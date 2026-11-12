//! Reading a safetensors checkpoint without a library: the shard headers and
//! `model.safetensors.index.json`, mirroring `engine/weights/safetensors.cpp`.
//!
//! The header of a shard is `u64 length`, then `length` bytes of JSON mapping a
//! tensor name to `{dtype, shape, data_offsets}`. Offsets are relative to the end
//! of the header, so a tensor's absolute byte range in the file is
//! `8 + length + data_offsets`.

use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::path::{Path, PathBuf};

use serde_json::Value;

use crate::dsw::DType;

#[derive(Debug, Clone)]
pub struct SrcTensor {
    pub name: String,
    pub dtype: DType,
    pub shape: Vec<u64>,
    /// Index into [`Checkpoint::shards`].
    pub shard: usize,
    /// Absolute offset of the tensor's first byte in the shard file.
    pub offset: u64,
    pub bytes: u64,
}

#[derive(Debug, Clone)]
pub struct Shard {
    pub name: String,
    /// `None` when the shard was described by its header alone (a streaming run
    /// that never has the file on disk).
    pub path: Option<PathBuf>,
    pub size: u64,
    /// Absolute offset of the first tensor byte (8 + the header's length).
    pub data_start: u64,
    /// The tensors this shard holds, in file order.
    pub tensors: Vec<usize>,
}

#[derive(Debug)]
pub struct Checkpoint {
    pub dir: PathBuf,
    pub shards: Vec<Shard>,
    pub tensors: Vec<SrcTensor>,
    by_name: HashMap<String, usize>,
}

impl Checkpoint {
    pub fn at(&self, name: &str) -> Option<&SrcTensor> {
        self.by_name.get(name).map(|&i| &self.tensors[i])
    }

    pub fn tensor(&self, name: &str) -> &SrcTensor {
        self.at(name).unwrap_or_else(|| panic!("no tensor {name} in the checkpoint"))
    }

    pub fn bytes(&self) -> u64 {
        self.tensors.iter().map(|t| t.bytes).sum()
    }

    /// Opens every `*.safetensors` in `dir` and cross-checks the index, when one
    /// is present, against the tensors actually found: the index must name exactly
    /// those tensors and put each in the shard that holds it.
    pub fn open(dir: &Path) -> Result<Checkpoint, String> {
        let mut shard_paths: Vec<PathBuf> = std::fs::read_dir(dir)
            .map_err(|e| format!("{}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().map(|x| x == "safetensors").unwrap_or(false))
            .collect();
        shard_paths.sort();
        if shard_paths.is_empty() {
            return Err(format!("no .safetensors files in {}", dir.display()));
        }
        let mut ck = Checkpoint { dir: dir.to_path_buf(), shards: Vec::new(), tensors: Vec::new(), by_name: HashMap::new() };
        for path in shard_paths {
            let name = path.file_name().unwrap().to_string_lossy().into_owned();
            let (header, header_len, size) = read_header(&path)?;
            ck.add_shard(name, Some(path), &header, header_len, size)?;
        }
        ck.check_index()?;
        Ok(ck)
    }

    /// A checkpoint described by shard headers alone: the plan needs nothing else,
    /// and a streaming run never has the files. Each item is one shard's name, the
    /// header length the file declares, and the header JSON a ranged GET returns.
    pub fn from_shard_headers(items: Vec<(String, u64, Value)>) -> Result<Checkpoint, String> {
        let mut ck = Checkpoint { dir: PathBuf::new(), shards: Vec::new(), tensors: Vec::new(), by_name: HashMap::new() };
        for (name, header_len, header) in items {
            ck.add_shard(name, None, &header, header_len, u64::MAX)?;
        }
        // With no file to measure, the size is the end of the last tensor.
        for s in &mut ck.shards {
            let end = s.tensors.iter().map(|&i| ck.tensors[i].offset + ck.tensors[i].bytes).max().unwrap_or(0);
            s.size = end;
        }
        Ok(ck)
    }

    fn add_shard(&mut self, name: String, path: Option<PathBuf>, header: &Value, header_len: u64, size: u64) -> Result<(), String> {
        let where_ = path.as_ref().map(|p| p.display().to_string()).unwrap_or_else(|| name.clone());
        let data_start = 8 + header_len;
        let mut shard = Shard { name, path, size, data_start, tensors: Vec::new() };
        let mut entries: Vec<(String, Value)> = header
            .as_object()
            .ok_or_else(|| format!("{where_}: header is not an object"))?
            .iter()
            .filter(|(k, _)| k.as_str() != "__metadata__")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        for (tname, info) in entries {
            let dtype = info["dtype"].as_str().ok_or_else(|| format!("{tname}: no dtype"))?;
            let dtype = DType::from_safetensors(dtype)
                .ok_or_else(|| format!("{tname}: dtype {dtype} is not in the text model"))?;
            let shape: Vec<u64> = info["shape"]
                .as_array()
                .ok_or_else(|| format!("{tname}: no shape"))?
                .iter()
                .map(|v| v.as_u64().ok_or_else(|| format!("{tname}: bad shape")))
                .collect::<Result<_, _>>()?;
            let offs = info["data_offsets"].as_array().ok_or_else(|| format!("{tname}: no offsets"))?;
            if offs.len() != 2 {
                return Err(format!("{tname}: data_offsets is not a pair"));
            }
            let begin = offs[0].as_u64().ok_or_else(|| format!("{tname}: bad begin"))?;
            let end = offs[1].as_u64().ok_or_else(|| format!("{tname}: bad end"))?;
            if end < begin {
                return Err(format!("{tname}: end before begin"));
            }
            let bytes = end - begin;
            let expect: u64 = shape.iter().product::<u64>() * dtype.bytes();
            if bytes != expect {
                return Err(format!("{tname}: {bytes} bytes but shape {shape:?} of {dtype} needs {expect}"));
            }
            if data_start + end > size {
                return Err(format!(
                    "{tname}: ends at {} but {where_} is {size} bytes (truncated download?)",
                    data_start + end
                ));
            }
            let idx = self.tensors.len();
            if self.by_name.insert(tname.clone(), idx).is_some() {
                return Err(format!("{tname} appears twice"));
            }
            self.tensors.push(SrcTensor { name: tname, dtype, shape, shard: self.shards.len(), offset: data_start + begin, bytes });
            shard.tensors.push(idx);
        }
        self.shards.push(shard);
        Ok(())
    }

    fn check_index(&self) -> Result<(), String> {
        let path = self.dir.join("model.safetensors.index.json");
        if !path.exists() {
            return Ok(());
        }
        let text = std::fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let index: Value = serde_json::from_str(&text).map_err(|e| format!("{}: {e}", path.display()))?;
        let map = index["weight_map"].as_object().ok_or_else(|| format!("{}: no weight_map", path.display()))?;
        if map.len() != self.tensors.len() {
            return Err(format!(
                "{}: the index names {} tensors, the shards hold {}",
                path.display(),
                map.len(),
                self.tensors.len()
            ));
        }
        for (name, shard) in map {
            let shard = shard.as_str().ok_or_else(|| format!("{name}: shard is not a string"))?;
            let t = self.at(name).ok_or_else(|| format!("{name}: in the index, not in the shards"))?;
            if self.shards[t.shard].name != shard {
                return Err(format!("{name}: the index says {shard}, the tensor is in {}", self.shards[t.shard].name));
            }
        }
        Ok(())
    }

}

/// The parsed header of one shard, its length, and the file's size.
pub fn read_header(path: &Path) -> Result<(Value, u64, u64), String> {
    let mut f = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let size = f.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len();
    let mut len_buf = [0u8; 8];
    f.read_exact(&mut len_buf).map_err(|e| format!("{}: {e}", path.display()))?;
    let header_len = u64::from_le_bytes(len_buf);
    if header_len == 0 || header_len > 100 << 20 {
        return Err(format!("{}: implausible header length {header_len}", path.display()));
    }
    let mut buf = vec![0u8; header_len as usize];
    f.read_exact(&mut buf).map_err(|e| format!("{}: {e}", path.display()))?;
    let v: Value = serde_json::from_slice(&buf).map_err(|e| format!("{}: header: {e}", path.display()))?;
    Ok((v, header_len, size))
}

/// Parses a shard header from bytes (used by the range reader, which never has
/// the whole file on disk).
pub fn parse_header(bytes: &[u8]) -> Result<(u64, Value), String> {
    if bytes.len() < 8 {
        return Err("shorter than the 8-byte header length".into());
    }
    let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap());
    if bytes.len() < 8 + header_len as usize {
        return Err(format!("{} bytes, the header needs {}", bytes.len(), 8 + header_len));
    }
    let v: Value = serde_json::from_slice(&bytes[8..8 + header_len as usize]).map_err(|e| format!("header: {e}"))?;
    Ok((header_len, v))
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    /// A minimal shard: header JSON then the tensor bytes.
    pub fn write_shard(path: &Path, tensors: &[(&str, &str, &[u64], &[u8])]) {
        let mut header = serde_json::Map::new();
        let mut data = Vec::new();
        for (name, dtype, shape, bytes) in tensors {
            let begin = data.len() as u64;
            data.extend_from_slice(bytes);
            header.insert(
                name.to_string(),
                serde_json::json!({"dtype": dtype, "shape": shape, "data_offsets": [begin, data.len() as u64]}),
            );
        }
        let text = serde_json::to_vec(&Value::Object(header)).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(&text);
        out.extend_from_slice(&data);
        std::fs::write(path, out).unwrap();
    }

    #[test]
    fn reads_shards_offsets_and_the_index() {
        let dir = std::env::temp_dir().join(format!("dsw-st-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_shard(
            &dir.join("model-00001-of-00002.safetensors"),
            &[("embed.weight", "BF16", &[2, 4], &[0u8; 16])],
        );
        write_shard(
            &dir.join("model-00002-of-00002.safetensors"),
            &[("layers.0.attn.wq_a.weight", "F8_E4M3", &[8, 64], &[7u8; 512])],
        );
        let index = serde_json::json!({"metadata": {"total_size": 528}, "weight_map": {
            "embed.weight": "model-00001-of-00002.safetensors",
            "layers.0.attn.wq_a.weight": "model-00002-of-00002.safetensors",
        }});
        std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&index).unwrap()).unwrap();

        let ck = Checkpoint::open(&dir).unwrap();
        assert_eq!(ck.shards.len(), 2);
        assert_eq!(ck.tensors.len(), 2);
        assert_eq!(ck.bytes(), 16 + 512);
        let t = ck.tensor("layers.0.attn.wq_a.weight");
        assert_eq!(t.dtype, DType::F8E4M3);
        assert_eq!(t.shape, vec![8, 64]);
        assert_eq!(t.offset, ck.shards[t.shard].data_start);

        // A short shard is an error, not a silent short read: this is what makes
        // `prepare` re-download a checkpoint a full disk truncated.
        let p = dir.join("model-00002-of-00002.safetensors");
        let mut bytes = std::fs::read(&p).unwrap();
        bytes.truncate(bytes.len() - 100);
        std::fs::write(&p, bytes).unwrap();
        let err = Checkpoint::open(&dir).unwrap_err();
        assert!(err.contains("truncated download"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rejects_a_shape_that_disagrees_with_its_bytes() {
        let dir = std::env::temp_dir().join(format!("dsw-st-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        write_shard(&dir.join("model.safetensors"), &[("embed.weight", "BF16", &[4, 4], &[0u8; 8])]);
        let err = Checkpoint::open(&dir).unwrap_err();
        assert!(err.contains("needs 32"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn parses_a_header_without_the_file() {
        let dir = std::env::temp_dir().join(format!("dsw-st-hdr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("model.safetensors");
        write_shard(&p, &[("embed.weight", "BF16", &[2, 4], &[0u8; 16])]);
        let all = std::fs::read(&p).unwrap();
        let (len, v) = parse_header(&all).unwrap();
        // 8 bytes of length, `len` of JSON, then the 16 bytes of tensor data.
        assert_eq!(len as usize + 8 + 16, all.len());
        assert_eq!(v["embed.weight"]["dtype"], "BF16");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
