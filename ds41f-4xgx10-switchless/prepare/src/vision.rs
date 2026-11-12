//! The small part of the checkpoint the vision encoder needs, saved so the rest
//! of the checkpoint does not have to survive.
//!
//! `tools/vision/vision_encoder.py` mounts the checkpoint read-only and uses the
//! publisher's own `inference/vision.py` and `inference/image_processor.py`. It
//! reads `model.safetensors.index.json`, then only the shards named there for
//! tensors starting with `vision.`, `aligner.` or `image_`. On 1017 those are two
//! small shards (~2.3 GB) out of 510 GB — worth keeping when the rest is released.

use std::path::{Path, PathBuf};

use crate::safetensors::Checkpoint;

/// The prefixes `vision_encoder.py` selects (`want` in its `load`).
pub const PREFIXES: [&str; 3] = ["vision.", "aligner.", "image_"];

/// The publisher's scripts the encoder imports, relative to the checkpoint.
pub const SCRIPTS: [&str; 3] = ["inference/vision.py", "inference/image_processor.py", "inference/config.json"];

/// The small files that travel with the weights. The index is rewritten, not
/// copied: a complete index would name shards the bundle does not hold.
pub const FILES: [&str; 2] = ["config.json", "tokenizer.json"];

pub fn is_vision(name: &str) -> bool {
    PREFIXES.iter().any(|p| name.starts_with(p))
}

/// The shards that hold any tensor the encoder asks for.
pub fn shards_needed(ck: &Checkpoint) -> Vec<usize> {
    let mut v: Vec<usize> = ck
        .tensors
        .iter()
        .filter(|t| is_vision(&t.name))
        .map(|t| t.shard)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

#[derive(Debug)]
pub struct Bundle {
    pub dest: PathBuf,
    pub shards: Vec<usize>,
    pub bytes: u64,
}

impl Bundle {
    /// The encoder's `--model` argument.
    pub fn model_dir(&self) -> &Path {
        &self.dest
    }
}

/// Copies the scripts, the index and the needed shards into `dest`.
///
/// `materialize` puts shard `i` at the given path — a copy for a checkpoint on
/// disk, a fetch for one streamed from the Hub.
pub fn write(
    ck: &Checkpoint,
    dest: &Path,
    materialize: &dyn Fn(usize, &Path) -> Result<(), String>,
) -> Result<Bundle, String> {
    let shards = shards_needed(ck);
    if shards.is_empty() {
        return Err("the checkpoint has no vision, aligner or image tensors".into());
    }
    std::fs::create_dir_all(dest.join("inference")).map_err(|e| format!("{}: {e}", dest.display()))?;
    let mut bytes = 0u64;
    for rel in SCRIPTS.iter().chain(FILES.iter()) {
        let src = ck.dir.join(rel);
        if !src.is_file() {
            // config.json and tokenizer.json are useful but not required to encode.
            if *rel == "config.json" || *rel == "tokenizer.json" {
                continue;
            }
            return Err(format!("{}: not in the checkpoint", src.display()));
        }
        let to = dest.join(rel);
        std::fs::copy(&src, &to).map_err(|e| format!("{}: {e}", to.display()))?;
        bytes += std::fs::metadata(&to).map_err(|e| e.to_string())?.len();
    }
    let mut copied = Vec::new();
    for &i in &shards {
        let name = &ck.shards[i].name;
        let to = dest.join(name);
        materialize(i, &to)?;
        bytes += std::fs::metadata(&to).map_err(|e| to.display().to_string() + &e.to_string())?.len();
        copied.push(i);
    }
    // An index naming only what the bundle holds, so the directory is consistent
    // on its own.
    let full: serde_json::Value = serde_json::from_slice(
        &std::fs::read(ck.dir.join("model.safetensors.index.json"))
            .map_err(|e| format!("{}: {e}", ck.dir.join("model.safetensors.index.json").display()))?,
    )
    .map_err(|e| format!("the index: {e}"))?;
    let mut map = serde_json::Map::new();
    if let Some(w) = full["weight_map"].as_object() {
        for (name, shard) in w {
            if is_vision(name) {
                map.insert(name.clone(), shard.clone());
            }
        }
    }
    let kept: u64 = ck.tensors.iter().filter(|t| is_vision(&t.name)).map(|t| t.bytes).sum();
    let index = serde_json::json!({"metadata": {"total_size": kept}, "weight_map": map});
    let idx_path = dest.join("model.safetensors.index.json");
    std::fs::write(&idx_path, serde_json::to_vec(&index).map_err(|e| e.to_string())?)
        .map_err(|e| format!("{}: {e}", idx_path.display()))?;
    bytes += std::fs::metadata(&idx_path).map_err(|e| e.to_string())?.len();
    Ok(Bundle { dest: dest.to_path_buf(), shards, bytes })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::tests::write_shard;
    use std::sync::atomic::{AtomicUsize, Ordering};

    fn checkpoint(tag: &str) -> (PathBuf, Checkpoint) {
        let d = std::env::temp_dir().join(format!("dsw-vision-{}-{tag}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(d.join("inference")).unwrap();
        for f in ["vision.py", "image_processor.py"] {
            std::fs::write(d.join("inference").join(f), b"# the publisher's\n").unwrap();
        }
        std::fs::write(d.join("inference/config.json"), b"{\"vision_n_layers\": 24}").unwrap();
        std::fs::write(d.join("config.json"), b"{}").unwrap();
        std::fs::write(d.join("tokenizer.json"), b"{}").unwrap();
        write_shard(
            &d.join("model-00001-of-00002.safetensors"),
            &[("layers.0.attn_norm.weight", "BF16", &[8], &[0u8; 16])],
        );
        write_shard(
            &d.join("model-00002-of-00002.safetensors"),
            &[
                ("vision.blocks.0.attn.wq.weight", "BF16", &[4], &[1u8; 8]),
                ("aligner.proj.weight", "BF16", &[4], &[2u8; 8]),
                ("image_start", "BF16", &[4], &[3u8; 8]),
            ],
        );
        let idx = serde_json::json!({"weight_map": {
            "layers.0.attn_norm.weight": "model-00001-of-00002.safetensors",
            "vision.blocks.0.attn.wq.weight": "model-00002-of-00002.safetensors",
            "aligner.proj.weight": "model-00002-of-00002.safetensors",
            "image_start": "model-00002-of-00002.safetensors",
        }});
        std::fs::write(d.join("model.safetensors.index.json"), serde_json::to_vec(&idx).unwrap()).unwrap();
        let ck = Checkpoint::open(&d).unwrap();
        (d, ck)
    }

    #[test]
    fn only_the_vision_shard_is_kept() {
        let (src, ck) = checkpoint("keep");
        let needed = shards_needed(&ck);
        assert_eq!(needed.len(), 1);
        assert_eq!(ck.shards[needed[0]].name, "model-00002-of-00002.safetensors");

        let dest = std::env::temp_dir().join(format!("dsw-vision-bundle-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dest);
        let b = write(&ck, &dest, &|i, to| {
            std::fs::copy(ck.shards[i].path.as_ref().unwrap(), to).map(|_| ()).map_err(|e| e.to_string())
        })
        .unwrap();
        assert_eq!(b.shards, needed);
        assert!(dest.join("model-00002-of-00002.safetensors").exists());
        assert!(!dest.join("model-00001-of-00002.safetensors").exists(), "the text shard must not be copied");
        assert!(dest.join("inference/vision.py").exists());
        assert!(dest.join("inference/image_processor.py").exists());
        assert!(dest.join("model.safetensors.index.json").exists());
        // The bundle is a consistent `--model` directory on its own: its index
        // names exactly the tensors it holds.
        let bundle = Checkpoint::open(&dest).unwrap();
        let mut names: Vec<&str> = bundle.tensors.iter().map(|t| t.name.as_str()).collect();
        names.sort_unstable();
        assert_eq!(names, ["aligner.proj.weight", "image_start", "vision.blocks.0.attn.wq.weight"]);
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&dest);
    }

    #[test]
    fn a_checkpoint_without_vision_is_an_error() {
        let d = std::env::temp_dir().join(format!("dsw-vision-none-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        write_shard(&d.join("model.safetensors"), &[("layers.0.attn_norm.weight", "BF16", &[8], &[0u8; 16])]);
        let ck = Checkpoint::open(&d).unwrap();
        assert!(shards_needed(&ck).is_empty());
        let err = write(&ck, &d.join("b"), &|_, _| Ok(())).unwrap_err();
        assert!(err.contains("no vision"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn every_needed_shard_is_materialised_once() {
        let (src, ck) = checkpoint("once");
        let calls = AtomicUsize::new(0);
        let dest = std::env::temp_dir().join(format!("dsw-vision-once-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dest);
        write(&ck, &dest, &|i, to| {
            calls.fetch_add(1, Ordering::Relaxed);
            std::fs::copy(ck.shards[i].path.as_ref().unwrap(), to).map(|_| ()).map_err(|e| e.to_string())
        })
        .unwrap();
        assert_eq!(calls.load(Ordering::Relaxed), 1);
        let _ = std::fs::remove_dir_all(&src);
        let _ = std::fs::remove_dir_all(&dest);
    }
}
