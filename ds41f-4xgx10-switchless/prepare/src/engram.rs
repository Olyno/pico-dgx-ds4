//! The per-rank Engram shards, transcribed from `engine/engram/tools/pack.cpp`
//! and `engine/engram/engram.h`.
//!
//! Each layer's table is `[rows, 256]` of e4m3 values with `[rows, 8]` of ue8m0
//! scales. Rank `r` of 4 owns rows `[rows*r/4, rows*(r+1)/4)`, stored as 264-byte
//! records (the row's 256 weight bytes, then its 8 scale bytes) behind a 4 KiB
//! header of six little-endian u64: the magic, the layer, lo, hi (exclusive), the
//! table's rows, and 264.

use crate::safetensors::Checkpoint;

/// `pack.cpp`'s `kShardMagic`, which `store.cpp` compares against. Written
/// little-endian its bytes are `DSV1EN41` — that is what the kit's shards carry,
/// whatever the comment in `pack.cpp` says, so the value is copied, not the text.
pub const MAGIC: u64 = 0x3134_4E45_3156_5344;
pub const HEADER_BYTES: u64 = 4096;
pub const ROW_BYTES: u64 = 264;
pub const WEIGHT_BYTES: u64 = 256;
pub const SCALE_BYTES: u64 = ROW_BYTES - WEIGHT_BYTES;
pub const WORLD: u32 = 4;

#[derive(Debug, Clone)]
pub struct Table {
    pub layer: u32,
    pub rows: u64,
    /// Indices into [`Checkpoint::tensors`].
    pub weight: usize,
    pub scale: usize,
}

/// Finds every `layers.<L>.engram.embed.{weight,scale}` pair.
pub fn tables(ck: &Checkpoint) -> Result<Vec<Table>, String> {
    let mut out: Vec<Table> = Vec::new();
    for (i, t) in ck.tensors.iter().enumerate() {
        let Some(rest) = t.name.strip_prefix("layers.") else { continue };
        let Some((layer, tail)) = rest.split_once(".engram.embed.") else { continue };
        let Ok(layer) = layer.parse::<u32>() else { continue };
        if !matches!(tail, "weight" | "scale") {
            continue;
        }
        let slot = match out.iter_mut().find(|x| x.layer == layer) {
            Some(s) => s,
            None => {
                out.push(Table { layer, rows: 0, weight: usize::MAX, scale: usize::MAX });
                out.last_mut().unwrap()
            }
        };
        if tail == "weight" {
            slot.weight = i;
        } else {
            slot.scale = i;
        }
    }
    out.sort_by_key(|t| t.layer);
    for t in &mut out {
        if t.weight == usize::MAX || t.scale == usize::MAX {
            return Err(format!("layer {}'s Engram table lacks its weight or its scale", t.layer));
        }
        let w = &ck.tensors[t.weight];
        let s = &ck.tensors[t.scale];
        if w.shape.len() != 2 || w.shape[1] != WEIGHT_BYTES || w.bytes != w.shape[0] * WEIGHT_BYTES {
            return Err(format!("{} is not [rows, {WEIGHT_BYTES}] e4m3", w.name));
        }
        if s.shape.len() != 2 || s.shape[1] != SCALE_BYTES || s.shape[0] != w.shape[0] {
            return Err(format!("{} is not [rows, {SCALE_BYTES}] ue8m0 matching its weight", s.name));
        }
        t.rows = w.shape[0];
    }
    if out.is_empty() {
        return Err(format!("{} has no layers.<L>.engram.embed tensors", ck.dir.display()));
    }
    Ok(out)
}

/// The rows of `t` that rank `r` owns: `[lo, hi)`.
pub fn rows_of(t: &Table, rank: u32) -> (u64, u64) {
    (t.rows * rank as u64 / WORLD as u64, t.rows * (rank as u64 + 1) / WORLD as u64)
}

pub fn shard_name(layer: u32, rank: u32) -> String {
    format!("engram-l{layer}-r{rank}of{WORLD}.bin")
}

pub fn shard_bytes(t: &Table, rank: u32) -> u64 {
    let (lo, hi) = rows_of(t, rank);
    HEADER_BYTES + (hi - lo) * ROW_BYTES
}

/// The 4 KiB header: six u64, then zeros.
pub fn header(layer: u32, lo: u64, hi: u64, rows: u64) -> [u8; HEADER_BYTES as usize] {
    let mut h = [0u8; HEADER_BYTES as usize];
    for (i, v) in [MAGIC, layer as u64, lo, hi, rows, ROW_BYTES].iter().enumerate() {
        h[i * 8..i * 8 + 8].copy_from_slice(&v.to_le_bytes());
    }
    h
}

/// One record: the row's weight bytes, then its scale bytes.
pub fn record(weight: &[u8], scale: &[u8]) -> [u8; ROW_BYTES as usize] {
    assert_eq!(weight.len(), WEIGHT_BYTES as usize);
    assert_eq!(scale.len(), SCALE_BYTES as usize);
    let mut r = [0u8; ROW_BYTES as usize];
    r[..weight.len()].copy_from_slice(weight);
    r[weight.len()..].copy_from_slice(scale);
    r
}

/// Splits a record back into its weight and scale bytes.
pub fn split_record(r: &[u8]) -> (&[u8], &[u8]) {
    r.split_at(WEIGHT_BYTES as usize)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::tests::write_shard;
    use std::path::PathBuf;

    fn ck(layers: &[u32], rows: u64) -> (PathBuf, Checkpoint) {
        let dir = std::env::temp_dir().join(format!("dsw-engram-{}-{}", std::process::id(), layers.len()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut names = Vec::new();
        for l in layers {
            names.push((format!("layers.{l}.engram.embed.weight"), "F8_E4M3", vec![rows, WEIGHT_BYTES]));
            names.push((format!("layers.{l}.engram.embed.scale"), "F8_E8M0", vec![rows, SCALE_BYTES]));
        }
        let bytes: Vec<Vec<u8>> = names
            .iter()
            .map(|(_, _, s)| vec![0u8; s.iter().product::<u64>() as usize])
            .collect();
        let refs: Vec<(&str, &str, &[u64], &[u8])> = names
            .iter()
            .zip(&bytes)
            .map(|((n, d, s), b)| (n.as_str(), *d, s.as_slice(), b.as_slice()))
            .collect();
        write_shard(&dir.join("model.safetensors"), &refs);
        let ck = Checkpoint::open(&dir).unwrap();
        (dir, ck)
    }

    #[test]
    fn finds_tables_and_their_row_counts() {
        let (dir, ck) = ck(&[1, 14], 8);
        let t = tables(&ck).unwrap();
        assert_eq!(t.len(), 2);
        assert_eq!(t[0].layer, 1);
        assert_eq!(t[1].layer, 14);
        assert_eq!(t[0].rows, 8);
        assert_eq!(ck.tensors[t[0].weight].name, "layers.1.engram.embed.weight");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_split_is_the_kits_four_way_one() {
        let (dir, ck) = ck(&[1], 1000);
        let t = &tables(&ck).unwrap()[0];
        assert_eq!(rows_of(t, 0), (0, 250));
        assert_eq!(rows_of(t, 1), (250, 500));
        assert_eq!(rows_of(t, 3), (750, 1000));
        assert_eq!(shard_bytes(t, 1), HEADER_BYTES + 250 * ROW_BYTES);
        assert_eq!(shard_name(14, 2), "engram-l14-r2of4.bin");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_header_is_six_little_endian_words_then_zeros() {
        let h = header(14, 250, 500, 1000);
        assert_eq!(h.len(), 4096);
        let w = |i: usize| u64::from_le_bytes(h[i * 8..i * 8 + 8].try_into().unwrap());
        assert_eq!(w(0), MAGIC);
        assert_eq!(&h[..8], b"DSV1EN41", "the bytes pack.cpp actually writes");
        assert_eq!(w(1), 14);
        assert_eq!(w(2), 250);
        assert_eq!(w(3), 500);
        assert_eq!(w(4), 1000);
        assert_eq!(w(5), 264);
        assert!(h[48..].iter().all(|&b| b == 0));
    }

    #[test]
    fn a_record_is_weight_then_scale() {
        let w: Vec<u8> = (0..256u32).map(|i| i as u8).collect();
        let s: Vec<u8> = vec![0xAB; 8];
        let r = record(&w, &s);
        assert_eq!(r.len(), 264);
        assert_eq!(&r[..256], &w[..]);
        assert_eq!(&r[256..], &s[..]);
        let (gw, gs) = split_record(&r);
        assert_eq!(gw, &w[..]);
        assert_eq!(gs, &s[..]);
    }

    #[test]
    fn a_missing_scale_is_an_error() {
        let dir = std::env::temp_dir().join(format!("dsw-engram-bad-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let b = vec![0u8; 256 * 8];
        write_shard(&dir.join("model.safetensors"), &[("layers.1.engram.embed.weight", "F8_E4M3", &[8, 256], &b)]);
        let ck = Checkpoint::open(&dir).unwrap();
        let err = tables(&ck).unwrap_err();
        assert!(err.contains("lacks"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
