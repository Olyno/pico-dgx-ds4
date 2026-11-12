//! Getting the checkpoint, and only when it is not already here.
//!
//! The rule the prepare command follows: a local checkpoint that opens and whose
//! shards reach the end of their last tensor is used as it is; anything else
//! (missing, short shard, bad index) triggers one download, which the Hub resumes,
//! and then the same validation runs again. The download itself is a closure, so
//! this module can be tested without a network.

use std::path::{Path, PathBuf};

use crate::safetensors::Checkpoint;

/// The checkpoint this machine already has, if any: the two directory names the
/// README uses under `$HOME`, the `/data/models` layout the fleet points at, and
/// the Hub cache `hf download` fills without `--local-dir`. First hit wins, and a
/// directory counts only when a safetensors shard sits in it.
///
/// `--ckpt DIR` still overrides; this is what keeps `prepare` from downloading
/// 507 GB that is already on the disk under another name.
pub fn discover(home: &Path) -> Option<PathBuf> {
    let mut candidates = vec![
        home.join("DeepSeek-V4.1-Flash-1017"),
        home.join("DeepSeek-V4.1-Flash"),
        PathBuf::from("/data/models/DeepSeek-V4.1-Flash-1017"),
        PathBuf::from("/data/models/DeepSeek-V4.1-Flash"),
    ];
    let snapshots = home.join(".cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4.1-Flash-1017/snapshots");
    if let Ok(entries) = std::fs::read_dir(&snapshots) {
        let mut revisions: Vec<PathBuf> = entries.flatten().map(|e| e.path()).collect();
        revisions.sort();
        candidates.extend(revisions);
    }
    candidates.into_iter().find(|dir| has_shards(dir))
}

pub fn has_shards(dir: &Path) -> bool {
    std::fs::read_dir(dir)
        .map(|entries| {
            entries.flatten().any(|e| e.path().extension().map(|x| x == "safetensors").unwrap_or(false))
        })
        .unwrap_or(false)
}

/// What the caller wants to know about the acquisition.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum How {
    /// The checkpoint was already complete.
    Local,
    /// It was missing or short, and the download repaired it.
    Downloaded,
}

/// Opens `dir`, or downloads it first when it is not usable.
///
/// `download` is called at most once, with the directory to fill; it must leave a
/// complete checkpoint there (or return an error).
pub fn ensure(dir: &Path, download: &dyn Fn(&Path) -> Result<(), String>) -> Result<(Checkpoint, How), String> {
    match Checkpoint::open(dir) {
        Ok(ck) => return Ok((ck, How::Local)),
        Err(why) => eprintln!("[dsw-prepare] {why}: downloading"),
    }
    std::fs::create_dir_all(dir).map_err(|e| format!("{}: {e}", dir.display()))?;
    download(dir)?;
    let ck = Checkpoint::open(dir).map_err(|e| format!("the download left the checkpoint unusable: {e}"))?;
    Ok((ck, How::Downloaded))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::cell::Cell;
    use std::path::PathBuf;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("dsw-ck-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        d
    }

    /// Writes a one-shard checkpoint whose single tensor is `bytes` long.
    fn fill(dir: &Path, bytes: usize) {
        std::fs::create_dir_all(dir).unwrap();
        let header = serde_json::json!({"t": {"dtype": "BF16", "shape": [bytes / 2], "data_offsets": [0, bytes]}});
        let text = serde_json::to_vec(&header).unwrap();
        let mut out = Vec::new();
        out.extend_from_slice(&(text.len() as u64).to_le_bytes());
        out.extend_from_slice(&text);
        out.extend_from_slice(&vec![7u8; bytes]);
        std::fs::write(dir.join("model.safetensors"), out).unwrap();
    }

    #[test]
    fn a_complete_checkpoint_is_used_without_downloading() {
        let d = tmp("local");
        fill(&d, 16);
        let calls = Cell::new(0);
        let (ck, how) = ensure(&d, &|_| {
            calls.set(calls.get() + 1);
            Ok(())
        })
        .unwrap();
        assert_eq!(how, How::Local);
        assert_eq!(calls.get(), 0, "nothing was missing, so nothing was fetched");
        assert_eq!(ck.tensors.len(), 1);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_missing_checkpoint_is_downloaded_once() {
        let d = tmp("missing");
        let calls = Cell::new(0);
        let (ck, how) = ensure(&d, &|p| {
            calls.set(calls.get() + 1);
            fill(p, 32);
            Ok(())
        })
        .unwrap();
        assert_eq!(how, How::Downloaded);
        assert_eq!(calls.get(), 1);
        assert_eq!(ck.tensors[0].bytes, 32);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_short_shard_is_repaired_by_downloading_again() {
        let d = tmp("short");
        fill(&d, 64);
        // A download that ran out of space leaves the shard short.
        let p = d.join("model.safetensors");
        let mut all = std::fs::read(&p).unwrap();
        all.truncate(all.len() - 20);
        std::fs::write(&p, all).unwrap();
        let calls = Cell::new(0);
        let (ck, how) = ensure(&d, &|p| {
            calls.set(calls.get() + 1);
            fill(p, 64);
            Ok(())
        })
        .unwrap();
        assert_eq!(how, How::Downloaded);
        assert_eq!(calls.get(), 1);
        assert_eq!(ck.tensors[0].bytes, 64);
        let _ = std::fs::remove_dir_all(&d);
    }

    #[test]
    fn a_failed_download_is_an_error_not_an_empty_checkpoint() {
        let d = tmp("fail");
        let err = ensure(&d, &|_| Err("hub unreachable".into())).unwrap_err();
        assert!(err.contains("hub unreachable"), "{err}");
        let _ = std::fs::remove_dir_all(&d);
    }
}
