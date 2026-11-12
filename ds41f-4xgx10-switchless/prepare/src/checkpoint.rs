//! Getting the checkpoint, and only when it is not already here.
//!
//! The rule the prepare command follows: a local checkpoint that opens and whose
//! shards reach the end of their last tensor is used as it is; anything else
//! (missing, short shard, bad index) triggers one download, which the Hub resumes,
//! and then the same validation runs again. The download itself is a closure, so
//! this module can be tested without a network.

use std::path::Path;

use crate::safetensors::Checkpoint;

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
