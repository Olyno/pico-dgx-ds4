//! Disk accounting and reclamation.
//!
//! Two rules keep a conversion inside its budget: outputs are allocated as they
//! are written (a sparse file whose blocks appear one tensor at a time), and a
//! source range is only released after its outputs are durable. Releasing means
//! `fallocate(FALLOC_FL_PUNCH_HOLE)`, which frees whole filesystem blocks and
//! leaves the file's length and offsets untouched; a filesystem without hole
//! punching can truncate a consumed tail instead.

use std::fs::File;
use std::io;
use std::os::unix::io::AsRawFd;
use std::path::Path;

/// Free bytes on the filesystem holding `path`, which need not exist yet: the
/// nearest existing ancestor decides, as it does for the eventual directory.
pub fn free_bytes(path: &Path) -> io::Result<u64> {
    let mut p = path;
    while !p.exists() {
        p = match p.parent() {
            Some(q) => q,
            None => return Err(io::Error::new(io::ErrorKind::NotFound, format!("{} has no ancestor", path.display()))),
        };
    }
    let c = std::ffi::CString::new(p.as_os_str().as_encoded_bytes())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "path has a NUL"))?;
    // SAFETY: statvfs fills the struct; the path is a valid C string.
    let mut st: libc::statvfs = unsafe { std::mem::zeroed() };
    if unsafe { libc::statvfs(c.as_ptr(), &mut st) } != 0 {
        return Err(io::Error::last_os_error());
    }
    Ok(st.f_bavail as u64 * st.f_frsize as u64)
}

/// Bytes the file actually occupies (not its logical length).
pub fn allocated_bytes(f: &File) -> io::Result<u64> {
    use std::os::unix::fs::MetadataExt;
    Ok(f.metadata()?.blocks() * 512)
}

/// Frees `[off, off+len)` of `f`, which must be block aligned for anything to be
/// freed. The file keeps its length; the range reads as zeros afterwards.
pub fn punch(f: &File, off: u64, len: u64) -> io::Result<()> {
    if len == 0 {
        return Ok(());
    }
    // SAFETY: fallocate only touches the descriptor's own allocation map.
    let rc = unsafe { libc::fallocate(f.as_raw_fd(), libc::FALLOC_FL_PUNCH_HOLE | libc::FALLOC_FL_KEEP_SIZE, off as i64, len as i64) };
    if rc == 0 {
        return Ok(());
    }
    let err = io::Error::last_os_error();
    match err.raw_os_error() {
        Some(libc::EOPNOTSUPP) | Some(libc::ENOSYS) | Some(libc::EINVAL) => {
            Err(io::Error::new(io::ErrorKind::Unsupported, format!("hole punching: {err}")))
        }
        _ => Err(err),
    }
}

/// Frees everything from `off` to the end of the file by truncating it. Only
/// valid when `off` is at or past the last byte any consumer still needs.
pub fn truncate_tail(f: &File, off: u64) -> io::Result<()> {
    f.set_len(off)
}

/// The peak bytes that must be free during a `--consume` run, in the same
/// filesystem as the checkpoint: outputs are written shard by shard, and a
/// shard's blocks are released only once every tensor that reads it is committed
/// (a tensor is written at the last shard its sources occupy). What has to fit is
/// therefore the high-water mark of `written - released`, not the total.
pub fn consume_peak(ck: &crate::safetensors::Checkpoint, world: u32, ranks: &[u32], engram: bool) -> Result<u64, String> {
    let n = ck.shards.len();
    let mut written_at = vec![0u64; n];
    let mut released_at = vec![0usize; n];
    for &r in ranks {
        let p = crate::plan::make_plan(&ck.tensors, r, world).map_err(|e| format!("rank {r}: {e}"))?;
        for t in &p.tensors {
            let last = t.pieces.iter().map(|pc| ck.tensors[pc.src].shard).max().unwrap_or(0);
            written_at[last] = written_at[last].saturating_add(t.bytes);
            for pc in &t.pieces {
                let s = ck.tensors[pc.src].shard;
                released_at[s] = released_at[s].max(last);
            }
        }
    }
    // An Engram table is written when its weight's shard is processed, and that
    // shard can only be released afterwards.
    if engram {
        for t in crate::engram::tables(ck)? {
            let s = ck.tensors[t.weight].shard;
            for &r in ranks {
                written_at[s] = written_at[s].saturating_add(crate::engram::shard_bytes(&t, r, world));
            }
            released_at[s] = released_at[s].max(s);
        }
    }
    let sizes: Vec<u64> = ck.shards.iter().map(|s| s.size).collect();
    Ok(simulate(&sizes, &written_at, &released_at))
}

/// The high-water mark of `written - released` in shard order. A shard's outputs
/// are written before the shards they finish are released, so the peak is taken
/// between the two, and a step that frees more than it writes cannot go below
/// zero (the space was already free).
fn simulate(sizes: &[u64], written_at: &[u64], released_at: &[usize]) -> u64 {
    let mut freed_at = vec![0u64; sizes.len()];
    for (s, &at) in released_at.iter().enumerate() {
        freed_at[at] = freed_at[at].saturating_add(sizes[s]);
    }
    let (mut written, mut freed, mut peak) = (0u64, 0u64, 0u64);
    for k in 0..sizes.len() {
        written = written.saturating_add(written_at.get(k).copied().unwrap_or(0));
        peak = peak.max(written.saturating_sub(freed));
        freed = freed.saturating_add(freed_at[k]);
    }
    peak
}

/// Admission for one output directory: never plan to write more than the disk can
/// hold alongside the reserve the run must keep free.
#[derive(Debug, Clone)]
pub struct Reservation {
    pub limit: u64,
    pub committed: u64,
}

impl Reservation {
    pub fn new(limit: u64) -> Reservation {
        Reservation { limit, committed: 0 }
    }

    /// Reserves `bytes` of future output. The caller must have subtracted the
    /// source it still has to keep (if any) from `limit`.
    pub fn take(&mut self, bytes: u64) -> Result<(), String> {
        if self.committed.saturating_add(bytes) > self.limit {
            return Err(format!(
                "the plan needs {:.1} GB of output and the budget has {:.1} GB left",
                (self.committed + bytes) as f64 / 1e9,
                self.limit.saturating_sub(self.committed) as f64 / 1e9
            ));
        }
        self.committed += bytes;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::os::unix::fs::MetadataExt;

    #[test]
    fn free_space_is_reported() {
        let f = free_bytes(Path::new("/tmp")).unwrap();
        assert!(f > 0);
    }

    #[test]
    fn punching_frees_whole_blocks_and_zeroes_the_range() {
        let dir = std::env::temp_dir().join(format!("dsw-space-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f");
        let mut f = std::fs::OpenOptions::new().create(true).read(true).write(true).open(&path).unwrap();
        f.write_all(&vec![0xABu8; 4 << 20]).unwrap();
        f.sync_all().unwrap();
        let before = f.metadata().unwrap().blocks();
        assert!(before > 0);
        // Aligned to the filesystem block size (4096 here).
        punch(&f, 4096, 1 << 20).unwrap();
        f.sync_all().unwrap();
        let after = f.metadata().unwrap().blocks();
        assert!(after < before, "{after} !< {before}");
        assert_eq!(f.metadata().unwrap().len(), 4 << 20, "the length is untouched");
        let mut buf = [1u8; 16];
        use std::os::unix::fs::FileExt;
        f.read_exact_at(&mut buf, 4096).unwrap();
        assert_eq!(buf, [0u8; 16], "a punched range reads as zeros");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn truncation_drops_the_tail() {
        let dir = std::env::temp_dir().join(format!("dsw-space-tr-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("f");
        let mut f = File::create(&path).unwrap();
        f.write_all(&[7u8; 8192]).unwrap();
        truncate_tail(&f, 4096).unwrap();
        assert_eq!(f.metadata().unwrap().len(), 4096);
        let _ = std::fs::remove_dir_all(&dir);
    }

    /// A two-shard checkpoint: `(shard, name, dtype, shape)` per tensor.
    fn checkpoint(tensors: &[(usize, &str, &str, &[u64])]) -> (std::path::PathBuf, crate::safetensors::Checkpoint) {
        use crate::safetensors::tests::write_shard;
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("dsw-space-ck-{}-{}", std::process::id(), n));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        for shard in 0..2 {
            let items: Vec<(&str, &str, &[u64], Vec<u8>)> = tensors
                .iter()
                .filter(|(s, ..)| *s == shard)
                .map(|(_, name, dtype, shape)| {
                    let bytes: u64 = shape.iter().product::<u64>() * crate::dsw::DType::from_safetensors(dtype).unwrap().bytes();
                    (*name, *dtype, *shape, vec![0u8; bytes as usize])
                })
                .collect();
            let refs: Vec<(&str, &str, &[u64], &[u8])> =
                items.iter().map(|(n, d, s, b)| (*n, *d, *s, b.as_slice())).collect();
            write_shard(&dir.join(format!("model-0000{}-of-00002.safetensors", shard + 1)), &refs);
        }
        let ck = crate::safetensors::Checkpoint::open(&dir).unwrap();
        (dir, ck)
    }

    #[test]
    fn consuming_peaks_at_one_shards_outputs() {
        let (dir, ck) = checkpoint(&[
            (0, "layers.0.attn.wq_b.weight", "F8_E4M3", &[2048, 64]),
            (1, "layers.1.attn.wq_b.weight", "F8_E4M3", &[2048, 64]),
        ]);
        let ranks = [0u32, 1, 2, 3];
        // Four ranks of 512 rows x 64 bytes, written when shard 0 is processed, and
        // shard 0 is released in the same step: the peak is one shard's worth.
        let peak = consume_peak(&ck, 4, &ranks, false).unwrap();
        assert_eq!(peak, 4 * 512 * 64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_tensor_spanning_two_shards_cannot_be_released_early() {
        // The two experts stack into one tensor, so nothing is released until the
        // last of the two shards has been read: the peak is the whole set.
        let (dir, ck) = checkpoint(&[
            (0, "layers.0.ffn.experts.0.w1.weight", "I8", &[64, 64]),
            (1, "layers.0.ffn.experts.1.w1.weight", "I8", &[64, 64]),
        ]);
        let ranks = [0u32, 1, 2, 3];
        let peak = consume_peak(&ck, 4, &ranks, false).unwrap();
        assert_eq!(peak, 4 * 2 * 16 * 64);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_peak_does_not_accumulate_across_shards() {
        // The real shape at TP=2: 46 shards of 7.4 GB whose rank files are 6.7 GB,
        // then the two 101.5 GB Engram shards whose shards are 50.7 GB. Written is
        // 409 GB in total, but every ordinary shard frees more than it writes, so
        // what has to be free is one Engram shard's output, not the sum.
        let ordinary = 7_400_000_000u64;
        let written = 6_700_000_000u64;
        let engram = 101_500_000_000u64;
        let engram_out = 50_700_000_000u64;
        let sizes: Vec<u64> = (0..46).map(|_| ordinary).chain([engram, engram]).collect();
        let written_at: Vec<u64> = (0..46).map(|_| written).chain([engram_out, engram_out]).collect();
        let released: Vec<usize> = (0..48).collect();
        let peak = simulate(&sizes, &written_at, &released);
        // Written so far (46 x 6.7 GB + one Engram shard) minus what those 46
        // shards already released (46 x 7.4 GB).
        assert_eq!(peak, 308_200_000_000u64 + engram_out - 340_400_000_000u64);
        assert!(peak < 20_000_000_000, "the peak is one Engram shard's output, not 409 GB: {peak}");
    }

    #[test]
    fn reservations_refuse_to_overcommit() {
        let mut r = Reservation::new(100);
        r.take(60).unwrap();
        assert!(r.take(50).is_err());
        r.take(40).unwrap();
        assert_eq!(r.committed, 100);
    }
}
