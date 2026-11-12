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

    #[test]
    fn reservations_refuse_to_overcommit() {
        let mut r = Reservation::new(100);
        r.take(60).unwrap();
        assert!(r.take(50).is_err());
        r.take(40).unwrap();
        assert_eq!(r.committed, 100);
    }
}
