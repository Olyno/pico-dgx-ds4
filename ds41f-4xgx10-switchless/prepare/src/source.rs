//! Where the checkpoint bytes come from.
//!
//! Two sources, one interface. A local shard is read with `pread` and, once its
//! last consumer is committed, its blocks can be released. A remote shard is read
//! through HTTP ranges, so a 101.5 GB shard never has to exist on disk at all —
//! only the ranges a batch needs, bounded by the caller's buffer.
//!
//! `curl` is the transport on purpose: it is already in the image, it inherits
//! `HF_ENDPOINT`/proxy settings, and `--max-filesize` caps the damage when a
//! proxy ignores `Range` and answers `200` with the whole file.

use std::fs::File;
use std::io::{self, Read};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

use crate::space;

/// A shard's bytes, addressed by absolute offset.
pub trait Source {
    fn len(&self) -> u64;
    /// Fills `buf` from `off`. Reads beyond the end are an error, not a short read.
    fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<()>;
}

/// Reads the checkpoint's shards, whichever way they are stored.
///
/// `prepare` is where a hybrid implementation fetches a whole file, and `finish`
/// is where it drops it, so the caller only has to bracket a shard.
pub trait Reader {
    fn prepare(&self, shard: usize) -> io::Result<()> {
        let _ = shard;
        Ok(())
    }

    fn read_at(&self, shard: usize, off: u64, buf: &mut [u8]) -> io::Result<()>;

    /// Releases `[off, off+len)` when this reader's copy is local. A remote copy
    /// has nothing to free.
    fn release(&self, shard: usize, off: u64, len: u64) -> io::Result<()> {
        let _ = (shard, off, len);
        Ok(())
    }

    /// Frees the whole copy of `shard`, for a filesystem whose `release` cannot
    /// punch holes.
    fn truncate(&self, shard: usize, len: u64) -> io::Result<()> {
        let _ = (shard, len);
        Ok(())
    }

    /// The path of this reader's own copy of `shard`, when it has one: what the
    /// vision bundle copies and what `--consume` releases.
    fn local_path(&self, shard: usize) -> Option<PathBuf> {
        let _ = shard;
        None
    }

    /// Drops what `prepare` left on disk.
    fn finish(&self, shard: usize) {
        let _ = shard;
    }
}

/// Every shard on disk: the plain local case, and the one `--consume` frees.
pub struct LocalReader {
    shards: Vec<LocalFile>,
    paths: Vec<PathBuf>,
}

impl LocalReader {
    pub fn open(paths: &[PathBuf]) -> io::Result<LocalReader> {
        Ok(LocalReader {
            shards: paths.iter().map(|p| LocalFile::open(p)).collect::<io::Result<_>>()?,
            paths: paths.to_vec(),
        })
    }
}

impl Reader for LocalReader {
    fn read_at(&self, shard: usize, off: u64, buf: &mut [u8]) -> io::Result<()> {
        self.shards[shard].read_at(off, buf)
    }

    fn release(&self, shard: usize, off: u64, len: u64) -> io::Result<()> {
        self.shards[shard].punch(off, len)
    }

    fn truncate(&self, shard: usize, len: u64) -> io::Result<()> {
        self.shards[shard].truncate_tail(len)
    }

    fn local_path(&self, shard: usize) -> Option<PathBuf> {
        Some(self.paths[shard].clone())
    }
}

/// Every shard read by HTTP range: nothing is stored, only the tensor in flight.
pub struct HttpReader {
    shards: Vec<HttpRange>,
}

impl HttpReader {
    pub fn new(shards: Vec<HttpRange>) -> HttpReader {
        HttpReader { shards }
    }
}

impl Reader for HttpReader {
    fn read_at(&self, shard: usize, off: u64, buf: &mut [u8]) -> io::Result<()> {
        self.shards[shard].read_at(off, buf)
    }
}

/// Whole small shards on disk, ranges for the large ones.
///
/// One shard is resident at a time (the one being converted), so the peak grows by
/// its size and nothing else; it is deleted as soon as that shard is done. A shard
/// above `max_cached` (the Engram tables) is never downloaded — it is read by
/// range, which keeps its 101.5 GB off the disk entirely.
pub struct HybridReader {
    remote: Vec<HttpRange>,
    names: Vec<String>,
    scratch: PathBuf,
    max_cached: u64,
    cached: std::sync::Mutex<Option<(usize, PathBuf, LocalFile)>>,
}

impl HybridReader {
    pub fn new(remote: Vec<HttpRange>, names: Vec<String>, scratch: PathBuf, max_cached: u64) -> HybridReader {
        HybridReader { remote, names, scratch, max_cached, cached: std::sync::Mutex::new(None) }
    }

    fn drop_cached(&self) {
        let mut c = self.cached.lock().unwrap();
        if let Some((_, path, _)) = c.take() {
            let _ = std::fs::remove_file(path);
        }
    }
}

impl Reader for HybridReader {
    fn prepare(&self, shard: usize) -> io::Result<()> {
        {
            let c = self.cached.lock().unwrap();
            if matches!(&*c, Some((s, _, _)) if *s == shard) {
                return Ok(());
            }
        }
        self.drop_cached();
        let len = self.remote[shard].len();
        if len > self.max_cached {
            return Ok(()); // read by range instead
        }
        std::fs::create_dir_all(&self.scratch)?;
        let path = self.scratch.join(&self.names[shard]);
        download_to(self.remote[shard].url(), &path, len, self.remote[shard].token(), 3)?;
        let local = LocalFile::open(&path)?;
        *self.cached.lock().unwrap() = Some((shard, path, local));
        Ok(())
    }

    fn read_at(&self, shard: usize, off: u64, buf: &mut [u8]) -> io::Result<()> {
        let c = self.cached.lock().unwrap();
        match &*c {
            Some((s, _, local)) if *s == shard => local.read_at(off, buf),
            _ => {
                drop(c);
                self.remote[shard].read_at(off, buf)
            }
        }
    }

    fn release(&self, shard: usize, off: u64, len: u64) -> io::Result<()> {
        let c = self.cached.lock().unwrap();
        match &*c {
            Some((s, _, local)) if *s == shard => local.punch(off, len),
            _ => Ok(()),
        }
    }

    fn truncate(&self, shard: usize, len: u64) -> io::Result<()> {
        let c = self.cached.lock().unwrap();
        match &*c {
            Some((s, _, local)) if *s == shard => local.truncate_tail(len),
            _ => Ok(()),
        }
    }

    fn local_path(&self, shard: usize) -> Option<PathBuf> {
        let c = self.cached.lock().unwrap();
        match &*c {
            Some((s, path, _)) if *s == shard => Some(path.clone()),
            _ => None,
        }
    }

    fn finish(&self, shard: usize) {
        let is_current = {
            let c = self.cached.lock().unwrap();
            matches!(&*c, Some((s, _, _)) if *s == shard)
        };
        if is_current {
            self.drop_cached();
        }
    }
}

/// Downloads one file, resuming a partial one, and checks its size.
pub fn download_to(url: &str, path: &Path, expect: u64, token: Option<&str>, retries: u32) -> io::Result<()> {
    if let Ok(m) = std::fs::metadata(path) {
        if expect != 0 && m.len() == expect {
            return Ok(());
        }
    }
    let mut attempt = 0;
    loop {
        let mut cmd = Command::new("curl");
        cmd.args(["-sSfL", "-C", "-", "--max-filesize", &(expect + 1).to_string(), "-o"]).arg(path);
        if let Some(t) = token {
            cmd.args(["-H", &format!("Authorization: Bearer {t}")]);
        }
        cmd.arg(url);
        let out = cmd.output()?;
        if out.status.success() {
            let got = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
            if expect == 0 || got == expect {
                return Ok(());
            }
            // A resumed transfer that overshot or stalled: start over once.
            let _ = std::fs::remove_file(path);
        }
        if attempt >= retries {
            return Err(io::Error::other(format!(
                "{url}: download failed after {attempt} retries: {}",
                String::from_utf8_lossy(&out.stderr).trim()
            )));
        }
        attempt += 1;
        std::thread::sleep(std::time::Duration::from_millis(500 << (attempt - 1).min(4)));
    }
}

pub struct LocalFile {
    file: File,
    len: u64,
}

impl LocalFile {
    /// Opens a shard. Read-write when the filesystem allows it, because releasing
    /// its blocks needs a writable descriptor; a read-only mount still converts.
    pub fn open(path: &Path) -> io::Result<LocalFile> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .open(path)
            .or_else(|e| if e.kind() == io::ErrorKind::PermissionDenied { File::open(path) } else { Err(e) })?;
        let len = file.metadata()?.len();
        Ok(LocalFile { file, len })
    }

    /// Releases `[off, off+len)`; see [`space::punch`].
    pub fn punch(&self, off: u64, len: u64) -> io::Result<()> {
        space::punch(&self.file, off, len)
    }

    pub fn truncate_tail(&self, off: u64) -> io::Result<()> {
        space::truncate_tail(&self.file, off)
    }
}

impl Source for LocalFile {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        if off.saturating_add(buf.len() as u64) > self.len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("read {}+{} past the {}-byte file", off, buf.len(), self.len),
            ));
        }
        self.file.read_exact_at(buf, off)
    }
}

/// The Hub URL of one shard at a pinned revision.
pub fn hub_url(endpoint: &str, repo: &str, rev: &str, shard: &str) -> String {
    format!("{}/{}/resolve/{}/{}", endpoint.trim_end_matches('/'), repo, rev, shard)
}

/// The curl invocation for one range. Pure, so the arguments are testable.
pub fn curl_args(url: &str, off: u64, len: u64, token: Option<&str>) -> Vec<String> {
    let mut a: Vec<String> = vec![
        "-sSfL".into(),
        // No `--max-filesize` here: it compares the *declared* body with the cap,
        // so a server that ignores Range (a Xet-backed Hub URL does) makes curl
        // abort with "(63) Maximum file size exceeded" before the reader can say
        // what happened. The reader asks for `len` bytes and treats the one extra
        // byte that means "the whole file" as the error it is.
        "-H".into(),
        format!("Range: bytes={}-{}", off, off + len - 1),
        "-H".into(),
        "Accept-Encoding: identity".into(),
    ];
    if let Some(t) = token {
        a.push("-H".into());
        a.push(format!("Authorization: Bearer {t}"));
    }
    a.push(url.into());
    a
}

pub struct HttpRange {
    url: String,
    len: u64,
    token: Option<String>,
    retries: u32,
}

impl HttpRange {
    pub fn new(url: String, len: u64, token: Option<String>, retries: u32) -> HttpRange {
        HttpRange { url, len, token, retries }
    }

    pub fn url(&self) -> &str {
        &self.url
    }

    pub fn token(&self) -> Option<&str> {
        self.token.as_deref()
    }

    fn once(&self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        let mut child = Command::new("curl")
            .args(curl_args(&self.url, off, buf.len() as u64, self.token.as_deref()))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let mut stdout = child.stdout.take().expect("piped");
        let mut got = 0usize;
        while got < buf.len() {
            match stdout.read(&mut buf[got..]) {
                Ok(0) => break,
                Ok(n) => got += n,
                Err(e) if e.kind() == io::ErrorKind::Interrupted => continue,
                Err(e) => return Err(e),
            }
        }
        // One byte more than asked for means the server sent the whole file.
        let mut extra = [0u8; 1];
        let over = matches!(stdout.read(&mut extra), Ok(n) if n > 0);
        drop(stdout);
        let out = child.wait_with_output()?;
        // Before the exit status: closing the pipe makes curl die of EPIPE, and
        // that would hide the only thing worth saying.
        if over {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!(
                    "the server sent the whole shard for bytes {off}+{} (it does not honour Range: \
                     Xet-backed Hub URLs answer 200): download the checkpoint first (hf download) \
                     and run with --source local",
                    buf.len()
                ),
            ));
        }
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            return Err(io::Error::other(err.trim().to_string()));
        }
        if got != buf.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("asked for {} bytes at {off}, got {got}", buf.len()),
            ));
        }
        if over {
            return Err(io::Error::other(format!(
                "the server answered with more than the requested range at {off} (Range ignored?)"
            )));
        }
        Ok(())
    }
}

impl Source for HttpRange {
    fn len(&self) -> u64 {
        self.len
    }

    fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<()> {
        if off.saturating_add(buf.len() as u64) > self.len {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                format!("read {}+{} past the {}-byte shard", off, buf.len(), self.len),
            ));
        }
        let mut attempt = 0;
        loop {
            match self.once(off, buf) {
                Ok(()) => return Ok(()),
                // A server that ignores Range will ignore it on every retry.
                Err(e) if e.kind() == io::ErrorKind::InvalidData => return Err(e),
                Err(e) if attempt < self.retries => {
                    attempt += 1;
                    std::thread::sleep(std::time::Duration::from_millis(200 << (attempt - 1).min(4)));
                    eprintln!("[prepare] range {off}+{} failed ({e}); retry {attempt}/{}", buf.len(), self.retries);
                }
                Err(e) => return Err(e),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;
    use std::net::TcpListener;

    #[test]
    fn local_reads_are_bounded() {
        let dir = std::env::temp_dir().join(format!("dsw-src-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("f");
        std::fs::write(&p, (0u8..=255).collect::<Vec<_>>()).unwrap();
        let s = LocalFile::open(&p).unwrap();
        assert_eq!(s.len(), 256);
        let mut b = [0u8; 4];
        s.read_at(250, &mut b).unwrap();
        assert_eq!(b, [250, 251, 252, 253]);
        assert!(s.read_at(254, &mut b).is_err(), "a read past the end is an error");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_url_pins_the_revision() {
        assert_eq!(
            hub_url("https://huggingface.co/", "deepseek-ai/DeepSeek-V4.1-Flash-1017", "1bae0e10", "model-00047-of-00048.safetensors"),
            "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/1bae0e10/model-00047-of-00048.safetensors"
        );
    }

    #[test]
    fn curl_is_told_the_exact_range_and_no_size_cap() {
        let a = curl_args("http://h/f", 100, 50, None);
        assert!(a.iter().any(|x| x == "Range: bytes=100-149"), "{a:?}");
        // A declared-size cap turns "this server ignores Range" into curl's
        // opaque (63); the reader detects that itself.
        assert!(!a.iter().any(|x| x == "--max-filesize"), "{a:?}");
        assert!(a.iter().any(|x| x == "Accept-Encoding: identity"));
        let a = curl_args("http://h/f", 0, 8, Some("hf_x"));
        assert!(a.iter().any(|x| x == "Authorization: Bearer hf_x"));
    }

    /// Serves one request: `206` with the asked range, or, when `ignore_range`,
    /// `200` with the whole body (the failure a proxy causes).
    fn serve_once(listener: TcpListener, body: Vec<u8>, ignore_range: bool) {
        let (mut s, _) = listener.accept().unwrap();
        let mut req = [0u8; 4096];
        let n = s.read(&mut req).unwrap();
        let text = String::from_utf8_lossy(&req[..n]).to_string();
        let (a, b) = text
            .lines()
            .find_map(|l| l.strip_prefix("range: bytes=").or_else(|| l.strip_prefix("Range: bytes=")))
            .map(|r| {
                let (a, b) = r.split_once('-').unwrap();
                (a.trim().parse::<usize>().unwrap(), b.trim().parse::<usize>().unwrap())
            })
            .unwrap();
        if ignore_range {
            let head = format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
            s.write_all(head.as_bytes()).unwrap();
            s.write_all(&body).unwrap();
        } else {
            let part = &body[a..=b];
            let head = format!(
                "HTTP/1.1 206 Partial Content\r\nContent-Range: bytes {a}-{b}/{}\r\nContent-Length: {}\r\nConnection: close\r\n\r\n",
                body.len(),
                part.len()
            );
            s.write_all(head.as_bytes()).unwrap();
            s.write_all(part).unwrap();
        }
    }

    #[test]
    fn http_reads_only_the_range_and_rejects_a_server_that_ignores_it() {
        let body: Vec<u8> = (0..4096u32).map(|i| (i % 251) as u8).collect();
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/shard", l.local_addr().unwrap());
        let b2 = body.clone();
        let t = std::thread::spawn(move || serve_once(l, b2, false));
        let s = HttpRange::new(url.clone(), body.len() as u64, None, 0);
        let mut got = [0u8; 64];
        s.read_at(1000, &mut got).unwrap();
        t.join().unwrap();
        assert_eq!(got, body[1000..1064]);

        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/shard", l.local_addr().unwrap());
        let b3 = body.clone();
        let t = std::thread::spawn(move || serve_once(l, b3, true));
        // Two retries are allowed: a 200 answer must fail once, not three times
        // (the server is gone after one request, so a retry could only report
        // "connection refused" instead).
        let s = HttpRange::new(url, body.len() as u64, None, 2);
        let err = s.read_at(0, &mut got).unwrap_err();
        t.join().unwrap();
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidData, "{err}");
        assert!(err.to_string().contains("does not honour Range"), "a 200 answer must name the cause: {err}");
    }
}
