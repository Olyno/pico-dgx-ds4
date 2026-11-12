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
use std::path::Path;
use std::process::{Command, Stdio};

use crate::space;

/// A shard's bytes, addressed by absolute offset.
pub trait Source {
    fn len(&self) -> u64;
    /// Fills `buf` from `off`. Reads beyond the end are an error, not a short read.
    fn read_at(&self, off: u64, buf: &mut [u8]) -> io::Result<()>;
}

/// One shard, either on disk (where its blocks can be released) or remote.
pub enum ShardSource {
    Local(LocalFile),
    Http(HttpRange),
}

impl ShardSource {
    pub fn as_source(&self) -> &dyn Source {
        match self {
            ShardSource::Local(l) => l,
            ShardSource::Http(h) => h,
        }
    }

    /// Only a local shard has blocks to release.
    pub fn local(&self) -> Option<&LocalFile> {
        match self {
            ShardSource::Local(l) => Some(l),
            ShardSource::Http(_) => None,
        }
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
        // A proxy that ignores Range would stream the whole shard; abort instead.
        "--max-filesize".into(),
        (len + 1).to_string(),
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
        if !out.status.success() {
            let err = String::from_utf8_lossy(&out.stderr);
            return Err(io::Error::other(format!("curl: {}", err.trim())));
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
    fn curl_is_told_the_exact_range_and_a_size_cap() {
        let a = curl_args("http://h/f", 100, 50, None);
        assert!(a.iter().any(|x| x == "Range: bytes=100-149"), "{a:?}");
        let cap = a.iter().position(|x| x == "--max-filesize").unwrap();
        assert_eq!(a[cap + 1], "51");
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
        let s = HttpRange::new(url, body.len() as u64, None, 0);
        let err = s.read_at(0, &mut got).unwrap_err();
        t.join().unwrap();
        assert!(
            err.to_string().contains("Range ignored") || err.to_string().contains("curl"),
            "a 200 answer must fail, not silently fill the buffer: {err}"
        );
    }
}
