//! The durable record of what is already on disk.
//!
//! One JSON object per line, appended and `fsync`ed before the source it used is
//! released. The order is what makes a crash safe: a batch is only marked
//! committed after its outputs are written, verified and synced, so a restart
//! never needs a source range that was already freed. A torn last line (the
//! process died mid-append) is discarded on load; every earlier line stands.

use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct OutputCommit {
    pub path: String,
    pub offset: u64,
    pub len: u64,
    pub crc32c: u32,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum Record {
    /// Written once: what this journal describes. A mismatch means the journal
    /// belongs to another plan and must not be reused.
    Begin {
        version: u32,
        fingerprint: String,
        rank: u32,
        world: u32,
        out: String,
    },
    /// A batch whose outputs are written, verified and synced.
    Commit { batch: String, outputs: Vec<OutputCommit> },
    /// The batch's source ranges have been released.
    Reclaim { batch: String },
    /// Every batch is committed and the outputs are finalised.
    Done { file: String, bytes: u64, tensors: u64 },
}

pub const VERSION: u32 = 1;

pub struct Journal {
    path: PathBuf,
    file: File,
}

#[derive(Debug, Default)]
pub struct State {
    pub committed: HashSet<String>,
    pub reclaimed: HashSet<String>,
    pub done: bool,
}

impl Journal {
    pub fn create(path: &Path) -> std::io::Result<Journal> {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir)?;
        }
        let file = OpenOptions::new().create(true).append(true).read(true).open(path)?;
        Ok(Journal { path: path.to_path_buf(), file })
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Appends one record and makes it durable. Callers must have synced the
    /// outputs it names first.
    pub fn append(&mut self, r: &Record) -> std::io::Result<()> {
        let mut line = serde_json::to_vec(r).expect("a journal record is serialisable");
        line.push(b'\n');
        self.file.write_all(&line)?;
        self.file.sync_data()
    }

    /// Reads the journal back. Returns the records and the state they describe;
    /// a truncated final line is dropped.
    pub fn load(path: &Path) -> std::io::Result<(Vec<Record>, State)> {
        let mut out = Vec::new();
        let mut state = State::default();
        let f = match File::open(path) {
            Ok(f) => f,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((out, state)),
            Err(e) => return Err(e),
        };
        let mut reader = BufReader::new(f);
        let mut line = Vec::new();
        loop {
            line.clear();
            let n = reader.read_until(b'\n', &mut line)?;
            if n == 0 {
                break;
            }
            let complete = line.last() == Some(&b'\n');
            if !complete {
                break; // torn append: the record was never durable
            }
            let text = std::str::from_utf8(&line).ok();
            let Some(text) = text else { break };
            match serde_json::from_str::<Record>(text) {
                Ok(r) => {
                    match &r {
                        Record::Commit { batch, .. } => {
                            state.committed.insert(batch.clone());
                        }
                        Record::Reclaim { batch } => {
                            state.reclaimed.insert(batch.clone());
                        }
                        Record::Done { .. } => state.done = true,
                        Record::Begin { .. } => {}
                    }
                    out.push(r);
                }
                Err(_) => break, // the first unparsable line ends the valid prefix
            }
        }
        Ok((out, state))
    }

    /// Checks that this journal describes `fingerprint` and this rank, so a
    /// resumed run never mixes two plans.
    pub fn begin_matches(records: &[Record], fingerprint: &str, rank: u32, world: u32) -> Result<(), String> {
        let begin = records.iter().find_map(|r| match r {
            Record::Begin { version, fingerprint: f, rank: r, world: w, .. } => Some((*version, f.clone(), *r, *w)),
            _ => None,
        });
        let Some((version, f, r, w)) = begin else {
            return Err("the journal has no begin record".into());
        };
        if version != VERSION {
            return Err(format!("the journal is version {version}, this build writes {VERSION}"));
        }
        if f != fingerprint {
            return Err("the journal was written for another plan (checkpoint or split changed)".into());
        }
        if r != rank || w != world {
            return Err(format!("the journal is for rank {r} of {w}, not rank {rank} of {world}"));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmp(name: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("dsw-journal-{}-{name}", std::process::id()));
        let _ = std::fs::remove_dir_all(&d);
        std::fs::create_dir_all(&d).unwrap();
        d.join("journal.jsonl")
    }

    #[test]
    fn commits_survive_a_restart_and_reclaim_is_idempotent() {
        let p = tmp("roundtrip");
        {
            let mut j = Journal::create(&p).unwrap();
            j.append(&Record::Begin { version: VERSION, fingerprint: "fp".into(), rank: 0, world: 4, out: "/x".into() })
                .unwrap();
            j.append(&Record::Commit {
                batch: "t7".into(),
                outputs: vec![OutputCommit { path: "/x/rank0.dsw".into(), offset: 4096, len: 512, crc32c: 0x1234 }],
            })
            .unwrap();
            j.append(&Record::Reclaim { batch: "t7".into() }).unwrap();
        }
        let (records, state) = Journal::load(&p).unwrap();
        assert_eq!(records.len(), 3);
        assert!(state.committed.contains("t7"));
        assert!(state.reclaimed.contains("t7"));
        assert!(!state.done);
        Journal::begin_matches(&records, "fp", 0, 4).unwrap();
        assert!(Journal::begin_matches(&records, "other", 0, 4).is_err());
        assert!(Journal::begin_matches(&records, "fp", 1, 4).is_err());
        let _ = std::fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn a_torn_last_line_is_discarded() {
        let p = tmp("torn");
        {
            let mut j = Journal::create(&p).unwrap();
            j.append(&Record::Begin { version: VERSION, fingerprint: "fp".into(), rank: 0, world: 4, out: "/x".into() })
                .unwrap();
            j.append(&Record::Commit { batch: "a".into(), outputs: vec![] }).unwrap();
        }
        let mut text = std::fs::read_to_string(&p).unwrap();
        text.push_str("{\"kind\":\"commit\",\"batch\":\"b\",\"outp");
        std::fs::write(&p, text).unwrap();
        let (_, state) = Journal::load(&p).unwrap();
        assert!(state.committed.contains("a"));
        assert!(!state.committed.contains("b"), "an unparsable line must not count");
        let _ = std::fs::remove_dir_all(p.parent().unwrap());
    }

    #[test]
    fn a_missing_journal_is_empty_not_an_error() {
        let p = std::env::temp_dir().join("dsw-journal-does-not-exist.jsonl");
        let (records, state) = Journal::load(&p).unwrap();
        assert!(records.is_empty());
        assert!(state.committed.is_empty());
    }
}
