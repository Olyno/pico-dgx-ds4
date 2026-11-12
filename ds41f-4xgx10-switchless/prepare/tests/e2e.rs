//! End to end: a synthetic checkpoint in, four verified rank files and their
//! Engram shards out — then resume, then reclamation, each checked against the
//! independent verifier and against the checkpoint's own bytes.

use std::collections::HashMap;
use std::fs::File;
use std::os::unix::fs::{FileExt, MetadataExt};
use std::path::{Path, PathBuf};

use dsw_prepare::convert::{self, Options};
use dsw_prepare::dsw::{Header, ALIGN, HEADER_BYTES};
use dsw_prepare::engram;
use dsw_prepare::journal::{Journal, Record};
use dsw_prepare::safetensors::Checkpoint;
use dsw_prepare::source::{HttpRange, HybridReader, LocalReader};

/// Writes one safetensors shard: `u64` header length, the JSON header, the bytes.
/// Deliberately its own implementation, so the reader is tested against a writer
/// that shares no code with it.
fn write_shard(path: &Path, tensors: &[(&str, &str, &[u64], &[u8])]) {
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
    let text = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = Vec::new();
    out.extend_from_slice(&(text.len() as u64).to_le_bytes());
    out.extend_from_slice(&text);
    out.extend_from_slice(&data);
    std::fs::write(path, out).unwrap();
}

/// A checkpoint that exercises every partition rule and both Engram layers.
/// Sizes are chosen so each split divides and each tiled layout fits.
fn write_checkpoint(dir: &Path) {
    let _ = std::fs::remove_dir_all(dir);
    std::fs::create_dir_all(dir).unwrap();
    let a = dir.join("model-00001-of-00002.safetensors");
    let b = dir.join("model-00002-of-00002.safetensors");
    write_shard(
        &a,
        &[
            ("embed.weight", "BF16", &[8, 4], &pattern(64, 1)),
            ("layers.0.attn_norm.weight", "BF16", &[8], &pattern(16, 2)),
            ("layers.0.attn.wq_b.weight", "F8_E4M3", &[64, 64], &pattern(4096, 3)),
            ("layers.0.attn.wo_b.weight", "F8_E4M3", &[64, 256], &pattern(16384, 4)),
            ("layers.0.attn.wo_b.scale", "F8_E8M0", &[2, 8], &pattern(16, 5)),
            ("layers.0.ffn.experts.0.w1.weight", "I8", &[64, 64], &pattern(4096, 6)),
            ("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", &[64, 4], &pattern(256, 14)),
            ("layers.0.ffn.experts.0.w2.weight", "I8", &[64, 128], &pattern(8192, 16)),
            ("layers.0.ffn.experts.0.w2.scale", "F8_E8M0", &[64, 8], &pattern(512, 17)),
            ("layers.0.ffn.experts.1.w1.weight", "I8", &[64, 64], &pattern(4096, 7)),
            ("layers.0.ffn.experts.1.w1.scale", "F8_E8M0", &[64, 4], &pattern(256, 15)),
            ("layers.0.ffn.experts.1.w2.weight", "I8", &[64, 128], &pattern(8192, 18)),
            ("layers.0.ffn.experts.1.w2.scale", "F8_E8M0", &[64, 8], &pattern(512, 19)),
            ("layers.0.engram.wkv.weight", "F8_E4M3", &[640, 128], &pattern(81920, 8)),
            ("layers.0.engram.wkv.scale", "F8_E8M0", &[20, 4], &pattern(80, 9)),
        ],
    );
    write_shard(
        &b,
        &[
            ("layers.1.engram.embed.weight", "F8_E4M3", &[8, 256], &pattern(2048, 10)),
            ("layers.1.engram.embed.scale", "F8_E8M0", &[8, 8], &pattern(64, 11)),
            ("layers.14.engram.embed.weight", "F8_E4M3", &[8, 256], &pattern(2048, 12)),
            ("layers.14.engram.embed.scale", "F8_E8M0", &[8, 8], &pattern(64, 13)),
        ],
    );
    let index = serde_json::json!({"metadata": {"total_size": 0}, "weight_map": {
        "embed.weight": "model-00001-of-00002.safetensors",
        "layers.0.attn_norm.weight": "model-00001-of-00002.safetensors",
        "layers.0.attn.wq_b.weight": "model-00001-of-00002.safetensors",
        "layers.0.attn.wo_b.weight": "model-00001-of-00002.safetensors",
        "layers.0.attn.wo_b.scale": "model-00001-of-00002.safetensors",
        "layers.0.ffn.experts.0.w1.weight": "model-00001-of-00002.safetensors",
        "layers.0.ffn.experts.0.w1.scale": "model-00001-of-00002.safetensors",
        "layers.0.ffn.experts.0.w2.weight": "model-00001-of-00002.safetensors",
        "layers.0.ffn.experts.0.w2.scale": "model-00001-of-00002.safetensors",
        "layers.0.ffn.experts.1.w1.weight": "model-00001-of-00002.safetensors",
        "layers.0.ffn.experts.1.w1.scale": "model-00001-of-00002.safetensors",
        "layers.0.ffn.experts.1.w2.weight": "model-00001-of-00002.safetensors",
        "layers.0.ffn.experts.1.w2.scale": "model-00001-of-00002.safetensors",
        "layers.0.engram.wkv.weight": "model-00001-of-00002.safetensors",
        "layers.0.engram.wkv.scale": "model-00001-of-00002.safetensors",
        "layers.1.engram.embed.weight": "model-00002-of-00002.safetensors",
        "layers.1.engram.embed.scale": "model-00002-of-00002.safetensors",
        "layers.14.engram.embed.weight": "model-00002-of-00002.safetensors",
        "layers.14.engram.embed.scale": "model-00002-of-00002.safetensors",
    }});
    std::fs::write(dir.join("model.safetensors.index.json"), serde_json::to_vec(&index).unwrap()).unwrap();
    std::fs::write(dir.join("config.json"), b"{}").unwrap();
    std::fs::write(dir.join("tokenizer.json"), b"{}").unwrap();
}

/// Deterministic, non-repeating bytes: a permutation or a slice bug changes them.
fn pattern(n: usize, seed: u64) -> Vec<u8> {
    (0..n)
        .map(|i| (i as u64 * 37 + seed * 101 + (i as u64 >> 3)) as u8)
        .collect()
}

fn out_dir(tag: &str) -> PathBuf {
    let d = std::env::temp_dir().join(format!("dsw-e2e-{}-{tag}", std::process::id()));
    let _ = std::fs::remove_dir_all(&d);
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn reader(ck: &Checkpoint) -> LocalReader {
    let paths: Vec<PathBuf> = ck.shards.iter().map(|s| s.path.clone().expect("local")).collect();
    LocalReader::open(&paths).unwrap()
}

fn options(out: &Path, consume: bool) -> Options {
    Options {
        world: 4,
        ranks: vec![0, 1, 2, 3],
        out: out.to_path_buf(),
        dest: HashMap::new(),
        consume,
        verify: true,
        resume: true,
        engram: true,
        vision: None,
        model: "deepseek-v4.1-flash".into(),
    }
}

#[test]
fn converts_all_ranks_and_every_output_verifies() {
    let ckdir = out_dir("ck");
    write_checkpoint(&ckdir);
    let out = out_dir("out");
    let ck = Checkpoint::open(&ckdir).unwrap();
    let o = options(&out, false);
    convert::run(&ck, &reader(&ck), &o, &ckdir.display().to_string()).unwrap();

    for r in 0..4u32 {
        let f = convert::dsw_path(&o, r);
        assert!(f.exists(), "{} missing", f.display());
        let hb = {
            let mut b = [0u8; HEADER_BYTES];
            File::open(&f).unwrap().read_exact_at(&mut b, 0).unwrap();
            Header::from_bytes(&b)
        };
        assert_eq!(hb.rank, r);
        assert_eq!(hb.world, 4);
        assert_eq!(&hb.magic, b"DSGB10W1");
        // embed, attn_norm, wq_b, wo_b (+scale), the stacked experts w1/w2 (+their
        // scales), engram.wkv (+scale)
        assert_eq!(hb.n_tensors, 11);
        assert_eq!(hb.file_bytes, f.metadata().unwrap().len());
        convert::verify_file(&ck, &f).unwrap_or_else(|e| panic!("rank {r}: {e}"));

        // Engram shards: header, size, and the rows this rank owns.
        for layer in [1u32, 14] {
            let p = convert::engram_path(&o, r, layer);
            let meta = p.metadata().unwrap_or_else(|e| panic!("{}: {e}", p.display()));
            let mut hdr = [0u8; 4096];
            File::open(&p).unwrap().read_exact_at(&mut hdr, 0).unwrap();
            assert_eq!(&hdr[..8], b"DSV1EN41");
            let word = |i: usize| u64::from_le_bytes(hdr[i * 8..i * 8 + 8].try_into().unwrap());
            assert_eq!(word(1), layer as u64);
            assert_eq!(word(2), 2 * r as u64); // 8 rows, 4 ranks
            assert_eq!(word(3), 2 * (r as u64 + 1));
            assert_eq!(word(4), 8);
            assert_eq!(meta.len(), engram::HEADER_BYTES + 2 * engram::ROW_BYTES);
            // The record's weight bytes are the checkpoint's row.
            let w = ck.tensor(&format!("layers.{layer}.engram.embed.weight"));
            let mut rec = [0u8; 264];
            File::open(&p).unwrap().read_exact_at(&mut rec, 4096).unwrap();
            let (gw, _gs) = engram::split_record(&rec);
            let mut want = vec![0u8; 256];
            File::open(ck.shards[w.shard].path.as_ref().unwrap())
                .unwrap()
                .read_exact_at(&mut want, w.offset + (2 * r as u64) * 256)
                .unwrap();
            assert_eq!(gw, &want[..], "rank {r} layer {layer}: first record");
        }
    }
    let _ = std::fs::remove_dir_all(&ckdir);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_crash_before_the_last_commit_replays_only_that_batch() {
    let ckdir = out_dir("ck-resume");
    write_checkpoint(&ckdir);
    let out = out_dir("out-resume");
    let ck = Checkpoint::open(&ckdir).unwrap();
    let o = options(&out, false);
    convert::run(&ck, &reader(&ck), &o, "test").unwrap();
    for r in 0..4u32 {
        convert::verify_file(&ck, &convert::dsw_path(&o, r)).unwrap();
    }

    // Simulate a crash after the outputs were written but before the last two
    // journal records landed: the file is renamed, the journal says "in progress".
    for r in 0..4u32 {
        let j = convert::journal_path(&o, r);
        let text = std::fs::read_to_string(&j).unwrap();
        let mut lines: Vec<&str> = text.lines().collect();
        lines.pop(); // Done
        lines.pop(); // the last Commit
        std::fs::write(&j, format!("{}\n", lines.join("\n"))).unwrap();
        // The final file is in place, not the .part: the resumed run must reopen it.
        assert!(convert::dsw_path(&o, r).exists());
    }
    let (_, state) = Journal::load(&convert::journal_path(&o, 0)).unwrap();
    assert!(!state.done);

    convert::run(&ck, &reader(&ck), &o, "test").unwrap();
    for r in 0..4u32 {
        convert::verify_file(&ck, &convert::dsw_path(&o, r)).unwrap();
        let (_, state) = Journal::load(&convert::journal_path(&o, r)).unwrap();
        assert!(state.done, "rank {r} did not finish");
    }
    let _ = std::fs::remove_dir_all(&ckdir);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn consume_releases_the_source_and_the_outputs_still_match() {
    let pristine = out_dir("ck-pristine");
    write_checkpoint(&pristine);
    let work = out_dir("ck-consume");
    write_checkpoint(&work);
    let out = out_dir("out-consume");
    let ck = Checkpoint::open(&work).unwrap();
    let before: Vec<u64> = ck
        .shards
        .iter()
        .map(|s| File::open(s.path.as_ref().unwrap()).unwrap().metadata().unwrap().blocks())
        .collect();
    let o = options(&out, true);
    convert::run(&ck, &reader(&ck), &o, "test").unwrap();

    let after: Vec<u64> = ck
        .shards
        .iter()
        .map(|s| File::open(s.path.as_ref().unwrap()).unwrap().metadata().unwrap().blocks())
        .collect();
    // The headers stay (a few blocks); the tensor payload must be gone.
    let (b0, a0) = (before.iter().sum::<u64>(), after.iter().sum::<u64>());
    assert!(a0 * 10 < b0, "the source was not released: {before:?} -> {after:?}");

    // The outputs are checked against an untouched copy of the checkpoint.
    let clean = Checkpoint::open(&pristine).unwrap();
    for r in 0..4u32 {
        convert::verify_file(&clean, &convert::dsw_path(&o, r)).unwrap();
    }
    let _ = std::fs::remove_dir_all(&pristine);
    let _ = std::fs::remove_dir_all(&work);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn consuming_requires_every_rank() {
    let ckdir = out_dir("ck-partial");
    write_checkpoint(&ckdir);
    let out = out_dir("out-partial");
    let ck = Checkpoint::open(&ckdir).unwrap();
    let mut o = options(&out, true);
    o.ranks = vec![0];
    let err = convert::run(&ck, &reader(&ck), &o, "test").unwrap_err();
    assert!(err.contains("every rank"), "{err}");
    let _ = std::fs::remove_dir_all(&ckdir);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_tampered_output_fails_verification() {
    let ckdir = out_dir("ck-tamper");
    write_checkpoint(&ckdir);
    let out = out_dir("out-tamper");
    let ck = Checkpoint::open(&ckdir).unwrap();
    let o = options(&out, false);
    convert::run(&ck, &reader(&ck), &o, "test").unwrap();
    let f = convert::dsw_path(&o, 0);
    // Flip one byte inside the first tensor's data.
    let meta = f.metadata().unwrap();
    let mut buf = [0u8; 1];
    let fh = std::fs::OpenOptions::new().read(true).write(true).open(&f).unwrap();
    fh.read_exact_at(&mut buf, ALIGN).unwrap();
    buf[0] ^= 0xFF;
    fh.write_all_at(&buf, ALIGN).unwrap();
    drop(fh);
    assert!(meta.len() > ALIGN);
    let err = convert::verify_file(&ck, &f).unwrap_err();
    assert!(err.contains("CRC") || err.contains("differ"), "{err}");
    let _ = std::fs::remove_dir_all(&ckdir);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_second_run_refuses_to_overwrite_a_finished_file() {
    let ckdir = out_dir("ck-twice");
    write_checkpoint(&ckdir);
    let out = out_dir("out-twice");
    let ck = Checkpoint::open(&ckdir).unwrap();
    let o = options(&out, false);
    convert::run(&ck, &reader(&ck), &o, "test").unwrap();
    let err = convert::run(&ck, &reader(&ck), &o, "test").unwrap_err();
    assert!(err.contains("already complete"), "{err}");
    let _ = std::fs::remove_dir_all(&ckdir);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn the_journal_begin_record_is_written_before_any_output() {
    let ckdir = out_dir("ck-begin");
    write_checkpoint(&ckdir);
    let out = out_dir("out-begin");
    let ck = Checkpoint::open(&ckdir).unwrap();
    let o = options(&out, false);
    convert::run(&ck, &reader(&ck), &o, "test").unwrap();
    let (records, _) = Journal::load(&convert::journal_path(&o, 2)).unwrap();
    assert!(matches!(records.first(), Some(Record::Begin { rank: 2, world: 4, .. })));
    let _ = std::fs::remove_dir_all(&ckdir);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn hybrid_downloads_whole_shards_and_leaves_no_scratch() {
    use std::io::{Read, Write};

    let ckdir = out_dir("ck-hybrid");
    write_checkpoint(&ckdir);
    let out = out_dir("out-hybrid");
    let scratch = out.join("scratch");

    // The shards served as whole files: a hybrid run downloads each once instead
    // of issuing a request per tensor.
    let bodies: Vec<(String, Vec<u8>)> = std::fs::read_dir(&ckdir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().map(|x| x == "safetensors").unwrap_or(false))
        .map(|e| (e.file_name().to_string_lossy().into_owned(), std::fs::read(e.path()).unwrap()))
        .collect();
    let listener = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
    let addr = listener.local_addr().unwrap();
    let served = bodies.clone();
    std::thread::spawn(move || {
        for s in listener.incoming() {
            let Ok(mut s) = s else { continue };
            let mut req = [0u8; 2048];
            let n = s.read(&mut req).unwrap_or(0);
            let text = String::from_utf8_lossy(&req[..n]).to_string();
            let path = text.split_whitespace().nth(1).unwrap_or("/").trim_start_matches('/').to_string();
            match served.iter().find(|(name, _)| *name == path) {
                Some((_, body)) => {
                    let head =
                        format!("HTTP/1.1 200 OK\r\nContent-Length: {}\r\nConnection: close\r\n\r\n", body.len());
                    let _ = s.write_all(head.as_bytes());
                    let _ = s.write_all(body);
                }
                None => {
                    let _ = s.write_all(b"HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n");
                }
            }
        }
    });

    // The plan comes from the headers alone, as it does in a streaming run.
    let mut items = Vec::new();
    let mut remote = Vec::new();
    let mut names = Vec::new();
    for (name, body) in &bodies {
        let (len, header) = dsw_prepare::safetensors::parse_header(body).unwrap();
        items.push((name.clone(), len, header));
        remote.push(HttpRange::new(format!("http://{addr}/{name}"), body.len() as u64, None, 2));
        names.push(name.clone());
    }
    let ck = Checkpoint::from_shard_headers(items).unwrap();
    let reader = HybridReader::new(remote, names, scratch.clone(), 64 << 20);
    let o = options(&out, false);
    convert::run(&ck, &reader, &o, "http").unwrap();

    let clean = Checkpoint::open(&ckdir).unwrap();
    for r in 0..4u32 {
        convert::verify_file(&clean, &convert::dsw_path(&o, r)).unwrap();
    }
    let left: Vec<String> = std::fs::read_dir(&scratch)
        .map(|d| d.filter_map(|e| e.ok()).map(|e| e.file_name().to_string_lossy().into_owned()).collect())
        .unwrap_or_default();
    assert!(left.is_empty(), "the scratch still holds {left:?}");

    let _ = std::fs::remove_dir_all(&ckdir);
    let _ = std::fs::remove_dir_all(&out);
}
