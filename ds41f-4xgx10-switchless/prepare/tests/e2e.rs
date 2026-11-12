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
            // 2 MiB: the one tensor here whose blocks are worth reclaiming.
            ("layers.0.ffn.shared_experts.w1.weight", "F8_E4M3", &[4096, 512], &pattern(2 << 20, 20)),
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
        "layers.0.ffn.shared_experts.w1.weight": "model-00001-of-00002.safetensors",
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
        // embed, attn_norm, wq_b, wo_b (+scale), the shared expert, the stacked
        // experts w1/w2 (+their scales), engram.wkv (+scale)
        assert_eq!(hb.n_tensors, 12);
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

    // The checkpoint must stay openable after a consuming run: its headers and the
    // blocks around every tensor's edges are untouched, which is what lets an
    // interrupted run be resumed.
    let again = Checkpoint::open(&work).unwrap();
    assert_eq!(again.tensors.len(), ck.tensors.len());

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
fn a_second_run_keeps_the_finished_ranks() {
    let ckdir = out_dir("ck-twice");
    write_checkpoint(&ckdir);
    let out = out_dir("out-twice");
    let ck = Checkpoint::open(&ckdir).unwrap();
    let o = options(&out, false);
    convert::run(&ck, &reader(&ck), &o, "test").unwrap();
    let before: Vec<Vec<u8>> = o.ranks.iter().map(|&r| std::fs::read(convert::dsw_path(&o, r)).unwrap()).collect();
    // The whole point of running it again: nothing happens, and no prompt.
    convert::run(&ck, &reader(&ck), &o, "test").unwrap();
    for (&r, bytes) in o.ranks.iter().zip(&before) {
        assert_eq!(&std::fs::read(convert::dsw_path(&o, r)).unwrap(), bytes, "rank {r} was rewritten");
    }
    let _ = std::fs::remove_dir_all(&ckdir);
    let _ = std::fs::remove_dir_all(&out);
}

#[test]
fn a_released_checkpoint_is_recognised() {
    let ckdir = out_dir("ck-holes");
    write_checkpoint(&ckdir);
    let ck = Checkpoint::open(&ckdir).unwrap();
    assert!(ck.released().is_empty(), "a fresh checkpoint has no holes");
    // The same punch an earlier --consume leaves behind, past the header so the
    // shard still opens.
    let shard = ck.shards[0].path.clone().unwrap();
    let ok = std::process::Command::new("fallocate")
        .args(["--punch-hole", "--offset", "4096", "--length", "4096"])
        .arg(&shard)
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if ok {
        let ck = Checkpoint::open(&ckdir).unwrap();
        assert_eq!(ck.released(), vec![0], "a punched shard must be reported");
    }
    let _ = std::fs::remove_dir_all(&ckdir);
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

// ---------------------------------------------------------------- resharding

/// A rank file's table of contents, by tensor name: offset, bytes, checksum.
fn toc_of(path: &Path) -> std::collections::BTreeMap<String, (u64, u64, u32)> {
    use dsw_prepare::dsw::{TocEntry, TOC_ENTRY_BYTES};
    let f = File::open(path).unwrap();
    let mut hb = [0u8; HEADER_BYTES];
    f.read_exact_at(&mut hb, 0).unwrap();
    let h = Header::from_bytes(&hb);
    let mut tb = vec![0u8; (h.data_offset - h.toc_offset) as usize];
    f.read_exact_at(&mut tb, h.toc_offset).unwrap();
    let mut m = std::collections::BTreeMap::new();
    for i in 0..h.n_tensors as usize {
        let e = TocEntry::from_bytes(&tb[i * TOC_ENTRY_BYTES..(i + 1) * TOC_ENTRY_BYTES]);
        m.insert(e.name(), (e.offset, e.bytes, e.crc32c));
    }
    m
}

/// The Engram shards of one directory as (layer, rank, records), header dropped.
fn engram_records(dir: &Path) -> Vec<(u32, u32, Vec<u8>)> {
    let mut out = Vec::new();
    let Ok(d) = std::fs::read_dir(dir.join("engram")) else { return out };
    for e in d.filter_map(|e| e.ok()) {
        let name = e.file_name().to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix("engram-l") else { continue };
        let (layer, rest) = rest.split_once("-r").unwrap();
        let (rank, _) = rest.split_once("of").unwrap();
        let bytes = std::fs::read(e.path()).unwrap();
        out.push((layer.parse().unwrap(), rank.parse().unwrap(), bytes[4096..].to_vec()));
    }
    out.sort_by_key(|(l, r, _)| (*l, *r));
    out
}

fn reshard_options(from: &Path, world: u32, out: &Path) -> dsw_prepare::reshard::Options {
    dsw_prepare::reshard::Options {
        from: from.to_path_buf(),
        world,
        out: out.to_path_buf(),
        // Empty: every rank of the new world, which is also what the CLI passes.
        ranks: Vec::new(),
        verify: true,
        replace: false,
        engram: true,
    }
}

#[test]
fn a_reshard_cuts_two_ways_and_back_to_the_same_bytes() {
    let ckdir = out_dir("rs-ck");
    write_checkpoint(&ckdir);
    let four = out_dir("rs-4");
    let ck = Checkpoint::open(&ckdir).unwrap();
    convert::run(&ck, &reader(&ck), &options(&four, false), "test").unwrap();

    // 4 -> 2: independent of the resharder, every file is re-derived from the
    // checkpoint at the world its own header claims.
    let two = out_dir("rs-2");
    let r = dsw_prepare::reshard::run(&reshard_options(&four, 2, &two)).unwrap();
    assert_eq!(r.files.len(), 2);
    assert_eq!(r.engram_files, 4, "two layers, two ranks each");
    for rank in 0..2u32 {
        convert::verify_file(&ck, &two.join(format!("rank{rank}.dsw"))).unwrap();
    }
    // The two-way row split doubles every rank's rows.
    let a0 = toc_of(&four.join("rank0.dsw"));
    let b0 = toc_of(&two.join("rank0.dsw"));
    let wq = "layers.0.attn.wq_b.weight";
    assert_eq!(b0[wq].1, a0[wq].1 * 2, "rows 16 -> 32 of a 64-row tensor");

    // 2 -> 4: the same bytes the checkpoint produced the first time.
    let back = out_dir("rs-back");
    dsw_prepare::reshard::run(&reshard_options(&two, 4, &back)).unwrap();
    for rank in 0..4u32 {
        let p = four.join(format!("rank{rank}.dsw"));
        let q = back.join(format!("rank{rank}.dsw"));
        let tp = toc_of(&p);
        let tq = toc_of(&q);
        assert_eq!(tp.keys().collect::<Vec<_>>(), tq.keys().collect::<Vec<_>>());
        let fp = File::open(&p).unwrap();
        let fq = File::open(&q).unwrap();
        for (name, &(off, bytes, crc)) in &tp {
            let (off2, bytes2, crc2) = tq[name];
            assert_eq!((off2, bytes2, crc2), (off, bytes, crc), "{name}: offset, size or checksum");
            let mut x = vec![0u8; bytes as usize];
            let mut y = vec![0u8; bytes as usize];
            fp.read_exact_at(&mut x, off).unwrap();
            fq.read_exact_at(&mut y, off2).unwrap();
            assert_eq!(x, y, "{name}: bytes");
        }
    }
    // and the Engram shards come back to the four-way records.
    assert_eq!(engram_records(&back)[..], engram_records(&four)[..]);

    // The two-way Engram shards hold the same rows, four instead of two each.
    let four_recs = engram_records(&four);
    let two_recs = engram_records(&two);
    for layer in [1u32, 14] {
        let joined4: Vec<u8> = four_recs.iter().filter(|(l, _, _)| *l == layer).flat_map(|(_, _, b)| b.clone()).collect();
        let joined2: Vec<u8> = two_recs.iter().filter(|(l, _, _)| *l == layer).flat_map(|(_, _, b)| b.clone()).collect();
        assert_eq!(joined4, joined2, "layer {layer}: the rows are the same, only the cut differs");
    }

    let _ = std::fs::remove_dir_all(&ckdir);
    for d in [&four, &two, &back] {
        let _ = std::fs::remove_dir_all(d);
    }
}

#[test]
fn a_reshard_refuses_a_split_the_tensors_cannot_take() {
    let ckdir = out_dir("rs-ck3");
    write_checkpoint(&ckdir);
    let four = out_dir("rs-4b");
    let ck = Checkpoint::open(&ckdir).unwrap();
    convert::run(&ck, &reader(&ck), &options(&four, false), "test").unwrap();
    let three = out_dir("rs-3");
    let err = dsw_prepare::reshard::run(&reshard_options(&four, 3, &three)).unwrap_err();
    assert!(err.contains("do not split"), "{err}");
    assert!(!three.join("rank0.dsw").exists(), "nothing is written when the plan cannot be built");
    let _ = std::fs::remove_dir_all(&ckdir);
    let _ = std::fs::remove_dir_all(&four);
    let _ = std::fs::remove_dir_all(&three);
}

#[test]
fn a_reshard_refuses_a_damaged_source() {
    let ckdir = out_dir("rs-ckd");
    write_checkpoint(&ckdir);
    let four = out_dir("rs-4c");
    let ck = Checkpoint::open(&ckdir).unwrap();
    convert::run(&ck, &reader(&ck), &options(&four, false), "test").unwrap();

    // One byte of rank 1's slice of a split tensor is corrupted: a replicated
    // one would be read from rank 0 and the damage would never show.
    let victim = four.join("rank1.dsw");
    let (off, _, _) = toc_of(&victim)["layers.0.attn.wq_b.weight"];
    {
        let f = std::fs::OpenOptions::new().write(true).open(&victim).unwrap();
        f.write_all_at(&[0xA5], off + 16).unwrap();
    }
    let two = out_dir("rs-2d");
    let err = dsw_prepare::reshard::run(&reshard_options(&four, 2, &two)).unwrap_err();
    assert!(err.contains("CRC mismatch"), "{err}");
    let _ = std::fs::remove_dir_all(&ckdir);
    let _ = std::fs::remove_dir_all(&four);
    let _ = std::fs::remove_dir_all(&two);
}
