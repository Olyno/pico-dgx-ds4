//! `dsw-prepare`: cut the checkpoint into per-rank `.dsw` files and Engram
//! shards, on disk or straight from the Hub, under a bounded disk budget.
//!
//!   dsw-prepare prepare [--ckpt DIR] [--out DIR] [--world N] [--ranks 0,1,..]
//!                       [--source local|http] [--consume] [--replace]
//!                       [--no-verify] [--no-resume] [--no-engram] [--dest R=DIR]
//!   dsw-prepare verify --ckpt DIR --file FILE
//!   dsw-prepare plan [--ckpt DIR] [--world N] [--rank R]
//!   dsw-prepare status [--out DIR]

use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::{Path, PathBuf};

use dsw_prepare::convert::{self, Options};
use dsw_prepare::plan;
use dsw_prepare::safetensors::Checkpoint;
use dsw_prepare::source::{hub_url, HttpRange, HttpReader, HybridReader, LocalReader, Reader, Source};
use dsw_prepare::space;

const REPO: &str = "deepseek-ai/DeepSeek-V4.1-Flash-1017";
const REV: &str = "1bae0e10";
/// Headroom the run must leave on the output filesystem: metadata, journals,
/// filesystem overhead, and the last tensor in flight.
const RESERVE: u64 = 10 << 30;

/// The largest shard a hybrid run downloads whole. Above it (the two 101.5 GB
/// Engram tables) the shard is read by range instead, which keeps it off the disk.
const HYBRID_MAX_CACHED: u64 = 16 << 30;

fn die(msg: &str) -> ! {
    eprintln!("[dsw-prepare] ERROR: {msg}");
    std::process::exit(1);
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

fn usage() -> ! {
    eprintln!(
        "usage: dsw-prepare <prepare|verify|plan|status> [options]\n\
         \n\
         prepare  --ckpt DIR  --out DIR  [--world N] [--ranks LIST] [--source local|http]\n\
         \x20        [--consume] [--replace] [--no-verify] [--no-resume] [--no-engram]\n\
         \x20        [--dest R=DIR]... [--repo OWNER/NAME] [--rev SHA] [--endpoint URL]\n\
         verify   --ckpt DIR --file FILE\n\
         plan     [--ckpt DIR] [--world N] [--rank R]\n\
         status   [--out DIR]"
    );
    std::process::exit(2);
}

fn arg<'a>(args: &'a [String], i: &mut usize, what: &str) -> &'a str {
    *i += 1;
    if *i >= args.len() {
        die(&format!("{what} needs a value"));
    }
    &args[*i]
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    if args.is_empty() {
        usage();
    }
    let cmd = args[0].clone();
    let mut ckpt = home().join("DeepSeek-V4.1-Flash-1017");
    let mut out = home().join("dsgb10/convert");
    let mut world = 4u32;
    let mut ranks: Option<Vec<u32>> = None;
    let mut source = "local".to_string();
    let mut consume = false;
    let mut replace = false;
    let mut verify = true;
    let mut resume = true;
    let mut engram = true;
    let mut dest: HashMap<u32, PathBuf> = HashMap::new();
    let mut repo = REPO.to_string();
    let mut rev = REV.to_string();
    let mut endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    let mut file: Option<PathBuf> = None;
    let mut rank: Option<u32> = None;
    let mut dump = false;

    let mut i = 1;
    while i < args.len() {
        let a = args[i].as_str();
        match a {
            "--ckpt" => ckpt = PathBuf::from(arg(&args, &mut i, a)),
            "--out" => out = PathBuf::from(arg(&args, &mut i, a)),
            "--world" => world = arg(&args, &mut i, a).parse().unwrap_or_else(|_| die("--world: not a number")),
            "--ranks" => {
                let v = arg(&args, &mut i, a);
                let mut r = Vec::new();
                for p in v.split(',') {
                    let n = p.trim().parse::<u32>().unwrap_or_else(|_| die("--ranks: not a list of numbers"));
                    r.push(n);
                }
                r.sort_unstable();
                r.dedup();
                ranks = Some(r);
            }
            "--rank" => rank = Some(arg(&args, &mut i, a).parse().unwrap_or_else(|_| die("--rank: not a number"))),
            "--source" => {
                let v = arg(&args, &mut i, a);
                if !matches!(v, "local" | "http" | "hybrid") {
                    die("--source is local, http or hybrid");
                }
                source = v.into();
            }
            "--consume" => consume = true,
            "--replace" => replace = true,
            "--no-verify" => verify = false,
            "--no-resume" => resume = false,
            "--no-engram" => engram = false,
            "--dest" => {
                let v = arg(&args, &mut i, a);
                let (r, p) = v.split_once('=').unwrap_or_else(|| die("--dest is R=DIR"));
                dest.insert(r.parse().unwrap_or_else(|_| die("--dest: rank is not a number")), PathBuf::from(p));
            }
            "--repo" => repo = arg(&args, &mut i, a).into(),
            "--rev" => rev = arg(&args, &mut i, a).into(),
            "--endpoint" => endpoint = arg(&args, &mut i, a).into(),
            "--file" => file = Some(PathBuf::from(arg(&args, &mut i, a))),
            // One line per tensor, the format `plan_dump` prints: the two plans are
            // compared with `diff` on the same checkpoint.
            "--dump" => dump = true,
            "-h" | "--help" => usage(),
            other => die(&format!("unknown argument {other}")),
        }
        i += 1;
    }

    match cmd.as_str() {
        "prepare" => {
            let ranks = ranks.unwrap_or_else(|| (0..world).collect());
            let o = Options { world, ranks, out: out.clone(), dest, consume, verify, resume, engram, model: "deepseek-v4.1-flash".into() };
            prepare(&ckpt, &o, &source, &repo, &rev, &endpoint, replace);
        }
        "verify" => {
            let f = file.unwrap_or_else(|| die("verify needs --file"));
            let ck = Checkpoint::open(&ckpt).unwrap_or_else(|e| die(&e));
            match convert::verify_file(&ck, &f) {
                Ok(()) => println!("{}: every tensor matches the checkpoint", f.display()),
                Err(e) => die(&e),
            }
        }
        "plan" => {
            let ck = Checkpoint::open(&ckpt).unwrap_or_else(|e| die(&e));
            let r = rank.unwrap_or(0);
            let p = plan::make_plan(&ck, r, world).unwrap_or_else(|e| die(&e));
            if dump {
                println!(
                    "rank {r} of {world}: {} tensors, {} bytes, {} skipped",
                    p.tensors.len(),
                    p.file_bytes,
                    p.skipped.len()
                );
                for t in &p.tensors {
                    let shape: String = t.shape.iter().map(|d| format!("{d},")).collect();
                    println!(
                        "{}|{}|{}|{}|{}|{}|{}|{}|{}|{}|{}",
                        t.name,
                        t.dtype as u32,
                        t.layout as u32,
                        t.bytes,
                        t.offset,
                        t.shape.len(),
                        shape,
                        t.kind as u32,
                        t.nstack,
                        t.nseg,
                        t.source
                    );
                }
                plan::check_coverage(&ck, world).unwrap_or_else(|e| die(&e));
                println!("coverage ok");
                return;
            }
            println!(
                "rank {r} of {world}: {} tensors, {:.1} GB, {} skipped, data at {}",
                p.tensors.len(),
                p.file_bytes as f64 / 1e9,
                p.skipped.len(),
                p.data_offset
            );
            for t in &p.tensors {
                println!(
                    "  {:<48} {:<8} {:>6} {:<10} {:>12} bytes  {}",
                    t.name,
                    t.dtype.name(),
                    t.shape.iter().product::<u64>(),
                    t.kind.name(),
                    t.bytes,
                    t.layout.name()
                );
            }
            plan::check_coverage(&ck, world).unwrap_or_else(|e| die(&e));
            println!("coverage: every tensor is whole on every rank or cut into disjoint, complete pieces");
        }
        "status" => status(&out, world),
        _ => usage(),
    }
}

fn prepare(ckpt: &Path, o: &Options, source: &str, repo: &str, rev: &str, endpoint: &str, replace: bool) {
    let (ck, reader, note): (Checkpoint, Box<dyn Reader>, String) = match source {
        "http" | "hybrid" => {
            let (ck, remote, names) = http_checkpoint(endpoint, repo, rev);
            let note = format!("{endpoint}/{repo}@{rev}");
            if source == "http" {
                (ck, Box::new(HttpReader::new(remote)), note)
            } else {
                let scratch = convert::dir_for(o, o.ranks[0]).join("scratch");
                println!(
                    "[dsw-prepare] hybrid: whole shards up to {:.1} GB, ranges above; scratch in {}",
                    HYBRID_MAX_CACHED as f64 / 1e9,
                    scratch.display()
                );
                (ck, Box::new(HybridReader::new(remote, names, scratch, HYBRID_MAX_CACHED)), note)
            }
        }
        _ => {
            let ck = ensure_checkpoint(ckpt, repo, rev);
            let paths: Vec<PathBuf> = ck.shards.iter().map(|s| s.path.clone().expect("a local checkpoint has paths")).collect();
            let r = LocalReader::open(&paths).unwrap_or_else(|e| die(&format!("opening the shards: {e}")));
            (ck, Box::new(r), ckpt.display().to_string())
        }
    };


    // What this run will write, per rank, from the plan itself.
    let mut need: u64 = 0;
    for &r in &o.ranks {
        let p = plan::make_plan(&ck, r, o.world).unwrap_or_else(|e| die(&e));
        let mut bytes = p.file_bytes;
        if o.engram {
            for t in dsw_prepare::engram::tables(&ck).unwrap_or_else(|e| die(&e)) {
                bytes += dsw_prepare::engram::shard_bytes(&t, r);
            }
        }
        println!(
            "rank {r}: {} tensors, {:.1} GB{}",
            p.tensors.len(),
            bytes as f64 / 1e9,
            o.dest.get(&r).map(|d| format!(" -> {}", d.display())).unwrap_or_default()
        );
        if o.dest.get(&r).is_none() {
            need += bytes;
        }
    }

    let existing = existing_outputs(o);
    if !existing.is_empty() {
        confirm_replace(&existing, replace);
        for p in &existing {
            if p.is_dir() {
                let _ = std::fs::remove_dir_all(p);
            } else {
                let _ = std::fs::remove_file(p);
            }
        }
    }

    // A hybrid run keeps one whole shard on disk while it converts it.
    if source == "hybrid" {
        need += ck.shards.iter().map(|s| s.size).filter(|&b| b <= HYBRID_MAX_CACHED).max().unwrap_or(0);
    }

    // Admission: the outputs must fit, plus the reserve. A local source is already
    // allocated, so it is not added again; --consume only frees it later.
    let free = space::free_bytes(&o.out).unwrap_or_else(|e| die(&format!("{}: {e}", o.out.display())));
    if need + RESERVE > free {
        die(&format!(
            "{} has {:.1} GB free; this run writes {:.1} GB and keeps {:.1} GB of headroom. \
             Use --dest for the other ranks, or --source http with --dest, or a bigger disk",
            o.out.display(),
            free as f64 / 1e9,
            need as f64 / 1e9,
            RESERVE as f64 / 1e9
        ));
    }

    convert::run(&ck, reader.as_ref(), o, &note).unwrap_or_else(|e| die(&e));
    if let Err(e) = &hashmap(&ck, o) {
        eprintln!("[dsw-prepare] WARN: the Engram hash map was not written: {e}");
    }
    // Serving needs a tokenizer, and the checkpoint may be deleted afterwards.
    let tok = ck.dir.join("tokenizer.json");
    if tok.is_file() {
        let to = convert::dir_for(o, o.ranks[0]).join("tokenizer.json");
        if let Err(e) = std::fs::copy(&tok, &to) {
            eprintln!("[dsw-prepare] WARN: {}: {e}", to.display());
        } else {
            println!("[dsw-prepare] tokenizer: {}", to.display());
        }
    }
    print_distribution(o);
}

/// The checkpoint, downloaded or repaired when it is not already here. `hf` is
/// tried first, then the older CLI; a `||` chain would mask "not found" as a
/// failed download.
fn ensure_checkpoint(ckpt: &Path, repo: &str, rev: &str) -> Checkpoint {
    let download = |dir: &Path| -> Result<(), String> {
        let try_hf = |bin: &str| -> bool {
            std::process::Command::new(bin)
                .args(["download", repo, "--revision", rev, "--local-dir"])
                .arg(dir)
                .status()
                .map(|s| s.success())
                .unwrap_or(false)
        };
        if try_hf("hf") || try_hf("huggingface-cli") {
            Ok(())
        } else {
            Err("the download failed (hf and huggingface-cli both failed)".into())
        }
    };
    match dsw_prepare::checkpoint::ensure(ckpt, &download) {
        Ok((ck, dsw_prepare::checkpoint::How::Local)) => {
            println!("[dsw-prepare] using the checkpoint at {}", ckpt.display());
            ck
        }
        Ok((ck, dsw_prepare::checkpoint::How::Downloaded)) => {
            println!("[dsw-prepare] downloaded {repo}@{rev} into {}", ckpt.display());
            ck
        }
        Err(e) => die(&e),
    }
}

/// The plan and the sources for a run that never stores the checkpoint.
fn http_checkpoint(endpoint: &str, repo: &str, rev: &str) -> (Checkpoint, Vec<HttpRange>, Vec<String>) {
    let token = std::env::var("HF_TOKEN").ok();
    let fetch = |url: &str, len: u64| -> Vec<u8> {
        let s = HttpRange::new(url.to_string(), len, token.clone(), 3);
        let mut b = vec![0u8; len as usize];
        s.read_at(0, &mut b).unwrap_or_else(|e| die(&format!("{url}: {e}")));
        b
    };
    let index_url = hub_url(endpoint, repo, rev, "model.safetensors.index.json");
    let index: serde_json::Value = {
        let out = std::process::Command::new("curl")
            .args(["-sSfL"])
            .arg(&index_url)
            .output()
            .unwrap_or_else(|e| die(&format!("curl: {e}")));
        if !out.status.success() {
            die(&format!("{}: {}", index_url, String::from_utf8_lossy(&out.stderr)));
        }
        serde_json::from_slice(&out.stdout).unwrap_or_else(|e| die(&format!("{}: {e}", index_url)))
    };
    let mut names: Vec<String> = index["weight_map"]
        .as_object()
        .unwrap_or_else(|| die("the index has no weight_map"))
        .values()
        .map(|v| v.as_str().unwrap_or_else(|| die("a shard name is not a string")).to_string())
        .collect();
    names.sort();
    names.dedup();
    let mut items = Vec::new();
    let mut sources = Vec::new();
    for name in &names {
        let url = hub_url(endpoint, repo, rev, name);
        let head = fetch(&url, 8);
        let header_len = u64::from_le_bytes(head[..8].try_into().unwrap());
        let mut raw = vec![0u8; 8 + header_len as usize];
        let s = HttpRange::new(url.clone(), u64::MAX, token.clone(), 3);
        s.read_at(0, &mut raw).unwrap_or_else(|e| die(&format!("{url}: {e}")));
        let (len, header) = dsw_prepare::safetensors::parse_header(&raw).unwrap_or_else(|e| die(&format!("{name}: {e}")));
        items.push((name.clone(), len, header));
        sources.push(s);
    }
    println!("[dsw-prepare] {} shards described from the Hub; nothing stored locally", items.len());
    let ck = Checkpoint::from_shard_headers(items).unwrap_or_else(|e| die(&e));
    (ck, sources, names)
}

fn existing_outputs(o: &Options) -> Vec<PathBuf> {
    let mut v = Vec::new();
    for &r in &o.ranks {
        let d = convert::dir_for(o, r);
        for p in [convert::dsw_path(o, r), convert::journal_path(o, r)] {
            if p.exists() {
                v.push(p.clone());
            }
            let mut part = p.into_os_string();
            part.push(".part");
            let part = PathBuf::from(part);
            if part.exists() {
                v.push(part);
            }
        }
        let e = d.join("engram");
        if e.is_dir() {
            v.push(e);
        }
    }
    v.sort();
    v.dedup();
    v
}

/// The interactive confirmation the prepare command always asks before it
/// deletes anything. A non-interactive run refuses instead of guessing.
fn confirm_replace(paths: &[PathBuf], replace: bool) {
    println!("[dsw-prepare] these outputs already exist:");
    for p in paths {
        println!("  {}", p.display());
    }
    if replace {
        println!("[dsw-prepare] --replace: deleting them");
        return;
    }
    let Ok(mut tty) = std::fs::OpenOptions::new().read(true).write(true).open("/dev/tty") else {
        die("they must be deleted to continue: re-run with --replace (no terminal to ask on)");
    };
    let _ = tty.write_all(b"delete them and continue? [y/N] ");
    let _ = tty.flush();
    let mut line = String::new();
    let mut buf = [0u8; 64];
    let n = tty.read(&mut buf).unwrap_or(0);
    line.push_str(&String::from_utf8_lossy(&buf[..n]));
    let yes = matches!(line.trim().to_ascii_lowercase().as_str(), "y" | "yes");
    if !yes {
        die("kept them; nothing was written");
    }
}

fn hashmap(ck: &Checkpoint, o: &Options) -> Result<(), String> {
    let Some(dir) = ck.dir.to_str() else { return Ok(()) };
    let out = convert::dir_for(o, o.ranks[0]).join("engram/engram_hashmap.bin");
    if let Some(p) = out.parent() {
        std::fs::create_dir_all(p).map_err(|e| e.to_string())?;
    }
    let script = Path::new("engine/engram/py/make_hashmap_stdlib.py");
    if !script.exists() {
        return Err("engine/engram/py/make_hashmap_stdlib.py not found (run from the repo)".into());
    }
    let st = std::process::Command::new("python3")
        .arg(script)
        .arg(dir)
        .arg(&out)
        .status()
        .map_err(|e| e.to_string())?;
    if !st.success() {
        return Err("make_hashmap_stdlib.py failed".into());
    }
    println!("[dsw-prepare] hash map: {}", out.display());
    Ok(())
}

fn print_distribution(o: &Options) {
    println!("[dsw-prepare] done. Install each rank's files where the fleet file says:");
    for &r in &o.ranks {
        let d = convert::dir_for(o, r);
        println!("  rank {r}: {} (rank{r}.dsw, engram/)", d.display());
        if o.dest.get(&r).is_none() && r != 0 {
            println!("    rsync -h --progress {} <RANK{r}_SSH>:<RANK{r}_WEIGHTS>", convert::dsw_path(o, r).display());
            println!(
                "    rsync -h --progress {} <RANK{r}_SSH>:<RANK{r}_ENGRAM>/",
                d.join("engram").display()
            );
        }
    }
}

fn status(out: &Path, world: u32) {
    let _ = world;
    let mut any = false;
    for r in 0..16u32 {
        let j = out.join(format!("rank{r}.dsw.journal"));
        if !j.exists() {
            continue;
        }
        any = true;
        match dsw_prepare::journal::Journal::load(&j) {
            Ok((_records, state)) => println!(
                "rank {r}: {} commits, {} released, {}",
                state.committed.len(),
                state.reclaimed.len(),
                if state.done { "complete" } else { "in progress" }
            ),
            Err(e) => println!("rank {r}: {}: {e}", j.display()),
        }
    }
    if !any {
        println!("no journals under {}", out.display());
    }
}
