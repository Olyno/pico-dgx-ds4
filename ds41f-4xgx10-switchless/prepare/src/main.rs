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

/// The device a path lives on; missing components are followed up to an ancestor,
/// as the space check does.
fn device_of(path: &Path) -> Option<u64> {
    use std::os::unix::fs::MetadataExt;
    let mut p = path;
    loop {
        if let Ok(m) = std::fs::metadata(p) {
            return Some(m.dev());
        }
        p = p.parent()?;
    }
}

/// Whether freeing the checkpoint's blocks would free space where the outputs go.
fn same_filesystem(a: &Path, b: &Path) -> bool {
    matches!((device_of(a), device_of(b)), (Some(x), Some(y)) if x == y)
}

/// Bytes a kept path already occupies: a file's allocated blocks, or every file
/// under a directory. A sparse `.part` counts only what it has written so far.
fn allocated(p: &Path) -> u64 {
    use std::os::unix::fs::MetadataExt;
    let Ok(m) = std::fs::metadata(p) else { return 0 };
    if !m.is_dir() {
        return m.blocks() * 512;
    }
    std::fs::read_dir(p).map(|e| e.flatten().map(|x| allocated(&x.path())).sum()).unwrap_or(0)
}

fn home() -> PathBuf {
    PathBuf::from(std::env::var("HOME").unwrap_or_else(|_| "/root".into()))
}

/// Whether the run may start, and why not when it may not. `peak` is what a
/// consuming run would need at its high-water mark; without it, or without
/// `--consume`, the whole output set has to fit.
fn admit(
    out: &Path,
    need: u64,
    free: u64,
    consume: bool,
    peak: Option<u64>,
    peak_why: Option<&str>,
) -> Result<(), String> {
    let fits = match (consume, peak) {
        (true, Some(p)) => p + RESERVE <= free,
        _ => need + RESERVE <= free,
    };
    if fits {
        return Ok(());
    }
    let mut msg = format!(
        "{} has {:.1} GB free and this run still has to write {:.1} GB, then keeps {:.1} GB of headroom",
        out.display(),
        free as f64 / 1e9,
        need as f64 / 1e9,
        RESERVE as f64 / 1e9
    );
    match peak {
        Some(p) if !consume && p + RESERVE <= free => msg.push_str(&format!(
            ". --consume fits: it converts one shard at a time and frees each once its outputs are verified, so the peak is {:.1} GB — the checkpoint cannot be kept afterwards",
            p as f64 / 1e9
        )),
        Some(p) => msg.push_str(&format!(
            ". --consume would peak at {:.1} GB, which does not fit either",
            p as f64 / 1e9
        )),
        None => {
            // Without a peak, say why: "run with --consume" would be advice that
            // cannot help, and the reason is the part the reader cannot guess.
            if let Some(why) = peak_why {
                msg.push_str(&format!("; {why}, so --consume would free nothing here"));
            }
        }
    }
    msg.push_str("; --dest writes a rank to another disk, and --source hybrid keeps only one shard on this one");
    Err(msg)
}

fn usage() -> ! {
    eprintln!(
        "usage: dsw-prepare <prepare|verify|plan|clear|reshard|status> [options]\n\
         \n\
         prepare  --ckpt DIR  --out DIR  [--world N] [--ranks LIST] [--source local|http]\n\
         \x20        [--consume] [--clear] [--no-verify] [--no-resume] [--no-engram]\n\
         \x20        [--dest R=DIR]... [--repo OWNER/NAME] [--rev SHA] [--endpoint URL]\n\
         verify   --ckpt DIR --file FILE\n\
         plan     [--ckpt DIR] [--world N] [--rank R]\n\
         clear    [--out DIR]   delete the rank files, journals, Engram shards and\n\
         \x20                     scratch of earlier runs (never the checkpoint)\n\
         reshard  [--from DIR] [--to DIR] --world N [--ranks LIST] [--replace]\n\
         \x20        [--no-verify] [--no-engram]\n\
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
    // Not a single fixed name: the checkpoint may already be here under another
    // one (/data, the Hub cache), and downloading 507 GB again is not a default.
    let mut ckpt = dsw_prepare::checkpoint::discover(&home())
        .unwrap_or_else(|| home().join("DeepSeek-V4.1-Flash-1017"));
    let mut out = home().join("dsgb10/convert");
    let mut world = 4u32;
    let mut ranks: Option<Vec<u32>> = None;
    // auto: the local checkpoint when it is already complete, otherwise hybrid
    // (whole shards downloaded one at a time, the oversized ones by range).
    let mut source = "auto".to_string();
    let mut consume = false;
    let mut replace = false;
    let mut repair = false;
    let mut clear = false;
    let mut verify = true;
    let mut resume = true;
    let mut engram = true;
    let mut dest: HashMap<u32, PathBuf> = HashMap::new();
    let mut repo = REPO.to_string();
    let mut rev = REV.to_string();
    let mut endpoint = std::env::var("HF_ENDPOINT").unwrap_or_else(|_| "https://huggingface.co".into());
    let mut file: Option<PathBuf> = None;
    let mut rank: Option<u32> = None;
    // reshard: the finished set to read and where the re-cut one goes.
    let mut from: Option<PathBuf> = None;
    let mut to: Option<PathBuf> = None;
    let mut dump = false;
    // VISION=1 keeps the encoder's slice of the checkpoint; the flags override it.
    let mut vision = std::env::var("VISION").map(|v| v == "1").unwrap_or(false);
    let mut vision_dir = home().join("dsgb10/vision");

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
                if !matches!(v, "auto" | "local" | "http" | "hybrid") {
                    die("--source is auto, local, http or hybrid");
                }
                source = v.into();
            }
            "--consume" => consume = true,
            "--replace" => replace = true,
            "--repair" => repair = true,
            "--clear" | "--reset" => clear = true,
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
            "--from" => from = Some(PathBuf::from(arg(&args, &mut i, a))),
            "--to" => to = Some(PathBuf::from(arg(&args, &mut i, a))),
            // One line per tensor, the format `plan_dump` prints: the two plans are
            // compared with `diff` on the same checkpoint.
            "--dump" => dump = true,
            "--vision" => vision = true,
            "--no-vision" => vision = false,
            "--vision-dir" => vision_dir = PathBuf::from(arg(&args, &mut i, a)),
            "-h" | "--help" => usage(),
            other => die(&format!("unknown argument {other}")),
        }
        i += 1;
    }

    match cmd.as_str() {
        "prepare" => {
            let ranks = ranks.unwrap_or_else(|| (0..world).collect());
            let o = Options {
                world,
                ranks,
                out: out.clone(),
                dest,
                consume,
                verify,
                resume,
                engram,
                vision: vision.then(|| vision_dir.clone()),
                model: "deepseek-v4.1-flash".into(),
            };
            prepare(&ckpt, &o, &source, &repo, &rev, &endpoint, replace, clear, repair);
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
            let p = plan::make_plan(&ck.tensors, r, world).unwrap_or_else(|e| die(&e));
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
        "clear" => {
            let (items, freed) = clear_outputs(&out).unwrap_or_else(|e| die(&e));
            if items == 0 {
                println!("[dsw-prepare] nothing to clear in {}", out.display());
            } else {
                println!(
                    "[dsw-prepare] {items} item(s) of earlier runs deleted, {:.1} GB freed (the checkpoint is untouched)",
                    freed as f64 / 1e9
                );
            }
        }
        "reshard" => {
            let ro = dsw_prepare::reshard::Options {
                from: from.unwrap_or_else(|| home().join("dsgb10/convert")),
                world,
                out: to.unwrap_or_else(|| home().join(format!("dsgb10/convert-tp{world}"))),
                ranks: ranks.unwrap_or_default(),
                verify,
                replace,
                engram,
            };
            let r = dsw_prepare::reshard::run(&ro).unwrap_or_else(|e| die(&e));
            println!(
                "[dsw-prepare] world {}: {} rank file(s), {} tensors, {:.1} GB written, {} Engram shard(s)",
                r.world,
                r.files.len(),
                r.tensors,
                r.bytes as f64 / 1e9,
                r.engram_files
            );
            if r.world != 4 {
                println!(
                    "[dsw-prepare] note: the engine serves TP=4 only (common/tp.h), so these files need a build that takes {}",
                    r.world
                );
            }
        }
        "status" => status(&out, world),
        _ => usage(),
    }
}

fn prepare(
    ckpt: &Path,
    o: &Options,
    source: &str,
    repo: &str,
    rev: &str,
    endpoint: &str,
    replace: bool,
    clear: bool,
    repair: bool,
) {
    // The local checkpoint wins whenever it is already complete; a download goes
    // through the hybrid source, which keeps one shard on disk at a time.
    let source: &str = if source == "auto" {
        match Checkpoint::open(ckpt) {
            Ok(_) => "local",
            // --repair: re-download into the same directory. `hf download` fetches
            // the files that are short or wrong and skips the rest, so this is a
            // repair, not 507 GB.
            Err(_) if repair => "local",
            // A directory that holds shards is not "nothing to lose": quietly
            // downloading 507 GB over a copy that will not open is how a run
            // that should have failed in a second hangs for hours.
            Err(why) if dsw_prepare::checkpoint::has_shards(ckpt) => die(&format!(
                "{why}\n  the checkpoint at {} is there but unusable: pass --repair to let the \
                 download fetch what is wrong, delete it to start over, point --ckpt at an intact \
                 one, or pass --source hybrid to ignore it",
                ckpt.display()
            )),
            Err(_) => "hybrid",
        }
    } else {
        source
    };
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
            // "local" means the weights are here: a missing checkpoint is an
            // error, not a 507 GB download. (Auto only resolves to local when it
            // opens, so this can only be the explicit flag.)
            if source == "local" && !dsw_prepare::checkpoint::has_shards(ckpt) {
                die(&format!(
                    "--source local: no checkpoint at {} (drop --source local to download it, or pass --ckpt DIR)",
                    ckpt.display()
                ));
            }
            let ck = ensure_checkpoint(ckpt, repo, rev);
            let paths: Vec<PathBuf> = ck.shards.iter().map(|s| s.path.clone().expect("a local checkpoint has paths")).collect();
            let r = LocalReader::open(&paths).unwrap_or_else(|e| die(&format!("opening the shards: {e}")));
            (ck, Box::new(r), ckpt.display().to_string())
        }
    };


    // --consume punches holes in the checkpoint as it goes, so a checkpoint that
    // an earlier run released has zeroes where the tensors were: converting from
    // it again would write those zeroes into the rank files without a word.
    // Only a run that has something to cut reads it, though: the usual re-run on
    // finished outputs walks in, says there is nothing to do and walks out.
    let needs_work = clear
        || !o.resume
        || o.ranks.iter().any(|&r| !convert::rank_done(&ck, o, r).unwrap_or(true));
    let released = if needs_work { ck.released() } else { Vec::new() };
    if !released.is_empty() {
        die(&format!(
            "{} of {} shard(s) in {} were released by an earlier --consume and now read as zeros; \
             converting from them would write zeros into the rank files. Download the checkpoint \
             again (remove the directory first) or point --ckpt at an intact copy",
            released.len(),
            ck.shards.len(),
            ckpt.display()
        ));
    }

    // What this run will write, per rank, from the plan itself.
    let mut need: u64 = 0;
    for &r in &o.ranks {
        let p = plan::make_plan(&ck.tensors, r, o.world).unwrap_or_else(|e| die(&e));
        let mut bytes = p.file_bytes;
        if o.engram {
            for t in dsw_prepare::engram::tables(&ck).unwrap_or_else(|e| die(&e)) {
                bytes += dsw_prepare::engram::shard_bytes(&t, r, o.world);
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

    // An interrupted run leaves its journal and its `.part` behind. Those are the
    // resume state, not stale output, so they are kept and the run continues where
    // it stopped; only what cannot be resumed is offered for deletion.
    let existing = if clear {
        let (items, freed) = clear_outputs(&o.out).unwrap_or_else(|e| die(&e));
        println!(
            "[dsw-prepare] --clear: {items} item(s) of earlier runs deleted, {:.1} GB freed (the checkpoint is untouched)",
            freed as f64 / 1e9
        );
        Vec::new()
    } else {
        existing_outputs(o)
    };
    let mut kept: Vec<PathBuf> = Vec::new();
    if !existing.is_empty() {
        let (keep, stale): (Vec<PathBuf>, Vec<PathBuf>) = if o.resume {
            existing.into_iter().partition(|p| keep_existing(&ck, o, p))
        } else {
            (Vec::new(), existing)
        };
        if !keep.is_empty() {
            println!(
                "[dsw-prepare] keeping {} file(s): finished ranks are left as they are, unfinished ones resume (--no-resume starts over)",
                keep.len()
            );
        }
        kept = keep;
        if !stale.is_empty() {
            confirm_replace(&stale, replace);
            for p in &stale {
                if p.is_dir() {
                    let _ = std::fs::remove_dir_all(p);
                } else {
                    let _ = std::fs::remove_file(p);
                }
            }
        }
    }

    // Bytes an earlier run already wrote are not written again: the space check
    // must ask for the remainder, not for the whole set. Without this a resumed
    // run was refused for a total that was already half on the disk.
    let already: u64 = kept.iter().map(|p| allocated(p)).sum();
    if already > 0 {
        need = need.saturating_sub(already);
        println!(
            "[dsw-prepare] {:.1} GB of the output is already on disk, so {:.1} GB still has to be written",
            already as f64 / 1e9,
            need as f64 / 1e9
        );
    }
    // A hybrid run keeps one whole shard on disk while it converts it.
    if source == "hybrid" {
        need += ck.shards.iter().map(|s| s.size).filter(|&b| b <= HYBRID_MAX_CACHED).max().unwrap_or(0);
    }
    // The vision bundle is a second copy of the shards the encoder reads, written
    // before anything is released, so it counts against both budgets.
    let mut always = 0u64;
    if o.vision.is_some() {
        always = dsw_prepare::vision::shards_needed(&ck).iter().map(|&i| ck.shards[i].size).sum();
        need += always;
    }

    // Admission. Without reclamation the whole output set must fit; a local source
    // is already allocated, so it is not added again. With --consume the checkpoint
    // is released as each shard's outputs are verified, so what has to fit is the
    // peak of (written - released) instead — computed whenever that could help,
    // which needs the outputs on the checkpoint's own filesystem.
    let free = space::free_bytes(&o.out).unwrap_or_else(|e| die(&format!("{}: {e}", o.out.display())));
    let (peak, peak_why) = if o.dest.is_empty() && source == "local" && same_filesystem(ckpt, &o.out) {
        match space::consume_peak(&ck, o.world, &o.ranks, o.engram).ok().map(|p| p + always) {
            Some(p) => (Some(p), None),
            None => (None, Some("the peak of this plan could not be computed")),
        }
    } else if !o.dest.is_empty() {
        (None, Some("the outputs are written to other disks (--dest)"))
    } else if source != "local" {
        (None, Some("this run streams the checkpoint, so there is nothing on this disk to release"))
    } else {
        (None, Some("the checkpoint and the outputs are on different filesystems"))
    };
    if o.consume {
        if let Some(p) = peak {
            println!(
                "[dsw-prepare] --consume: {:.1} GB free, peak {:.1} GB (one shard in flight), {:.1} GB of headroom",
                free as f64 / 1e9,
                p as f64 / 1e9,
                RESERVE as f64 / 1e9
            );
        }
    }
    if let Err(e) = admit(&o.out, need, free, o.consume, peak, peak_why) {
        die(&e);
    }

    // Before the conversion, so releasing the checkpoint cannot cost the encoder
    // its weights.
    if let Some(dir) = &o.vision {
        write_vision(&ck, dir, reader.as_ref());
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

/// Saves the vision encoder's slice of the checkpoint: its scripts, the index and
/// the shards holding `vision.`/`aligner.`/`image_` tensors. A warning, not a
/// failure, when the model has no vision at all.
fn write_vision(ck: &Checkpoint, dest: &Path, reader: &dyn Reader) {
    let materialize = |i: usize, to: &Path| -> Result<(), String> {
        if let Some(p) = reader.local_path(i) {
            return std::fs::copy(&p, to).map(|_| ()).map_err(|e| format!("{}: {e}", p.display()));
        }
        // A pure-range reader has no copy to give; a hybrid one fetches it.
        reader.prepare(i).map_err(|e| e.to_string())?;
        let out = match reader.local_path(i) {
            Some(p) => std::fs::copy(&p, to).map(|_| ()).map_err(|e| format!("{}: {e}", p.display())),
            None => Err("a vision bundle needs a local or hybrid source".into()),
        };
        reader.finish(i);
        out
    };
    match dsw_prepare::vision::write(ck, dest, &materialize) {
        Ok(b) => println!(
            "[dsw-prepare] vision bundle: {} ({} shards, {:.1} GB); point vision_encoder.py --model at it",
            b.dest.display(),
            b.shards.len(),
            b.bytes as f64 / 1e9
        ),
        Err(e) => eprintln!("[dsw-prepare] WARN: no vision bundle: {e}"),
    }
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

/// Every name this tool writes into an output directory: the rank files, their
/// journals and partial writes, the Engram shards and the hybrid scratch. The
/// checkpoint lives elsewhere and is never one of them.
fn owns(name: &str) -> bool {
    if name == "engram" || name == "scratch" {
        return true;
    }
    let Some(rest) = name.strip_prefix("rank") else { return false };
    let Some((rank, tail)) = rest.split_once('.') else { return false };
    rank.parse::<u32>().is_ok() && matches!(tail, "dsw" | "dsw.part" | "dsw.journal" | "dsw.journal.part")
}

/// Deletes them, and nothing else in the directory.
fn clear_outputs(out: &Path) -> Result<(usize, u64), String> {
    let Ok(dir) = std::fs::read_dir(out) else { return Ok((0, 0)) };
    let (mut items, mut freed) = (0usize, 0u64);
    for e in dir {
        let e = e.map_err(|err| format!("{}: {err}", out.display()))?;
        let name = e.file_name().to_string_lossy().into_owned();
        if !owns(&name) {
            continue;
        }
        let p = e.path();
        freed += bytes_of(&p);
        if p.is_dir() {
            std::fs::remove_dir_all(&p).map_err(|err| format!("{}: {err}", p.display()))?;
        } else {
            std::fs::remove_file(&p).map_err(|err| format!("{}: {err}", p.display()))?;
        }
        println!("  removed {}", p.display());
        items += 1;
    }
    Ok((items, freed))
}

fn bytes_of(p: &Path) -> u64 {
    if !p.is_dir() {
        return std::fs::metadata(p).map(|m| m.len()).unwrap_or(0);
    }
    std::fs::read_dir(p).into_iter().flatten().flatten().map(|e| bytes_of(&e.path())).sum()
}

/// Whether an existing file belongs to a run this tool keeps: `rank<r>.*` whose
/// journal parses and was written for this checkpoint, world and model, or the
/// Engram directory beside such a journal. A finished rank is kept as it is (the
/// conversion skips it), an unfinished one resumes; only what matches neither is
/// offered for deletion.
fn keep_existing(ck: &Checkpoint, o: &Options, p: &Path) -> bool {
    if p.is_dir() {
        return o.ranks.iter().any(|&r| rank_kept(ck, o, r));
    }
    let name = p.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default();
    let Some(rest) = name.strip_prefix("rank") else { return false };
    match rest.split('.').next().and_then(|s| s.parse::<u32>().ok()) {
        Some(r) => rank_kept(ck, o, r),
        None => false,
    }
}

fn rank_kept(ck: &Checkpoint, o: &Options, rank: u32) -> bool {
    let j = convert::journal_path(o, rank);
    if !j.exists() {
        return false;
    }
    // Unreadable is not kept: the next run could not continue it either.
    let Ok((records, _state)) = dsw_prepare::journal::Journal::load(&j) else { return false };
    match plan::make_plan(&ck.tensors, rank, o.world) {
        Ok(p) => dsw_prepare::journal::Journal::begin_matches(
            &records,
            &convert::fingerprint(&p, &o.model),
            rank,
            o.world,
        )
        .is_ok(),
        Err(_) => false,
    }
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
    // The map is a function of the checkpoint alone, so a rerun has nothing to
    // add to one that is already there.
    if out.metadata().map(|m| m.len() > 0).unwrap_or(false) {
        println!("[dsw-prepare] hash map: {} (kept)", out.display());
        return Ok(());
    }
    let script = Path::new("engine/engram/py/make_hashmap_stdlib.py");
    if !script.exists() {
        return Err("engine/engram/py/make_hashmap_stdlib.py not found (run from the repo)".into());
    }
    // Beside the target, then renamed: an interrupted script leaves no file the
    // engine would read as a complete table.
    let tmp = out.with_extension("bin.part");
    let st = std::process::Command::new("python3")
        .arg(script)
        .arg(dir)
        .arg(&tmp)
        .status()
        .map_err(|e| e.to_string())?;
    if !st.success() {
        let _ = std::fs::remove_file(&tmp);
        return Err("make_hashmap_stdlib.py failed".into());
    }
    std::fs::rename(&tmp, &out).map_err(|e| format!("{}: {e}", out.display()))?;
    println!("[dsw-prepare] hash map: {}", out.display());
    Ok(())
}

fn print_distribution(o: &Options) {
    println!("[dsw-prepare] done. Rank 0 reads its files where they were written; the others need three each:");
    for &r in &o.ranks {
        let d = convert::dir_for(o, r);
        println!("  rank {r}: {}", d.display());
        if r != 0 || o.dest.contains_key(&r) {
            println!("    rank{r}.dsw                     -> RANK{r}_WEIGHTS");
            println!("    engram/engram_hashmap.bin       -> RANK{r}_HASHMAP");
            println!("    engram/engram-l*-r{r}of{}.bin   -> RANK{r}_ENGRAM/", o.world);
        }
    }
    println!("[dsw-prepare] ./start.sh ships them and serves; flags: --consume --source --clear --vision");
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

#[cfg(test)]
mod tests {
    use super::*;

    const GB: u64 = 1_000_000_000;

    #[test]
    fn clear_takes_only_what_the_tool_wrote() {
        let dir = std::env::temp_dir().join(format!("dsw-clear-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("engram")).unwrap();
        std::fs::create_dir_all(dir.join("scratch")).unwrap();
        for (rel, bytes) in [
            ("rank0.dsw", 100),
            ("rank0.dsw.part", 50),
            ("rank0.dsw.journal", 20),
            ("rank3.dsw", 10),
            ("engram/engram-l1-r0of4.bin", 7),
            ("scratch/model-00001.safetensors", 3),
            ("notes.txt", 5),
            ("model.safetensors", 9),
        ] {
            std::fs::write(dir.join(rel), vec![0u8; bytes]).unwrap();
        }
        let (items, freed) = clear_outputs(&dir).unwrap();
        assert_eq!(items, 6, "rank0 x3, rank3, engram/, scratch/");
        assert_eq!(freed, 100 + 50 + 20 + 10 + 7 + 3);
        for gone in ["rank0.dsw", "rank0.dsw.part", "rank0.dsw.journal", "rank3.dsw", "engram", "scratch"] {
            assert!(!dir.join(gone).exists(), "{gone} should be gone");
        }
        for kept in ["notes.txt", "model.safetensors"] {
            assert!(dir.join(kept).exists(), "{kept} is not ours and must stay");
        }
        assert_eq!(clear_outputs(&dir).unwrap(), (0, 0), "a second clear finds nothing");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_rank_is_kept_while_it_resumes_and_once_it_is_done() {
        use dsw_prepare::journal::{Journal, Record, VERSION};
        let dir = std::env::temp_dir().join(format!("dsw-keep-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        let ckdir = dir.join("ck");
        std::fs::create_dir_all(&ckdir).unwrap();
        // One shard, one tensor: an 8-byte length, the header JSON, the data.
        let header = br#"{"embed.weight":{"dtype":"BF16","shape":[8,4],"data_offsets":[0,64]}}"#;
        let mut shard = (header.len() as u64).to_le_bytes().to_vec();
        shard.extend_from_slice(header);
        shard.extend_from_slice(&[0u8; 64]);
        std::fs::write(ckdir.join("model-00001-of-00001.safetensors"), shard).unwrap();
        let ck = Checkpoint::open(&ckdir).unwrap();
        let o = Options { ranks: vec![0], out: dir.join("out"), ..Options::default() };
        std::fs::create_dir_all(&o.out).unwrap();
        let j = convert::journal_path(&o, 0);
        let dsw = convert::dsw_path(&o, 0);

        // Nothing was written yet: nothing to keep.
        assert!(!keep_existing(&ck, &o, &j));
        assert!(!keep_existing(&ck, &o, &dsw));

        let plan = plan::make_plan(&ck.tensors, 0, o.world).unwrap();
        let fp = convert::fingerprint(&plan, &o.model);
        let begin = |fingerprint: &str| Record::Begin {
            version: VERSION,
            fingerprint: fingerprint.to_string(),
            rank: 0,
            world: o.world,
            out: dsw.display().to_string(),
        };
        // A journal written for another checkpoint cannot be continued, so it is
        // not kept either.
        let mut jr = Journal::create(&j).unwrap();
        jr.append(&begin("some other plan")).unwrap();
        assert!(!keep_existing(&ck, &o, &j), "a foreign journal is stale, not kept");
        std::fs::remove_file(&j).unwrap();

        let mut jr = Journal::create(&j).unwrap();
        jr.append(&begin(&fp)).unwrap();
        // Begun: the journal, the partial file and the Engram directory resume.
        assert!(keep_existing(&ck, &o, &j));
        let mut part = dsw.clone().into_os_string();
        part.push(".part");
        assert!(keep_existing(&ck, &o, Path::new(&part)));
        std::fs::create_dir_all(o.out.join("engram")).unwrap();
        assert!(keep_existing(&ck, &o, &o.out.join("engram")));

        // Finished: kept as it is (the conversion skips the rank), never a prompt.
        jr.append(&Record::Done { file: dsw.display().to_string(), bytes: 1, tensors: 1 }).unwrap();
        assert!(keep_existing(&ck, &o, &j));
        assert!(keep_existing(&ck, &o, &dsw));
        assert!(keep_existing(&ck, &o, &o.out.join("engram")));
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn kept_files_count_what_they_already_hold() {
        let dir = std::env::temp_dir().join(format!("dsw-alloc-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(dir.join("engram")).unwrap();
        std::fs::write(dir.join("rank0.dsw.part"), vec![2u8; 3 << 20]).unwrap();
        std::fs::write(dir.join("engram/shard.dsweng"), vec![1u8; 1 << 20]).unwrap();
        // A partial file is credited for what it holds, so the resume is not
        // asked to fit the whole rank again.
        assert!(allocated(&dir.join("rank0.dsw.part")) >= 3 << 20);
        // A kept Engram directory credits every shard inside it.
        assert!(allocated(&dir.join("engram")) >= 1 << 20);
        assert_eq!(allocated(&dir.join("missing")), 0);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn the_whole_set_must_fit_without_consumption() {
        let out = Path::new("/data/out");
        assert!(admit(out, 400 * GB, 500 * GB, false, None, None).is_ok());
        let e = admit(
            out,
            404 * GB,
            100 * GB,
            false,
            None,
            Some("the checkpoint and the outputs are on different filesystems"),
        )
        .unwrap_err();
        assert!(e.contains("still has to write 404.0 GB"), "{e}");
        // Without a peak the message names the reason instead of offering
        // --consume as a fix that could not help.
        assert!(e.contains("on different filesystems"), "{e}");
        assert!(e.contains("--consume would free nothing here"), "{e}");
        assert!(!e.contains("--consume fits"), "{e}");
    }

    #[test]
    fn consumption_is_judged_by_its_peak_not_the_total() {
        let out = Path::new("/data/out");
        // The report from a two-rank run: 409 GB of files, 95 GB free, and a
        // consuming peak of one Engram shard's output.
        let e = admit(out, 409 * GB, 95 * GB, false, Some(19 * GB), None).unwrap_err();
        assert!(e.contains("--consume fits"), "{e}");
        assert!(e.contains("peak is 19.0 GB"), "{e}");
        assert!(admit(out, 409 * GB, 95 * GB, true, Some(19 * GB), None).is_ok());
        // A peak that does not fit either is called out as such.
        let e = admit(out, 409 * GB, 95 * GB, true, Some(300 * GB), None).unwrap_err();
        assert!(e.contains("does not fit either"), "{e}");
        assert!(e.contains("--dest"), "{e}");
    }
}
