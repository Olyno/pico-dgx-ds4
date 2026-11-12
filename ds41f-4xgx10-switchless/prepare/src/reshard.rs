//! Re-cutting a finished rank set for a different world size.
//!
//! A reshard derives the new files from the old ones: no checkpoint, no network,
//! no GPU. It reads each old rank's tiled bytes, un-tiles them, copies the
//! rectangles the new split needs into the new rank's natural buffer, tiles that
//! with the new geometry, writes it, reads it back and compares.
//!
//! Every rectangle comes from [`plan::make_plan`] — the same table the converter
//! uses, evaluated at the old world for the source side and at the new world for
//! the destination. The two sides therefore cannot disagree about which rows a
//! rank owns, and the old files are checked against their own plan first: a set
//! cut by a different version of the table is refused, not silently re-cut.
//!
//! The engine still reads four ways (`engine/common/tp.h`), so the output of a
//! reshard is only servable once that changes; the bytes themselves are the
//! checkpoint's own, whichever way they are cut.

use std::collections::BTreeMap;
use std::fs::File;
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::convert::{read_padded, write_padded};
use crate::dsw::{crc32c, Header, Layout, SliceKind, TocEntry, HEADER_BYTES, MAGIC, TOC_ENTRY_BYTES};
use crate::engram;
use crate::plan::{self, OutTensor, Plan};
use crate::safetensors::SrcTensor;
use crate::tile;

/// A tensor bigger than this, summed over the old ranks that hold it, is re-read
/// for each new rank instead of being held in memory.
const CACHE_BUDGET: u64 = 4 << 30;

#[derive(Debug, Clone)]
pub struct Options {
    /// The finished set to read: `rank0.dsw` .. `rank{world-1}.dsw`.
    pub from: PathBuf,
    /// The world the source files were cut for is read from their headers; this
    /// is the world to cut for.
    pub world: u32,
    pub out: PathBuf,
    pub ranks: Vec<u32>,
    pub verify: bool,
    pub replace: bool,
    pub engram: bool,
}

#[derive(Debug)]
pub struct Report {
    pub world: u32,
    pub files: Vec<PathBuf>,
    pub tensors: usize,
    pub bytes: u64,
    pub engram_files: usize,
}

struct Old {
    world: u32,
    model: String,
    /// Rank to its tensors by name.
    ranks: BTreeMap<u32, BTreeMap<String, TocEntry>>,
    files: BTreeMap<u32, File>,
}

fn read_set(dir: &Path) -> Result<Old, String> {
    let mut ranks = BTreeMap::new();
    let mut files = BTreeMap::new();
    let mut world: Option<u32> = None;
    let mut model = String::new();
    for r in 0u32.. {
        let path = dir.join(format!("rank{r}.dsw"));
        if !path.exists() {
            break;
        }
        let file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        let size = file.metadata().map_err(|e| format!("{}: {e}", path.display()))?.len();
        let mut hb = [0u8; HEADER_BYTES];
        file.read_exact_at(&mut hb, 0).map_err(|e| format!("{}: {e}", path.display()))?;
        let h = Header::from_bytes(&hb);
        if h.magic != MAGIC {
            return Err(format!("{}: not a .dsw file", path.display()));
        }
        if h.header_bytes != HEADER_BYTES as u32 || h.toc_entry_bytes != TOC_ENTRY_BYTES as u32 {
            return Err(format!("{}: bad header sizes", path.display()));
        }
        if h.file_bytes != size {
            return Err(format!("{}: the header says {} bytes, the file is {size}", path.display(), h.file_bytes));
        }
        if h.rank != r {
            return Err(format!("{}: it says it is rank {} of {}", path.display(), h.rank, h.world));
        }
        match world {
            None => {
                world = Some(h.world);
                model = h.model_name();
            }
            Some(w) if w != h.world => {
                return Err(format!("{}: cut for {} ranks, the others for {w}", path.display(), h.world));
            }
            _ => {}
        }
        let toc_len = (h.data_offset - h.toc_offset) as usize;
        let mut tb = vec![0u8; toc_len];
        file.read_exact_at(&mut tb, h.toc_offset).map_err(|e| format!("{}: {e}", path.display()))?;
        if crc32c(&tb[..h.n_tensors as usize * TOC_ENTRY_BYTES]) != h.toc_crc32c {
            return Err(format!("{}: TOC CRC mismatch", path.display()));
        }
        let mut entries = BTreeMap::new();
        for i in 0..h.n_tensors as usize {
            let e = TocEntry::from_bytes(&tb[i * TOC_ENTRY_BYTES..(i + 1) * TOC_ENTRY_BYTES]);
            let name = e.name();
            if entries.insert(name.clone(), e).is_some() {
                return Err(format!("{}: {name} is listed twice", path.display()));
            }
        }
        ranks.insert(r, entries);
        files.insert(r, file);
    }
    let world = world.ok_or_else(|| format!("{} holds no rank0.dsw", dir.display()))?;
    if ranks.len() as u32 != world {
        return Err(format!(
            "{} holds {} rank files, but a set cut for {world} needs all {world} of them",
            dir.display(),
            ranks.len()
        ));
    }
    Ok(Old { world, model, ranks, files })
}

/// The checkpoint's tensors, rebuilt from the old files' own table of contents:
/// the source shapes, dtypes and expert stacks are what the plan needs.
fn tensors_of(old: &Old) -> Result<Vec<SrcTensor>, String> {
    let first = old.ranks.values().next().ok_or("empty set")?;
    let mut out = Vec::new();
    for e in first.values() {
        let nstack = e.nstack.max(1);
        let per = e.bytes / nstack as u64;
        let src = e.source_name();
        for k in 0..nstack {
            let name = if nstack == 1 { src.clone() } else { src.replace("{e}", &k.to_string()) };
            out.push(SrcTensor {
                name,
                dtype: e.dtype(),
                shape: e.src_shape().to_vec(),
                bytes: per,
                shard: 0,
                offset: 0,
            });
        }
    }
    out.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(out)
}

/// Rebuilds every old rank's plan from the tensor list and checks the files
/// against it, so a stale or foreign set cannot be re-cut as if it were current.
fn check_old_plan(old: &Old, tensors: &[SrcTensor]) -> Result<Vec<Plan>, String> {
    let mut plans = Vec::new();
    for (&r, entries) in &old.ranks {
        let p = plan::make_plan(tensors, r, old.world)
            .map_err(|e| format!("the plan for rank {r} of {}: {e}", old.world))?;
        if p.tensors.len() != entries.len() {
            return Err(format!(
                "rank {r} holds {} tensors, this table plans {} — the files were cut by another version",
                entries.len(),
                p.tensors.len()
            ));
        }
        for o in &p.tensors {
            let e = entries.get(&o.name).ok_or_else(|| format!("rank {r} has no {}", o.name))?;
            let what = if e.bytes != o.bytes {
                Some("size")
            } else if e.shape() != o.shape.as_slice() {
                Some("shape")
            } else if e.kind() != o.kind {
                Some("slice kind")
            } else if e.layout() != o.layout {
                Some("layout")
            } else if e.offset != o.offset {
                Some("offset")
            } else {
                None
            };
            if let Some(what) = what {
                return Err(format!(
                    "rank {r}'s {} differs from this table in its {what} — the files were cut by another version",
                    o.name
                ));
            }
        }
        plans.push(p);
    }
    Ok(plans)
}

/// One old rank's tensor, tiled bytes in and natural bytes out, checksum first.
fn old_natural(old: &Old, rank: u32, e: &TocEntry) -> Result<Vec<u8>, String> {
    let file = &old.files[&rank];
    let mut buf = vec![0u8; e.bytes as usize];
    file.read_exact_at(&mut buf, e.offset).map_err(|err| format!("rank {rank}'s {}: {err}", e.name()))?;
    if crc32c(&buf) != e.crc32c {
        return Err(format!("rank {rank}'s {}: CRC mismatch, the source file is damaged", e.name()));
    }
    if e.layout() == Layout::Natural {
        return Ok(buf);
    }
    let g = tile::tile_geom(e.layout(), e.dtype(), e.shape()).map_err(|w| format!("{}: {w}", e.name()))?;
    let mut nat = vec![0u8; e.bytes as usize];
    tile::retile(e.layout(), e.dtype(), g, &buf, &mut nat, true);
    Ok(nat)
}

/// Copies every rectangle the new rank needs out of the old ranks' natural
/// bytes. `cache`, when given, holds each old rank's tensor already un-tiled.
fn assemble(
    old: &Old,
    plans: &[Plan],
    o: &OutTensor,
    cache: Option<&[Vec<u8>]>,
) -> Result<Vec<u8>, String> {
    let mut nat = vec![0u8; o.bytes as usize];
    let mut covered: u64 = 0;
    // A replicated tensor is whole on every old rank: the first one is the whole
    // source, and counting the others would make the coverage check lie.
    let replicated = o.kind == SliceKind::Replicated;
    // A stacked tensor's pieces all name the same source rectangle, so pieces
    // only pair up inside one expert. The expert is the piece's destination
    // block, and that block is sized by its own side: an expert is half of a
    // four-way tensor and a quarter of an eight-way one.
    let new_per = (o.bytes / o.nstack.max(1) as u64).max(1);
    for (i, p) in plans.iter().enumerate() {
        if replicated && i > 0 {
            break;
        }
        let rank = i as u32;
        let Some(op) = p.tensors.iter().find(|t| t.name == o.name) else { continue };
        let entry = &old.ranks[&rank][&o.name];
        let owned;
        let buf: &[u8] = match cache {
            Some(c) => &c[i],
            None => {
                owned = old_natural(old, rank, entry)?;
                &owned
            }
        };
        for np in &o.pieces {
            let old_per = (op.bytes / o.nstack.max(1) as u64).max(1);
            for mp in &op.pieces {
                if mp.dst / old_per != np.dst / new_per {
                    continue;
                }
                let r0 = np.row0.max(mp.row0);
                let r1 = (np.row0 + np.nrows).min(mp.row0 + mp.nrows);
                let c0 = np.col0.max(mp.col0);
                let c1 = (np.col0 + np.ncols).min(mp.col0 + mp.ncols);
                if r0 >= r1 || c0 >= c1 {
                    continue;
                }
                let n = (c1 - c0) as usize;
                for r in r0..r1 {
                    let s = (mp.dst + (r - mp.row0) * mp.ncols + (c0 - mp.col0)) as usize;
                    let d = (np.dst + (r - np.row0) * np.ncols + (c0 - np.col0)) as usize;
                    nat[d..d + n].copy_from_slice(&buf[s..s + n]);
                }
                covered += (r1 - r0) * (c1 - c0);
            }
        }
    }
    if covered != o.bytes {
        return Err(format!(
            "{}: the old ranks cover {covered} of {} bytes — the source set is incomplete",
            o.name, o.bytes
        ));
    }
    Ok(nat)
}

/// One new rank's file, open while its tensors land.
struct Out {
    rank: u32,
    plan: Plan,
    file: File,
    work: PathBuf,
    path: PathBuf,
    crcs: Vec<u32>,
}

fn write_tensor(file: &File, crcs: &mut [u32], idx: usize, o: &OutTensor, nat: &[u8], verify: bool) -> Result<(), String> {
    let mut til = nat.to_vec();
    if o.layout != Layout::Natural {
        let g = tile::tile_geom(o.layout, o.dtype, &o.shape)?;
        tile::retile(o.layout, o.dtype, g, nat, &mut til, false);
    }
    write_padded(file, o.offset, &til).map_err(|e| format!("{}: write: {e}", o.name))?;
    crcs[idx] = crc32c(&til);
    if !verify {
        return Ok(());
    }
    let back = read_padded(file, o.offset, o.bytes).map_err(|e| format!("{}: read back: {e}", o.name))?;
    if back[o.bytes as usize..].iter().any(|&b| b != 0) {
        return Err(format!("{}: the padding after the tensor is not zeros", o.name));
    }
    let got = if o.layout == Layout::Natural {
        back[..o.bytes as usize].to_vec()
    } else {
        let g = tile::tile_geom(o.layout, o.dtype, &o.shape)?;
        let mut nat_back = vec![0u8; o.bytes as usize];
        tile::retile(o.layout, o.dtype, g, &back[..o.bytes as usize], &mut nat_back, true);
        nat_back
    };
    if got != nat {
        return Err(format!("{}: what was written does not read back as the same bytes", o.name));
    }
    Ok(())
}

fn finalise(f: &Out, model: &str, note: &str) -> Result<u64, String> {
    let mut toc = vec![0u8; (f.plan.tensors.len() * TOC_ENTRY_BYTES) as usize];
    for (i, o) in f.plan.tensors.iter().enumerate() {
        let mut e = o.toc_entry();
        e.crc32c = f.crcs[i];
        toc[i * TOC_ENTRY_BYTES..(i + 1) * TOC_ENTRY_BYTES].copy_from_slice(&e.to_bytes());
    }
    let toc_padded = (f.plan.data_offset - f.plan.toc_offset) as usize;
    let mut tbuf = vec![0u8; toc_padded];
    tbuf[..toc.len()].copy_from_slice(&toc);
    let mut h = Header::new(f.rank, f.plan.world, model, note, &crate::now_iso8601());
    h.n_tensors = f.plan.tensors.len() as u32;
    h.toc_offset = f.plan.toc_offset;
    h.data_offset = f.plan.data_offset;
    h.data_bytes = f.plan.file_bytes - f.plan.data_offset;
    h.file_bytes = f.plan.file_bytes;
    h.toc_crc32c = crc32c(&toc);
    f.file.write_all_at(&tbuf, f.plan.toc_offset).map_err(|e| format!("TOC: {e}"))?;
    f.file.write_all_at(&h.to_bytes(), 0).map_err(|e| format!("header: {e}"))?;
    f.file.sync_all().map_err(|e| format!("fsync: {e}"))?;
    if f.work != f.path {
        std::fs::rename(&f.work, &f.path).map_err(|e| format!("rename {}: {e}", f.work.display()))?;
    }
    Ok(f.plan.file_bytes)
}

fn part(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".part");
    PathBuf::from(s)
}

/// `engram-l{layer}-r{rank}of{world}.bin`, or why it is not that.
fn parse_shard(name: &str) -> Option<(u32, u32, u32)> {
    let rest = name.strip_prefix("engram-l")?;
    let (layer, rest) = rest.split_once("-r")?;
    let (rank, rest) = rest.split_once("of")?;
    let world = rest.strip_suffix(".bin")?;
    Some((layer.parse().ok()?, rank.parse().ok()?, world.parse().ok()?))
}

/// The 4 KiB header of an Engram shard: layer, first row, one past the last.
fn shard_header(path: &Path) -> Result<(u32, u64, u64, u64), String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let mut h = [0u8; engram::HEADER_BYTES as usize];
    file.read_exact_at(&mut h, 0).map_err(|e| format!("{}: {e}", path.display()))?;
    let u = |i: usize| u64::from_le_bytes(h[i * 8..i * 8 + 8].try_into().unwrap());
    if u(0) != engram::MAGIC {
        return Err(format!("{}: not an Engram shard", path.display()));
    }
    if u(5) != engram::ROW_BYTES {
        return Err(format!("{}: {} byte records, not {}", path.display(), u(5), engram::ROW_BYTES));
    }
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    let want = engram::HEADER_BYTES + (u(3) - u(2)) * engram::ROW_BYTES;
    if size != want {
        return Err(format!("{}: {} bytes, its header describes {want}", path.display(), size));
    }
    Ok((u(1) as u32, u(2), u(3), u(4)))
}

/// Re-cuts every Engram shard by rows: the records are whole rows, so a new
/// rank's range is a run of records from whichever old shard held them.
fn reshard_engram(o: &Options, old: &Old, ranks: &[u32]) -> Result<usize, String> {
    let dir = o.from.join("engram");
    if !dir.is_dir() {
        println!("[reshard] no Engram shards under {}", dir.display());
        return Ok(0);
    }
    let mut layers: BTreeMap<u32, BTreeMap<u32, PathBuf>> = BTreeMap::new();
    for e in std::fs::read_dir(&dir).map_err(|e| format!("{}: {e}", dir.display()))? {
        let e = e.map_err(|e| e.to_string())?;
        let name = e.file_name().to_string_lossy().into_owned();
        let Some((layer, rank, world)) = parse_shard(&name) else { continue };
        if world != old.world {
            return Err(format!("{name}: cut for {world} ways, the rank files say {}", old.world));
        }
        layers.entry(layer).or_default().insert(rank, e.path());
    }
    let out_dir = o.out.join("engram");
    std::fs::create_dir_all(&out_dir).map_err(|e| format!("{}: {e}", out_dir.display()))?;
    let mut written = 0usize;
    for (layer, shards) in &layers {
        if shards.len() as u32 != old.world {
            return Err(format!("Engram layer {layer} has {} of {} shards", shards.len(), old.world));
        }
        let mut rows = 0u64;
        let mut bounds = Vec::new();
        for r in 0..old.world {
            let p = shards.get(&r).ok_or_else(|| format!("Engram layer {layer} has no shard {r}"))?;
            let (l, lo, hi, n) = shard_header(p)?;
            if l != *layer {
                return Err(format!("{}: it says it holds layer {l}", p.display()));
            }
            if rows != 0 && n != rows {
                return Err(format!("{}: {n} rows, the others have {rows}", p.display()));
            }
            rows = n;
            bounds.push((lo, hi));
        }
        for &r in ranks {
            let (nlo, nhi) = engram::rows_of_for(rows, r, o.world);
            let path = out_dir.join(engram::shard_name(*layer, r, o.world));
            if path.exists() && !o.replace {
                return Err(format!("{} exists (pass --replace)", path.display()));
            }
            let work = part(&path);
            let f = std::fs::OpenOptions::new()
                .read(true)
                .write(true)
                .create(true)
                .truncate(true)
                .open(&work)
                .map_err(|e| format!("{}: {e}", work.display()))?;
            f.set_len(engram::HEADER_BYTES + (nhi - nlo) * engram::ROW_BYTES)
                .map_err(|e| format!("{}: {e}", work.display()))?;
            f.write_all_at(&engram::header(*layer, nlo, nhi, rows), 0)
                .map_err(|e| format!("{}: {e}", work.display()))?;
            let mut filled = nlo;
            let chunk = ((64u64 << 20) / engram::ROW_BYTES).max(1);
            for (i, &(lo, hi)) in bounds.iter().enumerate() {
                let a = nlo.max(lo);
                let b = nhi.min(hi);
                if a >= b {
                    continue;
                }
                let from = &shards[&(i as u32)];
                let src = File::open(from).map_err(|e| format!("{}: {e}", from.display()))?;
                let mut row = a;
                while row < b {
                    let n = (b - row).min(chunk);
                    let mut buf = vec![0u8; (n * engram::ROW_BYTES) as usize];
                    src.read_exact_at(&mut buf, engram::HEADER_BYTES + (row - lo) * engram::ROW_BYTES)
                        .map_err(|e| format!("{}: {e}", from.display()))?;
                    f.write_all_at(&buf, engram::HEADER_BYTES + (row - nlo) * engram::ROW_BYTES)
                        .map_err(|e| format!("{}: {e}", work.display()))?;
                    if o.verify {
                        let mut back = vec![0u8; buf.len()];
                        f.read_exact_at(&mut back, engram::HEADER_BYTES + (row - nlo) * engram::ROW_BYTES)
                            .map_err(|e| format!("{}: {e}", work.display()))?;
                        if back != buf {
                            return Err(format!("{}: rows {row}..{} do not read back", path.display(), row + n));
                        }
                    }
                    row += n;
                }
                filled += b - a;
            }
            if filled != nhi {
                return Err(format!("{}: rows {nlo}..{nhi} are not all in the old shards", path.display()));
            }
            f.sync_all().map_err(|e| e.to_string())?;
            std::fs::rename(&work, &path).map_err(|e| format!("{}: {e}", path.display()))?;
            written += 1;
        }
        println!("[reshard] engram layer {layer}: {rows} rows, {} shard(s)", ranks.len());
    }
    Ok(written)
}

pub fn run(o: &Options) -> Result<Report, String> {
    if o.world < 1 {
        return Err("--world must be at least 1".into());
    }
    let old = read_set(&o.from)?;
    if o.world == old.world {
        println!("[reshard] world {} -> {}: a rebuild, the bytes must come out identical", old.world, o.world);
    } else {
        println!("[reshard] world {} -> {}", old.world, o.world);
    }
    let tensors = tensors_of(&old)?;
    let old_plans = check_old_plan(&old, &tensors)?;
    let ranks: Vec<u32> = if o.ranks.is_empty() { (0..o.world).collect() } else { o.ranks.clone() };
    for &r in &ranks {
        if r >= o.world {
            return Err(format!("rank {r} is outside world {}", o.world));
        }
    }
    let mut plans = Vec::new();
    for &r in &ranks {
        let p = plan::make_plan(&tensors, r, o.world).map_err(|e| format!("rank {r} of {}: {e}", o.world))?;
        if p.tensors.len() != old_plans[0].tensors.len() {
            return Err(format!("rank {r} of {} would hold {} tensors, the source {} — the table is not reversible", o.world, p.tensors.len(), old_plans[0].tensors.len()));
        }
        plans.push(p);
    }
    for &r in &ranks {
        let path = o.out.join(format!("rank{r}.dsw"));
        if path.exists() && !o.replace {
            return Err(format!("{} exists (pass --replace)", path.display()));
        }
    }
    std::fs::create_dir_all(&o.out).map_err(|e| format!("{}: {e}", o.out.display()))?;

    let names: Vec<String> = plans[0].tensors.iter().map(|t| t.name.clone()).collect();
    let mut outs = Vec::new();
    for (r, plan) in ranks.iter().copied().zip(plans) {
        let path = o.out.join(format!("rank{r}.dsw"));
        let work = part(&path);
        // Read as well as write: every tensor is read back and compared.
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(true)
            .truncate(true)
            .open(&work)
            .map_err(|e| format!("{}: {e}", work.display()))?;
        file.set_len(plan.file_bytes).map_err(|e| format!("{}: {e}", work.display()))?;
        outs.push(Out { rank: r, crcs: vec![0u32; plan.tensors.len()], plan, file, work, path });
    }

    let mut written_bytes = 0u64;
    for (i, name) in names.iter().enumerate() {
        let bytes = outs[0].plan.tensors[i].bytes;
        let cache = (bytes as u128 * old.world as u128) <= CACHE_BUDGET as u128;
        let mut bufs = Vec::new();
        if cache {
            // A replicated tensor needs one old rank; a split one needs them all.
            let n = if outs[0].plan.tensors[i].kind == SliceKind::Replicated { 1 } else { old.world };
            for &r in old.ranks.keys().take(n as usize) {
                let entry = &old.ranks[&r][name];
                bufs.push(old_natural(&old, r, entry)?);
            }
        }
        for k in 0..outs.len() {
            let f = &mut outs[k];
            let nat = assemble(&old, &old_plans, &f.plan.tensors[i], cache.then_some(&bufs[..]))?;
            write_tensor(&f.file, &mut f.crcs, i, &f.plan.tensors[i], &nat, o.verify)?;
        }
        written_bytes += bytes * outs.len() as u64;
        if i % 2000 == 0 {
            println!("[reshard] {}/{} tensors, {:.1} GB written", i, names.len(), written_bytes as f64 / 1e9);
        }
    }

    let note = format!("resharded from world {} ({})", old.world, o.from.display());
    let mut files = Vec::new();
    for f in &outs {
        let bytes = finalise(f, &old.model, &note)?;
        println!(
            "[reshard] rank {} of {}: {} ({} tensors, {:.1} GB)",
            f.rank,
            o.world,
            f.path.display(),
            f.plan.tensors.len(),
            bytes as f64 / 1e9
        );
        files.push(f.path.clone());
    }
    let engram_files = if o.engram { reshard_engram(o, &old, &ranks)? } else { 0 };
    Ok(Report { world: o.world, files, tensors: names.len(), bytes: written_bytes, engram_files })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shard_names_round_trip() {
        assert_eq!(parse_shard("engram-l14-r2of4.bin"), Some((14, 2, 4)));
        assert_eq!(parse_shard("engram-l1-r0of2.bin"), Some((1, 0, 2)));
        assert_eq!(parse_shard("engram-l14-r2of4.bin.part"), None);
        assert_eq!(parse_shard("rank0.dsw"), None);
    }
}
