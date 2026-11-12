//! The conversion itself: read a source tensor once, write every rank's slice of
//! it, prove the bytes, commit, and only then release the source.
//!
//! The order per batch is fixed, and every step matters:
//!
//! 1. read the source ranges the batch needs;
//! 2. assemble each consumer's natural bytes, tile them, write at the tensor's
//!    final offset;
//! 3. read the bytes back, un-tile them with the inverse mapping, compare with the
//!    source-derived bytes, and check the padding is zeros;
//! 4. `fsync` the output, then append and `fsync` the journal record;
//! 5. only now may the source blocks be released.
//!
//! A crash before 4 replays the batch; a crash after 4 replays only the release.

use std::collections::{HashMap, HashSet};
use std::fs::{File, OpenOptions};
use std::os::unix::fs::FileExt;
use std::path::{Path, PathBuf};

use crate::dsw::{align_up, crc32c, Header, Layout, SliceKind, TocEntry, ALIGN, HEADER_BYTES, TOC_ENTRY_BYTES};
use crate::engram::{self, Table};
use crate::journal::{Journal, OutputCommit, Record};
use crate::plan::{self, OutTensor, Plan};
use crate::safetensors::{Checkpoint, SrcTensor};
use crate::source::Reader;
use crate::tile;

#[derive(Debug, Clone)]
pub struct Options {
    pub world: u32,
    pub ranks: Vec<u32>,
    /// Where the artifacts go when a rank has no destination of its own.
    pub out: PathBuf,
    /// Per-rank destinations (a mounted worker path, for instance).
    pub dest: HashMap<u32, PathBuf>,
    /// Release a source range once every rank's output is committed.
    pub consume: bool,
    /// Read every written tensor back and compare it with the checkpoint.
    pub verify: bool,
    /// Keep what an earlier run committed instead of starting over.
    pub resume: bool,
    /// Also write the Engram shards.
    pub engram: bool,
    /// Where to save the vision encoder's slice of the checkpoint, when it is
    /// wanted. Written before the conversion, so releasing the checkpoint cannot
    /// cost it; `run` itself does not read this.
    pub vision: Option<PathBuf>,
    pub model: String,
}

impl Default for Options {
    fn default() -> Self {
        Options {
            world: 4,
            ranks: vec![0],
            out: PathBuf::from("."),
            dest: HashMap::new(),
            consume: false,
            verify: true,
            resume: true,
            engram: true,
            vision: None,
            model: "deepseek-v4.1-flash".into(),
        }
    }
}

pub fn dir_for(o: &Options, rank: u32) -> PathBuf {
    o.dest.get(&rank).cloned().unwrap_or_else(|| o.out.clone())
}

pub fn dsw_path(o: &Options, rank: u32) -> PathBuf {
    dir_for(o, rank).join(format!("rank{rank}.dsw"))
}

pub fn journal_path(o: &Options, rank: u32) -> PathBuf {
    dir_for(o, rank).join(format!("rank{rank}.dsw.journal"))
}

pub fn engram_path(o: &Options, rank: u32, layer: u32) -> PathBuf {
    dir_for(o, rank).join("engram").join(engram::shard_name(layer, rank, o.world))
}

fn part(path: &Path) -> PathBuf {
    let mut s = path.as_os_str().to_os_string();
    s.push(".part");
    PathBuf::from(s)
}

/// Identifies the plan a journal belongs to: any change to the checkpoint, the
/// split, or the rank invalidates it.
pub fn fingerprint(plan: &Plan, model: &str) -> String {
    let mut s = format!("v1|{model}|{}|{}|{}|{}|", plan.rank, plan.world, plan.file_bytes, plan.tensors.len());
    for t in &plan.tensors {
        s.push_str(&format!(
            "{}|{}|{}|{}|{}|{:?}|{};",
            t.name,
            t.source,
            t.offset,
            t.bytes,
            t.dtype.name(),
            t.shape,
            t.layout.name()
        ));
    }
    format!("{:08x}", crc32c(s.as_bytes()))
}

struct RankState {
    rank: u32,
    plan: Plan,
    file: File,
    /// The file being written: `rank{r}.dsw.part` until the TOC and header land,
    /// or the final file when a resumed run reopened one.
    work: PathBuf,
    journal: Journal,
    committed: HashSet<String>,
    crcs: Vec<u32>,
}

impl RankState {
    fn tensor_index(&self, name: &str) -> Option<usize> {
        self.plan.tensors.iter().position(|t| t.name == name)
    }
}

fn view_row_bytes(t: &SrcTensor) -> u64 {
    let mut rb = t.dtype.bytes();
    for &d in t.shape.iter().skip(1) {
        rb *= d;
    }
    rb
}

/// Copies one piece of a source tensor into the output tensor's natural bytes.
/// `src` is the whole source tensor, row-major, `[rows, row_bytes]`.
fn place(dst: &mut [u8], src: &[u8], row_bytes: u64, p: &plan::Piece) {
    for r in 0..p.nrows {
        let s = ((p.row0 + r) * row_bytes + p.col0) as usize;
        let d = (p.dst + r * p.ncols) as usize;
        dst[d..d + p.ncols as usize].copy_from_slice(&src[s..s + p.ncols as usize]);
    }
}

fn assemble(o: &OutTensor, ck: &Checkpoint, buffers: &HashMap<usize, Vec<u8>>) -> Vec<u8> {
    let mut nat = vec![0u8; o.bytes as usize];
    for p in &o.pieces {
        let src = &buffers[&p.src];
        place(&mut nat, src, view_row_bytes(&ck.tensors[p.src]), p);
    }
    nat
}

fn read_tensor(reader: &dyn Reader, t: &SrcTensor) -> std::io::Result<Vec<u8>> {
    let mut buf = vec![0u8; t.bytes as usize];
    reader.read_at(t.shard, t.offset, &mut buf)?;
    Ok(buf)
}

pub(crate) fn write_padded(file: &File, offset: u64, bytes: &[u8]) -> std::io::Result<()> {
    let padded = align_up(bytes.len() as u64, ALIGN) as usize;
    let mut buf = vec![0u8; padded];
    buf[..bytes.len()].copy_from_slice(bytes);
    file.write_all_at(&buf, offset)
}

pub(crate) fn read_padded(file: &File, offset: u64, bytes: u64) -> std::io::Result<Vec<u8>> {
    let padded = align_up(bytes, ALIGN) as usize;
    let mut buf = vec![0u8; padded];
    file.read_exact_at(&mut buf, offset)?;
    Ok(buf)
}

/// One tensor for one rank: assemble, tile, write, prove.
fn convert_tensor(
    rs: &mut RankState,
    o: &OutTensor,
    buffers: &HashMap<usize, Vec<u8>>,
    ck: &Checkpoint,
    verify: bool,
) -> Result<(), String> {
    let nat = assemble(o, ck, buffers);
    let mut til = nat.clone();
    if o.layout != Layout::Natural {
        let g = tile::tile_geom(o.layout, o.dtype, &o.shape)?;
        tile::retile(o.layout, o.dtype, g, &nat, &mut til, false);
    }
    write_padded(&rs.file, o.offset, &til).map_err(|e| format!("{}: write: {e}", o.name))?;
    let crc = crc32c(&til);
    if verify {
        let back = read_padded(&rs.file, o.offset, o.bytes).map_err(|e| format!("{}: read back: {e}", o.name))?;
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
            return Err(format!("{}: the written bytes do not match the checkpoint slice", o.name));
        }
    }
    if let Some(i) = rs.tensor_index(&o.name) {
        rs.crcs[i] = crc;
    }
    Ok(())
}

/// Writes the table of contents and the header, then renames `.part` into place.
fn finalise(rs: &mut RankState, out: &Path, model: &str, source_note: &str) -> Result<u64, String> {
    let mut toc = vec![0u8; (rs.plan.tensors.len() * TOC_ENTRY_BYTES) as usize];
    for (i, o) in rs.plan.tensors.iter().enumerate() {
        let mut e: TocEntry = o.toc_entry();
        e.crc32c = rs.crcs[i];
        toc[i * TOC_ENTRY_BYTES..(i + 1) * TOC_ENTRY_BYTES].copy_from_slice(&e.to_bytes());
    }
    let toc_padded = (rs.plan.data_offset - rs.plan.toc_offset) as usize;
    let mut tbuf = vec![0u8; toc_padded];
    tbuf[..toc.len()].copy_from_slice(&toc);
    let mut h = Header::new(rs.rank, rs.plan.world, model, source_note, &crate::now_iso8601());
    h.n_tensors = rs.plan.tensors.len() as u32;
    h.toc_offset = rs.plan.toc_offset;
    h.data_offset = rs.plan.data_offset;
    h.data_bytes = rs.plan.file_bytes - rs.plan.data_offset;
    h.file_bytes = rs.plan.file_bytes;
    h.toc_crc32c = crc32c(&toc);
    rs.file.write_all_at(&tbuf, rs.plan.toc_offset).map_err(|e| format!("TOC: {e}"))?;
    rs.file.write_all_at(&h.to_bytes(), 0).map_err(|e| format!("header: {e}"))?;
    rs.file.sync_all().map_err(|e| format!("fsync: {e}"))?;
    if rs.work != out {
        std::fs::rename(&rs.work, out).map_err(|e| format!("rename {}: {e}", rs.work.display()))?;
    }
    Ok(rs.plan.file_bytes)
}

/// Opens (or resumes) one rank's output file and journal.
fn open_rank(ck: &Checkpoint, o: &Options, rank: u32) -> Result<RankState, String> {
    let plan = plan::make_plan(&ck.tensors, rank, o.world)?;
    let out = dsw_path(o, rank);
    if let Some(d) = out.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let tmp = part(&out);
    let jpath = journal_path(o, rank);
    let fp = fingerprint(&plan, &o.model);

    // The TOC's checksums live in the journal, not in the (still headerless) file:
    // a resumed run must restore them for the tensors it does not convert again.
    let mut crcs = vec![0u32; plan.tensors.len()];
    let (committed, journal) = if o.resume {
        let (records, state) = Journal::load(&jpath).map_err(|e| format!("{}: {e}", jpath.display()))?;
        if records.is_empty() {
            let mut j = Journal::create(&jpath).map_err(|e| format!("{}: {e}", jpath.display()))?;
            j.append(&Record::Begin {
                version: crate::journal::VERSION,
                fingerprint: fp,
                rank,
                world: o.world,
                out: out.display().to_string(),
            })
            .map_err(|e| format!("journal: {e}"))?;
            (HashSet::new(), j)
        } else {
            Journal::begin_matches(&records, &fp, rank, o.world)?;
            if state.done {
                return Err(format!("{} is already complete (delete it or pass --replace)", out.display()));
            }
            for r in &records {
                if let Record::Commit { batch, outputs } = r {
                    if let Some(oc) = outputs.first() {
                        if let Some(i) = plan.tensors.iter().position(|t| &t.name == batch) {
                            crcs[i] = oc.crc32c;
                        }
                    }
                }
            }
            (state.committed, Journal::create(&jpath).map_err(|e| format!("{}: {e}", jpath.display()))?)
        }
    } else {
        let _ = std::fs::remove_file(&jpath);
        let _ = std::fs::remove_file(&tmp);
        let mut j = Journal::create(&jpath).map_err(|e| format!("{}: {e}", jpath.display()))?;
        j.append(&Record::Begin {
            version: crate::journal::VERSION,
            fingerprint: fp,
            rank,
            world: o.world,
            out: out.display().to_string(),
        })
        .map_err(|e| format!("journal: {e}"))?;
        (HashSet::new(), j)
    };

    // A resumed run must reopen whatever already holds the committed tensors: the
    // `.part` file, or the final file if an earlier run got as far as renaming it.
    let work = if tmp.exists() {
        tmp
    } else if !committed.is_empty() && out.exists() {
        out.clone()
    } else {
        tmp
    };
    let file = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(committed.is_empty())
        .open(&work)
        .map_err(|e| format!("{}: {e}", work.display()))?;
    if file.metadata().map_err(|e| e.to_string())?.len() != plan.file_bytes {
        file.set_len(plan.file_bytes).map_err(|e| format!("{}: {e}", work.display()))?;
    }
    Ok(RankState { rank, plan, file, work, journal, committed, crcs })
}

/// Runs the conversion for the ranks in `o.ranks`, shard by shard.
pub fn run(ck: &Checkpoint, reader: &dyn Reader, o: &Options, source_note: &str) -> Result<(), String> {
    let mut states: Vec<RankState> = Vec::new();
    for &r in &o.ranks {
        states.push(open_rank(ck, o, r)?);
    }
    let tables: Vec<Table> = if o.engram { engram::tables(ck)? } else { Vec::new() };
    if o.consume && (o.ranks.len() != o.world as usize || !o.engram) {
        return Err(
            "--consume releases source blocks that other ranks still need: convert every rank and the Engram shards, or drop --consume"
                .into(),
        );
    }

    // The Engram shards first: releasing them early keeps the peak near the
    // checkpoint's own size instead of checkpoint + outputs.
    let engram_shards: HashSet<usize> =
        tables.iter().flat_map(|t| [ck.tensors[t.weight].shard, ck.tensors[t.scale].shard]).collect();
    let mut order: Vec<usize> = (0..ck.shards.len()).collect();
    order.sort_by_key(|&s| (!engram_shards.contains(&s), s));

    // A batch is one output tensor and every source tensor it reads. A stacked
    // expert tensor can span shards, so the batch runs at the last shard its
    // sources live in: until then, no shard it touches may be released.
    struct Batch {
        at: usize,
        srcs: Vec<usize>,
        consumers: Vec<(usize, usize)>,
    }
    let mut batches: Vec<Batch> = Vec::new();
    let mut by_srcs: HashMap<Vec<usize>, usize> = HashMap::new();
    for (ri, st) in states.iter().enumerate() {
        for (oi, ot) in st.plan.tensors.iter().enumerate() {
            let mut srcs: Vec<usize> = ot.pieces.iter().map(|p| p.src).collect();
            srcs.sort_unstable();
            srcs.dedup();
            if srcs.is_empty() {
                continue;
            }
            let at = srcs.iter().map(|&s| ck.tensors[s].shard).max().unwrap();
            match by_srcs.get(&srcs) {
                Some(&i) => batches[i].consumers.push((ri, oi)),
                None => {
                    by_srcs.insert(srcs.clone(), batches.len());
                    batches.push(Batch { at, srcs, consumers: vec![(ri, oi)] });
                }
            }
        }
    }

    let started = std::time::Instant::now();
    let totals: Vec<u64> = states.iter().map(|s| s.plan.tensors.iter().map(|t| t.bytes).sum()).collect();
    let mut released: u64 = 0;
    let mut warned_release = false;
    println!(
        "[dsw-prepare] converting {} rank(s): {} tensors and {:.1} GB each, {} shard(s), {}",
        states.len(),
        states[0].plan.tensors.len(),
        totals[0] as f64 / 1e9,
        order.len(),
        if o.consume { "the sources are released as they are consumed" } else { "the checkpoint is kept" }
    );

    for (step, &shard) in order.iter().enumerate() {
        reader.prepare(shard).map_err(|e| format!("{}: prepare: {e}", ck.shards[shard].name))?;
        // The scratch copy a hybrid reader fetched goes away even if this shard
        // fails, so a retry starts from a known state.
        let _guard = ShardGuard(reader, shard);
        // 1. The batches whose last source is in this shard.
        for b in batches.iter().filter(|b| b.at == shard) {
            if b.consumers.iter().all(|(ri, oi)| states[*ri].committed.contains(&states[*ri].plan.tensors[*oi].name)) {
                continue;
            }
            let mut buffers: HashMap<usize, Vec<u8>> = HashMap::new();
            for &s in &b.srcs {
                let t = &ck.tensors[s];
                buffers.insert(s, read_tensor(reader, t).map_err(|e| format!("{}: read: {e}", t.name))?);
            }
            for &(ri, oi) in &b.consumers {
                let st = &mut states[ri];
                let ot = st.plan.tensors[oi].clone();
                if st.committed.contains(&ot.name) {
                    continue;
                }
                convert_tensor(st, &ot, &buffers, ck, o.verify)?;
                st.journal
                    .append(&Record::Commit {
                        batch: ot.name.clone(),
                        outputs: vec![OutputCommit {
                            path: dsw_path(o, st.rank).display().to_string(),
                            offset: ot.offset,
                            len: ot.bytes,
                            crc32c: st.crcs[oi],
                        }],
                    })
                    .map_err(|e| format!("journal: {e}"))?;
                st.committed.insert(ot.name.clone());
            }
        }

        // 2. Engram tables held by this shard, once per rank.
        for table in &tables {
            if ck.tensors[table.weight].shard != shard {
                continue;
            }
            for st in states.iter_mut() {
                let batch = format!("engram:{}", table.layer);
                if st.committed.contains(&batch) {
                    continue;
                }
                write_engram(ck, st, table, reader, o)?;
                st.journal
                    .append(&Record::Commit {
                        batch: batch.clone(),
                        outputs: vec![OutputCommit {
                            path: engram_path(o, st.rank, table.layer).display().to_string(),
                            offset: 0,
                            len: engram::shard_bytes(table, st.rank, o.world),
                            crc32c: 0,
                        }],
                    })
                    .map_err(|e| format!("journal: {e}"))?;
                st.committed.insert(batch);
            }
        }

        // 3. Everything in this shard is committed for every rank: release it.
        //    The punch is aligned *inwards*, so it only ever covers whole blocks
        //    the tensor owns: the shard's own header (the safetensors JSON the
        //    checkpoint is re-opened from) and the bytes it shares with a
        //    neighbour stay intact, which is what makes a resume possible.
        if o.consume && fully_consumed(ck, shard, &tables, &states) {
            let mut freed_here = 0u64;
            for &ti in &ck.shards[shard].tensors {
                let t = &ck.tensors[ti];
                let start = align_up(t.offset, ALIGN);
                let end = (t.offset + t.bytes) / ALIGN * ALIGN;
                if end <= start {
                    continue;
                }
                match reader.release(shard, start, end - start) {
                    Ok(()) => freed_here += end - start,
                    Err(e) if e.kind() == std::io::ErrorKind::Unsupported => {
                        // No hole punching here: keep the checkpoint rather than
                        // truncate a file the next run has to open.
                        if !warned_release {
                            warned_release = true;
                            eprintln!(
                                "[dsw-prepare] WARN: {} cannot free blocks ({e}); the checkpoint is kept whole",
                                ck.shards[shard].path.as_ref().map(|p| p.display().to_string()).unwrap_or_default()
                            );
                        }
                        break;
                    }
                    Err(e) => return Err(format!("{}: release: {e}", t.name)),
                }
            }
            released += freed_here;
        }

        // Progress, once a shard: what each rank has committed, how much is on
        // disk, and how long the rest should take at the pace so far.
        let elapsed = started.elapsed();
        for (i, st) in states.iter().enumerate() {
            let done = st.committed.iter().filter(|n| !n.starts_with("engram:")).count();
            let written: u64 =
                st.plan.tensors.iter().filter(|t| st.committed.contains(&t.name)).map(|t| t.bytes).sum();
            let total = totals[i].max(1);
            let eta = if written > 0 && written < total {
                hhmmss(elapsed.mul_f64((total - written) as f64 / written as f64))
            } else {
                "-".to_string()
            };
            println!(
                "[{}/{}] rank {}: {}/{} tensors ({}%), {:.1}/{:.1} GB written{}{}, eta {eta}",
                step + 1,
                order.len(),
                st.rank,
                done,
                st.plan.tensors.len(),
                written * 100 / total,
                written as f64 / 1e9,
                total as f64 / 1e9,
                if o.consume { format!(", {:.1} GB released", released as f64 / 1e9) } else { String::new() },
                format!(", {}", hhmmss(elapsed)),
            );
        }
    }

    for st in &mut states {
        let out = dsw_path(o, st.rank);
        let bytes = finalise(st, &out, &o.model, source_note)?;
        st.journal
            .append(&Record::Done { file: out.display().to_string(), bytes, tensors: st.plan.tensors.len() as u64 })
            .map_err(|e| format!("journal: {e}"))?;
        println!("rank {}: {} ({} tensors, {:.1} GB)", st.rank, out.display(), st.plan.tensors.len(), bytes as f64 / 1e9);
    }
    Ok(())
}

/// `m:ss`, or `h:mm:ss` past an hour.
fn hhmmss(d: std::time::Duration) -> String {
    let s = d.as_secs();
    if s >= 3600 {
        format!("{}:{:02}:{:02}", s / 3600, s % 3600 / 60, s % 60)
    } else {
        format!("{}:{:02}", s / 60, s % 60)
    }
}

/// Runs `finish` on the reader whatever happens to the shard.
struct ShardGuard<'a>(&'a dyn Reader, usize);

impl Drop for ShardGuard<'_> {
    fn drop(&mut self) {
        self.0.finish(self.1);
    }
}

fn fully_consumed(ck: &Checkpoint, shard: usize, tables: &[Table], states: &[RankState]) -> bool {
    for st in states {
        for ot in &st.plan.tensors {
            if ot.pieces.iter().any(|p| ck.tensors[p.src].shard == shard) && !st.committed.contains(&ot.name) {
                return false;
            }
        }
        for t in tables {
            if ck.tensors[t.weight].shard == shard && !st.committed.contains(&format!("engram:{}", t.layer)) {
                return false;
            }
        }
    }
    true
}

fn write_engram(ck: &Checkpoint, st: &mut RankState, table: &Table, reader: &dyn Reader, o: &Options) -> Result<(), String> {
    let path = engram_path(o, st.rank, table.layer);
    if let Some(d) = path.parent() {
        std::fs::create_dir_all(d).map_err(|e| format!("{}: {e}", d.display()))?;
    }
    let tmp = part(&path);
    let (lo, hi) = engram::rows_of(table, st.rank, o.world);
    let w = &ck.tensors[table.weight];
    let s = &ck.tensors[table.scale];
    let f = OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .truncate(true)
        .open(&tmp)
        .map_err(|e| format!("{}: {e}", tmp.display()))?;
    f.set_len(engram::HEADER_BYTES + (hi - lo) * engram::ROW_BYTES).map_err(|e| e.to_string())?;
    f.write_all_at(&engram::header(table.layer, lo, hi, table.rows), 0).map_err(|e| e.to_string())?;
    let chunk_rows = ((64u64 << 20) / engram::ROW_BYTES).max(1);
    let mut row = lo;
    let mut last_pct = 0u64;
    while row < hi {
        let n = (hi - row).min(chunk_rows);
        let mut weight = vec![0u8; (n * engram::WEIGHT_BYTES) as usize];
        let mut scale = vec![0u8; (n * engram::SCALE_BYTES) as usize];
        reader
            .read_at(w.shard, w.offset + row * engram::WEIGHT_BYTES, &mut weight)
            .map_err(|e| format!("{}: {e}", w.name))?;
        reader
            .read_at(s.shard, s.offset + row * engram::SCALE_BYTES, &mut scale)
            .map_err(|e| format!("{}: {e}", s.name))?;
        let mut recs = vec![0u8; (n * engram::ROW_BYTES) as usize];
        for k in 0..n as usize {
            let a = k * engram::WEIGHT_BYTES as usize;
            let b = k * engram::SCALE_BYTES as usize;
            let r = engram::record(
                &weight[a..a + engram::WEIGHT_BYTES as usize],
                &scale[b..b + engram::SCALE_BYTES as usize],
            );
            let d = k * engram::ROW_BYTES as usize;
            recs[d..d + engram::ROW_BYTES as usize].copy_from_slice(&r);
        }
        let at = engram::HEADER_BYTES + (row - lo) * engram::ROW_BYTES;
        f.write_all_at(&recs, at).map_err(|e| format!("{}: {e}", path.display()))?;
        if o.verify {
            let mut back = vec![0u8; recs.len()];
            f.read_exact_at(&mut back, at).map_err(|e| e.to_string())?;
            if back != recs {
                return Err(format!("{}: records at row {row} do not read back", path.display()));
            }
        }
        row += n;
        // The tables are the long pole: a line every 10 %, which is minutes apart
        // on the real 95 GB shards.
        let pct = (row - lo) * 100 / (hi - lo).max(1);
        if pct >= last_pct + 10 {
            last_pct = pct;
            println!("  rank {}: Engram layer {} {pct}% ({} rows)", st.rank, table.layer, row - lo);
        }
    }
    f.sync_all().map_err(|e| e.to_string())?;
    std::fs::rename(&tmp, &path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(())
}

/// Independent verification of a finished file against the checkpoint: every TOC
/// entry is re-derived from its own declared description — kind, rank, world,
/// stacking, segments — not from the plan, un-tiled with the inverse mapping and
/// compared byte for byte. This is the oracle the writer is tested against, and
/// it needs the checkpoint on disk.
pub fn verify_file(ck: &Checkpoint, path: &Path) -> Result<(), String> {
    let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
    let size = file.metadata().map_err(|e| e.to_string())?.len();
    let mut hb = [0u8; HEADER_BYTES];
    file.read_exact_at(&mut hb, 0).map_err(|e| e.to_string())?;
    let h = Header::from_bytes(&hb);
    if h.magic != crate::dsw::MAGIC {
        return Err(format!("{}: not a .dsw file", path.display()));
    }
    if h.header_bytes != HEADER_BYTES as u32 || h.toc_entry_bytes != TOC_ENTRY_BYTES as u32 {
        return Err(format!("{}: bad header sizes", path.display()));
    }
    if h.file_bytes != size {
        return Err(format!("{}: header says {} bytes, the file is {size}", path.display(), h.file_bytes));
    }
    let toc_len = (h.data_offset - h.toc_offset) as usize;
    let mut tb = vec![0u8; toc_len];
    file.read_exact_at(&mut tb, h.toc_offset).map_err(|e| e.to_string())?;
    if crc32c(&tb[..h.n_tensors as usize * TOC_ENTRY_BYTES]) != h.toc_crc32c {
        return Err(format!("{}: TOC CRC mismatch", path.display()));
    }
    let mut handles: HashMap<usize, File> = HashMap::new();
    for i in 0..h.n_tensors as usize {
        let e = TocEntry::from_bytes(&tb[i * TOC_ENTRY_BYTES..(i + 1) * TOC_ENTRY_BYTES]);
        let padded = align_up(e.bytes, ALIGN) as usize;
        let mut data = vec![0u8; padded];
        file.read_exact_at(&mut data, e.offset).map_err(|e| e.to_string())?;
        if crc32c(&data[..e.bytes as usize]) != e.crc32c {
            return Err(format!("{}: {}: CRC mismatch", path.display(), e.name()));
        }
        if data[e.bytes as usize..].iter().any(|&b| b != 0) {
            return Err(format!("{}: {}: padding is not zeros", path.display(), e.name()));
        }
        let mut nat = data[..e.bytes as usize].to_vec();
        if e.layout() != Layout::Natural {
            let g = tile::tile_geom(e.layout(), e.dtype(), e.shape()).map_err(|w| format!("{}: {w}", e.name()))?;
            let mut out = vec![0u8; e.bytes as usize];
            tile::retile(e.layout(), e.dtype(), g, &nat, &mut out, true);
            nat = out;
        }
        let want = expected_natural(ck, &e, &mut handles, h.rank, h.world)?;
        if want.len() != nat.len() {
            return Err(format!("{}: {} bytes in the file, {} from the checkpoint", e.name(), nat.len(), want.len()));
        }
        if want != nat {
            return Err(format!("{}: bytes differ from the checkpoint", e.name()));
        }
    }
    Ok(())
}

/// Re-derives a TOC entry's natural bytes from the checkpoint, using only the
/// entry's own description of how it was cut.
fn expected_natural(
    ck: &Checkpoint,
    e: &TocEntry,
    handles: &mut HashMap<usize, File>,
    rank: u32,
    world: u32,
) -> Result<Vec<u8>, String> {
    let ns = if e.nstack > 0 { e.nstack as usize } else { 1 };
    let per = e.bytes / ns as u64;
    let mut out = Vec::with_capacity(e.bytes as usize);
    for k in 0..ns {
        let name = e.source_name().replace("{e}", &k.to_string());
        let t = ck.at(&name).ok_or_else(|| format!("{}: no source {name}", e.name()))?;
        let rb = view_row_bytes(t);
        let rows = t.shape.first().copied().unwrap_or(1);
        let rows_piece = match e.kind() {
            SliceKind::Replicated => (0, rows),
            SliceKind::Rows => {
                let n = rows / world as u64;
                (n * rank as u64, n)
            }
            SliceKind::Cols => (0, rows),
            SliceKind::RowsSeg => {
                let seg = rows / e.nseg as u64;
                let n = seg / world as u64;
                (0, n * e.nseg as u64)
            }
        };
        let cols_piece = match e.kind() {
            SliceKind::Cols => {
                let nc = t.shape[1] / world as u64;
                (nc * t.dtype.bytes() * rank as u64, nc * t.dtype.bytes())
            }
            _ => (0, rb),
        };
        let fd = match handles.entry(t.shard) {
            std::collections::hash_map::Entry::Occupied(o) => o.into_mut(),
            std::collections::hash_map::Entry::Vacant(v) => {
                let p = ck.shards[t.shard]
                    .path
                    .as_ref()
                    .ok_or_else(|| "the checkpoint is not on disk".to_string())?;
                v.insert(File::open(p).map_err(|e| format!("{}: {e}", p.display()))?)
            }
        };
        let mut part = vec![0u8; per as usize];
        match e.kind() {
            SliceKind::RowsSeg => {
                let seg = rows / e.nseg as u64;
                let n = seg / world as u64;
                for s in 0..e.nseg as u64 {
                    let row0 = s * seg + n * rank as u64;
                    let mut chunk = vec![0u8; (n * rb) as usize];
                    fd.read_exact_at(&mut chunk, t.offset + row0 * rb).map_err(|x| format!("{}: {x}", t.name))?;
                    let at = (s * n * rb) as usize;
                    part[at..at + chunk.len()].copy_from_slice(&chunk);
                }
            }
            SliceKind::Cols => {
                let (c0, nc) = cols_piece;
                for r in 0..rows_piece.1 {
                    let mut row = vec![0u8; nc as usize];
                    fd.read_exact_at(&mut row, t.offset + r * rb + c0).map_err(|x| format!("{}: {x}", t.name))?;
                    let at = (r * nc) as usize;
                    part[at..at + nc as usize].copy_from_slice(&row);
                }
            }
            _ => {
                let (r0, n) = rows_piece;
                fd.read_exact_at(&mut part, t.offset + r0 * rb).map_err(|x| format!("{}: {x}", t.name))?;
                let _ = n;
            }
        }
        out.extend_from_slice(&part);
    }
    Ok(out)
}


