//! Which slice of which checkpoint tensor each rank gets, transcribed from
//! `engine/weights/plan.cpp`. The table is the whole partition: a split changes
//! by editing one line here, exactly as it does there.
//!
//! This module is deliberately independent of `dsw_convert`: it must produce the
//! same plan, and a disagreement is a bug in one of the two. `tests/plan_dump` in
//! the repo is what compares them.

use std::collections::{BTreeMap, HashMap};

use crate::dsw::{align_up, DType, Layout, SliceKind, TocEntry, ALIGN, HEADER_BYTES, TOC_ENTRY_BYTES};
use crate::safetensors::{Checkpoint, SrcTensor};
use crate::tile::tile_geom;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rule {
    Skip,
    Rep,
    Rows,
    Cols,
    /// Dim 0 is `kSegs` equal segments; each segment's rows are split as in `Rows`.
    RowsSeg5,
}

const SEGS: u32 = 5;

#[derive(Debug, Clone, Copy)]
struct Entry {
    name: &'static str,
    rule: Rule,
    layout: Layout,
}

const fn e(name: &'static str, rule: Rule) -> Entry {
    Entry { name, rule, layout: Layout::Natural }
}

const fn t(name: &'static str, rule: Rule, layout: Layout) -> Entry {
    Entry { name, rule, layout }
}

const GLOBAL: &[Entry] = &[
    e("embed.weight", Rule::Rep), // a lookup reads one row, so no collective is needed
    t("head.weight", Rule::Rows, Layout::L2),
    e("norm.weight", Rule::Rep),
    e("image_start", Rule::Skip),
    e("image_end", Rule::Skip),
    e("image_newline", Rule::Skip),
];

/// Every main layer (`layers.N.`) and every DSpark block (`mtp.N.`).
const BLOCK: &[Entry] = &[
    e("attn_norm.weight", Rule::Rep),
    e("ffn_norm.weight", Rule::Rep),
    e("attn.attn_sink", Rule::Rows),
    t("attn.wq_a.weight", Rule::Rows, Layout::L1),
    e("attn.wq_a.scale", Rule::Rows),
    e("attn.q_norm.weight", Rule::Rep),
    t("attn.wkv.weight", Rule::Rows, Layout::L1),
    e("attn.wkv.scale", Rule::Rows),
    e("attn.kv_norm.weight", Rule::Rep),
    t("attn.wq_b.weight", Rule::Rows, Layout::L1),
    e("attn.wq_b.scale", Rule::Rows),
    t("attn.wo_a.weight", Rule::Rows, Layout::L1),
    e("attn.wo_a.scale", Rule::Rows),
    t("attn.wo_b.weight", Rule::Cols, Layout::L1),
    e("attn.wo_b.scale", Rule::Cols),
    e("hc_attn_fn", Rule::Cols),
    e("hc_ffn_fn", Rule::Cols),
    e("hc_attn_base", Rule::Rep),
    e("hc_ffn_base", Rule::Rep),
    e("hc_attn_scale", Rule::Rep),
    e("hc_ffn_scale", Rule::Rep),
    t("ffn.gate.weight", Rule::Rows, Layout::L2),
    e("ffn.gate.bias", Rule::Rep),
    e("ffn.gate.bias_vl", Rule::Rep),
    t("ffn.shared_experts.w1.weight", Rule::Rows, Layout::L1),
    e("ffn.shared_experts.w1.scale", Rule::Rows),
    t("ffn.shared_experts.w3.weight", Rule::Rows, Layout::L1),
    e("ffn.shared_experts.w3.scale", Rule::Rows),
    t("ffn.shared_experts.w2.weight", Rule::Cols, Layout::L1),
    e("ffn.shared_experts.w2.scale", Rule::Cols),
    t("ffn.experts.N.w1.weight", Rule::Rows, Layout::L3),
    t("ffn.experts.N.w1.scale", Rule::Rows, Layout::L3),
    t("ffn.experts.N.w3.weight", Rule::Rows, Layout::L3),
    t("ffn.experts.N.w3.scale", Rule::Rows, Layout::L3),
    t("ffn.experts.N.w2.weight", Rule::Cols, Layout::L3),
    t("ffn.experts.N.w2.scale", Rule::Cols, Layout::L3),
];

const LAYER_ONLY: &[Entry] = &[
    e("attn.compressor.wkv.weight", Rule::Rows),
    e("attn.compressor.wgate.weight", Rule::Rows),
    e("attn.compressor.norm.weight", Rule::Rep),
    t("attn.indexer.wq_b.weight", Rule::Rep, Layout::L1),
    e("attn.indexer.wq_b.scale", Rule::Rep),
    e("attn.indexer.weights_proj.weight", Rule::Rep),
    e("attn.indexer.wk.weight", Rule::Rep),
    e("attn.indexer.k_norm.weight", Rule::Rep),
    // 4 keys + 1 value, each 5120 wide: every rank takes the same quarter of each
    e("engram.wkv.weight", Rule::RowsSeg5),
    e("engram.wkv.scale", Rule::RowsSeg5),
    e("engram.q_weight", Rule::Rep),
    e("engram.k_weight", Rule::Rep),
    e("engram.embed.weight", Rule::Skip), // the tables live in the kit's per-rank shards
    e("engram.embed.scale", Rule::Skip),
];

const MTP_ONLY: &[Entry] = &[
    t("main_proj.weight", Rule::Rows, Layout::L1),
    e("main_proj.scale", Rule::Rows),
    e("main_norm.weight", Rule::Rep),
    e("norm.weight", Rule::Rep),
    e("markov_head.embed.weight", Rule::Rep),
    t("markov_head.head.weight", Rule::Rows, Layout::L2),
    e("confidence_head.proj.weight", Rule::Rep),
];

fn rules() -> &'static HashMap<String, Entry> {
    static RULES: std::sync::LazyLock<HashMap<String, Entry>> = std::sync::LazyLock::new(|| {
        let mut m = HashMap::new();
        let mut add = |k: String, e: Entry| {
            assert!(m.insert(k.clone(), e).is_none(), "plan table lists {k} twice");
        };
        for e in GLOBAL {
            add(e.name.to_string(), *e);
        }
        for e in BLOCK {
            add(format!("layers.N.{}", e.name), *e);
            add(format!("mtp.N.{}", e.name), *e);
        }
        for e in LAYER_ONLY {
            add(format!("layers.N.{}", e.name), *e);
        }
        for e in MTP_ONLY {
            add(format!("mtp.N.{}", e.name), *e);
        }
        m
    });
    &RULES
}

/// Every dot-separated all-digit component becomes `N`.
pub fn normalize(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    for (i, comp) in name.split('.').enumerate() {
        if i > 0 {
            out.push('.');
        }
        if !comp.is_empty() && comp.bytes().all(|c| c.is_ascii_digit()) {
            out.push('N');
        } else {
            out.push_str(comp);
        }
    }
    out
}

fn group_of(name: &str) -> u32 {
    if let Some(rest) = name.strip_prefix("layers.") {
        if let Some(n) = rest.split('.').next().and_then(|s| s.parse::<u32>().ok()) {
            return 1 + n;
        }
    }
    if let Some(rest) = name.strip_prefix("mtp.") {
        if let Some(n) = rest.split('.').next().and_then(|s| s.parse::<u32>().ok()) {
            return 1000 + n;
        }
    }
    0
}

fn component_of(n: &str) -> String {
    let mtp = n.starts_with("mtp.");
    let has = |s: &str| n.contains(s);
    let c = if n == "embed.weight" {
        "embed"
    } else if n == "head.weight" {
        "head"
    } else if has(".experts.N.") {
        "routed experts"
    } else if has("shared_experts") {
        "shared expert"
    } else if has("compressor") {
        "compressor"
    } else if has("indexer") {
        "indexer"
    } else if has("engram") {
        "engram wkv/gates"
    } else if has("hc_") {
        "mHC"
    } else if has("ffn.gate.") {
        "router"
    } else if has("markov_head") || has("confidence_head") || has("main_proj") {
        "draft heads"
    } else if has("norm") {
        "norms"
    } else if has("attn.") {
        "attention"
    } else {
        "other"
    };
    if mtp {
        format!("dspark {c}")
    } else {
        c.to_string()
    }
}

/// One rectangle of a source tensor, viewed as `[rows, row_bytes]`, copied to a
/// contiguous run of the output tensor (`rows` land back to back, `ncols` bytes
/// each).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Piece {
    /// Index into [`Checkpoint::tensors`].
    pub src: usize,
    pub row0: u64,
    pub nrows: u64,
    pub col0: u64,
    pub ncols: u64,
    /// Byte offset in the output tensor.
    pub dst: u64,
}

#[derive(Debug, Clone)]
pub struct OutTensor {
    pub name: String,
    /// Source name, `{e}` for the expert index when stacked.
    pub source: String,
    pub slice: String,
    pub component: String,
    pub dtype: DType,
    pub kind: SliceKind,
    pub layout: Layout,
    pub nstack: u32,
    pub nseg: u32,
    pub shape: Vec<u64>,
    pub src_shape: Vec<u64>,
    pub bytes: u64,
    pub offset: u64,
    pub pieces: Vec<Piece>,
}

impl OutTensor {
    /// The table-of-contents record (the CRC is filled in by the writer).
    pub fn toc_entry(&self) -> TocEntry {
        TocEntry::new(
            &self.name,
            &self.source,
            &self.slice,
            self.dtype,
            &self.shape,
            self.offset,
            self.bytes,
            self.kind,
            self.nstack,
            self.nseg,
            self.layout,
            &self.src_shape,
        )
    }
}

#[derive(Debug)]
pub struct Plan {
    pub rank: u32,
    pub world: u32,
    pub tensors: Vec<OutTensor>,
    pub skipped: Vec<String>,
    pub toc_offset: u64,
    pub data_offset: u64,
    pub file_bytes: u64,
}

impl Plan {
    pub fn total_bytes(&self) -> u64 {
        self.tensors.iter().map(|t| align_up(t.bytes, ALIGN)).sum()
    }
}

struct View {
    rows: u64,
    row_bytes: u64,
}

fn view_of(t: &SrcTensor) -> View {
    let mut rb = t.dtype.bytes();
    for &d in t.shape.iter().skip(1) {
        rb *= d;
    }
    View { rows: t.shape.first().copied().unwrap_or(1), row_bytes: rb }
}

fn dtype_of(t: &SrcTensor) -> Result<DType, String> {
    if t.dtype == DType::E2M1X2 {
        if !t.name.contains(".experts.") || t.name.contains("shared_experts") {
            return Err(format!("{} is packed FP4 but is not a routed expert weight", t.name));
        }
    }
    Ok(t.dtype)
}

struct Cut {
    pieces: Vec<Piece>,
    shape: Vec<u64>,
    bytes: u64,
    kind: SliceKind,
    nseg: u32,
    desc: String,
}

fn cut(t: &SrcTensor, src_idx: usize, rule: Rule, r: u32, w: u32, base: u64) -> Result<Cut, String> {
    let v = view_of(t);
    let (r64, w64) = (r as u64, w as u64);
    let mut c = Cut { pieces: Vec::new(), shape: t.shape.clone(), bytes: 0, kind: SliceKind::Replicated, nseg: 0, desc: String::new() };
    match rule {
        Rule::Rep => {
            c.pieces.push(Piece { src: src_idx, row0: 0, nrows: v.rows, col0: 0, ncols: v.row_bytes, dst: base });
            c.kind = SliceKind::Replicated;
            c.desc = "replicated".into();
        }
        Rule::Rows => {
            if t.shape.is_empty() || v.rows % w64 != 0 {
                return Err(format!("{}: {} rows do not split {w} ways", t.name, v.rows));
            }
            let n = v.rows / w64;
            c.pieces.push(Piece { src: src_idx, row0: n * r64, nrows: n, col0: 0, ncols: v.row_bytes, dst: base });
            c.shape[0] = n;
            c.kind = SliceKind::Rows;
            c.desc = format!("rows [{n}*r, +{n}) of {}", v.rows);
        }
        Rule::Cols => {
            if t.shape.len() != 2 || t.shape[1] % w64 != 0 {
                return Err(format!("{}: columns do not split {w} ways", t.name));
            }
            let nc = t.shape[1] / w64;
            let eb = t.dtype.bytes();
            c.pieces.push(Piece { src: src_idx, row0: 0, nrows: v.rows, col0: nc * eb * r64, ncols: nc * eb, dst: base });
            c.shape[1] = nc;
            c.kind = SliceKind::Cols;
            let packed = if t.dtype == DType::E2M1X2 { " (packed bytes)" } else { "" };
            c.desc = format!("cols [{nc}*r, +{nc}) of {}{packed}", t.shape[1]);
        }
        Rule::RowsSeg5 => {
            if t.shape.is_empty() || v.rows % (SEGS as u64 * w64) != 0 {
                return Err(format!("{}: rows do not split into {SEGS} x {w}", t.name));
            }
            let seg = v.rows / SEGS as u64;
            let n = seg / w64;
            for s in 0..SEGS as u64 {
                c.pieces.push(Piece {
                    src: src_idx,
                    row0: s * seg + n * r64,
                    nrows: n,
                    col0: 0,
                    ncols: v.row_bytes,
                    dst: base + s * n * v.row_bytes,
                });
            }
            c.shape[0] = n * SEGS as u64;
            c.kind = SliceKind::RowsSeg;
            c.nseg = SEGS;
            c.desc = format!("{SEGS} segments of {seg} rows, each rows [{n}*r, +{n})");
        }
        Rule::Skip => return Err(format!("cut() on the skipped tensor {}", t.name)),
    }
    c.bytes = c.pieces.iter().map(|p| p.nrows * p.ncols).sum();
    Ok(c)
}

/// Block geometry of a quantised weight, in *logical* elements: FP8 weights carry
/// one ue8m0 per 32x32 block, MXFP4 experts one per row and 32 elements along K.
/// A packed expert stores two values a byte, so its columns — which the table and
/// the pieces count in bytes — are doubled before the block arithmetic.
fn block_of(t: &SrcTensor) -> Result<(u64, u64, u64), String> {
    match t.dtype {
        DType::F8E4M3 => Ok((32, 32, 1)),
        DType::E2M1X2 => Ok((1, 32, 2)),
        _ => Err(format!("{} has a scale but dtype {}", t.name, t.dtype)),
    }
}

/// A scale must be cut exactly along the weight's blocks: same pieces, divided by
/// the block size, and its own shape must be the weight's in blocks.
fn check_scale_pair(ck: &Checkpoint, w: &OutTensor, s: &OutTensor) -> Result<(), String> {
    if w.pieces.len() != s.pieces.len() {
        return Err(format!("{} and its scale have different piece counts", w.name));
    }
    for (pw, ps) in w.pieces.iter().zip(&s.pieces) {
        let wt = &ck.tensors[pw.src];
        let st = &ck.tensors[ps.src];
        if st.dtype != DType::F8E8M0 {
            return Err(format!("{}: scale dtype {}", st.name, st.dtype));
        }
        let (br, bc, epb) = block_of(wt)?;
        let rows = wt.shape.first().copied().unwrap_or(1);
        let cols = wt.shape.get(1).copied().unwrap_or(1) * epb;
        if st.shape.len() != 2 || st.shape[0] != rows.div_ceil(br) || st.shape[1] != cols.div_ceil(bc) {
            return Err(format!("{}: scale shape {:?} does not match weight {:?}", st.name, st.shape, wt.shape));
        }
        let (c0, nc) = (pw.col0 * epb, pw.ncols * epb);
        let ok = pw.row0 % br == 0
            && pw.nrows % br == 0
            && c0 % bc == 0
            && nc % bc == 0
            && ps.row0 == pw.row0 / br
            && ps.nrows == pw.nrows / br
            && ps.col0 == c0 / bc
            && ps.ncols == nc / bc;
        if !ok {
            return Err(format!(
                "{}: slice rows [{}, +{}) cols [{c0}, +{nc}) cuts through its {br}x{bc} scale blocks",
                w.name, pw.row0, pw.nrows
            ));
        }
    }
    Ok(())
}

/// Builds rank `rank`'s plan and lays it out. Fails on a tensor the table does
/// not know, a dimension that does not divide, a slice that cuts through a scale
/// block, or a shape that does not fit its tiled layout.
pub fn make_plan(ck: &Checkpoint, rank: u32, world: u32) -> Result<Plan, String> {
    if world < 1 || rank >= world {
        return Err(format!("bad rank {rank} / world {world}"));
    }
    let mut plan = Plan {
        rank,
        world,
        tensors: Vec::new(),
        skipped: Vec::new(),
        toc_offset: HEADER_BYTES as u64,
        data_offset: 0,
        file_bytes: 0,
    };

    struct Group {
        rule: Rule,
        layout: Layout,
        norm: String,
        srcs: Vec<(i64, usize)>, // (expert index or -1, tensor index)
    }
    let mut groups: BTreeMap<String, Group> = BTreeMap::new();
    let mut unknown = Vec::new();
    for (idx, t) in ck.tensors.iter().enumerate() {
        if t.name.starts_with("vision.") || t.name.starts_with("aligner.") {
            plan.skipped.push(t.name.clone());
            continue;
        }
        let norm = normalize(&t.name);
        let Some(rule) = rules().get(&norm) else {
            unknown.push(t.name.clone());
            continue;
        };
        if rule.rule == Rule::Skip {
            plan.skipped.push(t.name.clone());
            continue;
        }
        let mut out = t.name.clone();
        let mut expert = -1i64;
        if let Some(pos) = t.name.find(".ffn.experts.") {
            let a = pos + ".ffn.experts.".len();
            let b = t.name[a..].find('.').map(|x| a + x).ok_or_else(|| format!("{}: bad expert name", t.name))?;
            expert = t.name[a..b].parse::<i64>().map_err(|_| format!("{}: bad expert index", t.name))?;
            out = format!("{}{}", &t.name[..a], &t.name[b + 1..]);
        }
        let g = groups.entry(out).or_insert_with(|| Group { rule: rule.rule, layout: rule.layout, norm: norm.clone(), srcs: Vec::new() });
        g.srcs.push((expert, idx));
    }
    if !unknown.is_empty() {
        unknown.sort();
        return Err(format!("{} checkpoint tensors have no rule: {}", unknown.len(), unknown.join(", ")));
    }
    plan.skipped.sort();

    for (out, g) in &mut groups {
        g.srcs.sort();
        let first_idx = g.srcs[0].1;
        let first = &ck.tensors[first_idx];
        let dtype = dtype_of(first)?;
        let src_shape = first.shape.clone();
        let stacked = g.srcs[0].0 >= 0;
        if stacked {
            for (e, (expert, idx)) in g.srcs.iter().enumerate() {
                if *expert != e as i64 {
                    return Err(format!("{out}: expert {e} missing"));
                }
                let t = &ck.tensors[*idx];
                if t.shape != first.shape || t.dtype != first.dtype {
                    return Err(format!("{out}: experts differ in shape or dtype"));
                }
            }
        } else if g.srcs.len() != 1 {
            return Err(format!("{out}: {} sources", g.srcs.len()));
        }
        let nstack = if stacked { g.srcs.len() as u32 } else { 0 };
        let source = if stacked {
            let a = out.find(".ffn.experts.").unwrap() + ".ffn.experts.".len();
            format!("{}{{e}}.{}", &out[..a], &out[a..])
        } else {
            first.name.clone()
        };

        let mut o = OutTensor {
            name: out.clone(),
            source,
            slice: String::new(),
            component: component_of(&g.norm),
            dtype,
            kind: SliceKind::Replicated,
            layout: g.layout,
            nstack,
            nseg: 0,
            shape: Vec::new(),
            src_shape,
            bytes: 0,
            offset: 0,
            pieces: Vec::new(),
        };
        let mut per = 0u64;
        for (e, (_expert, idx)) in g.srcs.iter().enumerate() {
            let c = cut(&ck.tensors[*idx], *idx, g.rule, rank, world, per * e as u64)?;
            if e == 0 {
                per = c.bytes;
                o.shape = c.shape.clone();
                o.kind = c.kind;
                o.nseg = c.nseg;
                o.slice = if stacked { format!("{nstack} experts, {}", c.desc) } else { c.desc.clone() };
            }
            o.pieces.extend(c.pieces);
            o.bytes += c.bytes;
        }
        if stacked {
            o.shape.insert(0, nstack as u64);
        }
        let elems: u64 = o.shape.iter().product();
        if elems * o.dtype.bytes() != o.bytes {
            return Err(format!("{out}: shape {:?} does not match {} bytes", o.shape, o.bytes));
        }
        if o.shape.len() > 4 {
            return Err(format!("{out}: more than 4 dims"));
        }
        if o.layout != Layout::Natural {
            tile_geom(o.layout, o.dtype, &o.shape).map_err(|why| {
                format!("{out}: shape {:?} does not fit layout {}: {why}", o.shape, o.layout.name())
            })?;
        }
        plan.tensors.push(o);
    }

    // Every scale is cut along its weight's blocks.
    let by_name: HashMap<&str, usize> =
        plan.tensors.iter().enumerate().map(|(i, o)| (o.name.as_str(), i)).collect();
    for i in 0..plan.tensors.len() {
        if !plan.tensors[i].name.ends_with(".scale") {
            continue;
        }
        let wname = format!("{}.weight", &plan.tensors[i].name[..plan.tensors[i].name.len() - 6]);
        let wi = *by_name.get(wname.as_str()).ok_or_else(|| format!("{} has no weight", plan.tensors[i].name))?;
        let (w, s) = (&plan.tensors[wi], &plan.tensors[i]);
        // The weight's source tensor decides the block size and the scale's must
        // be that shape in blocks; both are the checkpoint's, not the slice's.
        check_scale_pair(ck, w, s)?;
    }

    plan.tensors.sort_by(|a, b| {
        let (ga, gb) = (group_of(&a.name), group_of(&b.name));
        ga.cmp(&gb).then_with(|| a.name.cmp(&b.name))
    });

    plan.toc_offset = HEADER_BYTES as u64;
    plan.data_offset = align_up(plan.toc_offset + plan.tensors.len() as u64 * TOC_ENTRY_BYTES as u64, ALIGN);
    let mut off = plan.data_offset;
    for o in &mut plan.tensors {
        o.offset = off;
        off += align_up(o.bytes, ALIGN);
    }
    plan.file_bytes = off;
    Ok(plan)
}

/// Cross-rank check: for every source tensor, replicated ones are whole on every
/// rank, split ones are cut into disjoint pieces that together cover it exactly.
pub fn check_coverage(ck: &Checkpoint, world: u32) -> Result<(), String> {
    struct Rect {
        rank: u32,
        r0: u64,
        r1: u64,
        c0: u64,
        c1: u64,
    }
    let mut cover: HashMap<usize, Vec<Rect>> = HashMap::new();
    let mut kind: HashMap<usize, SliceKind> = HashMap::new();
    let mut skipped = 0usize;
    for r in 0..world {
        let p = make_plan(ck, r, world)?;
        skipped = p.skipped.len();
        for o in &p.tensors {
            for pc in &o.pieces {
                cover.entry(pc.src).or_default().push(Rect {
                    rank: r,
                    r0: pc.row0,
                    r1: pc.row0 + pc.nrows,
                    c0: pc.col0,
                    c1: pc.col0 + pc.ncols,
                });
                kind.insert(pc.src, o.kind);
            }
        }
    }
    if cover.len() + skipped != ck.tensors.len() {
        return Err(format!(
            "coverage: {} tensors placed + {skipped} skipped != {} in the checkpoint",
            cover.len(),
            ck.tensors.len()
        ));
    }
    for (src, rects) in &cover {
        let t = &ck.tensors[*src];
        let v = view_of(t);
        if kind[src] == SliceKind::Replicated {
            if rects.len() != world as usize {
                return Err(format!("coverage: {} replicated {} times", t.name, rects.len()));
            }
            for x in rects {
                if x.r0 != 0 || x.c0 != 0 || x.r1 != v.rows || x.c1 != v.row_bytes {
                    return Err(format!("coverage: {} not whole", t.name));
                }
            }
            continue;
        }
        let mut area = 0u64;
        for (i, a) in rects.iter().enumerate() {
            if a.r1 > v.rows || a.c1 > v.row_bytes {
                return Err(format!("coverage: {} piece out of bounds", t.name));
            }
            area += (a.r1 - a.r0) * (a.c1 - a.c0);
            for b in &rects[i + 1..] {
                if a.r0 < b.r1 && b.r0 < a.r1 && a.c0 < b.c1 && b.c0 < a.c1 {
                    return Err(format!("coverage: {} pieces of ranks {} and {} overlap", t.name, a.rank, b.rank));
                }
            }
        }
        if area != v.rows * v.row_bytes {
            return Err(format!("coverage: {} covered {area} of {} bytes", t.name, v.rows * v.row_bytes));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::safetensors::tests::write_shard;

    fn ck(tensors: &[(&str, &str, &[u64])]) -> (std::path::PathBuf, Checkpoint) {
        static N: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let n = N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let dir = std::env::temp_dir().join(format!("dsw-plan-{}-{}", std::process::id(), n));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let mut with_bytes = Vec::new();
        for (name, dtype, shape) in tensors {
            let elems: u64 = shape.iter().product();
            let bytes = vec![0u8; (elems * DType::from_safetensors(dtype).unwrap().bytes()) as usize];
            with_bytes.push((*name, *dtype, *shape, bytes));
        }
        let refs: Vec<(&str, &str, &[u64], &[u8])> =
            with_bytes.iter().map(|(n, d, s, b)| (*n, *d, *s, b.as_slice())).collect();
        write_shard(&dir.join("model-00001-of-00001.safetensors"), &refs);
        let ck = Checkpoint::open(&dir).unwrap();
        (dir, ck)
    }

    #[test]
    fn normalizes_numeric_components() {
        assert_eq!(normalize("layers.7.ffn.experts.123.w1.weight"), "layers.N.ffn.experts.N.w1.weight");
        assert_eq!(normalize("embed.weight"), "embed.weight");
        assert_eq!(normalize("mtp.0.main_proj.weight"), "mtp.N.main_proj.weight");
    }

    #[test]
    fn rows_split_disjointly_and_cover_the_source() {
        let (dir, ck) = ck(&[("layers.0.attn.wq_b.weight", "F8_E4M3", &[2048, 64])]);
        let plans: Vec<Plan> = (0..4).map(|r| make_plan(&ck, r, 4).unwrap()).collect();
        for (r, p) in plans.iter().enumerate() {
            let o = &p.tensors[0];
            assert_eq!(o.kind, SliceKind::Rows);
            assert_eq!(o.shape, vec![512, 64]);
            assert_eq!(o.pieces[0].row0, 512 * r as u64);
            assert_eq!(o.pieces[0].nrows, 512);
            assert_eq!(o.layout, Layout::L1);
        }
        check_coverage(&ck, 4).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn replicated_tensors_are_whole_on_every_rank() {
        let (dir, ck) = ck(&[("layers.0.attn_norm.weight", "BF16", &[64])]);
        for r in 0..4 {
            let p = make_plan(&ck, r, 4).unwrap();
            assert_eq!(p.tensors[0].kind, SliceKind::Replicated);
            assert_eq!(p.tensors[0].bytes, 128);
            assert_eq!(p.tensors[0].shape, vec![64]);
        }
        check_coverage(&ck, 4).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn experts_are_stacked_and_ordered() {
        let (dir, ck) = ck(&[
            ("layers.0.ffn.experts.2.w1.weight", "I8", &[64, 64]),
            ("layers.0.ffn.experts.0.w1.weight", "I8", &[64, 64]),
            ("layers.0.ffn.experts.1.w1.weight", "I8", &[64, 64]),
        ]);
        let p = make_plan(&ck, 1, 4).unwrap();
        let o = p.tensors.iter().find(|t| t.name == "layers.0.ffn.experts.w1.weight").unwrap();
        assert_eq!(o.nstack, 3);
        assert_eq!(o.shape, vec![3, 16, 64]); // 64 rows / 4 ranks, three experts
        assert_eq!(o.source, "layers.0.ffn.experts.{e}.w1.weight");
        assert_eq!(o.pieces.len(), 3);
        assert_eq!(o.pieces[1].dst, 16 * 64); // one expert's bytes: 16 rows of 64 bytes
        // expert e is source e, in order
        for (e, pc) in o.pieces.iter().enumerate() {
            assert_eq!(ck.tensors[pc.src].name, format!("layers.0.ffn.experts.{e}.w1.weight"));
        }
        check_coverage(&ck, 4).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn rows_seg_splits_each_segment() {
        let (dir, ck) = ck(&[("layers.0.engram.wkv.weight", "F8_E4M3", &[320, 16])]);
        let p = make_plan(&ck, 0, 4).unwrap();
        let o = &p.tensors[0];
        assert_eq!(o.kind, SliceKind::RowsSeg);
        assert_eq!(o.nseg, 5);
        assert_eq!(o.pieces.len(), 5);
        assert_eq!(o.shape, vec![80, 16]); // 320/5 = 64 a segment, /4 = 16 a rank
        for (s, pc) in o.pieces.iter().enumerate() {
            assert_eq!(pc.row0, s as u64 * 64);
            assert_eq!(pc.nrows, 16);
            assert_eq!(pc.dst, s as u64 * 16 * 16);
        }
        check_coverage(&ck, 4).unwrap();
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn offsets_are_aligned_and_the_file_is_consistent() {
        let (dir, ck) = ck(&[
            ("embed.weight", "BF16", &[8, 4]),
            ("layers.0.attn_norm.weight", "BF16", &[8]),
            ("layers.0.attn.wq_b.weight", "F8_E4M3", &[64, 64]),
        ]);
        let p = make_plan(&ck, 0, 4).unwrap();
        assert_eq!(p.toc_offset, 4096);
        assert_eq!(p.data_offset % 4096, 0);
        let mut expect = p.data_offset;
        for t in &p.tensors {
            assert_eq!(t.offset, expect);
            assert_eq!(t.offset % 4096, 0);
            expect += align_up(t.bytes, 4096);
        }
        assert_eq!(p.file_bytes, expect);
        // globals first, then layer 0
        assert_eq!(p.tensors.first().unwrap().name, "embed.weight");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_a_split_that_does_not_divide() {
        let (dir, ck) = ck(&[("layers.0.attn.wq_b.weight", "F8_E4M3", &[1002, 64])]);
        let err = make_plan(&ck, 0, 4).unwrap_err();
        assert!(err.contains("do not split"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn refuses_a_tensor_with_no_rule() {
        let (dir, ck) = ck(&[("layers.0.mystery.weight", "BF16", &[8])]);
        let err = make_plan(&ck, 0, 4).unwrap_err();
        assert!(err.contains("no rule"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn skips_vision_and_engram_tables() {
        let (dir, ck) = ck(&[
            ("vision.patch_embed.weight", "BF16", &[8]),
            ("layers.1.engram.embed.weight", "F8_E4M3", &[64, 256]),
            ("layers.0.attn_norm.weight", "BF16", &[8]),
        ]);
        let p = make_plan(&ck, 0, 4).unwrap();
        assert_eq!(p.skipped.len(), 2);
        assert_eq!(p.tensors.len(), 1);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn mxfp4_experts_cut_along_their_scale_blocks() {
        // 32 values a scale, two values a byte: a 128-byte row is 256 values, 8
        // scales. The columns are counted in bytes everywhere else, so the check
        // is the one place the factor of two must appear.
        let experts = |e: u32| {
            vec![
                (format!("layers.0.ffn.experts.{e}.w1.weight"), "I8".to_string(), vec![64u64, 128]),
                (format!("layers.0.ffn.experts.{e}.w1.scale"), "F8_E8M0".to_string(), vec![64, 8]),
                (format!("layers.0.ffn.experts.{e}.w2.weight"), "I8".to_string(), vec![64, 128]),
                (format!("layers.0.ffn.experts.{e}.w2.scale"), "F8_E8M0".to_string(), vec![64, 8]),
            ]
        };
        let owned: Vec<(String, String, Vec<u64>)> = experts(0).into_iter().chain(experts(1)).collect();
        let refs: Vec<(&str, &str, &[u64])> =
            owned.iter().map(|(n, d, s)| (n.as_str(), d.as_str(), s.as_slice())).collect();
        let (dir, ck) = ck(&refs);
        for world in [4u32, 2] {
            for r in 0..world {
                let p = make_plan(&ck, r, world).unwrap_or_else(|e| panic!("world {world} rank {r}: {e}"));
                let get = |n: &str| p.tensors.iter().find(|t| t.name == n).unwrap();
                let (w1, s1) = (get("layers.0.ffn.experts.w1.weight"), get("layers.0.ffn.experts.w1.scale"));
                // Rows: the rows split, the columns (and so the scales) stay whole.
                assert_eq!(w1.pieces[0].nrows, 64 / world as u64);
                assert_eq!(w1.pieces[0].ncols, 128);
                assert_eq!(s1.pieces[0].ncols, 8);
                // Cols: the scale follows the weight's bytes, two values per byte.
                let (w2, s2) = (get("layers.0.ffn.experts.w2.weight"), get("layers.0.ffn.experts.w2.scale"));
                assert_eq!(w2.pieces[0].ncols, 128 / world as u64);
                assert_eq!(s2.pieces[0].ncols, 8 / world as u64);
            }
        }
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn a_scale_of_the_wrong_shape_is_refused() {
        let (dir, ck) = ck(&[
            ("layers.0.ffn.experts.0.w1.weight", "I8", &[64, 128]),
            ("layers.0.ffn.experts.0.w1.scale", "F8_E8M0", &[64, 4]),
        ]);
        let err = make_plan(&ck, 0, 4).unwrap_err();
        assert!(err.contains("does not match weight"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
