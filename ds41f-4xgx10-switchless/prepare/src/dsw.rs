//! The per-rank weight file format, transcribed from `engine/weights/dsw.h`.
//!
//! A `.dsw` is a 4 KiB header, a table of contents (512 B per tensor) and the
//! tensor bytes, each starting on a 4 KiB boundary. Values are the checkpoint's;
//! the only transformations are slicing (which rows/columns a rank owns) and the
//! tiled layouts the decode kernels read.

use std::fmt;

pub const MAGIC: [u8; 8] = *b"DSGB10W1";
pub const VERSION: u32 = 2;
pub const MIN_VERSION: u32 = 1;
pub const ALIGN: u64 = 4096;
pub const HEADER_BYTES: usize = 4096;
pub const TOC_ENTRY_BYTES: usize = 512;

pub fn align_up(x: u64, a: u64) -> u64 {
    debug_assert!(a.is_power_of_two());
    (x + a - 1) & !(a - 1)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum DType {
    F32 = 1,
    BF16 = 2,
    F8E4M3 = 3,
    F8E8M0 = 4,
    E2M1X2 = 5,
}

impl DType {
    pub fn from_u32(v: u32) -> Option<DType> {
        Some(match v {
            1 => DType::F32,
            2 => DType::BF16,
            3 => DType::F8E4M3,
            4 => DType::F8E8M0,
            5 => DType::E2M1X2,
            _ => return None,
        })
    }

    /// Bytes per element (the packed FP4 counts two values per byte, so callers
    /// that work in elements must halve the last dimension themselves).
    pub fn bytes(self) -> u64 {
        match self {
            DType::F32 => 4,
            DType::BF16 => 2,
            DType::F8E4M3 | DType::F8E8M0 => 1,
            DType::E2M1X2 => 1,
        }
    }

    /// The safetensors dtype string (`I8` is how the checkpoint names packed FP4).
    pub fn from_safetensors(s: &str) -> Option<DType> {
        Some(match s {
            "F32" => DType::F32,
            "BF16" => DType::BF16,
            "F8_E4M3" => DType::F8E4M3,
            "F8_E8M0" => DType::F8E8M0,
            "I8" => DType::E2M1X2,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            DType::F32 => "F32",
            DType::BF16 => "BF16",
            DType::F8E4M3 => "F8_E4M3",
            DType::F8E8M0 => "F8_E8M0",
            DType::E2M1X2 => "E2M1X2",
        }
    }
}

/// How a rank's tensor was cut from its source, each viewed as `[rows, row_bytes]`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum SliceKind {
    Replicated = 1,
    Rows = 2,
    Cols = 3,
    RowsSeg = 4,
}

impl SliceKind {
    pub fn from_u32(v: u32) -> Option<SliceKind> {
        Some(match v {
            1 => SliceKind::Replicated,
            2 => SliceKind::Rows,
            3 => SliceKind::Cols,
            4 => SliceKind::RowsSeg,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            SliceKind::Replicated => "replicated",
            SliceKind::Rows => "rows",
            SliceKind::Cols => "cols",
            SliceKind::RowsSeg => "rows-seg",
        }
    }
}

/// How a tensor's bytes are arranged. `shape` stays the logical shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum Layout {
    Natural = 0,
    /// FP8 e4m3 `[N][K]`: `[N/8][K/64][8 rows][64 B]`.
    L1 = 1,
    /// bf16 `[N][K]` viewed as bytes `[N][2K]`, then L1.
    L2 = 2,
    /// MXFP4 `(tile_fp4)`: values and their ue8m0 scales, matrix by matrix.
    L3 = 3,
}

impl Layout {
    pub fn from_u32(v: u32) -> Option<Layout> {
        Some(match v {
            0 => Layout::Natural,
            1 => Layout::L1,
            2 => Layout::L2,
            3 => Layout::L3,
            _ => return None,
        })
    }

    pub fn name(self) -> &'static str {
        match self {
            Layout::Natural => "natural",
            Layout::L1 => "L1",
            Layout::L2 => "L2",
            Layout::L3 => "L3",
        }
    }
}

/// `[0, 4096)` of a `.dsw`.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct Header {
    pub magic: [u8; 8],
    pub version: u32,
    pub header_bytes: u32,
    pub rank: u32,
    pub world: u32,
    pub n_tensors: u32,
    pub toc_entry_bytes: u32,
    pub toc_offset: u64,
    pub data_offset: u64,
    pub data_bytes: u64,
    pub file_bytes: u64,
    pub alignment: u32,
    pub toc_crc32c: u32,
    pub model: [u8; 64],
    pub source: [u8; 256],
    pub created: [u8; 32],
    pub reserved: [u8; HEADER_BYTES - 424],
}

const _: () = assert!(std::mem::size_of::<Header>() == HEADER_BYTES);

/// One tensor of the table of contents.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TocEntry {
    pub name: [u8; 128],
    pub source: [u8; 128],
    pub slice: [u8; 96],
    pub dtype: u32,
    pub ndim: u32,
    pub shape: [u64; 4],
    pub offset: u64,
    pub bytes: u64,
    pub kind: u32,
    pub nstack: u32,
    pub nseg: u32,
    pub crc32c: u32,
    pub src_ndim: u32,
    pub layout: u32,
    pub src_shape: [u64; 4],
    pub reserved: [u8; 48],
}

const _: () = assert!(std::mem::size_of::<TocEntry>() == TOC_ENTRY_BYTES);

fn fixed<const N: usize>(s: &str, what: &str) -> [u8; N] {
    assert!(s.len() < N, "{what} does not fit in {N} bytes: {s}");
    let mut out = [0u8; N];
    out[..s.len()].copy_from_slice(s.as_bytes());
    out
}

fn text(b: &[u8]) -> String {
    let end = b.iter().position(|&c| c == 0).unwrap_or(b.len());
    String::from_utf8_lossy(&b[..end]).into_owned()
}

impl Header {
    pub fn new(rank: u32, world: u32, model: &str, source: &str, created: &str) -> Header {
        Header {
            magic: MAGIC,
            version: VERSION,
            header_bytes: HEADER_BYTES as u32,
            rank,
            world,
            n_tensors: 0,
            toc_entry_bytes: TOC_ENTRY_BYTES as u32,
            toc_offset: 0,
            data_offset: 0,
            data_bytes: 0,
            file_bytes: 0,
            alignment: ALIGN as u32,
            toc_crc32c: 0,
            model: fixed(model, "model"),
            source: fixed(source, "source"),
            created: fixed(created, "created"),
            reserved: [0; HEADER_BYTES - 424],
        }
    }

    pub fn to_bytes(&self) -> [u8; HEADER_BYTES] {
        let mut out = [0u8; HEADER_BYTES];
        // SAFETY: Header is repr(C), fully initialised and exactly HEADER_BYTES.
        unsafe { std::ptr::copy_nonoverlapping(self as *const Header as *const u8, out.as_mut_ptr(), HEADER_BYTES) };
        out
    }

    pub fn from_bytes(b: &[u8; HEADER_BYTES]) -> Header {
        // SAFETY: same size, and every field is a plain integer array.
        unsafe { std::ptr::read_unaligned(b.as_ptr() as *const Header) }
    }

    pub fn model_name(&self) -> String {
        text(&self.model)
    }

    /// "" when the file was cut for `rank` of `world`, else why it was not.
    pub fn rank_mismatch(&self, rank: u32, world: u32) -> String {
        if self.rank == rank && self.world == world {
            String::new()
        } else {
            format!(
                "the file was cut for rank {} of {}, but this process is rank {} of {}",
                self.rank, self.world, rank, world
            )
        }
    }
}

impl TocEntry {
    pub fn new(
        name: &str,
        source: &str,
        slice: &str,
        dtype: DType,
        shape: &[u64],
        offset: u64,
        bytes: u64,
        kind: SliceKind,
        nstack: u32,
        nseg: u32,
        layout: Layout,
        src_shape: &[u64],
    ) -> TocEntry {
        assert!(shape.len() <= 4, "{name}: more than 4 dims");
        assert!(src_shape.len() <= 4, "{name}: more than 4 source dims");
        let mut sh = [0u64; 4];
        sh[..shape.len()].copy_from_slice(shape);
        let mut ssh = [0u64; 4];
        ssh[..src_shape.len()].copy_from_slice(src_shape);
        TocEntry {
            name: fixed(name, "tensor name"),
            source: fixed(source, "source name"),
            slice: fixed(slice, "slice description"),
            dtype: dtype as u32,
            ndim: shape.len() as u32,
            shape: sh,
            offset,
            bytes,
            kind: kind as u32,
            nstack,
            nseg,
            crc32c: 0,
            src_ndim: src_shape.len() as u32,
            layout: layout as u32,
            src_shape: ssh,
            reserved: [0; 48],
        }
    }

    pub fn name(&self) -> String {
        text(&self.name)
    }

    pub fn source_name(&self) -> String {
        text(&self.source)
    }

    pub fn slice_desc(&self) -> String {
        text(&self.slice)
    }

    pub fn shape(&self) -> &[u64] {
        &self.shape[..self.ndim as usize]
    }

    pub fn src_shape(&self) -> &[u64] {
        &self.src_shape[..self.src_ndim as usize]
    }

    pub fn dtype(&self) -> DType {
        DType::from_u32(self.dtype).expect("bad dtype in TOC")
    }

    pub fn kind(&self) -> SliceKind {
        SliceKind::from_u32(self.kind).expect("bad slice kind in TOC")
    }

    pub fn layout(&self) -> Layout {
        Layout::from_u32(self.layout).expect("bad layout in TOC")
    }

    pub fn to_bytes(&self) -> [u8; TOC_ENTRY_BYTES] {
        let mut out = [0u8; TOC_ENTRY_BYTES];
        // SAFETY: repr(C), fully initialised, exactly TOC_ENTRY_BYTES.
        unsafe {
            std::ptr::copy_nonoverlapping(self as *const TocEntry as *const u8, out.as_mut_ptr(), TOC_ENTRY_BYTES)
        };
        out
    }

    pub fn from_bytes(b: &[u8]) -> TocEntry {
        assert_eq!(b.len(), TOC_ENTRY_BYTES);
        // SAFETY: same size, plain integers.
        unsafe { std::ptr::read_unaligned(b.as_ptr() as *const TocEntry) }
    }
}

/// CRC32C (Castagnoli), the checksum the format uses. Software here: the
/// converter's ARM instructions and this table must agree, and a unit test
/// pins the standard vector.
pub fn crc32c(data: &[u8]) -> u32 {
    crc32c_update(0, data)
}

pub fn crc32c_update(mut crc: u32, data: &[u8]) -> u32 {
    static TABLE: std::sync::LazyLock<[u32; 256]> = std::sync::LazyLock::new(|| {
        let mut t = [0u32; 256];
        for (i, slot) in t.iter_mut().enumerate() {
            let mut c = i as u32;
            for _ in 0..8 {
                c = if c & 1 != 0 { (c >> 1) ^ 0x82F6_3B78 } else { c >> 1 };
            }
            *slot = c;
        }
        t
    });
    let table = &*TABLE;
    crc = !crc;
    for &b in data {
        crc = table[((crc ^ b as u32) & 0xFF) as usize] ^ (crc >> 8);
    }
    !crc
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.name())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crc32c_matches_the_standard_vector() {
        // RFC 3720 / iSCSI: the Castagnoli check value of "123456789".
        assert_eq!(crc32c(b"123456789"), 0xE306_9283);
        assert_eq!(crc32c(b""), 0);
        // Chainable, as io.h documents.
        let all = crc32c(b"123456789");
        assert_eq!(crc32c_update(crc32c(b"1234"), b"56789"), all);
    }

    #[test]
    fn header_round_trips_at_the_format_size() {
        let h = Header::new(2, 4, "deepseek-v4.1-flash", "/data/ckpt", "2026-11-12T00:00:00Z");
        let b = h.to_bytes();
        assert_eq!(b.len(), 4096);
        assert_eq!(&b[..8], b"DSGB10W1");
        let back = Header::from_bytes(&b);
        assert_eq!(back.rank, 2);
        assert_eq!(back.world, 4);
        assert_eq!(back.header_bytes, 4096);
        assert_eq!(back.toc_entry_bytes, 512);
        assert_eq!(back.model_name(), "deepseek-v4.1-flash");
        assert_eq!(back.rank_mismatch(2, 4), "");
        assert!(back.rank_mismatch(3, 4).contains("rank 3 of 4"));
    }

    #[test]
    fn toc_entry_round_trips_at_the_format_size() {
        let e = TocEntry::new(
            "layers.7.attn.wq_b.weight",
            "layers.7.attn.wq_b.weight",
            "rows [5120*r, +5120) of 20480",
            DType::F8E4M3,
            &[5120, 4096],
            8192,
            20_971_520,
            SliceKind::Rows,
            0,
            0,
            Layout::L1,
            &[20480, 4096],
        );
        let b = e.to_bytes();
        assert_eq!(b.len(), 512);
        let back = TocEntry::from_bytes(&b);
        assert_eq!(back.name(), "layers.7.attn.wq_b.weight");
        assert_eq!(back.shape(), &[5120, 4096]);
        assert_eq!(back.src_shape(), &[20480, 4096]);
        assert_eq!(back.dtype(), DType::F8E4M3);
        assert_eq!(back.layout(), Layout::L1);
        assert_eq!(back.kind(), SliceKind::Rows);
        assert_eq!(back.bytes, 20_971_520);
    }

    #[test]
    fn sizes_and_alignment_are_the_formats() {
        assert_eq!(align_up(0, 4096), 0);
        assert_eq!(align_up(1, 4096), 4096);
        assert_eq!(align_up(4096, 4096), 4096);
        assert_eq!(DType::E2M1X2.bytes(), 1);
        assert_eq!(DType::BF16.bytes(), 2);
        assert_eq!(DType::from_safetensors("I8"), Some(DType::E2M1X2));
    }
}
