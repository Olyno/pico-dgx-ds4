//! The tiled byte layouts, transcribed from `engine/weights/layout.h`.
//!
//! `layout.h` is the host reference the verifier uses; the converter writes with
//! the kernels' own `dsk::tile_fp8`/`dsk::tile_fp4`. This module is a third
//! implementation, which is the point: a byte-for-byte comparison between two
//! independent implementations is what catches a wrong permutation, and the
//! round-trip test below pins that the mapping is a permutation at all.

use crate::dsw::{DType, Layout};

/// A tiled tensor as the tiling functions see it: `rows` rows over the whole
/// stack, each matrix of the stack `mat_rows` of them; `k` is the row length in
/// bytes for L1/L2 and the logical K (elements) for L3.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TileGeom {
    pub rows: u64,
    pub k: u64,
    pub mat_rows: u64,
}

/// The geometry of a tensor of dtype `dt` and natural shape `shape` in layout
/// `l`, or why the shape does not fit its tiles.
pub fn tile_geom(l: Layout, dt: DType, shape: &[u64]) -> Result<TileGeom, String> {
    if shape.len() < 2 || shape.len() > 4 {
        return Err("needs 2 to 4 dims".into());
    }
    let lead: u64 = shape[..shape.len() - 1].iter().product();
    let last = shape[shape.len() - 1];
    match l {
        Layout::L1 => {
            if dt != DType::F8E4M3 || shape.len() != 2 {
                return Err("L1 is for 2-d e4m3 weights".into());
            }
            let g = TileGeom { rows: shape[0], k: last, mat_rows: shape[0] };
            if g.rows % 8 != 0 || g.k % 64 != 0 {
                return Err("L1 needs N % 8 == 0 and K % 64 == 0".into());
            }
            Ok(g)
        }
        Layout::L2 => {
            if dt != DType::BF16 || shape.len() != 2 {
                return Err("L2 is for 2-d bf16 weights".into());
            }
            let g = TileGeom { rows: shape[0], k: 2 * last, mat_rows: shape[0] };
            if g.rows % 8 != 0 || g.k % 64 != 0 {
                return Err("L2 needs N % 8 == 0 and 2K % 64 == 0".into());
            }
            Ok(g)
        }
        Layout::L3 => {
            let (k, mat_rows) = match dt {
                DType::E2M1X2 => (2 * last, shape[shape.len() - 2]),
                DType::F8E8M0 => (32 * last, shape[shape.len() - 2]),
                _ => return Err("L3 is for packed e2m1 values and their ue8m0 scales".into()),
            };
            let g = TileGeom { rows: lead, k, mat_rows };
            if g.mat_rows % 16 != 0 || g.k % 64 != 0 {
                return Err("L3 needs N % 16 == 0 and K % 64 == 0".into());
            }
            Ok(g)
        }
        Layout::Natural => Err("natural is not tiled".into()),
    }
}

/// L1: natural `[N][K]` bytes <-> `[N/8][K/64][8 rows][64 B]`.
pub fn l1_offset(n: u64, k: u64, big_k: u64) -> u64 {
    ((n / 8) * (big_k / 64) + k / 64) * 512 + (n % 8) * 64 + k % 64
}

fn l1(src: &[u8], dst: &mut [u8], rows: u64, k: u64, untile: bool) {
    for n in 0..rows {
        for kk in (0..k).step_by(64) {
            let nat = (n * k + kk) as usize;
            let til = l1_offset(n, kk, k) as usize;
            if untile {
                dst[nat..nat + 64].copy_from_slice(&src[til..til + 64]);
            } else {
                dst[til..til + 64].copy_from_slice(&src[nat..nat + 64]);
            }
        }
    }
}

/// L3 values of one matrix: natural `[N][K/2]` bytes.
fn l3_value_offset(n: u64, p: u64, l: u64, t: u64, big_p: u64) -> u64 {
    let n16 = n / 16;
    let h = (n % 16) / 8;
    let g = n % 8;
    ((n16 * big_p + p) * 32 + 4 * g + t) * 16 + 4 * (2 * h + l)
}

/// L3 scales of one matrix: natural `[N][K/32]`.
fn l3_scale_offset(n: u64, p: u64, l: u64, big_p: u64) -> u64 {
    let n16 = n / 16;
    let h = (n % 16) / 8;
    let g = n % 8;
    ((n16 * big_p + p) * 8 + g) * 4 + 2 * h + l
}

/// One L3 matrix (values when `scales` is false), natural <-> tiled.
pub fn l3_matrix(src: &[u8], dst: &mut [u8], n_rows: u64, k: u64, scales: bool, untile: bool) {
    let p_count = k / 64;
    if scales {
        let rb = k / 32;
        for n in 0..n_rows {
            for p in 0..p_count {
                for l in 0..2 {
                    let nat = (n * rb + 2 * p + l) as usize;
                    let til = l3_scale_offset(n, p, l, p_count) as usize;
                    if untile {
                        dst[nat] = src[til];
                    } else {
                        dst[til] = src[nat];
                    }
                }
            }
        }
        return;
    }
    let rb = k / 2;
    for n in 0..n_rows {
        for p in 0..p_count {
            for l in 0..2 {
                for t in 0..4 {
                    let nat = (n * rb + 32 * p + 16 * l + 4 * t) as usize;
                    let til = l3_value_offset(n, p, l, t, p_count) as usize;
                    if untile {
                        dst[nat..nat + 4].copy_from_slice(&src[til..til + 4]);
                    } else {
                        dst[til..til + 4].copy_from_slice(&src[nat..nat + 4]);
                    }
                }
            }
        }
    }
}

/// Natural <-> tiled for a whole tensor (a stack is tiled matrix by matrix).
/// `src` is the input, `dst` the output; both are `bytes` long.
pub fn retile(l: Layout, dt: DType, g: TileGeom, src: &[u8], dst: &mut [u8], untile: bool) {
    match l {
        Layout::L1 | Layout::L2 => l1(src, dst, g.rows, g.k, untile),
        Layout::L3 => {
            let scales = dt == DType::F8E8M0;
            let per = g.mat_rows * (if scales { g.k / 32 } else { g.k / 2 });
            let mats = g.rows / g.mat_rows;
            for e in 0..mats {
                let a = (e * per) as usize;
                let b = ((e + 1) * per) as usize;
                l3_matrix(&src[a..b], &mut dst[a..b], g.mat_rows, g.k, scales, untile);
            }
        }
        Layout::Natural => panic!("retile on a natural tensor"),
    }
}

/// The bytes a tensor of this shape and layout occupies (its unpadded size).
pub fn tiled_bytes(l: Layout, dt: DType, shape: &[u64]) -> Result<u64, String> {
    let elems: u64 = shape.iter().product();
    let _ = tile_geom(l, dt, shape)?;
    Ok(match dt {
        DType::E2M1X2 => elems.div_ceil(2),
        _ => elems * dt.bytes(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pattern(n: usize) -> Vec<u8> {
        (0..n).map(|i| (i.wrapping_mul(37) ^ (i >> 3)) as u8).collect()
    }

    #[test]
    fn l1_is_a_permutation() {
        // 16 rows x 128 bytes: 2 row-tiles x 2 k-tiles.
        let (rows, k) = (16u64, 128u64);
        let nat = pattern((rows * k) as usize);
        let mut til = vec![0u8; nat.len()];
        retile(Layout::L1, DType::F8E4M3, TileGeom { rows, k, mat_rows: rows }, &nat, &mut til, false);
        assert_ne!(nat, til, "tiling must actually move bytes");
        // Every byte survives, exactly once.
        let mut seen = til.clone();
        seen.sort_unstable();
        let mut expect = nat.clone();
        expect.sort_unstable();
        assert_eq!(seen, expect);
        // The documented address of (n, k) holds nat[n][k].
        for n in 0..rows {
            for kk in (0..k).step_by(64) {
                let at = l1_offset(n, kk, k) as usize;
                assert_eq!(&til[at..at + 64], &nat[(n * k + kk) as usize..][..64]);
            }
        }
        let mut back = vec![0u8; nat.len()];
        retile(Layout::L1, DType::F8E4M3, TileGeom { rows, k, mat_rows: rows }, &til, &mut back, true);
        assert_eq!(back, nat);
    }

    #[test]
    fn l2_tiles_bf16_rows_as_bytes() {
        let (rows, k) = (8u64, 64u64); // 8 rows, 32 bf16 each -> 64 bytes a row
        let nat = pattern((rows * k) as usize);
        let mut til = vec![0u8; nat.len()];
        retile(Layout::L2, DType::BF16, TileGeom { rows, k, mat_rows: rows }, &nat, &mut til, false);
        let mut back = vec![0u8; nat.len()];
        retile(Layout::L2, DType::BF16, TileGeom { rows, k, mat_rows: rows }, &til, &mut back, true);
        assert_eq!(back, nat);
    }

    #[test]
    fn l3_values_and_scales_are_permutations() {
        // One matrix: 16 rows, K = 128 elements -> 64 value bytes and 4 scale bytes a row.
        let (n_rows, k) = (16u64, 128u64);
        let g = TileGeom { rows: n_rows, k, mat_rows: n_rows };
        let values = pattern((n_rows * (k / 2)) as usize);
        let mut tiled = vec![0u8; values.len()];
        retile(Layout::L3, DType::E2M1X2, g, &values, &mut tiled, false);
        assert_ne!(values, tiled);
        let mut back = vec![0u8; values.len()];
        retile(Layout::L3, DType::E2M1X2, g, &tiled, &mut back, true);
        assert_eq!(back, values);

        let scales = pattern((n_rows * (k / 32)) as usize);
        let mut tiled = vec![0u8; scales.len()];
        retile(Layout::L3, DType::F8E8M0, g, &scales, &mut tiled, false);
        let mut back = vec![0u8; scales.len()];
        retile(Layout::L3, DType::F8E8M0, g, &tiled, &mut back, true);
        assert_eq!(back, scales);
        // The scale at (n, 2p+l) is the documented one.
        for n in 0..n_rows {
            for p in 0..k / 64 {
                for l in 0..2 {
                    let nat = (n * (k / 32) + 2 * p + l) as usize;
                    let til = l3_scale_offset(n, p, l, k / 64) as usize;
                    assert_eq!(tiled[til], scales[nat]);
                }
            }
        }
    }

    #[test]
    fn stacked_matrices_tile_independently() {
        // 3 experts of 16 rows each, K = 64.
        let (mat_rows, k, mats) = (16u64, 64u64, 3u64);
        let g = TileGeom { rows: mat_rows * mats, k, mat_rows };
        let values = pattern((mats * mat_rows * (k / 2)) as usize);
        let mut tiled = vec![0u8; values.len()];
        retile(Layout::L3, DType::E2M1X2, g, &values, &mut tiled, false);
        // Expert 1 starts at its own size, not interleaved with expert 0.
        let per = (mat_rows * (k / 2)) as usize;
        let mut one = vec![0u8; per];
        retile(Layout::L3, DType::E2M1X2, TileGeom { rows: mat_rows, k, mat_rows }, &values[per..2 * per], &mut one, false);
        assert_eq!(one, tiled[per..2 * per]);
        let mut back = vec![0u8; values.len()];
        retile(Layout::L3, DType::E2M1X2, g, &tiled, &mut back, true);
        assert_eq!(back, values);
    }

    #[test]
    fn geometries_reject_shapes_the_kernels_cannot_read() {
        assert!(tile_geom(Layout::L1, DType::F8E4M3, &[7, 64]).is_err()); // N % 8
        assert!(tile_geom(Layout::L1, DType::F8E4M3, &[8, 63]).is_err()); // K % 64
        assert!(tile_geom(Layout::L1, DType::BF16, &[8, 64]).is_err()); // wrong dtype
        assert!(tile_geom(Layout::L3, DType::E2M1X2, &[16, 64]).is_ok());
        assert!(tile_geom(Layout::L3, DType::E2M1X2, &[15, 64]).is_err()); // N % 16
        assert!(tile_geom(Layout::Natural, DType::BF16, &[8, 8]).is_err());
        assert_eq!(tiled_bytes(Layout::L1, DType::F8E4M3, &[8, 64]).unwrap(), 512);
        assert_eq!(tiled_bytes(Layout::L2, DType::BF16, &[8, 32]).unwrap(), 512);
        assert_eq!(tiled_bytes(Layout::L3, DType::E2M1X2, &[16, 64]).unwrap(), 512);
    }
}
