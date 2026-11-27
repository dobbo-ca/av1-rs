//! Reconstruction: dequantization, inverse transform and adding the residual (spec 7.12.3).

#![allow(dead_code)]

use crate::consts::{DCT_DCT, IDTX};
use crate::predict::{EDGE_LEN, Edge};
use crate::quant::{QM_OFFSET, QUANTIZER_MATRIX};
use crate::simd::{self, Simd};
use crate::tables::{TX_HEIGHT_LOG2, TX_WIDTH_LOG2};
use crate::transform::{DQ_LEN, DQ_STRIDE, MID_LEN, Plan, dc_only};

/// Everything `reconstruct` needs besides the pixels and the coefficients.
#[derive(Clone, Copy, Debug)]
pub(crate) struct ReconParams {
    /// `txSz`.
    pub tx_sz: usize,
    /// `PlaneTxType`.
    pub tx_type: usize,
    /// `plane`.
    pub plane: usize,
    /// `get_dc_quant( plane )`, computed with `quant::dc_q`.
    pub dc_q: i32,
    /// `get_ac_quant( plane )`, computed with `quant::ac_q`.
    pub ac_q: i32,
    /// `SegQMLevel[ plane ][ segment_id ]`: `NUM_QM_LEVELS - 1` (15) when lossless or
    /// `using_qmatrix` is 0, else `qm_y`, `qm_u` or `qm_v`.
    pub qm_level: usize,
    /// `Lossless`.
    pub lossless: bool,
    /// `BitDepth`.
    pub bit_depth: u32,
}

/// Working buffers of [`reconstruct`], kept by the tile decoder so that no block zeroes them
/// (every entry is written before it is read).
#[derive(Debug)]
pub(crate) struct ItxScratch {
    /// `Dequant` of the coded region, layout `transform::DQ_STRIDE`.
    dq: [i32; DQ_LEN],
    /// The row pass output.
    mid: [i32; MID_LEN],
    /// The chroma-from-luma AC values of one block (`predict_cfl`), not cleared between uses.
    pub(crate) ac: [i32; 1024],
    /// The `AboveRow` and `LeftCol` of one intra prediction (`predict_intra_simd`), not
    /// cleared between uses.
    pub(crate) edges: [Edge; 2],
}

impl ItxScratch {
    pub(crate) fn new() -> Box<Self> {
        Box::new(Self {
            dq: [0; DQ_LEN],
            mid: [0; MID_LEN],
            ac: [0; 1024],
            edges: [[0; EDGE_LEN]; 2],
        })
    }
}

/// Spec 7.12.3 reconstruct process.
///
/// `quant` is the spec's `Quant[]` with `Quant[ i * tw + j ]`, `tw = Min( 32, w )`,
/// `th = Min( 32, h )`. The block at `(x, y)` of size `w x h` is updated in `px` (row stride
/// `stride`). The caller guarantees `tx_sz < 19`, `tx_type < 16`, `plane < 3`, `qm_level <= 15`,
/// `bit_depth` in 8..=12, `quant.len() >= tw * th` and that the block lies inside `px`; if not,
/// nothing is written.
///
/// Only the coded region (up to the last nonzero row and column of `quant`) is dequantized and
/// row-transformed. A DCT_DCT block with only the DC coefficient takes the flat fast path
/// ([`dc_only`]), which gives exactly the full transform's samples.
pub(crate) fn reconstruct(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    p: &ReconParams,
    quant: &[i32],
    scratch: &mut ItxScratch,
    simd: Simd,
) {
    let (Some(&log2_w), Some(&log2_h)) = (TX_WIDTH_LOG2.get(p.tx_sz), TX_HEIGHT_LOG2.get(p.tx_sz))
    else {
        return;
    };
    let (tw, th) = ((1usize << log2_w).min(32), (1usize << log2_h).min(32));
    if quant.len() < tw * th {
        return;
    }
    // The coded region. An all-zero block still runs as a 1x1 region, so the samples are clipped
    // exactly as the full transform clips them.
    let (mut rows, mut cols) = (1, 1);
    for i in 0..th {
        if let Some(j) = quant[i * tw..i * tw + tw].iter().rposition(|&v| v != 0) {
            rows = i + 1;
            cols = cols.max(j + 1);
        }
    }
    reconstruct_region(px, stride, x, y, p, quant, (rows, cols), scratch, simd);
}

/// [`reconstruct`] with the coded region given: `rows` x `cols` (each at least 1) must hold
/// every nonzero entry of `quant`, as the region `reconstruct` finds does. The coefficient
/// reader tracks it while it writes `Quant[ ]`, so the block is not scanned again.
pub(crate) fn reconstruct_region(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    p: &ReconParams,
    quant: &[i32],
    (rows, cols): (usize, usize),
    scratch: &mut ItxScratch,
    simd: Simd,
) {
    let (Some(&log2_w), Some(&log2_h)) = (TX_WIDTH_LOG2.get(p.tx_sz), TX_HEIGHT_LOG2.get(p.tx_sz))
    else {
        return;
    };
    let (w, h) = (1usize << log2_w, 1usize << log2_h);
    let (tw, th) = (w.min(32), h.min(32));
    if p.tx_type >= 16
        || p.qm_level > 15
        || quant.len() < tw * th
        || (y + h - 1) * stride + x + w > px.len()
    {
        return;
    }
    let Some(plan) = Plan::new(p.tx_sz, p.tx_type, p.lossless, p.bit_depth, rows, cols) else {
        return;
    };

    let dq_shift: u32 = match p.tx_sz {
        3 | 9 | 10 | 17 | 18 => 1, // TX_32X32, TX_16X32, TX_32X16, TX_16X64, TX_64X16
        4 | 11 | 12 => 2,          // TX_64X64, TX_32X64, TX_64X32
        _ => 0,
    };
    let use_qm = p.qm_level < 15 && p.tx_type < IDTX;
    let qm = &QUANTIZER_MATRIX[p.qm_level.min(14)][usize::from(p.plane > 0)];
    let qm_offset = QM_OFFSET[p.tx_sz];
    let lim = 1i64 << (7 + p.bit_depth);

    for i in 0..rows {
        for j in 0..cols {
            let q = if i == 0 && j == 0 { p.dc_q } else { p.ac_q };
            let q2 = if use_qm {
                (i64::from(q) * i64::from(qm[qm_offset + i * tw + j]) + 16) >> 5
            } else {
                i64::from(q)
            };
            let dq = i64::from(quant[i * tw + j]) * q2;
            let mag = (dq.unsigned_abs() & 0xFF_FFFF) as i64 >> dq_shift;
            let dq2 = if dq < 0 { -mag } else { mag };
            scratch.dq[j * DQ_STRIDE + i] = dq2.clamp(-lim, lim - 1) as i32;
        }
    }

    let blk = &mut px[y * stride + x..];
    if rows == 1 && cols == 1 && p.tx_type == DCT_DCT && !p.lossless {
        let d = dc_only(&plan, scratch.dq[0]);
        let max = (1i32 << p.bit_depth) - 1;
        for i in 0..h {
            for s in &mut blk[i * stride..i * stride + w] {
                *s = (i32::from(*s) + d).clamp(0, max) as u16;
            }
        }
        return;
    }
    simd::itx::inverse_transform_add(simd, &plan, &scratch.dq, &mut scratch.mid, blk, stride);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn params(tx_sz: usize, tx_type: usize) -> ReconParams {
        ReconParams {
            tx_sz,
            tx_type,
            plane: 0,
            dc_q: 8,
            ac_q: 8,
            qm_level: 15,
            lossless: false,
            bit_depth: 8,
        }
    }

    #[test]
    fn zero_coefficients_leave_pixels_unchanged() {
        let mut px = vec![100u16; 16 * 16];
        reconstruct(
            &mut px,
            16,
            4,
            4,
            &params(1, 0),
            &[0; 64],
            &mut ItxScratch::new(),
            simd::best(),
        );
        assert!(px.iter().all(|&v| v == 100));
    }

    #[test]
    fn dc_only_adds_flat_value() {
        // 4x4 DCT_DCT: Dequant = 125 * 8 = 1000, which gives a flat residual of 31
        // (see transform tests); the pixels outside the block stay.
        let mut px = vec![100u16; 8 * 8];
        let mut quant = [0i32; 16];
        quant[0] = 125;
        reconstruct(
            &mut px,
            8,
            2,
            2,
            &params(0, 0),
            &quant,
            &mut ItxScratch::new(),
            simd::best(),
        );
        for r in 0..8 {
            for c in 0..8 {
                let inside = (2..6).contains(&r) && (2..6).contains(&c);
                assert_eq!(px[r * 8 + c], if inside { 131 } else { 100 });
            }
        }
    }

    #[test]
    fn output_is_clipped() {
        let mut px = vec![250u16; 16];
        let mut quant = [0i32; 16];
        quant[0] = 1000;
        reconstruct(
            &mut px,
            4,
            0,
            0,
            &params(0, 0),
            &quant,
            &mut ItxScratch::new(),
            simd::best(),
        );
        assert!(px.iter().all(|&v| v == 255));
        quant[0] = -1000;
        reconstruct(
            &mut px,
            4,
            0,
            0,
            &params(0, 0),
            &quant,
            &mut ItxScratch::new(),
            simd::best(),
        );
        assert!(px.iter().all(|&v| v < 255));
    }

    #[test]
    fn flipadst_is_adst_mirrored() {
        let mut quant = [0i32; 64];
        quant[0] = 40;
        quant[1] = -17;
        quant[8] = 23;
        quant[9] = 5;
        let run = |tx_type| {
            let mut px = vec![128u16; 64];
            reconstruct(
                &mut px,
                8,
                0,
                0,
                &params(1, tx_type),
                &quant,
                &mut ItxScratch::new(),
                simd::best(),
            );
            px
        };
        // ADST_DCT (1) vs FLIPADST_DCT (4): mirrored vertically.
        let (a, f) = (run(1), run(4));
        // DCT_ADST (2) vs DCT_FLIPADST (5): mirrored horizontally.
        let (c, g) = (run(2), run(5));
        // ADST_ADST (3) vs FLIPADST_FLIPADST (6): mirrored both ways.
        let (d, e) = (run(3), run(6));
        for i in 0..8 {
            for j in 0..8 {
                assert_eq!(f[i * 8 + j], a[(7 - i) * 8 + j]);
                assert_eq!(g[i * 8 + j], c[i * 8 + 7 - j]);
                assert_eq!(e[i * 8 + j], d[(7 - i) * 8 + 7 - j]);
            }
        }
        assert_ne!(a, f);
    }

    #[test]
    fn dq_denom_halves_32x32() {
        // Dequant = 1000 * 8 / 2 = 4000, which gives a flat residual of 31
        // (row 2828, row shift -> 707, column 500, Round2( 500, 4 )). Without the
        // denominator it would be 62.
        let mut quant = [0i32; 32 * 32];
        quant[0] = 1000;
        let mut px = vec![100u16; 32 * 32];
        reconstruct(
            &mut px,
            32,
            0,
            0,
            &params(3, 0),
            &quant,
            &mut ItxScratch::new(),
            simd::best(),
        );
        assert!(px.iter().all(|&v| v == 131));
    }

    #[test]
    fn flat_matrix_level_equals_no_matrix_for_weight_32() {
        // Level 0 luma 4x4 first weight is 32, so Round2( q * 32, 5 ) = q for the DC term.
        let mut quant = [0i32; 16];
        quant[0] = 125;
        let mut a = vec![100u16; 16];
        let mut b = vec![100u16; 16];
        let mut p = params(0, 0);
        reconstruct(
            &mut a,
            4,
            0,
            0,
            &p,
            &quant,
            &mut ItxScratch::new(),
            simd::best(),
        );
        p.qm_level = 0;
        reconstruct(
            &mut b,
            4,
            0,
            0,
            &p,
            &quant,
            &mut ItxScratch::new(),
            simd::best(),
        );
        assert_eq!(a, b);
    }

    /// The region the coefficient reader tracks (the last nonzero row and column, plus 1, of
    /// the entries it writes) gives the samples of the scanning `reconstruct`, on random sparse
    /// blocks of every size, a few types and bit depths 8, 10 and 12.
    #[test]
    fn tracked_region_matches_the_scan() {
        let mut rng = crate::symbol::XorShift(3);
        for tx_sz in 0..19 {
            let (w, h) = (
                1usize << TX_WIDTH_LOG2[tx_sz],
                1usize << TX_HEIGHT_LOG2[tx_sz],
            );
            let (tw, th) = (w.min(32), h.min(32));
            for &(tx_type, bit_depth) in &[(DCT_DCT, 8), (1, 10), (IDTX, 12)] {
                let mut quant = [0i32; 1024];
                let (mut rows, mut cols) = (1, 1);
                for _ in 0..1 + rng.next_u32() % 6 {
                    let pos = rng.next_u32() as usize % (tw * th);
                    let v = (rng.next_u32() % 64) as i32 - 32;
                    quant[pos] = v;
                    if v != 0 {
                        rows = rows.max(pos / tw + 1);
                        cols = cols.max(pos % tw + 1);
                    }
                }
                let mut p = params(tx_sz, tx_type);
                p.bit_depth = bit_depth;
                let mut a = vec![1u16 << (bit_depth - 1); w * h];
                let mut b = a.clone();
                let mut scratch = ItxScratch::new();
                reconstruct(&mut a, w, 0, 0, &p, &quant, &mut scratch, simd::best());
                reconstruct_region(
                    &mut b,
                    w,
                    0,
                    0,
                    &p,
                    &quant,
                    (rows, cols),
                    &mut scratch,
                    simd::best(),
                );
                assert_eq!(a, b, "tx_sz {tx_sz} type {tx_type}");
            }
        }
    }
}
