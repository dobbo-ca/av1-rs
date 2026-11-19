//! Reconstruction: dequantization, inverse transform and adding the residual (spec 7.12.3).

#![allow(dead_code)]

use crate::consts::IDTX;
use crate::quant::{QM_OFFSET, QUANTIZER_MATRIX};
use crate::tables::{TX_HEIGHT_LOG2, TX_WIDTH_LOG2};
use crate::transform::{FLIP_LR, FLIP_UD, inverse_transform_2d};

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

/// Spec 7.12.3 reconstruct process.
///
/// `quant` is the spec's `Quant[]` with `Quant[ i * tw + j ]`, `tw = Min( 32, w )`,
/// `th = Min( 32, h )`. The block at `(x, y)` of size `w x h` is updated in `px` (row stride
/// `stride`). The caller guarantees `tx_sz < 19`, `tx_type < 16`, `plane < 3`, `qm_level <= 15`,
/// `quant.len() >= tw * th` and that the block lies inside `px`; if not, nothing is written.
pub(crate) fn reconstruct(
    px: &mut [u16],
    stride: usize,
    x: usize,
    y: usize,
    p: &ReconParams,
    quant: &[i32],
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

    let dq_denom: i64 = match p.tx_sz {
        3 | 9 | 10 | 17 | 18 => 2, // TX_32X32, TX_16X32, TX_32X16, TX_16X64, TX_64X16
        4 | 11 | 12 => 4,          // TX_64X64, TX_32X64, TX_64X32
        _ => 1,
    };
    let use_qm = p.qm_level < 15 && p.tx_type < IDTX;
    let qm = &QUANTIZER_MATRIX[p.qm_level.min(14)][usize::from(p.plane > 0)];
    let qm_offset = QM_OFFSET[p.tx_sz];
    let lim = 1i64 << (7 + p.bit_depth);

    let mut dequant = [0i32; 64 * 64];
    for i in 0..th {
        for j in 0..tw {
            let q = if i == 0 && j == 0 { p.dc_q } else { p.ac_q };
            let q2 = if use_qm {
                (i64::from(q) * i64::from(qm[qm_offset + i * tw + j]) + 16) >> 5
            } else {
                i64::from(q)
            };
            let dq = i64::from(quant[i * tw + j]) * q2;
            let mag = (dq.unsigned_abs() & 0xFF_FFFF) as i64 / dq_denom;
            let dq2 = if dq < 0 { -mag } else { mag };
            dequant[i * w + j] = dq2.clamp(-lim, lim - 1) as i32;
        }
    }

    let mut residual = [0i32; 64 * 64];
    inverse_transform_2d(
        &dequant,
        &mut residual,
        p.tx_sz,
        p.tx_type,
        p.lossless,
        p.bit_depth,
    );

    let (flip_ud, flip_lr) = (FLIP_UD[p.tx_type], FLIP_LR[p.tx_type]);
    let max = (1i32 << p.bit_depth) - 1;
    for i in 0..h {
        let yy = if flip_ud { h - i - 1 } else { i };
        let row = &mut px[(y + yy) * stride + x..(y + yy) * stride + x + w];
        for j in 0..w {
            let xx = if flip_lr { w - j - 1 } else { j };
            row[xx] = (i32::from(row[xx]) + residual[i * w + j]).clamp(0, max) as u16;
        }
    }
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
        reconstruct(&mut px, 16, 4, 4, &params(1, 0), &[0; 64]);
        assert!(px.iter().all(|&v| v == 100));
    }

    #[test]
    fn dc_only_adds_flat_value() {
        // 4x4 DCT_DCT: Dequant = 125 * 8 = 1000, which gives a flat residual of 31
        // (see transform tests); the pixels outside the block stay.
        let mut px = vec![100u16; 8 * 8];
        let mut quant = [0i32; 16];
        quant[0] = 125;
        reconstruct(&mut px, 8, 2, 2, &params(0, 0), &quant);
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
        reconstruct(&mut px, 4, 0, 0, &params(0, 0), &quant);
        assert!(px.iter().all(|&v| v == 255));
        quant[0] = -1000;
        reconstruct(&mut px, 4, 0, 0, &params(0, 0), &quant);
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
            reconstruct(&mut px, 8, 0, 0, &params(1, tx_type), &quant);
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
        reconstruct(&mut px, 32, 0, 0, &params(3, 0), &quant);
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
        reconstruct(&mut a, 4, 0, 0, &p, &quant);
        p.qm_level = 0;
        reconstruct(&mut b, 4, 0, 0, &p, &quant);
        assert_eq!(a, b);
    }
}
