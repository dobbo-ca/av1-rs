//! Upscaling process (spec 7.16): horizontal superres upscaling of a frame from `FrameWidth` to
//! `UpscaledWidth` samples.

use crate::consts::{
    FILTER_BITS, SUPERRES_EXTRA_BITS, SUPERRES_FILTER_OFFSET, SUPERRES_FILTER_SHIFTS,
    SUPERRES_FILTER_TAPS, SUPERRES_SCALE_BITS, SUPERRES_SCALE_MASK,
};
use crate::decoder::check_cancel;
use crate::error::Error;
use crate::frame::{FrameState, Plane};
use crate::header::FrameHeader;
use crate::seq::SequenceHeader;
use std::sync::atomic::AtomicBool;

/// `Upscale_Filter` (spec 7.16).
pub(crate) const UPSCALE_FILTER: [[i32; SUPERRES_FILTER_TAPS]; SUPERRES_FILTER_SHIFTS] = [
    [0, 0, 0, 128, 0, 0, 0, 0],
    [0, 0, -1, 128, 2, -1, 0, 0],
    [0, 1, -3, 127, 4, -2, 1, 0],
    [0, 1, -4, 127, 6, -3, 1, 0],
    [0, 2, -6, 126, 8, -3, 1, 0],
    [0, 2, -7, 125, 11, -4, 1, 0],
    [-1, 2, -8, 125, 13, -5, 2, 0],
    [-1, 3, -9, 124, 15, -6, 2, 0],
    [-1, 3, -10, 123, 18, -6, 2, -1],
    [-1, 3, -11, 122, 20, -7, 3, -1],
    [-1, 4, -12, 121, 22, -8, 3, -1],
    [-1, 4, -13, 120, 25, -9, 3, -1],
    [-1, 4, -14, 118, 28, -9, 3, -1],
    [-1, 4, -15, 117, 30, -10, 4, -1],
    [-1, 5, -16, 116, 32, -11, 4, -1],
    [-1, 5, -16, 114, 35, -12, 4, -1],
    [-1, 5, -17, 112, 38, -12, 4, -1],
    [-1, 5, -18, 111, 40, -13, 5, -1],
    [-1, 5, -18, 109, 43, -14, 5, -1],
    [-1, 6, -19, 107, 45, -14, 5, -1],
    [-1, 6, -19, 105, 48, -15, 5, -1],
    [-1, 6, -19, 103, 51, -16, 5, -1],
    [-1, 6, -20, 101, 53, -16, 6, -1],
    [-1, 6, -20, 99, 56, -17, 6, -1],
    [-1, 6, -20, 97, 58, -17, 6, -1],
    [-1, 6, -20, 95, 61, -18, 6, -1],
    [-2, 7, -20, 93, 64, -18, 6, -2],
    [-2, 7, -20, 91, 66, -19, 6, -1],
    [-2, 7, -20, 88, 69, -19, 6, -1],
    [-2, 7, -20, 86, 71, -19, 6, -1],
    [-2, 7, -20, 84, 74, -20, 7, -2],
    [-2, 7, -20, 81, 76, -20, 7, -1],
    [-2, 7, -20, 79, 79, -20, 7, -2],
    [-1, 7, -20, 76, 81, -20, 7, -2],
    [-2, 7, -20, 74, 84, -20, 7, -2],
    [-1, 6, -19, 71, 86, -20, 7, -2],
    [-1, 6, -19, 69, 88, -20, 7, -2],
    [-1, 6, -19, 66, 91, -20, 7, -2],
    [-2, 6, -18, 64, 93, -20, 7, -2],
    [-1, 6, -18, 61, 95, -20, 6, -1],
    [-1, 6, -17, 58, 97, -20, 6, -1],
    [-1, 6, -17, 56, 99, -20, 6, -1],
    [-1, 6, -16, 53, 101, -20, 6, -1],
    [-1, 5, -16, 51, 103, -19, 6, -1],
    [-1, 5, -15, 48, 105, -19, 6, -1],
    [-1, 5, -14, 45, 107, -19, 6, -1],
    [-1, 5, -14, 43, 109, -18, 5, -1],
    [-1, 5, -13, 40, 111, -18, 5, -1],
    [-1, 4, -12, 38, 112, -17, 5, -1],
    [-1, 4, -12, 35, 114, -16, 5, -1],
    [-1, 4, -11, 32, 116, -16, 5, -1],
    [-1, 4, -10, 30, 117, -15, 4, -1],
    [-1, 3, -9, 28, 118, -14, 4, -1],
    [-1, 3, -9, 25, 120, -13, 4, -1],
    [-1, 3, -8, 22, 121, -12, 4, -1],
    [-1, 3, -7, 20, 122, -11, 3, -1],
    [-1, 2, -6, 18, 123, -10, 3, -1],
    [0, 2, -6, 15, 124, -9, 3, -1],
    [0, 2, -5, 13, 125, -8, 2, -1],
    [0, 1, -4, 11, 125, -7, 2, 0],
    [0, 1, -3, 8, 126, -6, 2, 0],
    [0, 1, -3, 6, 127, -4, 1, 0],
    [0, 1, -2, 4, 127, -3, 1, 0],
    [0, 0, -1, 2, 128, -1, 0, 0],
];

/// Spec 7.16, the inner `x` loop for one row: `src` is one input row, `dst` the output row
/// (`dst.len()` samples are produced), `max_x` is `maxX` (already limited to the row), `max_val`
/// is `(1 << BitDepth) - 1`.
fn upscale_row(
    src: &[u16],
    dst: &mut [u16],
    initial_subpel_x: i64,
    step_x: i64,
    max_x: i64,
    max_val: i64,
) {
    for (x, d) in dst.iter_mut().enumerate() {
        let src_x = -(1i64 << SUPERRES_SCALE_BITS) + initial_subpel_x + x as i64 * step_x;
        let src_x_px = src_x >> SUPERRES_SCALE_BITS;
        let src_x_subpel = ((src_x & SUPERRES_SCALE_MASK as i64) >> SUPERRES_EXTRA_BITS) as usize;
        let filter = &UPSCALE_FILTER[src_x_subpel];
        let mut sum = 0i64;
        for (k, &f) in filter.iter().enumerate() {
            let sample_x = (src_x_px + (k as i64 - SUPERRES_FILTER_OFFSET as i64)).clamp(0, max_x);
            sum += i64::from(src[sample_x as usize]) * i64::from(f);
        }
        *d = ((sum + (1 << (FILTER_BITS - 1))) >> FILTER_BITS).clamp(0, max_val) as u16;
    }
}

/// Upscaling process (spec 7.16): upscales every plane of `input` (`FrameWidth` wide) into `out`
/// (`UpscaledWidth` wide; plane `p` needs at least `Round2(UpscaledWidth, subX)` x
/// `Round2(FrameHeight, subY)` samples, anything beyond that is left alone). Only called when
/// `use_superres` is set. Reads are limited to the input plane, writes to the output plane, so a
/// too small plane gives a wrong picture, never a panic.
pub(crate) fn upscale(
    input: &[Plane],
    fs: &FrameState,
    seq: &SequenceHeader,
    fh: &FrameHeader,
    out: &mut [Plane],
    cancel: &AtomicBool,
) -> Result<(), Error> {
    let max_val = (1i64 << seq.bit_depth) - 1;
    let planes = fs.num_planes.min(input.len()).min(out.len());
    for plane in 0..planes {
        let (sub_x, sub_y) = if plane > 0 {
            (u32::from(seq.subsampling_x), u32::from(seq.subsampling_y))
        } else {
            (0, 0)
        };
        let round2 = |v: u32, n: u32| i64::from((v + ((1 << n) >> 1)) >> n);
        let downscaled_plane_w = round2(fh.frame_width, sub_x);
        let upscaled_plane_w = round2(fh.upscaled_width, sub_x);
        let plane_h = round2(fh.frame_height, sub_y) as usize;
        if upscaled_plane_w == 0 {
            continue;
        }
        let step_x = ((downscaled_plane_w << SUPERRES_SCALE_BITS) + (upscaled_plane_w / 2))
            / upscaled_plane_w;
        let err = (upscaled_plane_w * step_x) - (downscaled_plane_w << SUPERRES_SCALE_BITS);
        let mut initial_subpel_x = (-((upscaled_plane_w - downscaled_plane_w)
            << (SUPERRES_SCALE_BITS - 1))
            + upscaled_plane_w / 2)
            / upscaled_plane_w
            + (1 << (SUPERRES_EXTRA_BITS - 1))
            - err / 2;
        initial_subpel_x &= SUPERRES_SCALE_MASK as i64;
        let mi_w = i64::from(fh.mi_cols >> sub_x);
        let (inp, dst) = (&input[plane], &mut out[plane]);
        if inp.width == 0 || inp.stride == 0 || dst.stride == 0 {
            continue;
        }
        let max_x = (mi_w * 4 - 1).min(inp.width as i64 - 1).max(0);
        let out_w = (upscaled_plane_w as usize).min(dst.width);
        let rows = plane_h.min(inp.height).min(dst.height);
        for y in 0..rows {
            if y.is_multiple_of(64) {
                check_cancel(cancel)?;
            }
            let src_row = &inp.px[y * inp.stride..y * inp.stride + inp.width];
            let dst_row = &mut dst.px[y * dst.stride..y * dst.stride + out_w];
            upscale_row(src_row, dst_row, initial_subpel_x, step_x, max_x, max_val);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn filter_rows_have_unit_gain() {
        for row in &UPSCALE_FILTER {
            assert_eq!(row.iter().sum::<i32>(), 128);
        }
    }

    #[test]
    fn step_one_pixel_phase_zero_reproduces_the_input() {
        // srcX = -(1 << 14) + init + x * (1 << 14) with init = 1 << 14 is exactly x, phase 0,
        // and filter row 0 is the unit impulse at tap 3 = SUPERRES_FILTER_OFFSET.
        let src: Vec<u16> = (0..16).map(|v| v * 37 % 255).collect();
        let mut dst = vec![0u16; 16];
        upscale_row(
            &src,
            &mut dst,
            1 << SUPERRES_SCALE_BITS,
            1 << SUPERRES_SCALE_BITS,
            15,
            255,
        );
        assert_eq!(dst, src);
    }

    #[test]
    fn constant_row_stays_constant() {
        let src = vec![100u16; 8];
        let mut dst = vec![0u16; 13];
        // 8 -> 13 samples: step = ((8 << 14) + 6) / 13, any init.
        upscale_row(&src, &mut dst, 5000, 10082, 7, 255);
        assert!(dst.iter().all(|&v| v == 100));
    }
}
