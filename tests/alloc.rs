//! Heap use of a decode with buffer reuse, measured by a counting global allocator: with
//! `max_threads = 1`, a second temporal unit of the same shape allocates nothing once the frames
//! of the first were handed back with `Decoder::recycle`, and no decode holds more heap than
//! `max_alloc` plus the few fixed-size buffers that are not charged.

use av1::{Decoder, Error, Format, Frame, Limits, Options};
use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::AtomicBool;

/// Counts the allocations and the live bytes of the thread that turned counting on.
struct Counting;

thread_local! {
    static ON: Cell<bool> = const { Cell::new(false) };
    static COUNT: Cell<usize> = const { Cell::new(0) };
    static LIVE: Cell<isize> = const { Cell::new(0) };
    static PEAK: Cell<isize> = const { Cell::new(0) };
}

/// Records an allocation (`new`) or a size change of `delta` bytes on this thread.
fn note(new: bool, delta: isize) {
    let _ = ON.try_with(|on| {
        if on.get() {
            let _ = COUNT.try_with(|c| c.set(c.get() + usize::from(new)));
            let _ = LIVE.try_with(|l| {
                l.set(l.get() + delta);
                let _ = PEAK.try_with(|p| p.set(p.get().max(l.get())));
            });
        }
    });
}

// SAFETY: every call goes to `System` unchanged; the counters are const-initialised thread
// locals without destructors, which never allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(true, layout.size() as isize);
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, passed on as is.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(true, layout.size() as isize);
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(true, new_size as isize - layout.size() as isize);
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract, passed on as is.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note(false, -(layout.size() as isize));
        // SAFETY: the caller upholds `GlobalAlloc::dealloc`'s contract, passed on as is.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Runs `f`; returns its result, the allocations it made on this thread and the most heap bytes
/// it held at once above what was live when it started.
fn measured<T>(f: impl FnOnce() -> T) -> (T, usize, isize) {
    COUNT.with(|c| c.set(0));
    LIVE.with(|l| l.set(0));
    PEAK.with(|p| p.set(0));
    ON.with(|on| on.set(true));
    let r = f();
    ON.with(|on| on.set(false));
    (r, COUNT.with(|c| c.get()), PEAK.with(|p| p.get()))
}

/// Heap the decoder uses without charging it: the frame and tile CDFs (about 13 KB each), the
/// two palette colour maps (4 KB each), the tile and plane lists and the output `Vec`s.
const UNCHARGED: isize = 64 << 10;

/// Writes bits high first.
#[derive(Default)]
struct Bits {
    bytes: Vec<u8>,
    n: usize,
}

impl Bits {
    fn put(&mut self, bits: u32, value: u64) {
        for i in (0..bits).rev() {
            if self.n.is_multiple_of(8) {
                self.bytes.push(0);
            }
            let last = self.bytes.len() - 1;
            self.bytes[last] |= (((value >> i) & 1) as u8) << (7 - self.n % 8);
            self.n += 1;
        }
    }

    fn align(&mut self) {
        while !self.n.is_multiple_of(8) {
            self.put(1, 0);
        }
    }
}

/// A low-overhead OBU with a one byte size field.
fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![(obu_type << 3) | 2, payload.len() as u8];
    v.extend_from_slice(payload);
    v
}

/// A reduced still picture sequence header: `width` x `height` (one superblock: at most 64x64),
/// 8-bit (10-bit when `high_bitdepth` is 1) 4:2:0, 64x64 superblocks, CDEF and loop restoration
/// on.
fn sequence_header(high_bitdepth: u64, width: u64, height: u64) -> Vec<u8> {
    let mut w = Bits::default();
    for (bits, value) in [
        (3, 0),             // seq_profile
        (1, 1),             // still_picture
        (1, 1),             // reduced_still_picture_header
        (5, 4),             // seq_level_idx[0]
        (4, 7),             // frame_width_bits_minus_1
        (4, 6),             // frame_height_bits_minus_1
        (8, width - 1),     // max_frame_width_minus_1
        (7, height - 1),    // max_frame_height_minus_1
        (1, 0),             // use_128x128_superblock
        (1, 1),             // enable_filter_intra
        (1, 1),             // enable_intra_edge_filter
        (1, 0),             // enable_superres
        (1, 1),             // enable_cdef
        (1, 1),             // enable_restoration
        (1, high_bitdepth), // high_bitdepth
        (1, 0),             // mono_chrome
        (1, 0),             // color_description_present_flag
        (1, 0),             // color_range
        (2, 0),             // chroma_sample_position
        (1, 0),             // separate_uv_delta_q
        (1, 0),             // film_grain_params_present
        (1, 1),             // trailing one bit
    ] {
        w.put(bits, value);
    }
    w.align();
    w.bytes
}

/// The key frame header of that sequence (loop filter, CDEF, switchable loop restoration on U),
/// byte aligned, then `tile` as the data of its one tile.
fn frame(tile: &[u8]) -> Vec<u8> {
    let mut w = Bits::default();
    w.put(1, 0); // disable_cdf_update
    w.put(1, 0); // allow_screen_content_tools
    w.put(1, 0); // render_and_frame_size_different
    w.put(1, 1); // uniform_tile_spacing_flag
    w.put(8, 100); // base_q_idx
    w.put(4, 0); // DeltaQYDc, DeltaQUDc, DeltaQUAc not coded; using_qmatrix 0
    w.put(1, 0); // segmentation_enabled
    w.put(1, 0); // delta_q_present
    for level in [10, 11, 12, 13] {
        w.put(6, level); // loop_filter_level[ i ]
    }
    w.put(3, 2); // loop_filter_sharpness
    w.put(1, 0); // loop_filter_delta_enabled
    w.put(2, 1); // cdef_damping_minus_3
    w.put(2, 1); // cdef_bits
    for i in 0..2 {
        w.put(4, i + 1); // cdef_y_pri_strength
        w.put(2, 3); // cdef_y_sec_strength
        w.put(4, 5); // cdef_uv_pri_strength
        w.put(2, 0); // cdef_uv_sec_strength
    }
    w.put(2, 0); // lr_type[0] = NONE
    w.put(2, 1); // lr_type[1] = SWITCHABLE
    w.put(2, 0); // lr_type[2] = NONE
    w.put(1, 1); // lr_unit_shift
    w.put(1, 0); // lr_unit_extra_shift
    w.put(1, 1); // lr_uv_shift
    w.put(1, 1); // tx_mode_select
    w.put(1, 0); // reduced_tx_set
    w.align();
    let mut payload = w.bytes;
    payload.extend_from_slice(tile);
    payload
}

/// Temporal delimiter, sequence header (`high_bitdepth` as given) and one key frame with tile
/// data `tile`.
fn unit(tile: &[u8], high_bitdepth: u64, size: (u64, u64)) -> Vec<u8> {
    [
        obu(2, &[]),
        obu(1, &sequence_header(high_bitdepth, size.0, size.1)),
        obu(6, &frame(tile)),
    ]
    .concat()
}

fn limits(max_alloc: usize) -> Limits {
    Limits {
        max_alloc,
        max_threads: 1,
        ..Limits::default()
    }
}

/// A unit that decodes to one frame (tile data is read as some syntax; the first candidate whose
/// syntax is valid), with the frame.
fn decodable_unit(high_bitdepth: u64, size: (u64, u64)) -> (Vec<u8>, Vec<Frame>) {
    let cancel = AtomicBool::new(false);
    let tiles: [&[u8]; 7] = [
        &[1, 2, 3, 4],
        &[0],
        &[0x80, 0x00],
        &[0x55; 16],
        &[0x12, 0x34, 0x56, 0x78, 0x9a],
        &[0; 64],
        &[0xff; 8],
    ];
    for t in tiles {
        let data = unit(t, high_bitdepth, size);
        let mut d = Decoder::new(Options::default(), limits(64 << 20));
        if let Ok(frames) = d.decode_temporal_unit(&data, Format::Section5, &cancel)
            && frames.len() == 1
        {
            return (data, frames);
        }
    }
    panic!("no candidate tile decodes");
}

#[test]
fn second_unit_after_recycle_allocates_nothing() {
    let (data, want) = decodable_unit(0, (64, 32));
    let cancel = AtomicBool::new(false);
    let mut d = Decoder::new(Options::default(), limits(64 << 20));
    for round in 0..3 {
        let (frames, allocations, _) =
            measured(|| d.decode_temporal_unit(&data, Format::Section5, &cancel));
        let frames = frames.unwrap();
        assert_eq!(frames, want, "round {round}");
        if round > 0 {
            assert_eq!(allocations, 0, "allocations in round {round}");
        }
        d.recycle(frames);
    }
}

#[test]
fn reuse_keeps_the_heap_within_max_alloc() {
    let (data, want) = decodable_unit(0, (64, 32));
    let cancel = AtomicBool::new(false);
    let mut fitted = 0;
    for max_alloc in (1..=64).map(|k| k * (8 << 10)) {
        let mut d = Decoder::new(Options::default(), limits(max_alloc));
        let mut ok = None;
        for round in 0..2 {
            let (r, _, peak) =
                measured(|| d.decode_temporal_unit(&data, Format::Section5, &cancel));
            assert!(
                peak <= max_alloc as isize + UNCHARGED,
                "max_alloc {max_alloc} round {round}: {peak} bytes held"
            );
            // Reuse never makes a unit fail that fitted before, nor the reverse.
            assert_eq!(
                *ok.get_or_insert(r.is_ok()),
                r.is_ok(),
                "max_alloc {max_alloc}"
            );
            match r {
                Ok(frames) => {
                    assert_eq!(frames, want);
                    fitted += 1;
                    d.recycle(frames);
                }
                Err(e) => {
                    assert_eq!(e, Error::AllocLimit, "max_alloc {max_alloc}");
                    d.take_frames();
                }
            }
        }
    }
    // Some limits fit and some do not, so both paths were checked.
    assert!(fitted > 0 && fitted < 128);
}

#[test]
fn decode_of_many_high_bitdepth_frames_keeps_the_heap_within_max_alloc() {
    // 10-bit output planes come from the pool of (larger) work planes: the frames `decode`
    // collects must be charged for what they hold, not for their samples.
    let (data, _) = decodable_unit(1, (64, 32));
    let stream = data.repeat(400);
    let max_alloc = 2 << 20;
    let cancel = AtomicBool::new(false);
    let (r, _, peak) = measured(|| {
        av1::decode(
            &stream,
            Format::Section5,
            &Options::default(),
            &limits(max_alloc),
            &cancel,
        )
    });
    assert!(
        peak <= max_alloc as isize + UNCHARGED,
        "{peak} bytes held, result {:?}",
        r.map(|f| f.len())
    );
}

#[test]
fn decode_of_many_tiny_frames_keeps_the_heap_within_max_alloc() {
    // A 1x1 frame has 3 bytes of samples: the `Frame`, its plane list and its slot in the
    // collected `Vec` must be charged too, or a long stream of them outgrows `max_alloc`.
    let (data, frames) = decodable_unit(0, (1, 1));
    assert_eq!((frames[0].width, frames[0].height), (1, 1));
    let stream = data.repeat(8000);
    let max_alloc = 1 << 20;
    let cancel = AtomicBool::new(false);
    let (r, _, peak) = measured(|| {
        av1::decode(
            &stream,
            Format::Section5,
            &Options::default(),
            &limits(max_alloc),
            &cancel,
        )
    });
    assert!(
        peak <= max_alloc as isize + UNCHARGED,
        "{peak} bytes held, result {:?}",
        r.map(|f| f.len())
    );
    assert_eq!(r.err(), Some(Error::AllocLimit));
}
