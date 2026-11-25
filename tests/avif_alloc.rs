//! Peak heap of `decode_avif`, measured by a counting global allocator over every thread: for a
//! plain item, an item with an alpha image and a 2x2 grid, at several `max_alloc` and
//! `max_threads`, the heap held at once (the parse, the item bytes, the item copy, the decoders,
//! the canvas and the output) stays within `max_alloc` plus the few fixed-size buffers that are not
//! charged.

use av1::{AvifImage, Decoder, Error, Format, Frame, Limits, Options};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicBool, AtomicIsize, Ordering::Relaxed};

struct Counting;

static ON: AtomicBool = AtomicBool::new(false);
static LIVE: AtomicIsize = AtomicIsize::new(0);
static PEAK: AtomicIsize = AtomicIsize::new(0);

fn note(delta: isize) {
    if ON.load(Relaxed) {
        let live = LIVE.fetch_add(delta, Relaxed) + delta;
        PEAK.fetch_max(live, Relaxed);
    }
}

// SAFETY: every call goes to `System` unchanged; the counters are atomics, which never allocate.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size() as isize);
        // SAFETY: the caller upholds `GlobalAlloc::alloc`'s contract, passed on as is.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size() as isize);
        // SAFETY: as for `alloc`.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, ptr: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        note(new_size as isize - layout.size() as isize);
        // SAFETY: the caller upholds `GlobalAlloc::realloc`'s contract, passed on as is.
        unsafe { System.realloc(ptr, layout, new_size) }
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        note(-(layout.size() as isize));
        // SAFETY: the caller upholds `GlobalAlloc::dealloc`'s contract, passed on as is.
        unsafe { System.dealloc(ptr, layout) }
    }
}

#[global_allocator]
static ALLOCATOR: Counting = Counting;

/// Runs `f`; returns its result and the most heap bytes, on all threads, it held at once above
/// what was live when it started (the result counts while it is alive).
fn measured<T>(f: impl FnOnce() -> T) -> (T, isize) {
    LIVE.store(0, Relaxed);
    PEAK.store(0, Relaxed);
    ON.store(true, Relaxed);
    let r = f();
    ON.store(false, Relaxed);
    (r, PEAK.load(Relaxed))
}

/// Heap one decoder uses without charging it (see tests/alloc.rs).
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

fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = ((body.len() as u32) + 8).to_be_bytes().to_vec();
    v.extend_from_slice(kind);
    v.extend_from_slice(body);
    v
}

/// An image item: id, type, data and its `ipma` associations.
struct Item {
    id: u8,
    kind: &'static [u8; 4],
    data: Vec<u8>,
    props: Vec<u8>,
}

/// An AVIF whose primary item is item 1. `props` is the `ipco` list, `refs` the `iref` boxes.
fn build(items: &[Item], props: &[Vec<u8>], refs: &[Vec<u8>]) -> Vec<u8> {
    let ftyp = bx(b"ftyp", b"avif\0\0\0\0avifmif1");
    let meta = |mdat_at: u32| {
        let hdlr = bx(b"hdlr", &[&[0; 8][..], b"pict", &[0; 13]].concat());
        let pitm = bx(b"pitm", &[0, 0, 0, 0, 0, 1]);
        let mut iinf = vec![0, 0, 0, 0, 0, items.len() as u8];
        let mut iloc = vec![0, 0, 0, 0, 0x44, 0, 0, items.len() as u8];
        let mut ipma = [&[0, 0, 0, 0][..], &(items.len() as u32).to_be_bytes()].concat();
        let mut at = mdat_at;
        for it in items {
            iinf.extend(bx(
                b"infe",
                &[&[2, 0, 0, 0, 0, it.id, 0, 0][..], it.kind, &[0]].concat(),
            ));
            iloc.extend_from_slice(&[0, it.id, 0, 0, 0, 1]);
            iloc.extend_from_slice(&at.to_be_bytes());
            iloc.extend_from_slice(&(it.data.len() as u32).to_be_bytes());
            at += it.data.len() as u32;
            ipma.extend_from_slice(&[0, it.id, it.props.len() as u8]);
            ipma.extend_from_slice(&it.props);
        }
        let ipco = bx(b"ipco", &props.concat());
        let iprp = bx(b"iprp", &[ipco, bx(b"ipma", &ipma)].concat());
        let iref = bx(b"iref", &[&[0; 4][..], &refs.concat()].concat());
        bx(
            b"meta",
            &[
                &[0; 4][..],
                &hdlr,
                &pitm,
                &bx(b"iinf", &iinf),
                &bx(b"iloc", &iloc),
                &iprp,
                &iref,
            ]
            .concat(),
        )
    };
    let at = (ftyp.len() + meta(0).len() + 8) as u32;
    let mdat = bx(
        b"mdat",
        &items
            .iter()
            .flat_map(|i| i.data.clone())
            .collect::<Vec<_>>(),
    );
    [ftyp, meta(at), mdat].concat()
}

fn ispe(w: u32, h: u32) -> Vec<u8> {
    bx(
        b"ispe",
        &[&[0; 4][..], &w.to_be_bytes(), &h.to_be_bytes()].concat(),
    )
}

/// The `ipco` list: tile `ispe` (1), `av1C` (2), alpha `auxC` (3), grid `ispe` (4).
fn props() -> Vec<Vec<u8>> {
    let urn = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0";
    vec![
        ispe(64, 32),
        bx(b"av1C", &[0x81, 0, 0x4c, 0]),
        bx(b"auxC", &[&[0; 4][..], urn].concat()),
        ispe(128, 64),
    ]
}

fn av01(id: u8, data: &[u8], props: &[u8]) -> Item {
    Item {
        id,
        kind: b"av01",
        data: data.to_vec(),
        props: props.to_vec(),
    }
}

fn refs(kind: &[u8; 4], from: u8, to: &[u8]) -> Vec<u8> {
    let mut v = vec![0, from, 0, to.len() as u8];
    for t in to {
        v.extend_from_slice(&[0, *t]);
    }
    bx(kind, &v)
}

fn files() -> Vec<(&'static str, Vec<u8>)> {
    let (data, _) = decodable_unit(0, (64, 32));
    let plain = build(&[av01(1, &data, &[1, 0x82])], &props(), &[]);
    let alpha = build(
        &[av01(1, &data, &[1, 0x82]), av01(2, &data, &[1, 0x82, 3])],
        &props(),
        &[refs(b"auxl", 2, &[1])],
    );
    let mut items = vec![Item {
        id: 1,
        kind: b"grid",
        data: vec![0, 0, 1, 1, 0, 128, 0, 64],
        props: vec![4],
    }];
    items.extend((2..=5).map(|id| av01(id, &data, &[1, 0x82])));
    let grid = build(&items, &props(), &[refs(b"dimg", 1, &[2, 3, 4, 5])]);
    vec![("plain", plain), ("alpha", alpha), ("grid", grid)]
}

#[test]
fn decode_avif_keeps_the_heap_within_max_alloc() {
    let cancel = AtomicBool::new(false);
    let decode = |file: &[u8], max_alloc: usize, max_threads: usize| {
        let limits = Limits {
            max_alloc,
            max_threads,
            ..Limits::default()
        };
        measured(|| av1::decode_avif(file, &Options::default(), &limits, &cancel))
    };
    for (name, file) in files() {
        let (want, _) = decode(&file, 64 << 20, 1);
        let want: AvifImage = want.unwrap_or_else(|e| panic!("{name}: reference decode: {e:?}"));
        let want_frames: Vec<&Frame> = std::iter::once(&want.image).chain(&want.alpha).collect();
        let mut fitted = 0;
        for threads in [1, 2, 4] {
            for max_alloc in (0..=11).map(|k| 16usize << k << 10).chain([64 << 20]) {
                let (r, peak) = decode(&file, max_alloc, threads);
                assert!(
                    peak <= max_alloc as isize + UNCHARGED * (threads as isize + 1),
                    "{name} max_alloc {max_alloc} threads {threads}: {peak} bytes held, result {:?}",
                    r.as_ref().map(|_| ())
                );
                match r {
                    Ok(got) => {
                        fitted += 1;
                        let frames: Vec<&Frame> =
                            std::iter::once(&got.image).chain(&got.alpha).collect();
                        assert_eq!(frames, want_frames, "{name} max_alloc {max_alloc}");
                    }
                    Err(e) => assert!(
                        matches!(
                            e,
                            Error::AllocLimit | Error::OutOfMemory | Error::Container(_)
                        ),
                        "{name} max_alloc {max_alloc} threads {threads}: {e:?}"
                    ),
                }
            }
        }
        // Some limits fit and some do not, so both paths were checked.
        assert!(fitted > 0 && fitted < 3 * 13, "{name}: {fitted} fitted");
    }
}
