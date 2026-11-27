//! Peak heap of `decode_avif` and of `Decoder::decode_temporal_units`, measured by a counting
//! global allocator over every thread. For AVIF: a plain item, an item with an alpha image, a
//! 2x2 grid, the same grid and a 1x2 grid with each tile stored in two extents, and an item and
//! a grid whose decoded frames are above the uncharged slack (rotated tiles, a mirrored grid, with
//! `apply_transforms` on), and a hostile grid of 16384 references to one tiny tile, at many
//! `max_alloc` and `max_threads` values, the heap held at once
//! (the parse, the item bytes, the item copy, the decoders, the transforms, the canvas and the
//! output) stays within the ledger's three quarters of `max_alloc` plus what the parse and the
//! item-byte copies held and the few fixed-size buffers that are not charged, and a single
//! item, and a grid up to its tile count, give the same result (the image or the error) at every
//! thread count. For a batch of units: the peak stays within `max_alloc` plus the uncharged
//! buffers at 2 and 4 threads.

use av1::{AvifImage, Decoder, Error, Format, Frame, Limits, Options};
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::Mutex;
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

/// The counters see every thread of the process, so the tests take turns.
static TURN: Mutex<()> = Mutex::new(());

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

/// A reduced still picture sequence header: `width` x `height` (at most 4096x4096), 8-bit
/// (10-bit when `high_bitdepth` is 1) 4:2:0, 64x64 superblocks, CDEF and loop restoration on.
fn sequence_header(high_bitdepth: u64, width: u64, height: u64) -> Vec<u8> {
    let mut w = Bits::default();
    for (bits, value) in [
        (3, 0),             // seq_profile
        (1, 1),             // still_picture
        (1, 1),             // reduced_still_picture_header
        (5, 4),             // seq_level_idx[0]
        (4, 11),            // frame_width_bits_minus_1
        (4, 11),            // frame_height_bits_minus_1
        (12, width - 1),    // max_frame_width_minus_1
        (12, height - 1),   // max_frame_height_minus_1
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
/// byte aligned, then `tile` as the data of its one tile (of a `width` x `height` frame).
fn frame(tile: &[u8], (width, height): (u64, u64)) -> Vec<u8> {
    let mut w = Bits::default();
    w.put(1, 0); // disable_cdf_update
    w.put(1, 0); // allow_screen_content_tools
    w.put(1, 0); // render_and_frame_size_different
    w.put(1, 1); // uniform_tile_spacing_flag
    if width > 64 {
        w.put(1, 0); // increment_tile_cols_log2: one tile column
    }
    if height > 64 {
        w.put(1, 0); // increment_tile_rows_log2: one tile row
    }
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
        obu(6, &frame(tile, size)),
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

/// A padding OBU of `n` zero bytes (its size field is LEB128).
fn padding(n: usize) -> Vec<u8> {
    let mut v = vec![(15 << 3) | 2];
    let mut s = n;
    while s >= 0x80 {
        v.push((s & 0x7f) as u8 | 0x80);
        s >>= 7;
    }
    v.push(s as u8);
    v.resize(v.len() + n, 0);
    v
}

/// An image item: id, type, data and its `ipma` associations. A `split` item is stored as two
/// `iloc` extents, so `isobmff` copies its bytes.
struct Item {
    id: u8,
    kind: &'static [u8; 4],
    data: Vec<u8>,
    props: Vec<u8>,
    split: bool,
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
            let n = it.data.len() as u32;
            let extents = if it.split {
                vec![(at, n / 2), (at + n / 2, n - n / 2)]
            } else {
                vec![(at, n)]
            };
            iloc.extend_from_slice(&[0, it.id, 0, 0, 0, extents.len() as u8]);
            for (offset, length) in extents {
                iloc.extend_from_slice(&offset.to_be_bytes());
                iloc.extend_from_slice(&length.to_be_bytes());
            }
            at += n;
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

/// The `ipco` list: small item and tile `ispe` (1), `av1C` (2), alpha `auxC` (3), small grid
/// `ispe` (4), `irot` 90 (5), `imir` (6), large item and tile `ispe` (7), large grid `ispe` (8),
/// 1x2 grid `ispe` (9), tiny tile `ispe` (10), hostile grid `ispe` (11).
fn props() -> Vec<Vec<u8>> {
    let urn = b"urn:mpeg:mpegB:cicp:systems:auxiliary:alpha\0";
    vec![
        ispe(64, 32),
        bx(b"av1C", &[0x81, 0, 0x4c, 0]),
        bx(b"auxC", &[&[0; 4][..], urn].concat()),
        ispe(128, 64),
        bx(b"irot", &[1]),
        bx(b"imir", &[0]),
        ispe(BIG_TILE, BIG_TILE),
        ispe(2 * BIG_TILE, 2 * BIG_TILE),
        ispe(128, 32),
        ispe(TINY, TINY),
        ispe(HOSTILE * TINY, HOSTILE * TINY),
    ]
}

/// The hostile grid has `HOSTILE` x `HOSTILE` tiles of `TINY` x `TINY`, every one a reference
/// to the same item: a small file and a small canvas, but a work list as long as the tile count.
const HOSTILE: u32 = 128;
const TINY: u32 = 2;

/// The large item and the tiles of the large grid are this square size: each decoded frame
/// (384 KiB at 8-bit 4:2:0), and the canvas even more, is above the uncharged slack at 4 threads
/// (320 KiB).
const BIG_TILE: u32 = 512;

fn av01(id: u8, data: &[u8], props: &[u8]) -> Item {
    Item {
        id,
        kind: b"av01",
        data: data.to_vec(),
        props: props.to_vec(),
        split: false,
    }
}

fn refs(kind: &[u8; 4], from: u8, to: &[u8]) -> Vec<u8> {
    let mut v = [&[0, from][..], &(to.len() as u16).to_be_bytes()].concat();
    for t in to {
        v.extend_from_slice(&[0, *t]);
    }
    bx(kind, &v)
}

/// The test files: name, file, and the thread counts up to which the result must be the same
/// as at one thread. That holds for a grid of at least as many tiles as threads (each tile
/// decodes with one thread, so its memory does not depend on the thread count), and for a single
/// item: it decodes with all the threads, and a frame whose threaded decode is refused decodes
/// again on one thread, so its result is the one of a serial decode.
fn files() -> Vec<(&'static str, Vec<u8>, usize)> {
    let (data, _) = decodable_unit(0, (64, 32));
    let plain = build(&[av01(1, &data, &[1, 0x82])], &props(), &[]);
    let alpha = build(
        &[av01(1, &data, &[1, 0x82]), av01(2, &data, &[1, 0x82, 3])],
        &props(),
        &[refs(b"auxl", 2, &[1])],
    );
    // 128x64 from four 64x32 tiles.
    let grid = |tiles: Vec<Item>| {
        let mut items = vec![Item {
            id: 1,
            kind: b"grid",
            data: vec![0, 0, 1, 1, 0, 128, 0, 64],
            props: vec![4],
            split: false,
        }];
        items.extend(tiles);
        build(&items, &props(), &[refs(b"dimg", 1, &[2, 3, 4, 5])])
    };
    let small_grid = grid((2..=5).map(|id| av01(id, &data, &[1, 0x82])).collect());
    // The same grid with every tile padded to 64 KiB and stored as two extents, so each tile's
    // bytes are copied under the item-read share of `max_alloc`.
    let padded = [data.clone(), padding(64 << 10)].concat();
    let split_grid = grid(
        (2..=5)
            .map(|id| Item {
                split: true,
                ..av01(id, &padded, &[1, 0x82])
            })
            .collect(),
    );
    // 128x32 from two of those split tiles: one tile decodes beside tile 0's canvas, so its read
    // must get the whole item-read share, as in a serial decode.
    let split_pair = build(
        &[
            Item {
                id: 1,
                kind: b"grid",
                data: vec![0, 0, 0, 1, 0, 128, 0, 32],
                props: vec![9],
                split: false,
            },
            Item {
                split: true,
                ..av01(2, &padded, &[1, 0x82])
            },
            Item {
                split: true,
                ..av01(3, &padded, &[1, 0x82])
            },
        ],
        &props(),
        &[refs(b"dimg", 1, &[2, 3])],
    );
    let (tile, _) = decodable_unit(0, (u64::from(BIG_TILE), u64::from(BIG_TILE)));
    // Rotated by 90 degrees, so the transform makes new planes.
    let big = build(&[av01(1, &tile, &[7, 0x82, 0x85])], &props(), &[]);
    // Every tile is rotated before it is pasted; the grid is mirrored after.
    let big_grid = {
        let side = (2 * BIG_TILE) as u16;
        let mut items = vec![Item {
            id: 1,
            kind: b"grid",
            data: [&[0, 0, 1, 1][..], &side.to_be_bytes(), &side.to_be_bytes()].concat(),
            props: vec![8, 0x86],
            split: false,
        }];
        items.extend((2..=5).map(|id| av01(id, &tile, &[7, 0x82, 0x85])));
        build(&items, &props(), &[refs(b"dimg", 1, &[2, 3, 4, 5])])
    };
    // 16384 references to one 2x2 tile (rotated, so each tile is transformed), mirrored after.
    let hostile = {
        let (tiny, _) = decodable_unit(0, (u64::from(TINY), u64::from(TINY)));
        let side = (HOSTILE * TINY) as u16;
        let n = (HOSTILE - 1) as u8;
        let grid = Item {
            id: 1,
            kind: b"grid",
            data: [&[0, 0, n, n][..], &side.to_be_bytes(), &side.to_be_bytes()].concat(),
            props: vec![11, 0x86],
            split: false,
        };
        let tiles = vec![2; (HOSTILE * HOSTILE) as usize];
        build(
            &[grid, av01(2, &tiny, &[10, 0x82, 0x85])],
            &props(),
            &[refs(b"dimg", 1, &tiles)],
        )
    };
    vec![
        ("plain", plain, 4),
        ("alpha", alpha, 4),
        ("grid", small_grid, 4),
        ("split grid", split_grid, 4),
        ("split grid 1x2", split_pair, 2),
        ("big", big, 4),
        ("big grid", big_grid, 4),
        ("hostile grid", hostile, 4),
    ]
}

/// `max_alloc` values from `from` to 64 MiB, a quarter apart, so the boundary between fitting
/// and not is crossed in small steps.
fn sweep(from: usize) -> Vec<usize> {
    let mut v = vec![from];
    while let Some(&m) = v.last()
        && m < 64 << 20
    {
        v.push(m + m / 4);
    }
    v
}

/// The sweep of [`files`] entry `name` at 1, 2 and 4 threads. The peak is checked against what
/// `decode_avif` may hold: its ledger (three quarters of `max_alloc`), what the `isobmff` parse
/// held (measured here; the parse result lives through the call), the item-byte copies (only a
/// split item is copied, within an eighth of `max_alloc`) and the uncharged buffers of each
/// decoder. So a frame or a transform output that is not booked shows up once it is larger than
/// that slack.
fn decode_avif_keeps_the_heap_within_max_alloc(name: &str) {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let (_, file, same) = files()
        .into_iter()
        .find(|f| f.0 == name)
        .expect("a test file");
    let cancel = AtomicBool::new(false);
    let options = Options {
        apply_transforms: true,
        ..Options::default()
    };
    let decode = |max_alloc: usize, max_threads: usize| {
        let limits = Limits {
            max_alloc,
            max_threads,
            ..Limits::default()
        };
        measured(|| av1::decode_avif(&file, &options, &limits, &cancel))
    };
    let parse = |max_alloc: usize, max_threads: usize| {
        let ilimits = av1::isobmff::Limits {
            max_alloc: max_alloc / 8,
            max_threads,
            max_pixels: Limits::default().max_pixels,
            ..av1::isobmff::Limits::default()
        };
        measured(|| av1::isobmff::parse(&file, &ilimits, &cancel)).1
    };
    let (want, _) = decode(64 << 20, 1);
    let want: AvifImage = want.unwrap_or_else(|e| panic!("{name}: reference decode: {e:?}"));
    let want_frames: Vec<&Frame> = std::iter::once(&want.image).chain(&want.alpha).collect();
    let (mut fitted, mut runs) = (0, 0);
    // The hostile grid decodes 16384 tiles per run and fits well below 16 MiB.
    let top = if name == "hostile grid" {
        16 << 20
    } else {
        64 << 20
    };
    for max_alloc in sweep(16 << 10).into_iter().take_while(|&m| m <= top) {
        let mut first: Option<Result<AvifImage, Error>> = None;
        for threads in [1, 2, 4] {
            let ledger = (max_alloc - 2 * (max_alloc / 8)) as isize;
            let reads = if name.starts_with("split grid") {
                (max_alloc / 8) as isize
            } else {
                0
            };
            let bound =
                ledger + parse(max_alloc, threads) + reads + UNCHARGED * (threads as isize + 1);
            let (r, peak) = decode(max_alloc, threads);
            runs += 1;
            assert!(
                peak <= bound,
                "{name} max_alloc {max_alloc} threads {threads}: {peak} bytes held, bound {bound}, result {:?}",
                r.as_ref().map(|_| ())
            );
            match &r {
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
            if threads > same {
                continue;
            }
            match &first {
                None => first = Some(r),
                Some(f) => assert!(
                    *f == r,
                    "{name} max_alloc {max_alloc}: {threads} threads give {:?}, 1 thread {:?}",
                    r.as_ref().map(|_| ()),
                    f.as_ref().map(|_| ())
                ),
            }
        }
    }
    // Some limits fit and some do not, so both paths were checked.
    assert!(
        fitted > 0 && fitted < runs,
        "{name}: {fitted} of {runs} fitted"
    );
}

#[test]
fn decode_avif_plain_keeps_the_heap_within_max_alloc() {
    decode_avif_keeps_the_heap_within_max_alloc("plain");
}

#[test]
fn decode_avif_alpha_keeps_the_heap_within_max_alloc() {
    decode_avif_keeps_the_heap_within_max_alloc("alpha");
}

#[test]
fn decode_avif_grid_keeps_the_heap_within_max_alloc() {
    decode_avif_keeps_the_heap_within_max_alloc("grid");
}

#[test]
fn decode_avif_split_grid_keeps_the_heap_within_max_alloc() {
    decode_avif_keeps_the_heap_within_max_alloc("split grid");
}

#[test]
fn decode_avif_split_grid_1x2_keeps_the_heap_within_max_alloc() {
    decode_avif_keeps_the_heap_within_max_alloc("split grid 1x2");
}

#[test]
fn decode_avif_big_keeps_the_heap_within_max_alloc() {
    decode_avif_keeps_the_heap_within_max_alloc("big");
}

#[test]
fn decode_avif_big_grid_keeps_the_heap_within_max_alloc() {
    decode_avif_keeps_the_heap_within_max_alloc("big grid");
}

#[test]
fn decode_temporal_units_keeps_the_heap_within_max_alloc() {
    let _turn = TURN.lock().unwrap_or_else(|e| e.into_inner());
    let cancel = AtomicBool::new(false);
    // Eight key frames far above the slack, decoded several at once.
    let (data, _) = decodable_unit(0, (u64::from(BIG_TILE), u64::from(BIG_TILE / 2)));
    let units: Vec<&[u8]> = vec![&data[..]; 8];
    let mut d = Decoder::new(Options::default(), limits(64 << 20));
    let mut want = Vec::new();
    for u in &units {
        want.extend(
            d.decode_temporal_unit(u, Format::Section5, &cancel)
                .unwrap(),
        );
    }
    let (mut fitted, mut runs) = (0, 0);
    for max_alloc in sweep(64 << 10) {
        let batch = |threads: usize| {
            let limits = Limits {
                max_alloc,
                max_threads: threads,
                ..Limits::default()
            };
            let mut d = Decoder::new(Options::default(), limits);
            measured(|| d.decode_temporal_units(&units, Format::Section5, &cancel))
        };
        let (one, _) = batch(1);
        for threads in [2, 4] {
            let (r, peak) = batch(threads);
            runs += 1;
            assert!(
                r == one,
                "max_alloc {max_alloc} threads {threads}: {:?}, 1 thread {:?}",
                r.as_ref().map(|f| f.len()),
                one.as_ref().map(|f| f.len())
            );
            assert!(
                peak <= max_alloc as isize + UNCHARGED * (threads as isize + 1),
                "max_alloc {max_alloc} threads {threads}: {peak} bytes held, result {:?}",
                r.as_ref().map(|f| f.len())
            );
            match r {
                Ok(got) => {
                    fitted += 1;
                    assert!(got == want, "max_alloc {max_alloc} threads {threads}");
                }
                Err(e) => assert!(
                    matches!(e, Error::AllocLimit | Error::OutOfMemory),
                    "max_alloc {max_alloc} threads {threads}: {e:?}"
                ),
            }
        }
    }
    assert!(fitted > 0 && fitted < runs, "{fitted} of {runs} fitted");
}

#[test]
fn decode_avif_hostile_grid_keeps_the_heap_within_max_alloc() {
    decode_avif_keeps_the_heap_within_max_alloc("hostile grid");
}
