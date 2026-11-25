//! API surface tests that need no corpus. Whole-frame decodes are covered by the parity gate.

use av1::{Decoder, Error, Format, Limits, Options, decode, probe};
use std::sync::atomic::AtomicBool;

/// A low-overhead OBU with a one byte size field.
fn obu(obu_type: u8, payload: &[u8]) -> Vec<u8> {
    let mut v = vec![(obu_type << 3) | 2, payload.len() as u8];
    v.extend_from_slice(payload);
    v
}

const OBU_TEMPORAL_DELIMITER: u8 = 2;
const OBU_SEQUENCE_HEADER: u8 = 1;

#[test]
fn empty_stream_has_no_frames() {
    let cancel = AtomicBool::new(false);
    let frames = decode(
        &[],
        Format::Section5,
        &Options::default(),
        &Limits::default(),
        &cancel,
    )
    .unwrap();
    assert!(frames.is_empty());
    assert!(
        probe(&[], Format::Section5, &Limits::default())
            .unwrap()
            .frames
            .is_empty()
    );
}

#[test]
fn truncated_sequence_header_is_reported() {
    // seq_profile 0, still_picture 1, reduced_still_picture_header 1, level 4, then nothing.
    let data = [
        obu(OBU_TEMPORAL_DELIMITER, &[]),
        obu(OBU_SEQUENCE_HEADER, &[0x19, 0x00]),
    ]
    .concat();
    let p = probe(&data, Format::Section5, &Limits::default()).unwrap();
    assert_eq!(p.stopped, Some(Error::Truncated));
    assert!(p.frames.is_empty());
}

#[test]
fn single_thread_decoder_reports_the_same_error() {
    let data = [
        obu(OBU_TEMPORAL_DELIMITER, &[]),
        obu(OBU_SEQUENCE_HEADER, &[0x19, 0x00]),
    ]
    .concat();
    let cancel = AtomicBool::new(false);
    for max_threads in [0, 1, 4] {
        let limits = Limits {
            max_threads,
            ..Limits::default()
        };
        let mut d = Decoder::new(Options::default(), limits);
        assert_eq!(
            d.decode_temporal_unit(&data, Format::Section5, &cancel),
            Err(Error::Truncated)
        );
    }
}

fn bx(kind: &[u8; 4], body: &[u8]) -> Vec<u8> {
    let mut v = ((body.len() as u32) + 8).to_be_bytes().to_vec();
    v.extend_from_slice(kind);
    v.extend_from_slice(body);
    v
}

/// A minimal AVIF: primary `av01` item 1 (`ispe` 10 x 20, `av1C`) whose three data bytes are not
/// an AV1 stream, and an Exif item 2.
fn tiny_avif() -> Vec<u8> {
    let ftyp = bx(b"ftyp", b"avif\0\0\0\0avifmif1");
    let build = |mdat_at: u32| {
        let hdlr = bx(b"hdlr", &[&[0; 8][..], b"pict", &[0; 13]].concat());
        let pitm = bx(b"pitm", &[0, 0, 0, 0, 0, 1]);
        let infe1 = bx(
            b"infe",
            &[&[2, 0, 0, 0, 0, 1, 0, 0][..], b"av01", &[0]].concat(),
        );
        let infe2 = bx(
            b"infe",
            &[&[2, 0, 0, 1, 0, 2, 0, 0][..], b"Exif", &[0]].concat(),
        );
        let iinf = bx(b"iinf", &[&[0, 0, 0, 0, 0, 2][..], &infe1, &infe2].concat());
        let mut iloc = vec![0, 0, 0, 0, 0x44, 0, 0, 2];
        iloc.extend_from_slice(&[0, 1, 0, 0, 0, 1]);
        iloc.extend_from_slice(&mdat_at.to_be_bytes());
        iloc.extend_from_slice(&3u32.to_be_bytes());
        iloc.extend_from_slice(&[0, 2, 0, 0, 0, 1]);
        iloc.extend_from_slice(&(mdat_at + 3).to_be_bytes());
        iloc.extend_from_slice(&2u32.to_be_bytes());
        let iloc = bx(b"iloc", &iloc);
        let ispe = bx(b"ispe", &[0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 20]);
        let av1c = bx(b"av1C", &[0x81, 0, 0x4c, 0, 0x0a]);
        let ipco = bx(b"ipco", &[ispe, av1c].concat());
        let ipma = bx(b"ipma", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 0x82]);
        let iprp = bx(b"iprp", &[ipco, ipma].concat());
        bx(
            b"meta",
            &[&[0; 4][..], &hdlr, &pitm, &iinf, &iloc, &iprp].concat(),
        )
    };
    let at = (ftyp.len() + build(0).len() + 8) as u32;
    let mdat = bx(b"mdat", &[1, 2, 3, b'M', b'M']);
    [ftyp, build(at), mdat].concat()
}

#[test]
fn avif_item_matches_the_primary_image() {
    let file = tiny_avif();
    let cancel = AtomicBool::new(false);
    let (o, l) = (Options::default(), Limits::default());
    let heif = av1::isobmff::parse(&file, &av1::isobmff::Limits::default(), &cancel).unwrap();
    // The item bytes are not AV1, so both paths fail the same way (the repo has no decodable
    // AVIF fixture without the parity corpus).
    let primary = av1::decode_avif(&file, &o, &l, &cancel).map(|a| a.image);
    let item = av1::decode_avif_item(&heif, &file, heif.primary, &o, &l, &cancel);
    assert_eq!(primary, item);
}

#[test]
fn avif_item_with_a_bad_id_is_an_error() {
    let file = tiny_avif();
    let cancel = AtomicBool::new(false);
    let heif = av1::isobmff::parse(&file, &av1::isobmff::Limits::default(), &cancel).unwrap();
    let (o, l) = (Options::default(), Limits::default());
    for id in [0, 9, u32::MAX] {
        assert!(matches!(
            av1::decode_avif_item(&heif, &file, id, &o, &l, &cancel),
            Err(Error::Container(_))
        ));
    }
    // An Exif item is not an image.
    assert!(av1::decode_avif_item(&heif, &file, 2, &o, &l, &cancel).is_err());
}

// Item bytes are read under an eighth of `max_alloc` (shared by the grid workers): a cap that
// leaves less than the item size is an error, not a decode.
#[test]
fn avif_item_bytes_over_the_split_are_an_error() {
    let file = tiny_avif();
    let cancel = AtomicBool::new(false);
    let heif = av1::isobmff::parse(&file, &av1::isobmff::Limits::default(), &cancel).unwrap();
    let o = Options::default();
    let l = Limits {
        max_alloc: 16,
        ..Limits::default()
    };
    assert!(av1::decode_avif_item(&heif, &file, heif.primary, &o, &l, &cancel).is_err());
}

/// An AVIF whose primary `av01` item is `n` zero bytes in two extents, so `isobmff` copies it.
fn two_extent_avif(n: u32) -> Vec<u8> {
    let ftyp = bx(b"ftyp", b"avif\0\0\0\0avifmif1");
    let build = |mdat_at: u32| {
        let hdlr = bx(b"hdlr", &[&[0; 8][..], b"pict", &[0; 13]].concat());
        let pitm = bx(b"pitm", &[0, 0, 0, 0, 0, 1]);
        let infe = bx(
            b"infe",
            &[&[2, 0, 0, 0, 0, 1, 0, 0][..], b"av01", &[0]].concat(),
        );
        let iinf = bx(b"iinf", &[&[0, 0, 0, 0, 0, 1][..], &infe].concat());
        let mut iloc = vec![0, 0, 0, 0, 0x44, 0, 0, 1];
        iloc.extend_from_slice(&[0, 1, 0, 0, 0, 2]);
        for (at, len) in [(mdat_at, n / 2), (mdat_at + n / 2, n - n / 2)] {
            iloc.extend_from_slice(&at.to_be_bytes());
            iloc.extend_from_slice(&len.to_be_bytes());
        }
        let iloc = bx(b"iloc", &iloc);
        let ispe = bx(b"ispe", &[0, 0, 0, 0, 0, 0, 0, 10, 0, 0, 0, 20]);
        let av1c = bx(b"av1C", &[0x81, 0, 0x4c, 0, 0x0a]);
        let ipco = bx(b"ipco", &[ispe, av1c].concat());
        let ipma = bx(b"ipma", &[0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 2, 1, 0x82]);
        let iprp = bx(b"iprp", &[ipco, ipma].concat());
        bx(
            b"meta",
            &[&[0; 4][..], &hdlr, &pitm, &iinf, &iloc, &iprp].concat(),
        )
    };
    let at = (ftyp.len() + build(0).len() + 8) as u32;
    let mdat = bx(b"mdat", &vec![0; n as usize]);
    [ftyp, build(at), mdat].concat()
}

// `decode_avif` reads the item bytes under an eighth of `max_alloc`, outside the ledger, so the
// copy and the ledger together stay within `max_alloc`.
#[test]
fn avif_item_bytes_over_an_eighth_of_max_alloc_are_a_container_error() {
    let file = two_extent_avif(4000);
    let cancel = AtomicBool::new(false);
    let o = Options::default();
    let at = |max_alloc| {
        let l = Limits {
            max_alloc,
            ..Limits::default()
        };
        av1::decode_avif(&file, &o, &l, &cancel).map(|a| a.image)
    };
    assert!(matches!(at(16 << 10), Err(Error::Container(_))));
    // With an eighth above the item size the bytes are read, and the decode fails on them.
    let r = at(64 << 10);
    assert!(
        r.is_err() && !matches!(r, Err(Error::Container(_))),
        "{r:?}"
    );
}

#[test]
fn decode_charges_the_unit_list_to_max_alloc() {
    // Every Annex B zero byte is an empty temporal unit: the list is 16 bytes per input byte.
    let limits = Limits {
        max_alloc: 1 << 10,
        ..Limits::default()
    };
    let r = decode(
        &vec![0u8; 1 << 12],
        Format::AnnexB,
        &Options::default(),
        &limits,
        &AtomicBool::new(false),
    );
    assert_eq!(r.err(), Some(Error::AllocLimit));
}
