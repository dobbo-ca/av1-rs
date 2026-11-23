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
