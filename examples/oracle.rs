//! Decodes every AV1 file in a directory and writes the results next to each other, for
//! comparing with other decoders.
//!
//! ```text
//! oracle [--info] [--null] [--threads N] [--no-filmgrain] [--op N] [--all-layers] <in-dir> <out-dir>
//! ```
//!
//! Inputs (sorted by name): `.ivf` (each IVF frame is one temporal unit), `.obu`, `.av1` and
//! `.bin` (low-overhead OBU stream), `.annexb` (Annex B), `.avif` (see below). Other files are
//! skipped. For each
//! `<stem>` the output directory gets:
//!
//! - `<stem>.yuv`: every output frame, planes Y, U, V one after the other, visible size only;
//!   8-bit samples as bytes, deeper samples as little-endian `u16` (the layout of `dav1d -o
//!   x.yuv`); monochrome writes Y only.
//! - `<stem>.frames`: one line per output frame, `W H bitdepth layout` (`420`, `422`, `444`,
//!   `400`).
//! - `<stem>.err`: only when the decode stopped on an error: the `Error::class()` token (the
//!   frames before the error are still written; `inter` is the expected stop of a stream with
//!   inter frames).
//!
//! An `.avif` input goes through `av1::decode_avif`: `<stem>.yuv` is the primary image (grid
//! assembled and cropped; transforms are not applied), `<stem>.frames` has its one line, and
//! `<stem>.alpha.yuv` the alpha image when there is one. `--info` does not apply to `.avif`.
//!
//! With `--null` the samples are decoded but no `.yuv` is written (the bench and profile runs).
//!
//! With `--info` the decode is replaced by `av1::probe` and `<stem>.info` has one line per
//! frame: `type show W H frameW bitdepth layout` (`show` is 1 when the frame is output).
//!
//! Exit status: 0 when every file was processed (whatever the decode result), 2 on a usage or
//! I/O error.

use av1::{
    Decoder, Error, Format, Frame, Limits, Options, PlaneData, decode_avif, probe,
    split_temporal_units,
};
use std::fmt::Write as _;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

struct Args {
    info: bool,
    null: bool,
    threads: Option<usize>,
    film_grain: bool,
    operating_point: usize,
    all_layers: bool,
    in_dir: PathBuf,
    out_dir: PathBuf,
}

#[derive(Clone, Copy)]
enum Kind {
    Ivf,
    Obu(Format),
    Avif,
}

fn parse_args() -> Result<Args, String> {
    let mut args = Args {
        info: false,
        null: false,
        threads: None,
        film_grain: true,
        operating_point: 0,
        all_layers: false,
        in_dir: PathBuf::new(),
        out_dir: PathBuf::new(),
    };
    let mut dirs = Vec::new();
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut number = |name: &str| -> Result<usize, String> {
            it.next()
                .and_then(|v| v.parse().ok())
                .ok_or_else(|| format!("{name} needs a number"))
        };
        match a.as_str() {
            "--info" => args.info = true,
            "--null" => args.null = true,
            "--no-filmgrain" => args.film_grain = false,
            "--all-layers" => args.all_layers = true,
            "--threads" => args.threads = Some(number("--threads")?),
            "--op" => args.operating_point = number("--op")?,
            s if s.starts_with("--") => return Err(format!("unknown option {s}")),
            _ => dirs.push(PathBuf::from(a)),
        }
    }
    let [in_dir, out_dir] =
        <[PathBuf; 2]>::try_from(dirs).map_err(|_| "expected <in-dir> <out-dir>".to_string())?;
    args.in_dir = in_dir;
    args.out_dir = out_dir;
    Ok(args)
}

fn kind_of(path: &Path) -> Option<Kind> {
    match path.extension()?.to_str()? {
        "ivf" => Some(Kind::Ivf),
        "obu" | "av1" | "bin" => Some(Kind::Obu(Format::Section5)),
        "annexb" => Some(Kind::Obu(Format::AnnexB)),
        "avif" => Some(Kind::Avif),
        _ => None,
    }
}

/// The payload of each IVF frame: a 32 byte file header ("DKIF"), then per frame a 12 byte
/// header (u32 size, u64 pts, little-endian) and the data.
fn ivf_frames(data: &[u8]) -> Result<Vec<&[u8]>, Error> {
    if data.len() < 32 || &data[..4] != b"DKIF" {
        return Err(Error::Invalid("not an IVF file"));
    }
    let header_len = usize::from(u16::from_le_bytes([data[6], data[7]]));
    let mut pos = header_len.max(32);
    let mut frames = Vec::new();
    while pos < data.len() {
        let size = data
            .get(pos..pos + 4)
            .map(|b| u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as usize)
            .ok_or(Error::Truncated)?;
        pos += 12;
        let end = pos.checked_add(size).ok_or(Error::Truncated)?;
        frames.push(data.get(pos..end).ok_or(Error::Truncated)?);
        pos = end;
    }
    Ok(frames)
}

fn layout(mono: bool, ssx: bool, ssy: bool) -> &'static str {
    match (mono, ssx, ssy) {
        (true, _, _) => "400",
        (false, true, true) => "420",
        (false, true, false) => "422",
        (false, false, false) => "444",
        (false, false, true) => "440",
    }
}

fn append_planes(yuv: &mut Vec<u8>, f: &Frame) {
    for p in &f.planes {
        match &p.data {
            PlaneData::U8(v) => yuv.extend_from_slice(v),
            PlaneData::U16(v) => yuv.extend(v.iter().flat_map(|s| s.to_le_bytes())),
        }
    }
}

/// `<stem>.<ext>`. Not `Path::with_extension`, which would replace a dot part of the stem
/// (`fox.yuv420` or `x.item1`).
fn out(stem: &Path, ext: &str) -> PathBuf {
    let mut s = stem.as_os_str().to_owned();
    s.push(".");
    s.push(ext);
    PathBuf::from(s)
}

fn write_frames(frames: &[Frame], stem: &Path, null: bool) -> std::io::Result<()> {
    let mut yuv = Vec::new();
    let mut lines = String::new();
    for f in frames {
        if !null {
            append_planes(&mut yuv, f);
        }
        let l = layout(f.mono_chrome, f.subsampling_x, f.subsampling_y);
        let _ = writeln!(lines, "{} {} {} {}", f.width, f.height, f.bit_depth, l);
    }
    if !null {
        fs::write(out(stem, "yuv"), yuv)?;
    }
    fs::write(out(stem, "frames"), lines)
}

fn write_err(stem: &Path, e: &Error) -> std::io::Result<()> {
    fs::write(out(stem, "why"), e.to_string())?;
    fs::write(out(stem, "err"), e.class())
}

/// Decodes or probes one file; only I/O errors are returned, decode errors go to `.err`.
fn run_file(
    path: &Path,
    kind: Kind,
    args: &Args,
    limits: &Limits,
    out_dir: &Path,
) -> std::io::Result<()> {
    let data = fs::read(path)?;
    let Some(name) = path.file_stem() else {
        return Ok(());
    };
    let stem = out_dir.join(name);
    if let Kind::Avif = kind {
        let options = Options {
            operating_point: args.operating_point,
            all_layers: args.all_layers,
            film_grain: args.film_grain,
            ..Options::default()
        };
        return match decode_avif(&data, &options, limits, &AtomicBool::new(false)) {
            Ok(avif) => {
                write_frames(std::slice::from_ref(&avif.image), &stem, args.null)?;
                if let Some(alpha) = avif.alpha.as_ref().filter(|_| !args.null) {
                    let mut yuv = Vec::new();
                    append_planes(&mut yuv, alpha);
                    fs::write(out(&stem, "alpha.yuv"), yuv)?;
                }
                Ok(())
            }
            Err(e) => {
                write_frames(&[], &stem, args.null)?;
                write_err(&stem, &e)
            }
        };
    }
    let units: Result<(Format, Vec<&[u8]>), Error> = match kind {
        Kind::Ivf => ivf_frames(&data).map(|u| (Format::Section5, u)),
        Kind::Obu(format) => split_temporal_units(&data, format, limits).map(|u| (format, u)),
        Kind::Avif => return Ok(()),
    };
    let (format, units) = match units {
        Ok(u) => u,
        Err(e) => {
            if args.info {
                fs::write(out(&stem, "info"), "")?;
            } else {
                write_frames(&[], &stem, args.null)?;
            }
            return write_err(&stem, &e);
        }
    };
    if args.info {
        // An IVF file's frames are whole temporal units; joined they are a low-overhead stream.
        let joined: Vec<u8>;
        let stream = match kind {
            Kind::Ivf => {
                joined = units.concat();
                &joined[..]
            }
            Kind::Obu(_) | Kind::Avif => &data[..],
        };
        let mut text = String::new();
        let stopped = match probe(stream, format, limits) {
            Ok(p) => {
                for f in &p.frames {
                    let show = u8::from(f.show_frame || f.show_existing_frame);
                    let l = layout(f.mono_chrome, f.subsampling_x, f.subsampling_y);
                    let _ = writeln!(
                        text,
                        "{} {} {} {} {} {} {}",
                        f.frame_type, show, f.width, f.height, f.frame_width, f.bit_depth, l
                    );
                }
                p.stopped
            }
            Err(e) => Some(e),
        };
        fs::write(out(&stem, "info"), text)?;
        return match stopped {
            Some(e) => write_err(&stem, &e),
            None => Ok(()),
        };
    }
    let options = Options {
        operating_point: args.operating_point,
        all_layers: args.all_layers,
        film_grain: args.film_grain,
        ..Options::default()
    };
    let cancel = AtomicBool::new(false);
    let mut decoder = Decoder::new(options, *limits);
    let mut frames = Vec::new();
    let mut stopped = None;
    for unit in units {
        match decoder.decode_temporal_unit(unit, format, &cancel) {
            Ok(f) => frames.extend(f),
            Err(e) => {
                frames.extend(decoder.take_frames());
                stopped = Some(e);
                break;
            }
        }
    }
    write_frames(&frames, &stem, args.null)?;
    match stopped {
        Some(e) => write_err(&stem, &e),
        None => Ok(()),
    }
}

fn run(args: &Args) -> std::io::Result<()> {
    let mut limits = Limits::default();
    if let Some(n) = args.threads {
        limits.max_threads = n;
    }
    fs::create_dir_all(&args.out_dir)?;
    let mut files: Vec<PathBuf> = fs::read_dir(&args.in_dir)?
        .map(|e| e.map(|e| e.path()))
        .collect::<Result<_, _>>()?;
    files.sort();
    for path in files {
        if let Some(kind) = kind_of(&path) {
            run_file(&path, kind, args, &limits, &args.out_dir)?;
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args = match parse_args() {
        Ok(a) => a,
        Err(msg) => {
            eprintln!("oracle: {msg}");
            eprintln!(
                "usage: oracle [--info] [--null] [--threads N] [--no-filmgrain] [--op N] [--all-layers] <in-dir> <out-dir>"
            );
            return ExitCode::from(2);
        }
    };
    match run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("oracle: {e}");
            ExitCode::from(2)
        }
    }
}
