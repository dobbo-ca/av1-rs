//! CI-only extraction tool: cuts the AV1 streams out of AVIF files so dav1d and aomdec, which
//! cannot read AVIF here, can decode the same bytes as av1-rs (see docs/parity.md). It is not
//! part of the library.
//!
//! ```text
//! avifx <out-dir> <file.avif>...
//! ```
//!
//! For every AV1 coded item reachable from the primary image (the primary itself, its grid
//! tiles, its alpha item and the alpha's grid tiles) it writes
//! `<out-dir>/<file stem>.item<id>.obu`: the `av1C` configOBUs followed by the item data, a
//! Section 5 stream. Each file gets one line, `<file>: <n> items`. A container error is printed
//! as `<file>: <error>` and the next file goes on.
//!
//! Exit status: 0 unless reading or writing a file fails (2, also for a usage error).

use av1::isobmff::{self, Av1c, Codec, CodecConfig, Heif, ImageKind, Limits};
use std::fs;
use std::path::{Path, PathBuf};
use std::process::ExitCode;
use std::sync::atomic::AtomicBool;

/// Adds the AV1 coded items of image `id` (itself, its tiles, its identity input) to `out`.
fn collect(
    heif: &Heif,
    id: u32,
    depth: u8,
    out: &mut Vec<(u32, Av1c)>,
) -> Result<(), isobmff::Error> {
    if depth > 8 {
        return Err(isobmff::Error::Malformed("nesting"));
    }
    match heif.image(id)?.kind {
        ImageKind::Coded {
            codec: Codec::Av1,
            config: CodecConfig::Av1(c),
        } => {
            if !out.iter().any(|(i, _)| *i == id) {
                out.push((id, c));
            }
        }
        ImageKind::Grid(g) => {
            for t in g.tiles {
                collect(heif, t, depth + 1, out)?;
            }
        }
        ImageKind::Identity { input } => collect(heif, input, depth + 1, out)?,
        _ => {}
    }
    Ok(())
}

/// The items of one file, or the container error that stopped the walk.
fn items(file: &[u8]) -> Result<Vec<(u32, Vec<u8>)>, isobmff::Error> {
    let limits = Limits::default();
    let cancel = AtomicBool::new(false);
    let heif = isobmff::parse(file, &limits, &cancel)?;
    let primary = heif.primary_image()?;
    let mut ids = Vec::new();
    collect(&heif, primary.id, 0, &mut ids)?;
    if let Some(alpha) = primary.alpha {
        collect(&heif, alpha, 0, &mut ids)?;
    }
    let mut found = Vec::new();
    for (id, c) in ids {
        let mut stream = Heif::slice(file, c.config_obus)
            .ok_or(isobmff::Error::Truncated)?
            .to_vec();
        stream.extend_from_slice(&heif.item_bytes(file, id, &limits, &cancel)?);
        found.push((id, stream));
    }
    Ok(found)
}

/// One temporal unit for dav1d and aomdec: a temporal delimiter, then every OBU of `stream`
/// except temporal delimiters (dav1d's Section 5 probe wants exactly one, first). Every OBU must
/// carry obu_size (AVIF requires it); a stream that does not parse is returned behind the
/// delimiter as it is.
fn temporal_unit(stream: &[u8]) -> Vec<u8> {
    let mut out = vec![0x12, 0x00];
    let mut kept = Vec::with_capacity(stream.len());
    let mut pos = 0;
    while pos < stream.len() {
        let h = stream[pos];
        let ext = usize::from((h >> 2) & 1);
        let mut p = pos + 1 + ext;
        if h & 2 == 0 || p > stream.len() {
            kept.clear();
            kept.extend_from_slice(stream);
            break;
        }
        let mut size = 0usize;
        let mut shift = 0;
        loop {
            let Some(&b) = stream.get(p) else { break };
            p += 1;
            size |= usize::from(b & 0x7f) << shift;
            shift += 7;
            if b & 0x80 == 0 || shift > 56 {
                break;
            }
        }
        let end = p.saturating_add(size).min(stream.len());
        if (h >> 3) & 15 != 2 {
            kept.extend_from_slice(&stream[pos..end]);
        }
        pos = end;
    }
    out.extend_from_slice(&kept);
    out
}

fn run(out_dir: &Path, files: &[PathBuf]) -> std::io::Result<()> {
    fs::create_dir_all(out_dir)?;
    for path in files {
        let data = fs::read(path)?;
        let name = path.file_stem().unwrap_or_default().to_string_lossy();
        match items(&data) {
            Ok(found) => {
                for (id, stream) in &found {
                    let obu = temporal_unit(stream);
                    fs::write(out_dir.join(format!("{name}.item{id}.obu")), obu)?;
                }
                println!("{}: {} items", path.display(), found.len());
            }
            Err(e) => println!("{}: {e}", path.display()),
        }
    }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let Some((out_dir, files)) = args.split_first() else {
        eprintln!("usage: avifx <out-dir> <file.avif>...");
        return ExitCode::from(2);
    };
    let files: Vec<PathBuf> = files.iter().map(PathBuf::from).collect();
    match run(Path::new(out_dir), &files) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("avifx: {e}");
            ExitCode::from(2)
        }
    }
}
