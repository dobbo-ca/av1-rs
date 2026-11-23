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

fn run(out_dir: &Path, files: &[PathBuf]) -> std::io::Result<()> {
    fs::create_dir_all(out_dir)?;
    for path in files {
        let data = fs::read(path)?;
        let name = path.file_stem().unwrap_or_default().to_string_lossy();
        match items(&data) {
            Ok(found) => {
                for (id, stream) in &found {
                    fs::write(out_dir.join(format!("{name}.item{id}.obu")), stream)?;
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
