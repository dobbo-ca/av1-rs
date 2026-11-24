# Dependencies

The crate depends on `isobmff` (our container parser, pinned by rev) and otherwise on std only. The fuzz crate (`tests/fuzz`) uses `libfuzzer-sys`.

## Sources used as guides

- The AV1 specification (`github.com/AOMediaCodec/av1-spec`, Copyright 2018 The Alliance for
  Open Media): the source of truth; every table is transcribed from it (NOTICE).
- dav1d and rav1d (BSD-2-Clause): read as design guides. Ported parts are listed here and in
  NOTICE, and named in the file that carries them:
  - `src/simd/cdef/{mod,avx2,neon}.rs`: the CDEF filter design of dav1d's `src/cdef_tmpl.c`
    and `src/x86/cdef*.asm` (a padded 16-bit block with the sentinel `0x8000` for unavailable
    taps; signed max and unsigned min skip it). Copyright © 2018, VideoLAN and dav1d authors;
    Copyright © 2018, Two Orioles, LLC.
  - `src/symbol.rs`: the symbol reader's window, refill, renormalisation and `read_bool` fast
    path, after dav1d's `src/msac.c`.
  - `src/coef.rs`: the padded levels array behind the `coeff_base` and `coeff_br` contexts,
    after dav1d's `src/recon_tmpl.c` (`decode_coefs`).
