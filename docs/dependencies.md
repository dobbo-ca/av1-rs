# Dependencies

The crate has no dependencies (std only). The fuzz crate (`tests/fuzz`) uses `libfuzzer-sys`.

## Sources used as guides

- The AV1 specification (`github.com/AOMediaCodec/av1-spec`, Copyright 2018 The Alliance for
  Open Media): the source of truth; every table is transcribed from it (NOTICE).
- dav1d and rav1d (BSD-2-Clause): read as design guides. Ported parts are listed here and in
  NOTICE, and named in the file that carries them. None so far.
