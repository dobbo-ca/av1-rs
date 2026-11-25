# CI image: the checks, oracles, benches and fuzzing run in this image, built with BuildKit on
# GitHub-hosted runners (ubuntu-24.04 for x86_64, ubuntu-24.04-arm for aarch64; every stage builds
# natively for the runner's architecture). The one git dependency, isobmff-rs, is fetched in
# `toolchain` with the App token (below); every later stage builds offline. The fuzz crate's
# dependency (libfuzzer-sys from crates.io) is fetched in `toolchain`, before the source is copied.
FROM rust:1.99.1-bookworm AS toolchain
RUN rustup component add rustfmt clippy
WORKDIR /src
COPY tests/fuzz/Cargo.toml tests/fuzz/Cargo.lock tests/fuzz/
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src tests/fuzz/fuzz_targets && touch src/lib.rs \
    && for t in probe decode frames_threads avif simd_lf simd_grain simd_lr simd_cdef simd_ipred simd_itx simd_threads simd_entropy simd_alloc; do touch tests/fuzz/fuzz_targets/$t.rs; done
# isobmff-rs is not public: the dobbo-ca-ci-read App token reaches this one RUN as
# the BuildKit secret gh_token, and the insteadOf rewrite is removed in the same command.
RUN --mount=type=secret,id=gh_token \
    git config --global url."https://x-access-token:$(cat /run/secrets/gh_token)@github.com/".insteadOf "https://github.com/" \
    && CARGO_NET_GIT_FETCH_WITH_CLI=true cargo fetch --locked \
    && CARGO_NET_GIT_FETCH_WITH_CLI=true cargo fetch --locked --manifest-path tests/fuzz/Cargo.toml \
    && git config --global --unset-all url."https://x-access-token:$(cat /run/secrets/gh_token)@github.com/".insteadOf \
    && rm -rf src tests/fuzz/fuzz_targets

FROM toolchain AS src
ENV CARGO_NET_OFFLINE=true
COPY . .

# The checks run in the build, so a red check fails the build. The target dir is a cache mount;
# every RUN touches the .rs files first, because the cache is shared by all branches and cargo
# compares mtimes.
FROM src AS ci
RUN --mount=type=cache,id=av1-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo fmt --check
RUN --mount=type=cache,id=av1-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo clippy --locked --all-targets -- -D warnings \
    && cargo clippy --locked --lib --features fuzzing -- -D warnings
# The fuzz targets (their own workspace, built for fuzzing only by fuzz.yml) must still compile.
RUN --mount=type=cache,id=av1-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo check --locked --manifest-path tests/fuzz/Cargo.toml --target-dir target/fuzz-check
RUN --mount=type=cache,id=av1-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo test --locked --no-fail-fast
# The whole suite again with each lower SIMD tier forced (src/simd; the default run above uses
# the best tier: AVX-512 when the runner's CPU has it, else AVX2). The cross-tier kernel tests
# compare every available tier anyway.
RUN --mount=type=cache,id=av1-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && AV1_SIMD=scalar cargo test --locked --no-fail-fast && AV1_SIMD=avx2 cargo test --locked --no-fail-fast

# aarch64 (job arm64, built natively on an arm64 runner): clippy and the whole suite, with the
# NEON tier and with the scalar tier forced.
FROM src AS arm64
RUN --mount=type=cache,id=av1-rs-target-arm64,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo clippy --locked --all-targets -- -D warnings \
    && cargo test --locked --no-fail-fast && AV1_SIMD=scalar cargo test --locked --no-fail-fast

# avifdec-new: libavif v1.4.2 (commit c5240fc79fe5) built from source, a second AVIF oracle that
# follows the dimg order (libavif >= 1.1.1). Its only codec is bookworm's libdav1d-dev (dav1d
# 1.0.0). The apps need libargparse at the pin of libavif's cmake/Modules/LocalLibargparse.cmake,
# fetched into ext/ beside it; the build runs without network. Based on the rust image only
# because that image is already pulled: nothing of the crate is copied.
FROM rust:1.99.1-bookworm AS avifdec-new
RUN apt-get update && apt-get install -y --no-install-recommends cmake libdav1d-dev libpng-dev libjpeg-dev zlib1g-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*
RUN git clone -q --depth 1 --branch v1.4.2 https://github.com/AOMediaCodec/libavif.git /libavif \
    && test "$(git -C /libavif rev-parse HEAD)" = c5240fc79fe5c2407e10afd35f5505ef6333ea49 \
    && git init -q /libavif/ext/libargparse \
    && git -C /libavif/ext/libargparse fetch -q --depth 1 https://github.com/kmurray/libargparse.git ee74d1b53bd680748af14e737378de57e2a0a954 \
    && git -C /libavif/ext/libargparse checkout -q FETCH_HEAD
RUN --network=none cmake -S /libavif -B /libavif/build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
       -DAVIF_CODEC_DAV1D=SYSTEM -DAVIF_LIBYUV=OFF -DAVIF_BUILD_APPS=ON \
    && cmake --build /libavif/build --target avifdec -j"$(nproc)" \
    && cp /libavif/build/avifdec /usr/local/bin/avifdec-new

# The C oracles (CI only, never linked into the crate): dav1d 1.0.0, aomdec 3.6.0 and avifdec
# 0.11.1 from bookworm, and avifdec-new (above; it needs the bookworm libdav1d6, libpng16-16 and
# libjpeg62-turbo). Nothing of the crate is copied here, so a crate change does not rebuild it.
FROM toolchain AS oracles
RUN apt-get update && apt-get install -y --no-install-recommends dav1d aom-tools libavif-bin libdav1d6 libpng16-16 libjpeg62-turbo \
    && rm -rf /var/lib/apt/lists/*
COPY --from=avifdec-new /usr/local/bin/avifdec-new /usr/local/bin/avifdec-new
RUN avifdec-new --version
ENV CARGO_NET_OFFLINE=true

# The public corpus (tests/parity/corpus.txt, every file checked against its pinned hash) in a
# cache mount, so a build downloads it once. sharing=locked: two builds never download at once.
# Only a list of the files reaches the layer. A cache mount lives only as long as the runner's
# builder, while this layer can come back from the Actions cache, so every RUN that mounts
# /data runs fetch.sh first (a no-op when the files are there).
FROM oracles AS data
COPY tests/parity/fetch.sh tests/parity/corpus.txt /parity/
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked \
    sh /parity/fetch.sh /parity/corpus.txt /data \
    && (cd /data && find . -type f \( -name '*.ivf' -o -name '*.obu' -o -name '*.avif' \) | sort) > /data.list && wc -l /data.list

# Parity gate (tests/parity/run.sh full): av1-rs vs dav1d and aomdec on the corpus, and one
# thread vs default threads. The stage does not fail on a diff: the report (summary.md, *.diff,
# status) is the build output (target parity-report), and the workflow fails on `status`. Job
# `parity` builds it on x86_64, job `parity-arm64` on aarch64 (NEON, and the scalar tier).
FROM oracles AS parity
COPY . .
COPY --from=data /data.list /data.list
RUN --mount=type=cache,id=av1-rs-target-parity,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo build --locked --release --example oracle --example avifx && cp target/release/examples/oracle /usr/local/bin/av1-oracle && cp target/release/examples/avifx /usr/local/bin/av1-avifx
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked \
    sh tests/parity/fetch.sh tests/parity/corpus.txt /data \
    && mkdir -p /report && { sh tests/parity/run.sh full /report; echo $? > /report/status; }

FROM scratch AS parity-report
COPY --from=parity /report /

# libheif v1.23.7 (LGPL-3.0, commit 12a259a5882d), the oracle of the transform gate (CI only, never
# linked into the crate): built from git, static, with dav1d from bookworm as its AV1 decoder and
# no other codec. Only transform_oracle.c is copied, after the library build, so a change to it
# does not rebuild libheif. The -dev packages ship static libraries; the runtime package of dav1d
# is installed in `transform` (`oracles`) in case the link falls back to the shared one.
FROM debian:bookworm AS c-oracle
RUN apt-get update && apt-get install -y --no-install-recommends build-essential cmake git ca-certificates pkg-config libdav1d-dev \
    && rm -rf /var/lib/apt/lists/*
RUN git clone -q --depth 1 --branch v1.23.7 https://github.com/strukturag/libheif /src/libheif \
    && test "$(git -C /src/libheif rev-parse HEAD)" = 12a259a5882dd99d45ec5fc77e296a3b620d1e2e \
    && cmake -S /src/libheif -B /b/heif -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=/opt/oracle \
       -DBUILD_SHARED_LIBS=OFF -DENABLE_PLUGIN_LOADING=OFF -DBUILD_TESTING=OFF -DBUILD_DOCUMENTATION=OFF \
       -DWITH_DAV1D=ON -DWITH_LIBDE265=OFF -DWITH_X265=OFF -DWITH_X264=OFF -DWITH_OpenH264_DECODER=OFF -DWITH_KVAZAAR=OFF \
       -DWITH_AOM_DECODER=OFF -DWITH_AOM_ENCODER=OFF -DWITH_RAV1E=OFF -DWITH_SvtEnc=OFF \
       -DWITH_JPEG_DECODER=OFF -DWITH_JPEG_ENCODER=OFF -DWITH_OpenJPEG_DECODER=OFF -DWITH_OpenJPEG_ENCODER=OFF \
       -DWITH_OPENJPH_ENCODER=OFF -DWITH_FFMPEG_DECODER=OFF -DWITH_UVG266=OFF -DWITH_VVDEC=OFF -DWITH_VVENC=OFF \
       -DWITH_UNCOMPRESSED_CODEC=OFF -DWITH_HEADER_COMPRESSION=OFF -DWITH_LIBSHARPYUV=OFF \
       -DWITH_EXAMPLES=OFF -DWITH_GDK_PIXBUF=OFF \
    && cmake --build /b/heif -j"$(nproc)" && cmake --install /b/heif \
    && rm -rf /src/libheif /b/heif
ENV PKG_CONFIG_PATH=/opt/oracle/lib/pkgconfig:/opt/oracle/lib/x86_64-linux-gnu/pkgconfig
COPY tests/parity/c/transform_oracle.c /c/transform_oracle.c
# libheif is C++: the C oracle is compiled with gcc and linked with g++.
RUN gcc -O2 -std=gnu99 -Wall -c -o /tmp/transform_oracle.o /c/transform_oracle.c $(pkg-config --cflags libheif) \
    && g++ -o /usr/local/bin/heif-transform-oracle /tmp/transform_oracle.o -static-libstdc++ -static-libgcc $(pkg-config --static --libs libheif)

# Transform gate (tests/parity/transform.sh): libheif against decode_avif with
# Options::apply_transforms, on the corpus and on files that avifenc (libavif 0.11.1 from bookworm,
# in `oracles`) writes with clap, irot and imir. Same layout as the parity stage: the stage never
# fails on a diff, the report (summary.md, transform.diff, status) is the build output (target
# transform-report), and the workflow fails on `status`.
FROM oracles AS transform
COPY . .
COPY --from=c-oracle /usr/local/bin/heif-transform-oracle /usr/local/bin/heif-transform-oracle
COPY --from=data /data.list /data.list
RUN --mount=type=cache,id=av1-rs-target-parity,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo build --locked --release --example oracle && cp target/release/examples/oracle /usr/local/bin/av1-oracle
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked \
    sh tests/parity/fetch.sh tests/parity/corpus.txt /data \
    && mkdir -p /report && { sh tests/parity/transform.sh /report; echo $? > /report/status; }

FROM scratch AS transform-report
COPY --from=transform /report /

# Fuzzing (fuzz.yml): nightly and cargo-fuzz, with the targets built into the image. The workflow
# runs it with `docker create` + `docker start`, so a BuildKit cache hit cannot skip the fuzzing.
FROM toolchain AS fuzz-tools
RUN rustup toolchain install nightly --profile minimal \
    && cargo install cargo-fuzz --locked --version 0.13.1
ENV CARGO_NET_OFFLINE=true

# Seeds: the raw corpus files (see run.sh seeds), each with one option
# byte at the end (the targets read their options from the last byte). The same stage on both
# architectures: on the arm64 runner the targets are built for aarch64, ASan included.
FROM fuzz-tools AS fuzz
COPY . .
COPY --from=data /data.list /data.list
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked \
    sh tests/parity/fetch.sh tests/parity/corpus.txt /data && sh tests/parity/run.sh seeds tests/fuzz/seeds
RUN cargo +nightly fuzz build --fuzz-dir tests/fuzz

# Benchmark (bench.yml) and profile (profile.yml) binaries: av1-oracle (the default release
# build), av1-oracle-v4 (x86_64 only, the release-binary flags: x86-64-v4,
# fat LTO, panic=abort) and av1-avifx (extracts AVIF items for the classes).
FROM oracles AS bench-bin
COPY . .
COPY --from=data /data.list /data.list
RUN --mount=type=cache,id=av1-rs-target-bench,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo build --locked --release --example oracle --example avifx \
    && cp target/release/examples/oracle /usr/local/bin/av1-oracle && cp target/release/examples/avifx /usr/local/bin/av1-avifx \
    && if [ "$(uname -m)" = x86_64 ]; then \
         RUSTFLAGS="-C target-cpu=x86-64-v4" \
         CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_PANIC=abort \
         cargo build --locked --release --example oracle --target-dir target/v4 \
         && cp target/v4/release/examples/oracle /usr/local/bin/av1-oracle-v4; fi

# Benchmark: dav1d, aomdec and av1-rs on the bench classes, in one container, so the ratio
# compares the same machine. Information only (no speed bar). BENCH_RUN changes per run, so a
# cache hit of the same source does not reuse an old timing.
FROM bench-bin AS bench
ARG BENCH_RUN=0
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked \
    echo "run $BENCH_RUN" && sh tests/parity/fetch.sh tests/parity/corpus.txt /data && mkdir -p /report && { sh tests/parity/bench.sh /report; echo $? > /report/status; }

FROM scratch AS bench-report
COPY --from=bench /report /

# Profile (profile.yml): the oracle with line tables and frame pointers, perf and valgrind, and
# the bench classes copied into the image (the workflow runs it with `docker run --privileged`,
# because perf_event_open is not allowed inside a BuildKit RUN).
FROM bench-bin AS profile
RUN apt-get update && apt-get install -y --no-install-recommends linux-perf valgrind \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,id=av1-rs-target-bench,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && RUSTFLAGS="-C force-frame-pointers=yes" CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
       cargo build --locked --release --example oracle --target-dir target/prof \
    && cp target/prof/release/examples/oracle /usr/local/bin/av1-oracle-prof \
    && RUSTFLAGS="-C force-frame-pointers=yes -C llvm-args=-inline-threshold=0" CARGO_PROFILE_RELEASE_DEBUG=line-tables-only \
       cargo build --locked --release --example oracle --target-dir target/noinline \
    && cp target/noinline/release/examples/oracle /usr/local/bin/av1-oracle-noinline
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked \
    sh tests/parity/fetch.sh tests/parity/corpus.txt /data && sh tests/parity/profile.sh prep /bench && du -sh /bench

# v4-diag.yml: which false-deps target features this rustc knows (tests/parity/v4-diag.sh).
# Needs no crate source and no token.
FROM rust:1.99.1-bookworm AS v4-diag
ARG DIAG_RUN=0
COPY tests/parity/v4-diag.sh /v4-diag.sh
RUN echo "run $DIAG_RUN" && mkdir -p /report && sh /v4-diag.sh > /report/v4-diag.txt 2>&1; echo 0 > /report/status

FROM scratch AS v4-diag-report
COPY --from=v4-diag /report /
