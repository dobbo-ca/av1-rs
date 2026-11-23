# CI image: Rust runs here, on the in-cluster BuildKit, because the ARC runner has no toolchain.
# The one private dependency, isobmff-rs, is fetched in `toolchain` with the App token (below);
# every later stage builds offline. The fuzz crate's dependency
# (libfuzzer-sys from crates.io) is fetched in `toolchain`, before the source is copied.
FROM rust:1.99.1-bookworm AS toolchain
# gcc-aarch64-linux-gnu: the linker of `cargo build --target aarch64-unknown-linux-gnu`.
RUN rustup component add rustfmt clippy && rustup target add aarch64-unknown-linux-gnu \
    && apt-get update && apt-get install -y --no-install-recommends gcc-aarch64-linux-gnu libc6-dev-arm64-cross \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc
WORKDIR /src
COPY tests/fuzz/Cargo.toml tests/fuzz/Cargo.lock tests/fuzz/
COPY Cargo.toml Cargo.lock ./
RUN mkdir -p src tests/fuzz/fuzz_targets && touch src/lib.rs \
    && for t in probe decode avif; do touch tests/fuzz/fuzz_targets/$t.rs; done
# isobmff-rs is a private dobbo-ca port: the dobbo-ca-ci-read App token reaches this one RUN as
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
    find src examples tests -name '*.rs' -exec touch {} + && cargo clippy --locked --all-targets -- -D warnings
RUN --mount=type=cache,id=av1-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo clippy --locked --all-targets --target aarch64-unknown-linux-gnu -- -D warnings
RUN --mount=type=cache,id=av1-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo test --locked --no-fail-fast
# The whole suite again with each lower SIMD tier forced (src/simd; the default run above uses
# the best tier, AVX-512 on the ARC nodes). The cross-tier kernel tests compare every tier anyway.
RUN --mount=type=cache,id=av1-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && AV1_SIMD=scalar cargo test --locked --no-fail-fast && AV1_SIMD=avx2 cargo test --locked --no-fail-fast
# The examples are built too, so the cross linker runs (a lib alone is not linked).
RUN --mount=type=cache,id=av1-rs-target-ci,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + && cargo build --locked --release --target aarch64-unknown-linux-gnu --examples

# qemu-user, for the arm64 stages (the cross gcc is in `toolchain`).
FROM toolchain AS qemu-tools
RUN apt-get update && apt-get install -y --no-install-recommends qemu-user \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_RUNNER="qemu-aarch64 -L /usr/aarch64-linux-gnu" \
    CARGO_NET_OFFLINE=true

FROM qemu-tools AS qemu-arm64
COPY . .

# aarch64 tests under qemu-user: the library unit tests (they hold the SIMD kernel, cross-tier
# and dispatch tests; the full suite runs natively in `ci`), with the NEON tier and with the
# scalar tier forced.
FROM qemu-arm64 AS arm64
RUN --mount=type=cache,id=av1-rs-target-arm64,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo test --locked --release --target aarch64-unknown-linux-gnu --lib \
    && AV1_SIMD=scalar cargo test --locked --release --target aarch64-unknown-linux-gnu --lib

# The C oracles (CI only, never linked into the crate): dav1d 1.0.0 and aomdec 3.6.0 from
# bookworm. Nothing of the crate is copied here, so a crate change does not rebuild it.
FROM toolchain AS oracles
RUN apt-get update && apt-get install -y --no-install-recommends dav1d aom-tools libavif-bin \
    && rm -rf /var/lib/apt/lists/*
ENV CARGO_NET_OFFLINE=true

# The public corpus (tests/parity/corpus.txt, every file checked against its pinned hash) in a
# cache mount, so a build downloads it once. sharing=locked: two builds never download at once.
# Only a list of the files reaches the layer.
FROM oracles AS data
COPY tests/parity/fetch.sh tests/parity/corpus.txt /parity/
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked \
    sh /parity/fetch.sh /parity/corpus.txt /data \
    && (cd /data && find . -type f \( -name '*.ivf' -o -name '*.obu' -o -name '*.avif' \) | sort) > /data.list && wc -l /data.list

# The Rust oracle on the x86_64 host, one thread: the tree that the aarch64 oracle must reproduce.
FROM src AS x86-ref
COPY --from=data /data.list /data.list
RUN --mount=type=cache,id=av1-rs-target-parity,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo build --locked --release --example oracle --example avifx && cp target/release/examples/oracle /usr/local/bin/av1-oracle && cp target/release/examples/avifx /usr/local/bin/av1-avifx
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked,readonly \
    sh tests/parity/run.sh ref /ref

# Parity gate (tests/parity/run.sh full): av1-rs vs dav1d and aomdec on the corpus, and one
# thread vs default threads. The stage does not fail on a diff: the report (summary.md, *.diff,
# status) is the build output (target parity-report), and the workflow fails on `status`.
FROM oracles AS parity
COPY . .
COPY --from=data /data.list /data.list
RUN --mount=type=cache,id=av1-rs-target-parity,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo build --locked --release --example oracle --example avifx && cp target/release/examples/oracle /usr/local/bin/av1-oracle && cp target/release/examples/avifx /usr/local/bin/av1-avifx
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked,readonly \
    mkdir -p /report && { sh tests/parity/run.sh full /report; echo $? > /report/status; }

FROM scratch AS parity-report
COPY --from=parity /report /

# The oracle built for aarch64, run under qemu-user on a subset of the corpus, compared with the
# x86_64 output above.
FROM qemu-arm64 AS parity-arm64
COPY --from=data /data.list /data.list
RUN --mount=type=cache,id=av1-rs-target-arm64,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo build --locked --release --target aarch64-unknown-linux-gnu --example oracle \
    && cp target/aarch64-unknown-linux-gnu/release/examples/oracle /usr/local/bin/av1-oracle-arm64
RUN --mount=type=bind,from=x86-ref,source=/ref,target=/ref,rw \
    --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked,readonly \
    sh tests/parity/run.sh arm /ref

# Fuzzing (fuzz.yml): nightly and cargo-fuzz, with the targets built into the image. The workflow
# runs it with `docker create` + `docker start`, so a BuildKit cache hit cannot skip the fuzzing.
FROM toolchain AS fuzz-tools
RUN rustup toolchain install nightly --profile minimal \
    && cargo install cargo-fuzz --locked --version 0.13.1
ENV CARGO_NET_OFFLINE=true

# Seeds: the raw corpus files (see run.sh seeds), each with one option
# byte at the end (the targets read their options from the last byte).
FROM fuzz-tools AS fuzz
COPY . .
COPY --from=data /data.list /data.list
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked,readonly \
    sh tests/parity/run.sh seeds tests/fuzz/seeds
RUN cargo +nightly fuzz build --fuzz-dir tests/fuzz

# Benchmark (bench.yml) and profile (profile.yml) binaries: av1-oracle (the default release
# build), av1-oracle-v4 (the release-binary flags of docs/conventions.md: x86-64-v4 with the
# false-deps features, fat LTO, panic=abort) and av1-avifx (extracts AVIF items for the classes).
FROM oracles AS bench-bin
COPY . .
COPY --from=data /data.list /data.list
RUN --mount=type=cache,id=av1-rs-target-bench,sharing=locked,target=/src/target \
    find src examples tests -name '*.rs' -exec touch {} + \
    && cargo build --locked --release --example oracle --example avifx \
    && cp target/release/examples/oracle /usr/local/bin/av1-oracle && cp target/release/examples/avifx /usr/local/bin/av1-avifx \
    && RUSTFLAGS="-C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls" \
       CARGO_PROFILE_RELEASE_LTO=fat CARGO_PROFILE_RELEASE_PANIC=abort \
       cargo build --locked --release --example oracle --target-dir target/v4 \
    && cp target/v4/release/examples/oracle /usr/local/bin/av1-oracle-v4

# Benchmark: dav1d, aomdec and av1-rs on the bench classes, in one container, so the ratio
# compares the same machine. Information only (no speed bar). BENCH_RUN changes per run, so a
# cache hit of the same source does not reuse an old timing.
FROM bench-bin AS bench
ARG BENCH_RUN=0
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked,readonly \
    echo "run $BENCH_RUN" && mkdir -p /report && { sh tests/parity/bench.sh /report; echo $? > /report/status; }

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
RUN --mount=type=cache,id=av1-rs-data,target=/data,sharing=locked,readonly \
    sh tests/parity/profile.sh prep /bench && du -sh /bench
