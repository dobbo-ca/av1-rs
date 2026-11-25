#!/bin/sh
# Which false-deps target features does this rustc know, and what does the x86-64-v4 build enable?
# Run by .github/workflows/v4-diag.yml in the v4-diag image. Information only; prints to stdout.
T=x86_64-unknown-linux-gnu
d=$(mktemp -d); cd "$d" || exit 1
echo "== $(rustc -V)"
echo "== rustc --print target-features: false-deps*"
rustc --print target-features --target $T | grep -i false-deps
echo "== rustc -C target-cpu=x86-64-v4 --print cfg: target_feature"
rustc -C target-cpu=x86-64-v4 --target $T --print cfg | grep target_feature
echo "== target-cpus znver*"
rustc --print target-cpus --target $T | grep -i znver
# A one-line lib that uses tzcnt, lzcnt, blsr and popcnt, compiled with each spelling.
cat > t.rs <<'RS'
#![crate_type = "lib"]
#[no_mangle] pub fn tz(x: u64) -> u32 { x.trailing_zeros() }
#[no_mangle] pub fn lz(x: u64) -> u32 { x.leading_zeros() }
#[no_mangle] pub fn bl(x: u64) -> u64 { x & x.wrapping_sub(1) }
#[no_mangle] pub fn pc(x: u64) -> u32 { x.count_ones() }
RS
for f in false-deps-tzcnt false-deps-bls false-deps-lzcnt-tzcnt false-deps-popcnt false-deps-perm false-deps-range false-deps-getmant false-deps-mulc false-deps-mullq; do
  echo "== build with +$f (stderr)"
  rustc -C opt-level=3 -C target-cpu=x86-64-v4 -C target-feature=+$f --target $T --emit=asm -o "$f.s" t.rs 2>&1
  echo "(exit $?)"
done
echo "== build with the current flags (stderr)"
rustc -C opt-level=3 -C target-cpu=x86-64-v4 -C target-feature=+false-deps-tzcnt,+false-deps-bls --target $T --emit=asm -o cur.s t.rs 2>&1
echo "== asm: tz/lz/pc, default x86-64-v4 vs each CPU and each candidate (xor before the instruction = false dependency broken)"
for c in "-C target-cpu=x86-64-v4" "-C target-cpu=znver4" "-C target-cpu=znver5" "-C target-cpu=x86-64-v4 -C target-feature=+false-deps-lzcnt-tzcnt" "-C target-cpu=x86-64-v4 -C target-feature=+false-deps-popcnt"; do
  echo "-- $c"
  # shellcheck disable=SC2086
  rustc -C opt-level=3 $c --target $T --emit=asm -o a.s t.rs 2>&1 | grep -v '^$'
  grep -E '^(tz|lz|bl|pc):|tzcnt|lzcnt|blsr|popcnt|xor' a.s
done
