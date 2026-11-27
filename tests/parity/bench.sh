#!/bin/sh
# bench.sh <report-dir>: dav1d, aomdec and av1-rs (examples/oracle.rs, the default build and the
# x86-64-v4 build) decode the same frames of each bench class (tests/parity/bench-lib.sh), with
# one thread and with the default threads, in one container. Information only: there is no speed
# bar (docs/benchmarks.md). Wall time of the whole class, best of 5. No decoder writes samples.
# On aarch64: dav1d (with its NEON asm and with --cpumask none) and av1-rs with AV1_SIMD=scalar
# and with NEON, no aomdec and no v4 build.
set -u
W=/work
REP=$1
NT=$(nproc)
ARCH=$(uname -m)
CPU=$(lscpu 2>/dev/null | sed -n 's/^Model name: *//p' | head -n 1)
# The v4 build and the avx512 tier need AVX-512 (F, BW, VL, DQ, CD); not every hosted x86_64
# runner has it.
AVX512=yes
for f in avx512f avx512bw avx512vl avx512dq avx512cd; do grep -qw "$f" /proc/cpuinfo || AVX512=; done
. tests/parity/bench-lib.sh
rm -rf $W; mkdir -p "$REP" $W
: > "$REP/failures.txt"
classes $W/c 2>> "$REP/failures.txt"

# The frames av1-rs outputs per stream (the leading intra frames); dav1d and aomdec decode the
# same number.
for c in $CLASSES; do
  av1-oracle --null --threads 1 $W/c/$c $W/f/$c >/dev/null 2>&1 || echo "av1-oracle exited non-zero on $c" >> "$REP/failures.txt"
done

ms() {
  t0=$(date +%s%N); "$@" >/dev/null 2>&1 || echo "failed: $*" >> "$REP/failures.txt"; t1=$(date +%s%N)
  echo $(((t1 - t0) / 1000000))
}
best() {
  b=
  for _ in 1 2 3 4 5; do
    t=$(ms "$@")
    if [ -z "$b" ] || [ "$t" -lt "$b" ]; then b=$t; fi
  done
  echo "$b"
}
# each <class> <cmd...>: runs <cmd> <file> <frames> for every file of the class.
each() {
  c=$1; shift
  for p in $W/c/$c/*; do
    n=$(wc -l < "$W/f/$c/$(basename "${p%.*}").frames" 2>/dev/null || echo 0)
    [ "$n" -gt 0 ] || continue
    "$@" "$p" "$n" || return 1
  done
}
dav() { dav1d -q --threads "$1" --limit "$3" -i "$2" -o /dev/null --muxer null; }
davc() { dav1d -q --cpumask none --threads "$1" --limit "$3" -i "$2" -o /dev/null --muxer null; }
aom() { aomdec --rawvideo --threads="$1" --limit="$3" -o /dev/null "$2"; }
rs() { "$1" --null --threads "$2" "$W/c/$3" $W/o; }
num() { case ${1:-} in '' | *[!0-9]*) return 1 ;; esac; }
ratio() { num "$1" && num "$2" && [ "$2" -gt 0 ] && echo "$(($1 * 100 / $2))%" || echo "-"; }

table() {
  tn=$1 dt=$2 at=$3
  echo "| class | streams | frames | dav1d ms | aomdec ms | av1-rs ms | av1-rs v4 ms | av1-rs / dav1d | v4 / dav1d | av1-rs / aomdec |"
  echo "|---|---|---|---|---|---|---|---|---|---|"
  sd=0 sa=0 sr=0 sv=0
  for c in $CLASSES; do
    d=$(best each $c dav $dt); a=$(best each $c aom $at)
    r=$(best rs av1-oracle $tn $c)
    v=-; [ -n "$AVX512" ] && v=$(best rs av1-oracle-v4 $tn $c) && sv=$((sv + v))
    sd=$((sd + d)) sa=$((sa + a)) sr=$((sr + r))
    echo "| $c | $(ls $W/c/$c | wc -l) | $(cat $W/f/$c/*.frames 2>/dev/null | wc -l) | $d | $a | $r | $v | $(ratio $r $d) | $(ratio $v $d) | $(ratio $r $a) |"
  done
  [ -n "$AVX512" ] || sv=-
  echo "| all | | | $sd | $sa | $sr | $sv | $(ratio $sr $sd) | $(ratio $sv $sd) | $(ratio $sr $sa) |"
}

# ab <threads>: the default build of the base commit (av1-oracle-base, bench.yml input `base`)
# against this commit's, the two runs of each round one after the other, best of 7 rounds; the
# ratio in tenths of a percent.
permille() { num "$1" && num "$2" && [ "$2" -gt 0 ] && { x=$(($1 * 1000 / $2)); echo "$((x / 10)).$((x % 10))%"; } || echo "-"; }
ab() {
  echo "| class | base ms | this ms | this / base |"
  echo "|---|---|---|---|"
  sb=0 st=0
  for c in $CLASSES; do
    b= t=
    for _ in 1 2 3 4 5 6 7; do
      x=$(ms rs av1-oracle-base $1 $c); { [ -z "$b" ] || [ "$x" -lt "$b" ]; } && b=$x
      x=$(ms rs av1-oracle $1 $c); { [ -z "$t" ] || [ "$x" -lt "$t" ]; } && t=$x
    done
    sb=$((sb + b)) st=$((st + t))
    echo "| $c | $b | $t | $(permille $t $b) |"
  done
  echo "| all | $sb | $st | $(permille $st $sb) |"
}

# tiers: av1-rs with each SIMD tier forced (AV1_SIMD), one thread; which tier wins per class.
tiers() {
  echo "| class | scalar ms | avx2 ms | avx512 ms (best) | avx2 / scalar | avx512 / avx2 |"
  echo "|---|---|---|---|---|---|"
  for c in $CLASSES; do
    export AV1_SIMD=scalar; s=$(best rs av1-oracle 1 $c)
    export AV1_SIMD=avx2; x=$(best rs av1-oracle 1 $c)
    unset AV1_SIMD; z=-; [ -n "$AVX512" ] && z=$(best rs av1-oracle 1 $c)
    echo "| $c | $s | $x | $z | $(ratio $x $s) | $(ratio $z $x) |"
  done
}

# neon <av1-rs threads> <dav1d threads>: aarch64, dav1d with and without its asm, av1-rs scalar
# and NEON (the best tier).
neon() {
  tn=$1 dt=$2
  echo "| class | streams | frames | dav1d ms | dav1d C ms | av1-rs scalar ms | av1-rs neon ms | neon / scalar | neon / dav1d | dav1d / dav1d C |"
  echo "|---|---|---|---|---|---|---|---|---|---|"
  sd=0 sc=0 ss=0 sn=0
  for c in $CLASSES; do
    d=$(best each $c dav $dt); dc=$(best each $c davc $dt)
    export AV1_SIMD=scalar; s=$(best rs av1-oracle $tn $c)
    unset AV1_SIMD; n=$(best rs av1-oracle $tn $c)
    sd=$((sd + d)) sc=$((sc + dc)) ss=$((ss + s)) sn=$((sn + n))
    echo "| $c | $(ls $W/c/$c | wc -l) | $(cat $W/f/$c/*.frames 2>/dev/null | wc -l) | $d | $dc | $s | $n | $(ratio $n $s) | $(ratio $n $d) | $(ratio $d $dc) |"
  done
  echo "| all | | | $sd | $sc | $ss | $sn | $(ratio $sn $ss) | $(ratio $sn $sd) | $(ratio $sd $sc) |"
}

{
  echo "## av1-rs bench (information only)"
  echo
  echo "$ARCH, ${CPU:-unknown CPU}, $NT threads (nproc). Classes: tests/parity/bench-lib.sh. Best of 5 wall times; no decoder writes samples."
  echo
  if command -v av1-oracle-base >/dev/null; then
    echo "### A/B: this commit against the base $(cat /ab-base.rev 2>/dev/null) (default build, best of 7 rounds)"
    echo
    echo "1 thread:"
    echo
    ab 1
    echo
    echo "default threads ($NT):"
    echo
    ab $NT
    echo
  fi
  if [ "$ARCH" = aarch64 ]; then
    echo "dav1d C: dav1d --cpumask none (its C code, no NEON asm). av1-rs scalar: AV1_SIMD=scalar."
    echo
    echo "### 1 thread"
    echo
    neon 1 1
    echo
    echo "### default threads ($NT)"
    echo
    neon $NT 0
  else
    echo "av1-rs v4: built with -C target-cpu=x86-64-v4, fat LTO and panic=abort.$([ -n "$AVX512" ] || echo " This CPU has no AVX-512: no v4 times and no avx512 tier.")"
    echo
    echo "### 1 thread"
    echo
    table 1 1 1
    echo
    echo "### default threads ($NT)"
    echo
    table $NT 0 $NT
    echo
    echo "### SIMD tiers (av1-rs default build, 1 thread)"
    echo
    tiers
  fi
} > "$REP/summary.md"
[ -s "$REP/failures.txt" ] && { echo; echo "WARNING: a command failed, its times are not meaningful:"; sed 's/^/    /' "$REP/failures.txt"; } >> "$REP/summary.md"
cat "$REP/summary.md"
