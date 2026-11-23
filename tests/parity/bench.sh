#!/bin/sh
# bench.sh <report-dir>: dav1d, aomdec and av1-rs (examples/oracle.rs, the default build and the
# x86-64-v4 build) decode the same frames of each bench class (tests/parity/bench-lib.sh), with
# one thread and with the default threads, in one container. Information only: there is no speed
# bar (docs/benchmarks.md). Wall time of the whole class, best of 5. No decoder writes samples.
set -u
W=/work
REP=$1
NT=$(nproc)
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
aom() { aomdec --rawvideo --threads="$1" --limit="$3" -o /dev/null "$2"; }
rs() { "$1" --null --threads "$2" "$W/c/$3" $W/o; }
ratio() { [ "${2:-0}" -gt 0 ] && echo "$(($1 * 100 / $2))%" || echo "-"; }

table() {
  tn=$1 dt=$2 at=$3
  echo "| class | streams | frames | dav1d ms | aomdec ms | av1-rs ms | av1-rs v4 ms | av1-rs / dav1d | v4 / dav1d | av1-rs / aomdec |"
  echo "|---|---|---|---|---|---|---|---|---|---|"
  sd=0 sa=0 sr=0 sv=0
  for c in $CLASSES; do
    d=$(best each $c dav $dt); a=$(best each $c aom $at)
    r=$(best rs av1-oracle $tn $c); v=$(best rs av1-oracle-v4 $tn $c)
    sd=$((sd + d)) sa=$((sa + a)) sr=$((sr + r)) sv=$((sv + v))
    echo "| $c | $(ls $W/c/$c | wc -l) | $(cat $W/f/$c/*.frames 2>/dev/null | wc -l) | $d | $a | $r | $v | $(ratio $r $d) | $(ratio $v $d) | $(ratio $r $a) |"
  done
  echo "| all | | | $sd | $sa | $sr | $sv | $(ratio $sr $sd) | $(ratio $sv $sd) | $(ratio $sr $sa) |"
}

{
  echo "## av1-rs bench (information only)"
  echo
  echo "Classes: tests/parity/bench-lib.sh. Best of 5 wall times; no decoder writes samples. av1-rs v4: built with -C target-cpu=x86-64-v4 and the false-deps flags."
  echo
  echo "### 1 thread"
  echo
  table 1 1 1
  echo
  echo "### default threads ($NT)"
  echo
  table $NT 0 $NT
} > "$REP/summary.md"
[ -s "$REP/failures.txt" ] && { echo; echo "WARNING: a command failed, its times are not meaningful:"; sed 's/^/    /' "$REP/failures.txt"; } >> "$REP/summary.md"
cat "$REP/summary.md"
