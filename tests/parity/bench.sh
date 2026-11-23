#!/bin/sh
# bench.sh <report-dir>: dav1d and av1-rs (examples/oracle.rs) decode the same frames of the
# corpus, one thread and default threads, in one container. Information only: there is no speed
# bar (docs/benchmarks.md). Wall time of the whole set, best of 3.
set -u
W=/work
REP=$1
rm -rf $W; mkdir -p "$REP" $W/in $W/r $W/d
grep -E 'quantizer-(00|31|63)|allintra|size-(196|226)' /data.list | while IFS= read -r f; do
  ln -sf "/data/${f#./}" "$W/in/$(printf '%s' "${f#./}" | sed 's#/#__#g')"
done
# The frames av1-rs outputs (the leading intra frames); dav1d decodes the same number.
av1-oracle --threads 1 $W/in $W/r >/dev/null 2>&1 || echo "av1-oracle exited non-zero in the frame-count run" >> "$REP/failures.txt"

best() {
  b=
  for _ in 1 2 3; do
    t0=$(date +%s%N); "$@" >/dev/null 2>&1 || echo "failed: $*" >> "$REP/failures.txt"; t1=$(date +%s%N)
    t=$(((t1 - t0) / 1000000))
    if [ -z "$b" ] || [ "$t" -lt "$b" ]; then b=$t; fi
  done
  echo "$b"
}

dav() {
  for p in $W/in/*; do
    s=$(basename "${p%.*}")
    n=$(wc -l < "$W/r/$s.frames" 2>/dev/null || echo 0)
    [ "$n" -gt 0 ] || continue
    dav1d -q --threads "$1" --limit "$n" -i "$p" -o /dev/null --muxer null
  done
}

frames=$(cat $W/r/*.frames 2>/dev/null | wc -l)
r1=$(best av1-oracle --threads 1 $W/in $W/d)
rn=$(best av1-oracle $W/in $W/d)
d1=$(best dav 1)
dn=$(best dav 0)
ratio() { [ "$2" -gt 0 ] && echo "$(($1 * 100 / $2))%" || echo "-"; }
{
  echo "## av1-rs bench (information only)"
  echo
  echo "$(ls $W/in | wc -l) streams, $frames frames (av1-rs output incl. writing .yuv; dav1d to null)."
  echo
  echo "| threads | dav1d ms | av1-rs ms | av1-rs / dav1d |"
  echo "|---|---|---|---|"
  echo "| 1 | $d1 | $r1 | $(ratio "$r1" "$d1") |"
  echo "| default | $dn | $rn | $(ratio "$rn" "$dn") |"
} > "$REP/summary.md"
[ -f "$REP/failures.txt" ] && { echo; echo "WARNING: a timed command failed, times are not meaningful:"; cat "$REP/failures.txt"; } >> "$REP/summary.md"
cat "$REP/summary.md"
