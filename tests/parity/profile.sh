#!/bin/sh
# profile.sh prep <dir> | run <report-dir> (CI only, .github/workflows/profile.yml).
#   prep: copies the bench classes (bench-lib.sh) into <dir>; runs in the image build.
#   run:  where av1-oracle-prof (debug line tables, frame pointers) spends its time with one
#         thread, per class and over all classes: perf when the container may use it, else
#         callgrind. Reports: <class>.srcfile.txt (self time by source file),
#         <class>.symbol.txt, all.srcline.txt, summary.md.
set -u
W=/work
B=/bench
case $1 in
prep)
  . tests/parity/bench-lib.sh
  classes "$2"
  exit 0 ;;
run) REP=$2 ;;
*) echo "usage: profile.sh prep <dir> | run <report-dir>" >&2; exit 2 ;;
esac
. tests/parity/bench-lib.sh
rm -rf $W; mkdir -p $W/all "$REP"
for c in $CLASSES; do for f in $B/$c/*; do ln -sf "$f" $W/all/; done; done

PERF=
for p in perf $(ls /usr/bin/perf_* 2>/dev/null); do
  if $p record -q -o $W/probe.data -- true >/dev/null 2>&1; then PERF=$p; break; fi
done
{
  echo "## av1-rs profile (1 thread, scalar unless AV1_SIMD says otherwise)"
  echo
} > "$REP/summary.md"
for c in all $CLASSES; do
  d=$B/$c; [ $c = all ] && d=$W/all
  if [ -n "$PERF" ]; then
    $PERF record -q --all-user -F 1999 -g -o $W/$c.data -- sh -c "for i in 1 2 3 4 5; do av1-oracle-prof --null --threads 1 $d $W/o; done" >/dev/null 2>&1
    $PERF report -i $W/$c.data --stdio --no-children --sort srcfile --percent-limit 0.3 -g none > "$REP/$c.srcfile.txt" 2> "$REP/$c.srcfile.err"
    $PERF report -i $W/$c.data --stdio --no-children --sort symbol --percent-limit 0.2 -g none > "$REP/$c.symbol.txt" 2>/dev/null
    top=$REP/$c.srcfile.txt
    grep -q '%' "$top" || top=$REP/$c.symbol.txt
    if [ $c = all ]; then
      $PERF report -i $W/$c.data --stdio --no-children --sort srcline --percent-limit 0.1 -g none > "$REP/all.srcline.txt" 2> "$REP/all.srcline.err"
      # The same run with LLVM inlining off (av1-oracle-noinline): the time of each function
      # on its own, instead of folded into its caller (the absolute times are higher).
      $PERF record -q --all-user -F 1999 -o $W/noinline.data -- sh -c "for i in 1 2 3; do av1-oracle-noinline --null --threads 1 $d $W/o; done" >/dev/null 2>&1
      $PERF report -i $W/noinline.data --stdio --no-children --sort symbol --percent-limit 0.1 -g none > "$REP/all.noinline.symbol.txt" 2>/dev/null
      {
        echo "### all, inlining off"
        echo
        echo '```'
        grep -v '^#' "$REP/all.noinline.symbol.txt" | grep -v '^$' | head -n 40
        echo '```'
        echo
      } >> "$REP/summary.md"
    fi
  else
    valgrind --tool=callgrind --callgrind-out-file=$W/$c.cg av1-oracle-prof --null --threads 1 $d $W/o >/dev/null 2>&1
    callgrind_annotate --inclusive=no $W/$c.cg > "$REP/$c.symbol.txt" 2>/dev/null
    top=$REP/$c.symbol.txt
  fi
  {
    echo "### $c"
    echo
    echo '```'
    grep -v '^#' "$top" | grep -v '^$' | head -n 25
    echo '```'
    echo
  } >> "$REP/summary.md"
done
[ -n "$PERF" ] && echo "Tool: $PERF (user-space samples, 1999 Hz)." >> "$REP/summary.md" || echo "Tool: callgrind (perf was not usable in the container)." >> "$REP/summary.md"
cat "$REP/summary.md"
