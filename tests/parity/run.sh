#!/bin/sh
# Parity gates of av1-rs (docs/parity.md). CI only, called by .github/ci/Dockerfile; needs
# av1-oracle (examples/oracle.rs), dav1d and aomdec on PATH and the corpus in /data (/data.list).
#   run.sh full <report-dir>   av1-rs vs dav1d (every stream), aomdec and the published per-frame
#                              md5 (libaom vectors), and --threads 1 vs default threads
#   run.sh ref <out-dir>       av1-oracle, one thread, on the arm64 subset
#   run.sh arm <ref-dir>       av1-oracle-arm64 under qemu on the same subset vs <ref-dir>
#   run.sh seeds <dst>         the corpus as fuzz seeds, one option byte (0x00) appended
# PARITY_MODE (default: the word in tests/parity/mode): `info` (headers only: frame sizes and counts against dav1d's y4m) or `pixels`
# (every output plane byte-exact). Exit status 0 only when every gate passes.
set -u
W=/work
MODE=${PARITY_MODE:-$(cat tests/parity/mode)}

# inputs <dst> [pattern]: link the corpus files into <dst>, flattened (/ -> __). A file whose
# name says annexb gets the .annexb extension, so the oracle reads it as Annex B.
inputs() {
  mkdir -p "$1"
  grep -E "${2:-.}" /data.list | while IFS= read -r f; do
    n=$(printf '%s' "${f#./}" | sed 's#/#__#g')
    case $n in *annexb*.obu) n=${n%.obu}.annexb ;; esac
    ln -sf "/data/${f#./}" "$1/$n"
  done
}

stem() { printf '%s' "${1%.*}"; }

# where <frames-file> <byte offset, 0-based>: "frame F plane P x X y Y" of a sample in the .yuv.
where() {
  off=$2 i=0
  while read -r w h bd lay; do
    b=1; [ "$bd" -gt 8 ] && b=2
    cw=$w ch=$h np=3
    case $lay in 420) cw=$(((w + 1) / 2)) ch=$(((h + 1) / 2)) ;; 422) cw=$(((w + 1) / 2)) ;; 400) np=1 ;; esac
    for p in $(seq 0 $((np - 1))); do
      pw=$w ph=$h; [ "$p" -gt 0 ] && pw=$cw ph=$ch
      sz=$((pw * ph * b))
      if [ "$off" -lt "$sz" ]; then
        s=$((off / b)); echo "frame $i plane $p x $((s % pw)) y $((s / pw)) (${w}x$h $lay $bd-bit)"; return
      fi
      off=$((off - sz))
    done
    i=$((i + 1))
  done < "$1"
  echo "past the end"
}

# mismatch <a.yuv> <b.yuv> <frames>: where the first difference is, and how many bytes differ.
mismatch() {
  first=$(cmp "$1" "$2" 2>&1 | head -n 1)
  case $first in *differ:*) o=$(printf '%s' "$first" | sed -E 's/.*char ([0-9]+).*/\1/'); echo "first at $(where "$3" $((o - 1))), $(cmp -l "$1" "$2" 2>/dev/null | head -n 100000 | wc -l) bytes differ (capped 1e5)" ;; *) echo "$first" ;; esac
}

# planes size of one frame: W H bitdepth layout
fsize() {
  b=1; [ "$3" -gt 8 ] && b=2
  case $4 in
    400) echo $(($1 * $2 * b)) ;;
    420) echo $((($1 * $2 + 2 * (($1 + 1) / 2) * (($2 + 1) / 2)) * b)) ;;
    422) echo $((($1 * $2 + 2 * (($1 + 1) / 2) * $2) * b)) ;;
    444) echo $((3 * $1 * $2 * b)) ;;
  esac
}

full() {
  REP=$1 bad=0
  rm -rf $W; mkdir -p "$REP" $W/in $W/r1 $W/rn $W/dav $W/aom
  inputs $W/in
  n=$(ls $W/in | wc -l)
  # 240 libaom vectors + 49 dav1d-test-data streams: a mount that did not arrive must fail.
  [ "$n" -ge 285 ] || { echo "corpus has $n streams, want at least 285" >&2; bad=1; }
  flag=; [ "$MODE" = info ] && flag=--info
  av1-oracle $flag --threads 1 $W/in $W/r1 || bad=1
  av1-oracle $flag $W/in $W/rn || bad=1
  : > $W/d-dav1d; : > $W/d-aom; : > $W/d-md5; : > $W/d-err
  frames=0 cmpd=0 aomd=0 md5d=0
  for p in $W/in/*; do
    f=$(basename "$p"); s=$(stem "$f")
    if [ "$MODE" = info ]; then out=$W/r1/$s.info; else out=$W/r1/$s.frames; fi
    [ -f "$out" ] || : > "$out"
    if [ "$MODE" = info ]; then
      # shown frames: column 2 is show (1) for an output frame
      N=$(awk '$2 == 1' "$out" | wc -l)
    else
      N=$(wc -l < "$out")
    fi
    if [ -f $W/r1/$s.err ] && [ "$(cat $W/r1/$s.err)" != inter ]; then
      echo "$f: av1-rs error $(cat $W/r1/$s.err) ($(cat $W/r1/$s.why 2>/dev/null)) after $N frames" >> $W/d-err
    fi
    [ "$N" -gt 0 ] || continue
    frames=$((frames + N))
    if [ "$MODE" = info ]; then
      dav1d -q --threads 1 --limit "$N" -i "$p" -o $W/dav/$s.y4m 2>/dev/null || { echo "$f: dav1d failed" >> $W/d-dav1d; continue; }
      set -- $(awk '$2 == 1 { print $3, $4, $6, $7; exit }' "$out")
      hdr=$(head -c 200 $W/dav/$s.y4m | head -n 1)
      want=$(stat -c %s $W/dav/$s.y4m)
      got=$((${#hdr} + 1 + N * (6 + $(fsize "$1" "$2" "$3" "$4"))))
      case "$hdr" in *" W$1 H$2 "*) ;; *) echo "$f: y4m '$hdr', av1-rs $1x$2" >> $W/d-dav1d ;; esac
      [ "$want" = "$got" ] || echo "$f: dav1d y4m $want bytes, av1-rs frames give $got" >> $W/d-dav1d
      cmpd=$((cmpd + 1))
      continue
    fi
    dav1d -q --threads 1 --filmgrain 1 --limit "$N" -i "$p" -o $W/dav/$s.yuv 2>/dev/null \
      || echo "$f: dav1d failed" >> $W/d-dav1d
    cmp -s $W/dav/$s.yuv $W/r1/$s.yuv || echo "$f: differs from dav1d ($N frames): $(mismatch $W/dav/$s.yuv $W/r1/$s.yuv $W/r1/$s.frames)" >> $W/d-dav1d
    cmpd=$((cmpd + 1))
    case $f in
      aom__*)
        aomdec --rawvideo --limit="$N" -o $W/aom/$s.yuv "$p" 2>/dev/null || echo "$f: aomdec failed" >> $W/d-aom
        cmp -s $W/aom/$s.yuv $W/r1/$s.yuv || echo "$f: differs from aomdec ($N frames): $(mismatch $W/aom/$s.yuv $W/r1/$s.yuv $W/r1/$s.frames)" >> $W/d-aom
        aomd=$((aomd + 1))
        # The published per-frame md5 (always 3 planes, so monochrome is checked by aomdec only).
        m=/data/aom/${f#aom__}.md5
        if [ -f "$m" ] && ! grep -q ' 400$' $W/r1/$s.frames; then
          i=0 off=0
          while read -r w h bd lay; do
            i=$((i + 1)); sz=$(fsize "$w" "$h" "$bd" "$lay")
            got=$(tail -c +$((off + 1)) $W/r1/$s.yuv | head -c "$sz" | md5sum | cut -d' ' -f1)
            want=$(sed -n "${i}p" "$m" | cut -d' ' -f1)
            [ "$got" = "$want" ] || echo "$f: frame $i md5 $got, published $want" >> $W/d-md5
            off=$((off + sz))
          done < $W/r1/$s.frames
          md5d=$((md5d + 1))
        fi
        ;;
    esac
  done
  diff -r $W/r1 $W/rn > $W/d-threads
  {
    echo "## av1-rs parity ($MODE)"
    echo
    echo "$n streams, $frames frames compared; $cmpd streams vs dav1d, $aomd vs aomdec, $md5d vs the published md5."
    echo
    echo "| gate | result |"
    echo "|---|---|"
  } > "$REP/summary.md"
  for g in err dav1d aom md5 threads; do
    l=$(wc -l < $W/d-$g)
    if [ "$l" -eq 0 ]; then echo "| $g | ok |" >> "$REP/summary.md"
    else echo "| $g | FAIL, $l lines |" >> "$REP/summary.md"; bad=1; head -n 300 $W/d-$g > "$REP/$g.diff"; fi
  done
  echo >> "$REP/summary.md"
  echo "Stop classes (av1-rs .err, one thread): $(for e in $W/r1/*.err; do [ -f "$e" ] && printf '%s\n' "$(cat "$e")"; done | sort | uniq -c | tr -s ' \n' ' ')" >> "$REP/summary.md"
  cat "$REP/summary.md"
  return $bad
}

SUBSET='size-(16x16|66x66|196x196|226x226)|quantizer-(00|31|63)|monochrome|film_grain-50|allintra|features__(annexb|section5)'

ref() {
  rm -rf $W; mkdir -p $W/in "$1"
  inputs $W/in "$SUBSET"
  flag=; [ "$MODE" = info ] && flag=--info
  av1-oracle $flag --threads 1 $W/in "$1"
}

arm() {
  rm -rf $W; mkdir -p $W/in $W/arm
  inputs $W/in "$SUBSET"
  flag=; [ "$MODE" = info ] && flag=--info
  qemu-aarch64 -L /usr/aarch64-linux-gnu "$(command -v av1-oracle-arm64)" $flag --threads 1 $W/in $W/arm || exit 1
  diff -r "$1" $W/arm && echo "arm64 matches x86_64 on $(ls $W/in | wc -l) streams"
}

seeds() {
  rm -rf $W; mkdir -p $W/in "$1"
  inputs $W/in
  for p in $W/in/*; do
    cp "$p" "$1/$(basename "$p")" && printf '\0' >> "$1/$(basename "$p")"
  done
  echo "$(ls "$1" | wc -l) seeds"
}

case ${1:-} in
  full) full "$2" ;;
  ref) ref "$2" ;;
  arm) arm "$2" ;;
  seeds) seeds "$2" ;;
  *) echo "usage: run.sh full|ref|arm|seeds <dir>" >&2; exit 2 ;;
esac
