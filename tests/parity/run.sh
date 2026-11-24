#!/bin/sh
# Parity gates of av1-rs (docs/parity.md). CI only, called by .github/ci/Dockerfile; needs
# av1-oracle (examples/oracle.rs), av1-avifx, dav1d, aomdec, avifdec and avifdec-new on PATH and
# the corpus in /data (/data.list).
#   run.sh full <report-dir>   av1-rs vs dav1d (every stream), aomdec and the published per-frame
#                              md5 (libaom vectors), the AVIF files vs avifdec and avifdec-new,
#                              --threads 1 vs default threads, and the scalar and AVX2 SIMD
#                              tiers vs the best tier (AV1_SIMD)
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

# avifcmp <avifdec> <work-dir>: the whole AVIF image $p (grid assembled, before irot/imir/clap)
# against an avifdec (libavif with dav1d); its y4m minus the header and the one FRAME line is the
# same layout as .yuv, plus the alpha plane when there is one. A failure goes to
# $REP/<avifdec>-failed.txt, and fails the gate unless tests/parity/<avifdec>-known-failures.txt
# lists it, so a libavif update cannot silently drop files from the comparison. Status 0 when
# compared.
avifcmp() {
  "$1" --jobs 1 "$p" "$2/$s.y4m" >/dev/null 2>&1 || {
    echo "$f" >> "$REP/$1-failed.txt"
    grep -qxF "$f" "tests/parity/$1-known-failures.txt" \
      || echo "$f: $1 failed, not in $1-known-failures.txt" >> $W/d-avif
    return 1
  }
  hl=$(head -n 1 "$2/$s.y4m" | wc -c)
  tail -c +$((hl + 7)) "$2/$s.y4m" > "$2/$s.yuv"
  # - color_grid_alpha_nogrid: the alpha items hang on the grid tiles, not on the grid;
  #   libavif >= 1.0 assembles them into an alpha plane, av1-rs returns none: color planes only.
  case $1:$f in avifdec-new:*color_grid_alpha_nogrid.avif)
    head -c "$(wc -c < $W/r1/$s.yuv)" "$2/$s.yuv" > "$2/$s.c" && mv "$2/$s.c" "$2/$s.yuv" ;;
  esac
  cat $W/r1/$s.yuv $W/r1/$s.alpha.yuv > "$2/$s.ours" 2>/dev/null
  cmp -s "$2/$s.yuv" $W/r1/$s.yuv || cmp -s "$2/$s.yuv" "$2/$s.ours" \
    || echo "$f: differs from $1: $(mismatch "$2/$s.yuv" $W/r1/$s.yuv $W/r1/$s.frames)" >> $W/d-avif
  return 0
}

full() {
  REP=$1 bad=0
  rm -rf $W; mkdir -p "$REP" $W/in $W/r1 $W/rn $W/dav $W/aom $W/avd $W/avn
  rm -f "$REP/avifdec-failed.txt" "$REP/avifdec-new-failed.txt"
  inputs $W/in
  # AVIF: every coded item (av1C config OBUs + item bytes) becomes a Section 5 stream in the same
  # directory, so dav1d and aomdec check the AV1 decode of each item (examples/avifx.rs).
  ls $W/in/*.avif >/dev/null 2>&1 && { av1-avifx $W/in $W/in/*.avif > "$REP/avifx.log" 2>&1 || bad=1; }
  n=$(ls $W/in | wc -l)
  # 289 streams (240 libaom + 49 dav1d-test-data) and 151 AVIF files: a mount that did not arrive
  # must fail. The streams are counted in the list, before the extracted AVIF items join them.
  nav=$(ls $W/in/*.avif 2>/dev/null | wc -l)
  [ "$nav" -ge 151 ] || { echo "corpus has $nav AVIF files, want at least 151" >&2; bad=1; }
  nst=$(grep -vc '\.avif$' /data.list)
  [ "$nst" -ge 289 ] || { echo "corpus has $nst streams, want at least 289" >&2; bad=1; }
  flag=; [ "$MODE" = info ] && flag=--info
  av1-oracle $flag --threads 1 $W/in $W/r1 || bad=1
  av1-oracle $flag $W/in $W/rn || bad=1
  # Every lower SIMD tier (src/simd) must give the output of the best tier (r1), byte for byte.
  for t in scalar avx2; do
    AV1_SIMD=$t av1-oracle $flag --threads 1 $W/in $W/t-$t || bad=1
  done
  : > $W/d-dav1d; : > $W/d-aom; : > $W/d-md5; : > $W/d-err; : > $W/d-avif
  frames=0 cmpd=0 itemd=0 aomd=0 md5d=0 avifd=0 avnd=0
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
    # Known oracle differences (docs/parity.md, "Known oracle differences"):
    # - libavif poc_* files are security reproducers (hostile input): fuzz seeds, no oracle.
    case $f in libavif__poc_*) continue ;; esac
    # - animated_alpha_sample_without_shown_frame: the second sample of the alpha track has no
    #   shown frame; av1-rs returns no alpha, both avifdecs output one: compared through its
    #   items only.
    if [ -f $W/r1/$s.err ] && [ "$(cat $W/r1/$s.err)" != inter ]; then
      case $f in
        *_tile_shared_in_dimg.avif) ;;
        *) echo "$f: av1-rs error $(cat $W/r1/$s.err) ($(cat $W/r1/$s.why 2>/dev/null)) after $N frames" >> $W/d-err ;;
      esac
    fi
    # - *_tile_shared_in_dimg: an item used twice in dimg; av1-rs and avifdec-new (libavif >= 1.1.1)
    #   must both refuse it.
    case $f in *_tile_shared_in_dimg.avif)
      [ -f $W/r1/$s.err ] || echo "$f: av1-rs decoded it, want a refusal" >> $W/d-err
      avifdec-new --jobs 1 "$p" $W/avn/$s.y4m >/dev/null 2>&1 \
        && echo "$f: avifdec-new decoded it, want a refusal as av1-rs" >> $W/d-avif
      continue ;;
    esac
    [ "$N" -gt 0 ] || continue
    case $f in *.avif)
      case $f in *animated_alpha_sample_without_shown_frame.avif) continue ;; esac
      # - *_reversed_dimg_order: libavif 0.11 ignores the dimg order (fixed in 1.1.1), so only
      #   avifdec-new compares it.
      case $f in
        *_reversed_dimg_order.avif) ;;
        *) avifcmp avifdec $W/avd && avifd=$((avifd + 1)) ;;
      esac
      avifcmp avifdec-new $W/avn && avnd=$((avnd + 1))
      continue ;;
    esac
    frames=$((frames + N))
    if [ "$MODE" = info ]; then
      dav1d -q --threads 1 --limit "$N" -i "$p" -o $W/dav/$s.y4m 2>/dev/null || { echo "$f: dav1d failed" >> $W/d-dav1d; continue; }
      set -- $(awk '$2 == 1 { print $3, $4, $6, $7; exit }' "$out")
      hdr=$(head -c 200 $W/dav/$s.y4m | head -n 1)
      want=$(stat -c %s $W/dav/$s.y4m)
      got=$((${#hdr} + 1 + N * (6 + $(fsize "$1" "$2" "$3" "$4"))))
      case "$hdr" in *" W$1 H$2 "*) ;; *) echo "$f: y4m '$hdr', av1-rs $1x$2" >> $W/d-dav1d ;; esac
      [ "$want" = "$got" ] || echo "$f: dav1d y4m $want bytes, av1-rs frames give $got" >> $W/d-dav1d
      case $f in *.item[0-9]*.obu) itemd=$((itemd + 1)) ;; *) cmpd=$((cmpd + 1)) ;; esac
      continue
    fi
    dav1d -q --threads 1 --filmgrain 1 --limit "$N" -i "$p" -o $W/dav/$s.yuv 2>$W/stderr \
      || echo "$f: dav1d failed: $(head -c 160 $W/stderr | tr '\n' ' ')" >> $W/d-dav1d
    cmp -s $W/dav/$s.yuv $W/r1/$s.yuv || echo "$f: differs from dav1d ($N frames): $(mismatch $W/dav/$s.yuv $W/r1/$s.yuv $W/r1/$s.frames)" >> $W/d-dav1d
    case $f in *.item[0-9]*.obu) itemd=$((itemd + 1)) ;; *) cmpd=$((cmpd + 1)) ;; esac
    case $f in
      # Two spatial layers: the upper layer is an inter frame (it predicts from the base), so
      # av1-rs and dav1d (all layers) output the base layer, while libaom outputs the upper
      # layer. Checked against dav1d only (docs/parity.md).
      *svc-L2*|av1-avif__Xiph__*) ;;
      *)
        ab=; case $f in *.annexb) ab=--annexb ;; esac
        aomdec $ab --rawvideo --limit="$N" -o $W/aom/$s.yuv "$p" 2>$W/stderr || echo "$f: aomdec failed: $(head -c 160 $W/stderr | tr '\n' ' ')" >> $W/d-aom
        cmp -s $W/aom/$s.yuv $W/r1/$s.yuv || echo "$f: differs from aomdec ($N frames): $(mismatch $W/aom/$s.yuv $W/r1/$s.yuv $W/r1/$s.frames)" >> $W/d-aom
        aomd=$((aomd + 1))
        # The published per-frame md5 (always 3 planes, so monochrome is checked by aomdec only).
        m=/data/aom/${f#aom__}.md5
        if [ "${f#aom__}" != "$f" ] && [ -f "$m" ] && ! grep -q ' 400$' $W/r1/$s.frames; then
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
  # Floors: a gate that compared (almost) nothing must not pass. Pin these from a green run.
  # A listed avifdec failure that decoded after all is a stale entry.
  for o in avifdec avifdec-new; do
    while IFS= read -r k; do
      grep -qxF "$k" "$REP/$o-failed.txt" 2>/dev/null \
        || echo "$k: listed in $o-known-failures.txt but $o decoded it" >> $W/d-avif
    done < tests/parity/$o-known-failures.txt
  done
  # Every real stream outputs at least its first key frame; the extracted AVIF items are counted
  # apart, so they cannot make up for missing streams.
  [ "$cmpd" -ge 289 ] || echo "only $cmpd streams compared against dav1d, want at least 289" >> $W/d-err
  [ "$itemd" -ge 238 ] || echo "only $itemd AVIF items compared against dav1d, want at least 238" >> $W/d-err
  [ "$avifd" -ge 133 ] || echo "only $avifd AVIF files compared against avifdec, want at least 133" >> $W/d-err
  [ "$avnd" -ge 134 ] || echo "only $avnd AVIF files compared against avifdec-new, want at least 134" >> $W/d-err
  [ "$frames" -ge 604 ] || echo "only $frames frames compared against dav1d, want at least 604" >> $W/d-err
  diff -r $W/r1 $W/rn > $W/d-threads
  : > $W/d-tiers
  for t in scalar avx2; do diff -r $W/r1 $W/t-$t >> $W/d-tiers; done
  {
    echo "## av1-rs parity ($MODE)"
    echo
    echo "$n inputs (streams, AVIF files and their extracted items), $frames frames compared; $cmpd streams and $itemd AVIF items vs dav1d, $aomd vs aomdec, $md5d vs the published md5, $avifd AVIF files vs avifdec, $avnd vs avifdec-new."
    echo
    echo "| gate | result |"
    echo "|---|---|"
  } > "$REP/summary.md"
  for g in err dav1d aom md5 avif threads tiers; do
    l=$(wc -l < $W/d-$g)
    if [ "$l" -eq 0 ]; then echo "| $g | ok |" >> "$REP/summary.md"
    else echo "| $g | FAIL, $l lines |" >> "$REP/summary.md"; bad=1; head -n 300 $W/d-$g > "$REP/$g.diff"; fi
  done
  echo >> "$REP/summary.md"
  [ -f "$REP/avifdec-failed.txt" ] && echo "avifdec (libavif 0.11) could not decode $(wc -l < "$REP/avifdec-failed.txt") AVIF files; av1-rs output for them is checked through their items only (avifdec-failed.txt)." >> "$REP/summary.md"
  [ -f "$REP/avifdec-new-failed.txt" ] && echo "avifdec-new (libavif 1.4.2) could not decode $(wc -l < "$REP/avifdec-new-failed.txt") AVIF files (avifdec-new-failed.txt)." >> "$REP/summary.md"
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
  # The scalar tier on aarch64 too (the NEON tier is the default run above).
  AV1_SIMD=scalar qemu-aarch64 -L /usr/aarch64-linux-gnu "$(command -v av1-oracle-arm64)" $flag --threads 1 $W/in $W/arm-scalar || exit 1
  diff -r "$1" $W/arm-scalar || exit 1
  [ "$(ls $W/in | wc -l)" -ge 50 ] || { echo "arm64 subset has only $(ls $W/in | wc -l) streams, want at least 50" >&2; exit 1; }
  diff -r "$1" $W/arm && echo "arm64 matches x86_64 on $(ls $W/in | wc -l) streams"
}

seeds() {
  rm -rf $W; mkdir -p $W/in "$1"
  inputs $W/in
  for p in $W/in/*; do
    b=$(basename "$p")
    # Option byte: bit 0 Annex B, bit 1 two threads (the parallel tile and grid paths).
    o=0; case $b in *.annexb) o=1 ;; esac
    cp "$p" "$1/$b" && printf "\\$(printf %03o $o)" >> "$1/$b"
    cp "$p" "$1/t2-$b" && printf "\\$(printf %03o $((o + 2)))" >> "$1/t2-$b"
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
