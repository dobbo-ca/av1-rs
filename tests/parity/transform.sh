#!/bin/sh
# Transform gate of av1-rs (docs/parity.md, "Transforms"). CI only, called by .github/ci/Dockerfile
# (stage `transform`) from the repo root:
#   transform.sh <report-dir>   libheif (with dav1d) against `decode_avif` with
#                               Options::apply_transforms, on the corpus AVIF files and on files
#                               that avifenc writes with clap, irot and imir; the report is
#                               summary.md and transform.diff; the caller writes `status`
#                               (0 only when the gate passed)
# Binaries on PATH: av1-oracle (examples/oracle.rs), heif-transform-oracle
# (tests/parity/c/transform_oracle.c) and avifenc. The corpus is in /data (/data.list).
# Environment: TRANSFORM_MIN_CORPUS, TRANSFORM_MIN_TRANSFORMED and TRANSFORM_MIN_GENERATED: the
# least number of corpus images, of corpus images with a transform property and of generated images
# that both decoders decode, so a tool that did not work fails the gate instead of shrinking it.
set -u
export LC_ALL=C
W=/work-tr
MIN_CORPUS=${TRANSFORM_MIN_CORPUS:-100}
MIN_TR=${TRANSFORM_MIN_TRANSFORMED:-5}
MIN_GEN=${TRANSFORM_MIN_GENERATED:-60}

# gen <dst>: 4:2:0, 4:2:2 and 4:4:4 images of three sizes (a gradient with some noise, so a wrong
# row or column shows) written by avifenc with every combination of clap, irot and imir below. A
# combination that avifenc refuses (a crop that 4:2:0 cannot hold) is skipped.
gen() {
  mkdir -p $W/gen
  for sz in 64x48 63x47 33x17; do
    w=${sz%x*} h=${sz#*x}
    cw=$((w / 4 * 2)) ch=$((h / 4 * 2))
    for c in 420 422 444; do
      tag=C${c}; [ "$c" = 420 ] && tag=C420jpeg
      perl -e '($w, $h, $c, $t) = @ARGV; print "YUV4MPEG2 W$w H$h F30:1 Ip A1:1 $t\nFRAME\n";
        ($cw, $ch) = $c == 444 ? ($w, $h) : $c == 422 ? (int(($w + 1) / 2), $h) : (int(($w + 1) / 2), int(($h + 1) / 2));
        for $y (0 .. $h - 1) { print pack("C*", map { ($_ * 7 + $y * 13 + ($_ * $y) % 31) % 256 } 0 .. $w - 1) }
        for $p (1, 2) { for $y (0 .. $ch - 1) { print pack("C*", map { ($_ * 5 + $y * 11 + $p * 40 + ($_ * $y) % 17) % 256 } 0 .. $cw - 1) } }' \
        $w $h $c $tag > $W/gen/src.y4m
      i=0
      while IFS= read -r opts; do
        i=$((i + 1))
        o=$1/gen_${w}x${h}_${c}_$i.avif
        avifenc -s 10 --min 20 --max 30 --jobs 1 -y $c $opts $W/gen/src.y4m $o >/dev/null 2>&1 || rm -f $o
      done <<EOF

--irot 1
--irot 2
--irot 3
--imir 0
--imir 1
--clap $cw,1,$ch,1,0,1,0,1
--clap $cw,1,$ch,1,4,1,-2,1
--clap $cw,1,$ch,1,0,1,0,1 --irot 1
--irot 3 --imir 1
--clap $cw,1,$ch,1,0,1,0,1 --irot 2 --imir 0
--clap $cw,1,$ch,1,4,1,-2,1 --irot 1 --imir 1
EOF
    done
  done
}

# lumacmp <ours.yuv> <libheif.yuv> <bytes> <bit depth>: the first <bytes> bytes (the luma plane)
# are equal, or for 8-bit samples within 2 (libheif converts through RGB, which can rescale the
# luma by a little: docs/parity.md). Prints exact or rgb, status 1 when neither.
lumacmp() {
  if cmp -s -n "$3" "$1" "$2"; then echo exact; return 0; fi
  [ "$4" -eq 8 ] || return 1
  cmp -l -n "$3" "$1" "$2" | awk '{ a = 0; b = 0; for (i = 1; i <= length($2); i++) a = a * 8 + substr($2, i, 1);
    for (i = 1; i <= length($3); i++) b = b * 8 + substr($3, i, 1);
    d = a - b; if (d < 0) d = -d; if (d > 2) bad = 1 } END { exit bad }' || return 1
  echo rgb
}

full() {
  REP=$1 bad=0
  rm -rf $W; mkdir -p "$REP" $W/in $W/raw $W/ours $W/lh
  grep '\.avif$' /data.list | while IFS= read -r f; do
    n=$(printf '%s' "${f#./}" | sed 's#/#__#g')
    # libavif poc_* files are security reproducers (hostile input): fuzz seeds, no oracle.
    case $n in libavif__poc_*) continue ;; esac
    ln -sf "/data/${f#./}" "$W/in/$n"
  done
  cn=$(ls $W/in | wc -l)
  gen $W/in
  gn=$(( $(ls $W/in | wc -l) - cn ))
  # raw (the default), with the option on, and libheif with its transforms applied
  av1-oracle $W/in $W/raw || bad=1
  av1-oracle --transforms $W/in $W/ours || bad=1
  heif-transform-oracle $W/in $W/lh 2> $W/lh.err || bad=1
  : > $W/transform.diff
  c_exact=0 c_doc=0 c_rgb=0 c_skip=0 c_tr=0 c_cmp=0 g_exact=0 g_doc=0 g_rgb=0 g_skip=0 g_cmp=0 g_tr=0
  for p in $W/in/*; do
    f=${p##*/}; s=${f%.*}
    case $f in gen_*) k=g ;; *) k=c ;; esac
    if [ -f $W/ours/$s.err ]; then
      if [ -f $W/lh/$s.err ]; then eval "${k}_skip=\$((${k}_skip + 1))"
      else echo "$f: av1-rs error $(cat $W/ours/$s.err), libheif decoded it" >> $W/transform.diff; fi
      continue
    fi
    # libheif refuses it (or hits its own limits) and av1-rs decodes: not compared, counted.
    [ -f $W/lh/$s.err ] && { eval "${k}_skip=\$((${k}_skip + 1))"; continue; }
    [ -f $W/lh/$s.frames ] || { echo "$f: libheif wrote nothing" >> $W/transform.diff; continue; }
    ntr=$(cat $W/ours/$s.tr 2>/dev/null || echo 0)
    eval "${k}_cmp=\$((${k}_cmp + 1))"
    [ "$ntr" -gt 0 ] && eval "${k}_tr=\$((${k}_tr + 1))"
    read -r ow oh obd olay < $W/ours/$s.frames
    read -r lw lhh lbd llay < $W/lh/$s.frames
    read -r _ _ _ rlay < $W/raw/$s.frames 2>/dev/null || rlay=none
    if [ "$ow $oh $obd $olay" = "$lw $lhh $lbd $llay" ] && cmp -s $W/ours/$s.yuv $W/lh/$s.yuv; then
      eval "${k}_exact=\$((${k}_exact + 1))"; continue
    fi
    # The documented difference (docs/parity.md): libheif converts a 4:2:0 or 4:2:2 image to 4:4:4
    # for odd geometry, and av1-rs keeps the format (or, for 4:2:2 rotated by 90 or 270, makes
    # 4:4:4 by another rule). Same size and depth, and the chroma format of either decoder differs
    # from the raw one: the luma plane is compared.
    b=1; [ "$obd" -gt 8 ] && b=2
    if [ "$ow $oh $obd" = "$lw $lhh $lbd" ] && { [ "$llay" != "$rlay" ] || [ "$olay" != "$rlay" ]; }; then
      how=$(lumacmp $W/ours/$s.yuv $W/lh/$s.yuv $((ow * oh * b)) "$obd") && {
        case $how in exact) eval "${k}_doc=\$((${k}_doc + 1))" ;; *) eval "${k}_rgb=\$((${k}_rgb + 1))" ;; esac
        continue
      }
      echo "$f: luma differs from libheif (raw $rlay, av1-rs $olay, libheif $llay, ${ow}x$oh, $obd-bit, $ntr transforms)" >> $W/transform.diff
    else
      echo "$f: differs from libheif (raw $rlay; av1-rs $ow $oh $obd $olay; libheif $lw $lhh $lbd $llay; $ntr transforms)" >> $W/transform.diff
    fi
  done
  : > $W/d-cover
  [ "$c_cmp" -ge "$MIN_CORPUS" ] || echo "only $c_cmp corpus images compared, want at least $MIN_CORPUS" >> $W/d-cover
  [ "$c_tr" -ge "$MIN_TR" ] || echo "only $c_tr corpus images with transform properties compared, want at least $MIN_TR" >> $W/d-cover
  [ "$g_cmp" -ge "$MIN_GEN" ] || echo "only $g_cmp generated images compared, want at least $MIN_GEN" >> $W/d-cover
  {
    echo "## av1-rs transform parity (libheif 1.23.7 with dav1d)"; echo
    echo "| set | files | compared | with transform properties | exact | luma exact, chroma format differs (documented) | luma within 2 via RGB (documented) | not compared (both fail, or libheif refuses) |"
    echo "|---|---|---|---|---|---|---|---|"
    echo "| corpus | $cn | $c_cmp | $c_tr | $c_exact | $c_doc | $c_rgb | $c_skip |"
    echo "| generated (avifenc) | $gn | $g_cmp | $g_tr | $g_exact | $g_doc | $g_rgb | $g_skip |"
    echo; echo "| gate | result |"; echo "|---|---|"
  } > "$REP/summary.md"
  for g in transform.diff d-cover; do
    lines=$(wc -l < $W/$g); lines=$((lines + 0))
    if [ "$lines" -eq 0 ]; then echo "| $g | ok |" >> "$REP/summary.md"
    else echo "| $g | FAIL, $lines lines |" >> "$REP/summary.md"; bad=1; fi
  done
  cat $W/d-cover $W/transform.diff | head -n 100 | sed 's/^/    /' >> "$REP/summary.md"
  head -n 5000 $W/transform.diff > "$REP/transform.diff"
  head -n 5000 $W/lh.err > "$REP/libheif.err" 2>/dev/null
  rm -rf $W
  return $bad
}

[ $# -eq 1 ] || { echo "usage: transform.sh <report-dir>" >&2; exit 2; }
full "$1"
