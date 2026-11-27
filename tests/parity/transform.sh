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
MIN_CORPUS=${TRANSFORM_MIN_CORPUS:-140}
MIN_TR=${TRANSFORM_MIN_TRANSFORMED:-10}
MIN_GEN=${TRANSFORM_MIN_GENERATED:-70}
# The most images that pass on "luma within 2 via RGB" (the 29 of the run that pinned it). A
# higher count means files left the exact classes: look at transform-files.txt.
MAX_RGB=${TRANSFORM_MAX_RGB:-29}
# The most a chroma sample of av1-rs may be from the range of the libheif samples of its cell, in
# 8-bit units (scaled up for deeper samples), see chrompos.
CHROMA_CAP=${TRANSFORM_CHROMA_CAP:-3}
# The least number of images whose chroma is compared with libheif's by position, and of
# images whose crop is compared sample by sample with av1-rs's raw decode (cropcmp), so these
# checks cannot shrink to nothing.
MIN_CHROMA=${TRANSFORM_MIN_CHROMA:-20}
MIN_CROP=${TRANSFORM_MIN_CROP:-4}
# The least number of files where the self-check (raw chroma with transformed luma) fails
# chrompos, as it must.
MIN_MUT=${TRANSFORM_MIN_MUT:-4}

# gen <dst>: 4:2:0, 4:2:2 and 4:4:4 images of three sizes written by avifenc with every
# combination of clap, irot and imir below. Luma is a gradient with some noise inside the RGB
# gamut, so a wrong row or column shows. Chroma is smooth, with a different slope across and down
# and per plane, so that a chroma plane left unmirrored or rotated the wrong way is far from
# libheif's samples in its window (chrompos; a shift of one chroma sample stays inside it, which
# only cropcmp sees, for a crop alone); it stays within 120..135, near grey, because
# libheif goes through RGB for odd geometry and a colour outside the gamut clips and moves the
# luma (88..168 failed the luma check on 25 files). A combination that avifenc refuses (a crop that
# 4:2:0 or 4:2:2 cannot hold) is skipped.
gen() {
  mkdir -p $W/gen
  for sz in 64x48 63x47 33x17; do
    w=${sz%x*} h=${sz#*x}
    cw=$((w / 4 * 2)) ch=$((h / 4 * 2))
    # A crop with an odd origin (and an odd size for an odd side): (w - ow) is even, so the
    # centre is a whole sample, and the offset 1 makes the origin odd.
    ow=$((w / 2)) oh=$((h / 2))
    [ $(((w - ow) % 2)) -eq 0 ] || ow=$((ow + 1))
    [ $(((h - oh) % 2)) -eq 0 ] || oh=$((oh + 1))
    for c in 420 422 444; do
      tag=C${c}; [ "$c" = 420 ] && tag=C420jpeg
      perl -e '($w, $h, $c, $t) = @ARGV; print "YUV4MPEG2 W$w H$h F30:1 Ip A1:1 $t\nFRAME\n";
        ($cw, $ch) = $c == 444 ? ($w, $h) : $c == 422 ? (int(($w + 1) / 2), $h) : (int(($w + 1) / 2), int(($h + 1) / 2));
        for $y (0 .. $h - 1) { print pack("C*", map { 40 + ($_ * 7 + $y * 13 + ($_ * $y) % 31) % 176 } 0 .. $w - 1) }
        for $p (1, 2) { for $y (0 .. $ch - 1) { print pack("C*", map { $p == 1 ? 120 + int($_ * 12 / $cw) + int($y * 4 / $ch) : 120 + int($y * 12 / $ch) + int($_ * 4 / $cw) } 0 .. $cw - 1) } }' \
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
--clap $ow,1,$oh,1,1,1,1,1
--clap $cw,1,$ch,1,0,1,0,1 --irot 1
--irot 3 --imir 1
--clap $cw,1,$ch,1,0,1,0,1 --irot 2 --imir 0
--clap $cw,1,$ch,1,4,1,-2,1 --irot 1 --imir 1
EOF
    done
  done
}

# lumacmp <ours.yuv> <libheif.yuv> <bytes> <bit depth>: the first <bytes> bytes (the luma plane)
# are equal, or for 8-bit samples libheif's luma is within 2 of ours or of ours rescaled from
# studio to full range, round((v - 16) * 255 / 219) (libheif converts through RGB, which moves the
# luma to another range and rounds it twice: docs/parity.md). Prints exact or rgb, status 1 when
# neither.
lumacmp() {
  if cmp -s -n "$3" "$1" "$2"; then echo exact; return 0; fi
  [ "$4" -eq 8 ] || return 1
  od -An -v -tu1 -w1 -N "$3" "$1" > $W/lum.a
  od -An -v -tu1 -w1 -N "$3" "$2" > $W/lum.b
  paste $W/lum.a $W/lum.b | awk '{ d = $1 - $2; if (d < 0) d = -d; if (d > 2) b0 = 1
    r = int(($1 - 16) * 255 / 219 + 0.5); if (r < 0) r = 0; if (r > 255) r = 255
    d = r - $2; if (d < 0) d = -d; if (d > 2) b1 = 1 } END { exit (b0 && b1) }' || return 1
  echo rgb
}

# chromacmp <ours.yuv> <raw.yuv> <raw luma bytes> <w> <h> <bytes per sample> <layout> <transforms>:
# av1-rs's chroma, which libheif's differs from by design, checked against av1-rs's raw decode
# (no transform property in the file): the file size is the one of <layout>, and the chroma is equal
# to the raw chroma when there is no transform, or the same samples in another order (rotate and
# mirror keep the multiset) when the sizes are equal. A crop (clap) only gets the size check
# here; a crop alone is compared sample by sample in cropcmp.
chromacmp() {
  cw=$4 ch=$5
  case $7 in 420) cw=$((($4 + 1) / 2)); ch=$((($5 + 1) / 2)) ;; 422) cw=$((($4 + 1) / 2)) ;; 400) cw=0 ch=0 ;; esac
  [ "$(wc -c < $1)" -eq $((($4 * $5 + 2 * cw * ch) * $6)) ] || return 1
  [ "$8" -gt 0 ] || { cmp -s $1 $2; return; }
  [ "$(wc -c < $1)" -eq "$(wc -c < $2)" ] || return 0
  tail -c +$(($3 + 1)) $1 | od -An -v -tu$6 -w$6 | sort -n > $W/ch.a
  tail -c +$(($3 + 1)) $2 | od -An -v -tu$6 -w$6 | sort -n > $W/ch.b
  cmp -s $W/ch.a $W/ch.b
}

# chrompos <ours.yuv> <libheif.yuv> <w> <h> <bytes per sample> <ours layout> <libheif layout> <bit depth> <rgb>:
# av1-rs's chroma against libheif's by position, in the output geometry (rotation, mirror and crop
# are already applied to both, so they need no mapping). A chroma sample sits at the luma positions
# of its cell (2x2 for 4:2:0, 2x1 for 4:2:2, one for 4:4:4). Every libheif sample, at its luma
# position, must lie within CHROMA_CAP (scaled to the bit depth) of the range of the av1-rs samples
# whose cells touch the 5x5 luma window around it: libheif's bilinear 4:2:0 and 4:2:2 to 4:4:4 step
# (and, for 4:2:2 rotated by 90 or 270, the interpolation of the sample copied into two columns by
# av1-rs) is a blend of neighbouring samples, and an odd crop or mirror moves av1-rs's chroma by one
# luma sample. With <rgb> = rgb (libheif went through RGB) the av1-rs samples rescaled from studio
# to full range, round((v - 128) * 255 / 224 + 128), may be used instead. Prints exact (every
# libheif sample equals the av1-rs sample of its cell) or cap and the largest distance; status 1
# above the cap.
chrompos() {
  case $6 in 420) sx=2 sy=2 ;; 422) sx=2 sy=1 ;; *) sx=1 sy=1 ;; esac
  case $7 in 420) lsx=2 lsy=2 ;; 422) lsx=2 lsy=1 ;; *) lsx=1 lsy=1 ;; esac
  cw=$((($3 + sx - 1) / sx)) ch=$((($4 + sy - 1) / sy))
  lcw=$((($3 + lsx - 1) / lsx)) lch=$((($4 + lsy - 1) / lsy))
  tail -c +$(($3 * $4 * $5 + 1)) $1 | od -An -v -tu$5 -w$5 > $W/cp.a
  tail -c +$(($3 * $4 * $5 + 1)) $2 | od -An -v -tu$5 -w$5 > $W/cp.b
  # Both chroma parts must be whole, or a missing plane would compare nothing and pass.
  [ "$(wc -l < $W/cp.a)" -eq $((2 * cw * ch)) ] && [ "$(wc -l < $W/cp.b)" -eq $((2 * lcw * lch)) ] ||
    { echo "size"; return 1; }
  awk -v w=$3 -v h=$4 -v cw=$cw -v ch=$ch -v sx=$sx -v sy=$sy -v lcw=$lcw -v lch=$lch -v lsx=$lsx -v lsy=$lsy \
    -v cap=$((CHROMA_CAP << ($8 - 8))) -v rgb="$9" '
    function rs(v,  t) { t = (v - 128) * 255 / 224 + 128.5; return t < 0 ? 0 : int(t) }
    NR == FNR { o[NR - 1] = $1; next }
    { i = FNR - 1; n = lcw * lch; p = int(i / n); r = i - p * n; y = int(r / lcw) * lsy; x = (r - int(r / lcw) * lcw) * lsx
      x0 = x < 2 ? 0 : x - 2; x1 = x + 2 > w - 1 ? w - 1 : x + 2; y0 = y < 2 ? 0 : y - 2; y1 = y + 2 > h - 1 ? h - 1 : y + 2
      lo = 1e9; hi = -1
      for (yy = int(y0 / sy); yy <= int(y1 / sy); yy++) for (xx = int(x0 / sx); xx <= int(x1 / sx); xx++) {
        v = o[p * cw * ch + yy * cw + xx]; if (v < lo) lo = v; if (v > hi) hi = v }
      d = $1 < lo ? lo - $1 : ($1 > hi ? $1 - hi : 0)
      if (rgb == "rgb") { e = $1 < rs(lo) ? rs(lo) - $1 : ($1 > rs(hi) ? $1 - rs(hi) : 0); if (e < d) d = e }
      if (d > worst) worst = d
      if ($1 != o[p * cw * ch + int(y / sy) * cw + int(x / sx)]) ne = 1 }
    END { if (worst > cap) { print "over " worst; exit 1 }
      if (ne) print "cap " worst + 0; else print "exact" }' $W/cp.a $W/cp.b
}

# cropcmp <ours.yuv> <raw.yuv> <raw w> <raw h> <bytes per sample> <layout> <x> <y> <w> <h>: the
# output of a file whose one transform is a crop (`<stem>.crop` of av1-oracle) is, sample by
# sample, the rectangle of av1-rs's raw decode: luma x, y, w, h, and each chroma plane with
# factors sx, sy from floor(x / sx), floor(y / sy), ceil(w / sx) x ceil(h / sy) samples (the
# rule of isobmff::apply_transforms; the chroma format is kept). Status 1 when any sample differs.
cropcmp() {
  case $6 in 420) sx=2 sy=2 ;; 422) sx=2 sy=1 ;; 400) sx=0 sy=0 ;; *) sx=1 sy=1 ;; esac
  od -An -v -tu$5 -w$5 $2 > $W/cr.raw
  awk -v W=$3 -v H=$4 -v sx=$sx -v sy=$sy -v x=$7 -v y=$8 -v w=$9 -v h=${10} '
    { v[NR - 1] = $1 }
    END {
      for (r = y; r < y + h; r++) for (c = x; c < x + w; c++) print v[r * W + c]
      if (sx == 0) exit
      cw = int((W + sx - 1) / sx); ch = int((H + sy - 1) / sy)
      x0 = int(x / sx); y0 = int(y / sy); ow = int((w + sx - 1) / sx); oh = int((h + sy - 1) / sy)
      for (p = 0; p < 2; p++) for (r = y0; r < y0 + oh; r++) for (c = x0; c < x0 + ow; c++)
        print v[W * H + p * cw * ch + r * cw + c]
    }' $W/cr.raw > $W/cr.want
  od -An -v -tu$5 -w$5 $1 | awk '{ print $1 }' > $W/cr.got
  cmp -s $W/cr.got $W/cr.want
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
  : > $W/transform.diff; : > $W/files.txt
  mut=0 crops=0 c_cx=0 c_cc=0 g_cx=0 g_cc=0 c_exact=0 c_doc=0 c_rgb=0 c_skip=0 c_tr=0 c_cmp=0 g_exact=0 g_doc=0 g_rgb=0 g_skip=0 g_cmp=0 g_tr=0
  for p in $W/in/*; do
    f=${p##*/}; s=${f%.*}
    case $f in gen_*) k=g ;; *) k=c ;; esac
    if [ -f $W/ours/$s.err ]; then
      # inter: a layered item (Xiph) whose frames av1-rs stops at, as in run.sh; libheif decodes one layer.
      if [ -f $W/lh/$s.err ] || [ "$(cat $W/ours/$s.err)" = inter ]; then eval "${k}_skip=\$((${k}_skip + 1))"
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
    [ -f $W/raw/$s.frames ] || { echo "$f: no raw decode of av1-rs" >> $W/transform.diff; continue; }
    read -r rw rh _ rlay < $W/raw/$s.frames
    # A crop alone is checked against av1-rs's raw decode sample by sample, whatever libheif gives.
    if [ -f $W/ours/$s.crop ]; then
      read -r cx cy cwid chei < $W/ours/$s.crop
      rb=1; [ "$obd" -gt 8 ] && rb=2
      cropcmp $W/ours/$s.yuv $W/raw/$s.yuv $rw $rh $rb "$rlay" $cx $cy $cwid $chei ||
        { echo "$f: av1-rs crop differs from its raw decode (crop $cx,$cy ${cwid}x$chei of raw ${rw}x$rh $rlay)" >> $W/transform.diff; continue; }
      crops=$((crops + 1))
    fi
    if [ "$ow $oh $obd $olay" = "$lw $lhh $lbd $llay" ] && cmp -s $W/ours/$s.yuv $W/lh/$s.yuv; then
      eval "${k}_exact=\$((${k}_exact + 1))"; echo "$f exact" >> $W/files.txt; continue
    fi
    # The documented difference (docs/parity.md): libheif converts a 4:2:0 or 4:2:2 image to 4:4:4
    # for odd geometry, and av1-rs keeps the format (or, for 4:2:2 rotated by 90 or 270, makes
    # 4:4:4 by another rule). Same size and depth, libheif's chroma format differs from the raw
    # one, and av1-rs's is the expected one (the raw one, or 4:4:4 for raw 4:2:2): the luma plane
    # is compared with libheif's, and the chroma of av1-rs with its own raw decode.
    b=1; [ "$obd" -gt 8 ] && b=2
    if [ "$ow $oh $obd" = "$lw $lhh $lbd" ] && [ "$llay" != "$rlay" ] &&
      { [ "$olay" = "$rlay" ] || { [ "$rlay" = 422 ] && [ "$olay" = 444 ]; }; }; then
      how=$(lumacmp $W/ours/$s.yuv $W/lh/$s.yuv $((ow * oh * b)) "$obd") && {
        # A layout that differs from the raw one (4:2:2 turned 4:4:4) has no raw chroma to compare.
        if [ "$olay" = "$rlay" ]; then
          chromacmp $W/ours/$s.yuv $W/raw/$s.yuv $((ow * oh * b)) $ow $oh $b "$olay" "$ntr" ||
            { echo "$f: av1-rs chroma differs from its raw decode (raw $rlay, ${ow}x$oh, $obd-bit, $ntr transforms)" >> $W/transform.diff; continue; }
        fi
        # The chroma by position against libheif's 4:4:4 (or other) chroma, after any rotate, mirror or crop.
        # No chroma to compare by position when either side is 4:0:0.
        cp=none
        if [ "$olay" != 400 ] && [ "$llay" != 400 ]; then
          cp=$(chrompos $W/ours/$s.yuv $W/lh/$s.yuv $ow $oh $b "$olay" "$llay" "$obd" "$how") ||
            { echo "$f: av1-rs chroma differs from libheif by position ($cp, cap $CHROMA_CAP; raw $rlay, av1-rs $olay, libheif $llay, ${ow}x$oh, $obd-bit, $ntr transforms)" >> $W/transform.diff; continue; }
        fi
        # Self-check of chrompos: the same luma with av1-rs's raw (untransformed) chroma, as a
        # decoder that mirrors or rotates the luma only would give, must fail it. Only for an
        # output of the raw size and layout whose chroma the transform changed.
        if [ "$cp" != none ] && [ "$olay" = "$rlay" ] &&
          [ "$(wc -c < $W/ours/$s.yuv)" -eq "$(wc -c < $W/raw/$s.yuv)" ]; then
          lb=$((ow * oh * b))
          { head -c $lb $W/ours/$s.yuv; tail -c +$((lb + 1)) $W/raw/$s.yuv; } > $W/mut.yuv
          if cmp -s $W/mut.yuv $W/ours/$s.yuv; then :
          elif chrompos $W/mut.yuv $W/lh/$s.yuv $ow $oh $b "$olay" "$llay" "$obd" "$how" > /dev/null; then
            echo "$f: chrompos also passes the untransformed raw chroma, so it cannot see a wrong chroma transform here" >> $W/transform.diff
          else mut=$((mut + 1)); fi
        fi
        case $cp in none) ;; exact) eval "${k}_cx=\$((${k}_cx + 1))" ;; *) eval "${k}_cc=\$((${k}_cc + 1))" ;; esac
        case $how in
          exact) eval "${k}_doc=\$((${k}_doc + 1))" ;;
          *) eval "${k}_rgb=\$((${k}_rgb + 1))" ;;
        esac
        echo "$f luma-$how chroma-$cp raw=$rlay ours=$olay libheif=$llay ${ow}x$oh $obd-bit $ntr-transforms" >> $W/files.txt
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
  [ $((c_rgb + g_rgb)) -le "$MAX_RGB" ] || echo "$((c_rgb + g_rgb)) images pass on luma within 2 via RGB, want at most $MAX_RGB" >> $W/d-cover
  [ $((c_cx + c_cc + g_cx + g_cc)) -ge "$MIN_CHROMA" ] || echo "only $((c_cx + c_cc + g_cx + g_cc)) images had their chroma compared by position, want at least $MIN_CHROMA" >> $W/d-cover
  [ "$crops" -ge "$MIN_CROP" ] || echo "only $crops crops compared with the raw decode, want at least $MIN_CROP" >> $W/d-cover
  [ "$mut" -ge "$MIN_MUT" ] || echo "only $mut untransformed-chroma self-checks failed chrompos as they must, want at least $MIN_MUT" >> $W/d-cover
  {
    echo "## av1-rs transform parity (libheif 1.23.7 with dav1d)"; echo
    echo "| set | files | compared | with transform properties | exact | luma exact, chroma format differs (documented) | luma within 2 via RGB (documented) | of those, chroma by position equal | of those, chroma by position within the cap | not compared (both fail, or libheif refuses) |"
    echo "|---|---|---|---|---|---|---|---|---|---|"
    echo "| corpus | $cn | $c_cmp | $c_tr | $c_exact | $c_doc | $c_rgb | $c_cx | $c_cc | $c_skip |"
    echo "| generated (avifenc) | $gn | $g_cmp | $g_tr | $g_exact | $g_doc | $g_rgb | $g_cx | $g_cc | $g_skip |"
    echo; echo "Crops (one \`clap\`, no other transform) equal to the rectangle of the raw decode, sample by sample: $crops. Self-checks (transformed luma, raw chroma) that chrompos refused, as it must: $mut."
    echo; echo "| gate | result |"; echo "|---|---|"
  } > "$REP/summary.md"
  for g in transform.diff d-cover; do
    lines=$(wc -l < $W/$g); lines=$((lines + 0))
    if [ "$lines" -eq 0 ]; then echo "| $g | ok |" >> "$REP/summary.md"
    else echo "| $g | FAIL, $lines lines |" >> "$REP/summary.md"; bad=1; fi
  done
  cat $W/d-cover $W/transform.diff | head -n 100 | sed 's/^/    /' >> "$REP/summary.md"
  head -n 5000 $W/transform.diff > "$REP/transform.diff"
  cp $W/files.txt "$REP/transform-files.txt"
  head -n 5000 $W/lh.err > "$REP/libheif.err" 2>/dev/null
  rm -rf $W
  return $bad
}

[ $# -eq 1 ] || { echo "usage: transform.sh <report-dir>" >&2; exit 2; }
full "$1"
