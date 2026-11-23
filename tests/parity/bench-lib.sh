# bench-lib.sh: sourced by bench.sh and profile.sh (CI only). `classes <dir>` copies the bench
# inputs from the corpus (/data, /data.list) into one directory per class. Needs av1-avifx,
# dav1d and aomenc on PATH.
CLASSES="q8 q10 allintra size grain avif8 avifhi tiles8 tiles10"

# pick <dst> <ERE>: the corpus files whose path matches, flattened (/ -> __).
pick() {
  mkdir -p "$1"
  grep -E "$2" /data.list | while IFS= read -r f; do
    cp "/data/${f#./}" "$1/$(printf '%s' "${f#./}" | sed 's#/#__#g')"
  done
}

# items <dst> <ERE>: every AV1 coded item of the matching AVIF files, as Section 5 streams
# (examples/avifx.rs), so all decoders read the same input.
items() {
  pick "$1.src" "$2"
  mkdir -p "$1"
  av1-avifx "$1" "$1.src"/*.avif >/dev/null 2>&1 || echo "av1-avifx failed on $1" >&2
  rm -rf "$1.src"
}

# multitile <dst.ivf> <item.obu> <bitdepth>: the item's picture 8 times, every frame a key frame,
# 4 tile columns x 2 tile rows, encoded by aomenc (the corpus has no large multi-tile intra stream).
multitile() {
  t=$(dirname "$1")/mt
  dav1d -q -i "$2" -o "$t.y4m" || return 1
  hl=$(head -n 1 "$t.y4m" | wc -c)
  { head -n 1 "$t.y4m"; for _ in 1 2 3 4 5 6 7 8; do tail -c +$((hl + 1)) "$t.y4m"; done; } > "$t.8.y4m"
  aomenc -q --ivf --passes=1 --limit=8 --kf-max-dist=0 --lag-in-frames=0 --cpu-used=6 \
    --end-usage=q --cq-level=24 --tile-columns=2 --tile-rows=1 --bit-depth="$3" \
    -o "$1" "$t.8.y4m" >/dev/null 2>&1 || echo "aomenc failed for $1" >&2
  rm -f "$t.y4m" "$t.8.y4m"
}

classes() {
  C=$1
  pick $C/q8 'b8-00-quantizer-(00|16|31|48|63)\.ivf$'
  pick $C/q10 'b10-00-quantizer-(00|16|31|48|63)\.ivf$'
  pick $C/allintra 'b8-02-allintra\.ivf$'
  pick $C/size 'b8-01-size-(196|200|208|226)x(196|200|208|226)\.ivf$'
  pick $C/grain 'aom/av1-1-b(8|10)-23-film_grain-50\.ivf$'
  items $C/avif8 '/(fox\.profile0\.8bpc\.yuv420|kimono|kodim03_yuv420_8bpc|kodim23_yuv420_8bpc)\.avif$'
  items $C/avifhi '/(fox\.profile0\.10bpc\.yuv420|fox\.profile2\.12bpc\.yuv444|cosmos1650_yuv444_10bpc_p3pq|hato\.profile0\.10bpc\.yuv420\.no-cdef\.no-restoration)\.avif$'
  mkdir -p $C/tiles8 $C/tiles10
  multitile $C/tiles8/fox8-tiles4x2.ivf "$(ls $C/avif8/*fox.profile0.8bpc*.obu | head -n 1)" 8
  multitile $C/tiles10/fox10-tiles4x2.ivf "$(ls $C/avifhi/*fox.profile0.10bpc*.obu | head -n 1)" 10
}
