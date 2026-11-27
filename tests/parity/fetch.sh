#!/bin/sh
# fetch.sh <corpus.txt> <dir>: download every file of corpus.txt into <dir>/<local name> once, at
# most 20 MB/s, and check it against its pinned hash: `sha1` (sha1sum, libaom's test-data.sha1)
# or `blob` (git hash-object, the blob id at the pinned commit of the source repo). A file with
# its stamp (<file>.ok, written after the hash matched) is not downloaded again, so a second run
# is a no-op. Run it inside the BuildKit cache mount id=av1-rs-data, sharing=locked.
set -eu
[ $# -eq 2 ] || { echo "usage: fetch.sh <corpus.txt> <dir>" >&2; exit 2; }
list=$1 dir=$2
grep -v '^#' "$list" | while read -r kind sum url name; do
  [ -n "$name" ] || continue
  f=$dir/$name
  [ -f "$f.ok" ] && [ -f "$f" ] && continue
  mkdir -p "$(dirname "$f")"
  rm -f "$f.ok"
  # A host can answer 200 with another body for a while (a rate limit or bot check page), so a
  # wrong hash is tried again, 3 times in all, before it fails.
  for try in 1 2 3; do
    curl -fsSL --retry 3 --limit-rate 20M -o "$f.part" "$url"
    case $kind in
      sha1) have=$(sha1sum "$f.part" | cut -d' ' -f1) ;;
      blob) have=$(git hash-object "$f.part") ;;
      *) echo "$name: unknown hash kind $kind" >&2; exit 1 ;;
    esac
    [ "$have" = "$sum" ] && break
    echo "$name: hash $have, want $sum (try $try of 3)" >&2
    rm -f "$f.part"
    [ "$try" = 3 ] && exit 1
    sleep $((try * 20))
  done
  mv "$f.part" "$f"
  touch "$f.ok"
done
echo "av1-rs-data: $(find "$dir" -type f ! -name '*.ok' | wc -l) files, $(du -sh "$dir" | cut -f1) in $dir"
