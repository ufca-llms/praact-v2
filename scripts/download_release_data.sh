#!/usr/bin/env bash
set -euo pipefail

RELEASE_API_URL="https://api.github.com/repos/ufca-llms/praact-v2/releases/tags/data"
OUTPUT_DIR="${1:-data}"
TMP_DIR="$(mktemp -d)"

cleanup() {
  rm -rf "$TMP_DIR"
}
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"

curl -fsSL "$RELEASE_API_URL" | \
grep '"browser_download_url":' | \
sed -E 's/.*"([^"]+)".*/\1/' | \
while read -r url; do
  file="$TMP_DIR/${url##*/}"
  echo "Downloading $url"
  curl -fL "$url" -o "$file"

  case "$file" in
    *.zip)
      unzip -o "$file" -d "$OUTPUT_DIR"
      ;;
    *.tar.gz|*.tgz)
      tar -xzf "$file" -C "$OUTPUT_DIR"
      ;;
    *)
      cp "$file" "$OUTPUT_DIR/"
      ;;
  esac
done

echo "Data extracted to $OUTPUT_DIR"
