#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$ROOT/third_party"
mkdir -p "$DEST"

clone_or_update() {
  local url="$1"
  local dir="$2"
  if [[ -d "$dir/.git" ]]; then
    echo "Updating $dir"
    git -C "$dir" pull --ff-only
  else
    echo "Cloning $url -> $dir"
    git clone --depth 1 "$url" "$dir"
  fi
}

clone_or_update "https://github.com/Ayanami0730/deep_research_bench.git" "$DEST/deep_research_bench"
clone_or_update "https://github.com/cxcscmu/deepresearch_benchmarking.git" "$DEST/deepresearch_benchmarking"

echo
echo "Official eval repos are in $DEST"
echo "Install their extras if you will run judges:"
echo "  pip install -r third_party/deep_research_bench/requirements.txt"
echo "  pip install openai crawl4ai   # Gym judges + optional citation crawl"
