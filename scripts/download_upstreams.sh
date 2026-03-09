#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
THIRD_PARTY="$ROOT_DIR/third_party"
mkdir -p "$THIRD_PARTY"

clone_or_skip() {
  local url="$1"
  local dir="$2"
  if [[ -d "$THIRD_PARTY/$dir/.git" ]]; then
    echo "[skip] $dir already cloned"
  else
    echo "[clone] $url -> $THIRD_PARTY/$dir"
    git clone "$url" "$THIRD_PARTY/$dir"
  fi
}

clone_or_skip "https://github.com/microsoft/OmniParser.git" "OmniParser"
clone_or_skip "https://github.com/ultralytics/ultralytics.git" "ultralytics"
clone_or_skip "https://github.com/huggingface/transformers.git" "transformers"
clone_or_skip "https://github.com/huggingface/peft.git" "peft"
clone_or_skip "https://github.com/huggingface/trl.git" "trl"

echo "[ok] Upstream tools are available under $THIRD_PARTY"
