#!/usr/bin/env bash
set -euo pipefail

FILE_ID="1-aE1NfzpRCLxA4GUxX9ITI3F9LlbtEGP"
OUTPUT_RELATIVE_PATH="ckpts/speaker/wavlm_large_finetune.pth"
MIN_BYTES=$((1024 * 1024 * 1024))

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
OUTPUT_PATH="${REPO_ROOT}/${OUTPUT_RELATIVE_PATH}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required to download the speaker checkpoint." >&2
  exit 1
fi

existing_size=0
if [[ -f "${OUTPUT_PATH}" ]]; then
  existing_size="$(wc -c < "${OUTPUT_PATH}" | tr -d ' ')"
fi

if [[ "${existing_size}" -ge "${MIN_BYTES}" ]]; then
  echo "Speaker checkpoint already exists: ${OUTPUT_RELATIVE_PATH}"
  echo "Size: ${existing_size} bytes"
  exit 0
fi

if [[ -f "${OUTPUT_PATH}" ]]; then
  echo "Removing incomplete checkpoint: ${OUTPUT_RELATIVE_PATH} (${existing_size} bytes)"
  rm -f "${OUTPUT_PATH}"
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

if ! python3 -c "import gdown" >/dev/null 2>&1; then
  echo "Installing gdown for the current user..."
  python3 -m pip install --user gdown
fi

echo "Downloading UniSpeech WavLM-large speaker checkpoint..."
python3 -m gdown "https://drive.google.com/uc?id=${FILE_ID}" -O "${OUTPUT_PATH}"

downloaded_size="$(wc -c < "${OUTPUT_PATH}" | tr -d ' ')"
if [[ "${downloaded_size}" -lt "${MIN_BYTES}" ]]; then
  echo "Downloaded file is unexpectedly small: ${downloaded_size} bytes" >&2
  echo "Delete ${OUTPUT_RELATIVE_PATH} and retry, or download manually from:" >&2
  echo "https://drive.google.com/file/d/${FILE_ID}/view?usp=sharing" >&2
  exit 1
fi

echo "Speaker checkpoint ready: ${OUTPUT_RELATIVE_PATH}"
