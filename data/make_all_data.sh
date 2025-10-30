#!/usr/bin/env bash
# Batch-generate rule datasets starting at a given rule index up through rule 25 inclusive.
# Usage:
#   ./make_all_data.sh START_RULE
# Example:
#   ./make_all_data.sh 6   # regenerates rule_6, rule_7, ..., rule_25

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 START_RULE" >&2
  echo "Example: $0 6" >&2
  exit 1
fi

START_RULE="$1"
END_RULE=25

if ! [[ "${START_RULE}" =~ ^[0-9]+$ ]]; then
  echo "START_RULE must be an integer." >&2
  exit 1
fi

if (( START_RULE > END_RULE )); then
  echo "START_RULE must be less than or equal to ${END_RULE}." >&2
  exit 1
fi

for (( rule=START_RULE; rule<=END_RULE; rule++ )); do
  printf '\n=== Generating dataset for rule_%d ===\n' "${rule}"
  python3 "${SCRIPT_DIR}/make_data.py" \
    --rule-number "${rule}" \
    --examples 300 \
    --output-dir "${SCRIPT_DIR}"
done

echo -e "\nAll datasets generated up to rule_${END_RULE}."

