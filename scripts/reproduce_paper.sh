#!/usr/bin/env bash
# Regenerate every number, table and figure in the report from one seeded simulation, and check the numbers against
# paper/results.sha256. Writes paper/data/results.json, paper/data/numbers.json, paper/tables/*.tex and
# paper/figures/*.pdf. Requires the `paper` extra (matplotlib).
set -euo pipefail
cd "$(dirname "$0")/../paper"
python make_data.py > /dev/null
python check_results.py
python figures.py
echo "Wrote paper/data, paper/tables and paper/figures."
