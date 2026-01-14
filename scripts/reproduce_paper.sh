#!/usr/bin/env bash
# Regenerate every number, table and figure in the report from one seeded simulation.
# Writes paper/data/results.json, paper/data/numbers.json, paper/tables/*.tex and paper/figures/*.pdf.
set -euo pipefail
cd "$(dirname "$0")/../paper"
python make_data.py
python figures.py
echo "Wrote paper/data, paper/tables and paper/figures."
