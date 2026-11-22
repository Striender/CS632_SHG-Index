#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/conf_runs"
R="../.."

echo "=== Starting Conference Evaluation Suite ==="
date

echo "1/3 Running Enron (d=1369)..."
../bench_conference $R/enron/enron.data_new $R/enron/enron_query.fvecs $R/enron/enron_groundtruth.ivecs 200 enron > run_enron.log 2>&1
echo "Enron finished $(date +%T)"

echo "2/3 Running SIFT1M (d=128)..."
../bench_conference $R/sift/sift.data_new $R/sift/sift_query.fvecs $R/sift/sift_groundtruth.ivecs 10000 sift > run_sift.log 2>&1
echo "SIFT1M finished $(date +%T)"

echo "3/3 Running GIST1M (d=960)..."
../bench_conference $R/gist/gist.data_new $R/gist/gist_query.fvecs $R/gist/gist_groundtruth.ivecs 1000 gist > run_gist.log 2>&1
echo "GIST1M finished $(date +%T)"

echo "=== Generating All Academic Publication Plots ==="
python3 ../plot_conference_results.py .
echo "=== ALL COMPLETE ==="
date
