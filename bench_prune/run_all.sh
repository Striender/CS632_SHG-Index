#!/usr/bin/env bash
# Same-graph, per-query-interleaved comparison of HNSW and SHG under every prefetch / pruning setting.
set -e
cd "$(dirname "$0")"; R=../..
run(){ ds=$1; shift; mkdir -p out_$ds; (cd out_$ds && ../bench_prune "$@" > run.log 2>&1); echo "$ds done $(date +%T)"; }
free -g | head -2
run enron   $R/enron/enron.data_new     $R/enron/enron_query.fvecs    $R/enron/enron_groundtruth.ivecs
run sift    $R/sift/sift.data_new       $R/sift/sift_query.fvecs      $R/sift/sift_groundtruth.ivecs
run gist    $R/gist/gist.data_new       $R/gist/gist_query.fvecs      $R/gist/gist_groundtruth.ivecs
run sift10m $R/sift10m/sift10m.data_new $R/sift10m/bigann_query.bvecs $R/sift10m/gnd/idx_10M.ivecs
