#!/usr/bin/env bash
# CPU-time breakdown per base-layer distance evaluation (stall / compute / overhead), all four datasets.
set -e
cd "$(dirname "$0")"; R=../..
run(){ ds=$1; shift; mkdir -p stall_$ds; (cd stall_$ds && ../bench_stall "$@" > run.log 2>&1); echo "$ds done $(date +%T)"; }
run enron   $R/enron/enron.data_new     $R/enron/enron_query.fvecs    $R/enron/enron_groundtruth.ivecs
run sift    $R/sift/sift.data_new       $R/sift/sift_query.fvecs      $R/sift/sift_groundtruth.ivecs
run gist    $R/gist/gist.data_new       $R/gist/gist_query.fvecs      $R/gist/gist_groundtruth.ivecs
run sift10m $R/sift10m/sift10m.data_new $R/sift10m/bigann_query.bvecs $R/sift10m/gnd/idx_10M.ivecs
