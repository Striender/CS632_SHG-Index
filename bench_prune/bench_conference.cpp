#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>
#include <numeric>
#include <unordered_set>
#include <omp.h>
#include <sched.h>
#include <sys/resource.h>

#include "hnswlib_patched/hnswlib.h"

using namespace std;

static long rss_kb() {
    FILE* f = fopen("/proc/self/status", "r");
    if (!f) return 0;
    char l[256]; long v = 0;
    while (fgets(l, 256, f)) {
        if (!strncmp(l, "VmRSS:", 6)) v = atol(l + 6);
    }
    fclose(f);
    return v;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <base.data_new> <query.fvecs> <groundtruth.ivecs> [NQ] [dataset_name]\n", argv[0]);
        return 1;
    }
    string fb = argv[1], fq = argv[2], fg = argv[3];
    int NQ = (argc > 4) ? atoi(argv[4]) : 1000000;
    string dataset = (argc > 5) ? argv[5] : "dataset";
    const int K = 20;

    // Load Queries
    int d = 0;
    vector<float> Q;
    bool bv = fq.size() > 6 && fq.substr(fq.size() - 6) == ".bvecs";
    {
        FILE* f = fopen(fq.c_str(), "rb");
        if (!f) { fprintf(stderr, "Cannot open query file %s\n", fq.c_str()); return 1; }
        int dim, n = 0;
        while (n < NQ && fread(&dim, 4, 1, f) == 1) {
            vector<float> t(dim);
            if (bv) {
                vector<unsigned char> b(dim);
                (void)fread(b.data(), 1, dim, f);
                for (int j = 0; j < dim; j++) t[j] = b[j];
            } else {
                (void)fread(t.data(), 4, dim, f);
            }
            Q.insert(Q.end(), t.begin(), t.end());
            d = dim;
            n++;
        }
        fclose(f);
        NQ = n;
    }

    // Load Ground Truth
    vector<vector<int>> gt(NQ);
    {
        FILE* f = fopen(fg.c_str(), "rb");
        if (!f) { fprintf(stderr, "Cannot open groundtruth file %s\n", fg.c_str()); return 1; }
        for (int i = 0; i < NQ; i++) {
            int k;
            (void)fread(&k, 4, 1, f);
            vector<int> t(k);
            (void)fread(t.data(), 4, k, f);
            gt[i].assign(t.begin(), t.begin() + min(k, K));
        }
        fclose(f);
    }

    // Load Base Data
    FILE* fbase = fopen(fb.c_str(), "rb");
    if (!fbase) { fprintf(stderr, "Cannot open base file %s\n", fb.c_str()); return 1; }
    int hdr[3];
    (void)fread(hdr, 4, 3, fbase);
    size_t N = hdr[1];
    if (argc > 6) N = min(N, (size_t)atol(argv[6]));
    if (hdr[2] != d) {
        fprintf(stderr, "Dimension mismatch: base=%d query=%d\n", hdr[2], d);
        fclose(fbase);
        return 1;
    }

    printf("======================================================================\n");
    printf("CONFERENCE BENCHMARK: dataset=%s N=%zu d=%d NQ=%d\n", dataset.c_str(), N, d, NQ);
    printf("Available OpenMP Max Threads: %d\n", omp_get_max_threads());
    printf("======================================================================\n");
    fflush(stdout);

    hnswlib::L2Space sp(d);
    Performance per;
    auto T0 = chrono::steady_clock::now();
    auto sec = [&] { return chrono::duration<double>(chrono::steady_clock::now() - T0).count(); };

    auto* S = new hnswlib::HEDS<float>(&sp, d, N + NQ, 48, 80);
    {
        vector<float> x(d);
        for (size_t i = 0; i < N; i++) {
            (void)fread(x.data(), 4, d, fbase);
            S->addDataPoint(x.data(), i, 1, per);
            if (i % 200000 == 0 && i > 0) {
                printf("  Inserted %zu vectors (%.1fs)...\n", i, sec());
                fflush(stdout);
            }
        }
        fclose(fbase);
    }
    double build_time = sec();
    T0 = chrono::steady_clock::now();
    int ni = N;
    S->buildShortcuts(ni);
    double sc_time = sec();

    for (int i = 0; i < NQ; i++) {
        S->addDataPoint(&Q[(size_t)i * d], N + i, -1, per);
    }
    printf("Index Ready: build=%.1fs shortcuts=%.2fs RSS=%.1fMB\n", build_time, sc_time, rss_kb() / 1024.0);
    fflush(stdout);

    const int EF = 64;
    S->setEf(EF);

    // ======================================================================
    // SECTION 1: Analytical Cost Model Calibration & Empirical Lookahead Sweep
    // ======================================================================
    printf("\n----------------------------------------------------------------------\n");
    printf("SECTION 1: Theoretical Cost Model Calibration & Lookahead Sweep (1 Thread)\n");
    printf("----------------------------------------------------------------------\n");

    const void* v1 = S->getDataByInternalId(0);
    const void* v2 = S->getDataByInternalId(min((size_t)1, N - 1));
    auto model_cfg = hnswlib::PrefetchController::compute_optimal_parameters(
        d, sp.get_dist_func(), sp.get_dist_func_param(), v1, v2, 1
    );

    printf("Analytical Cost Model Calibration for d=%d:\n", d);
    printf("  T_DRAM      = %.1f core cycles\n", model_cfg.t_dram);
    printf("  T_SIMD(d)   = %.1f cycles (measured cache-resident distance)\n", model_cfg.t_simd);
    printf("  P_unvisited = %.2f\n", model_cfg.p_unvisited);
    printf("  tau_iter    = %.1f cycles/neighbor\n", model_cfg.tau_iter);
    printf("  ==> Predicted Optimal Lookahead D*(d) = %zu, Prefetch Lines L* = %zu\n",
           model_cfg.d_star, model_cfg.l_star);
    fflush(stdout);

    struct SingleThreadEval {
        string name;
        hnswlib::PrefetchMode mode;
        size_t dist;
        size_t lines;
        bool filter;
    };

    vector<SingleThreadEval> st_configs = {
        {"HNSW_baseline", hnswlib::PrefetchMode::BASELINE, 0, 0, false},
        {"HNSW_stock", hnswlib::PrefetchMode::STOCK_HNSW, 0, 0, false},
        {"Pf_Filter_D1", hnswlib::PrefetchMode::IDEA2_V2, 1, 3, true},
        {"Pf_Filter_D2", hnswlib::PrefetchMode::IDEA2_V2, 2, 3, true},
        {"Pf_Filter_D3", hnswlib::PrefetchMode::IDEA2_V2, 3, 3, true},
        {"Pf_Filter_D4", hnswlib::PrefetchMode::IDEA2_V2, 4, 3, true},
        {"Pf_Filter_D6", hnswlib::PrefetchMode::IDEA2_V2, 6, 3, true},
        {"Pf_Filter_AutoTune", hnswlib::PrefetchMode::IDEA2_V2, model_cfg.d_star, model_cfg.l_star, true}
    };

    string fn_autotune = "results_autotune_" + dataset + ".csv";
    FILE* f_at = fopen(fn_autotune.c_str(), "w");
    fprintf(f_at, "dataset,dim,method,lookahead_D,prefetch_lines_L,predicted_D_star,recall,lat_mean_us,lat_p50_us,lat_p95_us,lat_p99_us,qps\n");

    // Save original affinity and pin thread for single-threaded evaluation
    cpu_set_t orig_cs;
    sched_getaffinity(0, sizeof(orig_cs), &orig_cs);
    cpu_set_t cs;
    CPU_ZERO(&cs);
    CPU_SET(2, &cs);
    sched_setaffinity(0, sizeof(cs), &cs);

    for (const auto& cfg : st_configs) {
        hnswlib::PrefetchController::set_mode(cfg.mode);
        if (cfg.mode == hnswlib::PrefetchMode::IDEA2_V2) {
            hnswlib::PrefetchController::v2_dist = cfg.dist;
            hnswlib::PrefetchController::v2_lines = cfg.lines;
            hnswlib::PrefetchController::v2_warmup = true;
            hnswlib::PrefetchController::v2_filter = cfg.filter;
            hnswlib::PrefetchController::v2_push_ll = true;
        }

        // Warmup
        for (int i = 0; i < min(NQ, 50); i++) {
            Query q(N + i, K);
            (void)S->searchKnnHNSW(q);
        }

        vector<double> lats(NQ);
        float total_hits = 0;
        for (int i = 0; i < NQ; i++) {
            auto t_start = chrono::steady_clock::now();
            Query q(N + i, K);
            auto r = S->searchKnnHNSW(q);
            auto t_end = chrono::steady_clock::now();
            lats[i] = chrono::duration<double, micro>(t_end - t_start).count();

            unordered_set<int> gt_set(gt[i].begin(), gt[i].end());
            int hits = 0;
            while (!r.empty()) {
                if (gt_set.count(r.top().second)) hits++;
                r.pop();
            }
            total_hits += (float)hits / K;
        }

        float recall = total_hits / NQ;
        vector<double> sorted_lats = lats;
        sort(sorted_lats.begin(), sorted_lats.end());
        double lat_mean = accumulate(lats.begin(), lats.end(), 0.0) / NQ;
        double lat_p50 = sorted_lats[(size_t)(0.50 * NQ)];
        double lat_p95 = sorted_lats[(size_t)(0.95 * NQ)];
        double lat_p99 = sorted_lats[(size_t)(0.99 * NQ)];
        double qps = 1e6 / lat_mean;

        fprintf(f_at, "%s,%d,%s,%zu,%zu,%zu,%.4f,%.2f,%.2f,%.2f,%.2f,%.1f\n",
                dataset.c_str(), d, cfg.name.c_str(), cfg.dist, cfg.lines, model_cfg.d_star,
                recall, lat_mean, lat_p50, lat_p95, lat_p99, qps);
        printf("  %-20s | D=%zu L=%zu | Recall=%.4f | Lat: mean=%6.1fus p50=%6.1fus p99=%6.1fus | QPS=%7.1f\n",
               cfg.name.c_str(), cfg.dist, cfg.lines, recall, lat_mean, lat_p50, lat_p99, qps);
        fflush(stdout);
    }
    fclose(f_at);
    printf("Wrote Section 1 results to %s\n", fn_autotune.c_str());

    // ======================================================================
    // SECTION 2: Multi-Threaded Concurrency Scaling (1, 2, 4, 8, 16, 32, 64 Threads)
    // ======================================================================
    printf("\n----------------------------------------------------------------------\n");
    printf("SECTION 2: Multi-Threaded Concurrency Scaling & Contention Adaptation\n");
    printf("----------------------------------------------------------------------\n");

    // Restore original unpinned affinity to utilize all CPU cores
    sched_setaffinity(0, sizeof(orig_cs), &orig_cs);

    vector<int> thread_counts = {1, 2, 4, 8, 16, 32, 64};
    int n_reps = (NQ < 1000) ? 10 : ((NQ < 5000) ? 3 : 1);
    int total_eval_queries = NQ * n_reps;

    string fn_mt = "results_multithread_" + dataset + ".csv";
    FILE* f_mt = fopen(fn_mt.c_str(), "w");
    fprintf(f_mt, "dataset,dim,threads,method,qps,speedup_vs_1thread,gain_vs_baseline,lat_mean_us,lat_p50_us,lat_p95_us,lat_p99_us,total_queries,wall_sec\n");

    struct MTMethod {
        string name;
        int mode_id; // 0=baseline, 1=stock, 2=pf_filter_static, 3=pf_filter_adaptive
    };
    vector<MTMethod> mt_methods = {
        {"HNSW_baseline", 0},
        {"HNSW_stock", 1},
        {"Pf_Filter_Static", 2},
        {"Pf_Filter_Adaptive", 3}
    };

    // Store 1-thread QPS for speedup computation
    unordered_map<string, double> base_qps_1t;

    for (int T : thread_counts) {
        printf("\n>>> Evaluating Concurrency: %2d Threads (Queries per run: %d) <<<\n", T, total_eval_queries);
        double base_qps_current_T = 1.0;
        for (const auto& method : mt_methods) {
            // Adaptive parameters based on thread concurrency
            size_t eff_d = model_cfg.d_star;
            size_t eff_l = model_cfg.l_star;
            if (method.mode_id == 3) { // Adaptive
                if (T >= 32) {
                    eff_l = 1; // Throttle cache lines to prevent MSHR saturation
                    if (eff_d > 1) eff_d -= 1;
                } else if (T >= 16) {
                    eff_l = min(eff_l, (size_t)2);
                }
            } else if (method.mode_id == 2) { // Static
                eff_d = 2;
                eff_l = 3;
            }

            vector<double> latencies(total_eval_queries, 0.0);

            auto t_wall_start = chrono::steady_clock::now();

            #pragma omp parallel num_threads(T)
            {
                // Set per-thread prefetch configuration
                if (method.mode_id == 0) {
                    hnswlib::PrefetchController::set_mode(hnswlib::PrefetchMode::BASELINE);
                } else if (method.mode_id == 1) {
                    hnswlib::PrefetchController::set_mode(hnswlib::PrefetchMode::STOCK_HNSW);
                } else {
                    hnswlib::PrefetchController::set_mode(hnswlib::PrefetchMode::IDEA2_V2);
                    hnswlib::PrefetchController::v2_dist = eff_d;
                    hnswlib::PrefetchController::v2_lines = eff_l;
                    hnswlib::PrefetchController::v2_warmup = true;
                    hnswlib::PrefetchController::v2_filter = true;
                    hnswlib::PrefetchController::v2_push_ll = true;
                }

                #pragma omp for schedule(dynamic, 16)
                for (int qi = 0; qi < total_eval_queries; qi++) {
                    int qid = qi % NQ;
                    auto q_t0 = chrono::steady_clock::now();
                    Query q(N + qid, K);
                    auto r = S->searchKnnHNSW(q);
                    auto q_t1 = chrono::steady_clock::now();
                    latencies[qi] = chrono::duration<double, micro>(q_t1 - q_t0).count();
                }
            }

            auto t_wall_end = chrono::steady_clock::now();
            double wall_sec = chrono::duration<double>(t_wall_end - t_wall_start).count();
            double qps = (double)total_eval_queries / wall_sec;

            if (T == 1) {
                base_qps_1t[method.name] = qps;
            }

            double speedup_vs_1t = qps / base_qps_1t[method.name];

            // Latency percentiles
            vector<double> sorted_lat = latencies;
            sort(sorted_lat.begin(), sorted_lat.end());
            double lat_mean = accumulate(latencies.begin(), latencies.end(), 0.0) / total_eval_queries;
            double lat_p50 = sorted_lat[(size_t)(0.50 * total_eval_queries)];
            double lat_p95 = sorted_lat[(size_t)(0.95 * total_eval_queries)];
            double lat_p99 = sorted_lat[(size_t)(0.99 * total_eval_queries)];

            if (method.mode_id == 0) {
                base_qps_current_T = qps;
            }
            double gain_vs_base = (base_qps_current_T > 0.0) ? (qps / base_qps_current_T) : 1.0;
            // Store temporarily
            fprintf(f_mt, "%s,%d,%d,%s,%.1f,%.2f,%.4f,%.2f,%.2f,%.2f,%.2f,%d,%.3f\n",
                    dataset.c_str(), d, T, method.name.c_str(), qps, speedup_vs_1t, gain_vs_base,
                    lat_mean, lat_p50, lat_p95, lat_p99, total_eval_queries, wall_sec);
            printf("  [%2d Threads] %-20s | QPS=%9.1f | Speedup=%5.2fx | Lat: mean=%6.1fus p50=%6.1fus p99=%6.1fus\n",
                   T, method.name.c_str(), qps, speedup_vs_1t, lat_mean, lat_p50, lat_p99);
            fflush(stdout);
        }
        fflush(f_mt);
    }
    fclose(f_mt);
    printf("\nWrote Section 2 results to %s\n", fn_mt.c_str());

    printf("\n======================================================================\n");
    printf("COMPLETED ALL BENCHMARKS FOR DATASET: %s\n", dataset.c_str());
    printf("======================================================================\n");
    return 0;
}
