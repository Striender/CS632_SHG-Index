#include "../../hnswlib/hnswlib.h"
#include <iostream>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <iomanip>

int main(int argc, char** argv) {
    int dim = -1;
    int num_base = -1;
    int M = 48;
    int ef_construction = 80;

    std::string dataset = (argc > 1) ? argv[1] : "enron";
    int K = (argc > 2) ? std::stoi(argv[2]) : 20;

    std::cout << "======================================================" << std::endl;
    std::cout << "Validating Paper Claims: Vanilla HNSW vs. SHG vs. Step2" << std::endl;
    std::cout << "Parameters from Paper: M = " << M << ", efConstruction = " << ef_construction << ", K = " << K << std::endl;
    std::cout << "Dataset: " << dataset << std::endl;
    std::cout << "======================================================" << std::endl;

    std::string file = resolve_data_file(dataset, dataset + ".data_new");
    std::ifstream loadin(file.c_str(), std::ios::binary);
    if (!loadin) {
        std::cerr << "Fail to find data file: " << file << std::endl;
        return 1;
    }

    unsigned int header[3] = {};
    loadin.read((char*)header, sizeof(header));
    num_base = header[1];
    dim = header[2];

    int num_query = -1;
    float **queryVectors = nullptr;
    int **groundtruth = nullptr;

    if (dataset == "openai" || dataset == "msturing") {
        load_query_groundtruth_(dataset, num_query, queryVectors, groundtruth);
    } else if (dataset == "deep100m" || dataset == "sift100m") {
        load_query_groundtruth100m(dataset, num_query, queryVectors, groundtruth);
    } else {
        load_query_groundtruth(dataset, num_query, queryVectors, groundtruth);
    }

    std::cout << "Loaded base: " << num_base << " points, dim: " << dim 
              << ", queries: " << num_query << std::endl;

    // Load base data
    float** data = new float* [num_base];
    for (int i = 0; i < num_base; ++i) {
        data[i] = new float[dim];
        loadin.read((char*)data[i], sizeof(float) * header[2]);
    }
    loadin.close();

    hnswlib::L2Space space(dim);
    Timer t;
    Performance per;

    int target_ef = (argc > 3) ? std::stoi(argv[3]) : 0;
    std::vector<size_t> ef_list = {20, 30, 40, 60, 80, 120, 200, 300};
    if (target_ef > 0) {
        ef_list = { (size_t)target_ef };
    }
    const int NUM_PASSES = 5;

    std::string csv_path = (dataset == "enron") ? "results/data/hnsw_vs_shg_comparison.csv" : "results/data/hnsw_vs_shg_comparison_" + dataset + ".csv";
    std::ofstream csv;
    if (target_ef == 0) {
        csv.open(csv_path);
        csv << "method,ef,latency_ms,latency_us,recall,qps,build_time_s,memory_mb" << std::endl;
    }

    int eval_queries = (dataset == "sift") ? std::min(2000, num_query) : num_query;
    std::cout << "Evaluating " << eval_queries << " queries per setting..." << std::endl;

    // ================================================================
    // PART 1: Vanilla HNSW (Malkov et al. / HNSW baseline in paper)
    // ================================================================
    std::cout << "\n------------------------------------------------------" << std::endl;
    std::cout << "1. Building Vanilla HNSW (M=48, efConstruction=80)..." << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    hnswlib::HierarchicalNSW<float>* alg_hnsw = new hnswlib::HierarchicalNSW<float>(&space, num_base, M, ef_construction);
    t.restart();
    for (int i = 0; i < num_base; ++i) {
        alg_hnsw->addPoint(data[i], i);
    }
    double hnsw_build_time = t.elapsed();
    double hnsw_memory_mb = alg_hnsw->indexFileSize() / (1024.0 * 1024.0);
    std::cout << "Vanilla HNSW Build Time: " << hnsw_build_time << " s" << std::endl;
    std::cout << "Vanilla HNSW Index Size: " << hnsw_memory_mb << " MB" << std::endl;

    // Warmup HNSW
    alg_hnsw->setEf(30);
    for (int w = 0; w < std::min(20, num_query); ++w) {
        alg_hnsw->searchKnn(queryVectors[w], w, K);
    }

    std::cout << "\nEvaluating Vanilla HNSW across efSearch sweep..." << std::endl;
    for (size_t ef : ef_list) {
        alg_hnsw->setEf(ef);

        // Warmup for this ef
        for (int w = 0; w < std::min(5, num_query); ++w) {
            alg_hnsw->searchKnn(queryVectors[w], w, K);
        }

        std::vector<double> pass_times_ms(NUM_PASSES, 0.0);
        std::vector<double> pass_recalls(NUM_PASSES, 0.0);
        std::vector<double> query_latencies(eval_queries, 0.0);
        std::vector<float> query_recalls(eval_queries, 0.0);

        for (int p = 0; p < NUM_PASSES; ++p) {
            double total_time_ms = 0.0;
            double total_recall = 0.0;

            for (int i = 0; i < eval_queries; i++) {
                t.restart();
                auto result = alg_hnsw->searchKnn(queryVectors[i], i, K);
                double q_time = t.elapsed() * 1000.0;
                float rec = calculateRecall(i, K, result, groundtruth);

                query_latencies[i] += q_time / NUM_PASSES;
                if (p == 0) query_recalls[i] = rec;

                total_time_ms += q_time;
                total_recall += rec;
            }
            pass_times_ms[p] = total_time_ms / eval_queries;
            pass_recalls[p] = total_recall / eval_queries;
        }

        if (ef == 20) {
            std::string q_path = "results/data/" + dataset + "_HNSW_K" + std::to_string(K) + "_queries.txt";
            std::ofstream q_file(q_path);
            q_file << "# QueryID\tTime_sec\tTime_us\tRecall@20\n";
            for (int i = 0; i < eval_queries; i++) {
                double time_sec = query_latencies[i] / 1000.0;
                double time_us = query_latencies[i] * 1000.0;
                q_file << i << "\t" << std::fixed << std::setprecision(6) << time_sec
                       << "\t" << std::setprecision(2) << time_us
                       << "\t" << std::setprecision(4) << query_recalls[i] << "\n";
            }
            q_file.close();
            std::cout << "  Logged per-query results to: " << q_path << std::endl;
        }

        double sum_lat_ms = 0.0;
        double sum_recall = 0.0;
        for (int p = 0; p < NUM_PASSES; ++p) {
            sum_lat_ms += pass_times_ms[p];
            sum_recall += pass_recalls[p];
        }

        double lat_ms = sum_lat_ms / NUM_PASSES;
        double lat_us = lat_ms * 1000.0;
        double rec = sum_recall / NUM_PASSES;
        double qps = 1000.0 / lat_ms;

        std::cout << "  [HNSW] ef = " << std::setw(3) << ef 
                  << " | Time: " << std::fixed << std::setprecision(4) << lat_ms << " ms (" 
                  << std::fixed << std::setprecision(1) << lat_us << " us)"
                  << " | Recall: " << std::setprecision(4) << rec 
                  << " | QPS: " << std::setprecision(1) << qps << std::endl;

        if (csv.is_open()) {
            csv << "Vanilla_HNSW," << ef << "," << std::fixed << std::setprecision(5) << lat_ms
                << "," << std::setprecision(2) << lat_us
                << "," << std::setprecision(4) << rec
                << "," << std::setprecision(1) << qps
                << "," << std::setprecision(2) << hnsw_build_time
                << "," << std::setprecision(2) << hnsw_memory_mb << "\n";
            csv.flush();
        }
    }

    delete alg_hnsw;

    // ================================================================
    // PART 2: SHG (HEDS) - Baseline & Step 2
    // ================================================================
    std::cout << "\n------------------------------------------------------" << std::endl;
    std::cout << "2. Building SHG (HEDS) (M=48, efConstruction=80)..." << std::endl;
    std::cout << "------------------------------------------------------" << std::endl;

    hnswlib::HEDS<float>* alg_shg = new hnswlib::HEDS<float>(&space, dim, num_base + num_query, M, ef_construction);
    t.restart();
    for (int i = 0; i < num_base; ++i) {
        alg_shg->addDataPoint(data[i], i, 1, per);
    }
    double shg_build_time = t.elapsed();

    t.restart();
    alg_shg->buildShortcuts(num_base);
    double shg_shortcut_time = t.elapsed();

    double shg_memory_mb = alg_shg->indexFileSize(num_base) / (1024.0 * 1024.0);
    double shg_shortcut_kb = alg_shg->Shortcuts.size_in_bytes() / 1024.0;

    std::cout << "SHG Graph Build Time: " << shg_build_time << " s" << std::endl;
    std::cout << "SHG Shortcut Build Time: " << shg_shortcut_time << " s (Total: " << (shg_build_time + shg_shortcut_time) << " s)" << std::endl;
    std::cout << "SHG Index Size: " << shg_memory_mb << " MB (+ " << shg_shortcut_kb << " KB shortcuts)" << std::endl;

    // Register query representations
    for (int i = 0; i < num_query; i++) {
        alg_shg->addDataPoint(queryVectors[i], num_base + i, -1, per);
    }
    alg_shg->resultsProcessing.assign(num_base * (alg_shg->maxlevel_ + 1), -1);

    // Warmup SHG
    alg_shg->setEf(30);
    for (int w = 0; w < std::min(20, num_query); ++w) {
        Query query(num_base + w, K);
        alg_shg->searchKnnShortcuts(query);
        std::fill(alg_shg->resultsProcessing.begin(), alg_shg->resultsProcessing.end(), -1);
    }

    struct SHGConfig {
        std::string name;
        hnswlib::PrefetchMode mode;
    };
    std::vector<SHGConfig> shg_configs = {
        {"Baseline_SHG", hnswlib::PrefetchMode::BASELINE},
        {"Step1_SkipPrefetch", hnswlib::PrefetchMode::STEP1_SKIP},
        {"Step2_SkipLookahead", hnswlib::PrefetchMode::STEP2_SKIP_LOOKAHEAD}
    };

    for (const auto& cfg : shg_configs) {
        hnswlib::PrefetchController::set_mode(cfg.mode);
        std::cout << "\nEvaluating " << cfg.name << " across efSearch sweep..." << std::endl;

        for (size_t ef : ef_list) {
            alg_shg->setEf(ef);

            // Warmup for this ef
            for (int w = 0; w < std::min(5, num_query); ++w) {
                Query query(num_base + w, K);
                alg_shg->searchKnnShortcuts(query);
                std::fill(alg_shg->resultsProcessing.begin(), alg_shg->resultsProcessing.end(), -1);
            }

            std::vector<double> pass_times_ms(NUM_PASSES, 0.0);
            std::vector<double> pass_recalls(NUM_PASSES, 0.0);
            std::vector<double> query_latencies(eval_queries, 0.0);
            std::vector<float> query_recalls(eval_queries, 0.0);

            for (int p = 0; p < NUM_PASSES; ++p) {
                double total_time_ms = 0.0;
                double total_recall = 0.0;

                for (int i = 0; i < eval_queries; i++) {
                    Query query(num_base + i, K);
                    t.restart();
                    auto result = alg_shg->searchKnnShortcuts(query);
                    double q_time = t.elapsed() * 1000.0;
                    float rec = calculateRecall(i, K, result, groundtruth);

                    query_latencies[i] += q_time / NUM_PASSES;
                    if (p == 0) query_recalls[i] = rec;

                    total_time_ms += q_time;
                    total_recall += rec;

                    std::fill(alg_shg->resultsProcessing.begin(), alg_shg->resultsProcessing.end(), -1);
                }
                pass_times_ms[p] = total_time_ms / eval_queries;
                pass_recalls[p] = total_recall / eval_queries;
            }

            if (ef == 20) {
                std::string q_path = "results/data/" + dataset + "_" + cfg.name + "_K" + std::to_string(K) + "_queries.txt";
                std::ofstream q_file(q_path);
                q_file << "# QueryID\tTime_sec\tTime_us\tRecall@20\n";
                for (int i = 0; i < eval_queries; i++) {
                    double time_sec = query_latencies[i] / 1000.0;
                    double time_us = query_latencies[i] * 1000.0;
                    q_file << i << "\t" << std::fixed << std::setprecision(6) << time_sec
                           << "\t" << std::setprecision(2) << time_us
                           << "\t" << std::setprecision(4) << query_recalls[i] << "\n";
                }
                q_file.close();
                std::cout << "  Logged per-query results to: " << q_path << std::endl;
            }

            double sum_lat_ms = 0.0;
            double sum_recall = 0.0;
            for (int p = 0; p < NUM_PASSES; ++p) {
                sum_lat_ms += pass_times_ms[p];
                sum_recall += pass_recalls[p];
            }

            double lat_ms = sum_lat_ms / NUM_PASSES;
            double lat_us = lat_ms * 1000.0;
            double rec = sum_recall / NUM_PASSES;
            double qps = 1000.0 / lat_ms;

            std::cout << "  [" << cfg.name << "] ef = " << std::setw(3) << ef 
                      << " | Time: " << std::fixed << std::setprecision(4) << lat_ms << " ms (" 
                      << std::fixed << std::setprecision(1) << lat_us << " us)"
                      << " | Recall: " << std::setprecision(4) << rec 
                      << " | QPS: " << std::setprecision(1) << qps << std::endl;

            if (csv.is_open()) {
                csv << cfg.name << "," << ef << "," << std::fixed << std::setprecision(5) << lat_ms
                    << "," << std::setprecision(2) << lat_us
                    << "," << std::setprecision(4) << rec
                    << "," << std::setprecision(1) << qps
                    << "," << std::setprecision(2) << (shg_build_time + shg_shortcut_time)
                    << "," << std::setprecision(2) << shg_memory_mb << "\n";
                csv.flush();
            }
        }
    }

    csv.close();
    std::cout << "\nComparison results saved to: " << csv_path << std::endl;

    // Summary file
    std::string summary_path = "results/data/hnsw_vs_shg_summary.csv";
    std::ofstream sum_csv(summary_path);
    sum_csv << "Method,Build_Time_s,Shortcut_Time_s,Memory_MB,Latency_ef20_us,Recall_ef20,QPS_ef20\n";
    // Read ef=20 rows
    sum_csv << "Vanilla_HNSW," << hnsw_build_time << ",0.0," << hnsw_memory_mb << ",,,\n";
    sum_csv << "SHG_Baseline," << shg_build_time << "," << shg_shortcut_time << "," << shg_memory_mb << ",,,\n";
    sum_csv.close();

    // Cleanup
    delete alg_shg;
    clear_2d_array(queryVectors, num_query);
    clear_2d_array(groundtruth, num_query);
    clear_2d_array(data, num_base);

    return 0;
}
