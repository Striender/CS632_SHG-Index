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

    std::string algo = (argc > 1) ? argv[1] : "heds";
    std::string dataset = (argc > 2) ? argv[2] : "enron";
    int K = (argc > 3) ? std::stoi(argv[3]) : 20;

    std::cout << "======================================================" << std::endl;
    std::cout << "Starting Noise-Isolated Search Performance Evaluation" << std::endl;
    std::cout << "Dataset: " << dataset << ", Algo: " << algo << ", K: " << K << std::endl;
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

    // Build Index
    hnswlib::L2Space space(dim);
    hnswlib::HEDS<float>* alg_hnsw = new hnswlib::HEDS<float>(&space, dim, num_base + num_query, M, ef_construction);
    Performance per;
    Timer t;

    float** data = new float* [num_base];
    for (int i = 0; i < num_base; ++i) {
        data[i] = new float[dim];
        loadin.read((char*)data[i], sizeof(float) * header[2]);
    }

    std::cout << "Building SHG Index..." << std::endl;
    t.restart();
    for (int i = 0; i < num_base; ++i) {
        alg_hnsw->addDataPoint(data[i], i, 1, per);
    }
    double build_time = t.elapsed();
    std::cout << "Index build time: " << build_time << " s" << std::endl;

    t.restart();
    alg_hnsw->buildShortcuts(num_base);
    double shortcut_time = t.elapsed();
    std::cout << "Shortcut build time: " << shortcut_time << " s" << std::endl;

    // Register query vectors
    std::cout << "Registering " << num_query << " query vectors..." << std::endl;
    for (int i = 0; i < num_query; i++) {
        alg_hnsw->addDataPoint(queryVectors[i], num_base + i, -1, per);
    }

    alg_hnsw->resultsProcessing.assign(num_base * (alg_hnsw->maxlevel_ + 1), -1);

    // Warm up CPU governor and caches with 30 queries
    std::cout << "Performing CPU frequency ramp-up warmup (30 queries)..." << std::endl;
    alg_hnsw->setEf(30);
    for (int w = 0; w < 30; ++w) {
        Query query(num_base + (w % num_query), K);
        alg_hnsw->searchKnnShortcuts(query);
        std::fill(alg_hnsw->resultsProcessing.begin(), alg_hnsw->resultsProcessing.end(), -1);
    }

    // Prepare CSV output file for ef sweep
    std::string csv_path = "results/data/ef_sweep_curves_" + dataset + ".csv";
    std::ofstream csv(csv_path);
    csv << "method,ef,latency_ms,latency_us,recall,qps" << std::endl;

    std::vector<size_t> ef_list = {20, 25, 30, 40, 50, 60, 80, 100, 120, 150, 200, 300};

    int eval_queries = std::min(num_query, (dataset == "sift" ? 2000 : num_query));
    std::cout << "Evaluating " << eval_queries << " queries per setting..." << std::endl;

    struct TestConfig {
        std::string name;
        hnswlib::PrefetchMode mode;
        std::string per_query_filename;
    };

    std::vector<TestConfig> configs = {
        {"Baseline_SHG", hnswlib::PrefetchMode::BASELINE, "results/data/" + dataset + "_heds_K" + std::to_string(K) + "_baseline_queries.txt"},
        {"Step1_SkipPrefetch", hnswlib::PrefetchMode::STEP1_SKIP, "results/data/" + dataset + "_heds_K" + std::to_string(K) + "_step1_skip_prefetch_queries.txt"},
        {"Step2_SkipLookahead", hnswlib::PrefetchMode::STEP2_SKIP_LOOKAHEAD, "results/data/" + dataset + "_heds_K" + std::to_string(K) + "_step2_skip_lookahead_prefetch_queries.txt"}
    };

    std::cout << "\n======================================================" << std::endl;
    std::cout << "Running Noise-Isolated Sweep across methods (3 passes/setting)..." << std::endl;
    std::cout << "======================================================" << std::endl;

    const int NUM_PASSES = 3;

    for (const auto& cfg : configs) {
        hnswlib::PrefetchController::set_mode(cfg.mode);
        std::cout << "\n--- Testing Configuration: " << cfg.name << " ---" << std::endl;

        for (size_t ef : ef_list) {
            alg_hnsw->setEf(ef);

            // Warmup for this specific ef
            for (int w = 0; w < 5; ++w) {
                Query query(num_base + (w % num_query), K);
                alg_hnsw->searchKnnShortcuts(query);
                std::fill(alg_hnsw->resultsProcessing.begin(), alg_hnsw->resultsProcessing.end(), -1);
            }

            // Run NUM_PASSES and take best (minimum noise) pass
            std::vector<double> pass_times_ms(NUM_PASSES, 0.0);
            std::vector<double> pass_recalls(NUM_PASSES, 0.0);
            std::vector<std::vector<double>> pass_per_query_time(NUM_PASSES, std::vector<double>(eval_queries));
            std::vector<std::vector<float>> pass_per_query_recall(NUM_PASSES, std::vector<float>(eval_queries));

            for (int p = 0; p < NUM_PASSES; ++p) {
                double total_time_ms = 0.0;
                double total_recall = 0.0;

                for (int i = 0; i < eval_queries; i++) {
                    Query query(num_base + i, K);
                    t.restart();
                    auto result = alg_hnsw->searchKnnShortcuts(query);
                    double q_time = t.elapsed() * 1000.0; // ms
                    float rec = calculateRecall(i, K, result, groundtruth);

                    total_time_ms += q_time;
                    total_recall += rec;

                    pass_per_query_time[p][i] = q_time;
                    pass_per_query_recall[p][i] = rec;

                    std::fill(alg_hnsw->resultsProcessing.begin(), alg_hnsw->resultsProcessing.end(), -1);
                }

                pass_times_ms[p] = total_time_ms / eval_queries;
                pass_recalls[p] = total_recall / eval_queries;
            }

            // Select pass with minimum latency (standard practice to eliminate context switches)
            int best_p = 0;
            for (int p = 1; p < NUM_PASSES; ++p) {
                if (pass_times_ms[p] < pass_times_ms[best_p]) {
                    best_p = p;
                }
            }

            double avg_latency_ms = pass_times_ms[best_p];
            double avg_latency_us = avg_latency_ms * 1000.0;
            double avg_recall = pass_recalls[best_p];
            double qps = 1000.0 / avg_latency_ms;

            std::cout << "  ef = " << std::setw(3) << ef 
                      << " | Time: " << std::fixed << std::setprecision(4) << avg_latency_ms << " ms (" 
                      << std::fixed << std::setprecision(1) << avg_latency_us << " us)"
                      << " | Recall: " << std::setprecision(4) << avg_recall
                      << " | QPS: " << std::setprecision(1) << qps << std::endl;

            csv << cfg.name << "," << ef << "," << std::fixed << std::setprecision(5) << avg_latency_ms
                << "," << std::setprecision(2) << avg_latency_us
                << "," << std::setprecision(4) << avg_recall
                << "," << std::setprecision(1) << qps << "\n";
            csv.flush();

            // If ef == 20 (default benchmark setting), update the per-query log file
            if (ef == 20) {
                std::ofstream qlog(cfg.per_query_filename);
                qlog << "# QueryID\tTime_sec\tTime_us\tRecall@20\n";
                for (int i = 0; i < eval_queries; ++i) {
                    double t_ms = pass_per_query_time[best_p][i];
                    float r = pass_per_query_recall[best_p][i];
                    qlog << i << "\t" << std::fixed << std::setprecision(6) << (t_ms / 1000.0)
                         << "\t" << std::setprecision(2) << (t_ms * 1000.0)
                         << "\t" << std::setprecision(4) << r << "\n";
                }
                qlog.close();
            }
        }
    }
    csv.close();
    std::cout << "\nSweep results successfully saved to: " << csv_path << std::endl;

    // Cleanup
    clear_2d_array(queryVectors, num_query);
    clear_2d_array(groundtruth, num_query);
    clear_2d_array(data, num_base);
    delete alg_hnsw;

    return 0;
}
