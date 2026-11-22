class Timer
{
public:
    Timer() : time_begin(std::chrono::steady_clock::now()) {};
    ~Timer() {};
    /**
        * Restart the timer.
        */
    void restart()
    {
        time_begin = std::chrono::steady_clock::now();
    }
    /**
        * Measures elapsed time.
        *
        * @return The elapsed time
        */
    double elapsed()
    {
        std::chrono::steady_clock::time_point time_end = std::chrono::steady_clock::now();
        return (std::chrono::duration_cast<std::chrono::microseconds>(time_end - time_begin).count())*1e-6;// / CLOCKS_PER_SEC;
    }
private:
    std::chrono::steady_clock::time_point time_begin;
};


class Query
{

public:

    int query_Id;

    int k;

    float queryTime = 0;

    float recall = 0;


    std::vector<std::pair<float, int>> nearestNeighbor; // The result of query : curObj, dis

    std::vector<std::tuple<int, int, float>> queryRes; // The processing of query : curObj, level, dis


public:
    Query() {};

    Query(int query_Id_, int k_){
        query_Id = query_Id_;
        k = k_;
    }

    void setQueryTime(float t){
        queryTime = t;
    }
    float getQueryTime(){ return queryTime;}

    int getID() {return query_Id;}

    void setRecall(float r){
        recall = r;
    }
    float getRecall() {return recall;}

    //void search();

    ~Query() {};
};


class Performance
{
public:

    int num = 0;

    int levelofHNSW = 0;

    float timeQuery = 0;

    float timeBuildindex = 0;

    float timePreprocessing = 0;

    float timeShortcut = 0;

    float recall = 0;

    float indexSize = 0;


public:
    Performance() {};
    //update the query results
    void update(Query &q){
        num ++;
        timeQuery += q.queryTime;
    }

    int getNum(){ return num;}

    void setLevel(int n) {levelofHNSW = n;}
    int getLevel() {return levelofHNSW;}

    float getTimeQuery(){ return timeQuery;}

    float getTimePreprocessing() {return timePreprocessing;}

    float getTimeBuildindex() {return timeBuildindex;}
    void setTimeBuildindex(float t) {timeBuildindex = t;}

    float getTimeShortcut() {return timeShortcut;}
    void setTimeShortcut(float t) {timeShortcut = t;}


    ~Performance() {};
};


float calculateRecall(int &queryID, int &K, std::priority_queue<std::pair<float, hnswlib::labeltype>> &result, int** &groundtruth){

    int cnt = 0;
    std::vector<int> queryResults;
    std::vector<int> queryGroundtruth(groundtruth[queryID],groundtruth[queryID]+K);

    while(result.size() > 0){
        std::pair<float, int> rez = result.top();
        queryResults.push_back(rez.second);
        result.pop();        
    }

    for(int id: queryResults){
        auto it = std::find(queryGroundtruth.begin(), queryGroundtruth.end(), id);
        if (it != queryGroundtruth.end()){ cnt++ ;} 
    }
    return float(cnt)/float(K);
}

inline std::string resolve_data_file(const std::string &dataset, const std::string &filename) {
    std::vector<std::string> candidates = {
        dataset + "/" + filename,
        "../" + dataset + "/" + filename,
        "/home/striender/Course/ADBMS/SHG-Index/" + dataset + "/" + filename,
        "/home/data/zgongae/VectorsIndex/HEDS/" + dataset + "/" + filename
    };
    const char* env_path = std::getenv("DATA_PATH");
    if (env_path) {
        candidates.insert(candidates.begin(), std::string(env_path) + "/" + dataset + "/" + filename);
        candidates.insert(candidates.begin(), std::string(env_path) + "/" + filename);
    }
    for (const auto& p : candidates) {
        std::ifstream check(p.c_str(), std::ios::binary);
        if (check.is_open()) return p;
    }
    return candidates[0];
}

inline std::string resolve_results_dir() {
    std::vector<std::string> candidates = {
        "results",
        "../results",
        "/home/striender/Course/ADBMS/SHG-Index/results"
    };
    for (const auto& dir : candidates) {
        std::ifstream check(dir.c_str());
        if (check.good()) return dir;
    }
    int r = system("mkdir -p results");
    (void)r;
    return "results";
}

void load_query_groundtruth(std::string &dataset, int &num_query, float** &queryVectors, int** &groundtruth){

    std::string query_input = resolve_data_file(dataset, dataset + "_query.fvecs");
    std::string groundtruth_input = resolve_data_file(dataset, dataset + "_groundtruth.ivecs");

    std::ifstream file_query(query_input, std::ios::binary);
    std::ifstream file_truth(groundtruth_input, std::ios::binary);
    if (!file_query.is_open() || !file_truth.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }

    uint32_t d, top;
    file_query.read(reinterpret_cast<char*>(&d), sizeof(d));
    file_truth.read(reinterpret_cast<char*>(&top), sizeof(top));

    size_t vecsizeof = sizeof(uint32_t) + d * sizeof(float); // 4 bytes for dimension + d floats

    file_query.seekg(0, std::ios::end);
    size_t fileSize = file_query.tellg();
    file_query.seekg(0, std::ios::beg);
    file_truth.seekg(0, std::ios::beg);
    num_query = fileSize / vecsizeof;

    std::cout << num_query << " " << d << " " << top <<std::endl;

    queryVectors = new float*[num_query];
    groundtruth = new int*[num_query];

    for (uint32_t i = 0; i < num_query; ++i) {

        uint32_t d_, top_;
        file_query.read(reinterpret_cast<char*>(&d_), sizeof(d_)); 
        file_truth.read(reinterpret_cast<char*>(&top_), sizeof(top_)); 
        queryVectors[i] = new float[d];
        file_query.read((char*)queryVectors[i], sizeof(float) * d);
        groundtruth[i] = new int[top];
        file_truth.read((char*)groundtruth[i], sizeof(int) * top);      
    }
    std::cout << "Finish reading ..." << std::endl;
}

void load_query_groundtruth_(std::string &dataset, int &num_query, float** &queryVectors, int** &groundtruth){
    
    std::string query_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + dataset +"_query.fbin";
    std::string groundtruth_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + dataset +"_groundtruth";

    std::ifstream file_query(query_input, std::ios::binary);
    std::ifstream file_truth(groundtruth_input, std::ios::binary);
    if (!file_query.is_open() || !file_truth.is_open()) {
        std::cerr << "Failed to open file." << std::endl;
        return;
    }
    file_query.read(reinterpret_cast<char*>(&num_query), sizeof(num_query));
    file_truth.read(reinterpret_cast<char*>(&num_query), sizeof(num_query));

    uint32_t top, d;
    file_truth.read(reinterpret_cast<char*>(&top), sizeof(top));
    file_query.read(reinterpret_cast<char*>(&d), sizeof(d));

    queryVectors = new float*[num_query];
    groundtruth = new int*[num_query];
    for (uint32_t i = 0; i < num_query; ++i) {

        queryVectors[i] = new float[d];
        file_query.read((char*)queryVectors[i], sizeof(float) * d);
        groundtruth[i] = new int[top];
        file_truth.read((char*)groundtruth[i], sizeof(int) * top);      
    }
    std::cout << "Finish reading ..." << std::endl;    


}


void load_query_groundtruth100m(std::string &dataset, int &num_query, float** &queryVectors, int** &groundtruth){
    std::string query_input, groundtruth_input;
    if(dataset == "deep100m"){
        query_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + "query.public.10K.fbin";
        groundtruth_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + "deep-100M";
    }
    else{
        query_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + "query.public.10K.u8bin";
        groundtruth_input = "/home/data/zgongae/VectorsIndex/datasets/" + dataset + "/" + "bigann-100M";
    }

    int d;
    int top;

    std::ifstream file_query(query_input, std::ios::binary);
    std::ifstream file_truth(groundtruth_input, std::ios::binary); 

    file_query.read(reinterpret_cast<char*>(&num_query), sizeof(num_query));
    file_truth.read(reinterpret_cast<char*>(&num_query), sizeof(num_query));

    file_truth.read(reinterpret_cast<char*>(&top), sizeof(top));
    file_query.read(reinterpret_cast<char*>(&d), sizeof(d));

    //std::cout << num_query << " " << top << " " << d << std::endl;

    queryVectors = new float*[num_query];
    groundtruth = new int*[num_query];
    for (uint32_t i = 0; i < num_query; ++i) {

        queryVectors[i] = new float[d];
        if(dataset == "sift10m"){
            for(uint32_t j = 0; j < d; ++j){
                unsigned char value;
                file_query.read(reinterpret_cast<char*>(&value), 1);
                float number = static_cast<float>(value);
                queryVectors[i][j] = number;
            }
        }else{
            file_query.read((char*)queryVectors[i], sizeof(float) * d);
        }

        file_query.read((char*)queryVectors[i], sizeof(float) * d);
        groundtruth[i] = new int[top];
        file_truth.read((char*)groundtruth[i], sizeof(int) * top);

    }
}

template <class T>
void clear_2d_array(T** array, int n)
{
	for (int i = 0; i < n; ++i) {
		delete[] array[i];
	}
	delete[] array;
}