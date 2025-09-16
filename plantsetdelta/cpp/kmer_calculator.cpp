// kmer_calculator.cpp

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <unordered_map>
#include <chrono>
#include <cctype>
#include <thread>
#include <mutex>
#include <queue>

using namespace std;

// 定义Kmer类型为64位无符号整数
using Kmer = uint64_t;

// 将碱基字符转换为整数表示（A:0, C:1, G:2, T:3）
inline int char_to_int(char c) {
    switch(toupper(c)) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default: return -1;
    }
}

// 将整数表示的k-mer转换回字符串
string kmerToString(Kmer kmer, int k) {
    static const char bases[] = {'A', 'C', 'G', 'T'};
    string result(k, 'A');
    for (int i = k - 1; i >= 0; --i) {
        result[i] = bases[kmer & 3];
        kmer >>= 2;
    }
    return result;
}

// 定义互斥量用于线程同步
mutex mtx;

// 输出文件的全局ofstream对象
ofstream outFile;

// 处理单个序列，计算指定k值的k-mer计数
void processSequence(const string& seq, const string& label, const vector<int>& kValues, int minCount) {
    for (int k : kValues) {
        if (k > 32) {
            cerr << "Warning: k > 32 is not supported. Skipping k = " << k << endl;
            continue;
        }

        unordered_map<Kmer, int> kmerCounts;
        Kmer kmer = 0;
        Kmer mask = (1ULL << (2 * k)) - 1;

        int validBases = 0;
        for (size_t i = 0; i < seq.length(); ++i) {
            int base = char_to_int(seq[i]);
            if (base == -1) {
                kmer = 0;
                validBases = 0;
                continue;
            }
            kmer = ((kmer << 2) | base) & mask;
            validBases++;
            if (validBases >= k) {
                kmerCounts[kmer]++;
            }
        }

        // 锁定输出文件，确保线程安全
        lock_guard<mutex> lock(mtx);
        for (const auto& pair : kmerCounts) {
            Kmer kmerVal = pair.first;
            int count = pair.second;
            if (count >= minCount) {
                outFile << label << "\t" << k << "\t" << kmerToString(kmerVal, k) << "\t" << count << "\n";
            }
        }
    }
}

int main(int argc, char* argv[]) {
    string filename;
    string output_prefix;
    vector<int> kValues;
    int minCount = 1;
    int num_threads = 4;

    // 解析命令行参数
    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if (arg == "-i") {
            if (i + 1 < argc) {
                filename = argv[++i];
            } else {
                cerr << "Error: Missing argument for -i" << endl;
                return 1;
            }
        } else if (arg == "-o") {
            if (i + 1 < argc) {
                output_prefix = argv[++i];
            } else {
                cerr << "Error: Missing argument for -o" << endl;
                return 1;
            }
        } else if (arg == "-k") {
            while (i + 1 < argc && argv[i+1][0] != '-') {
                kValues.push_back(stoi(argv[++i]));
            }
            if (kValues.empty()) {
                cerr << "Error: Missing k-mer sizes after -k" << endl;
                return 1;
            }
        } else if (arg == "-m") {
            if (i + 1 < argc) {
                minCount = stoi(argv[++i]);
            } else {
                cerr << "Error: Missing argument for -m" << endl;
                return 1;
            }
        } else if (arg == "-t") {
            if (i + 1 < argc) {
                num_threads = stoi(argv[++i]);
            } else {
                cerr << "Error: Missing argument for -t" << endl;
                return 1;
            }
        } else {
            cerr << "Unknown option: " << arg << endl;
            return 1;
        }
    }

    // 检查必需参数
    if (filename.empty()) {
        cerr << "Error: Input file not specified (-i)" << endl;
        return 1;
    }
    if (output_prefix.empty()) {
        cerr << "Error: Output prefix not specified (-o)" << endl;
        return 1;
    }
    if (kValues.empty()) {
        cerr << "Error: K-mer sizes not specified (-k)" << endl;
        return 1;
    }

    // 打印解析后的参数
    cout << "Input file: " << filename << endl;
    cout << "Output prefix: " << output_prefix << endl;
    cout << "Min count: " << minCount << endl;
    cout << "Num threads: " << num_threads << endl;
    cout << "K-mer sizes:";
    for (int k : kValues) {
        cout << " " << k;
    }
    cout << endl;

    // 打开输入文件
    ifstream inFile(filename);
    if (!inFile.is_open()) {
        cerr << "Error: Cannot open input file " << filename << endl;
        return 1;
    }

    // 打开输出文件
    string output_filename = output_prefix + "_kmer_counts.txt";
    outFile.open(output_filename);
    if (!outFile.is_open()) {
        cerr << "Error: Cannot open output file " << output_filename << endl;
        return 1;
    }
    outFile << "Sequence\tK\tKMer\tCount\n";

    vector<pair<string, string>> sequences;

    string line, seq, label;
    auto start = chrono::high_resolution_clock::now();

    // 读取FASTA文件中的序列
    while (getline(inFile, line)) {
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (!seq.empty()) {
                sequences.emplace_back(label, seq);
                seq.clear();
            }
            label = line.substr(1);
        } else {
            seq += line;
        }
    }

    // 处理最后一个序列
    if (!seq.empty()) {
        sequences.emplace_back(label, seq);
    }

    // 检查线程数量
    if (num_threads <= 0) {
        num_threads = thread::hardware_concurrency();
        if (num_threads == 0) num_threads = 4;
    }

    // 定义队列用于存储待处理的序列
    queue<pair<string, string>> seqQueue;
    for (const auto& s : sequences) {
        seqQueue.push(s);
    }

    // 定义线程函数
    auto worker = [&]() {
        while (true) {
            pair<string, string> seqPair;
            {
                lock_guard<mutex> lock(mtx);
                if (seqQueue.empty()) {
                    break;
                }
                seqPair = seqQueue.front();
                seqQueue.pop();
            }
            processSequence(seqPair.second, seqPair.first, kValues, minCount);
        }
    };

    // 创建线程池
    vector<thread> threads;
    for (int i = 0; i < num_threads; ++i) {
        threads.emplace_back(worker);
    }

    // 等待所有线程完成
    for (auto& t : threads) {
        t.join();
    }

    outFile.close();

    auto end = chrono::high_resolution_clock::now();
    chrono::duration<double> diff = end - start;
    cerr << "Time taken: " << diff.count() << " s" << endl;

    return 0;
}
