// PlantSetDelta C++ k-mer counter. Output is sparse raw counts.
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
#include <cstdint>

using namespace std;
using Kmer = uint64_t;

inline int char_to_int(char c) {
    switch (toupper(static_cast<unsigned char>(c))) {
        case 'A': return 0;
        case 'C': return 1;
        case 'G': return 2;
        case 'T': return 3;
        default: return -1;
    }
}

string kmerToString(Kmer value, int k) {
    static const char bases[] = {'A', 'C', 'G', 'T'};
    string result(k, 'A');
    for (int i = k - 1; i >= 0; --i) {
        result[i] = bases[value & 3];
        value >>= 2;
    }
    return result;
}

mutex queueMutex;
mutex outputMutex;
ofstream outFile;

void processSequence(const string& seq, const string& label, int k, int minCount) {
    unordered_map<Kmer, int> counts;
    Kmer value = 0;
    const Kmer mask = (1ULL << (2 * k)) - 1;
    int validBases = 0;

    for (char c : seq) {
        const int base = char_to_int(c);
        if (base < 0) {
            value = 0;
            validBases = 0;
            continue;
        }
        value = ((value << 2) | static_cast<Kmer>(base)) & mask;
        ++validBases;
        if (validBases >= k) ++counts[value];
    }

    lock_guard<mutex> lock(outputMutex);
    for (const auto& item : counts) {
        if (item.second >= minCount) {
            outFile << label << "\t" << k << "\t"
                    << kmerToString(item.first, k) << "\t" << item.second << "\n";
        }
    }
}

int main(int argc, char* argv[]) {
    string filename, outputPrefix;
    int k = 7, minCount = 1, numThreads = 4;

    for (int i = 1; i < argc; ++i) {
        string arg = argv[i];
        if ((arg == "-i" || arg == "-o" || arg == "-k" || arg == "-m" || arg == "-t") && i + 1 >= argc) {
            cerr << "Missing value for " << arg << endl;
            return 1;
        }
        if (arg == "-i") filename = argv[++i];
        else if (arg == "-o") outputPrefix = argv[++i];
        else if (arg == "-k") k = stoi(argv[++i]);
        else if (arg == "-m") minCount = stoi(argv[++i]);
        else if (arg == "-t") numThreads = stoi(argv[++i]);
        else {
            cerr << "Unknown option: " << arg << endl;
            return 1;
        }
    }

    if (filename.empty() || outputPrefix.empty()) {
        cerr << "Both -i and -o are required." << endl;
        return 1;
    }
    // The Python package currently supports k=4..7. Keep the standalone
    // executable's accepted range identical and avoid undefined 64-bit shifts.
    if (k < 4 || k > 7) {
        cerr << "k must be between 4 and 7." << endl;
        return 1;
    }
    if (minCount < 1) {
        cerr << "Minimum count must be >= 1." << endl;
        return 1;
    }
    if (numThreads < 1) numThreads = 1;

    ifstream inFile(filename);
    if (!inFile) {
        cerr << "Cannot open input FASTA: " << filename << endl;
        return 1;
    }

    const string outputName = outputPrefix + "_kmer_counts.txt";
    outFile.open(outputName);
    if (!outFile) {
        cerr << "Cannot open output file: " << outputName << endl;
        return 1;
    }
    outFile << "Sequence\tK\tKMer\tCount\n";

    vector<pair<string, string>> sequences;
    string line, sequence, label;
    bool haveRecord = false;
    while (getline(inFile, line)) {
        if (!line.empty() && line.back() == '\r') line.pop_back();
        if (line.empty()) continue;
        if (line[0] == '>') {
            if (haveRecord) sequences.emplace_back(label, sequence);
            string header = line.substr(1);
            const size_t firstSpace = header.find_first_of(" \t");
            label = header.substr(0, firstSpace); // Match Biopython record.id.
            sequence.clear();
            haveRecord = true;
        } else if (haveRecord) {
            for (char c : line) {
                if (!isspace(static_cast<unsigned char>(c))) {
                    sequence.push_back(static_cast<char>(toupper(static_cast<unsigned char>(c))));
                }
            }
        }
    }
    if (haveRecord) sequences.emplace_back(label, sequence);

    queue<pair<string, string>> work;
    for (const auto& record : sequences) work.push(record);

    auto worker = [&]() {
        while (true) {
            pair<string, string> record;
            {
                lock_guard<mutex> lock(queueMutex);
                if (work.empty()) return;
                record = work.front();
                work.pop();
            }
            processSequence(record.second, record.first, k, minCount);
        }
    };

    vector<thread> workers;
    for (int i = 0; i < numThreads; ++i) workers.emplace_back(worker);
    for (auto& thread : workers) thread.join();
    outFile.close();
    return outFile ? 0 : 1;
}
