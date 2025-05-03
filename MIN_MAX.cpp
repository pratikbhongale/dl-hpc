#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;

class ParallelMinMax {
private:
    vector<int> arr;
    int size;
    int min_seq, max_seq, min_par, max_par;
    long long sum_seq, sum_par;
    double avg_seq, avg_par;
    double time_min_seq, time_max_seq, time_sum_seq, time_avg_seq;
    double time_min_par, time_max_par, time_sum_par, time_avg_par;

public:
    ParallelMinMax(int size) : size(size), min_seq(0), max_seq(0), min_par(0), max_par(0), sum_seq(0), sum_par(0), avg_seq(0), avg_par(0),
        time_min_seq(0), time_max_seq(0), time_sum_seq(0), time_avg_seq(0), time_min_par(0), time_max_par(0), time_sum_par(0), time_avg_par(0) {
        arr.resize(size);
    }

    void generateRandomArray() {
        srand(time(0));
        for (int i = 0; i < size; i++) {
            arr[i] = rand() % 100 + 1;
        }
    }

    void computeMinSequential() {
        auto start = chrono::high_resolution_clock::now();
        min_seq = arr[0];
        for (int i = 1; i < size; i++) {
            if (arr[i] < min_seq) min_seq = arr[i];
        }
        auto end = chrono::high_resolution_clock::now();
        time_min_seq = chrono::duration<double>(end - start).count();
    }

    void computeMaxSequential() {
        auto start = chrono::high_resolution_clock::now();
        max_seq = arr[0];
        for (int i = 1; i < size; i++) {
            if (arr[i] > max_seq) max_seq = arr[i];
        }
        auto end = chrono::high_resolution_clock::now();
        time_max_seq = chrono::duration<double>(end - start).count();
    }

    void computeSumSequential() {
        auto start = chrono::high_resolution_clock::now();
        sum_seq = 0;
        for (int i = 0; i < size; i++) {
            sum_seq += arr[i];
        }
        auto end = chrono::high_resolution_clock::now();
        time_sum_seq = chrono::duration<double>(end - start).count();
    }

    void computeAvgSequential() {
        auto start = chrono::high_resolution_clock::now();
        avg_seq = static_cast<double>(sum_seq) / size;
        auto end = chrono::high_resolution_clock::now();
        time_avg_seq = chrono::duration<double>(end - start).count();
    }

    void computeMinParallel() {
        auto start = chrono::high_resolution_clock::now();
        min_par = arr[0];
        #pragma omp parallel for reduction(min:min_par)
        for (int i = 1; i < size; i++) {
            if (arr[i] < min_par) min_par = arr[i];
        }
        auto end = chrono::high_resolution_clock::now();
        time_min_par = chrono::duration<double>(end - start).count();
    }

    void computeMaxParallel() {
        auto start = chrono::high_resolution_clock::now();
        max_par = arr[0];
        #pragma omp parallel for reduction(max:max_par)
        for (int i = 1; i < size; i++) {
            if (arr[i] > max_par) max_par = arr[i];
        }
        auto end = chrono::high_resolution_clock::now();
        time_max_par = chrono::duration<double>(end - start).count();
    }

    void computeSumParallel() {
        auto start = chrono::high_resolution_clock::now();
        sum_par = 0;
        #pragma omp parallel for reduction(+:sum_par)
        for (int i = 0; i < size; i++) {
            sum_par += arr[i];
        }
        auto end = chrono::high_resolution_clock::now();
        time_sum_par = chrono::duration<double>(end - start).count();
    }

    void computeAvgParallel() {
        auto start = chrono::high_resolution_clock::now();
        avg_par = static_cast<double>(sum_par) / size;
        auto end = chrono::high_resolution_clock::now();
        time_avg_par = chrono::duration<double>(end - start).count();
    }

    void sequentialComputation() {
        computeMinSequential();
        computeMaxSequential();
        computeSumSequential();
        computeAvgSequential();
    }

    void parallelComputation() {
        computeMinParallel();
        computeMaxParallel();
        computeSumParallel();
        computeAvgParallel();
    }

    void displayResults() {
        cout << "---- Sequential Computation ----\n";
        cout << "Min: " << min_seq << " | Time: " << time_min_seq << " sec\n";
        cout << "Max: " << max_seq << " | Time: " << time_max_seq << " sec\n";
        cout << "Sum: " << sum_seq << " | Time: " << time_sum_seq << " sec\n";
        cout << "Average: " << avg_seq << " | Time: " << time_avg_seq << " sec\n\n";

        cout << "---- Parallel Computation ----\n";
        cout << "Min: " << min_par << " | Time: " << time_min_par << " sec\n";
        cout << "Max: " << max_par << " | Time: " << time_max_par << " sec\n";
        cout << "Sum: " << sum_par << " | Time: " << time_sum_par << " sec\n";
        cout << "Average: " << avg_par << " | Time: " << time_avg_par << " sec\n\n";

        cout << "---- Speedup Factors ----\n";
        cout << "Speedup (Min): " << (time_min_seq / time_min_par) << "x\n";
        cout << "Speedup (Max): " << (time_max_seq / time_max_par) << "x\n";
        cout << "Speedup (Sum): " << (time_sum_seq / time_sum_par) << "x\n";
        cout << "Speedup (Average): " << (time_avg_seq / time_avg_par) << "x\n";
    }
};

int main() {
    int size;
    cout << "Enter array size: ";
    cin >> size;
    if (size <= 0) {
        cout << "Invalid size!" << endl;
        return 1;
    }

    ParallelMinMax pm(size);
    pm.generateRandomArray();
    pm.sequentialComputation();
    pm.parallelComputation();
    pm.displayResults();

    return 0;
}