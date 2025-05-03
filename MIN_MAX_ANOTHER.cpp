#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

using namespace std;

int main() {
    int size;
    cout << "Enter array size: ";
    cin >> size;
    if (size <= 0) {
        cout << "Invalid size!" << endl;
        return 1;
    }

    vector<int> arr(size);
    srand(time(0));
    
    for (int i = 0; i < size; i++) {
        arr[i] = rand() % 100 + 1;
    }

    // Sequential computation
    int min_seq = arr[0], max_seq = arr[0];
    long long sum_seq = 0;
    double avg_seq = 0;
    auto start = chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i++) {
        if (arr[i] < min_seq) min_seq = arr[i];
    }
    auto end = chrono::high_resolution_clock::now();
    double time_min_seq = chrono::duration<double>(end - start).count();

    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i++) {
        if (arr[i] > max_seq) max_seq = arr[i];
    }
    end = chrono::high_resolution_clock::now();
    double time_max_seq = chrono::duration<double>(end - start).count();

    start = chrono::high_resolution_clock::now();
    for (int i = 0; i < size; i++) {
        sum_seq += arr[i];
    }
    end = chrono::high_resolution_clock::now();
    double time_sum_seq = chrono::duration<double>(end - start).count();

    start = chrono::high_resolution_clock::now();
    avg_seq = static_cast<double>(sum_seq) / size;
    end = chrono::high_resolution_clock::now();
    double time_avg_seq = chrono::duration<double>(end - start).count();

    // Parallel computation
    int min_par = arr[0], max_par = arr[0];
    long long sum_par = 0;
    double avg_par = 0;

    start = chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(min:min_par)
    for (int i = 0; i < size; i++) {
        if (arr[i] < min_par) min_par = arr[i];
    }
    end = chrono::high_resolution_clock::now();
    double time_min_par = chrono::duration<double>(end - start).count();

    start = chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(max:max_par)
    for (int i = 0; i < size; i++) {
        if (arr[i] > max_par) max_par = arr[i];
    }
    end = chrono::high_resolution_clock::now();
    double time_max_par = chrono::duration<double>(end - start).count();

    start = chrono::high_resolution_clock::now();
    #pragma omp parallel for reduction(+:sum_par)
    for (int i = 0; i < size; i++) {
        sum_par += arr[i];
    }
    end = chrono::high_resolution_clock::now();
    double time_sum_par = chrono::duration<double>(end - start).count();

    start = chrono::high_resolution_clock::now();
    avg_par = static_cast<double>(sum_par) / size;
    end = chrono::high_resolution_clock::now();
    double time_avg_par = chrono::duration<double>(end - start).count();

    // Display results
    cout << "\n---- Sequential Computation ----\n";
    cout << "Min: " << min_seq << " | Time: " << time_min_seq << " sec\n";
    cout << "Max: " << max_seq << " | Time: " << time_max_seq << " sec\n";
    cout << "Sum: " << sum_seq << " | Time: " << time_sum_seq << " sec\n";
    cout << "Avg: " << avg_seq << " | Time: " << time_avg_seq << " sec\n";

    cout << "\n---- Parallel Computation ----\n";
    cout << "Min: " << min_par << " | Time: " << time_min_par << " sec\n";
    cout << "Max: " << max_par << " | Time: " << time_max_par << " sec\n";
    cout << "Sum: " << sum_par << " | Time: " << time_sum_par << " sec\n";
    cout << "Avg: " << avg_par << " | Time: " << time_avg_par << " sec\n";

    cout << "\n---- Speedup (Sequential / Parallel) ----\n";
    cout << "Min Speedup: " << (time_min_seq / time_min_par) << "x\n";
    cout << "Max Speedup: " << (time_max_seq / time_max_par) << "x\n";
    cout << "Sum Speedup: " << (time_sum_seq / time_sum_par) << "x\n";
    cout << "Avg Speedup: " << (time_avg_seq / time_avg_par) << "x\n";

    return 0;
}

