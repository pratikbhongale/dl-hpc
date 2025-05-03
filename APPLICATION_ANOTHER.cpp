#include <iostream>
#include <cstdlib>
#include <ctime>
#include <chrono>
#include <omp.h>

#define DATA_SIZE 1000000
#define TEST_SIZE 100000

using namespace std;
using namespace std::chrono;

// ?? Linear Regression: y = ax + b
// ?? Sequential Version
void linear_regression_sequential(float* x, float* y, int n, float& a, float& b) {
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;

    for (int i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    b = (sum_y - a * sum_x) / n;
}

// ?? Parallel Version using OpenMP
void linear_regression_parallel(float* x, float* y, int n, float& a, float& b) {
    float sum_x = 0, sum_y = 0, sum_xy = 0, sum_xx = 0;

    #pragma omp parallel for reduction(+:sum_x, sum_y, sum_xy, sum_xx)
    for (int i = 0; i < n; ++i) {
        sum_x += x[i];
        sum_y += y[i];
        sum_xy += x[i] * y[i];
        sum_xx += x[i] * x[i];
    }

    a = (n * sum_xy - sum_x * sum_y) / (n * sum_xx - sum_x * sum_x);
    b = (sum_y - a * sum_x) / n;
}

// ?? Mean Squared Error (MSE)
float calculate_mse(float* x_test, float* y_test, int size, float a, float b) {
    float mse = 0;

    #pragma omp parallel for reduction(+:mse)
    for (int i = 0; i < size; ++i) {
        float predicted = a * x_test[i] + b;
        mse += (predicted - y_test[i]) * (predicted - y_test[i]);
    }

    return mse / size;
}

int main() {
    float* x = new float[DATA_SIZE];
    float* y = new float[DATA_SIZE];

    srand(time(0));

    // ?? Generate linear data: y = 5x + small random noise
    for (int i = 0; i < DATA_SIZE; ++i) {
        x[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        y[i] = 5.0f * x[i] + static_cast<float>(rand()) / RAND_MAX;
    }

    float a_seq, b_seq, a_par, b_par;

    // ?? Run Sequential
    auto start_seq = high_resolution_clock::now();
    linear_regression_sequential(x, y, DATA_SIZE, a_seq, b_seq);
    auto end_seq = high_resolution_clock::now();
    double time_seq = duration<double>(end_seq - start_seq).count();

    // ? Run Parallel
    auto start_par = high_resolution_clock::now();
    linear_regression_parallel(x, y, DATA_SIZE, a_par, b_par);
    auto end_par = high_resolution_clock::now();
    double time_par = duration<double>(end_par - start_par).count();

    // ?? Test Data for MSE
    float* x_test = new float[TEST_SIZE];
    float* y_test = new float[TEST_SIZE];
    for (int i = 0; i < TEST_SIZE; ++i) {
        x_test[i] = static_cast<float>(rand()) / RAND_MAX * 100.0f;
        y_test[i] = 5.0f * x_test[i] + static_cast<float>(rand()) / RAND_MAX;
    }

    float mse_seq = calculate_mse(x_test, y_test, TEST_SIZE, a_seq, b_seq);
    float mse_par = calculate_mse(x_test, y_test, TEST_SIZE, a_par, b_par);
    float speedup = time_seq / time_par;

    // ?? Print Results
    cout << fixed;
    cout << "\n?? Time (Sequential): " << time_seq << "s";
    cout << "\n?? Time (Parallel)  : " << time_par << "s";
    cout << "\n? Speedup          : " << speedup;

    cout << "\n\n?? Predicted Equations:";
    cout << "\nSequential : y = " << a_seq << "x + " << b_seq;
    cout << "\nParallel   : y = " << a_par << "x + " << b_par;

    cout << "\n\n?? Sample Comparison (first 2 values):";
    cout << "\nX\t\tActual Y\tSeq Y\t\tPar Y";
    for (int i = 0; i < 2; ++i) {
        float y_seq = a_seq * x_test[i] + b_seq;
        float y_par = a_par * x_test[i] + b_par;
        cout << "\n" << x_test[i] << "\t" << y_test[i] << "\t" << y_seq << "\t" << y_par;
    }

    cout << "\n\n?? Mean Squared Error:";
    cout << "\nSequential : " << mse_seq;
    cout << "\nParallel   : " << mse_par << endl;

    // ?? Clean up
    delete[] x;
    delete[] y;
    delete[] x_test;
    delete[] y_test;

    return 0;
}

