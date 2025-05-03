#include <iostream>
#include <ctime>
#include <cstdlib>
#include <omp.h>

using namespace std;

void swap(int &a, int &b) {
    int temp = a;
    a = b;
    b = temp;
}

void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; ++i) {
        for (int j = 0; j < n - i - 1; ++j) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void parallelBubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; ++i) {
        #pragma omp parallel for
        for (int j = 0; j < n - i - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }

        #pragma omp parallel for
        for (int j = 1; j < n - i - 1; j += 2) {
            if (arr[j] > arr[j + 1]) {
                swap(arr[j], arr[j + 1]);
            }
        }
    }
}

void merge(int arr[], int left, int mid, int right) {
    int n1 = mid - left + 1;
    int n2 = right - mid;

    int* L = new int[n1];
    int* R = new int[n2];

    for (int i = 0; i < n1; i++)
        L[i] = arr[left + i];
    for (int j = 0; j < n2; j++)
        R[j] = arr[mid + 1 + j];

    int i = 0, j = 0, k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k++] = L[i++];
        } else {
            arr[k++] = R[j++];
        }
    }

    while (i < n1)
        arr[k++] = L[i++];
    while (j < n2)
        arr[k++] = R[j++];

    delete[] L;
    delete[] R;
}

void mergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;
        mergeSort(arr, left, mid);
        mergeSort(arr, mid + 1, right);
        merge(arr, left, mid, right);
    }
}

void parallelMergeSort(int arr[], int left, int right) {
    if (left < right) {
        int mid = left + (right - left) / 2;

        #pragma omp parallel sections
        {
            #pragma omp section
            parallelMergeSort(arr, left, mid);

            #pragma omp section
            parallelMergeSort(arr, mid + 1, right);
        }

        merge(arr, left, mid, right);
    }
}

int main() {
    srand(time(0));  

    int n;
    cout << "Enter the size of the array: ";
    cin >> n;

    int* originalArr = new int[n];

    for (int i = 0; i < n; i++) {
        originalArr[i] = rand() % 1000;  
    }

    int* arr1 = new int[n];
    int* arr2 = new int[n];
    int* arr3 = new int[n];
    int* arr4 = new int[n];

    // Copy original array into different arrays
    copy(originalArr, originalArr + n, arr1);
    copy(originalArr, originalArr + n, arr2);
    copy(originalArr, originalArr + n, arr3);
    copy(originalArr, originalArr + n, arr4);

    // Sequential Bubble Sort
    clock_t start = clock();
    bubbleSort(arr1, n);
    clock_t end = clock();
    double sequentialBubbleTime = double(end - start) / CLOCKS_PER_SEC;

    start = clock();
    parallelBubbleSort(arr2, n);
    end = clock();
    double parallelBubbleTime = double(end - start) / CLOCKS_PER_SEC;

    start = clock();
    mergeSort(arr3, 0, n - 1);
    end = clock();
    double sequentialMergeTime = double(end - start) / CLOCKS_PER_SEC;

    start = clock();
    parallelMergeSort(arr4, 0, n - 1);
    end = clock();
    double parallelMergeTime = double(end - start) / CLOCKS_PER_SEC;

    cout << "Sequential Bubble Sort Time: " << sequentialBubbleTime << " seconds" << endl;
    cout << "Parallel Bubble Sort Time: " << parallelBubbleTime << " seconds" << endl;
    cout << "Speedup for Bubble Sort: " << (sequentialBubbleTime / parallelBubbleTime) << endl;

    cout << "Sequential Merge Sort Time: " << sequentialMergeTime << " seconds" << endl;
    cout << "Parallel Merge Sort Time: " << parallelMergeTime << " seconds" << endl;
    cout << "Speedup for Merge Sort: " << (sequentialMergeTime / parallelMergeTime) << endl;

    // Free dynamically allocated memory
    delete[] originalArr;
    delete[] arr1;
    delete[] arr2;
    delete[] arr3;
    delete[] arr4;

    return 0;
}
