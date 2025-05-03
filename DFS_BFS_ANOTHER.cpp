#include <iostream>
#include <vector>
#include <queue>
#include <stack>
#include <omp.h>
#include <chrono>
#include <cstdlib>
#include <ctime>

using namespace std;
using namespace std::chrono;

class Graph {
    int V;
    vector<vector<int>> adj;

public:
    Graph(int V) {
        this->V = V;
        adj.resize(V);
    }

    void addEdge(int v, int w) {
        adj[v].push_back(w);
        adj[w].push_back(v);
    }

    void serialBFS(int start);
    void parallelBFS(int start);
    void serialDFS(int start);
    void parallelDFS(int start);
    void measurePerformance(int startNode);
};

void Graph::serialBFS(int start) {
    vector<bool> visited(V, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int node = q.front();
        q.pop();
        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                visited[neighbor] = true;
                q.push(neighbor);
            }
        }
    }
}

void Graph::parallelBFS(int start) {
    vector<bool> visited(V, false);
    queue<int> q;
    visited[start] = true;
    q.push(start);

    while (!q.empty()) {
        int qSize = q.size();

        #pragma omp parallel for
        for (int i = 0; i < qSize; i++) {
            int node;
            #pragma omp critical
            {
                if (!q.empty()) {
                    node = q.front();
                    q.pop();
                } else {
                    continue;
                }
            }

            for (int neighbor : adj[node]) {
                #pragma omp critical
                {
                    if (!visited[neighbor]) {
                        visited[neighbor] = true;
                        q.push(neighbor);
                    }
                }
            }
        }
    }
}

void Graph::serialDFS(int start) {
    vector<bool> visited(V, false);
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int node = s.top();
        s.pop();

        if (!visited[node]) {
            visited[node] = true;
        }

        for (int neighbor : adj[node]) {
            if (!visited[neighbor]) {
                s.push(neighbor);
            }
        }
    }
}

void Graph::parallelDFS(int start) {
    vector<bool> visited(V, false);
    stack<int> s;
    s.push(start);

    while (!s.empty()) {
        int node;
        #pragma omp critical
        {
            if (!s.empty()) {
                node = s.top();
                s.pop();
            } else {
                continue;
            }
        }

        if (!visited[node]) {
            visited[node] = true;
        }

        #pragma omp parallel for
        for (int i = 0; i < adj[node].size(); i++) {
            int neighbor = adj[node][i];
            #pragma omp critical
            {
                if (!visited[neighbor]) {
                    s.push(neighbor);
                }
            }
        }
    }
}

void Graph::measurePerformance(int startNode) {
    clock_t t1 = clock();
    serialBFS(startNode);
    clock_t t2 = clock();
    double serialBfsTime = double(t2 - t1) / CLOCKS_PER_SEC;

    clock_t t3 = clock();
    parallelBFS(startNode);
    clock_t t4 = clock();
    double parallelBfsTime = double(t4 - t3) / CLOCKS_PER_SEC;

    clock_t t5 = clock();
    serialDFS(startNode);
    clock_t t6 = clock();
    double serialDfsTime = double(t6 - t5) / CLOCKS_PER_SEC;

    clock_t t7 = clock();
    parallelDFS(startNode);
    clock_t t8 = clock();
    double parallelDfsTime = double(t8 - t7) / CLOCKS_PER_SEC;

    cout << "\nPerformance:\n";
    cout << "Serial BFS Time: " << serialBfsTime << " sec\n";
    cout << "Parallel BFS Time: " << parallelBfsTime << " sec\n";
     cout << "Speedup for BFS Sort: " << (serialBfsTime / parallelBfsTime) << endl;

    cout << "Serial DFS Time: " << serialDfsTime << " sec\n";
    cout << "Parallel DFS Time: " << parallelDfsTime << " sec\n";
    
    cout << "Speedup for DFS Sort: " << (serialDfsTime / parallelDfsTime) << endl;
}

int main() {
    int N = 999999;
    Graph g(N);

    cout << "Graph with " << N << " nodes is being created...\n";

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < 5; j++) {
            int neighbor = rand() % N;
            if (neighbor != i) {
                g.addEdge(i, neighbor);
            }
        }
    }

    g.measurePerformance(0);

    return 0;
}

