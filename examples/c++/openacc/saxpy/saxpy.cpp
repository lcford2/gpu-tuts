#include <chrono>
#include <iostream>

using namespace std;

int main(int argc, char **argv) {
  // initialize some information
  int N = 1000000;
  // note that GPUs generally use single precision FPs
  // that is why these are floats instead of doubles
  float a = 3.0f;
  float x[N], y[N];

  for (int i = 0; i < N; ++i) {
    x[i] = 2.0f;
    y[i] = 1.0f;
  }

  // open acc kernel (parallel loop) with timing
  auto gpu_start = chrono::high_resolution_clock::now();
#pragma acc kernels
  for (int i = 0; i < N; ++i) {
    y[i] = a * x[i] + y[i];
  }
  auto gpu_stop = chrono::high_resolution_clock::now();

  // cpu loop with timing
  auto cpu_start = chrono::high_resolution_clock::now();
  for (int i = 0; i < N; ++i) {
    y[i] = a * x[i] + y[i];
  }
  auto cpu_stop = chrono::high_resolution_clock::now();

  // get duriations as doubles
  chrono::duration<double, milli> gpu_time = gpu_stop - gpu_start;
  chrono::duration<double, milli> cpu_time = cpu_stop - cpu_start;

  cout << "GPU Time: " << gpu_time.count() << " milliseconds" << endl;
  cout << "CPU Time: " << cpu_time.count() << " milliseconds" << endl;
}
