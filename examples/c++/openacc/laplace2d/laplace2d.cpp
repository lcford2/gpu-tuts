#include <iostream>
#include <chrono>
#include <cmath>

using namespace std;

void serial(long rows, long columns)
{
    float A[rows][columns];
    float A_new[rows][columns];

    // init matrices
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<columns; ++j) {
            A[i][j] = 0.0;
            A_new[i][j] = 0.0;
        }
    }
    // give A a spike in the middle
    A[rows/2][columns/2] = 2.0;

    // setup jacobi information
    float dt=1.0, max_error=1e-4;
    int iter=0, max_iter=1000;

    // start timer
    auto start_time = chrono::high_resolution_clock::now();
    // loop until error critera or iters is reached
    while (dt > max_error && iter <= max_iter) {
        for (int i=1; i<rows-1; ++i) {
            for (int j=1; j<columns-1; ++j) {
                A_new[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j+1] + A[i][j-1]);
            }
        }

        dt = 0.0;
        for (int i=1; i<rows-1; ++i) {
            for (int j=1; j<columns-1; ++j) {
                dt = fmax(fabs(A_new[i][j] - A[i][j]), dt);
                A[i][j] = A_new[i][j];
            }
        }
        ++iter;
    }
    auto stop_time = chrono::high_resolution_clock::now();

    chrono::duration<float, milli> elapsed = stop_time - start_time;
    cout << "Serial Time: " << elapsed.count() / 1000 << " seconds\n";
}

void bad_acceleration(long rows, long columns)
{
    float A[rows][columns];
    float A_new[rows][columns];

    // init matrices
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<columns; ++j) {
            A[i][j] = 0.0;
            A_new[i][j] = 0.0;
        }
    }
    // give A a spike in the middle
    A[rows/2][columns/2] = 2.0;

    // setup jacobi information
    float dt=1.0, max_error=1e-4;
    int iter=0, max_iter=1000;

    // start timer
    auto start_time = chrono::high_resolution_clock::now();
    // loop until error critera or iters is reached
    // the outer loop cannot be accelerated because
    // of the data dependency on dt
    while (dt > max_error && iter <= max_iter) {
#pragma acc kernels
        for (int i=1; i<rows-1; ++i) {
            for (int j=1; j<columns-1; ++j) {
                A_new[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j+1] + A[i][j-1]);
            }
        }

        dt = 0.0;
#pragma acc parallel loop reduction(max:dt)
        for (int i=1; i<rows-1; ++i) {
            for (int j=1; j<columns-1; ++j) {
                dt = fmax(fabs(A_new[i][j] - A[i][j]), dt);
                A[i][j] = A_new[i][j];
            }
        }
        ++iter;
    }
    auto stop_time = chrono::high_resolution_clock::now();

    chrono::duration<float, milli> elapsed = stop_time - start_time;
    cout << "Bad Acceleration Time: " << elapsed.count() / 1000 << " seconds\n";
}

void good_acceleration(long rows, long columns)
{
    float A[rows][columns];
    float A_new[rows][columns];

    // init matrices
    for (int i=0; i<rows; ++i) {
        for (int j=0; j<columns; ++j) {
            A[i][j] = 0.0;
            A_new[i][j] = 0.0;
        }
    }
    // give A a spike in the middle
    A[rows/2][columns/2] = 2.0;

    // setup jacobi information
    float dt=1.0, max_error=1e-4;
    int iter=0, max_iter=1000;

    // start timer
    auto start_time = chrono::high_resolution_clock::now();
    // loop until error critera or iters is reached
    // the outer loop cannot be accelerated because
    // of the data dependency on dt
#pragma acc data copy(A) create(A_new) copy(dt) copy(max_error) copy(iter) copy(max_iter)
    while (dt > max_error && iter <= max_iter) {
#pragma acc kernels
        for (int i=1; i<rows-1; ++i) {
            for (int j=1; j<columns-1; ++j) {
                A_new[i][j] = 0.25 * (A[i+1][j] + A[i-1][j] + A[i][j+1] + A[i][j-1]);
            }
        }

        dt = 0.0;
#pragma acc parallel loop reduction(max:dt)
        for (int i=1; i<rows-1; ++i) {
            for (int j=1; j<columns-1; ++j) {
                dt = fmax(fabs(A_new[i][j] - A[i][j]), dt);
                A[i][j] = A_new[i][j];
            }
        }
        ++iter;
    }
    auto stop_time = chrono::high_resolution_clock::now();

    chrono::duration<float, milli> elapsed = stop_time - start_time;
    cout << "Good Acceleration Time: " << elapsed.count() / 1000 << " seconds\n";
}


int main(int argc, char **argv)
{
    long rows, columns;
    if (argc > 2) {
        rows = atoi(argv[1]);
        columns = atoi(argv[2]);
    } else {
        rows = 1000;
        columns = 1000;
    }

    cout << "Grid Size: [" << rows << ", " << columns << "]\n";
    serial(rows, columns);
    bad_acceleration(rows, columns);
    good_acceleration(rows, columns);
}
