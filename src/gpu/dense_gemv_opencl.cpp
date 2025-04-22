#include <CL/cl.h>
#include <vector>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <chrono>
#include <cblas.h>
#include "sparse_types.h"
#include "dense_gemv_opencl.h"
#include <cassert>
#include <algorithm>
#include <cmath>

// Constants to be used by the host code - must match kernel definitions
static const int ROWS_PER_WORKITEM = 4;  // Each thread processes multiple rows
#define ROWS_PER_WI ROWS_PER_WORKITEM    // Define ROWS_PER_WI to match kernel
static const int TILE_SIZE_K = 512;      // Tile size for x vector

//--------------------------------------------------------------
// Numerically stable kernels with optimized performance
//--------------------------------------------------------------
static const char* kernelSource = R"CLC(
    #pragma OPENCL EXTENSION cl_khr_subgroups : enable
    #define VEC_SIZE 4
    #define ROWS_PER_WI 4
    #ifndef TILE_K
    #define TILE_K 256
    #endif
    
    __kernel void optimized_kahan_gemv(
        __global const float* A,
        __global const float* x,
        const int cols,
        const int rows,
        __global float* y,
        __local float* tileX)
    {
        int gid = get_global_id(0);
        int row_base = gid * ROWS_PER_WI;
        int lid = get_local_id(0);
        int wg  = get_local_size(0);
    
        for (int k = 0; k < cols; k += TILE_K) {
            int tile = min(TILE_K, cols - k);
            for (int i = lid * VEC_SIZE; i < tile; i += wg * VEC_SIZE) {
                if (i + VEC_SIZE <= tile) {
                    float4 v = vload4(0, &x[k + i]);
                    vstore4(v, 0, &tileX[i]);
                } else {
                    for (int j = i; j < tile; ++j)
                        tileX[j] = x[k + j];
                }
            }
            barrier(CLK_LOCAL_MEM_FENCE);
    
            for (int r = 0; r < ROWS_PER_WI; ++r) {
                int row = row_base + r;
                if (row >= rows) continue;
                int offset = row * cols + k;
                float sum = 0.f, c = 0.f;
                for (int j = 0; j < tile; ++j) {
                    float prod = A[offset + j] * tileX[j];
                    float y = prod - c;
                    float t = sum + y;
                    c = (t - sum) - y;
                    sum = t;
                }
                if (k == 0)
                    y[row] = sum;
                else
                    y[row] += sum;
            }
            barrier(CLK_LOCAL_MEM_FENCE);
        }
    }
    )CLC";

//--------------------------------------------------------------
// Helper to pick the best available device
//--------------------------------------------------------------
static void pickBestDevice(cl_device_id &dev) {
    cl_uint nPlat;
    clGetPlatformIDs(0, nullptr, &nPlat);
    std::vector<cl_platform_id> plats(nPlat);
    clGetPlatformIDs(nPlat, plats.data(), nullptr);
    cl_device_id best = nullptr;
    cl_uint bestCU = 0;
    for (auto p : plats) {
        cl_uint nDev;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &nDev) == CL_SUCCESS && nDev) {
            std::vector<cl_device_id> devs(nDev);
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, nDev, devs.data(), nullptr);
            for (auto d : devs) {
                cl_uint cu;
                clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cu), &cu, nullptr);
                if (cu > bestCU) { bestCU = cu; best = d; }
            }
        }
    }
    if (!best) {
        for (auto p : plats) {
            if (clGetDeviceIDs(p, CL_DEVICE_TYPE_CPU, 1, &best, nullptr) == CL_SUCCESS)
                break;
        }
    }
    dev = best;
}

GEMVResult denseGEMV_OpenCL(const DenseVector &x, const DenseMatrix &A) {
    if (A.cols != x.size)
        throw std::runtime_error("Dimension mismatch");

    auto start = std::chrono::high_resolution_clock::now();
    cl_device_id dev;
    pickBestDevice(dev);
    cl_int err;
    cl_context ctx = clCreateContext(nullptr, 1, &dev, nullptr, nullptr, &err);
    cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
    cl_command_queue q = clCreateCommandQueueWithProperties(ctx, dev, props, &err);
    cl_ulong lm;
    clGetDeviceInfo(dev, CL_DEVICE_LOCAL_MEM_SIZE, sizeof(lm), &lm, nullptr);
    size_t tileK = std::min<size_t>(A.cols, (lm / 2) / sizeof(float));
    std::string opts = "-cl-std=CL2.0 -DTILE_K=" + std::to_string(tileK) + " -cl-fast-relaxed-math";
    const char* src = kernelSource;
    size_t sl = strlen(src);
    cl_program prog = clCreateProgramWithSource(ctx, 1, &src, &sl, &err);
    err = clBuildProgram(prog, 1, &dev, opts.c_str(), nullptr, nullptr);
    cl_kernel k = clCreateKernel(prog, "optimized_kahan_gemv", &err);

    size_t wg = 64;
    size_t gsz = ((A.rows + ROWS_PER_WI - 1) / ROWS_PER_WI + wg - 1) / wg * wg;
    size_t mBytes = sizeof(float)*A.rows*A.cols;
    size_t vBytes = sizeof(float)*x.size;
    size_t yBytes = sizeof(float)*A.rows;
    cl_mem dA = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, mBytes, (void*)A.values.data(), &err);
    cl_mem dX = clCreateBuffer(ctx, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR, vBytes, (void*)x.values.data(), &err);
    cl_mem dY = clCreateBuffer(ctx, CL_MEM_READ_WRITE, yBytes, nullptr, &err);
    clSetKernelArg(k, 0, sizeof(cl_mem), &dA);
    clSetKernelArg(k, 1, sizeof(cl_mem), &dX);
    clSetKernelArg(k, 2, sizeof(int), &A.cols);
    clSetKernelArg(k, 3, sizeof(int), &A.rows);
    clSetKernelArg(k, 4, sizeof(cl_mem), &dY);
    clSetKernelArg(k, 5, tileK*sizeof(float), nullptr);

    for (int i=0; i<3; ++i) {
        clEnqueueNDRangeKernel(q, k, 1, nullptr, &gsz, &wg, 0, nullptr, nullptr);
        clFinish(q);
    }

    double t = 0;
    for (int i=0; i<10; ++i) {
        cl_event ev;
        clEnqueueNDRangeKernel(q, k, 1, nullptr, &gsz, &wg, 0, nullptr, &ev);
        clWaitForEvents(1,&ev);
        cl_ulong st, en;
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(st), &st, nullptr);
        clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(en), &en, nullptr);
        t += (en - st) * 1e-6;
        clReleaseEvent(ev);
    }

    GEMVResult res;
    res.time_ms = t / 10;
    res.gflops = 2.0 * A.rows * A.cols / (res.time_ms * 1e6);
    res.output.resize(A.rows);
    clEnqueueReadBuffer(q, dY, CL_TRUE, 0, yBytes, res.output.data(), 0, nullptr, nullptr);
    clReleaseMemObject(dA);
    clReleaseMemObject(dX);
    clReleaseMemObject(dY);
    clReleaseKernel(k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);
    auto end = std::chrono::high_resolution_clock::now();
    res.end_to_end_ms = std::chrono::duration<double, std::milli>(end - start).count();
    return res;
}

//-----------------------------------------------------------------------------
// CPU version: OpenBLAS
//-----------------------------------------------------------------------------
GEMVResult denseGEMV_OpenBLAS(const DenseVector &x, const DenseMatrix &A)
{
    assert(A.values.size() == size_t(A.rows) * size_t(A.cols));
    assert(x.values.size() == size_t(x.size));
    using Clock = std::chrono::high_resolution_clock;

    auto t0 = Clock::now();

    if (A.cols != x.size)
        throw std::runtime_error("Dim mismatch");

    std::vector<float> y(A.rows);
    auto t00 = Clock::now();
    auto setup_time = std::chrono::duration_cast<std::chrono::milliseconds>(t00 - t0).count(); // milliseconds

    auto t1 = Clock::now();

    // Warmup runs
    for (int i = 0; i < 10; ++i) {
        cblas_sgemv(CblasRowMajor, CblasNoTrans, A.rows, A.cols, 1.0f, A.values.data(), A.cols, x.values.data(), 1, 0.0f, y.data(), 1);
    }
    auto t2 = Clock::now();

    // Timing runs
    double total_ms = 0.0;
    for (int i = 0; i < 50; ++i) {
        auto ti0 = Clock::now();
        cblas_sgemv(CblasRowMajor, CblasNoTrans, A.rows, A.cols, 1.0f, A.values.data(), A.cols, x.values.data(), 1, 0.0f, y.data(), 1);
        auto ti1 = Clock::now();
        total_ms += std::chrono::duration<double, std::milli>(ti1 - ti0).count();
    }
    double avg_ms = total_ms / 50.0;
    double flops  = 2.0 * double(A.rows) * double(A.cols);
    double gflops = flops / (avg_ms * 1e6);

    return GEMVResult{y, avg_ms, gflops, setup_time + avg_ms};
}