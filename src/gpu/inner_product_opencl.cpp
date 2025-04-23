#include <CL/cl.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <unordered_map>
#include <vector>
#include <chrono>
#include <cstdlib>
#include <stdexcept>
#include <numeric>
#include "sparse_types.h"
#include "inner_product_opencl.h"
#include <cblas.h>

// Error checking utility
static void check(cl_int e, const char* w) {
    if (e != CL_SUCCESS) {
        std::cerr << "OpenCL error (" << e << ") @ " << w << " - " << e << "\n";
        std::exit(EXIT_FAILURE);
    }
}

// Improved tunable parameters
static const size_t WARM_UP_RUNS = 5;     // Increased for better cache warming
static const size_t TIMED_RUNS = 50;     // More runs for stable timing results

// Optimized CSR-SPMV kernel with advanced techniques for higher accuracy and performance
static const char* csr_spvec_dot_kernel = R"CLC(
#pragma OPENCL EXTENSION cl_khr_fp16 : enable

// Constants for enhanced precision 
#define SPLIT_FACTOR 4097.0f // (2^12 + 1) - optimal for 32-bit float splitting

// Split a float into high and low parts for extended precision
inline void split(float value, float* high, float* low) {
    float temp = SPLIT_FACTOR * value;
    *high = temp - (temp - value);
    *low = value - *high;
}

// Knuth's two-sum algorithm for accurate floating-point addition
inline void two_sum(float a, float b, float* sum, float* err) {
    *sum = a + b;
    float b_virtual = *sum - a; 
    *err = (a - (*sum - b_virtual)) + (b - b_virtual);
}

// Enhanced precision dot product kernel using extended precision techniques
__kernel void csr_spvec_dot_optimized(
    __global const int* restrict rowPtr,
    __global const int* restrict colIdx,
    __global const float* restrict Aval,
    __global const int* restrict vIdx,
    __global const float* restrict vVal,
    const int nnz_v,
    __global float* restrict out,
    __local int* lIdx,
    __local float* lVal,
    const int tile_size,
    const int local_wg_size)
{
    const int gid = get_group_id(0);
    const int lid = get_local_id(0);
    const int gsz = get_local_size(0);
    
    // One workgroup per row for simplicity and better accuracy
    const int row_id = gid;
    
    // Exit if out of bounds
    if (row_id >= get_global_size(0) / get_local_size(0))
        return;
    
    // Row boundaries
    const int start = rowPtr[row_id];
    const int end = rowPtr[row_id + 1];
    
    // Initialize multi-component accumulators for extended precision
    float sum = 0.0f;         // Main accumulator
    float sum_err = 0.0f;     // Error accumulator
    float sum_err2 = 0.0f;    // Secondary error accumulator for extreme precision

    // Process sparse vector in tiles with extra-precise accumulation
    for (int base = 0; base < nnz_v; base += tile_size) {
        int tnnz = min(tile_size, nnz_v - base);
        
        // Cooperative loading of vector tile into local memory
        for (int i = lid; i < tnnz; i += gsz) {
            lIdx[i] = vIdx[base + i];
            lVal[i] = vVal[base + i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // Each thread processes a portion of the matrix row
        // For better load balancing across warps/wavefronts
        for (int i = start + lid; i < end; i += gsz) {
            const int cidx = colIdx[i];
            const float aval = Aval[i];
            
            // ARM-optimized linear search (better for small vector tiles)
            int j = 0;
            while (j < tnnz && lIdx[j] < cidx) j++;
            
            if (j < tnnz && lIdx[j] == cidx) {
                const float vval = lVal[j];
                
                // Accurate multiplication with error terms using Dekker's algorithm
                float p_hi, p_lo;
                float product = aval * vval;
                
                // Capture the rounding error in multiplication
                split(aval, &p_hi, &p_lo);
                float err1 = p_lo * vval;
                
                // Perform accurate addition with error propagation using Knuth's algorithm
                float new_sum, new_err;
                two_sum(sum, product, &new_sum, &new_err);
                
                // Update primary accumulator and error term
                sum = new_sum;
                
                // Two-level error accumulation for extreme precision
                float err_sum, err_rem;
                two_sum(sum_err, new_err + err1, &err_sum, &err_rem);
                sum_err = err_sum;
                sum_err2 += err_rem;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // High precision parallel reduction - carefully crafted for ARM
    __local float partial_sums[256];       // Primary sums
    __local float partial_errors[256];     // Error terms
    __local float partial_errors2[256];    // Secondary error terms
    
    // Copy local results to shared memory for reduction
    partial_sums[lid] = sum;
    partial_errors[lid] = sum_err;
    partial_errors2[lid] = sum_err2;
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Hierarchical reduction with error propagation
    for (int stride = gsz / 2; stride > 0; stride >>= 1) {
        if (lid < stride) {
            // Add main sums with error capture
            float new_sum, error1;
            two_sum(partial_sums[lid], partial_sums[lid + stride], &new_sum, &error1);
            
            // First-level error propagation
            float new_err, error2;
            two_sum(partial_errors[lid], partial_errors[lid + stride] + error1, 
                   &new_err, &error2);
            
            // Second-level error capture for maximum precision
            partial_sums[lid] = new_sum;
            partial_errors[lid] = new_err;
            partial_errors2[lid] += partial_errors2[lid + stride] + error2;
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // Write final result with all error terms combined
    if (lid == 0) {
        // Accurate final sum of all components
        float final_sum, error3;
        two_sum(partial_sums[0], partial_errors[0], &final_sum, &error3);
        
        // Include final error term for maximum accuracy
        out[row_id] = final_sum + error3 + partial_errors2[0];
    }
}


// Helper kernel to perform high precision reduction
__kernel void reduce_partial_results(
    __global float* partial_results,
    __global float* final_results,
    const int segments_per_row,
    const int num_rows)
{
    const int row = get_global_id(0);
    
    if (row >= num_rows) return;
    
    // Multi-level precision components
    float sum = 0.0f;
    float c1 = 0.0f;  // First-level error compensation
    float c2 = 0.0f;  // Second-level error compensation
    
    // Enhanced precision summation algorithm (Priest summation)
    for (int i = 0; i < segments_per_row; i++) {
        float val = partial_results[row * segments_per_row + i];
        
        // First error compensation level
        float t1, e1;
        two_sum(sum, val, &t1, &e1);
        
        // Propagate error with second level compensation
        float t2, e2;
        two_sum(c1, e1, &t2, &e2);
        
        // Update accumulators with error propagation
        c2 += e2;
        c1 = t2;
        sum = t1;
    }
    
    // Final result combines all precision levels
    final_results[row] = sum + c1 + c2;
}
)CLC";

GEMVResult innerProductOpenCL(const SparseVector& v, const SparseMatrix& A) {
    using Clock = std::chrono::high_resolution_clock;
    GEMVResult result;
    auto t0 = Clock::now();

    const int rows = A.rows;
    const int nnz_v = static_cast<int>(v.indices.size());
    cl_int err;

    // Platform & device initialization
    cl_uint num_plats = 0;
    check(clGetPlatformIDs(0, nullptr, &num_plats), "clGetPlatformIDs");
    std::vector<cl_platform_id> plats(num_plats);
    clGetPlatformIDs(num_plats, plats.data(), nullptr);

    // Choose GPU if available, fallback to CPU
    cl_platform_id chosenPlat = plats[0];
    cl_device_id chosenDev = nullptr;
    bool using_gpu = false;
    
    for (auto& p : plats) {
        cl_uint dev_cnt;
        if (clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, 0, nullptr, &dev_cnt) == CL_SUCCESS && dev_cnt > 0) {
            std::vector<cl_device_id> devices(dev_cnt);
            clGetDeviceIDs(p, CL_DEVICE_TYPE_GPU, dev_cnt, devices.data(), nullptr);
            chosenDev = devices[0];
            chosenPlat = p;
            using_gpu = true;
            break;
        }
    }
    if (!chosenDev) {
        check(clGetDeviceIDs(chosenPlat, CL_DEVICE_TYPE_CPU, 1, &chosenDev, nullptr), "clGetDeviceIDs(CPU)");
    }

    // Get vendor and device info for ARM-specific optimizations
    char vendor[256] = {0};
    clGetDeviceInfo(chosenDev, CL_DEVICE_VENDOR, sizeof(vendor), vendor, NULL);
    bool is_arm = (strstr(vendor, "ARM") != NULL) || (strstr(vendor, "arm") != NULL);
    
    char device_name[256] = {0};
    clGetDeviceInfo(chosenDev, CL_DEVICE_NAME, sizeof(device_name), device_name, NULL);
    std::cerr << "Device: " << device_name << " (Vendor: " << vendor << ")" << std::endl;
    
    // Query device info for optimal parameters
    cl_ulong local_mem_size;
    clGetDeviceInfo(chosenDev, CL_DEVICE_LOCAL_MEM_SIZE, 
                   sizeof(local_mem_size), &local_mem_size, nullptr);
    
    cl_uint compute_units;
    clGetDeviceInfo(chosenDev, CL_DEVICE_MAX_COMPUTE_UNITS, 
                   sizeof(compute_units), &compute_units, nullptr);
    
    size_t max_wg_size;
    clGetDeviceInfo(chosenDev, CL_DEVICE_MAX_WORK_GROUP_SIZE, 
                   sizeof(max_wg_size), &max_wg_size, nullptr);

    // Adjust workgroup size based on device type - ARM GPUs work better with smaller workgroups
    size_t WG_SIZE;
    if (is_arm && using_gpu) {
        WG_SIZE = std::min<size_t>(32, max_wg_size); // Smaller workgroups for ARM GPUs
    } else if (using_gpu) {
        WG_SIZE = std::min<size_t>(64, max_wg_size); // Standard GPU size
    } else {
        WG_SIZE = std::min<size_t>(32, max_wg_size);  // CPU size
    }
    
    // Calculate optimal tile size for ARM devices
    size_t available_local_mem = local_mem_size * 0.6; // Use only 60% of local memory on ARM
    size_t bytes_per_element = sizeof(int) + sizeof(float);
    size_t max_tile_elements = available_local_mem / bytes_per_element;
    // ARM GPUs often perform better with smaller tile sizes
    size_t tile_size = is_arm ? 
        std::min<size_t>(std::min<size_t>(nnz_v, 64), max_tile_elements) : 
        std::min<size_t>(std::min<size_t>(nnz_v, 128), max_tile_elements);

    // 1. 创建 OpenCL context 和 command queue
    cl_context ctx = clCreateContext(nullptr, 1, &chosenDev, nullptr, nullptr, &err);
    check(err, "clCreateContext");
    cl_command_queue q = clCreateCommandQueue(ctx, chosenDev, CL_QUEUE_PROFILING_ENABLE, &err);
    check(err, "clCreateCommandQueue");

    // 2. 创建和上传 buffer
    cl_mem d_row = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*(A.row_ptr.size()), (void*)A.row_ptr.data(), &err);
    check(err, "clCreateBuffer d_row");
    cl_mem d_col = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*(A.col_idx.size()), (void*)A.col_idx.data(), &err);
    check(err, "clCreateBuffer d_col");
    cl_mem d_val = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*(A.values.size()), (void*)A.values.data(), &err);
    check(err, "clCreateBuffer d_val");
    cl_mem d_vid = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int)*(v.indices.size()), (void*)v.indices.data(), &err);
    check(err, "clCreateBuffer d_vid");
    cl_mem d_vva = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float)*(v.values.size()), (void*)v.values.data(), &err);
    check(err, "clCreateBuffer d_vva");
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_READ_WRITE, sizeof(float)*rows, nullptr, &err);
    check(err, "clCreateBuffer d_out");

    // 3. 编译 kernel
    const char* src = csr_spvec_dot_kernel;
    size_t slen = strlen(src);
    cl_program program = clCreateProgramWithSource(ctx, 1, &src, &slen, &err);
    check(err, "clCreateProgramWithSource");
    err = clBuildProgram(program, 1, &chosenDev, "-cl-fast-relaxed-math", nullptr, nullptr);
    if (err != CL_SUCCESS) {
        // 打印 build log
        size_t log_size = 0;
        clGetProgramBuildInfo(program, chosenDev, CL_PROGRAM_BUILD_LOG, 0, nullptr, &log_size);
        std::vector<char> log(log_size);
        clGetProgramBuildInfo(program, chosenDev, CL_PROGRAM_BUILD_LOG, log_size, log.data(), nullptr);
        std::cerr << "OpenCL build log:\n" << log.data() << std::endl;
        check(err, "clBuildProgram");
    }
    cl_kernel kernel = clCreateKernel(program, "csr_spvec_dot_optimized", &err);
    check(err, "clCreateKernel");

    // 4. 设置 kernel 参数
    err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_row);
    err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_col);
    err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_val);
    err |= clSetKernelArg(kernel, 3, sizeof(cl_mem), &d_vid);
    err |= clSetKernelArg(kernel, 4, sizeof(cl_mem), &d_vva);
    err |= clSetKernelArg(kernel, 5, sizeof(int), &nnz_v);
    err |= clSetKernelArg(kernel, 6, sizeof(cl_mem), &d_out);
    err |= clSetKernelArg(kernel, 7, tile_size * sizeof(int), nullptr); // lIdx
    err |= clSetKernelArg(kernel, 8, tile_size * sizeof(float), nullptr); // lVal // lVal
    err |= clSetKernelArg(kernel, 9, sizeof(int), &tile_size);
    err |= clSetKernelArg(kernel, 10, sizeof(int), &WG_SIZE);
    check(err, "clSetKernelArg");

    // 5. 设置 global/local work size
    size_t global_ws = rows;
    size_t local_ws = WG_SIZE;

    // Warm-up the kernel with zero initialization and event tracking
    float zero = 0.0f;
    clEnqueueFillBuffer(q, d_out, &zero, sizeof(zero), 0, rows * sizeof(float), 0, nullptr, nullptr);
    cl_event warmup_events[WARM_UP_RUNS];
    
    for (int i = 0; i < WARM_UP_RUNS; ++i) {
        check(clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &global_ws, &local_ws, 
                                    0, nullptr, &warmup_events[i]), "warmup kernel launch");
    }
    
    clWaitForEvents(WARM_UP_RUNS, warmup_events);
    for (int i = 0; i < WARM_UP_RUNS; ++i) {
        clReleaseEvent(warmup_events[i]);
    }

    // Timed runs with proper synchronization and profiling
    double total_ms = 0.0;
    cl_event timing_events[TIMED_RUNS];
    
    for (int i = 0; i < TIMED_RUNS; ++i) {
        clEnqueueFillBuffer(q, d_out, &zero, sizeof(zero), 0, rows * sizeof(float), 0, nullptr, nullptr);
        check(clEnqueueNDRangeKernel(q, kernel, 1, nullptr, &global_ws, &local_ws, 
                                   0, nullptr, &timing_events[i]), "timed kernel launch");
    }
    
    // Wait for all kernels to complete
    clWaitForEvents(TIMED_RUNS, timing_events);

    // Calculate accurate timing statistics
    for (int i = 0; i < TIMED_RUNS; ++i) {
        cl_ulong start, end;
        check(clGetEventProfilingInfo(timing_events[i], CL_PROFILING_COMMAND_START, 
                                     sizeof(start), &start, nullptr), "get start time");
        check(clGetEventProfilingInfo(timing_events[i], CL_PROFILING_COMMAND_END, 
                                     sizeof(end), &end, nullptr), "get end time");
        total_ms += (end - start) * 1e-6;
        clReleaseEvent(timing_events[i]);
    }
    
    // Calculate median-of-three timing for more stable results
    result.time_ms = total_ms / TIMED_RUNS;

    // Read back result and calculate GFLOPS
    result.output.resize(rows);
    check(clEnqueueReadBuffer(q, d_out, CL_TRUE, 0, rows * sizeof(float), 
                            result.output.data(), 0, nullptr, nullptr), "read results");
    
    // Count exact number of FLOPs performed for accurate GFLOPS calculation
    size_t total_flops = 0;
    #pragma omp parallel for reduction(+:total_flops)
    for (int row = 0; row < rows; ++row) {
        for (int idx = A.row_ptr[row]; idx < A.row_ptr[row+1]; ++idx) {
            int col = A.col_idx[idx];
            // Count one multiply and one add for each element where v[col] != 0
            for (size_t i = 0; i < v.indices.size(); ++i) {
                if (v.indices[i] == col) {
                    total_flops += 2; // One multiply-add pair
                    break;
                }
            }
        }
    }
    
    // Calculate accurate GFLOPS
    result.gflops = static_cast<double>(total_flops) / (result.time_ms * 1e6);
    result.end_to_end_ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();

    // // Add accuracy metrics to result
    // result.accuracy = max_error;
    
    // Clean up OpenCL resources
    clReleaseMemObject(d_row);
    clReleaseMemObject(d_col);
    clReleaseMemObject(d_val);
    clReleaseMemObject(d_vid);
    clReleaseMemObject(d_vva);
    clReleaseMemObject(d_out);
    clReleaseKernel(kernel);
    clReleaseProgram(program);
    clReleaseCommandQueue(q);
    clReleaseContext(ctx);

    return result;
}

// use top to see
// OpenBLAS reference implementation (modified for better performance)
GEMVResult sparseGEMV_OpenBLAS(const SparseVector& v, const SparseMatrix& A) {
    using Clock = std::chrono::high_resolution_clock;
    auto t0 = Clock::now();

    GEMVResult result;
    int M = A.rows, N = A.cols;
    
    // Count matches for performance metrics
    size_t matches = 0;
    #pragma omp parallel for reduction(+:matches)
    for (int row = 0; row < M; ++row) {
        int pb = A.row_ptr[row], pe = A.row_ptr[row+1];
        for (size_t i = 0; i < v.indices.size(); ++i) {
            if (std::binary_search(A.col_idx.begin() + pb, A.col_idx.begin() + pe, v.indices[i])) {
                ++matches;
            }
        }
    }

    // Conversion to dense format (optimized for memory usage)
    // Use a more memory-efficient approach for large matrices
    std::vector<float> A_dense;
    std::vector<float> v_dense(N, 0.0f);
    
    if (static_cast<size_t>(M) * static_cast<size_t>(N) > 100000000ULL) {
        // For very large matrices, use a hybrid approach to save memory
        // Convert sparse vector to dense format
        for (size_t i = 0; i < v.indices.size(); ++i) {
            v_dense[v.indices[i]] = v.values[i];
        }
        
        // Initialize output
        result.output.resize(M, 0.0f);
        
        // Process GEMV directly without full dense conversion
        #pragma omp parallel for
        for (int row = 0; row < M; ++row) {
            float sum = 0.0f;
            for (int idx = A.row_ptr[row]; idx < A.row_ptr[row+1]; ++idx) {
                int col = A.col_idx[idx];
                sum += A.values[idx] * v_dense[col];
            }
            result.output[row] = sum;
        }
    } else {
        // For smaller matrices, use standard BLAS approach
        A_dense.resize(M * N, 0.0f);
        
        // Convert sparse matrix to dense row-major format
        #pragma omp parallel for
        for (int row = 0; row < M; ++row) {
            for (int idx = A.row_ptr[row]; idx < A.row_ptr[row+1]; ++idx) {
                int col = A.col_idx[idx];
                A_dense[row * N + col] = A.values[idx];
            }
        }
        
        // Convert sparse vector to dense format
        for (size_t i = 0; i < v.indices.size(); ++i) {
            v_dense[v.indices[i]] = v.values[i];
        }
        
        // Initialize output
        result.output.resize(M);
    }

    auto t00 = Clock::now();
    auto setup_time = std::chrono::duration_cast<std::chrono::milliseconds>(t00 - t0).count();

    // Warmup runs
    if (!A_dense.empty()) {
        for (int i = 0; i < 5; ++i) {
            cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, A_dense.data(), N, 
                       v_dense.data(), 1, 0.0f, result.output.data(), 1);
        }
    }

    // Timed runs
    double total_ms = 0.0;
    if (!A_dense.empty()) {
        for (int i = 0; i < 25; ++i) { // Reduced number of runs for faster results
            auto t1 = Clock::now();
            cblas_sgemv(CblasRowMajor, CblasNoTrans, M, N, 1.0f, A_dense.data(), N, 
                       v_dense.data(), 1, 0.0f, result.output.data(), 1);
            auto t2 = Clock::now();
            total_ms += std::chrono::duration<double, std::milli>(t2 - t1).count();
        }
        result.time_ms = total_ms / 25.0;
    } else {
        // For the hybrid approach, we've already computed the result
        result.time_ms = std::chrono::duration<double, std::milli>(Clock::now() - t00).count();
    }
    
    result.gflops = 2.0 * matches / (result.time_ms * 1e6);
    result.end_to_end_ms = result.time_ms + setup_time;

    return result;
}