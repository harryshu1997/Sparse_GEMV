#include <iostream>
#include <vector>
#include <algorithm>
#include <random>
#include <chrono>
#include <cmath>
#include <fstream>

#include "types/sparse_types.h"
#include "cpu/inner_product_cpu.h"
#include "cpu/dense_gemv_cpu.h"
#ifdef ENABLE_OPENCL
#include "gpu/inner_product_opencl.h"
#include "gpu/dense_gemv_opencl.h"
#endif

constexpr float SPARSITY = 0.99f;


static SparseVector generateSparseVector(int size, float sparsity) {
    std::default_random_engine eng(42);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    SparseVector v; v.size = size;
    for (int i = 0; i < size; ++i) {
        if (dist(eng) > sparsity) {
            v.indices.push_back(i);
            v.values.push_back(dist(eng));
        }
    }
    return v;
}

static SparseMatrix generateSparseMatrix(int rows, int cols, float sparsity) {
    std::default_random_engine eng(43);
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    int nnz_per_row = std::ceil((1.f - sparsity) * cols);
    SparseMatrix A; A.rows = rows; A.cols = cols;
    A.row_ptr.push_back(0);
    std::vector<int> col_buf(cols);
    std::iota(col_buf.begin(), col_buf.end(), 0);
    for (int r = 0; r < rows; ++r) {
        std::shuffle(col_buf.begin(), col_buf.end(), eng);
        for (int i = 0; i < nnz_per_row; ++i) {
            A.col_idx.push_back(col_buf[i]);
            A.values.push_back(dist(eng));
        }
        A.row_ptr.push_back(A.row_ptr.back() + nnz_per_row);
    }
    return A;
}

static DenseVector generateDenseVector(int size) {
    std::default_random_engine eng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    DenseVector v; v.size = size;
    v.values.resize(size);
    for (int i = 0; i < size; ++i) v.values[i] = dist(eng);
    return v;
}

static DenseMatrix generateDenseMatrix(int rows, int cols) {
    std::default_random_engine eng(std::random_device{}());
    std::uniform_real_distribution<float> dist(0.f, 1.f);
    DenseMatrix m; m.rows = rows; m.cols = cols;
    m.values.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) m.values[i] = dist(eng);
    return m;
}

static double calc_l2_error(const std::vector<float>& ref, const std::vector<float>& test) {
    if (ref.size() != test.size()) return -1.0;
    double err = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        double diff = double(ref[i]) - double(test[i]);
        err += diff * diff;
    }
    return std::sqrt(err);
}

static void log_result(std::ofstream& log, const std::string& type, int rows, int cols, float sparsity, double time_ms, double gflops, double end_to_end, double err = 0.0) {
    log << type << "," << rows << "," << cols << "," << sparsity << "," << time_ms << "," << gflops << "," << end_to_end << ","  << err << "\n";
}

static void print_log(){
    std::ifstream log("/data/local/tmp/sparse_gemv/benchmark_log.csv");
    std::string line;
    while (std::getline(log, line)) {
        std::cout << line << "\n";
    }
    log.close();
}


int main() {
    std::ofstream log("/data/local/tmp/sparse_gemv/benchmark_log.csv");
    log << "Type,Rows,Cols,Sparsity,Avg Inference (50 times) Time (ms),GFLOPS,SetUp Time(ms),Error\n";

    std::vector<int> vec_sizes = {512, 1024, 2048, 4096, 8192};
    std::vector<int> mat_rows  = {1000, 5000, 10000, 20000, 30000, 40000, 50000, 60000 };
    std::vector<float> sparsities = {0.90f, 0.95f, 0.99f};

    // std::vector<int> vec_sizes = {4096};
    // std::vector<int> mat_rows  = {10000};
    // std::vector<float> sparsities = {0.95f};

    for (int vec_size : vec_sizes) {
        for (int rows : mat_rows) {
            for (float sparsity : sparsities) {
                std::cout << "Testing vec=" << vec_size << ", rows=" << rows << ", sparsity=" << sparsity << "\n";

                SparseVector vec = generateSparseVector(vec_size, sparsity);
                SparseMatrix mat = generateSparseMatrix(rows, vec_size, sparsity);
                DenseVector dvec = generateDenseVector(vec_size);
                DenseMatrix dmat = generateDenseMatrix(rows, vec_size);

                std::vector<float> d_cpu = denseGEMV_CPU(dvec, dmat);

#ifdef ENABLE_OPENCL
                if(rows <= 20000){
                    try {
                        GEMVResult d_gpu = denseGEMV_OpenCL(dvec, dmat);
                        double dense_gpu_err = calc_l2_error(d_cpu, d_gpu.output);
                        log_result(log, "DenseGPU", rows, vec_size, 0.0f, d_gpu.time_ms, d_gpu.gflops, d_gpu.end_to_end_ms, dense_gpu_err);
                    } catch (...) {
                        log_result(log, "DenseGPU", rows, vec_size, 0.0f, -1, -1, -1);
                    }
                    try {
                        GEMVResult do_gpu = denseGEMV_OpenBLAS(dvec, dmat);
                        double dense_blas_err = calc_l2_error(d_cpu, do_gpu.output);
                        log_result(log, "DenseBLAS", rows, vec_size, 0.0f, do_gpu.time_ms, do_gpu.gflops, do_gpu.end_to_end_ms, dense_blas_err);
                    } catch (...) {
                        log_result(log, "DenseBLAS", rows, vec_size, 0.0f, -1, -1, -1);
                    }
                }
#endif

#ifdef ENABLE_OPENCL
                // 稀疏CPU基准
                std::vector<float> sp_cpu_ref = innerProductCPU(vec, mat, false);

                try {
                    GEMVResult sp_gpu = innerProductOpenCL(vec, mat);
                    double sparse_gpu_err = calc_l2_error(sp_cpu_ref, sp_gpu.output);
                    log_result(log, "SparseGPU", rows, vec_size, sparsity, sp_gpu.time_ms, sp_gpu.gflops, sp_gpu.end_to_end_ms, sparse_gpu_err);
                } catch (...) {
                    log_result(log, "SparseGPU", rows, vec_size, sparsity, -1, -1, -1);
                }

                if(rows <= 20000){
                    try {
                        GEMVResult sp_cpu = sparseGEMV_OpenBLAS(vec, mat);
                        double sparse_blas_err = calc_l2_error(sp_cpu_ref, sp_cpu.output);
                        log_result(log, "SparseBLAS", rows, vec_size, sparsity, sp_cpu.time_ms, sp_cpu.gflops, sp_cpu.end_to_end_ms, sparse_blas_err);
                    } catch (...) {
                        log_result(log, "SparseBLAS", rows, vec_size, sparsity, -1, -1, -1);
                    }
                }
#endif
            }
        }
    }

    log.close();
    print_log();
    return 0;
}
