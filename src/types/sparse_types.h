// filepath: /home/monsterharry/Documents/sparse_gemv/sparse-inner-product/src/types/sparse_types.h
#ifndef SPARSE_TYPES_H
#define SPARSE_TYPES_H

#include <vector>

struct SparseVector {
    std::vector<float> values; // Non-zero values of the sparse vector
    std::vector<int> indices;  // Indices of the non-zero values
    int size;                  // Total size of the vector
};

struct SparseMatrix {
    std::vector<float> values; // Non-zero values of the sparse matrix
    std::vector<int> col_idx;  // Column indices of the non-zero values
    std::vector<int> row_ptr;   // Row pointers for the sparse matrix
    int rows;                  // Number of rows in the matrix
    int cols;                  // Number of columns in the matrix
};

struct DenseVector {
    std::vector<float> values;
    int size;
};

struct DenseMatrix {
    std::vector<float> values; // 按行主序存储
    int rows;
    int cols;
};

struct GEMVResult {
    std::vector<float> output;
    double time_ms;
    double gflops;
    double end_to_end_ms;
};


#endif // SPARSE_TYPES_H