#include <iostream>
#include <vector>
#include <unordered_map>
#include <cmath>
#include "sparse_types.h"

// Function to perform inner product for sparse vectors and matrices
std::vector<float> innerProductCPU(const SparseVector& v, const SparseMatrix& A, bool cosine) {
    const int rows = A.rows;
    const int cols = A.cols;

    // 1) Build dense lookup table v_dense[col] = value  (0 if missing)
    std::vector<float> v_dense(cols, 0.f);
    for (size_t k = 0; k < v.indices.size(); ++k)
        v_dense[v.indices[k]] = v.values[k];

    // pre‑compute ‖v‖ for cosine
    float v_norm = 1.f;
    if (cosine) {
        double s = 0.0;
        for (float x : v.values) s += x * x;
        v_norm = std::sqrt(s);
        if (v_norm == 0.f) return std::vector<float>(rows, 0.f);
    }

    std::vector<float> out(rows);

    // 2) Parallel over rows
#pragma omp parallel for schedule(static)
    for (int r = 0; r < rows; ++r) {
        float dot = 0.f;
        float row_norm_sq = 0.f;

        int start = A.row_ptr[r];
        int end   = A.row_ptr[r + 1];

        // pointer alias hints help autovectoriser
        const int   * __restrict col_p = A.col_idx.data() + start;
        const float * __restrict val_p = A.values .data() + start;

        for (int p = 0; p < end - start; ++p) {
            int   c = col_p[p];
            float a = val_p[p];

            row_norm_sq += a * a;
            dot += a * v_dense[c];
        }

        if (!cosine) {
            out[r] = dot;
        } else {
            float row_norm = std::sqrt(row_norm_sq);
            out[r] = (row_norm > 0.f) ? dot / (row_norm * v_norm) : 0.f;
        }
    }
    return out;
}