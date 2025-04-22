#include "sparse_types.h"
#include <vector>
#include <algorithm>

std::vector<float> denseGEMV_CPU(const DenseVector& v, const DenseMatrix& A) {
    constexpr int TILE = 64;
    constexpr int UNROLL = 4;
    std::vector<float> out(A.rows, 0.f);

    for (int r = 0; r < A.rows; ++r) {
        float sum = 0.f;
        int c = 0;
        // Tiling
        for (; c + TILE <= A.cols; c += TILE) {
            float tile_sum = 0.f;
            // Unrolling within tile
            int t = 0;
            for (; t + UNROLL <= TILE; t += UNROLL) {
                int base = r * A.cols + c + t;
                tile_sum += A.values[base + 0] * v.values[c + t + 0];
                tile_sum += A.values[base + 1] * v.values[c + t + 1];
                tile_sum += A.values[base + 2] * v.values[c + t + 2];
                tile_sum += A.values[base + 3] * v.values[c + t + 3];
            }
            // Handle remaining within tile
            for (; t < TILE; ++t) {
                int base = r * A.cols + c + t;
                tile_sum += A.values[base] * v.values[c + t];
            }
            sum += tile_sum;
        }
        // Handle remaining
        for (; c < A.cols; ++c) {
            sum += A.values[r * A.cols + c] * v.values[c];
        }
        out[r] = sum;
    }
    return out;
}