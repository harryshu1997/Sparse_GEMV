#pragma once
#include "sparse_types.h"
#include <vector>

std::vector<float> innerProductCPU(const SparseVector& vec, const SparseMatrix& mat, bool cosine);