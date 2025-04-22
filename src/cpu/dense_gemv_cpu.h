#pragma once

#include "sparse_types.h"
#include <vector>
#include <algorithm>

std::vector<float> denseGEMV_CPU(const DenseVector& v, const DenseMatrix& A);
