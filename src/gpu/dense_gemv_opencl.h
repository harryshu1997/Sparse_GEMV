#pragma once
#include <vector>
#include "sparse_types.h"


GEMVResult denseGEMV_OpenCL(const DenseVector &x, const DenseMatrix &A);
GEMVResult denseGEMV_OpenBLAS(const DenseVector &x, const DenseMatrix &A);
