#pragma once
#include <vector>
#include "sparse_types.h"


GEMVResult innerProductOpenCL(const SparseVector& v, const SparseMatrix& A);
GEMVResult sparseGEMV_OpenBLAS(const SparseVector& v, const SparseMatrix& A);
