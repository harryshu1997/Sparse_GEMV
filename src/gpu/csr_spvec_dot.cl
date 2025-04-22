__kernel void csr_spvec_dot(
        __global const int   *row_ptr,
        __global const int   *col_idx,
        __global const float *A_val,
        __global const int   *v_idx,
        __global const float *v_val,
        const  int            nnz_v,
        __global float       *out)          // size = rows
{
    int row = get_global_id(0);
    int a_p  = row_ptr[row];
    int a_end= row_ptr[row+1];

    // two‑pointer walk through the row and the sparse vector
    int v_p  = 0;
    float dot = 0.0f;

    while (a_p < a_end && v_p < nnz_v) {
        int cA = col_idx[a_p];
        int cV = v_idx[v_p];

        if (cA == cV) {
            dot += A_val[a_p] * v_val[v_p];
            ++a_p; ++v_p;
        } else if (cA < cV) {
            ++a_p;
        } else {
            ++v_p;
        }
    }
    out[row] = dot;    
}