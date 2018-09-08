/*
 * Author: Tyler Allen
 * USAGE: ./concomp path-to-file
 *
 * Connected component finder. Works only on undirected matrices. Expects Matrix-Market format input file. Also expects
 * edges to have weights, due to sparse matrix library.
 *
 * Sparse matrix library is BeBOP SMC, included but not part of this project. I do not claim any of their code or 
 * library as my own.
 *
 */

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <vector>
#include <iterator>
#include<unordered_set>
#include <assert.h>
#include "safecuda.h"

extern "C" {
#include "bebop/smc/csr_matrix.h"
#include "bebop/smc/sparse_matrix.h"
#include "bebop/smc/sparse_matrix_ops.h"
}

#define MAX_ITER (m)

/* 
 * GPU memory initialization kernel for connected components
 * Args:
 * m - from m x m matrix
 * I - buffer of length m+1 containing CSR-format rows
 * values - buffer of length M + 1 to be overwritten
 *
 * Values contains a unique global ID, which is just the row entry in the matrix.
 */
__global__ void init_cc(const int m, const int* const __restrict__ I, int* __restrict__ values)
{
    const int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < m + 1)
    {
        values[idx] = idx;
    }
}

/*
 *  connected components solving kernel
 *  Args:
 *  m, I, values are same as init_cc
 *  J - The column index array from CSR-format
 *
 *  each node corresponds to an index in the values array. At the beginning, all IDs are unique.
 *  Each iteration of this kernel, 1 thread is assigned to each node. This node replaces its value
 *  with the value of its lowest-value neighbor, including itself. After m calls to this kernel, all connected
 *  nodes within a single component will have the same minimum value. 
 */
__global__ void cudacc(const int m, const int* const __restrict__ I, const int* const __restrict__ J, int* __restrict__ values)
{
    int first_edge = 0;
    int num_edges = 0;
    const int idx = blockDim.x * blockIdx.x + threadIdx.x + 1;
    // in range
    if (idx < m + 1)
    {
        int minimum = values[idx];
        // not 0 edges
        if (I[idx] - I[idx - 1] > 0)
        {
            first_edge = I[idx - 1];
            num_edges = I[idx] - first_edge;
            minimum = values[idx];
            for (int j = first_edge; j < first_edge + num_edges; j++)
            {
                // find neighbor edge, see their true value
                minimum = min(minimum, values[J[j] + 1]);
            }
            // weak race condition. For huge matrices, may require some additional iterations
            // However, (m) iterations is already accounting worst-case behavior for the pathological list
            // network. In practice, this is far more than enough iterations. Also, thanks to
            // warp-synchronicity, the pathological list case likely will not fall into this issue either.
            // Therefore, this is a non-issue.
            values[idx] = minimum;
        }
    }
}

int main(int argc, char* argv[])
{
    if (argc != 2)
    {
        printf("Recv'd argc: %d\n", argc);
        exit(1);
    }
    char* filename = argv[1];
    // load mm
    struct sparse_matrix_t* mat = load_sparse_matrix(sparse_matrix_file_format_t::MATRIX_MARKET, filename);

    if (mat == nullptr)
    {
        printf("mat pointer: %p\nmaybe bad file name?\n", mat);
        exit(1);
    }
    // convert mm to csr
    int err = sparse_matrix_convert (mat, sparse_matrix_storage_format_t::CSR);
    
    // extract csr
    struct csr_matrix_t* cmat = (struct csr_matrix_t*) mat->repr;
    err = valid_csr_matrix_p (cmat);
    if (err == 0)
    {
        printf("not valid matrix before expansion\n");
        exit(1);
    }
    // expand to symmetry, assumed symmetric
    err = csr_matrix_expand_symmetric_storage (cmat);
    if (err != 0)
    {
        printf("some error expanding symmetric storage: %d\n", err);
        exit(1);
    }
    // makes sure matrix is sane
    err = valid_csr_matrix_p (cmat);
    if (err == 0)
    {
        printf("not valid matrix after expansion\n");
        exit(1);
    }
    int nnz = cmat->nnz;
    int m = cmat->m;
    int* I;
    int* values;
    printf("m, nnz: %d, %d\n", m, nnz);
    // map of real row indexes to sparse row indexes
    int* J;

    // alloc GPU memory for I, J, values defined in cuda section
    cudaMalloc(&I, sizeof(int) * (m + 1));
    CHECK_CUDA_ERROR();
    cudaMalloc(&J, sizeof(int) * nnz);
    CHECK_CUDA_ERROR();
    cudaMalloc(&values, sizeof(int) * (m + 1));
    CHECK_CUDA_ERROR();

    // lets you know if you alloc too much memory : )
    printf("Sparse Structure requires %lu MB GPU memory\n", (sizeof(int) * nnz + 2 * sizeof(int) * (m + 1)) / 1000000);
    
    // copy sparse matrix into gpu memory
    cudaMemcpy(I, cmat->rowptr, sizeof(int) * (m + 1), cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();
    cudaMemcpy(J, cmat->colidx, sizeof(int) * nnz, cudaMemcpyHostToDevice);
    CHECK_CUDA_ERROR();

    // init connected components algorithm
    init_cc<<<(m + 1) / 128 + 1, 128>>>(m, I, values);
    // repeat defined number of iterations
    for (int i = 0; i < MAX_ITER; i++)
    {
        cudacc<<<(m+1) / 128 + 1, 128>>>(cmat->m, I, J, values);
    }
    // wait for results
    cudaDeviceSynchronize();
    CHECK_CUDA_ERROR();
    
    // copy result back
    int h_values[m + 1];
    cudaMemcpy(h_values, values, sizeof(int) * (m + 1), cudaMemcpyDeviceToHost);

    // count number of unique components via one list iteration
    std::unordered_set<int> uniques;
    for (int i = 1; i < m + 1; i++)
    {
        uniques.insert(h_values[i]);
    }
    printf("# of connected components: %lu\n", uniques.size());
    
    // clean house
    cudaFree(I);
    cudaFree(J);
    cudaFree(values);
    destroy_sparse_matrix(mat);
    return 0;
}

