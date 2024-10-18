/*
 * Implementation of Incomplete-LU preconditioned conjugate-gradient for symmetric PSD systems using CUDA 12.0.
 * The Matrix Vector multiplication has been implemented in CSR format with one warp taking care of one row of matrix
 * The sparse lower and upper triangular solver  have been implemented using cuSPARSE library.
 * 
 * Considering the ILU factorization is deprecated in the new version of cuSPARSE, I am calculating it manually on the host side.
 * 
 * Author: Harshal Raut
 */
// includes, system
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <iostream>
#include <time.h>
#include <assert.h>
#include <cuda_runtime.h>
#include <cusparse.h>	//cusparse library
#include <cublas_v2.h>	//cublas library
#include <sys/time.h>

extern "C" {
#include "helper.h"
#include "sequential.h"
}

// vecVec
#define BLOCK_DIM_VEC 32

//matVec
#define NB_ELEM_MAT 32
#define BLOCK_SIZE_MAT 32

/*
 * --Naive implementation--
 * Computes a (square) matrix vector product
 * Input: pointer to 1D-array-stored matrix, 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
__global__ void matVec(float* A, float* b, float* out, int* SIZE) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < *SIZE) {
		float tmp = 0;
		for (int i = 0; i < *SIZE; i++) {
			tmp += b[i] * A[*SIZE * index_x + i];
		}
		out[index_x] = tmp;
	}
}

/*
 * --More efficient implementation--
 * Computes a (square) symmetric matrix vector product
 * Input: pointer to 1D-array-stored matrix, 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
__global__ void matVec2(float* ACSR, int* IA, int* JA, int* SIZE, float* b, float* out) {


	__shared__ volatile float t_sum[BLOCK_SIZE_MAT]; // thread sum
	int j;

	int t_id =  blockDim.x * blockIdx.x + threadIdx.x; // thread id
	int warpsPerBlock = blockDim.x / 32 ;
	int row = threadIdx.x / 32 + blockIdx.x* warpsPerBlock; // one warp per row
	int t_warp = threadIdx.x & (32-1); // thread number within a given warp

	// don't compute for a row value greater than the total in our matrix!
	if (row < *SIZE){

		// compute running sum per thread in warp
		float dotProduct =0;
		// t_sum[threadIdx.x] = 0;

		for (j = IA[row] + t_warp; j < IA[row+1]; j += 32)
		{
			dotProduct += ACSR[j] * b[JA[j]];
		}

		t_sum[threadIdx.x] = dotProduct;

		__syncthreads();
		// Parallel reduction of result in shared memory for one warp
		if (t_warp < 16) t_sum[threadIdx.x] += t_sum[threadIdx.x+16];
		if (t_warp < 8) t_sum[threadIdx.x] += t_sum[threadIdx.x+8];
		if (t_warp < 4) t_sum[threadIdx.x] += t_sum[threadIdx.x+4];
		if (t_warp < 2) t_sum[threadIdx.x] += t_sum[threadIdx.x+2];
		if (t_warp < 1) t_sum[threadIdx.x] += t_sum[threadIdx.x+1];
		// first thread within warp contains desired y[row] result so write it to y
		if (t_warp == 0)
			out[row] = t_sum[threadIdx.x];
	}
	//__syncthreads();
}

/*
 * Computes the sum of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 */
__global__ void vecPlusVec(float* a, float* b, float* out, int* SIZE) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < *SIZE) {
		out[index_x] = b[index_x] + a[index_x];
	}
}

/*
 * Computes the sum of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 * Also 0's the vector b
 */
__global__ void vecPlusVec2(float* a, float* b, float* out, int* SIZE) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < *SIZE) {
		out[index_x] = b[index_x] + a[index_x];
		b[index_x] = 0.0;
	}
}

/*
 * Computes the difference of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 */
__global__ void vecMinVec(float* a, float* b, float* out, int* SIZE) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < *SIZE) {
		out[index_x] = a[index_x] - b[index_x];
	}
}

/*
 * --Naive implementation--
 * Computes the inner product of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
__global__ void vecVec(float* a, float* b, float* out, int* SIZE) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	float tmp = 0.0;
	if (index_x == 0) {
		for (int i = 0; i < *SIZE; i++) {
			tmp += b[i] * a[i];
		}
		*out = tmp;
	}
}

/*
 * --More efficient implementation--
 * Computes the inner product of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
__global__ void vecVec2(float* a, float* b, float* out, int* SIZE) {
	// each block has it's own shared_tmp of size BLOCK_DIM_VEC
	__shared__ volatile float shared_tmp[BLOCK_DIM_VEC];

	// needed for atomicAdd
	if (threadIdx.x + blockDim.x * blockIdx.x == 0) {
		*out = 0.0;
	}


	if (blockIdx.x * blockDim.x + threadIdx.x < *SIZE) {
		shared_tmp[threadIdx.x] = a[blockIdx.x * blockDim.x + threadIdx.x]
				* b[blockIdx.x * blockDim.x + threadIdx.x];
	} else {
		// needed for the reduction
		shared_tmp[threadIdx.x] = 0.0;
	}

	// reduction within block
	for (int i = blockDim.x / 2; i >= 1; i = i / 2) {
		// threads access memory position written by other threads so sync is needed
		__syncthreads();
		if (threadIdx.x < i) {
			shared_tmp[threadIdx.x] += shared_tmp[threadIdx.x + i];
		}
	}
	__syncthreads();
	// atomic add the partial reduction in out
	if (threadIdx.x == 0) {
		atomicAdd(out, shared_tmp[0]);
	}
}

/*
 * Computes the product of a scalar with a vector
 * Input: pointer to scalar, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 */
__global__ void scalarVec(float* scalar, float* a, float* out, int* SIZE) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < *SIZE) {
		out[index_x] = a[index_x] * *scalar;
	}
}

/*
 * Copies the content of vector in to vector out
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 */
__global__ void memCopy(float* in, float* out, int* SIZE) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x < *SIZE) {
		out[index_x] = in[index_x];
	}
}

/*
 * Computes the quotient of 2 scalars
 * Input: pointer to scalar, pointer to scalar
 * Stores the quotient in memory at the location of the pointer out
 */
__global__ void divide(float* num, float* den, float* out) {
	unsigned int index_x = blockIdx.x * blockDim.x + threadIdx.x;
	if (index_x == 0) {
		*out = *num / *den;
	}
}

/*
 * Main CG solver
 * All the given pointers are device pointers, with correct initial values
 */

void solveCG_cuda(float* d_A, float* A_CSR, int* JA, int* IA, float* d_UD, int* d_JU, int* d_IU, float* d_LD, int* d_JL, int* d_IL, float* b, float* x, float* p, float* r, float* z, float* temp,
		float* alpha, float* beta, float* r_norm, float* r_norm_old,
		float* temp_scal, float* h_x, float* h_r_norm, int M, int NNZ, int NNL, int NNU) {

	dim3 vec_block_dim(BLOCK_DIM_VEC); 
	dim3 vec_grid_dim((M + BLOCK_DIM_VEC - 1) / BLOCK_DIM_VEC); 
	dim3 mat_block_dim(BLOCK_SIZE_MAT);
	dim3 mat_grid_dim((M+BLOCK_SIZE_MAT - 1)  / BLOCK_SIZE_MAT * 32); //because each warp is taking care of one row of matrix during Mat-Vec multiplication

	float* numerator;
	cudaMalloc((void **) &numerator, sizeof(float));

	float* denominator;
	cudaMalloc((void **) &denominator, sizeof(float));

	int* SIZE;
	cudaMalloc((void**)&SIZE, sizeof(int));
	cudaMemcpy(SIZE, &M, sizeof(int), cudaMemcpyHostToDevice);

    cusparseHandle_t handle;
    cusparseCreate(&handle);

	cusparseSpMatDescr_t matL, matU;
	cusparseDnVecDescr_t vecB, vecX, vecZ;
	cusparseCreateCsr(&matL, M, M, NNL,
					d_IL, d_JL, d_LD,
					CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
					CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
	cusparseCreateCsr(&matU, M, M, NNU,
					d_IU, d_JU, d_UD,
					CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
					CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);
	cusparseCreateDnVec(&vecB, M, r, CUDA_R_32F);
	cusparseCreateDnVec(&vecX, M, temp, CUDA_R_32F);
	cusparseCreateDnVec(&vecZ, M, z, CUDA_R_32F);

	cusparseSpSVDescr_t spsvDescrL;
	cusparseSpSV_createDescr(&spsvDescrL);

	cusparseSpSVDescr_t spsvDescrU;
	cusparseSpSV_createDescr(&spsvDescrU);

	size_t bufferSize = 0;
	size_t stmp = 0;
	float alpha2 = 1.0f;
	cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_TRANSPOSE,
							&alpha2, matU, vecB, vecX, CUDA_R_32F,
							CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, &stmp);
	if (stmp > bufferSize)	bufferSize = stmp;

	cusparseSpSV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
							&alpha2, matU, vecX, vecZ, CUDA_R_32F,
							CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, &stmp);
	if (stmp > bufferSize)	bufferSize = stmp;
	void* dBuffer;
	cudaMalloc(&dBuffer, bufferSize);

	// Analysis step
	cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_TRANSPOSE,
						&alpha2, matU, vecB, vecX, CUDA_R_32F,
						CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL, dBuffer);
	cusparseSpSV_analysis(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						&alpha2, matU, vecX, vecZ, CUDA_R_32F,
						CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU, dBuffer);

	vecVec2<<<vec_grid_dim, vec_block_dim>>>(r, r, r_norm_old, SIZE);

	int k = 0;
	long micro_begin_gpu = getMicrotime();
	while ((k < MAX_ITER) && (*h_r_norm > EPS)) {


        // preconditioner application: z = U^-1 L^-1 r
        cusparseSpSV_solve(handle, CUSPARSE_OPERATION_TRANSPOSE,
						&alpha2, matU, vecB, vecX, CUDA_R_32F,
						CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrL);
        cusparseSpSV_solve(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
						&alpha2, matU, vecX, vecZ, CUDA_R_32F,
						CUSPARSE_SPSV_ALG_DEFAULT, spsvDescrU);

        if (k == 0)
        {
			memCopy<<<vec_grid_dim, vec_block_dim>>>(z, p, SIZE);
			vecVec2<<<vec_grid_dim, vec_block_dim>>>(r, z, numerator, SIZE);
        }
        else
        {
			// beta_k = ...
			memCopy<<<1, 1>>>(numerator, denominator, SIZE);
			vecVec2<<<vec_grid_dim, vec_block_dim>>>(r, z, numerator, SIZE);
			divide<<<1, 1>>>(numerator, denominator, beta);

			// p_{k+1} = ...
			scalarVec<<<vec_grid_dim, vec_block_dim>>>(beta, p, temp, SIZE);
			vecPlusVec2<<<vec_grid_dim, vec_block_dim>>>(z, temp, p, SIZE);
        }

		matVec2<<< mat_grid_dim, mat_block_dim>>>(A_CSR, IA, JA, SIZE, p, temp);

		// alpha_k = ...
		vecVec2<<<vec_grid_dim, vec_block_dim>>>(p, temp, temp_scal, SIZE); //temp_scalar is denominator for alpha
		divide<<<1, 1>>>(numerator, temp_scal, alpha);

		// r_{k+1} = ...
		scalarVec<<<vec_grid_dim, vec_block_dim>>>(alpha, temp, temp, SIZE);
		vecMinVec<<<vec_grid_dim, vec_block_dim>>>(r, temp, r, SIZE);

		// x_{k+1} = ...
		scalarVec<<<vec_grid_dim, vec_block_dim>>>(alpha, p, temp, SIZE);
		vecPlusVec<<<vec_grid_dim, vec_block_dim>>>(x, temp, x, SIZE);

		vecVec2<<<vec_grid_dim, vec_block_dim>>>(r, r, r_norm, SIZE);

		// set r_norm_old to r_norm
		memCopy<<<1, 1>>>(r_norm, r_norm_old, SIZE);

		// copy to r_norm to CPU (to evaluate stop condition)
		cudaMemcpy(h_r_norm, r_norm, sizeof(float), cudaMemcpyDeviceToHost);
		k++;
		// printing the error and iteration number in CPU
		if(k % 1000==0) printf("iteraction:%d, Errnorm: %f \n", k, *h_r_norm);

	}
	long micro_end_gpu = getMicrotime();
	printf("iterations: %d \n",k);
	printf("norm: %f\n", *h_r_norm);
	printf("Time spent gpu per iter [s]: %e\n", (float) ((micro_end_gpu - micro_begin_gpu)/k) / 1e6);

	cudaFree(dBuffer);
	cusparseDestroy(handle);
    cusparseSpSV_destroyDescr(spsvDescrU);
    cusparseSpSV_destroyDescr(spsvDescrL);
    cusparseDestroySpMat(matL);
    cusparseDestroyDnVec(vecB);
    cusparseDestroyDnVec(vecX);
    cusparseDestroyDnVec(vecZ);

}



////////////////////////////////////////////////////////////////////////////////
// MAIN
////////////////////////////////////////////////////////////////////////////////
int main() {

	int M =32; // size of matrix

	// Parameters for sparce matrix
	double p_diag = 1.0; //probability of element along diagonal
	double p_nondiag = 0.15;

	// Parameters for CSR format for matrix A
	int *h_IA, *h_JA; 
	float *h_ACSR;
	int NNZ; // total non-zero elements

	//Parameters for CSR format for L and U from ilu decomposition
	int *h_IU, *h_IL, *h_JU, *h_JL;
	float *h_UCSR, *h_LCSR;
	int NNL, NNU;

	// allocate host memory
	float* h_A = generateA(M, p_diag, p_nondiag);
	float*h_LD, *h_UD;
	get_ILU_decomposition(&h_A, M, &h_LD, &h_UD);

	get_CSR_format(&h_A, &h_ACSR, &h_IA, &h_JA, &NNZ, M, p_diag, p_nondiag);

	//Get CSR for L and U matrix as well
	get_CSR(&h_LD, &h_LCSR, &h_IL, &h_JL, &NNL, M, NNZ);
	get_CSR(&h_UD, &h_UCSR, &h_IU, &h_JU, &NNU, M, NNZ);

	free(h_LD);
	free(h_UD);
	float* h_b = generateb(M);
	float* h_x = (float *) calloc(M, sizeof(float));
	float* h_r_norm = (float *) malloc(sizeof(float));
	*h_r_norm = 1.0;

	// allocate device memory
	float *d_A;
	float* d_ACSR, *d_LD, *d_UD;
	int* d_IA, *d_JA, *d_IL, *d_JL, *d_IU, *d_JU;
	float* d_b;
	float* d_x;
	float* d_p;
	float* d_r;
	float* d_z;
	float* d_temp;

	std::cout << "SIZE: " <<M << std::endl; //HR
	cudaMalloc((void **) &d_A, M * M * sizeof(float));
	cudaMalloc((void **) &d_ACSR, NNZ * sizeof(float));
	cudaMalloc((void **) &d_JA, NNZ * sizeof(int));
	cudaMalloc((void **) &d_IA, (M+1) * sizeof(int));
	cudaMalloc((void **) &d_LD, NNL * sizeof(float));
	cudaMalloc((void **) &d_JL, NNL * sizeof(int));
	cudaMalloc((void **) &d_IL, (M+1) * sizeof(int));
	cudaMalloc((void **) &d_UD, NNU * sizeof(float));
	cudaMalloc((void **) &d_JU, NNU * sizeof(int));
	cudaMalloc((void **) &d_IU, (M+1) * sizeof(int));

	cudaMalloc((void **) &d_b, M * sizeof(float));
	cudaMalloc((void **) &d_x, M * sizeof(float));
	cudaMalloc((void **) &d_p, M * sizeof(float));
	cudaMalloc((void **) &d_r, M * sizeof(float));
	cudaMalloc((void **) &d_z, M * sizeof(float));
	cudaMalloc((void **) &d_temp, M * sizeof(float));

	// copy host memory to device
	cudaMemcpy(d_A, h_A, M * M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_ACSR, h_ACSR, NNZ * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_JA, h_JA, NNZ * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_IA, h_IA, (M+1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_UD, h_UCSR, NNU * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_JU, h_JU, NNU * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_IU, h_IU, (M+1)* sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_LD, h_LCSR, NNL * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_JL, h_JL, NNL * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_IL, h_IL, (M+1)* sizeof(int), cudaMemcpyHostToDevice);


	cudaMemcpy(d_b, h_b, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_x, h_x, M * sizeof(float), cudaMemcpyHostToDevice);
	// assume x0 = 0
	cudaMemcpy(d_p, h_b, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_r, h_b, M * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_z, h_b, M * sizeof(float), cudaMemcpyHostToDevice);

	// 5 floats needed
	float* d_beta;
	float* d_alpha;
	float* d_r_norm;
	float* d_r_norm_old;
	float* d_temp_scal;
	cudaMalloc((void **) &d_beta, sizeof(float));
	cudaMalloc((void **) &d_alpha, sizeof(float));
	cudaMalloc((void **) &d_r_norm, sizeof(float));
	cudaMalloc((void **) &d_r_norm_old, sizeof(float));
	cudaMalloc((void **) &d_temp_scal, sizeof(float));

	// run the main function
	solveCG_cuda(d_A,d_ACSR, d_JA, d_IA, d_UD, d_JU, d_IU, d_LD,d_JL, d_IL, d_b, d_x, d_p, d_r, d_z, d_temp, d_alpha, d_beta, d_r_norm,
			d_r_norm_old, d_temp_scal, h_x, h_r_norm, M, NNZ, NNL, NNU);

	// allocate memory for the result on host side
	cudaDeviceSynchronize();
	// copy result from device to host
	cudaMemcpy(h_x, d_x, sizeof(float) * M, cudaMemcpyDeviceToHost);

	// compare output with sequential version
	float* h_x_seq = (float *) calloc(M, sizeof(float));
	solveCG_seq(h_A, h_b, h_x_seq, M);

	// for(int i=0; i<M; i++)
	// {
	// 	printf("%2.2f : %2.2f \n",h_x[i], h_x_seq[i]);
	// }
    assert(moreOrLessEqual(h_x, h_x_seq) == 1);

    printf("\nAssertion passed!\n");

	// cleanup memory host
	free(h_A);
	free(h_ACSR);
	free(h_UCSR);
	free(h_LCSR);
	free(h_IA);
	free(h_JA);
	free(h_b);
	free(h_x);
	free(h_r_norm);

	// cleanup memory device
	// cudaFree(d_A);
	cudaFree(d_ACSR);
	cudaFree(d_LD);
	cudaFree(d_UD);
	cudaFree(d_IA);
	cudaFree(d_JA);
	cudaFree(d_IL);
	cudaFree(d_JL);
	cudaFree(d_IU);
	cudaFree(d_JU);
	cudaFree(d_b);
	cudaFree(d_x);
	cudaFree(d_p);
	cudaFree(d_r);
	cudaFree(d_temp);
	cudaFree(d_alpha);
	cudaFree(d_beta);
	cudaFree(d_r_norm);
	cudaFree(d_r_norm_old);
	cudaFree(d_temp_scal);

	return 0;
}
