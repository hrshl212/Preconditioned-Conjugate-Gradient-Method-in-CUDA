// Sequential version of CG
// Author: Harshal Raut

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>
#include "helper.h"

#define TOL 0.005


/*
 * Computes a (square) matrix vector product
 * Input: pointer to 1D-array-stored matrix (row major), 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
void matVec(float* A, float* b, float* out, int M) {
	int i, j;
	for (j = 0; j < M; j++) {
		out[j] = 0;
		for (i = 0; i < M; i++) {
			out[j] += A[j*M+i] * b(i);
		}
	}
}

/*
 * Computes the scalar product of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer 1D-array-stored vector
 * Output: float scalar product
 */
float vecVec(float* vec1, float* vec2, int M) {
	int i;
	float product = 0;
	for (i = 0; i < M; i++) {
		product += vec1[i] * vec2[i];
	}
	return product;
}

/*
 * Computes the sum of 2 vectors
 * Input: pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * Stores the sum in memory at the location of the pointer out
 */
void vecPlusVec(float* vec1, float* vec2, float* out, int M) {
	int i;
	for (i = 0; i < M; i++) {
		out[i] = vec1[i] + vec2[i];
	}
}

/*
 * Computes a scalar vector product
 * Input: scalar, pointer to 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
void scalarVec(float alpha, float* vec2, float* out, int M) {
	int i;
	for (i = 0; i < M; i++) {
		out[i] = alpha * vec2[i];
	}
}

/*
 * Computes a scalar (square) matrix vector product
 * Input: scalar, pointer to 1D-array-stored matrix (row major), pointer to 1D-array-stored vector
 * Stores the product in memory at the location of the pointer out
 */
void scalarMatVec(float alpha, float* A, float* b, float* out, int M) {
	int i, j;
	for (j = 0; j < M; j++) {
		out[j] = 0;
		for (i = 0; i < M; i++) {
			out[j] += alpha * A[j* M + i] * b(i);
		}
	}
}

/*
 * Computes the 2-norm of a vector
 * Input: pointer to 1D-array-stored vector
 * Output: value of the norm of the vector
 */
float norm2d(float* a, int M) {
	return sqrt(vecVec(a, a, M));
}


/*
 * Checks if 2 vectors or matrices are equal up to some precision
 * Input: 2 1D-array-stored vector or 2 1D-array-stored matrices
 * Output: true if same (up to some precision) else false
 */
int moreOrLessEqual(float* a, float* b, int M) {
	int i;
	for (i = 0; i < M; i++) {
		if (fabs(a[i] - b[i]) > TOL) {
			return 0;
		}
	}
	return 1;
}

/*
 * Solve the system Ax=b using the CG method
 * Input: pointer to 1D-array-stored matrix, pointer to 1D-array-stored vector, pointer to 1D-array-stored vector
 * float* x is used as initial condition and the final output is written there as well
 */
void solveCG_seq(float* A, float* b, float* x, int M) {
	// Initialize temporary variables
	float* p = (float*) calloc(sizeof(float), M);
	float* r = (float*) calloc(sizeof(float), M);
	float* temp = (float*) calloc(sizeof(float), M);
	float beta, alpha, rNormOld = 0.0;
	float rNorm = 1.0;
	int k = 0;

	// Set initial variables
	scalarMatVec(-1.0, A, x, temp, M);
	vecPlusVec(b, temp, r, M);
	scalarVec(1.0, r, p, M);
	rNormOld = vecVec(r, r,M);

	long micro_begin_seq = getMicrotime();
	while ((rNorm > EPS) && (k < 2*MAX_ITER)) {
		// temp = A* p (only compute matrix vector product once)
		matVec(A, p, temp, M);
		// alpha_k = ...
		alpha = rNormOld / vecVec(p, temp,M);
		// r_{k+1} = ...
		scalarVec(-alpha, temp, temp, M);
		vecPlusVec(r, temp, r, M);
		// x_{k+1} = ...
		scalarVec(alpha, p, temp, M);
		vecPlusVec(x, temp, x, M);
		// beta_k = ...
		rNorm = vecVec(r, r, M);
		beta = rNorm / rNormOld;
		// p_{k+1} = ...
		scalarVec(beta, p, temp, M);
		vecPlusVec(r, temp, p, M);
		// set rOld to r
		rNormOld = rNorm;
		k++;
		if(k % 1000==0) printf("Seq iteraction:%d, Errnorm: %f \n", k, rNorm);

	}
	long micro_end_seq = getMicrotime();
	printf("Seq code iterations: %d \n",k);

	printf("Time spent seq per iter [s]: %e\n", (float) ((micro_end_seq - micro_begin_seq)/k) / 1e6);
	// free temporary memory
	free(p);
	free(r);
	free(temp);
}
