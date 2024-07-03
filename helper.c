#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <sys/time.h>
//#include <iostream>
#include "helper.h"

/*
 * Generates a sparse PSD symmetric N x N matrix.
 */
float* generateA(const int N, const double p_diag, const double p_nondiag) {

	float* A_total = (float *)malloc(sizeof(float)*N*N);

	double randProb; float randNum;

	int i,j;
	for (i = 0; i < N; ++i)
	{
		for (j = 0; j <= i; ++j) // To make matrix symmetric, filling only the lower half
		{
			randProb = ((double)rand())/RAND_MAX;
			if (i == j) // on diagonal - use p_diag
			{
				if (randProb < p_diag) // insert non-zero element
				{
					randNum = getRandomFloat(0, 1);
					A_total[i*N+j] = randNum;
				}
			}
			else
			{
				if (randProb < p_nondiag) 
				{
					randNum = getRandomFloat(0, 1);
					A_total[i*N+j] = randNum;
					A_total[j*N+i] = randNum; //Symmetry creation
				}									
			}
			// printf("%2.5f ",A_total[i*N+j]);
			
		}
		// printf("\n");
	}

	return A_total;
}

float getRandomFloat(const float min, const float max)
{
        return ((((float)rand())/RAND_MAX)*(max-min)+min);
}


void get_CSR_format(float **A_o, float **A_p, int **IA_p, int **JA_p, int *NNZ_p, int N, double p_diag, double p_nondiag)
{
	// estimate size of A, JA arrays because they vary between realization
	// but are same for a given realization
	int estSize = N*p_diag + N*(N-1)*p_nondiag;
	
	// allocate IA because size is fixed (size of IA = N + 1)
	*IA_p = (int *)malloc(sizeof(int)*(N+1));
	
	// define buffer space for undetermined arrays
	int bufferSize = (int)ceil(1.33*estSize);
	
	// allocate buffer*estSize for A & JA so we can probably fit everything in those
	float* A_temp = (float *)malloc(sizeof(float)*bufferSize);
	int* JA_temp = (int *)malloc(sizeof(float)*bufferSize);
	
	// Setup inital conditions for sparse matrix
	*NNZ_p = 0; (*IA_p)[0] = 0;

	int i,j;
	for (i = 0; i < N; ++i)
	{
		(*IA_p)[i+1] = (*IA_p)[i];
		
		// printf("%d: %d\n",i, (*IA_p)[i]);
		for (j = 0; j < N; ++j)
		{			
			if((*A_o)[i*N+j] != 0.0)
			{
				if((*NNZ_p) == bufferSize) // Placing element will exceed allowed buffer!
				{
					resizeSpMatrixArraysAndCopy(&A_temp, &JA_temp, &bufferSize, 1.33); // resize arrays so we can insert element!
				}				

				A_temp[(*NNZ_p)] = (*A_o)[i*N+j];
				JA_temp[(*NNZ_p)] = j;
				(*IA_p)[i+1]++;
				(*NNZ_p)++;
			}
		}
	}

	// By this point we have not exceeded memory limit so lets create
	// actual A and IA array now that we have determined the size
	*A_p = (float *)malloc(sizeof(float)*(*NNZ_p));
	*JA_p = (int *)malloc(sizeof(float)*(*NNZ_p));
	
	// Add elements from temp arrays to actual arrays
	for (i = 0; i < (*NNZ_p); ++i)
	{
		(*A_p)[i] = A_temp[i];
		// printf("%2.3f: %d \n",A_temp[i], JA_temp[i]); 
		(*JA_p)[i] = JA_temp[i];
	}
	
	// free no longer used temp arrays
	free(A_temp); A_temp = NULL;
	free(JA_temp); JA_temp = NULL;
	
	return;
}

void get_CSR(float **A_o, float **A_p, int **IA_p, int **JA_p, int *NNZ_p, int N, int NNZ)
{
	// estimate size of A, JA arrays because they vary between realization
	// but are same for a given realization
	int estSize = NNZ;
	
	// allocate IA because size is fixed (size of IA = N + 1)
	*IA_p = (int *)malloc(sizeof(int)*(N+1));
	
	// define buffer space for undetermined arrays
	int bufferSize = (int)ceil(2.5*estSize);
	
	// allocate buffer*estSize for A & JA so we can probably fit everything in those
	float* A_temp = (float *)malloc(sizeof(float)*bufferSize);
	int* JA_temp = (int *)malloc(sizeof(float)*bufferSize);
	
	// Setup inital conditions for sparse matrix
	*NNZ_p = 0; (*IA_p)[0] = 0;

	int i,j;
	for (i = 0; i < N; ++i)
	{
		(*IA_p)[i+1] = (*IA_p)[i];
		
	 	// printf("%d: %d\n",i, (*IA_p)[i]);
		for (j = 0; j < N; ++j)
		{			
			if((*A_o)[i*N+j] != 0.0)
			{
				if((*NNZ_p) == bufferSize) // Placing element will exceed allowed buffer!
				{
					resizeSpMatrixArraysAndCopy(&A_temp, &JA_temp, &bufferSize, 1.33); // resize arrays so we can insert element!
				}				

				A_temp[(*NNZ_p)] = (*A_o)[i*N+j];
				JA_temp[(*NNZ_p)] = j;
				(*IA_p)[i+1]++;
				(*NNZ_p)++;
			}
		}
	}

	// By this point we have not exceeded memory limit so lets create
	// actual A and IA array now that we have determined the size
	*A_p = (float *)malloc(sizeof(float)*(*NNZ_p));
	*JA_p = (int *)malloc(sizeof(float)*(*NNZ_p));
	
	// printf("Following is l or u: \n");
	// Add elements from temp arrays to actual arrays
	for (i = 0; i < (*NNZ_p); ++i)
	{
		(*A_p)[i] = A_temp[i];
	 	// printf("%2.3f: %d \n",A_temp[i], JA_temp[i]); 
		(*JA_p)[i] = JA_temp[i];
	}
	
	// free no longer used temp arrays
	free(A_temp); A_temp = NULL;
	free(JA_temp); JA_temp = NULL;
	
	return;
}


void get_ILU_decomposition(float **A,  int N, float **L, float **U)
{
	*L = (float *)calloc(N*N, sizeof(float));
	*U = (float *)calloc(N*N, sizeof(float));

    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
            if (i == j) {
                (*U)[i*N+j] = (*A)[i*N+j];
                (*L)[i*N+j] = 1.0; // Diagonal elements of L are 1
            } else if (i < j) {
                (*U)[i*N+j] = (*A)[i*N+j];
            } else {
                (*L)[i*N+j] = (*A)[i*N+j];
            }
        }
    }

    // Perform the factorization
    for (int i = 1; i < N; ++i) {
        for (int k = 0; k < i; ++k) {
            // if (inP(i, k, P)) {
                (*L)[i*N+k] /= (*U)[k*N+k];
                for (int j = k + 1; j < i; ++j) {
                    // if (inP(i, j, P)) {
                        (*L)[i*N+j] -= (*L)[i*N+k] * (*U)[k*N+j];
                    // }
                }
                for (int j = i; j < N; ++j) {
                    // if (inP(i, j, P)) {
                        (*U)[i*N+j] -= (*L)[i*N+k] * (*U)[k*N+j];
                    // }
                }
            // }
			// printf("%2.3f ",L[i*N+k]);
        }
		// printf("\n");
    }

	// print the matrices
	// printf("Printing LU matrices: \n");
    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
	// 		printf("%2.5f ",(*L)[i*N+j]);
	// 	}
	// 	printf("\n");
	// }

    // for (int i = 0; i < N; ++i) {
    //     for (int j = 0; j < N; ++j) {
	// 		printf("%2.5f ",(*U)[i*N+j]);
	// 	}
	// 	printf("\n");
	// }

return ;

}

void resizeSpMatrixArraysAndCopy(float **A_temp_p, int **JA_temp_p, int *bufferSize_p, double RESIZE_FACTOR)
{

        printf("Executing resize!!\n");
        if (RESIZE_FACTOR <= 1) // RESIZE_FACTOR should not be less than one!
                RESIZE_FACTOR = 1.33; // if so, set to default value of 1.33

        int oldLength = (*bufferSize_p);
        int newLength = (int)ceil((*bufferSize_p)*RESIZE_FACTOR);
        float *A_temp_new;
        int *JA_temp_new;

        // allocate the new resized memory
        A_temp_new = (float *)malloc(sizeof(float)*newLength);
        JA_temp_new = (int *)malloc(sizeof(int)*newLength);

        // copy old elements into new array
        int i;
        for (i = 0; i < oldLength; ++i)
        {
                A_temp_new[i] = (*A_temp_p)[i];
                JA_temp_new[i] = (*JA_temp_p)[i];
        }

        // free memory from old arrays
        free(*A_temp_p);
        free(*JA_temp_p);

        // update pointers
        *A_temp_p = A_temp_new; A_temp_new = NULL;
        *JA_temp_p = JA_temp_new; A_temp_new = NULL;

        // update bufferSize
        *bufferSize_p = newLength;
}


/*
 * Returns the time in microseconds
 * Taken from https://gist.github.com/sevko/d23646ba07c77c15fde9
 */
long getMicrotime(){
	struct timeval currentTime;
	gettimeofday(&currentTime, NULL);
	return currentTime.tv_sec * (int)1e6 + currentTime.tv_usec;
}

/*
 * Generates a random vector of size SIZE
 */
float* generateb(int N) {
	int i;
	float* b = (float *)malloc(sizeof(float) * N);
	for (i = 0; i < N; i++) {
		b[i] = (float)rand()/RAND_MAX;
	}
	return b;
}

