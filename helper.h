#ifndef HELPER_H_
#define HELPER_H_

// #define SIZE 32
#define EPS 1e-10
#define MAX_ITER 20000

// #define A(row,col) (A[(row)*SIZE + (col)])
#define b(x) (b[(x)])

float* generateA(const int N, 
                const double p_diag, 
                const double p_nondiag);
/* Generates a symmetric square sparse matrix of dimension N with elements of the diagonal according to probability p_diag and
 * elements off the diagonal according to probability p_nondiag
 */
void get_CSR_format(float **A_o, float **A_p, int **IA_p, int **JA_p, int *NNZ_p, int N, double p_diag, double p_nondiag);

void get_CSR(float **A_o, float **A_p, int **IA_p, int **JA_p, int *NNZ_p, int N, int NNZ);

void get_ILU_decomposition(float **A_p, int M, float **L, float **U);

//&h_ACSR, &h_IA, &h_JA, &h_UCSR, &h_IU, &h_JU, &h_LCSR, &h_IL, &h_JL, &NNZ

void resizeSpMatrixArraysAndCopy(       float **A_temp_p,
                                        int **JA_temp_p,
                                        int *bufferSize_p,
                                        double RESIZE_FACTOR);
/* Called from within generateSquareSpMatrix function when we have run out of room to store new elements
 * in A & JA. Will create new arrays of length RESIZE_FACTOR times the original length (bufferSize) and 
 * copy the elements into the new larger array.
 */
float getRandomFloat(const float min, const float max);
/* Returns a quasi-uniformly distributed random float between min and max
 */

float* generateb(int N);
void printMat(float* A);
void printVec(float* b);
float getMaxDiffSquared(float* a, float* b);
long getMicrotime();

#endif /* HELPER_H_ */
