# Preconditioned-Conjugate-Gradient-Method-in-CUDA
Implementation of Incomplete-LU preconditioned conjugate-gradient for symmetric PSD systems using CUDA 12.0.

The Matrix-Vector multiplication has been implemented in CSR format with one warp taking care of one row of matrix. The sparse lower and upper triangular solver have been implemented using cuSPARSE library. Considering the ILU factorization is deprecated in the new version of cuSPARSE, it has been calculated manually on the host side. 

I wanted to extend this code to preconditioned bicg method and hence wanted to use ILU preconditioner instead of Incomplete-Cholesky. Also I do not make use of the symmetry of the matrix anywhere within the code.
