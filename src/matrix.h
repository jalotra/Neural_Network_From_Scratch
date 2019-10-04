#ifndef MATRIX_H
#define MATRIX_H

// Macros for for loops 
#define for_loop(x, n) for(int x = 0; x < n; ++x)

// Data Structure for matrix representation
typedef struct matrix{
	int rows;
	int cols;
	double **data;            // Data contains all the values from (0,0) to (n-1,m-1)
	int shallow;              // Not Sure
}matrix;


typedef struct LUP
{
	matrix *L;
	matrix *U;
	int *P;
	int n;
} LUP;


// Useful methods

// Creating Matrices and low-level matix functions
matrix make_matrix(int rows, int cols);
matrix copy_matrix(matrix originalMatrix);
void print_matrix(matrix originalMatrix);
void free_matrix(matrix m);

// Neural Netwok spefic methods
matrix make_identity(int rows, int cols);
matrix make_identity_homography();       // Creates a identity matrix 
matrix make_translation_homography(float dx, float dy);          // Creates a translation matrix
double mag_matrix(matrix originalMatrix);             // Returns the sqrt of sum of all the values
double *sle_solve(matrix A, double *b);

matrix matrix_mult_matrix(matrix A, matrix B);        // Multiplies two matrices
matrix matrix_elmult_matrix(matrix a, matrix b);		// Element wise multiplication of two matrices                          // Prints matrix
double **n_principal_components(matrix m, int n);
matrix solve_system(matrix M, matrix b);
matrix matrix_invert(matrix m);
matrix random_matrix(int rows, int cols, double s);
matrix transpose_matrix(matrix m);
matrix axpy_matrix(double a, matrix x, matrix y);
matrix augment_matrix(matrix m);


// test matrix 
void test_matrix();
#endif 




