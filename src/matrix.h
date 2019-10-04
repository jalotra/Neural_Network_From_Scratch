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



// Useful methods

// Creating Matrices and low-level matix functions
matrix make_matrix(int rows, int cols);
matrix copy_matrix(matrix originalMatrix);
void print_matrix(matrix originalMatrix);
void free_matrix(matrix m);
matrix matrix_sub_matrix(matrix a, matrix b);
matrix matrix_add_matrix(matrix a, matrix b);


matrix scale_matrix(double s, matrix A);
matrix matrix_mult_matrix(matrix A, matrix B);        // Multiplies two matrices
matrix matrix_elmult_matrix(matrix a, matrix b);		// Element wise multiplication of two matrices                          // Prints matrix
matrix random_matrix(int rows, int cols, double s);
matrix transpose_matrix(matrix m);
matrix axpy_matrix(double a, matrix x, matrix y);


// test matrix 
void test_matrix();
#endif 




