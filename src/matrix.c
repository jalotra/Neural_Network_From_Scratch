// This file contains all the matrix operations that a neural network requires


#include "matrix.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>
#include <math.h>


// Implementing various methods

// Creates a matrix of size (row*cols)
matrix make_matrix(int rows, int cols)
{
	matrix new_matrix;
	new_matrix.rows = rows;
	new_matrix.cols = cols;
	new_matrix.shallow = 0;

	// Now I have to create a 2D matrix 
	// Using calloc : calloc stands for continuos allocation
	// Calloc also initializes all bits to 0
	// Calloc return a ptr to the first memory location
	new_matrix.data = callloc(new_matrix.rows, sizeof(double *));
	for(int i = 0; i < new_matrix.rows; i++ )
	{
		new_matrix.data[i] = calloc(new_matrix.cols, sizeof(double *));
	}

	// Finally return new_matrix

	return new_matrix; 

}

matrix copy_matrix(matrix originalMatrix)
{
	// Take use of make_matrix funxtion
	matrix m = make_matrix(originalMatrix.rows, originalMatrix.cols);

	for(int i = 0; i < m.rows; i++)
	{
		for(int j = 0; j < m.cols; j++)
		{
			m.data[i][j] = originalMatrix.data[i][j];
		}
	}

	return originalMatrix;
}
void print_matrix(matrix originalMatrix)
{
	for(int i = 0; i < originalMatrix.rows; i++ )
	{
		for(int j = 0; j < originalMatrix.cols; j++)
		{
			printf("%15.7f", m.data[i][j]);
		}
	}
}

// Important methods

// Creates a identity matrix of size(rows, cols)
matrix make_identity(int rows, int cols)
{
	matrix m = make_matrix(rows, cols);
	for(int i = 0; i < rows && i < cols; i++)
	{
		m.data[]i[i] = 1;
	}

	return m;

}

// Creates a identity matrix of size (3*3)
matrix make_identity_homography()
{
	matrix m = make_identity(3, 3);

	return m;
}

// Refer to this wiki https://en.wikipedia.org/wiki/Transformation_matrix
// Scales first two columns of a matrix if mutiplied by
matrix make_translation_homography(float dx, float dy)
{
	matrix identity = make_identity_homography();
	identity.data[0][2] = dx;
	identity.data[1][2] = dy;


	return identity;
}

// Returns the sqrt of the sum of all the values of a matrix
// In other words norm
double mag_matrix(matrix originalMatrix)
{	
	double sum = 0;
	for_loop(i, originalMatrix.rows)
	{
		for_loop(j, originalMatrix.cols)
		{	
			sum += originalMatrix[i][j];
		}
	}

	return sqrt(sum);
}

double *sle_solve(matrix A, double *b){
	// Have to implement it
}

// Multiply matrix a with  matrix b
matrix matrix_mult_matrix(matrix A, matrix B)
{
	matrix new_matrix = make_matrix(A.rows, B.cols);
 	assert(A.cols == B.rows);
	for_loop(i, new_matrix.rows){
        for_loop(j, new_matrix.cols){
            for_loop(k, A.cols){
                new_matrix.data[i][j] += A.data[i][k]*B.data[k][j];
            }
        }
    }

    return new_matrix;
}

// ELement wise multiplication
matrix matrix_elmult_matrix(matrix a, matrix b)
{
	matrix new_matrix = make_matrix(a.rows, a.cols);
	assert(a.rows == b.rows);
	assert(a.cols == b.cols);

	for_loop(i, a.rows)
	{
		for_loop(j, a.cols)
		{
			new_matrix.data[i][j] = a.data[i][j] + b.data[i][j];
		}

	}

	return new_matrix;
}	

