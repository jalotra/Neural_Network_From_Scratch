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
	new_matrix.data = calloc(new_matrix.rows, sizeof(double *));
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
			printf("%15.7f", originalMatrix.data[i][j]);
		}
	}
}
void free_matrix(matrix m)
{
	if (m.data) {
        int i;
        if (!m.shallow) for(i = 0; i < m.rows; ++i) free(m.data[i]);
        free(m.data);
    }
}

matrix matrix_sub_matrix(matrix a, matrix b)
{
    // Matrix will be possible only if the dimensions match up
    assert(a.rows == b.rows);
    assert(a.cols == b.cols);

    matrix m = make_matrix(a.rows, a.cols);
    for_loop(i, a.rows)
    {
        int j ;
        for_loop(j, a.cols)
        {
            m.data[i][j] = a.data[i][j] - b.data[i][j];
        }
    }

    return m;
}
matrix matrix_add_matrix(matrix a, matrix b)
{
    // Matrix will be possible only if the dimensions match up
    assert(a.rows == b.rows);
    assert(b.cols == a.cols);

    matrix m = make_matrix(a.rows, a.cols);
    for_loop(i, a.rows)
    {
        int j ;
        for_loop(j, a.cols)
        {
            m.data[i][j] = a.data[i][j] + b.data[i][j];
        }
    }

    return m;
}
// Important methods

// Creates a identity matrix of size(rows, cols)
matrix make_identity(int rows, int cols)
{
	matrix m = make_matrix(rows, cols);
	for(int i = 0; i < rows && i < cols; i++)
	{
		m.data[i][i] = 1;
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
			sum += originalMatrix.data[i][j];
		}
	}

	return sqrt(sum);
}

double *sle_solve(matrix A, double *b){
	// Have to implement it
	return b;
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

double **n_principal_components(matrix m, int n)
{
	// Have to implement it
	return m.data;
}
matrix matrix_mult_scalar(double b, matrix A){
    matrix m = make_matrix(A.rows, A.cols);

    for_loop(i, m.rows)
    {
        int j ;
        for_loop(j, m.cols)
        {
            m.data[i][j] = b*A.data[i][j];
        }
    }


    return m;
}


matrix solve_system(matrix M, matrix b)
{
	matrix none = {0};
    matrix Mt = transpose_matrix(M);
    matrix MtM = matrix_mult_matrix(Mt, M);
    matrix MtMinv = matrix_invert(MtM);
    if(!MtMinv.data) return none;
    matrix Mdag = matrix_mult_matrix(MtMinv, Mt);
    matrix a = matrix_mult_matrix(Mdag, b);
    free_matrix(Mt); free_matrix(MtM); free_matrix(MtMinv); free_matrix(Mdag);
    return a;
}

// SOME methods That I couldn't understand
double *LUP_solve(matrix L, matrix U, int *p, double *b)
{
    int i, j;
    double *c = calloc(L.rows, sizeof (double));
    for(i = 0; i < L.rows; ++i){
        int pi = p[i];
        c[i] = b[pi];
        for(j = 0; j < i; ++ j){
            c[i] -= L.data[i][j]*c[j];
        }
    }
    for(i = U.rows-1; i >= 0; --i){
        for(j = i+1; j < U.cols; ++j){
            c[i] -= U.data[i][j]*c[j];
        }
        c[i] /= U.data[i][i];
    }
    return c;
}

matrix matrix_invert(matrix m)
{
    //print_matrix(m);
    matrix none = {0};
    if(m.rows != m.cols){
        fprintf(stderr, "Matrix not square\n");
        return none;
    }
    matrix c = augment_matrix(m);
    //print_matrix(c);

    int i, j, k;
    for(k = 0; k < c.rows; ++k){
        double p = 0.;
        int index = -1;
        for(i = k; i < c.rows; ++i){
            double val = fabs(c.data[i][k]);
            if(val > p){
                p = val;
                index = i;
            }
        }
        if(index == -1){
            fprintf(stderr, "Can't do it, sorry!\n");
            free_matrix(c);
            return none;
        }

        double *swap = c.data[index];
        c.data[index] = c.data[k];
        c.data[k] = swap;

        double val = c.data[k][k];
        c.data[k][k] = 1;
        for(j = k+1; j < c.cols; ++j){
            c.data[k][j] /= val;
        }
        for(i = k+1; i < c.rows; ++i){
            double s = -c.data[i][k];
            c.data[i][k] = 0;
            for(j = k+1; j < c.cols; ++j){
                c.data[i][j] +=  s*c.data[k][j];
            }
        }
    }
    for(k = c.rows-1; k > 0; --k){
        for(i = 0; i < k; ++i){
            double s = -c.data[i][k];
            c.data[i][k] = 0;
            for(j = k+1; j < c.cols; ++j){
                c.data[i][j] += s*c.data[k][j];
            }
        }
    }
    //print_matrix(c);
    matrix inv = make_matrix(m.rows, m.cols);
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            inv.data[i][j] = c.data[i][j+m.cols];
        }
    }
    free_matrix(c);
    //print_matrix(inv);
    return inv;
}

int* in_place_LUP(matrix m)
{
    int *pivot = calloc(m.rows, sizeof(int));
    if(m.rows != m.cols){
        fprintf(stderr, "Matrix not square\n");
        return 0;
    }

    int i, j, k;
    for(k = 0; k < m.rows; ++k) pivot[k] = k;
    for(k = 0; k < m.rows; ++k){
        double p = 0.;
        int index = -1;
        for(i = k; i < m.rows; ++i){
            double val = fabs(m.data[i][k]);
            if(val > p){
                p = val;
                index = i;
            }
        }
        if(index == -1){
            fprintf(stderr, "Matrix is singular\n");
            return 0;
        }

        int swapi = pivot[k];
        pivot[k] = pivot[index];
        pivot[index] = swapi;

        double *swap = m.data[index];
        m.data[index] = m.data[k];
        m.data[k] = swap;

        for(i = k+1; i < m.rows; ++i){
            m.data[i][k] = m.data[i][k]/m.data[k][k];
            for(j = k+1; j < m.cols; ++j){
                m.data[i][j] -= m.data[i][k] * m.data[k][j];
            }
        }
    }
    return pivot;
}

matrix random_matrix(int rows, int cols, double s)
{
    matrix m = make_matrix(rows, cols);
    int i, j;
    for(i = 0; i < rows; ++i){
        for(j = 0; j < cols; ++j){
            m.data[i][j] = 2*s*(rand()%1000/1000.0) - s;    
        }
    }
    return m;
}
matrix transpose_matrix(matrix m)
{
    matrix t;
    t.rows = m.cols;
    t.cols = m.rows;
    t.data = calloc(t.rows, sizeof(double *));
    t.shallow = 0;
    int i, j;
    for(i = 0; i < t.rows; ++i){
        t.data[i] = calloc(t.cols, sizeof(double));
        for(j = 0; j < t.cols; ++j){
            t.data[i][j] = m.data[j][i];
        }
    }
    return t;
}
matrix augment_matrix(matrix m)
{
    int i,j;
    matrix c = make_matrix(m.rows, m.cols*2);
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            c.data[i][j] = m.data[i][j];
        }
    }
    for(j = 0; j < m.rows; ++j){
        c.data[j][j+m.cols] = 1;
    }
    return c;
}
matrix axpy_matrix(double a, matrix x, matrix y)
{
    assert(x.cols == y.cols);
    assert(x.rows == y.rows);
    int i, j;
    matrix p = make_matrix(x.rows, x.cols);
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            p.data[i][j] = a*x.data[i][j] + y.data[i][j];
        }
    }
    return p;
}

// Test Matrix
void test_matrix()
{
	for_loop(i,100)
	{
		int s = rand()%4 + 3;
        matrix m = random_matrix(s, s, 10);
        matrix inv = matrix_invert(m);
        matrix res = matrix_mult_matrix(m, inv);
        print_matrix(res);
	}
}