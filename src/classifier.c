// This file contains all the code that a Cnn uses namely Backward Prop, Forward Prop, Activation Functions etc.

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "image.h"
#include "matrix.h"
#include <assert.h>


// Runs activation function on every element of input matrix
// It modifies the matrix in place
// matrix m : Input to activation function
// ACTIVATION a : Activation FUnction to run
void activate_matrix(matrix m, ACTIVATION a)
{
	for_loop(i, m.rows)
	{
		double sum = 0; 
		for_loop(j, m.cols)
		{
			double data = m.data[i][j];
			if(a == LINEAR)
			{
				const double linear_constant = 0.3;
				m.data[i][j] = data*linear_constant;
			}
			if(a == LOGISTIC)
			
				m.data[i][j] = 1/(1 + exp(-data)); 	
			}
			if (a == RELU)
			{
				if (data >= 0)
				{
					m.data[i][j] = data; 
				}
				else 
				{
					m.data[i][j] = 0
				}
			}
			if (a == LRELU)
			{
				double alpha = 0.1;
				if (data >= 0 )
				{
					m.data[i][j] = data;
				}
				else
				{
					m.data[i][j] = alpha*data;
				}
			}
			if (a == SOFTMAX)
			{
				m.data[i][j] = 	exp(data);
				sum += m.data[i][j]; 
			}
		if(a == SOFTMAX)
		{
			for_loop(j, m.cols)
			{
				m.data[i][j] /= sum; 
			}
		}
	}
}


// Calculates the gradient of an activation function and multiplies it into
// the delta for a layer
// matrix m: an activated layer output
// ACTIVATION a: activation function for a layer
// matrix d: delta before activation gradient
void gradient_matrix(matrix m, ACTIVATION a, matrix d)
{
    int i, j;
    for(i = 0; i < m.rows; ++i){
        for(j = 0; j < m.cols; ++j){
            double x = m.data[i][j];
            if(a == LINEAR)
            {
            	d.data[i][j] *= 1; 
            }
            if (a == SOFTMAX)
            {
            	// We use the softmax as the ouput with cross-entropy loss function
            	// Thus gradient becomes one
            	// Check this link : https://math.stackexchange.com/questions/945871/derivative-of-softmax-loss-function?utm_medium=organic&utm_source=google_rich_qa&utm_campaign=google_rich_qa
            	d.data[i][j] *= 1; 
            }
            if (a == LOGISTIC)
            {
            	d.data[i][j] *= x*(1-x); 
            }
            if (a == RELU)
            {
            	if (x >= 0)
            	{
            		d.data[i][j] *= 1;
            	}
            	else
            	{
            		d.data[i][j] = 0;
            	}
            }
            if (a == LRELU)
            {
            	double alpha = 0.1;
            	if (x >= 0)
            	{
            		d.data[i][j] *= 1
            	}
            	else
            	{
            		d.data[i][j] *= alpha;
            	}
            }
        }
    }
}

// Forward propagate information through a layer
// layer *l: pointer to the layer
// matrix in: input to layer
// returns: matrix that is output of the layer
matrix forward_layer(layer *l, matrix in)
{

    l->in = in;  // Save the input for backpropagation


    // TODO: fix this! multiply input by weights and apply activation function.
    matrix out = make_matrix(in.rows, l->w.cols);


    free_matrix(l->out);// free the old output
    l->out = out;       // Save the current output for gradient calculation
    return out;
}

