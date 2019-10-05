# Neural Network from Scratch  
## What is a neural Network?
Wiki says that a neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes.These artificial networks may be used for predictive modeling, adaptive control and applications where they can be trained via a dataset. Self-learning resulting from experience can occur within networks, which can derive conclusions from a complex and seemingly unrelated set of information.

Look at this picture. ![Neural Network](ReadmeData/Neural_Network_Example.png)

## Why this?
I created this file to act as a starter template for students like me that want to implement a neural network and learn from it. Most of the time I wanted to learn neural netwok, I came across a number of different parameters like weight initialization, # of neurons, decay rate, learning rate, batch size etc. And I always thought that there must be something or a course that will tell you all of these because most of the teachers online say that it's all about trial and error and it is about trial and error. But finally I thought of creating a neural network on my own and came across some the best teachers in the world Joseph Redmon aka Pjreddie. He has open-sourced a number of lectures and they are a must to watch if you want to learn computer-vision . This the course link : [Ancient Secrets of Computer Vision](https://pjreddie.com/courses/computer-vision/)

## Implementing neural networks ##

A neural network is made up of a number of layers or fully connected layers. Each layer has following entites:
1. Input neurons
2. Weights corresponding to each neuron
3. Current weight Updates while backpropogating
4. Past weight updates fo use with momentum
5. Activation Function to add non-linearity in model
5. Output neurons


### Activation functions ###

An important part of machine learning, be it linear classifiers or neural networks, is the activation function you use. This has been implemented in `void activate_matrix(matrix m, ACTIVATION a)` and is used to modify `m` to be `f(m)` applied elemetwise all along the output matrix.

The different Activation functions that I have implemented are:
1. LINEAR        // Just a linear activation
2. LOGISTIC      // Not so good due to gradient loss if x becomes large or > 7 
3. RELU          // Rectified-linear-Unit
4. LRELU         // Leaky-Relu
5. SOFTMAX       // Mainly used before the output layer 

### Taking gradients ###

As we are calculating the partial derivatives for the weights of our model we have to remember to factor in the gradient from our activation function at every layer. We will have the derivative of the loss with respect to the output of a layer (after activation) and we need the derivative of the loss with respect to the input (before activation). To do that we take each element in our delta (the partial derivative) and multiply it by the gradient of our activation function.

Normally, to calculate `f'(x)` we would need to remember what the input to the function, `x`, was. However, all of our activation functions have the nice property that we can calculate the gradient `f'(x)` only with the output `f(x)`. This means we have to remember less stuff when running our model.

In my view you must differentiate all the functions yourself and then see if I have implemented it rightly or not. Check this readthedocs for detailed information [Activation Functions Described](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html)


After learning about the gradients check this function `void gradient_matrix(matrix m, ACTIVATION a, matrix d)`
for implementation details.

### Forward propagation ###

Now we can fill in how to forward-propagate information through a layer. First check out our layer struct:

    typedef struct {
        matrix in;              // Saved input to a layer
        matrix w;               // Current weights for a layer
        matrix dw;              // Current weight updates
        matrix v;               // Past weight updates (for use with momentum)
        matrix out;             // Saved output from the layer
        ACTIVATION activation;  // Activation the layer uses
    } layer;

During forward propagation we will do a few things. We'll multiply the input matrix by our weight matrix `w`. We'll apply the activation function `activation`. We'll also save the input and output matrices in the layer so that we can use them during backpropagation. But this is already done for you.

Check `matrix forward_layer(layer *l, matrix in)`.

Using matrix operations we can batch-process data. Each row in the input `in` is a separate data point and each row in the returned matrix is the result of passing that data point through the layer.

### Backward propagation ###

Back Propogation is the most difficult to see through for me but as I implemented all the things became clear.

We have to backpropagate error through our layer. We will be given `dL/dy`, the derivative of the loss with respect to the output of the layer, `y`. The output `y` is given by `y = f(xw)` where `x` is the input, `w` is our weights for that layer, and `f` is the activation function. What we want to calculate is `dL/dw` so we can update our weights and `dL/dx` which is the backpropagated loss to the previous layer. Check  `matrix backward_layer(layer *l, matrix delta)`.

### Gradient of activation function ###

First we need to calculate `dL/d(xw)` using the gradient of our activation function. Recall:

    dL/d(xw) = dL/dy * dy/d(xw)
             = dL/dy * df(xw)/d(xw)
             = dL/dy * f'(xw)

We use `void gradient_matrix(matrix m, ACTIVATION a, matrix d)` function to change delta from `dL/dy` to `dL/d(xw)`

### Derivative of loss w.r.t. weights ###

Next we want to calculate the derivative with respect to our weights, `dL/dw`. 

    dL/dw = dL/d(xw) * d(xw)/dw
          = dL/d(xw) * x

but remember to make the matrix dimensions work out right we acutally do the matrix operiation of `xt * dL/d(xw)` where `xt` is the transpose of the input matrix `x`.

In our layer we saved the input as `l->in`. Calculate `xt` using that and the matrix transpose function in our library, `matrix transpose_matrix(matrix m)`. Then calculate `dL/dw` and save it into `l->dw` (free the old one first to not leak memory!). We'll use this later when updating our weights.

### 1.4.3 Derivative of loss w.r.t. input ###

Next we want to calculate the derivative with respect to the input as well, `dL/dx`. Recall:

    dL/dx = dL/d(xw) * d(xw)/dx
          = dL/d(xw) * w

again, we have to make the matrix dimensions line up so it actually ends up being `dL/d(xw) * wt` where `wt` is the transpose of our weights, `w`. Calculate `wt` and then calculate dL/dx. This is the matrix we will return.

### Weight updates ###

After we've performed a round of forward and backward propagation we want to update our weights. Check  `void update_layer(layer *l, double rate, double momentum, double decay)` .

Remember from class with momentum and decay our weight update rule is:

    Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    w_{t+1} = w_t + ηΔw_t

We'll be doing gradient ascent (the partial derivative component is positive) instead of descent because we'll flip the sign of our partial derivative when we calculate it at the output. We saved `dL/dw_t` as `l->dw` and by convention we'll use `l->v` to store the previous weight change `Δw_{t-1}`.

Calculate the current weight change as a weighted sum of `l->dw`, `l->w`, and `l->v`. The function `matrix axpy_matrix(double a, matrix x, matrix y)` will be useful here, it calculates the result of the operation `ax+y` where `a` is a scalar and `x` and `y` are matrices. Save the current weight change in `l->v` for next round (remember to free the current `l->v` first).

Finally, apply the weight change to your weights by adding a scaled amount of the change based on the learning rate.







