# Neural Network from Scratch  
## What is a neural Network?
Wiki says that a neural network is a network or circuit of neurons, or in a modern sense, an artificial neural network, composed of artificial neurons or nodes.These artificial networks may be used for predictive modeling, adaptive control and applications where they can be trained via a dataset. Self-learning resulting from experience can occur within networks, which can derive conclusions from a complex and seemingly unrelated set of information.

Look at this picture. 

![Neural Network](ReadmeData/Neural_Network_Example.png)


## Why this?
I created this file to act as a starter template for students like me that want to implement a neural network and learn from it. 

Most of the time I wanted to learn neural netwok, I came across a number of different parameters like weight initialization, # of neurons, decay rate, learning rate, batch size etc. And I always thought that there must be something or a course that will tell you all of these because most of the teachers online say that it's all about trial and error and it is about trial and error.

 But finally I thought of creating a neural network on my own and came across some the best teachers in the world Joseph Redmon aka Pjreddie. He has open-sourced a number of lectures and they are a must to watch if you want to learn computer-vision . This the course link : [Ancient Secrets of Computer Vision](https://pjreddie.com/courses/computer-vision/)

## Implementing neural networks ##

*I strongly recommend that you atleast see Lectures 12 and 13 to have a basic understanding of neural networks* 

All the methods related to neural network are implemented in `src\classifier.c`. Check all the mentioned functions in this file.

A neural network is made up of a number of layers or fully connected layers. Each layer has following entites:
1. Input neurons
2. Weights corresponding to each neuron
3. Current weight updates while backpropogating
4. Past weight updates to use with momentum
5. Activation Function to add non-linearity in model
5. Output neurons


### Activation functions ###

An important part of machine learning, be it classifiers or neural networks, is the activation function you use. This has been implemented in `void activate_matrix(matrix m, ACTIVATION a)` and is used to modify `m` to be `f(m)` applied elemetwise all along the output matrix.

The different Activation functions that I have implemented are:
1. LINEAR        --> Just a linear activation
2. LOGISTIC OR SIGMOID     --> Not so good due to vanishing gradients
3. RELU          --> Rectified-linear-Unit
4. LRELU         --> Leaky-Relu
5. SOFTMAX       --> Mainly used before the output layer 

### Taking gradients ###

As we are calculating the partial derivatives for the weights of our model we have to remember to factor in the gradient from our activation function at every layer. We will have the derivative of the loss with respect to the output of a layer (after activation) and we need the derivative of the loss with respect to the input (before activation). To do that we take each element in our delta (the partial derivative) and multiply it by the gradient of our activation function.

Normally, to calculate `f'(x)` we would need to remember what the input to the function, `x`, was. However, all of our activation functions have the nice property that we can calculate the gradient `f'(x)` only with the output `f(x)`. This means we have to remember less stuff when running our model.

In my view you must differentiate all the functions yourself and then see if I have implemented it rightly or not. Check this readthedocs for detailed information [Activation Functions Described](https://ml-cheatsheet.readthedocs.io/en/latest/activation_functions.html).


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

Remember that with momentum and decay our weight update rule is:

    Δw_t = dL/dw_t - λw_t + mΔw_{t-1}
    w_{t+1} = w_t + ηΔw_t

We'll be doing gradient ascent (the partial derivative component is positive) instead of descent because we'll flip the sign of our partial derivative when we calculate it at the output. We saved `dL/dw_t` as `l->dw` and by convention we'll use `l->v` to store the previous weight change `Δw_{t-1}`.

Calculate the current weight change as a weighted sum of `l->dw`, `l->w`, and `l->v`. The function `matrix axpy_matrix(double a, matrix x, matrix y)` will be useful here, it calculates the result of the operation `ax+y` where `a` is a scalar and `x` and `y` are matrices. Save the current weight change in `l->v` for next round (remember to free the current `l->v` first).

Finally, apply the weight change to your weights by adding a scaled amount of the change based on the learning rate.

## Creating and learning a Model

### Creating a New Layer
To create a model that can do something, you have to create a number of hidden layers in the first place and for all those hidden layers you have to think of the best suited activation function and different hyperparameters that come along. 

Lets see how to create a new layer:

As you can see in line 245 `src/classifier.c`.

* We initialize the `l->in` matrix and `l->out` with one neuron.

* We do the weight initialization quite effectively. There a few methods to do so. One is known as Xavier Initialization. See it here [Xavier Initialization](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), here I initialize the weights as a random matrix of size `input*output` with values that lie in the range `(-sqrt(2/input), sqrt(2/input))`. And this works pretty well

* We create the past weights updates as l->v, current weight updates l->dw of same dimensions as that of l->w.

* Final Step is to have a activation function for the layer. And we do this by setting `l->activation`.

### Running the model Forward
After every time we backpropogate and do weight updation we have to do forward propagation. 

Check line 261 for that `matrix forward_model(model m, matrix X)`.
Now this function does is call the forward_layer function for all the layers in the model and set the weights.
`X = foward_layer(m.(layers+i), X)` sets X that is the input matrix to the output matrix at each layer.

### Running the model Backward
Once the weights are all set, we have to backpropogate the error through the model. Check line 273 in the `src/classifier.c` for more information on the above method. We send the matrix Loss_D (which represnts loss of each layer) in backward sense throgh the `backward_layer(layer* working_layer, matrix Loss_D)` .


### Training the Model
See line 357. The function declaration looks like this : `void train_model(model m, data d, int batch, int iters, double rate, double momentum, double decay)`.

Now lets see what this function do.
1. The model trains for `int iters` number of times.
2. Creates a random data batch of batch size `batch` from the whole data points.
3. Do a forward_propagation taking in (model, batch.X)
4. Calculate the cross entropy loss for the last layer.
5. Calculate the loss matrix and backpropogate it through the model.
6. Update the model weight through ` update_model(m, rate/batch, momentum, decay);`. See for SGD the rate changes to rate/batchsize.

## MUST READ
*After reading each and every bit of information that I have written here you are equipped with most of information needed to write something similar to this repo. I strongly feel that you should create your own Neural Network in C specially because you will be defining all the matrix methods your-self and also implementing all these functions your self.* 

You don't have to code everything from scratch. I have build this on top of my existing repo. Check the repo [Image_Processing_Starter_Packge](https://github.com/jalotra/Image_Processing_Starter_Package)

## Installation:

1. Clone this repository and make sure that you have Gcc installed. If you want to use any other C compiler, feel free to do that in the MakeFile that's presents in the root of this repo.
See line 15 in `Makefile`. Also  make sure you have `make` installed.

2. Fire `make` from a Cli. This will create all the the objects and link them to each other.

3. Finally create you own neural nets and play with them as whole of the project has a python wrapper. So you just have to define models in a `.py` file. Example models are present in the neuralNetworkTrainer.py file.

### Datasets to play with 
1. MNIST DATASET

To run your model you'll need the dataset. The training images can be found [here](https://pjreddie.com/media/files/mnist_train.tar.gz) and the test images are [here](https://pjreddie.com/media/files/mnist_test.tar.gz), I've preprocessed them for you into PNG format. To get the data you can run:


    wget https://pjreddie.com/media/files/mnist_train.tar.gz
    wget https://pjreddie.com/media/files/mnist_test.tar.gz
    tar xzf mnist_train.tar.gz
    tar xzf mnist_test.tar.gz

We'll also need a list of the images in our training and test set. To do this you can run:

    find train -name \*.png > mnist.train
    find test -name \*.png > mnist.test  

2. CIFAR DATASET

We have to do a similar process as last time, getting the data and creating files to hold the paths. Run:

  ```
    wget http://pjreddie.com/media/files/cifar.tgz
    tar xzf cifar.tgz
    find cifar/train -name \*.png > cifar.train
    find cifar/test -name \*.png > cifar.test
```

## Further Developments
Since fo now each layer is connected to the next layer in a fully connected way. I want to implement some other layers also like Convolution Layer, Max Pooling Layer, Dropout Layer etc.


## Contributing 
1. If you have some doubt reagarding any line of code email me right away or create an issue. [Email](jalotrashivam9@gmail.com)

2. If you would like to create custom layers as I have written above. Create a issue and discuss with me as I want this repo to be as clean as possible.

## Authors
**Shivam Jalotra** - (https://github.com/jalotra)

## Acknowledgments

* I thank Pjreddie from the bottom of my heart for creating such a greate course and wonderful assignments.
* I think that everybody who wants to learn Computer Vision or Neural Network in general should watch do this course [Ancient Secrets of Computer Vision](https://pjreddie.com/courses/computer-vision/)
* Finally I thank you for reading through all this. 








