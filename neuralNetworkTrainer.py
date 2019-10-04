from uwimg import *

def softmax_model(inputs, outputs):
    l = [make_layer(inputs, outputs, SOFTMAX)]
    return make_model(l)

def neural_net(inputs, outputs):
    print (inputs)
    l = [make_layer(inputs, 32, LOGISTIC),
            make_layer(32, outputs, SOFTMAX)]
    return make_model(l)



if __name__ == "__main__":
	print("loading data...")
	train_file_path = "mnist.train"
	labels_path = "mnist.labels"
	test_file_path = "mnist.test"

	train = load_classification_data(c_char_p(train_file_path.encode('utf-8')), c_char_p(labels_path.encode('utf-8')), 1)
	test  = load_classification_data(c_char_p(test_file_path.encode('utf-8')),c_char_p(labels_path.encode('utf-8')) , 1)
	print("done")
	print()

	print("training model...")
	batch = 128
	iters = 1000
	rate = .01
	momentum = .9
	decay = .0005

	m = softmax_model(train.X.cols, train.y.cols)
	train_model(m, train, batch, iters, rate, momentum, decay)
	print("done")
	print()

	print("evaluating model...")
	print("training accuracy: %f", accuracy_model(m, train))
	print("test accuracy:     %f", accuracy_model(m, test))
