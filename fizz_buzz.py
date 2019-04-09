# Fizz Buzz in Tensorflow!
# Our goal is to produce fizzbuzz for the numbers 1 to 100.
# So it would be unfair to include these in our training data.
# Accordingly, the training data corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
# see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
from Base.plot_confusion_matrix import plot_confusion_matrix as plot
import numpy #as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix







# Represent each input by an array of its binary digits.
def binary_encode(i:int, num_digits:int):
    return numpy.array([i >> d & 1 for d in range(num_digits)])

# Generate random weights.
def WeightsInit(in_digits:int, num_hidden:int, out_digits:int):
    # We'll want to randomly initialize weights.
    def init_weights(shape):
        return tf.Variable(tf.random_normal(shape, stddev=0.01))

    # Initialize the weights.
    weights_h = init_weights([in_digits, num_hidden])  #Hidden layer weights.
    weights_o = init_weights([num_hidden, out_digits]) #Output layer weights.
    return(weights_h, weights_o)


# Our model is a standard 1-hidden-layer multi-layer-perceptron with ReLU activation.
# The softmax (which turns arbitrary real-valued outputs into probabilities) gets applied in the cost function.
def model(X, weights_h, weights_o):
    h = tf.nn.relu(tf.matmul(X, weights_h))
    return tf.matmul(h, weights_o)

#Create optimization function.
def Optimizer(X, Y, w_h, w_o):
    # Predict y given x using the model.
    py_x = model(X, w_h, w_o)

    # We'll train our model by minimizing a cost function.
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
    train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

    # And we'll make predictions by choosing the largest output.
    predict_op = tf.argmax(py_x, 1)
    return(train_op, predict_op)



# Our variables. The input has width in_digits, and the output has width 4.
def PlaceHolders(in_digits:int, out_digits):
    X = tf.placeholder("float", [None, in_digits])
    Y = tf.placeholder("float", [None, out_digits])
    return(X,Y);



def TrainData(test_start:int, in_digits, binary_encode, fizz_buzz_encode):
    trainX = numpy.array([binary_encode(i, in_digits) for i in range(test_start, 2 ** in_digits)])
    trainY = numpy.array([fizz_buzz_encode(i) for i in range(test_start, 2 ** in_digits)])
    return(trainX, trainY)

# Launch the graph in a session
def TensorTrain(X, Y, test_size:int, batch_size:int, trainX:list, trainY:list, train_op, predict_op, sess:tf.Session):
    for epoch in range(test_size):
        # Shuffle the data before each training iteration.
        p = numpy.random.permutation(range(len(trainX)))
        trainX, trainY = trainX[p], trainY[p]

        # Train in batches of 128 inputs.
        for start in range(0, len(trainX), batch_size):
            end = start + batch_size
            sess.run(train_op, feed_dict={X: trainX[start:end], Y: trainY[start:end]})

        # And print the current accuracy on the training data.
        print(epoch, numpy.mean(numpy.argmax(trainY, axis=1) == sess.run(predict_op, feed_dict={X: trainX, Y: trainY})))
    return;


def TensorTest(sess:tf.Session, test_start:int, in_digits:int, X, predict_op):
    # And now for some fizz buzz
    numbers = numpy.arange(1, test_start + 1)
    teX = numpy.transpose(binary_encode(numbers, in_digits))
    teY = sess.run(predict_op, feed_dict={X: teX})
    return (numbers, teY)


#Report the network performance.
def Report(numbers:list, teY:list, fizz_buzz_name, fizz_buzz_cls):
    output_str = numpy.vectorize(fizz_buzz_name)(numbers, teY)
    print(output_str)

    expected_vec = numpy.vectorize(fizz_buzz_cls)(numbers)
    conf_matrix = confusion_matrix(expected_vec, teY)
    print(conf_matrix)
    plot(y_true=expected_vec, y_pred=teY, classes=["num", "fizz", "buzz", "fizzbuzz"])
    return;


def TensorFlowRun(input_encoder, exp_output, exp_class, class_name, in_digits:int, out_digits:int):
    NUM_HIDDEN = 5000  # How many units in the hidden layer.
    TEST_LAST = 100
    TRAIN_SIZE = 1000
    BATCH_SIZE = 128  # Number of samples to test.

    with tf.Session() as sess:
        w_h, w_o = WeightsInit(in_digits, NUM_HIDDEN, out_digits);
        X, Y = PlaceHolders(in_digits, out_digits);
        train_op, predict_op = Optimizer(X, Y, w_h, w_o);
        tf.initialize_all_variables().run()

        for i in range(2):
            trainX, trainY = TrainData(TEST_LAST + 1, in_digits, input_encoder, exp_output)
            TensorTrain(X, Y, TRAIN_SIZE, BATCH_SIZE, trainX, trainY, train_op, predict_op, sess)

        numbers, test_output = TensorTest(sess, TEST_LAST + 1, in_digits, X, predict_op)

        Report(numbers, test_output, class_name, exp_class)
        # Calculate the expected vector.
    return;








# One-hot encode the desired network outputs: [number, "fizz", "buzz", "fizzbuzz"])
def fizz_buzz_encode(i):
    if   i % 15 == 0: return numpy.array([0, 0, 0, 1])
    elif i % 5  == 0: return numpy.array([0, 0, 1, 0])
    elif i % 3  == 0: return numpy.array([0, 1, 0, 0])
    else:             return numpy.array([1, 0, 0, 0])

# Finally, we need a way to turn a prediction (and an original number) into a fizz buzz output
def fizz_buzz_cls(num):
    if   num % 15 == 0: return 3
    elif num % 5  == 0: return 2
    elif num % 3  == 0: return 1
    else: return 0

def fizz_buzz_name(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]







# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz", "mish", "mishfizz", "mishbuzz"]
def mish_buzz_encode(i):
    if   i % 15 == 0: return numpy.array([0, 0, 0, 0, 0, 0, 1])
    elif i % 10 == 0: return numpy.array([0, 0, 0, 0, 0, 1, 0])
    elif i % 6 == 0:  return numpy.array([0, 0, 0, 0, 1, 0, 0])
    elif i % 5  == 0: return numpy.array([0, 0, 0, 1, 0, 0, 0])
    elif i % 3  == 0: return numpy.array([0, 0, 1, 0, 0, 0, 0])
    elif i % 2 == 0:  return numpy.array([0, 1, 0, 0, 0, 0, 0])
    else:             return numpy.array([1, 0, 0, 0, 0, 0, 0])


def mish_buzz_cls(num):
    if   num % 15 == 0: return 6
    elif num % 10 == 0: return 5
    elif num % 6 == 0:  return 4
    elif num % 5  == 0: return 3
    elif num % 3  == 0: return 2
    elif num % 2  == 0: return 1
    else: return 0

def mish_buzz_name(i, prediction):
    return [str(i), "mish", "fizz", "buzz", "mishfizz", "mishbuzz", "fizzbuzz"][prediction]

NUM_IN_DIGITS = 10   # Number of binary digits (Maximum number)
NUM_OUT_DIGITS = 4;
#TensorFlowRun(binary_encode, fizz_buzz_encode, fizz_buzz_cls, fizz_buzz_name, NUM_IN_DIGITS, NUM_OUT_DIGITS);

NUM_IN_DIGITS = 10   # Number of binary digits (Maximum number)
NUM_OUT_DIGITS = 7;
TensorFlowRun(binary_encode, mish_buzz_encode, mish_buzz_cls, mish_buzz_name, NUM_IN_DIGITS, NUM_OUT_DIGITS);

input("Press Enter to continue...")