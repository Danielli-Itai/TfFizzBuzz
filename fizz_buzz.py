# Fizz Buzz in Tensorflow!
# Our goal is to produce fizzbuzz for the numbers 1 to 100.
# So it would be unfair to include these in our training data.
# Accordingly, the training data corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
# see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
from PyBaseGUI.Mplot import  plot_confusion_matrix as plot
import numpy #as np
import tensorflow as tf
from sklearn import metrics
from sympy import sieve
from sympy.ntheory import factorint






#*************************************************************
#   Maximum number to clasify
PROGRESS_SHOW = False
MAX_NUM = 2**10





#*************************************************************
#   Binary Encoding
# Represent each input by an array of its binary digits.
def binary_digits(number:int)->int:
    count = 0
    while (number > 0):
        number = number // 2
        count = count + 1
    return(count)

def binary_encode(number:int, num_digits:int)->list:
    encode : list= numpy.array([number >> d & 1 for d in range(num_digits)]);
    return encode;




#*************************************************************
#   Primes Encoding
# Represent each input by an array of its primary number multiplier as digits.
def primes_digits(number:int)->int:
    sieve.extend(number)
    return(len(sieve._list))

def prime_encode(number:int, num_digits:int)->list:
    sieve.extend(number)
    prime_nums = sieve._list
    prim_factored = factorint(int(number))
    encode:list = list(range(num_digits));
    for prime in prime_nums:
        if prime in prim_factored:
            encode[prime_nums.index(prime)]=prim_factored[prime]
        else:
            encode[prime_nums.index(prime)]=0
    return encode;

def primes_encode(numbers, num_digits:int)->list:
    encode: list=[]
    if(type(numbers)==int):
        encode = prime_encode(numbers, num_digits);
    else:
        for num in numbers:
            encode.append(prime_encode(num, num_digits));
        encode = numpy.transpose(encode)
    return encode;



#*************************************************************
# Create the network model.
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




#*************************************************************
#   Network data place holders
# Our variables. The input has width in_digits, and the output has width out_digits.
def PlaceHolders(in_digits:int, out_digits):
    X = tf.placeholder("float", [None, in_digits])
    Y = tf.placeholder("float", [None, out_digits])
    return(X,Y);




#*************************************************************
#   Network Training
# Create Labled training data.

#   Fizz-Buzz data lables generation.
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

fizz_buzz_names=["num", "fizz", "buzz", "fizzbuzz"];
def fizz_buzz_name(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]


#   Mish-Buzz data lables generation.
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

mish_buzz_names:list=["num", "mish", "fizz", "buzz", "mishfizz", "mishbuzz", "fizzbuzz"];
def mish_buzz_name(i, prediction):
    return [str(i), "mish", "fizz", "buzz", "mishfizz", "mishbuzz", "fizzbuzz"][prediction]


def TrainData(test_start:int, max_in, in_digits, input_encoder, fizz_buzz_encode):
    trainX = numpy.array([input_encoder(i, in_digits) for i in range(test_start, max_in)])
    trainY = numpy.array([fizz_buzz_encode(i) for i in range(test_start, max_in)])
    return(trainX, trainY)

# Train the network using the labled data.
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
            if(PROGRESS_SHOW):
                print(epoch, numpy.mean(numpy.argmax(trainY, axis=1) == sess.run(predict_op, feed_dict={X: trainX, Y: trainY})))
    return;




#*************************************************************
#   Network testing.
def TensorTest(sess:tf.Session, input_encoder, test_last:int, in_digits:int, X, predict_op):
    # And now for some fizz buzz
    numbers = numpy.arange(1, test_last + 1)
    teX = numpy.transpose(input_encoder(numbers, in_digits))
    teY = sess.run(predict_op, feed_dict={X: teX})
    return (numbers, teY)




#*************************************************************
#   Performance report.
#Report the network performance.
def Report(numbers:list, predicted_vec:list, fizz_buzz_name, fizz_buzz_cls, class_names:list):
    # Calculate the expected vector.
    expected_vec = numpy.vectorize(fizz_buzz_cls)(numbers)

    #Print the output vector.
    output_vec = numpy.vectorize(fizz_buzz_name)(numbers, predicted_vec)
    print(output_vec)

    #Print accuracy mesure.
    accuracy = metrics.accuracy_score(expected_vec, predicted_vec)
    print('Accuracy : ' + str(accuracy));

    #Print confusion matrix.
    conf_matrix = metrics.confusion_matrix(expected_vec, predicted_vec)
    print('Confusion matrix : ' + str(conf_matrix))
    plot(actual_cls=expected_vec, predict_cls=predicted_vec, classes=class_names)

    return;




#*************************************************************
#   Run the network learning algorithm.
def TensorFlowRun(num_hidden:int, train_size:int, input_encoder, exp_output, exp_class, class_name, max_in:int, in_digits:int, class_names, report:bool):
    TEST_LAST = 100
    BATCH_SIZE = 128  # Number of samples to test.
    TRAIN_SIZE = BATCH_SIZE * train_size

    out_digits: int = len(class_names);
    with tf.Session() as sess:
        w_h, w_o = WeightsInit(in_digits, num_hidden, out_digits);
        X, Y = PlaceHolders(in_digits, out_digits);
        train_op, predict_op = Optimizer(X, Y, w_h, w_o);
        tf.initialize_all_variables().run()

        trainX, trainY = TrainData(TEST_LAST + 1, max_in, in_digits, input_encoder, exp_output)
        TensorTrain(X, Y, TRAIN_SIZE, BATCH_SIZE, trainX, trainY, train_op, predict_op, sess)

        numbers, test_output = TensorTest(sess, input_encoder, TEST_LAST + 1, in_digits, X, predict_op)
        if(report): Report(numbers, test_output, class_name, exp_class, class_names)
    return;








#*************************************************************
#   Running experiments

#  questions 1-4
#1. Take the FizzBuzz example and make it work: It works.
#2. Add to the code a function that analyzes the results.
#	a. Print the accuracy of the classifier: See the report function and the batch accuracy reort.
#	b. Generate a confusion matrix:          See the report function.
#3. Run it once and put the results in a word file: the loop can be set to once or 10 times.
NUM_HIDDEN = 100  # How many units in the hidden layer.
NUM_IN_DIGITS = binary_digits(MAX_NUM)  # Number of binary digits (Maximum number)
TRAIN_SIZE = 8
print("Run bizz buzz 1 time with progress combinations of 2,3,6,10")
for i in range (1):
	TensorFlowRun(NUM_HIDDEN, TRAIN_SIZE, binary_encode, fizz_buzz_encode, fizz_buzz_cls, fizz_buzz_name, MAX_NUM, NUM_IN_DIGITS, fizz_buzz_names, True);

#4. Rerun the algorithm 10 times and see if there are differences in the accuracy:  The alogrithem is executed 10 times.
NUM_HIDDEN = 100  # How many units in the hidden layer.
NUM_IN_DIGITS = binary_digits(MAX_NUM)  # Number of binary digits (Maximum number)
TRAIN_SIZE = 8
print("Run 8 times without progress report to compare results")
for i in range (9):
	TensorFlowRun(NUM_HIDDEN, TRAIN_SIZE, binary_encode, fizz_buzz_encode, fizz_buzz_cls, fizz_buzz_name, MAX_NUM, NUM_IN_DIGITS, fizz_buzz_names, False);
TensorFlowRun(NUM_HIDDEN, TRAIN_SIZE, binary_encode, fizz_buzz_encode, fizz_buzz_cls, fizz_buzz_name, MAX_NUM, NUM_IN_DIGITS, fizz_buzz_names, True);


# questions 5-6
#5. Change the algorithm to also deal with the number 2 besides the numbers 3
#	and 5 with all the relevant combinations (more classes2,3,5): we added the mish_buzz ancoding and class functions and class names.
#6. Run it once and put the results in a word file.
NUM_HIDDEN = 100  # How many units in the hidden layer.
print("Add mish buzz combinations of 2,3,5,6,10,15")
NUM_IN_DIGITS = binary_digits(MAX_NUM)   # Number of binary digits (Maximum number)
TRAIN_SIZE = 8
for i in range (1):
	TensorFlowRun(NUM_HIDDEN, TRAIN_SIZE, binary_encode, mish_buzz_encode, mish_buzz_cls, mish_buzz_name, MAX_NUM, NUM_IN_DIGITS, mish_buzz_names, True);


# a question 7
#7. Try to improve the results by more training, change the network etc.
NUM_HIDDEN = 5000  # How many units in the hidden layer. we increase from 100 to 5000
NUM_IN_DIGITS = primes_digits(MAX_NUM)   # Number of binary digits (Maximum number)
TRAIN_SIZE = 1
print("improve binary encode training and hidden layer")
for i in range (1):
    TensorFlowRun(NUM_HIDDEN, TRAIN_SIZE, primes_encode, mish_buzz_encode, mish_buzz_cls, mish_buzz_name, MAX_NUM, NUM_IN_DIGITS, mish_buzz_names, True);

#  a question 8 for fizzbuzz
#8. Change the representation from binary to prime based(encode the numbers as
#	prime number multiplayer). That means each number is coded by how many
#	time each prime appears in the product. For large primes you can put them all
#	in one bucket.
#	Example: 24 is coded as 2^3* 3^1 -? [3,1,0,0,0,0…]
#	Do you think the algorithms (first and second) will work better? Run them
#	and write the results and compare them.

NUM_HIDDEN = 100  # How many units in the hidden layer.
NUM_IN_DIGITS = primes_digits(MAX_NUM)  # Number of binary digits (Maximum number)
TRAIN_SIZE = 2
print("perform prim encoding with small hidden layer and small number of train")
for i in range(0):
    TensorFlowRun(NUM_HIDDEN, TRAIN_SIZE, primes_encode, mish_buzz_encode, mish_buzz_cls, mish_buzz_name, MAX_NUM, NUM_IN_DIGITS, mish_buzz_names, True);


input("Press Enter to continue...")