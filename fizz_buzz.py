# Fizz Buzz in Tensorflow!
# see http://joelgrus.com/2016/05/23/fizz-buzz-in-tensorflow/
from Base.plot_confusion_matrix import plot_confusion_matrix as plot
import numpy #as np
import tensorflow as tf
from sklearn.metrics import confusion_matrix

NUM_DIGITS = 10

# Represent each input by an array of its binary digits.
def binary_encode(i, num_digits):
    return numpy.array([i >> d & 1 for d in range(num_digits)])

# One-hot encode the desired outputs: [number, "fizz", "buzz", "fizzbuzz"]
def fizz_buzz_encode(i):
    if   i % 15 == 0: return numpy.array([0, 0, 0, 1])
    elif i % 5  == 0: return numpy.array([0, 0, 1, 0])
    elif i % 3  == 0: return numpy.array([0, 1, 0, 0])
    else:             return numpy.array([1, 0, 0, 0])

# Our goal is to produce fizzbuzz for the numbers 1 to 100. So it would be
# unfair to include these in our training data. Accordingly, the training data
# corresponds to the numbers 101 to (2 ** NUM_DIGITS - 1).
trX = numpy.array([binary_encode(i, NUM_DIGITS) for i in range(101, 2 ** NUM_DIGITS)])
trY = numpy.array([fizz_buzz_encode(i)          for i in range(101, 2 ** NUM_DIGITS)])

# We'll want to randomly initialize weights.
def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

# Our model is a standard 1-hidden-layer multi-layer-perceptron with ReLU
# activation. The softmax (which turns arbitrary real-valued outputs into
# probabilities) gets applied in the cost function.
def model(X, w_h, w_o):
    h = tf.nn.relu(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

# Our variables. The input has width NUM_DIGITS, and the output has width 4.
X = tf.placeholder("float", [None, NUM_DIGITS])
Y = tf.placeholder("float", [None, 4])

# How many units in the hidden layer.
NUM_HIDDEN = 100

# Initialize the weights.
w_h = init_weights([NUM_DIGITS, NUM_HIDDEN])
w_o = init_weights([NUM_HIDDEN, 4])

# Predict y given x using the model.
py_x = model(X, w_h, w_o)

# We'll train our model by minimizing a cost function.
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x, labels=Y))
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost)

# And we'll make predictions by choosing the largest output.
predict_op = tf.argmax(py_x, 1)

# Finally, we need a way to turn a prediction (and an original number)
# into a fizz buzz output
def fizz_buzz_cls(num):
    if   num % 15 == 0: return 3
    elif num % 5  == 0: return 2
    elif num % 3  == 0: return 1
    else: return 0

def fizz_buzz_name(i, prediction):
    return [str(i), "fizz", "buzz", "fizzbuzz"][prediction]

def fizz_buzz_str(num):
    return(fizz_buzz_name(num,fizz_buzz_cls(num)));



BATCH_SIZE = 128

# Launch the graph in a session
with tf.Session() as sess:
    tf.initialize_all_variables().run()

    for epoch in range(1000):
        # Shuffle the data before each training iteration.
        p = numpy.random.permutation(range(len(trX)))
        trX, trY = trX[p], trY[p]

        # Train in batches of 128 inputs.
        for start in range(0, len(trX), BATCH_SIZE):
            end = start + BATCH_SIZE
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})

        # And print the current accuracy on the training data.
        print(epoch, numpy.mean(numpy.argmax(trY, axis=1) == sess.run(predict_op, feed_dict={X: trX, Y: trY})))

    # And now for some fizz buzz
    numbers = numpy.arange(1, 101)
    teX = numpy.transpose(binary_encode(numbers, NUM_DIGITS))
    teY = sess.run(predict_op, feed_dict={X: teX})
    output_str = numpy.vectorize(fizz_buzz_name)(numbers, teY)

    #Calculate the expected vector.
    print(output_str)

    expected_vec = numpy.vectorize(fizz_buzz_cls)(numbers)
    output_cls = numpy.vectorize(fizz_buzz_cls)(teY)
    conf_matrix = confusion_matrix(expected_vec, output_cls)
    print(conf_matrix)
    plot(y_true=expected_vec, y_pred=output_cls, classes=["num","fizz","buzz","fizzbuzz"])