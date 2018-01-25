# Import MNIST data
import input_data
mnist = input_data.read_data_sets("D:\Studia\Inne\TensorFlow_Demo\data", one_hot=True)

import tensorflow as tf

# Set parameters
learning_rate = 0.01
training_iteration = 20
batch_size = 100

# TF graph input
x = tf.placeholder("float", [None, 784], name='input') # mnist data image of shape 28*28=784
y = tf.placeholder("float", [None, 10], name='output') # 0-9 digits recognition => 10 classes

# Create a model

# Set model weights
W = tf.Variable(tf.zeros([784, 10]), name='weight')
b = tf.Variable(tf.zeros([10]), name='bias')

with tf.variable_scope("Wx_b"):
    # Construct a linear model
    model = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Add summary ops to collect data
w_h = tf.summary.histogram(W.op.name, W)
b_h = tf.summary.histogram(b.op.name, b)

# Create loss function
with tf.variable_scope("cost_function"):
    # Minimize error using cross entropy
    cost_function = -tf.reduce_sum(y*tf.log(model))
    # Create a summary to monitor the cost function
    tf.summary.scalar(cost_function.op.name, cost_function)

with tf.variable_scope("train"):
    # Gradient descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost_function)

# Initializing the variables
init = tf.global_variables_initializer()

# Merge all summaries into a single operator
merged_summary_op = tf.summary.merge_all()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Set the logs writer
    # summary_writer = tf.summary.FileWriter('D:\Studia\Inne\TensorFlow_Demo', graph=sess.graph)
    summary_writer = tf.summary.FileWriter('board_log', sess.graph)

    # Training cycle
    for iteration in range(training_iteration):
        avg_cost = 0.
        total_batch = int(mnist.train.num_examples/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            # Fit training using batch data
            sess.run(optimizer, feed_dict={x: batch_xs, y: batch_ys})
            # Compute the average loss
            avg_cost += sess.run(cost_function, feed_dict={x: batch_xs, y: batch_ys})/total_batch
            # Write logs for each iteration
            summary_str = sess.run(merged_summary_op, feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary_str, iteration*total_batch + i)
        # Display logs per iteration step
        print("Iteration:", '%04d' % (iteration + 1), "cost=", "{:.9f}".format(avg_cost))

    print("Training completed!")

    # Test the model
    predictions = tf.equal(tf.argmax(model, 1), tf.argmax(y, 1))
    # Calculate accuracy
    accuracy = tf.reduce_mean(tf.cast(predictions, "float"))
    print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))


# tensorboard --logdir=D:\Studia\Inne\TensorFlow_Demo

