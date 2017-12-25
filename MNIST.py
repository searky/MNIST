import os
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0, shape=shape, dtype=tf.float32)
    return tf.Variable(initial)

def model(img):
    x_image = tf.reshape(img, [-1, 28, 28, 1])

    # conv1
    # in LeNet paper, the input is 32x32x1, but here is 29x28x1.
    # so use 'SAME' padding to make sure the input for next convolution layer is 28x28x1.
    kernel1 = weight_variable([5, 5, 1, 6])
    bias1 = bias_variable([6])
    conv1 = tf.nn.conv2d(x_image, kernel1, [1, 1, 1, 1], padding='SAME')
    activation1 = tf.nn.sigmoid(tf.nn.bias_add(conv1, bias1), 'conv1')

    # pool1 
    pool1 = tf.nn.avg_pool(activation1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    #conv2
    kernel2 = weight_variable([5, 5, 6, 16])
    bias2 = bias_variable([16])
    conv2 = tf.nn.conv2d(pool1, kernel2, [1, 1, 1, 1], padding='VALID')
    activation2 = tf.nn.sigmoid(tf.nn.bias_add(conv2, bias2), 'conv2')

    # pool2 
    pool2 = tf.nn.avg_pool(activation2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    pool2_flat = tf.reshape(pool2, [-1, 5 * 5 * 16])

    # fc1
    weight_fc1 = weight_variable([5 * 5 * 16, 120])
    bias_fc1 = bias_variable([120])
    activation_fc1 = tf.nn.sigmoid(tf.nn.bias_add(tf.matmul(pool2_flat, weight_fc1), bias_fc1), 'fc1')

    # fc2
    weight_fc2 = weight_variable([120, 84])
    bias_fc2 = bias_variable([84])
    activation_fc2 = tf.nn.bias_add(tf.matmul(activation_fc1, weight_fc2), bias_fc2)

    # fc3/softmax
    weight_fc3 = weight_variable([84, 10])
    bias_fc3 = bias_variable([10])
    activation_fc3 = tf.nn.bias_add(tf.matmul(activation_fc2, weight_fc3), bias_fc3)

    return activation_fc3

def main():
    mnist = input_data.read_data_sets("./MNIST_data/", one_hot=True)
    # input image is 28x28=784
    x = tf.placeholder(tf.float32, [None, 784], name='x')
    # label
    y = tf.placeholder(tf.float32, [None, 10], name='y')

    # prediction
    y_ = model(x)

    # cross entropy loss function
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_)
    loss = tf.reduce_mean(loss)

    # learning rate: 0.001
    train_op = tf.train.AdamOptimizer(0.001).minimize(loss)

    # calculate the accuracy
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    correct_prediction = tf.cast(correct_prediction, tf.float32)
    accuracy = tf.reduce_mean(correct_prediction)

    # log
    logs_dir = 'C:/log/'
    if os.path.exists(logs_dir) == False:
        os.mkdir(logs_dir)
    tf.summary.scalar('loss', loss)
    tf.summary.scalar('accuracy', accuracy)
    summary_op = tf.summary.merge_all()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        writer = tf.summary.FileWriter(logs_dir,sess.graph)

        for i in range(10000):
            batch = mnist.train.next_batch(64)
            
            _, summary =  sess.run([train_op, summary_op], feed_dict={x: batch[0], y: batch[1]})

            if i % 100 == 0:
                train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1]})
                print('step %d, training accuracy: %g' % (i, train_accuracy))
            writer.add_summary(summary, global_step=i)

        print('test accuracy: %g' % accuracy.eval(feed_dict={x: mnist.test.images, y: mnist.test.labels}))

if __name__ == '__main__':
    main()