import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import random


def convert2tensor(my_type, num):
    return tf.convert_to_tensor(my_type[num], dtype=tf.float32)


def prepare_vocabulary(my_data):
    idx = 0
    for sentence in new_data_x:
        for word in sentence.split():
            if word not in word2location:
                word2location[word] = idx
                idx += 1
    return idx


def convert2vec(sentence):
    res_vec = np.zeros(vocabulary_size)
    for word in sentence.split():
        if word in word2location:
            res_vec[word2location[word]] += 1
    return res_vec


counter = 0
accuracy = 0
vocabulary_size = 0
word2location = {}

data = pd.read_csv("C:/Users/Computer/Desktop/Input.csv").values

new_data_x = []
new_data_y = []

for i in range(len(data)):
    new_data_x.append(data[i][0])
    new_data_y.append(data[i][1])

vocabulary_size = prepare_vocabulary(new_data_x)
features = vocabulary_size
categories = 3

x = tf.placeholder(tf.float32, [None, features])
y = tf.placeholder(tf.float32, [None, categories])
W1 = tf.Variable(tf.zeros([features, features]))
b1 = tf.Variable(tf.zeros([features]))
W2 = tf.Variable(tf.zeros([features, categories]))
b2 = tf.Variable(tf.zeros([categories]))

z1 = tf.sigmoid(tf.matmul(x, W1) + b1)
a1 = tf.nn.relu(z1)
y_ = tf.nn.softmax(tf.matmul(a1, W2) + b2)

book_name = tf.convert_to_tensor(['Genesis', 'Isaiah', 'Kings'], dtype=tf.string)

loss = tf.reduce_sum(tf.square(y - y_))
update = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

X_train, X_test, y_train, y_test = train_test_split(new_data_x, new_data_y, test_size=0.2)

data_x = np.array([convert2vec(X_train[i]) for i in range(len(X_train))])
data_y = np.array([np.array(y_train[i].split(",")) for i in range(len(y_train))])

test_y = np.array([np.array(y_test[i].split(",")) for i in range(len(y_test))])

sess = tf.Session()

sess.run(tf.global_variables_initializer())

step = 6
for i in range(step):
    for j in range(1000):
        sess.run(update, feed_dict={x: data_x, y: data_y})
    predict = sess.run(tf.argmax(y_, 1), feed_dict={x: data_x, y: data_y})
    label = sess.run(tf.argmax(y, 1), feed_dict={y: data_y})
    print("Step: ", i + 1, "/", step)
    counter = 0
    for i in range(len(predict)):
        if predict[i] == label[i]:
            counter += 1
    accuracy = 100*counter/len(data_x)
    print("Loss: ", sess.run(loss, feed_dict={x: data_x, y: data_y}), "\nAccuracy: ", round(accuracy, 3), '%')
    print()

counter = 0
for j in range(len(X_test)):
    print(j+1, ': Prediction for: "', X_test[j], ': "', sess.run(y_, feed_dict={x: [convert2vec(X_test[j])]}))
    print("Book name is: ", sess.run(tf.gather_nd(book_name, tf.argmax(y_, 1)), feed_dict={x: [convert2vec(X_test[j])]})
          , "Prediction is:", sess.run(tf.equal(tf.argmax(y_, 1), tf.argmax(convert2tensor(test_y, j))),
                                       feed_dict={x: [convert2vec(X_test[j])]}), "Correct Answer is book: ",
          sess.run(tf.gather(book_name, tf.argmax(convert2tensor(test_y, j))), feed_dict={x: [convert2vec(X_test[j])]}))
    predict = sess.run(tf.argmax(y_, 1), feed_dict={x: [convert2vec(X_test[j])]})
    label = sess.run(tf.argmax(convert2tensor(test_y, j)), feed_dict={x: [convert2vec(X_test[j])]})
    print()
    if predict == label:
        counter += 1
accuracy = 100*counter/len(X_test)
print("Accuracy: ", round(accuracy, 3), '%', "Right answers: ", counter, "/", len(X_test))
