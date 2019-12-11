import tensorflow as tf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


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


vocabulary_size = 0
word2location = {}

data = pd.read_csv("C:/Users/Computer/Desktop/Input.csv").values
# validate = pd.read_csv("C:/Users/Computer/Desktop/Validate.csv").values

new_data_x = []
new_data_y = []
count = []

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

book_name = ["Genesis", "Isaiah", "Kings"]

loss = tf.reduce_sum(tf.square(y - y_))
update = tf.train.GradientDescentOptimizer(0.0005).minimize(loss)

X_train, X_test, y_train, y_test = train_test_split(new_data_x, new_data_y, test_size=0.2, random_state=42)

data_x = np.array([convert2vec(X_train[i]) for i in range(len(X_train))])
data_y = np.array([np.array(y_train[i].split(",")) for i in range(len(y_train))])

test_y = np.array([np.array(y_test[i].split(",")) for i in range(len(y_test))])

sess = tf.Session()

sess.run(tf.global_variables_initializer())

for i in range(10):
    print(i+1, "/ 10")
    for j in range(10000):
        sess.run(update, feed_dict={x: data_x, y: data_y})

for j in range(len(X_test)):
    print('Prediction for: "', X_test[j], ': "', sess.run(y_, feed_dict={x: [convert2vec(X_test[j])]}), "Prediction is:"
          , sess.run(tf.equal(tf.argmax(y_, 1), tf.argmax(convert2tensor(test_y, j))), feed_dict={x: [convert2vec(X_test[j])]}))

