import matplotlib.pyplot as pl
import numpy as np
import tensorflow as tf
import math as m

def getAccuracy(model, data, labels):
    predicted = model.predict(data)

    same = 0
    for i in range(len(labels)):
        same += np.argmax(predicted[i]) == np.argmax(labels[i])
    return float(same)/float(len(data))

dsize  = 500
positive = np.vstack((
    np.concatenate(
        (np.random.normal(loc=1.0, scale=0.8, size=dsize),
         np.random.normal(loc=-1.0, scale=0.8, size=dsize))
    ),
    np.concatenate(
        (np.random.normal(loc=1.0, scale=0.8, size=dsize),
         np.random.normal(loc=-1.0, scale=0.8, size=dsize))
    )
)).T

negative = np.vstack((
    np.concatenate(
        (np.random.normal(loc=1.0, scale=0.8, size=dsize),
         np.random.normal(loc=-1.0, scale=0.8, size=dsize))
    ),
    np.concatenate(
        (np.random.normal(loc=-1.0, scale=0.8, size=dsize),
         np.random.normal(loc=1.0, scale=0.8, size=dsize))
    )
)).T

data_label_pairs = list(zip(np.vstack((positive, negative)), np.vstack(([[1, 0]]*400, [[0, 1]]*400))))
np.random.shuffle(data_label_pairs)

train_label_pairs = data_label_pairs[:m.floor(0.7*len(data_label_pairs))]
test_label_pairs = data_label_pairs[m.floor(0.7*len(data_label_pairs)):]

train_data = np.zeros((len(train_label_pairs), 2));
train_labels = np.zeros((len(train_label_pairs), 2));

test_data = np.zeros((len(test_label_pairs), 2));
test_labels = np.zeros((len(test_label_pairs), 2));

for i in range(len(train_label_pairs)):
    train_data[i, :] = train_label_pairs[i][0]
    train_labels[i, :] = train_label_pairs[i][1]

for i in range(len(test_label_pairs)):
    test_data[i, :] = test_label_pairs[i][0]
    test_labels[i, :] = test_label_pairs[i][1]


train_accuracy = np.zeros(10)
test_accuracy = np.zeros(10);

for i in range(10):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense((i+1)*10, input_shape=(2,), activation="tanh", kernel_initializer=tf.keras.initializers.GlorotNormal(),),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metric=['accuracy']);


    model.fit(train_data, train_labels, epochs=1000, verbose=0)


    train_accuracy[i] = getAccuracy(model, train_data, train_labels)
    test_accuracy[i] = getAccuracy(model, test_data, test_labels)

pl.plot((np.array(range(10))+1) * 5, train_accuracy)
pl.plot((np.array(range(10))+1) * 5, test_accuracy)
pl.show()
