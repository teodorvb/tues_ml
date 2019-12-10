import matplotlib.pyplot as pl
import numpy as np
import tensorflow as tf
import math as m

def random_split(data, labels, split_point):
    data_label_pairs = list(zip(data, labels))
    np.random.shuffle(data_label_pairs)

    l_labels, d_labels = np.shape(labels);
    l_data, d_data = np.shape(data);

    assert (l_labels == l_data), "Data lenght different from labels length"
    
    train_label_pairs = data_label_pairs[:m.floor(split_point*l_data)]
    test_label_pairs = data_label_pairs[m.floor(split_point*l_data):]

    train_data = np.zeros((len(train_label_pairs), d_data));
    train_labels = np.zeros((len(train_label_pairs), d_labels));

    test_data = np.zeros((len(test_label_pairs), d_data));
    test_labels = np.zeros((len(test_label_pairs), d_labels));

    for i in range(len(train_label_pairs)):
        train_data[i, :] = train_label_pairs[i][0]
        train_labels[i, :] = train_label_pairs[i][1]

    for i in range(len(test_label_pairs)):
        test_data[i, :] = test_label_pairs[i][0]
        test_labels[i, :] = test_label_pairs[i][1]

    return (train_data, train_labels, test_data, test_labels);


def getAccuracy(model, data, labels):
    predicted = model.predict(data)

    same = 0
    for i in range(len(labels)):
        same += np.argmax(predicted[i]) == np.argmax(labels[i])
    return float(same)/float(len(data))

sigma = 4

dsize  = 200
positive = np.vstack(
    (np.random.normal(loc=1.0, scale=sigma, size=dsize),
     np.random.normal(loc=1.0, scale=sigma, size=dsize),
     np.random.normal(loc=1.0, scale=sigma, size=dsize),
     np.random.normal(loc=1.0, scale=sigma, size=dsize),
     np.random.normal(loc=1.0, scale=sigma, size=dsize),
     np.random.normal(loc=1.0, scale=sigma, size=dsize),
     np.random.normal(loc=1.0, scale=sigma, size=dsize),
     np.random.normal(loc=1.0, scale=sigma, size=dsize),
     np.random.normal(loc=1.0, scale=sigma, size=dsize),
     np.random.normal(loc=1.0, scale=sigma, size=dsize)),
    ).T

negative = np.vstack(
    (np.random.normal(loc=-1.0, scale=sigma, size=dsize),
     np.random.normal(loc=-1.0, scale=sigma, size=dsize),
     np.random.normal(loc=-1.0, scale=sigma, size=dsize),
     np.random.normal(loc=-1.0, scale=sigma, size=dsize),
     np.random.normal(loc=-1.0, scale=sigma, size=dsize),
     np.random.normal(loc=-1.0, scale=sigma, size=dsize),
     np.random.normal(loc=-1.0, scale=sigma, size=dsize),
     np.random.normal(loc=-1.0, scale=sigma, size=dsize),
     np.random.normal(loc=-1.0, scale=sigma, size=dsize),
     np.random.normal(loc=-1.0, scale=sigma, size=dsize)),
    ).T

data = np.vstack((positive, negative));
labels = np.vstack((
    np.array([[1, 0]]*len(positive)),
    np.array([[0, 1]]*len(negative))
))


train_accuracy = np.zeros(10)
test_accuracy = np.zeros(10);

train_data, train_labels, test_data, test_labels = random_split(data, labels, 0.7)

for i in range(10):
    print("Training network with ", (i+1), " hidden units");

    model = tf.keras.Sequential([
        tf.keras.layers.Dense((i+1), input_shape=(10,), activation="tanh", kernel_initializer=tf.keras.initializers.GlorotNormal(),),
        tf.keras.layers.Dense(2, activation="softmax")
    ])

    model.compile(optimizer='adam',
                  loss = 'categorical_crossentropy',
                  metric=['accuracy']);


    model.fit(train_data, train_labels, epochs=500, verbose=0)


    train_accuracy[i] = getAccuracy(model, train_data, train_labels)
    test_accuracy[i] = getAccuracy(model, test_data, test_labels)

pl.plot((np.array(range(10))+1) * 5, train_accuracy)
pl.plot((np.array(range(10))+1) * 5, test_accuracy)
pl.show()
