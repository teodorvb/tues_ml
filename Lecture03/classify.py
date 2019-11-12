import matplotlib.pyplot as pl
import numpy as np
import tensorflow as tf
positive = np.vstack((
    np.concatenate(
        (np.random.normal(loc=1.0, scale=0.5, size=200),
         np.random.normal(loc=-1.0, scale=0.5, size=200))
    ),
    np.concatenate(
        (np.random.normal(loc=1.0, scale=0.5, size=200),
         np.random.normal(loc=-1.0, scale=0.5, size=200))
    )
)).T

print(positive)
negative = np.vstack((
    np.concatenate(
        (np.random.normal(loc=1.0, scale=0.5, size=200),
         np.random.normal(loc=-1.0, scale=0.5, size=200))
    ),
    np.concatenate(
        (np.random.normal(loc=-1.0, scale=0.5, size=200),
         np.random.normal(loc=1.0, scale=0.5, size=200))
    )
)).T
data = np.vstack((positive, negative))
labels = np.vstack(([[1, 0]]*400, [[0, 1]]*400));

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, input_shape=(2,), activation="tanh", kernel_initializer=tf.keras.initializers.GlorotNormal(),),
    tf.keras.layers.Dense(2, activation="softmax")
])

model.compile(optimizer='adam',
              loss = 'categorical_crossentropy',
              metric=['accuracy']);


model.fit(data, labels, epochs=100)

predicted = model.predict(data)

same = 0
for i in range(len(labels)):
    same += np.argmax(predicted[i]) == np.argmax(labels[i])

print("Accuracy: ", float(same)/float(len(labels)))

pl.scatter(positive[:,0], positive[:, 1])
pl.scatter(negative[:,0], negative[:, 1])
pl.show()
