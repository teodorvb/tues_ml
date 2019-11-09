import pylab as pl
import numpy as np
import tensorflow as tf

dsize = 500
msize = 100
x = np.linspace(-1, 1, dsize)
y = x**2 + np.random.normal(size=dsize)/4

model = tf.keras.Sequential([
    tf.keras.layers.Dense(2, activation="tanh", kernel_initializer=tf.keras.initializers.GlorotNormal(), input_shape=(1,) ),
    tf.keras.layers.Dense(1, activation="linear", kernel_initializer=tf.keras.initializers.GlorotNormal())
])

model.compile(optimizer="adam",
              loss = tf.keras.losses.MeanSquaredError())

model.fit(x, y, epochs=1000)
pl.plot(np.linspace(-1, 1, msize), model.predict(np.linspace(-1, 1, msize)), "r-")

pl.scatter(x, y)
pl.show()
