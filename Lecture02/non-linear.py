import pylab as pl
import numpy as np
import tensorflow as tf

x = np.linspace(-1, 1, 500);
y = x**2 + np.random.normal(size=500)/4

model = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="tanh"),
    tf.keras.layers.Dense(1, activation="linear")
])

model.compile(optimizer="adam",
              loss = tf.keras.losses.MeanSquaredError())

model.fit(x, y, epochs=200)
pl.plot(np.linspace(-1, 1, 500), model.predict(np.linspace(-1, 1, 500)), "r-")

pl.scatter(x, y)
pl.show()
