import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


mnist = tf.keras.datasets.mnist

(x_train, y_train),(x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
input_dim=x_train.shape[1]
print(input_dim)

class MnistModel(tf.keras.Model):

  def __init__(self, num_classes=10):
    super(MnistModel, self).__init__(name='MnistModel')
    self.num_classes = num_classes
    self.flatten = layers.Flatten(input_shape=(28,28))
    self.dense_1 = layers.Dense(128, activation='relu')
    self.dropout_1 = layers.Dropout(0.2)
    self.dense_2 = layers.Dense(num_classes, activation='softmax')

  def call(self, inputs):
    x = self.flatten(inputs)
    x = self.dense_1(x)
    x = self.dropout_1(x)
    return self.dense_2(x)

model = MnistModel(num_classes=10)


model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test,  y_test, verbose=2)