import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import optimizers
from tensorflow.keras.optimizers import Adam


(X_train,_),(X_test,_) = tf.keras.datasets.mnist.load_data()

samples = X_train.shape[0]
X_train = X_train.flatten().reshape(samples,784)/255.0
val_x = X_test.reshape(10000,784) / 255.0


autoencoder = Sequential()
autoencoder.add(Dense(512, activation='elu', input_shape=(784,)))
autoencoder.add(Dense(128,  activation='elu'))
autoencoder.add(Dense(10,    activation='linear', name="bottleneck"))
autoencoder.add(Dense(128,  activation='elu'))
autoencoder.add(Dense(512,  activation='elu'))
autoencoder.add(Dense(784,  activation='sigmoid'))
autoencoder.compile(loss='mean_squared_error', optimizer = Adam())
trained_model = autoencoder.fit(X_train, X_train, batch_size=1024, epochs=10, verbose=1, validation_data=(val_x, val_x))
encoder = Model(autoencoder.input, autoencoder.get_layer('bottleneck').output)


encoded_data = encoder.predict(X_train)  # bottleneck representation
decoded_output = autoencoder.predict(X_train)        # reconstruction
encoding_dim = 10

print ("Encoded Image: {}".format(encoder.predict(X_train[0].reshape(1,-1))[0]))
encoded_image = encoder.predict(X_train[0].reshape(1,-1))[0]


# return the decoder
encoded_input = Input(shape=(encoding_dim,))
decoder = autoencoder.layers[-3](encoded_input)
decoder = autoencoder.layers[-2](decoder)
decoder = autoencoder.layers[-1](decoder)
decoder = Model(encoded_input, decoder)

print ("DECODED IMAGE: {}".format(decoder.predict(encoded_image.reshape(1,-1))))


