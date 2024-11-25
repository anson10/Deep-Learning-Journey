import tensorflow as tf 

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype("float32") / 255.0
X_test = X_test.astype("float32") / 255.0


model  = tf.keras.Sequential()
model.add(tf.keras.Input(shape=(None, 28)))
model.add(
    tf.keras.layers.GRU(256, return_sequences=True),
)

model.add(tf.keras.layers.GRU(256, activation='relu'))
model.add(tf.keras.layers.Dense(10))

# print(model.summary())

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics = ["accuracy"]
)

model.fit(X_train, y_train, batch_size=10, verbose=2)
model.evaluate(X_test, y_test, batch_size=64, verbose=2)