import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import *

X_train,X_test,Y_train,Y_test = np.load(
    './saved_np/lyrics_dataset_150_73006.npy',
    allow_pickle=True)
print(X_train.shape, Y_train.shape)
print(X_test.shape, Y_test.shape)

model = Sequential()
model.add(Embedding(73006, 700, input_length=150))      # 73006개의 단어집합을 700으로 차원 축소 / 입력 길이는 150
model.add(Conv1D(64, kernel_size=13, padding="same", activation="relu"))
model.add(MaxPooling1D(pool_size=1))

model.add(LSTM(128, activation="tanh", return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(64, activation="tanh", return_sequences=True))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(7, activation="softmax"))
print(model.summary())

model.compile(loss="categorical_crossentropy", optimizer="adam",
              metrics=["accuracy"])

fit_hist = model.fit(X_train, Y_train, batch_size=200, epochs=1, validation_data=(X_test,Y_test))

model.save("./models/lyrics_category_model_{}.h5".format(
    fit_hist.history["val_accuracy"][-1]))

