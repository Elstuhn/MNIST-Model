import tensorflow as tf
import numpy as np
from tensorflow.keras.datasets import mnist
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D, Dropout
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

(trainX, trainy), (testX, testy) = mnist.load_data()
trainX, testX = trainX / 255.0, testX / 255.0
trainX = np.expand_dims(trainX, -1)
testX = np.expand_dims(testX, -1)
earlystop = EarlyStopping(monitor = "val_accuracy", patience = 3)
model_save = ModelCheckpoint('CNNmodel.hdf5',
                             save_best_only=True)

model = Sequential()
model.add(Conv2D(32, kernel_size = (3, 3), activation = 'relu', input_shape = (28, 28, 1)))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Conv2D(64, kernel_size = (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Flatten())
model.add(Dropout(0.5))
model.add(Dense(10, activation = 'softmax'))
model.compile(loss="sparse_categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
history = model.fit(trainX, trainy, epochs=10000, validation_data = (testX, testy), callbacks = [earlystop, model_save])
#model.fit(trainX, trainy, batch_size=128, epochs=15, validation_split=0.1)
score = model.evaluate(testX, testy, verbose = 0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

def check_acc(model, X_test, y_test):
    incorrect = 0
    correct = 0
    for i in range(len(X_test)):
        prediction = model.predict(X_test[i].reshape(1, 28, 28, 1)).argmax()
        actual = y_test[i]
        if int(prediction) != int(actual):
            incorrect += 1
        else:
            correct += 1
            
    fig = plt.figure()
    plt.ylabel("Amount")
    plt.xlabel(f"Accuracy: {(correct/(correct+incorrect))*100}%")
    plt.bar(["correct", "incorrect"], [correct, incorrect])
    plt.show()
    print(f"Accuracy: {(correct/(correct+incorrect))*100}%")
    
check_acc(model, testX, testy)

def plothist(history, metric):
    plt.figure()
    if metric == "accuracy":
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('Model Accuracy')
        plt.ylabel("Accuracy")
    elif metric == "loss":
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title("Model Loss")
        plt.ylabel("Loss")
    plt.xlabel("Epoch")
    plt.legend(["Train", "Test"])
    plt.show()
    
plothist(history, "accuracy")
plothist(history, "loss")
