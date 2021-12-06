import tensorflow as tf
from tensorflow.keras.datasets import mnist
from keras.layers import Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential, load_model
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV
from keras.callbacks import ModelCheckpoint, EarlyStopping
import matplotlib.pyplot as plt

(trainX, trainy), (testX, testy) = mnist.load_data()
trainX, testX = trainX / 255.0, testX / 255.0

earlystop = EarlyStopping(monitor = "val_accuracy", patience = 3)
"""Randomized Search To Find Best Parameters"""
"""
def create_model(nn):
    model = Sequential()
    model.add(Flatten())
    model.add(Dense(nn, activation = 'relu'))
    model.add(Dropout(0.2,))
    model.add(Dense(10))
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])
    return model

model = KerasClassifier(build_fn=create_model, epochs=6, batch_size=16)
params = dict(
    nn = range(1, 500))

random_search = RandomizedSearchCV(model, param_distributions=params, cv=3)
random_search_results = random_search.fit(trainX, trainy, callbacks = [earlystop], validation_data = [testX, testy])
print(random_search_results.best_score_) 
print(random_search_results.best_params_)

"""
model = Sequential()
model.add(Flatten(input_shape = (28, 28)))
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation = 'softmax'))
model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])


model_save = ModelCheckpoint('best_model.hdf5',
                             save_best_only=True)
history = model.fit(trainX, trainy, epochs = 10000, callbacks = [model_save, earlystop], validation_data = [testX, testy])

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

model.evaluate(testX, testy, verbose = 2)
