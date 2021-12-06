from keras.models import load_model
from tensorflow.keras.datasets import mnist
from time import sleep
from matplotlib import image
import matplotlib.pyplot as plt
model = load_model('CNNmodel.hdf5')
(trainX, trainy), (testX, testy) = mnist.load_data()
trainX, testX = trainX / 255.0, testX / 255.0
plt.figure(figsize=(10,10))
plt.title("Picture Input")
plt.imshow(testX[143])
plt.show()
prediction = model.predict(testX[143].reshape(1, 28, 28, 1))
print(f"Prediction: {prediction.argmax()}")
print(f"Actual: {testy[143]}")


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
