from numpy import argmax
from sklearn.utils import shuffle
from tensorflow.keras.datasets import mnist
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.python.keras.layers import Flatten


# one conventional layer and one Pooling layer and one hidden layer
def model1(X_train, Y_train, X_test, Y_test):
    global model
    X_train, Y_train = shuffle(X_train, Y_train, random_state=1)
    # prepare cross validation
    train_size = 0.75
    train_index = int(len(X_train) * train_size)
    trainX, trainY, validateX, validateY = X_train[0:train_index], Y_train[0:train_index], \
                                           X_train[train_index:], Y_train[train_index:]

    model = Sequential()
    # add conventional layer with filter /features 32
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    # out
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    fit = model.fit(trainX, trainY, epochs=10, validation_data=(validateX, validateY))
    print(fit)
    (testLoss, testAcc) = model.evaluate(X_test, Y_test)
    print("test loss : ", testLoss, "%", "\ntest accuracy: ", testAcc * 100, "%")
    predicate = model.predict(X_test)
    sum = 0
    Y_preficated = list()
    i = 0
    while i < len(predicate):
        Y_preficated.append(argmax(predicate[i]))
        i += 1
    i = 0
    while i < len(predicate):
        if Y_test[i] == Y_preficated[i]:
            sum += 1
        i += 1
    acc = sum / len(predicate)
    print('Accuracy of testing', acc * 100, "%")


def model2(X_train, Y_train, X_test, Y_test):
    global model
    X_train, Y_train = shuffle(X_train, Y_train, random_state=1)
    # prepare cross validation
    train_size = 0.75
    train_index = int(len(X_train) * train_size)
    trainX, trainY, validateX, validateY = X_train[0:train_index], Y_train[0:train_index], \
                                           X_train[train_index:], Y_train[train_index:]

    model = Sequential()
    # add conventional layer with filter /features 64
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X_train[0].shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # out
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    fit = model.fit(trainX, trainY, epochs=10, validation_data=(validateX, validateY))
    print(fit)
    (testLoss, testAcc) = model.evaluate(X_test, Y_test)
    print("test loss : ", testLoss, "%", "\ntest accuracy: ", testAcc * 100, "%")
    predicate = model.predict(X_test)
    sum = 0
    Y_preficated = list()
    i = 0
    while (i < len(predicate)):
        Y_preficated.append(argmax(predicate[i]))
        i += 1
    i = 0
    while (i < len(predicate)):
        if (Y_test[i] == Y_preficated[i]):
            sum += 1
        i += 1
    acc = sum / len((predicate))
    print('Accuracy of testing', acc * 100, "%")


def model3(X_train, Y_train, X_test, Y_test):
    global model
    X_train, Y_train = shuffle(X_train, Y_train, random_state=1)
    # prepare cross validation
    train_size = 0.75
    train_index = int(len(X_train) * train_size)
    trainX, trainY, validateX, validateY = X_train[0:train_index], Y_train[0:train_index], \
                                           X_train[train_index:], Y_train[train_index:]

    model = Sequential()
    # add conventional layer with filter /features 64
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X_train[0].shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # out
    model.add(Dense(32, activation='softmax'))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    fit = model.fit(trainX, trainY, epochs=10, validation_data=(validateX, validateY))
    print(fit)
    (testLoss, testAcc) = model.evaluate(X_test, Y_test)
    print("test loss : ", testLoss, "%", "\ntest accuracy: ", testAcc * 100, "%")
    predicate = model.predict(X_test)
    sum = 0
    Y_preficated = list()
    i = 0
    while (i < len(predicate)):
        Y_preficated.append(argmax(predicate[i]))
        i += 1
    i = 0
    while (i < len(predicate)):
        if (Y_test[i] == Y_preficated[i]):
            sum += 1
        i += 1
    acc = sum / len((predicate))
    print('Accuracy of testing', acc * 100, "%")


def model4(X_train, Y_train, X_test, Y_test):
    global model
    X_train, Y_train = shuffle(X_train, Y_train, random_state=1)
    # prepare cross validation
    train_size = 0.75
    train_index = int(len(X_train) * train_size)
    trainX, trainY, validateX, validateY = X_train[0:train_index], Y_train[0:train_index], \
                                           X_train[train_index:], Y_train[train_index:]

    model = Sequential()
    # add conventional layer with filter /features 64
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=X_train[0].shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X_train[0].shape))
    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=X_train[0].shape))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(128, activation='relu'))
    # out
    model.add(Dense(32, activation='softmax'))
    model.add(Dense(10, activation='softmax'))
    print(model.summary())
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    fit = model.fit(trainX, trainY, epochs=10, validation_data=(validateX, validateY))
    print(fit)
    (testLoss, testAcc) = model.evaluate(X_test, Y_test)
    print("test loss : ", testLoss, "%", "\ntest accuracy: ", testAcc * 100, "%")
    predicate = model.predict(X_test)
    sum = 0
    Y_preficated = list()
    i = 0
    while (i < len(predicate)):
        Y_preficated.append(argmax(predicate[i]))
        i += 1
    i = 0
    while (i < len(predicate)):
        if (Y_test[i] == Y_preficated[i]):
            sum += 1
        i += 1
    acc = sum / len((predicate))
    print('Accuracy of testing', acc * 100, "%")


if __name__ == '__main__':
    # load data set
    (X_train, Y_train), (X_test, Y_test) = mnist.load_data()
    # make matrix vector
    X_train = X_train.reshape((X_train.shape[0], 28, 28, 1))
    X_test = X_test.reshape((X_test.shape[0], 28, 28, 1))
    # make type float for normalization
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
    # divide by 255 because gray levels
    X_train = X_train / 255.0
    X_test = X_test / 255.0
    model1(X_train, Y_train, X_test, Y_test)
    model2(X_train, Y_train, X_test, Y_test)
    model3(X_train, Y_train, X_test, Y_test)
    model4(X_train, Y_train, X_test, Y_test)
