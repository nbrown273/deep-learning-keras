from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import SGD
from numpy import array, asscalar

def define_model():
    model = Sequential()
    model.add(Dense(8, activation='tanh', input_dim=2))
    model.add(Dense(1, activation='sigmoid'))
    return model

def define_optimization(model):
    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

def train_model(model, training_data):
    model.fit(*training_data, batch_size=4, nb_epoch=1000)

def test_model(model, testing_data):
    print("Final Loss, Accuracy: " + str(model.evaluate(*testing_data)))

def apply_model(model, data):
    for x in data:
        y = asscalar(model.predict(array([x])))
        print("Model(%s) = %.6f ~ %d" % (str(x),y,round(y)))

if __name__=="__main__":
    # XOR logical operator
    xdata = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    ydata = array([[0], [1], [1], [0]])

    model = define_model()
    define_optimization(model)
    train_model(model, (xdata, ydata))
    test_model(model, (xdata, ydata))

    apply_model(model, xdata)
    