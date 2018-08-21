from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.optimizers import Adam
from keras.datasets import mnist
from keras.utils import np_utils
from numpy import array, argmax
from random import choices
import matplotlib.pyplot as plt

def load_data():
    """Collects and, if necessary, cleans datasets used for training
    the model, testing the model, and making predictions. The training
    and testing data set need to provide the independent features (x) 
    and the result (y) for each example in the data set. The prediction
    data set only needs to provide the independent features to make
    predictions for.
    
    Returns:
        tuple -- The three data sets ((training_x, training_y), (testing_x, testing_y), prediction_x)
    """

    train, test = mnist.load_data()
    train = reshape_data(*train)
    test = reshape_data(*test)
    predict = list(zip(*choices(list(zip(*test)), k=10)))
    return (train, test, predict)

def reshape_data(x, y):
    x2 = x.reshape(x.shape[0], 28, 28, 1).astype('float32') / 255
    y2 = np_utils.to_categorical(y)
    return (x2, y2)

def define_model():
    """Creates a Keras model representing some architecture
    for a neural network. In these examples, the architectures will
    always be a sequential set of layers. The specific structure of
    each layer is determined by the class used from keras.layers.
    
    Returns:
        Model - A Keras neural network model
    """

    model = Sequential()
    model.add(Conv2D(32, (5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(10, activation='softmax'))
    return model

def define_optimization(model):
    """Compiles a Keras model be specifying how to evaluate how well
    a model performs and what algorithm to use to improve the model.
    
    Arguments:
        model {Model} -- The Keras model to compile
    """

    model.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])

def train_model(model, training_data):
    """Fine tunes the parameters of a neural network so that it performs
    well on the specified training data set.
    
    Arguments:
        model {Model} -- The Keras model to train
        training_data {tuple[array]} -- The features and results for each example
    """

    model.fit(*training_data, batch_size=4, nb_epoch=2)

def test_model(model, testing_data):
    """Evalutes the model on a test data set, printing the trained loss/cost 
    and accuracy of the model on the test set.
    
    Arguments:
        model {Model} -- The Keras model to test
        testing_data {tuple[array]} -- The features and results for each example
    """

    print("Final Loss, Accuracy: " + str(model.evaluate(*testing_data)))

def apply_model(model, data):
    """Prints the predicted result for each feature set in the provided
    data set, based on the provided neural network.
    
    Arguments:
        model {Model} -- The Keras model used to make predictions
        data {array} -- The features to make predictions for
    """
    y = model.predict(array(data[0]))
    for i, x, y, ytrue in zip(range(len(y)), data[0], y, data[1]):
        y = argmax(y)
        ytrue = argmax(ytrue)
        print("Model(%d) = %.6e should be %.6e" % (i, y, ytrue))
        plt.subplot(2, 5, i+1)
        plt.imshow(x.reshape(28, 28) * 255, cmap=plt.get_cmap('gray'))
    plt.show()

def main():
    """The main routine for working with a neural network. While presenting
    sequentially here, this is usually an interactive, iterative process
    that depends heavily on the results of investigation along the way.
    """

    train_data, test_data, to_predict = load_data()
    model = define_model()
    define_optimization(model)
    train_model(model, train_data)
    test_model(model, test_data)
    apply_model(model, to_predict)

if __name__=="__main__":
    main()
    