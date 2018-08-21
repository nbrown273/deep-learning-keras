from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import SGD
from numpy import array, asscalar

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

    # CHANGE THE CODE HERE: What should the dataset be for XOR?
    xdata = array([[0, 0], [0, 1], [1, 0], [1, 1]])
    ydata = array([[0], [1], [1], [1]])
    return ((xdata, ydata), (xdata, ydata), xdata)

def define_model():
    """Creates a Keras model representing some architecture
    for a neural network. In these examples, the architectures will
    always be a sequential set of layers. The specific structure of
    each layer is determined by the class used from keras.layers.
    
    Returns:
        Model - A Keras neural network model
    """

    # CHANGE THE CODE HERE: Is one layer enough? How many nodes should be in the layer(s)?
    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=2))
    return model

def define_optimization(model):
    """Compiles a Keras model be specifying how to evaluate how well
    a model performs and what algorithm to use to improve the model.
    
    Arguments:
        model {Model} -- The Keras model to compile
    """

    model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.1), metrics=['accuracy'])

def train_model(model, training_data):
    """Fine tunes the parameters of a neural network so that it performs
    well on the specified training data set.
    
    Arguments:
        model {Model} -- The Keras model to train
        training_data {tuple[array]} -- The features and results for each example
    """

    model.fit(*training_data, batch_size=4, nb_epoch=1000)

def test_model(model, testing_data):
    """Evalutes the model on a test data set, printing the trained loss/cost 
    and accuracy of the model on the test set.
    
    Arguments:
        model {Model} -- The Keras model to test
        testing_data {tuple[array]} -- The features and results for each example
    """

    print("Final Loss, Accuracy: " + str(model.evaluate(*testing_data)))

def apply_model(model, xdata):
    """Prints the predicted result for each feature set in the provided
    data set, based on the provided neural network.
    
    Arguments:
        model {Model} -- The Keras model used to make predictions
        data {array} -- The features to make predictions for
    """
    y = model.predict(xdata)
    for x, y in zip(xdata, y):
        y = asscalar(y)
        print("Model(%s) = %.6f ~ %d" % (str(x),y,round(y)))

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
    