from keras.models import Sequential
from keras.layers import Dense, LeakyReLU
from keras.optimizers import Adam
from keras.datasets import boston_housing
from keras.metrics import mae
from numpy import array, asscalar
from random import choices

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

    train, test = boston_housing.load_data()
    predict = list(zip(*choices(list(zip(*test)), k=10)))
    return (train, test, predict)

def define_model():
    """Creates a Keras model representing some architecture
    for a neural network. In these examples, the architectures will
    always be a sequential set of layers. The specific structure of
    each layer is determined by the class used from keras.layers.
    
    Returns:
        Model - A Keras neural network model
    """

    model = Sequential()
    model.add(Dense(20, input_dim=13, activation=LeakyReLU(alpha=0.1)))
    model.add(Dense(10, activation=LeakyReLU(alpha=0.1)))
    model.add(Dense(1))
    return model

def define_optimization(model):
    """Compiles a Keras model be specifying how to evaluate how well
    a model performs and what algorithm to use to improve the model.
    
    Arguments:
        model {Model} -- The Keras model to compile
    """

    model.compile(loss='mean_squared_error', optimizer=Adam(), metrics=[mae])

def train_model(model, training_data):
    """Fine tunes the parameters of a neural network so that it performs
    well on the specified training data set.
    
    Arguments:
        model {Model} -- The Keras model to train
        training_data {tuple[array]} -- The features and results for each example
    """

    model.fit(*training_data, nb_epoch=1000)

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
    for x, y, ytrue in zip(data[0], y, data[1]):
        y = asscalar(y)
        print("Model(%s) = %.6e should be %.6e" % (str(x), y, ytrue))

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
    