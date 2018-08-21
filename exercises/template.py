
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

    pass

def define_model():
    """Creates a Keras model representing some architecture
    for a neural network. In these examples, the architectures will
    always be a sequential set of layers. The specific structure of
    each layer is determined by the class used from keras.layers.
    
    Returns:
        Model - A Keras neural network model
    """

    pass

def define_optimization(model):
    """Compiles a Keras model be specifying how to evaluate how well
    a model performs and what algorithm to use to improve the model.
    
    Arguments:
        model {Model} -- The Keras model to compile
    """

    pass

def train_model(model, training_data):
    """Fine tunes the parameters of a neural network so that it performs
    well on the specified training data set.
    
    Arguments:
        model {Model} -- The Keras model to train
        training_data {tuple[array]} -- The features and results for each example
    """

    pass

def test_model(model, testing_data):
    """Evalutes the model on a test data set, printing the trained loss/cost 
    and accuracy of the model on the test set.
    
    Arguments:
        model {Model} -- The Keras model to test
        testing_data {tuple[array]} -- The features and results for each example
    """

    pass

def apply_model(model, xdata):
    """Prints the predicted result for each feature set in the provided
    data set, based on the provided neural network.
    
    Arguments:
        model {Model} -- The Keras model used to make predictions
        data {array} -- The features to make predictions for
    """
    pass

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
    