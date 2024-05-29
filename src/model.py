
from __future__ import annotations

import pathlib
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Float: TypeAlias = np.float64
FloatArray: TypeAlias = npt.NDArray[Float]

DATA_FILE_PATH: str = str(pathlib.Path().resolve()) + "/src/data.txt"

NUM_TRAINING: int = 400  # First ~80% of data used for training, the rest used for model evaluation
POLY_DEGREE: int = 3  # The degree of the polynomial of the regression model
REG_PARAM: Float = np.float64(0.5)  # Controls how much the model is punished for higher weights
LEARNING_RATE: Float = np.float64(0.1)  # Determines how much the weights and bias are updated with each iteration
ITERATIONS: int = 10000


def read_data(file_name: str) -> tuple[FloatArray, FloatArray]:
    """
    ### Description
    Read training examples from the specified file.

    ### Parameters
    `file_name: str` - The path of the file from which to read the data

    ### Return Values
    `feature_matrix: FloatArray` - A 2D array containing the input features for training the model. Each row should
    represent an individual training example and each column should contain the values for a specific feature.

    `target_vector: FloatArray` - A 1D array containing the target values for each corresponding training example.
    """

    data_matrix = np.loadtxt(file_name)
    np.random.shuffle(data_matrix)  # Shuffle rows around to mitigate patterns in the training versus testing data
    return data_matrix[:, :-1], data_matrix[:, -1]


def map_features(feature_matrix: FloatArray, degree: int) -> FloatArray:
    """
    ### Description
    Map a set of linear features to polynomial ones.

    ### Parameters
    `feature_matrix: FloatArray` - A 2D array containing the input features to be mapped. Each row should represent an
    individual training example and each column should contain the values for a specific feature.

    `degree: int` - The degree of the resulting polynomial; the highest power to map each feature to.

    ### Return Values
    `mapped_features: FloatArray` - A 2D array containing the newly mapped polynomial features. Each column in
    `feature_matrix` is rasied to each power between 1 and `degree` inclusive. The resulting columns are concatenated
    together to form the `mapped_features` matrix.
    """

    columns: list[FloatArray] = []
    for j in range(feature_matrix.shape[1]):
        for pow in range(1, degree + 1):
            columns.append(feature_matrix[:, j] ** pow)
    return np.column_stack(columns)


def zscore_normalize(feature_matrix: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    ### Description
    Perform z-score normalization.

    ### Parameters
    `feature_matrix: FloatArray` - A 2D array containing the input features to be normalized. Each row should represent
    an individual training example and each column should contain the values for a specific feature.

    ### Return Values
    `norm_matrix: FloatArray` - A 2D array containing the normalized values from `feature_matrix`.

    `mean: FloatArray` - A 1D array containing the mean of each column from `feature_matrix`.

    `std_dev: FloatArray` - A 1D array containing the standard deviation of each column from `feature_matrix`.
    """

    mean: FloatArray = np.mean(feature_matrix, axis=0)
    std_dev: FloatArray = np.std(feature_matrix, axis=0)
    return ((feature_matrix - mean) / std_dev), mean, std_dev


def model_prediction(example: FloatArray, weights: FloatArray, bias: Float) -> Float:
    """
    ### Description
    The model's prediction of the output for a given input example.

    ### Parameters
    `example: FloatArray` - A 1D array containing the values for each feature of a single example.

    `weights: FloatArray` - A 1D array containing parameters for the model to use for each corresponding feature.

    `bias: Float` - Another parameter of the model; represents the model's prediction when all input features are zero.

    ### Return Values
    `prediction: Float` - The model's estimated target value.
    """

    return np.dot(example, weights) + bias


def compute_cost(feature_matrix: FloatArray, target_vector: FloatArray, weights: FloatArray, bias: Float, reg_param: Float) -> Float:
    """
    ### Description
    Compute the mean squared-error cost function of the model.

    ### Parameters
    `feature_matrix: FloatArray` - A 2D array containing input features. Each row should represent an individual training
    example and each column should contain the values for a specific feature.

    `target_vector: FloatArray` - A 1D array containing the target values for each corresponding training example.

    `weights: FloatArray` - A 1D array containing parameters for the model to use for each corresponding feature.

    `bias: Float` - Another parameter of the model; represents the model's prediction when all input features are zero.

    `reg_param: Float` - The regularization parameter determining how much the model is punished for higher weights;
    used to mitigate overfitting.

    ### Return Values
    `cost: Float` - A value representing how closely the model's predictions match the expected target values across the
    entire input dataset.
    """

    num_examples: int = feature_matrix.shape[0]
    cost: Float = np.float64(0.0)
    for i in range(num_examples):
        cost += (model_prediction(feature_matrix[i], weights, bias) - target_vector[i]) ** 2
    cost /= 2 * num_examples
    cost += (reg_param / (2 * num_examples)) * np.sum(weights ** 2)  # Add regularization term
    return cost


def compute_gradient(feature_matrix: FloatArray, target_vector: FloatArray, weights: FloatArray, bias: Float, reg_param: Float) -> tuple[FloatArray, Float]:
    """
    ### Description
    Compute the gradient of the cost function, the direction of greatest decrease.

    ### Parameters
    `feature_matrix: FloatArray` - A 2D array containing input features. Each row should represent an individual training
    example and each column should contain the values for a specific feature.

    `target_vector: FloatArray` - A 1D array containing the target values for each corresponding training example.

    `weights: FloatArray` - A 1D array containing parameters for the model to use for each corresponding feature.

    `bias: Float` - Another parameter of the model; represents the model's prediction when all input features are zero.

    `reg_param: Float` - The regularization parameter determining how much the model is punished for higher weights;
    used to mitigate overfitting.

    ### Return Values
    `weights_gradients: FloatArray` - A 1D array containing the partial derivatives of the cost function with respect to
    each weight.

    `bias_gradient: Float` - The partial derivative of the cost function with respect to the bias.
    """

    num_examples: int
    num_features: int
    num_examples, num_features = feature_matrix.shape
    weights_gradients: FloatArray = np.zeros(num_features)
    bias_gradient: Float = np.float64(0.0)
    for i in range(num_examples):
        loss: Float = model_prediction(feature_matrix[i], weights, bias) - target_vector[i]
        weights_gradients += loss * feature_matrix[i]
        bias_gradient += loss
    weights_gradients /= num_examples
    weights_gradients += (reg_param / num_examples) * weights  # Add regularization term
    bias_gradient /= num_examples
    return weights_gradients, bias_gradient


def gradient_descent(feature_matrix: FloatArray, target_vector: FloatArray, weights: FloatArray, bias: Float, reg_param: Float, learning_rate: Float, iterations: int) -> tuple[FloatArray, Float]:
    """
    ### Description
    Perform batch gradient descent to achieve optimal parameters for the model.

    ### Parameters
    `feature_matrix: FloatArray` - A 2D array containing input features. Each row should represent an individual training
    example and each column should contain the values for a specific feature.

    `target_vector: FloatArray` - A 1D array containing the target values for each corresponding training example.

    `weights: FloatArray` - A 1D array containing parameters for the model to use for each corresponding feature.

    `bias: Float` - Another parameter of the model; represents the model's prediction when all input features are zero.

    `reg_param: Float` - The regularization parameter determining how much the model is punished for higher weights;
    used to mitigate overfitting.

    `learning_rate: Float` - Controls how quickly the model descends towards local the minima of the cost function.

    `iterations: int` - The number of times for which to run a single loop of gradient descent.

    ### Return Values
    `final_weights: FloatArray` - A 1D array containing the "optimal" weights found by the gradient descent process.

    `final_bias: Float` - The "optimal" bias found by the gradient descent process.
    """

    for iter in range(1, iterations + 1):
        weights_gradients: FloatArray
        bias_gradient: Float
        weights_gradients, bias_gradient = compute_gradient(feature_matrix, target_vector, weights, bias, reg_param)
        weights -= learning_rate * weights_gradients
        bias -= learning_rate * bias_gradient
        if (iter % (iterations // 10) == 0):
            print(f"Cost at iteration {iter}: {compute_cost(feature_matrix, target_vector, weights, bias, reg_param)}")
    return weights, bias


def main() -> None:
    """
    ### Description
    Train and test the multivariate polynomial regression model.

    ### Parameters
    None

    ### Return Values
    None
    """

    # Read data and preprocess
    feature_matrix: FloatArray
    target_vector: FloatArray
    feature_matrix, target_vector = read_data(DATA_FILE_PATH)

    feature_matrix = map_features(feature_matrix, POLY_DEGREE)
    feature_matrix, _, _ = zscore_normalize(feature_matrix)

    training_features: FloatArray = feature_matrix[:NUM_TRAINING, :]
    training_targets: FloatArray = target_vector[:NUM_TRAINING]
    testing_features: FloatArray = feature_matrix[NUM_TRAINING:, :]
    testing_targets: FloatArray = target_vector[NUM_TRAINING:]

    # Train the model
    weights: FloatArray = np.zeros(training_features.shape[1])
    bias: Float = np.float64(0.0)
    weights, bias = gradient_descent(training_features, training_targets, weights, bias, REG_PARAM, LEARNING_RATE, ITERATIONS)

    # Print final parameters and cost
    print()
    print("Final weights:")
    print(weights)
    print()
    print("Final bias")
    print(bias)
    print()
    print("Final cost")
    print(compute_cost(training_features, training_targets, weights, bias, REG_PARAM))
    print()

    # Perform simple model evaluation, compare predictions to expected values
    training_error: Float = np.float64(0.0)
    for i in range(training_features.shape[0]):
        training_error += abs((model_prediction(training_features[i], weights, bias) - training_targets[i]) / training_targets[i]) * 100
    print(f"Average percent error on training data: {training_error / training_features.shape[0]}%")
    testing_error: Float = np.float64(0.0)
    for i in range(testing_features.shape[0]):
        testing_error += abs((model_prediction(testing_features[i], weights, bias) - testing_targets[i]) / testing_targets[i]) * 100
    print(f"Average percent error on testing data: {testing_error / testing_features.shape[0]}%")
    print(f"Average percent error on all data: {(training_error + testing_error) / feature_matrix.shape[0]}%")


if __name__ == "__main__":
    main()
