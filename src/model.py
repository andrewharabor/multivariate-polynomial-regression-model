
from __future__ import annotations

import pathlib
from typing import TypeAlias

import numpy as np
import numpy.linalg as npla
import numpy.typing as npt

Float: TypeAlias = np.float64
FloatArray: TypeAlias = npt.NDArray[Float]

DATA_FILE_PATH: str = str(pathlib.Path().resolve()) + "/src/boston-house-prices.txt"

PERCENT_TRAINING: Float = np.float64(0.80)  # First ~80% of data used for training, the rest used for model evaluation
POLY_DEGREE: int = 3  # The degree of the polynomial of the regression model
REG_PARAM: Float = np.float64(0.5)  # Controls how much the model is punished for higher weights
LEARNING_RATE: Float = np.float64(0.1)  # Determines how much the weights and bias are updated with each iteration
ITERATIONS: int = 100_000


def read_data(file_name: str) -> tuple[FloatArray, FloatArray]:
    """
    ### Description
    Read training examples from the specified file.

    ### Parameters
    `file_name: str` - The path of the file from which to read the data

    ### Return Values
    `features_matrix: FloatArray` - A 2D array containing the input features for training the model. Each row should
    represent an individual training example and each column should contain the values for a specific feature.

    `target_vector: FloatArray` - A 1D array containing the target values for each corresponding training example.
    """

    data_matrix = np.loadtxt(file_name)
    np.random.shuffle(data_matrix)  # Shuffle rows around to mitigate patterns in the training versus testing data
    return data_matrix[:, :-1], data_matrix[:, -1]


def map_features(features_matrix: FloatArray, degree: int) -> FloatArray:
    """
    ### Description
    Map a set of linear features to polynomial ones.

    ### Parameters
    `features_matrix: FloatArray` - A 2D array containing the input features to be mapped. Each row should represent an
    individual training example and each column should contain the values for a specific feature.

    `degree: int` - The degree of the resulting polynomial; the highest power to map each feature to.

    ### Return Values
    `mapped_features: FloatArray` - A 2D array containing the newly mapped polynomial features. Each column in
    `features_matrix` is rasied to each power between 1 and `degree` inclusive. The resulting columns are concatenated
    together to form the `mapped_features` matrix.
    """

    columns: list[FloatArray] = []
    for j in range(features_matrix.shape[1]):
        for pow in range(1, degree + 1):
            columns.append(features_matrix[:, j] ** pow)
    return np.column_stack(columns)


def zscore_normalize(features_matrix: FloatArray) -> FloatArray:
    """
    ### Description
    Perform z-score normalization.

    ### Parameters
    `features_matrix: FloatArray` - A 2D array containing the input features to be normalized. Each row should represent
    an individual training example and each column should contain the values for a specific feature.

    ### Return Values
    `norm_matrix: FloatArray` - A 2D array containing the normalized values from `features_matrix`.
    """

    mean: FloatArray = np.mean(features_matrix, axis=0)
    std_dev: FloatArray = np.std(features_matrix, axis=0)
    return ((features_matrix - mean) / std_dev)


def model_prediction(features_matrix: FloatArray, weights: FloatArray, bias: Float) -> FloatArray:
    """
    ### Description
    The model's prediction of the output for all given examples.

    ### Parameters
    `features_matrix: FloatArray` - A 2D array containing input features. Each row should represent an individual training
    example and each column should contain the values for a specific feature.

    `weights: FloatArray` - A 1D array containing parameters for the model to use for each corresponding feature.

    `bias: Float` - Another parameter of the model; represents the model's prediction when all input features are zero.

    ### Return Values
    `prediction_vector: Float` - The model's estimated target values for each respective example.
    """

    return features_matrix @ weights + bias


def compute_cost(features_matrix: FloatArray, target_vector: FloatArray, weights: FloatArray, bias: Float, reg_param: Float) -> Float:
    """
    ### Description
    Compute the mean squared-error cost function of the model.

    ### Parameters
    `features_matrix: FloatArray` - A 2D array containing input features. Each row should represent an individual training
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

    loss_norm: Float = npla.norm(model_prediction(features_matrix, weights, bias) - target_vector) ** 2
    reg_norm: Float =  reg_param * (npla.norm(weights) ** 2)
    return (loss_norm + reg_norm) / (2 * features_matrix.shape[0])


def compute_gradient(features_matrix: FloatArray, target_vector: FloatArray, weights: FloatArray, bias: Float, reg_param: Float) -> tuple[FloatArray, Float]:
    """
    ### Description
    Compute the gradient of the cost function, the direction of greatest increase.

    ### Parameters
    `features_matrix: FloatArray` - A 2D array containing input features. Each row should represent an individual training
    example and each column should contain the values for a specific feature.

    `target_vector: FloatArray` - A 1D array containing the target values for each corresponding training example.

    `weights: FloatArray` - A 1D array containing parameters for the model to use for each corresponding feature.

    `bias: Float` - Another parameter of the model; represents the model's prediction when all input features are zero.

    `reg_param: Float` - The regularization parameter determining how much the model is punished for higher weights;
    used to mitigate overfitting.

    ### Return Values
    `weights_derivatives: FloatArray` - A 1D array containing the partial derivatives of the cost function with respect
    to each weight.

    `bias_derivative: Float` - The partial derivative of the cost function with respect to the bias.
    """

    weights_derivatives: FloatArray = (np.transpose(features_matrix) @ (model_prediction(features_matrix, weights, bias) - target_vector) + reg_param * weights) / features_matrix.shape[0]
    bias_derivative: Float = np.sum(model_prediction(features_matrix, weights, bias) - target_vector) / features_matrix.shape[0]
    return weights_derivatives, bias_derivative


def gradient_descent(features_matrix: FloatArray, target_vector: FloatArray, weights: FloatArray, bias: Float, reg_param: Float, learning_rate: Float, iterations: int) -> tuple[FloatArray, Float]:
    """
    ### Description
    Perform batch gradient descent to achieve optimal parameters for the model.

    ### Parameters
    `features_matrix: FloatArray` - A 2D array containing input features. Each row should represent an individual training
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
        weights_derivatives: FloatArray
        bias_derivative: Float
        weights_derivatives, bias_derivative = compute_gradient(features_matrix, target_vector, weights, bias, reg_param)
        weights -= learning_rate * weights_derivatives
        bias -= learning_rate * bias_derivative
        if (iter % (iterations // 10) == 0):
            print(f"Cost at iteration {iter}: {compute_cost(features_matrix, target_vector, weights, bias, reg_param)}")
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
    features_matrix: FloatArray
    target_vector: FloatArray
    features_matrix, target_vector = read_data(DATA_FILE_PATH)
    features_matrix = zscore_normalize(map_features(features_matrix, POLY_DEGREE))

    # Split training versus testing data
    num_training: int = round(PERCENT_TRAINING * features_matrix.shape[0])
    training_features: FloatArray = features_matrix[:num_training, :]
    training_targets: FloatArray = target_vector[:num_training]
    testing_features: FloatArray = features_matrix[num_training:, :]
    testing_targets: FloatArray = target_vector[num_training:]

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

    # Perform simple model evaluation
    training_mape: Float = npla.norm((model_prediction(training_features, weights, bias) - training_targets) / training_targets, ord=1) / training_features.shape[0]
    testing_mape: Float = npla.norm((model_prediction(testing_features, weights, bias) - testing_targets) / testing_targets, ord=1) / testing_features.shape[0]
    print(f"Mean absolute percent error on training data: {training_mape * 100}%")
    print(f"Mean absolute percent error on testing data: {testing_mape * 100}%")


if __name__ == "__main__":
    main()
