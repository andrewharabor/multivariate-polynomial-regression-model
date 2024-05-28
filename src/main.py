
from __future__ import annotations

import pathlib
from typing import TypeAlias

import numpy as np
import numpy.typing as npt

Float: TypeAlias = np.float64
FloatArray: TypeAlias = npt.NDArray[Float]

DATA_PATH: str = str(pathlib.Path().absolute()) + "/src/data.txt"

EXAMPLES_FOR_TRAINING: int = 400  # First ~80% of data used for training, the rest used for model evaluation
POLY_DEGREE: int = 3  # The degree of the polynomial for the polynomial regression model
REG_PARAM: Float = np.float64(1.0)  # Controls how much the model is punished for higher weights
LEARNING_RATE: Float = np.float64(0.1)  # Determines how much the weights and bias are updated with each iteration
ITERATIONS: int = 10000


def read_data(file_name: str) -> tuple[FloatArray, FloatArray]:
    """
    ### Description
    Read training examples from the specified file. Each row should represent a specific training example and each column
    should represent a specific feature. The last column should be the expected target value for each example.

    ### Parameters
    `file_name: str` - The path of the file from which to read the data

    ### Return Values
    `input_matrix: FloatArray` - A 2D array containing the input features to be mapped. Each row should represent an
    individual training example and each column should contain the values for a specific feature.

    `target_vector: FloatArray` - A 1D array countaining the target values for each corresponding training example.
    """

    data_matrix = np.loadtxt(file_name)
    np.random.shuffle(data_matrix)  # Shuffle rows around to mitigate patterns in the training versus testing data
    input_matrix: FloatArray = data_matrix[:, :-1]
    target_vector: FloatArray = data_matrix[:, -1]  # Extract last column
    return input_matrix, target_vector


def map_features(input_matrix: FloatArray, degree: int) -> FloatArray:
    """
    ### Description
    Map a set of linear features to polynomial ones.

    ### Parameters
    `input_matrix: FloatArray` - A 2D array containing the input features to be mapped. Each row should represent an
    individual training example and each column should contain the values for a specific feature.

    `degree: int` - The degree of the resulting polynomial features; the highest power to map each feature to.

    ### Return Values
    `mapped_inputs: FloatArray` - A 2D array with the same number of rows as `input_matrix` but `degree`
    times as many columns. Each column in `input_matrix` is rasied to each power between 1 and `degree` inclusive. The
    resulting columns are concatenated together to form the `mapped_inputs` matrix.
    """

    columns: list[FloatArray] = []
    for j in range(input_matrix.shape[1]):
        for pow in range(1, degree + 1):
            columns.append(input_matrix[:, j] ** pow)
    return np.column_stack(columns)


def zscore_normalize(input_matrix: FloatArray) -> tuple[FloatArray, FloatArray, FloatArray]:
    """
    ### Description
    Perform z-score normalization.

    ### Parameters
    `input_matrix: FloatArray` - A 2D array containing the input features to be normalized. Each row should represent an
    individual training example and each column should contain the values for a specific feature.

    ### Return Values
    `norm_matrix: FloatArray` - A 2D array containing the normalized values from `input_matrix`.

    `mean: FloatArray` - A 1D array containing the mean of each column from `input_matrix`.

    `std_dev: FloatArray` - A 1D array containing the standard deviation of each column from `input_matrix`.
    """

    mean: FloatArray = np.mean(input_matrix, axis=0)
    std_dev: FloatArray = np.std(input_matrix, axis=0)
    return (input_matrix - mean) / std_dev, mean, std_dev



def model_prediction(input: FloatArray, weights: FloatArray, bias: Float) -> Float:
    """
    ### Description
    Have the model predict the output for a given input example.

    ### Parameters
    `input: FloatArray` - A 1D array containing the values for each feature of a single example.

    `weights: FloatArray` - A 1D array containing weights for the model to use; should have the same number of weights
    as the number of features.

    `bias: Float` - Another parameter of the model; represents the model's prediction when all input features are zero.

    ### Return Values
    `prediction: Float` - The model's estimated target value.
    """

    return np.dot(input, weights) + bias


def compute_cost(input_matrix: FloatArray, target_vector: FloatArray, weights: FloatArray, bias: Float, reg_param: Float) -> Float:
    """
    ### Description
    Compute the mean squared-error cost function of the model.

    ### Parameters
    `input_matrix: FloatArray` - A 2D array containing input features. Each row should represent an individual training
    example and each column should contain the values for a specific feature.

    `target_vector: FloatArray` - A 1D array countaining the target values for each corresponding training example.

    `weights: FloatArray` - A 1D array containing weights for the model to use; should have the same number of weights
    as the number of features.

    `bias: Float` - Another parameter of the model; represents the model's prediction when all input features are zero.

    `reg_param: Float` - The regularization parameter determining how much the model is punished for higher weights;
    used to mitigate overfitting.

    ### Return Values
    `cost: Float` - A value representing how closely the model's predictions match the expected target values.
    """

    num_examples: int = input_matrix.shape[0]
    cost: Float = np.float64(0.0)
    for i in range(num_examples):
        cost += (model_prediction(input_matrix[i], weights, bias) - target_vector[i]) ** 2
    cost /= (2 * num_examples)
    cost += (reg_param / (2 * num_examples)) * np.sum(weights ** 2)  # Add regularization term
    return cost


def compute_gradient(input_matrix: FloatArray, target_vector: FloatArray, weights: FloatArray, bias: Float, reg_param: Float) -> tuple[FloatArray, Float]:
    """
    ### Description
    Compute the gradient of the cost function, the direction of greatest decrease.

    ### Parameters
    `input_matrix: FloatArray` - A 2D array containing input features. Each row should represent an individual training
    example and each column should contain the values for a specific feature.

    `target_vector: FloatArray` - A 1D array countaining the target values for each corresponding training example.

    `weights: FloatArray` - A 1D array containing weights for the model to use; should have the same number of weights
    as the number of features.

    `bias: Float` - Another parameter of the model; represents the model's prediction when all input features are zero.

    `reg_param: Float` - The regularization parameter determining how much the model is punished for higher weights;
    used to mitigate overfitting.

    ### Return Values
    `weights_gradients: FloatArray` - A 1D array of the same size as the `weights` array containing the partial
    derivative of the cost function with respect to each weight.

    `bias_gradient: Float` - The partial derivative of the cost function with respect to the bias.
    """

    num_examples: int
    num_features: int
    num_examples, num_features = input_matrix.shape
    weights_gradients: FloatArray = np.zeros(num_features)
    bias_gradient: Float = np.float64(0.0)
    for i in range(num_examples):
        loss: Float = model_prediction(input_matrix[i], weights, bias) - target_vector[i]
        weights_gradients += loss * input_matrix[i]
        bias_gradient += loss
    weights_gradients /= num_examples
    bias_gradient /= num_examples
    weights_gradients += (reg_param / num_examples) * weights  # Add regularization term
    return weights_gradients, bias_gradient


def gradient_descent(input_matrix: FloatArray, target_vector: FloatArray, weights: FloatArray, bias: Float, reg_param: Float, learning_rate: Float, iterations: int) -> tuple[FloatArray, Float]:
    """
    ### Description
    Perform batch gradient descent to achieve optimal parameters for the model.

    ### Parameters
    `input_matrix: FloatArray` - A 2D array containing input features. Each row should represent an individual training
    example and each column should contain the values for a specific feature.

    `target_vector: FloatArray` - A 1D array countaining the target values for each corresponding training example.

    `weights: FloatArray` - A 1D array containing weights for the model to use; should have the same number of weights
    as the number of features.

    `bias: Float` - Another parameter of the model; represents the model's prediction when all input features are zero.

    `reg_param: Float` - The regularization parameter determining how much the model is punished for higher weights;
    used to mitigate overfitting.

    `learning_rate: Float` - Controls how quickly the model descends towards local minima; a high value can cause the
    model not to converge.

    `iterations: int` - The number of iterations for which to run gradient descent.

    ### Return Values
    `final_weights: FloatArray` - A 1D array containing the "optimal" weights found by the gradient descent process.

    `final_bias: Float` - The "optimal" bias found by the gradient descent process.
    """

    for iter in range(1, iterations + 1):
        weights_gradients: FloatArray
        bias_gradient: Float
        weights_gradients, bias_gradient = compute_gradient(input_matrix, target_vector, weights, bias, reg_param)
        weights -= learning_rate * weights_gradients
        bias -= learning_rate * bias_gradient
        if (iter % (iterations // 10) == 0):
            print(f"Cost at iteration {iter}: {compute_cost(input_matrix, target_vector, weights, bias, REG_PARAM)}")

    return weights, bias


def main() -> None:
    """
    ### Description
    Train and test the multiple polynomial regression model.

    ### Parameters
    None

    ### Return Values
    None
    """

    # Read training examples
    input_matrix: FloatArray
    target_vector: FloatArray
    input_matrix, target_vector = read_data(DATA_PATH)

    # Map input features to polynomial ones, normalize data
    input_matrix = map_features(input_matrix, POLY_DEGREE)
    input_matrix, _, _ = zscore_normalize(input_matrix)

    # Train the model to achieve optimal parameters
    weights: FloatArray = np.zeros(input_matrix.shape[1])
    bias: Float = np.float64(0.0)
    weights, bias = gradient_descent(input_matrix[:EXAMPLES_FOR_TRAINING, :], target_vector[:EXAMPLES_FOR_TRAINING], weights, bias, REG_PARAM, LEARNING_RATE, ITERATIONS)

    # Print final parameters and cost
    print()
    print("Final weights:")
    print(weights)
    print()
    print("Final bias")
    print(bias)
    print()
    print("Final cost")
    print(compute_cost(input_matrix[:EXAMPLES_FOR_TRAINING], target_vector[:EXAMPLES_FOR_TRAINING], weights, bias, REG_PARAM))
    print()

    # Perform simple model evaluation on the last ~20% of the data reserved for testing
    total_error: Float = np.float64(0.0)
    test_error: Float = np.float64(0.0)
    for i in range(input_matrix.shape[0]):
        error: Float = abs(model_prediction(input_matrix[i], weights, bias) - target_vector[i]) / target_vector[i] * 100
        total_error += error
        if i >= EXAMPLES_FOR_TRAINING:
            test_error += error
    print(f"Average percent error on all data: {total_error / input_matrix.shape[0]}%")
    print(f"Average percent error on test data: {test_error / (input_matrix.shape[0] - EXAMPLES_FOR_TRAINING)}%")


if __name__ == "__main__":
    main()
