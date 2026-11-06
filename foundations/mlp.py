import numpy as np
from numpy.typing import NDArray
from typing import List


class Solution:
    def forward(self, x: NDArray[np.float64], weights: List[NDArray[np.float64]], biases: List[NDArray[np.float64]]) -> NDArray[np.float64]:
        # x: 1D input array
        # weights: list of 2D weight matrices
        # biases: list of 1D bias vectors
        # Apply ReLU after each hidden layer, no activation on output layer
        # return np.round(your_answer, 5)
        m = len(weights)
        x = np.array(x)

        for i in range(m-1):
            zi = x @ np.array(weights[i]) + np.array(biases[i])
            ai = np.maximum(0, zi)
            x = ai
        
        y_hat = x @ np.array(weights[m-1]) + np.array(biases[m-1])
        return np.round(y_hat, 5)

