"""
Logistic Regression implemented from scratch using:
- Sigmoid activation function
- Binary Cross-Entropy (Log Loss)
- Gradient Descent
"""

import numpy as np


class LogisticRegressionScratch:
    """
    Binary Logistic Regression built entirely from scratch with NumPy.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent updates.
    n_iterations : int
        Number of full passes over the training data.
    """

    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Activation
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        """Numerically stable sigmoid: σ(z) = 1 / (1 + e^{-z})."""
        # Clip to avoid overflow in exp
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------
    # Loss
    # ------------------------------------------------------------------
    @staticmethod
    def _binary_cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Binary Cross-Entropy (Log Loss):
            L = -1/N * Σ [ y*log(ŷ) + (1-y)*log(1-ŷ) ]
        """
        eps = 1e-15  # prevent log(0)
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(
            -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionScratch":
        """
        Train the model via gradient descent.

        Gradient of BCE w.r.t. weights:  dL/dw = (1/N) * Xᵀ(ŷ - y)
        Gradient of BCE w.r.t. bias:     dL/db = (1/N) * Σ(ŷ - y)
        """
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            # Forward pass
            linear_output = np.dot(X, self.weights) + self.bias
            y_pred = self._sigmoid(linear_output)

            # Compute and record loss
            loss = self._binary_cross_entropy(y, y_pred)
            self.loss_history.append(loss)

            # Gradients
            error = y_pred - y
            dw = np.dot(X.T, error) / n_samples
            db = np.mean(error)

            # Parameter update
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return raw sigmoid probability for each sample."""
        linear_output = np.dot(X, self.weights) + self.bias
        return self._sigmoid(linear_output)

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class labels (0 or 1)."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        """Fraction of correctly classified samples."""
        return float(np.mean(self.predict(X) == y))
