"""
Support Vector Machine (SVM) Classifier implemented from scratch using:
- Hyperplane decision boundary via dot products
- Hinge Loss for optimization
- Gradient Descent for weight/bias updates
"""

import numpy as np


class SVMClassifierScratch:
    """
    Linear SVM for binary classification built entirely from scratch.

    The primal SVM objective (soft-margin):
        min  (1/2)||w||² + C * Σ max(0, 1 - yᵢ(w·xᵢ + b))

    Labels must be {0, 1}; internally converted to {-1, +1}.

    Parameters
    ----------
    learning_rate : float
        Step size for gradient descent.
    lambda_param : float
        Regularisation strength (1/C).
    n_iterations : int
        Number of gradient descent steps.
    """

    def __init__(
        self,
        learning_rate: float = 0.001,
        lambda_param: float = 0.01,
        n_iterations: int = 1000,
    ):
        self.learning_rate = learning_rate
        self.lambda_param = lambda_param
        self.n_iterations = n_iterations
        self.weights: np.ndarray | None = None
        self.bias: float = 0.0
        self.loss_history: list[float] = []

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _to_svm_labels(y: np.ndarray) -> np.ndarray:
        """Convert {0,1} labels to {-1,+1} for SVM math."""
        return np.where(y <= 0, -1, 1)

    def _hinge_loss(self, X: np.ndarray, y_svm: np.ndarray) -> float:
        """
        Hinge loss with L2 regularisation:
            L = (λ/2)||w||² + (1/N) Σ max(0, 1 - yᵢ(w·xᵢ + b))
        """
        margins = y_svm * (np.dot(X, self.weights) + self.bias)
        hinge = np.maximum(0, 1 - margins)
        reg = 0.5 * self.lambda_param * np.dot(self.weights, self.weights)
        return float(reg + np.mean(hinge))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "SVMClassifierScratch":
        """
        Train via sub-gradient descent on the hinge loss.

        Gradient rules per sample i:
          - If yᵢ(w·xᵢ + b) >= 1  (correctly classified, outside margin):
              dw = λ·w,   db = 0
          - Else (inside margin or misclassified):
              dw = λ·w - yᵢ·xᵢ,   db = -yᵢ
        """
        n_samples, n_features = X.shape
        y_svm = self._to_svm_labels(y)

        self.weights = np.zeros(n_features)
        self.bias = 0.0
        self.loss_history = []

        for _ in range(self.n_iterations):
            for idx, x_i in enumerate(X):
                condition = y_svm[idx] * (np.dot(x_i, self.weights) + self.bias) >= 1
                if condition:
                    dw = self.lambda_param * self.weights
                    db = 0.0
                else:
                    dw = self.lambda_param * self.weights - y_svm[idx] * x_i
                    db = -float(y_svm[idx])

                self.weights -= self.learning_rate * dw
                self.bias -= self.learning_rate * db

            self.loss_history.append(self._hinge_loss(X, y_svm))

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Raw signed distance from the hyperplane: w·x + b."""
        return np.dot(X, self.weights) + self.bias

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Approximate probability via sigmoid of the decision function.
        Not a true probability, but useful for risk display.
        """
        scores = self.decision_function(X)
        scores = np.clip(scores, -500, 500)
        return 1.0 / (1.0 + np.exp(-scores))

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return binary class labels {0, 1}."""
        return (self.decision_function(X) >= 0).astype(int)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
