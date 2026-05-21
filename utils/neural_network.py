"""
Single-Layer Neural Network (Perceptron) implemented from scratch using:
- Sigmoid activation function
- Forward computation
- Cross-Entropy Loss
- Mini-Batch Gradient Descent
"""

import numpy as np


class NeuralNetworkScratch:
    """
    Single hidden-layer neural network for binary classification.

    Architecture:  Input → Dense(hidden_size, sigmoid) → Dense(1, sigmoid)

    Parameters
    ----------
    hidden_size : int
        Number of neurons in the hidden layer.
    learning_rate : float
        Step size for mini-batch gradient descent.
    n_epochs : int
        Number of full passes over the training data.
    batch_size : int
        Number of samples per mini-batch.
    """

    def __init__(
        self,
        hidden_size: int = 16,
        learning_rate: float = 0.01,
        n_epochs: int = 200,
        batch_size: int = 32,
    ):
        self.hidden_size = hidden_size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.loss_history: list[float] = []

        # Weights initialised in fit()
        self.W1: np.ndarray | None = None
        self.b1: np.ndarray | None = None
        self.W2: np.ndarray | None = None
        self.b2: np.ndarray | None = None

    # ------------------------------------------------------------------
    # Activation & Loss
    # ------------------------------------------------------------------
    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -500, 500)
        return 1.0 / (1.0 + np.exp(-z))

    @staticmethod
    def _sigmoid_derivative(a: np.ndarray) -> np.ndarray:
        """Derivative of sigmoid given its output a: a*(1-a)."""
        return a * (1.0 - a)

    @staticmethod
    def _cross_entropy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        eps = 1e-15
        y_pred = np.clip(y_pred, eps, 1 - eps)
        return float(
            -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        )

    # ------------------------------------------------------------------
    # Weight Initialisation
    # ------------------------------------------------------------------
    def _init_weights(self, n_features: int) -> None:
        """Xavier / Glorot uniform initialisation for stable training."""
        rng = np.random.default_rng(42)
        limit1 = np.sqrt(6.0 / (n_features + self.hidden_size))
        limit2 = np.sqrt(6.0 / (self.hidden_size + 1))

        self.W1 = rng.uniform(-limit1, limit1, (n_features, self.hidden_size))
        self.b1 = np.zeros((1, self.hidden_size))
        self.W2 = rng.uniform(-limit2, limit2, (self.hidden_size, 1))
        self.b2 = np.zeros((1, 1))

    # ------------------------------------------------------------------
    # Forward Pass
    # ------------------------------------------------------------------
    def _forward(self, X: np.ndarray):
        """
        Forward computation through two layers.

        Returns
        -------
        A1 : hidden layer activations  (N, hidden_size)
        A2 : output layer activations  (N, 1)
        """
        Z1 = np.dot(X, self.W1) + self.b1   # (N, hidden_size)
        A1 = self._sigmoid(Z1)               # (N, hidden_size)
        Z2 = np.dot(A1, self.W2) + self.b2  # (N, 1)
        A2 = self._sigmoid(Z2)               # (N, 1)
        return A1, A2

    # ------------------------------------------------------------------
    # Backward Pass
    # ------------------------------------------------------------------
    def _backward(self, X: np.ndarray, y: np.ndarray, A1: np.ndarray, A2: np.ndarray):
        """
        Backpropagation through both layers.

        Output layer gradient:
            dL/dZ2 = A2 - y   (BCE + sigmoid simplification)
        Hidden layer gradient:
            dL/dZ1 = (dL/dZ2 · W2ᵀ) ⊙ σ'(A1)
        """
        n = X.shape[0]
        y = y.reshape(-1, 1)

        # Output layer
        dZ2 = A2 - y                                    # (N, 1)
        dW2 = np.dot(A1.T, dZ2) / n                    # (hidden, 1)
        db2 = np.mean(dZ2, axis=0, keepdims=True)       # (1, 1)

        # Hidden layer
        dA1 = np.dot(dZ2, self.W2.T)                   # (N, hidden)
        dZ1 = dA1 * self._sigmoid_derivative(A1)        # (N, hidden)
        dW1 = np.dot(X.T, dZ1) / n                     # (features, hidden)
        db1 = np.mean(dZ1, axis=0, keepdims=True)       # (1, hidden)

        return dW1, db1, dW2, db2

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "NeuralNetworkScratch":
        """Train with mini-batch gradient descent."""
        n_samples, n_features = X.shape
        self._init_weights(n_features)
        self.loss_history = []

        for epoch in range(self.n_epochs):
            # Shuffle data each epoch
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]

            epoch_losses = []
            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                X_batch = X_shuffled[start:end]
                y_batch = y_shuffled[start:end]

                # Forward
                A1, A2 = self._forward(X_batch)

                # Loss
                batch_loss = self._cross_entropy(y_batch, A2.ravel())
                epoch_losses.append(batch_loss)

                # Backward
                dW1, db1, dW2, db2 = self._backward(X_batch, y_batch, A1, A2)

                # Update
                self.W1 -= self.learning_rate * dW1
                self.b1 -= self.learning_rate * db1
                self.W2 -= self.learning_rate * dW2
                self.b2 -= self.learning_rate * db2

            self.loss_history.append(float(np.mean(epoch_losses)))

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return sigmoid probability for each sample."""
        _, A2 = self._forward(X)
        return A2.ravel()

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        """Return binary class labels {0, 1}."""
        return (self.predict_proba(X) >= threshold).astype(int)

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------
    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))
