from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm


from typing import Dict, Tuple

import numpy as np
from tqdm import tqdm


class UVDecomposition:
    def __init__(self, seed: int = None):
        """
        Initializes UVDecomposition object with optional seed.

        Args:
            seed (int, optional): Seed for numpy random number generator.
        """
        self.U: np.ndarray = None
        self.V: np.ndarray = None
        self.mean: Dict[int, float] = {}
        self.seed = seed
        if self.seed is not None:
            np.random.seed(self.seed)

    @staticmethod
    def _RMSE(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Calculates Root Mean Square Error between two numpy arrays.

        Args:
            y_true (np.ndarray): True values.
            y_pred (np.ndarray): Predicted values.

        Returns:
            float: The RMSE value.
        """
        return np.sqrt(np.square(np.subtract(y_true, y_pred)).mean())

    def _calculate_RMSE(self, M: np.ndarray) -> float:
        """
        Calculates RMSE between the product of matrices U and V and a given matrix.

        Args:
            M (np.ndarray): Given matrix.

        Returns:
            float: The RMSE value.
        """
        UV_product = self.U @ self.V
        mask = M != 0
        return self._RMSE(M[mask], UV_product[mask])

    def _normalize(self, user_id: int, user_ratings: np.ndarray) -> np.ndarray:
        """
        Normalizes user ratings by subtracting the mean of non-zero ratings.

        Args:
            user_id (int): User ID.
            user_ratings (np.ndarray): User ratings.

        Returns:
            np.ndarray: Normalized user ratings.
        """
        non_zero_indices = np.where(user_ratings != 0)
        non_zero_ratings = user_ratings[non_zero_indices]
        mean_rating = np.mean(non_zero_ratings) if non_zero_ratings.size != 0 else 0
        self.mean[user_id] = mean_rating
        user_ratings[non_zero_indices] -= mean_rating
        return user_ratings

    def _denormalize(self, user_id: int, normalized_ratings: float) -> float:
        """
        Denormalizes user ratings by adding the mean of non-zero ratings.

        Args:
            user_id (int): User ID.
            normalized_ratings (float): Normalized user ratings.

        Returns:
            float: Denormalized user ratings.
        """
        mean_rating = self.mean[user_id]
        return normalized_ratings + mean_rating

    def _initialize_U_and_V(
        self, M: np.ndarray, d: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Initializes matrices U and V with random values.

        Args:
            M (np.ndarray): Given matrix.
            d (int): The number of latent factors.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Initialized matrices U and V.
        """
        U = np.random.rand(M.shape[0], d)
        V = np.random.rand(d, M.shape[1])
        return U, V

    def _calculate_gradients(
        self, M: np.ndarray, indices: np.ndarray, reg_factor: float = 0.01
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculates the gradients of the objective function with respect to U and V.

        Args:
            M (np.ndarray): Given matrix.
            indices (np.ndarray): Indices of non-zero values in the given matrix.
            reg_factor (float, optional): The regularization factor. Default is 0.01.

        Returns:
            Tuple[np.ndarray, np.ndarray]: Gradients of the objective function with respect to U and V.
        """
        UV_product = self.U @ self.V
        mask = np.zeros_like(M, dtype=bool)
        mask[indices[:, 0], indices[:, 1]] = M[indices[:, 0], indices[:, 1]] != 0
        diff = np.zeros_like(M)
        diff[mask] = M[mask] - UV_product[mask]

        gradient_U = -2 * (diff @ self.V.T) + 2 * reg_factor * self.U
        gradient_V = -2 * (self.U.T @ diff) + 2 * reg_factor * self.V

        return gradient_U, gradient_V

    def _train_step(
        self,
        M: np.ndarray,
        learning_rate: float,
        reg_factor: float,
        GD_type: str = "BGD",
        batch_size: int = None,
    ) -> float:
        non_zero_indices = np.argwhere(M != 0)
        """
        Performs one training step.

        Args:
            M (np.ndarray): Given matrix.
            learning_rate (float): The learning rate.
            GD_type (str, optional): Type of gradient descent. Can be 'BGD', 'SGD' or 'MBGD'. Default is 'BGD'.
            batch_size (int, optional): Batch size for 'MBGD'. Default is None.

        Returns:
            float: RMSE after the training step.
        """
        if GD_type == "BGD":
            indices = np.indices(M.shape).reshape(2, -1).T
        elif GD_type == "SGD":
            index = np.random.choice(non_zero_indices.shape[0])
            indices = non_zero_indices[index][None, :]
        elif GD_type == "MBGD":
            if batch_size is None:
                raise ValueError(
                    "Batch size must be provided for mini-batch gradient descent."
                )
            batch_indices = np.random.choice(
                non_zero_indices.shape[0], batch_size, replace=False
            )
            indices = non_zero_indices[batch_indices]
        else:
            raise ValueError(
                f"Unknown GD_type {GD_type}. Please use 'BGD', 'SGD', or 'MBGD'."
            )

        gradient_U, gradient_V = self._calculate_gradients(M, indices, reg_factor)
        self.U -= learning_rate * gradient_U
        self.V -= learning_rate * gradient_V

        return self._calculate_RMSE(M)

    def train(
        self,
        M: np.ndarray,
        d: int,
        learning_rate: float,
        max_iterations: int,
        reg_factor: float = 0.01,
        patience: int = 50,
        GD_type: str = "BGD",
        batch_size: int = None,
        verbose: bool = True,
    ):
        """
        Train the model.

        Args:
            M (np.ndarray): The utility matrix.
            d (int): The number of latent factors.
            learning_rate (float): The learning rate.
            max_iterations (int): The maximum number of iterations for training.
            patience (int): The maximum number of iterations where the training doesn't improve
            GD_type (str): The type of gradient descent to use, one of 'BGD', 'SGD', or 'MBGD'. Default is 'BGD'.
            batch_size (int): The batch size for 'MBGD'. Default is None.
            verbose (bool, optional): Whether to print progress messages. Defaults to True.

        Returns:
            training_history (list): List of RMSE values at each iteration
        """
        training_history = []

        for i in range(M.shape[0]):
            M[i, :] = self._normalize(i, M[i, :])

        self.U, self.V = self._initialize_U_and_V(M, d)

        old_RMSE = self._calculate_RMSE(M)
        pbar = tqdm(range(max_iterations), desc="", ncols=100)
        no_progress = 0
        for i in pbar:
            new_RMSE = self._train_step(
                M, learning_rate, reg_factor, GD_type, batch_size
            )
            if np.isnan(new_RMSE):
                print("RMSE is nan, breaking the training.")
                break
            training_history.append(new_RMSE)
            if verbose:
                pbar.set_description(f"Iteration nr {i+1}, RMSE = {new_RMSE:.4f}")
                pbar.refresh()
            if old_RMSE < new_RMSE:
                no_progress += 1
            elif old_RMSE > new_RMSE:
                no_progress = 0
                if no_progress == patience:
                    print(
                        f"Stopping training due to no progress in last {patience} iterations"
                    )
                    break
            old_RMSE = new_RMSE

        return training_history

    def predict(self, user_index: int, item_index: int) -> float:
        """
        Predicts the rating of a user for an item.

        Args:
            user_id (int): User ID.
            item_id (int): Item ID.

        Returns:
            float: The predicted rating.
        """
        return self._denormalize(
            user_index, (self.U[user_index, :] @ self.V[:, item_index])
        )
