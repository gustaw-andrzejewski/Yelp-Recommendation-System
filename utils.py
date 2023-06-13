import random

import numpy as np
import plotly.express as px
from pyspark.sql import DataFrame
from pyspark.sql.functions import col
from pyspark.sql.functions import first as F_first
from tqdm import tqdm

from UV import UVDecomposition


def create_utility_matrix(indexed_df: DataFrame, chunk_size: int) -> np.ndarray:
    """
    Creates a utility matrix from a DataFrame of user-item ratings, processing the DataFrame in chunks.

    Args:
        indexed_df (DataFrame): A Spark DataFrame containing userIndex, businessIndex, and average_stars columns.
        chunk_size (int): The size of the chunks to process the DataFrame in.

    Returns:
        np.ndarray: A numpy array representation of the utility matrix, with userIndex as rows, businessIndex as columns, and average_stars as values.
    """
    all_business_ids = sorted(
        indexed_df.select("businessIndex").distinct().rdd.flatMap(lambda x: x).collect()
    )
    chunks = []
    num_chunks = indexed_df.count() // chunk_size + (
        indexed_df.count() % chunk_size != 0
    )

    for i in tqdm(range(num_chunks)):
        start = i * chunk_size
        end = start + chunk_size - 1
        chunk_df = indexed_df.filter(col("userIndex").between(start, end))
        utility_matrix_df = (
            chunk_df.groupBy("userIndex")
            .pivot("businessIndex", all_business_ids)
            .agg(F_first("average_stars"))
        )
        utility_matrix_df = utility_matrix_df.fillna(0)
        chunks.append(
            utility_matrix_df.toPandas().set_index("userIndex").sort_index().values
        )

    utility_matrix_array = np.concatenate(chunks, axis=0)

    return utility_matrix_array


# utils


def plot_history(history: list) -> None:
    """plots history of training"""
    indices = np.arange(len(history))
    fig = px.line(x=indices, y=history)
    fig.update_layout(
        title="Training history",
        xaxis_title="Iteration",
        yaxis_title="RMSE",
        font=dict(size=18),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=18),
        ),
    )
    fig.show()


def display_random_predictions(uv: UVDecomposition, train_set: np.ndarray):
    """
    Displays predictions for a random sample of users from the original training set.

    Args:
        uv (UVDecomposition): The trained UVDecomposition object.
        train_set (np.ndarray): The original training set.
        num_users (int): The number of users for which to display predictions.
    """
    user_indices = np.random.choice(train_set.shape[0], 5, replace=False)
    for user_index in user_indices:
        user_ratings = train_set[user_index, :]
        non_zero_indices = np.where(user_ratings != 0)[0]
        if len(non_zero_indices) > 0:
            actual_ratings = user_ratings[non_zero_indices]
            predicted_ratings = [
                uv.predict(user_index, item_index) for item_index in non_zero_indices
            ]
            print(f"User {user_index} Ratings:")
            print("Actual: ", actual_ratings)
            print("Predicted: ", predicted_ratings)
            print("\n")


def train_test_split(train_set: np.ndarray, test_samples: int):
    """
    Create a test set by randomly selecting samples from the training set.

    Args:
        train_set (np.ndarray): The training set.
        num_samples (int): The number of samples to select for the test set.

    Returns:
        np.ndarray: The test set.
    """
    test_indices = random.sample(range(train_set.shape[0]), test_samples)
    test_set = train_set[test_indices]
    train_set = np.delete(train_set, test_indices, axis=0)
    return train_set, test_set
