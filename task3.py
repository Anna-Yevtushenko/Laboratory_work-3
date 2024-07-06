import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt

# Завантаження даних
ratings_path = 'ml-latest-small/ratings.csv'
movies_path = 'ml-latest-small/movies.csv'

ratings = pd.read_csv(ratings_path)
movies = pd.read_csv(movies_path)

# Створення матриці рейтингів
ratings_matrix = ratings.pivot(index='userId', columns='movieId', values='rating')

# Видалення рідкісних користувачів та фільмів
ratings_matrix = ratings_matrix.dropna(thresh=10, axis=0)  # Користувачі, що оцінили менше 10 фільмів
ratings_matrix = ratings_matrix.dropna(thresh=20, axis=1)  # Фільми, що мають менше 20 оцінок