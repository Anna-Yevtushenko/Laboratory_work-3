import pandas as pd
import numpy as np
from scipy.sparse.linalg import svds
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

file_path = 'ratings.csv'
df = pd.read_csv(file_path)

ratings_matrix = df.pivot(index='userId', columns='movieId', values='rating')

# Видалення рідкісних користувачів та фільмів
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=0)  # Користувачі, що оцінили менше 100 фільмів
ratings_matrix = ratings_matrix.dropna(thresh=100, axis=1)  # Фільми, що мають менше 100 оцінок

# ratings_matrix_filled_zero = ratings_matrix.fillna(0)
# ratings_matrix_filled_mean = ratings_matrix.fillna(2.5)

if ratings_matrix.empty:
    print("Matrix is emty after ratings_matrix.dropna")
else:
    # Заповнення відсутніх значень
    # ratings_matrix_filled = ratings_matrix.fillna(0)
    ratings_matrix_filled = ratings_matrix.fillna(2.5)
    print(ratings_matrix_filled)
    # віднімаємо середню оцінку яку давав користувач
    R = ratings_matrix_filled.values  # перетворення в масив
    user_ratings_mean = np.mean(R, axis=1)  # за рядками
    R_demeaned = R - user_ratings_mean.reshape(-1, 1)

