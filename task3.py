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

ratings_matrix_filled = ratings_matrix.fillna(2.5)
R = ratings_matrix_filled.values
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# Виконання SVD
U, sigma, Vt = svds(R_demeaned, k=3)
sigma = np.diag(sigma)

# Отримання матриці з прогнозованими оцінками
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)

# Створення DataFrame для прогнозованих оцінок
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)

# Вибір перших 10 стовпців для виведення
selected_columns = ratings_matrix.columns[:10]

print("Дані до прогнозування:")
print(ratings_matrix[selected_columns].head(10))


print("\nДані після прогнозування:")
print(preds_df[selected_columns].head(10))

preds_only_df = preds_df.mask(~ratings_matrix.isna())
print("\nТільки прогнозовані дані:")
print(preds_only_df[selected_columns].head(10))




