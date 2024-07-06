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

# Заповнення відсутніх значень
ratings_matrix_filled = ratings_matrix.fillna(2.5)

# Перетворення на масив NumPy
R = ratings_matrix_filled.values

# Усунення особливостей оцінювання кожним користувачем
user_ratings_mean = np.mean(R, axis=1)
R_demeaned = R - user_ratings_mean.reshape(-1, 1)

# Виконання SVD
U, sigma, Vt = svds(R_demeaned, k=3)

# Перетворення sigma на діагональну матрицю
sigma = np.diag(sigma)

# Створення матриці з прогнозованими оцінками
all_user_predicted_ratings = np.dot(np.dot(U, sigma), Vt) + user_ratings_mean.reshape(-1, 1)
preds_df = pd.DataFrame(all_user_predicted_ratings, columns=ratings_matrix.columns, index=ratings_matrix.index)


# Функція для надання рекомендацій користувачам
def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=10):
    user_row_number = userID - 1  # userId починається з 1, а індексація в DataFrame з 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    # Фільми, які користувач вже оцінив
    user_data = original_ratings_df[original_ratings_df.userId == userID]
    user_full = user_data.merge(movies_df, how='left', on='movieId').sort_values(['rating'], ascending=False)

    # Рекомендації
    recommendations = movies_df[~movies_df['movieId'].isin(user_full['movieId'])]
    recommendations = recommendations.merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left',
                                            on='movieId')
    recommendations = recommendations.rename(columns={user_row_number: 'Predictions'}).sort_values('Predictions',
                                                                                                   ascending=False)

    return user_full, recommendations.head(num_recommendations)


# Виправлена функція для надання рекомендацій
def recommend_movies_fixed(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=10):
    user_row_number = userID - 1  # userId починається з 1, а індексація в DataFrame з 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False).reset_index()
    sorted_user_predictions.columns = ['movieId', 'Predictions']

    # Фільми, які користувач вже оцінив
    user_data = original_ratings_df[original_ratings_df.userId == userID]
    user_full = user_data.merge(movies_df, how='left', on='movieId').sort_values(['rating'], ascending=False)

    # Рекомендації
    recommendations = movies_df[~movies_df['movieId'].isin(user_full['movieId'])]
    recommendations = recommendations.merge(sorted_user_predictions, how='left', on='movieId')
    recommendations = recommendations.sort_values('Predictions', ascending=False)

    return user_full, recommendations.head(num_recommendations)


# Отримання рекомендацій для користувача з ID = 1
already_rated, predictions = recommend_movies_fixed(preds_df, 1, movies, ratings, 10)

print("ID рекомендованих фільмів:")
print(predictions['movieId'].values)

print("\nНазви рекомендованих фільмів:")
print(predictions[['title', 'genres']])
