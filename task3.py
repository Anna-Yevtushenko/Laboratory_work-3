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

# Виведення даних до прогнозування
print("Дані до прогнозування:")
print(ratings_matrix[selected_columns].head(10))

# Виведення даних після прогнозування
print("\nДані після прогнозування:")
print(preds_df[selected_columns].head(10))


preds_only_df = preds_df.mask(~ratings_matrix.isna())
print("\nТільки прогнозовані дані:")
print(preds_only_df[selected_columns].head(10))



def recommend_movies(predictions_df, userID, movies_df, original_ratings_df, num_recommendations=10):
    user_row_number = userID - 1  # userId починається з 1, а індексація в DataFrame з 0
    sorted_user_predictions = predictions_df.iloc[user_row_number].sort_values(ascending=False)

    # Фільми, які користувач вже оцінив
    user_data = original_ratings_df[original_ratings_df.userId == userID]
    user_full = user_data.merge(movies_df, how='left', on='movieId').sort_values(['rating'], ascending=False)

    # Рекомендації
    recommendations = movies_df[~movies_df['movieId'].isin(user_full['movieId'])]
    recommendations = recommendations.merge(pd.DataFrame(sorted_user_predictions).reset_index(), how='left', on='movieId')
    recommendations = recommendations.rename(columns={user_row_number + 1: 'Predictions'}).sort_values('Predictions', ascending=False)

    return user_full, recommendations.head(num_recommendations)

already_rated, predictions = recommend_movies(preds_df, 1, movies, ratings, 10)

predictions = predictions.reset_index(drop=True)
predictions.index = np.arange(1, len(predictions) + 1)

print("ID рекомендованих фільмів:")
print(predictions['movieId'].values)

print("\nНазви рекомендованих фільмів:")
print(predictions[['title', 'genres']])

