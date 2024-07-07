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

    try:
        U, sigma, Vt = svds(R_demeaned, k=3)  # Виконуємо SVD  k- розмірність даних, яку зберігаємо
        sigma = np.diag(sigma)
        user_factors = pd.DataFrame(U, index=ratings_matrix.index,
                                    columns=[f'Feature_{i}' for i in range(1, U.shape[1] + 1)])
        movie_factors = pd.DataFrame(Vt.T, index=ratings_matrix.columns,
                                     columns=[f'Feature_{i}' for i in range(1, Vt.T.shape[1] + 1)])

        print("Матриця U (подібністі смаків у різних користувачів)(перші 15 рядків):")
        print(user_factors.head(15))

        print("Матриця V (схожість фільмів)):")
        print(movie_factors.head(15))

        fig = plt.figure(figsize=(14, 6))

        ax = fig.add_subplot(121, projection='3d')


        ax.scatter(user_factors['Feature_1'][:15], user_factors['Feature_2'][:15], user_factors['Feature_3'][:15],
                   c='r', label='Users')

        ax.set_title('Users in 3D space')

        ax.legend()

        # Візуалізація фільмів у 3D просторі
        ax = fig.add_subplot(122, projection='3d')
        ax.scatter(movie_factors['Feature_1'][:15], movie_factors['Feature_2'][:15],
                   movie_factors['Feature_3'][:15], c='b', label='Movies')
        ax.set_title('Movies in 3D space')

        ax.legend()

        plt.show()

    except Exception as e:
        print(f"An error occurred while executing SVD: {e}")


