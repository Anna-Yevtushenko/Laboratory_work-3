import numpy as np

#M = np.random.rand(4, 3)
M = np.array([[1, 2, 0], [0, 1, 3], [4, 0, 0]])

def custom_svd(M):

    MTM = np.dot(M.T, M)
    MMT = np.dot(M, M.T)

    eigvals_MTM, V = np.linalg.eigh(MTM)
    eigvals_MMT, U = np.linalg.eigh(MMT)

    sorted_indices_MTM = np.argsort(eigvals_MTM)[::-1]
    sorted_indices_MMT = np.argsort(eigvals_MMT)[::-1]

    eigvals_MTM = eigvals_MTM[sorted_indices_MTM]
    V = V[:, sorted_indices_MTM]

    eigvals_MMT = eigvals_MMT[sorted_indices_MMT]
    U = U[:, sorted_indices_MMT]

    singular_values = np.sqrt(eigvals_MTM)

    Sigma = np.zeros(M.shape)
    np.fill_diagonal(Sigma, singular_values)

    U = np.dot(U, np.diag(np.sign(np.diag(np.dot(U.T, U)))))
    V = np.dot(V, np.diag(np.sign(np.diag(np.dot(V.T, V)))))

    return U, Sigma, V.T


def realization_svd(M, U, Sigma, VT):
    M_reconstructed = np.dot(U, np.dot(Sigma, VT))
    return np.allclose(M, M_reconstructed)

U, Sigma, VT = custom_svd(M)

print("U:\n", U)
print("Sigma:\n", Sigma)
print("V^T:\n", VT)

is_correct = realization_svd(M, U, Sigma, VT)
print("Is SVD decomposition correct?", is_correct)