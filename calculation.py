def A_Calculation(M, result):
    B = result[0].reshape(3, 1)
    gradient = result[1].reshape(3, 3)
    G = gradient.T

    def Sk(M):
        return np.array([
            [0, -M[2], M[1]],
            [M[2], 0, -M[0]],
            [-M[1], M[0], 0]
        ])

    T = Sk(M) @ B
    F = (M.reshape(3, 1).T @ G).T
    A = np.vstack([T, F])
    return A

def Calculation_A(M, results):
    A2 = []
    for i in range(0, 8):
        result = results[i]
        A1 = A_Calculation(M, result)
        A2.append(A1)
    A = np.hstack(A2)
    return A



