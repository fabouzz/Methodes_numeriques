#!/usr/bin/env python3
"""Computing first derivatives of simple signals using finite diffeence methods.

Four different methods will be compared :
    - the first order finite difference method
    - the compact finite difference method
    - the hybrid finite difference method proposed by Lele in 1992
"""
import numpy as np
import matplotlib.pyplot as plt

# %% Computing the first derivative of f(x) = sin(x) (which is cos(x))


def tridiagonal_matrix_solver(A, B):
    """Solve a A * X = B tridiagonal matrix problem using Thomas algorithm.

    See https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm and https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_%28Thomas_algorithm%29 for details.

    Keywords arguments :
    A -- the (N * 3) shaped tridiagonal matrix containing each coefficient.
    B -- the (N * 1) shaped input vector

    Returns :
    X -- the (N * 1) output vector
    """
    N = len(B)
    X = np.zeros(N)
    a = A[:, 0]
    b = A[:, 1]
    c = A[:, 2]
    d = B[:]

    # Computing new coefficients
    for i in range(1, N):
        m = a[i] / b[i - 1]
        b[i] = b[i] - m * c[i - 1]
        d[i] = d[i] - m * d[i - 1]

    # Backward substitution
    # X = b
    X[-1] = d[-1] / b[-1]
    for j in range(N - 2, -1, -1):
        X[j] = (d[j] - c[j] * X[j + 1]) / b[j]

    return X


def compact_diff(u):
    """Compute the derivative of a signal u using compact finite difference method."""
    N = len(u)
    dx = u[1] - u[0]
    a = 1 / 6
    b = 2 / 3
    # Computing the tridiagonal matrix
    A = np.zeros((N, 3))
    A[:, 0] = A[:, -1] = a
    A[:, 1] = b

    # Computing the second member
    B = np.zeros(N)
    B[0] = u[1]
    B[-1] = -u[-2]
    for i in range(1, N - 1):
        B[i] = B[i + 1] - B[i - 1]

    B = B / (2 * dx)

    # Adding boundary conditions
    # A[0, :] = A[-1, :] = 0
    # A[0, 1] = A[-1, -2] = 1
    # B[0] = 1
    # B[-1] = 1

    V = tridiagonal_matrix_solver(A, B)
    return V


def diff(u, dx):
    """Compute the derivative of a signal u using centered finite difference method.

    Keywords arguments :
    u -- the input signal
    dx -- step between two samples of the u signal

    Returns :
    du / (2 * dx) -- the derivative of the u signal computed using centered finite difference method
    """

    du = np.array([u[i + 1] - u[i - 1] for i in range(1, len(u) - 1)])
    return du / (2 * dx)


if __name__ == '__main__':

    # # Testing the tridiagonal matrix solver
    A = np.array([[10, 2, 0, 0], [3, 10, 4, 0], [0, 1, 7, 5], [0, 0, 3, 4]], dtype=float)
    a = np.array([0, 3, 1, 3], dtype=float)
    b = np.array([10, 10, 7, 4], dtype=float)
    c = np.array([2, 4, 5, 0], dtype=float)
    d = np.array([3, 4, 5, 6.], dtype=float)
    AA = np.array([a, b, c]).T
    if np.allclose(np.linalg.solve(A, d), tridiagonal_matrix_solver(AA, d)):
        print("Function 'tridiagonal_matrix_solver' ok !")

    # Computing x axis and signal to derivate
    x = np.linspace(0, 2 * np.pi, 100)
    dx = x[1] - x[0]
    u = np.cos(x)
    N = len(u)

    # Derivation using finite differences
    # du = np.zeros(N)
    # du[0] = du[-1] = 1
    du = diff(u, dx)

    # Derivation using compact finite differences
    du_compact = compact_diff(u)

    # Plotting some stuff
    fig, ax = plt.subplots()
    ax.plot(x, np.cos(x), label="True derivative")
    ax.plot(x[1:-1], du, label="First order finite difference")
    ax.plot(x, du_compact, label="Compact finite difference")
    ax.set_xlabel("x")
    ax.grid()
    ax.legend()
    plt.show()
    # plt.savefig('compact_FD_x_squared.pdf')
