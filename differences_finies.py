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
    """Solve a A * X = B tridiagonal matrix problem using Thomas algorithm, see https://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm and https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_%28Thomas_algorithm%29 for details.

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
    cp = np.zeros(N)
    dp = np.zeros(N)
    cp[0] = c[0] / b[0]
    dp[0] = d[0] / b[0]
    for i in range(1, N):
        cp[i] = c[i] / (b[i] - a[i] * cp[i - 1])
        dp[i] = (d[i] - a[i] * dp[i - 1]) / (b[i] - a[i] * cp[i - 1])
    # Backward substitution
    X[-1] = dp[-1]
    for i in range(N - 2, -1, -1):
        X[i] = dp[i] - cp[i] * X[i + 1]

    return X


def compact_diff(u):
    """Compute th derivative of a singal u using compact finite difference method."""
    N = len(u)
    dx = u[1] - u[0]
    a = 1 / 6
    b = 2 / 3
    # Computing the tridiagonal matrix
    A = np.zeros((N, 3))
    A[:, 0] = A[:, -1] = a
    A[:, 1] = b
    A[0, 0] = A[-1, -1] = 0

    # Computing the second member
    B = np.zeros(N)
    B[0] = u[1]
    B[-1] = -u[-2]
    for i in range(1, N - 1):
        B[i] = B[i + 1] - B[i - 1]
    B = B / (2 * dx)

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
    x = np.linspace(0, 2 * np.pi, 100)
    dx = x[1] - x[0]
    u = np.sin(x)
    N = len(u)

    du = np.zeros(N)
    du[0] = du[-1] = 1
    du[1:-1] = diff(u, dx)

    du_compact = compact_diff(u)

    fig, ax = plt.subplots()
    ax.plot(x, np.cos(x), label="True derivative")
    ax.plot(x, du, label="First order finite difference")
    ax.plot(x, du_compact, label="Compact finite difference")
    ax.set_xlabel("x")
    ax.grid()
    ax.legend()
    plt.show()