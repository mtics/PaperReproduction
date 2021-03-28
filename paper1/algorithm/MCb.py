import numpy as np
from numpy import linalg as la
import math
from tools import constants
from tools import globals


def MCb(Z0, X, Y, b0, mu0, tau_b, tau_z):
    # Initialization
    tempMu = mu0
    mus = [tempMu]
    while tempMu > 1e-5:
        tempMu = tempMu * constants.ETA
        mus.append(tempMu)

    Z = Z0
    b = b0

    lastZ = np.zeros((Z.shape[0], Z.shape[1]))

    # Iteration for convergence
    iters_mu = 0
    for mu in mus:

        # Count to print
        iters_mu += 1
        iters_while = 0

        loss = 65535
        while loss > 1e-5:
            iters_while += 1
            print("[mu iters: %d, while iters: %d, loss: %.10f]" % (iters_mu, iters_while, loss))

            lastZ = Z

            b = b - tau_b * gb(Z, Y, b)
            A = Z - tau_z * gz(Z, X, Y, b)

            U, Sigma, VT = la.svd(A)
            S = np.zeros((U.shape[1], VT.shape[0]))
            for i in range(0, Sigma.size):
                # S[i][i] = max(Sigma[i] - tau_z * mu, 0)
                S[i][i] = max(Sigma[i] - mu, 0)

            Z = U @ S @ VT

            loss = la.norm(Z - lastZ)

    return Z, b


def gb(Z, Y, b):
    """
    Formulation (4)
    :param Z:
    :param Y:
    :param b:
    :return:
    """

    colY = Y.shape[1]

    bc = b.copy()
    for i in range(0, b.shape[0]):
        su = 0  # sum
        for j in range(0, colY):
            if globals.corruptedPositionY.T[i, j] == 0:
                su += -Y[i, j] / (1 + math.exp(Y[i, j] * (Z[i, j] + b[i])))

        bc[i] = constants.LAMBDA / globals.omegaY * su

    return bc


def gz(Z, X, Y, b):
    """
    Formulation (5)
    :param Z:
    :param X:
    :param Y:
    :param b:
    :return:
    """

    rowY = Y.shape[0]

    Zc = Z.copy()
    # The condition is not complete in the part,
    # must compete it later!
    for i in range(0, Z.shape[0]):
        for j in range(0, Z.shape[1]):
            if i < rowY and globals.corruptedPositionY.T[i, j] == 0:
                Zc[i, j] = (constants.LAMBDA / globals.omegaY) * (-Y[i, j]) / (1 + math.exp(Y[i, j] * (Z[i, j] + b[i])))
            elif i >= rowY and globals.corruptedPositionX.T[i - rowY, j] == 0:
                Zc[i, j] = 1 / globals.omegaX * (Z[i, j] - X[i - rowY, j])
            else:
                Zc[i, j] = 0

    return Zc
