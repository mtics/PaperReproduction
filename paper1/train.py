from tools import tools
from tools import constants
from tools import globals
from numpy import linalg as la
import numpy as np
from algorithm import MCb

# Define some constants

if __name__ == "__main__":

    # Load data
    filepath = "dataset/emotions/emotions-train.arff"
    [samples, cla] = tools.getDataFromArff(filepath)

    # X: feature matrix
    # Y: label matrix
    X = samples[:, 0:72]
    Y = samples[:, 72:]

    # Set global variants' values
    globals.rowX = X.shape[0]
    globals.colX = X.shape[1]
    globals.sizeX = X.size
    globals.omegaX = int(X.size * constants.RATE)

    globals.rowY = Y.shape[0]
    globals.colY = Y.shape[1]
    globals.sizeY = Y.size
    globals.omegaY = int(Y.size * constants.RATE)

    # Initialization
    b0 = np.zeros([Y.shape[1]])

    globals.corruptedPositionX = np.zeros((X.shape[0], X.shape[1]))
    globals.corruptedPositionY = np.zeros((Y.shape[0], Y.shape[1]))

    incompleteX = tools.mask(X, globals.corruptedPositionX, constants.RATE)
    incompleteY = tools.mask(Y, globals.corruptedPositionY, constants.RATE)
    tau_z = min(3.8 * globals.omegaY / constants.LAMBDA, globals.omegaX)
    tau_b = 3.8 * globals.omegaY / (constants.LAMBDA * Y.shape[0])

    Z0 = np.concatenate([incompleteY.T, incompleteX.T], axis=0).astype(dtype=float)
    orgZ = np.concatenate([Y.T, X.T], axis=0).astype(dtype=float)

    U, S, VT = la.svd(Z0)
    mu = S.max() * constants.ETA

    # Training
    newZ, newb = MCb.MCb(Z0, incompleteX.T, incompleteY.T, b0, mu, tau_b, tau_z)

    np.savetxt("output/Z.txt", newZ)
    np.savetxt("output/b.txt", newb)

    W = MCb.getW(newZ, newb)
    np.savetxt("output/W.txt", W)

    rfie = tools.relativeFeatureImputationError(orgZ, newZ)
    tle = tools.transductiveLabelError(orgZ, newZ)

    print("Relative feature Imputation Error: %f \n Transductive Label Error: %f" %(rfie, tle))
