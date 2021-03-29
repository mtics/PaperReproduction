from tools import tools
from tools import constants
from tools import globals
from numpy import linalg as la
import numpy as np
from algorithm import MCb

# Define some constants

if __name__ == "__main__":

    # Load data
    filepath = "dataset/emotions/emotions-test.arff"
    [samples, cla] = tools.getDataFromArff(filepath)

    # X: feature matrix
    # Y: label matrix
    X = samples[:, 0:72].astype(float)
    realY = samples[:, 72:].astype(float)

    # Set global variants' values
    globals.rowX = X.shape[0]
    globals.colX = X.shape[1]
    globals.sizeX = X.size
    globals.omegaX = int(X.size * constants.RATE)

    globals.rowY = realY.shape[0]
    globals.colY = realY.shape[1]
    globals.sizeY = realY.size
    globals.omegaY = int(realY.size * constants.RATE)

    b = np.loadtxt("output/b.txt").astype(float)
    W = np.loadtxt("output/W.txt").astype(float)

    testY = W@X.T
    for j in range(testY.shape[1]):
        testY[:, j] = testY[:, j] + b

    err = la.norm(testY-realY.T)

    print(err)
