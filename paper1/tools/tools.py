from scipy.io import arff
import pandas as pd
import random
from tools import globals


def getDataFromArff(filepath):
    data = arff.loadarff(filepath)
    dataFrame = pd.DataFrame(data[0])

    # Sample
    sample = dataFrame.values[:, 0:len(dataFrame.values[0])]

    # Handle labels
    labels = dataFrame.head()
    cla = []
    for label in labels:
        test = label
        cla.append(test)

    return sample, cla


def mask(mat, corruptedPosition, rate):
    """
    Randomly choose some elements to set to 0
    :param mat: a 2D ndarray
    :param corruptedPosition:
    :param rate: a fixed rate of elements chosed
    :return: the changed matrix
    """

    num = int(mat.size * rate)

    cmat = mat.astype(dtype=float)

    row = mat.shape[0]
    column = mat.shape[1]

    while num != 0:
        i = random.randint(0, row - 1)
        j = random.randint(0, column - 1)
        cmat[i, j] = 0
        corruptedPosition[i, j] = 1
        num -= 1

    return cmat


def relativeFeatureImputationError(orgZ, newZ):
    orgX = orgZ[globals.colY:, :].astype(float)
    newX = newZ[globals.colY:, :].astype(float)

    total = 0
    su = 0

    for i in range(orgX.shape[0]):
        for j in range(orgX.shape[1]):
            total += orgX[i, j] ** 2
            if globals.corruptedPositionX.T[i, j] == 1:
                su += (orgX[i, j] - newX[i, j]) ** 2

    return su / total


def transductiveLabelError(orgZ, newZ):
    orgY = orgZ[:globals.colY, :].astype(float)
    newY = newZ[:globals.colY, :].astype(float)

    total = 0

    for i in range(orgY.shape[0]):
        for j in range(orgY.shape[1]):
            if newY - orgY != 0:
                total += 1

    return total / (globals.sizeY - globals.omegaY)
