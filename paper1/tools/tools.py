from scipy.io import arff
import pandas as pd
import random


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
