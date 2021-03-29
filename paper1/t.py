from tools import tools
from tools import constants
from tools import globals
from numpy import linalg as la
import numpy as np
from algorithm import MCb

Z = np.loadtxt("output/Z.txt")
b = np.loadtxt("output/b.txt")

globals.rowY = 6
globals.colY = 391

W = MCb.getW(Z, b)

print(W)
