import numpy as np
import operator as op
import functools
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, range(n, n-r, -1))
    denom = functools.reduce(op.mul, range(1, r+1))
    return numer//denom

# calculate the chance to reach the target number of hits
def chance(num, hits, target = 4):
    sum_chance = 0
    fail = (target - 1) / 6
    for k in range(hits, num+1):
        sum_chance += ncr(num, k) * pow(fail,k) * pow((1-fail), (num-k))
    return sum_chance

pool = np.arange(1, 11)
targetnum = np.arange(1,11)

fig = plt.figure()
ax1 = fig.add_subplot(111, projection='3d')

xpos = [1,2,3,4,5,6,7,8,9,10]
ypos = [2,3,4,5,1,6,2,1,7,2]
num_elements = len(xpos)
zpos = [0,0,0,0,0,0,0,0,0,0]
dx = np.ones(10)
dy = np.ones(10)
dz = [1,2,3,4,5,6,7,8,9,10]

ypos2 = [4,1,5,2,7,3,8,9,1,6]
dz2 = [6,7,8,9,10,1,2,3,4,5]

ax1.bar3d(xpos, ypos, zpos, dx, dy, dz, color='#00ceaa')
ax1.bar3d(xpos, ypos2, zpos, dx, dy, dz2, color='#ff0000')
plt.show()
