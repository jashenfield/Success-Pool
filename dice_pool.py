import numpy as np
import operator as op
import functools
import matplotlib.pyplot as plt
from matplotlib import cm



######################
# FUNCTIONS FOR MATH #
######################

# Function to caluculate nCr that I found on StackExchange
# Credit to user dheerosaur
def ncr(n, r):
    r = min(r, n-r)
    if r == 0: return 1
    numer = functools.reduce(op.mul, np.arange(n, n-r, -1))
    denom = functools.reduce(op.mul, np.arange(1, r+1))
    return numer//denom

# Calculate the chance to reach the target number of hits
def chance(num, hits, target = 4):
    sum_chance = 0  # Bootleg Sigma
    fail = (target - 1) / 6
    for k in np.arange(hits, num+1):
        sum_chance += ncr(num, k) * pow(fail,k) * pow((1-fail), (num-k))
    return sum_chance




#############################
# SETTING THE DATA MATRICES #
#############################

# X = the size of the dice pool
# Y = the number of sucesses needed
x = y = np.arange(1, 11)
X, Y = np.meshgrid(x, y)      # Setting up matrices of X and Y values
Z = np.zeros((10, 10))        # Set up a matrix of zeroes to alter later

for i in x:
    for j in y:
        if i >= j:
            prob = chance(x[i-1], y[j-1])
            # Need to switch x and y coordinates
            # That's just how it works
            # Took me forever to figure it out
            Z[j-1][i-1] = prob






########################
# SETTING UP THE PLOTS #
########################


#makes figure twice as tall as wide
fig = plt.figure(figsize=plt.figaspect(2.))

# wire subplot
ax = fig.add_subplot(2, 1, 1, projection='3d')
wire = ax.plot_wireframe(X,Y,Z, rstride=1, cstride=1)
ax.set_xlabel('Pool Size')
ax.set_ylabel('Target Number')
ax.set_zlabel('Chance of Success')

#surface subplot

ax = fig.add_subplot(2, 1, 2, projection='3d')
surf = ax.plot_surface(X, Y, Z, linewidth=0, cstride=1,
                       rstride=1, cmap=cm.coolwarm,
                       antialiased=False)
ax.set_xlabel('Pool Size')
ax.set_ylabel('Target Number')
ax.set_zlabel('Chance of Success')

# color bar for the surface plot
fig.colorbar(surf, shrink=1, aspect=5)


plt.show()












