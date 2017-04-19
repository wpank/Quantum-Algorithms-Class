from numpy import *
from matplotlib.pyplot import *

N = 100 #number of steps
P = 2*N+1 #number of positions

# create a matrix array in numpy
coin0 = array([1,0]) # |0>
coin1 = array([0,1]) # |1>

# outer = matrix multiplication using numpy
C00 = outer(coin0,coin0) # |0><0|
C01 = outer(coin0,coin1) # |0><1|
C10 = outer(coin1,coin0) # |1><0|
C11 = outer(coin1,coin1) # |1><1|

print("C00:")
print(C00)
print()

print("C01:")
print(C01)
print()

print("C10")
print(C10)
print()

print("C11")
print(C11)
print()

print("C00 + C01 + C10 - C11:")
print(C00+C01+C10 - C11)
print()

C_hat = (C00 + C01 + C10 - C11)/sqrt(2.)

print("C_hat")
print(C_hat)


# roll = numpy function to flatten the matrix, shift the elements by a certain amount, then return an array matrix of original proportions
# eye = returns an N x N array, with zeros everywhere, except for 1's on the diagonals.
# kron = kronecker product of two arrays. Assumes each array is the same size, if not appending 1's to the smaller array.
ShiftPlus = roll(eye(P), 1, axis=0)
ShiftMinus = roll(eye(P), -1, axis=0)
S_hat = kron(ShiftPlus, C00) + kron(ShiftMinus, C11)

print("ShiftPlus")
print(ShiftPlus)
print()

print("ShiftMinus")
print(ShiftMinus)
print()

U = S_hat.dot(kron(eye(P),C_hat))

posn0 = zeros(P)
posn0[N] = 1 # array indexing starts from 0, so index N is the central position
psi0 = kron(posn0,(coin0+coin1*1j)/sqrt(2.))


psiN = linalg.matrix_power(U, N).dot(psi0)

prob = empty(P)
for k in range(P):
    posn = zeros(P)
    posn[k] = 1
    M_hat_k = kron( outer(posn,posn), eye(2))
    proj = M_hat_k.dot(psiN)
    prob[k] = proj.dot(proj.conjugate()).real

fig = figure()
ax = fig.add_subplot(111)
plot(arange(P), prob)
#plot(arange(P), prob, 'o')
loc = range (0, P, 20) #location of ticks
xticks(loc)
xlim(0, P)
ax.set_xticklabels(range (-N,N+1, 20))
show()
