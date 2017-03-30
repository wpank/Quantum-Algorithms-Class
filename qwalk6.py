import numpy as np
from matplotlib.pyplot import *

N = 5 #number of steps
P = 2*N+1 #number of positions in each direction

coin00 = np.array([1,0,0,0]) # |00>
coin01 = np.array([0,1,0,0]) # |01>
coin10 = np.array([0,0,1,0]) # |10>
coin11 = np.array([0,0,0,1]) # |11>

C0000 = np.outer(coin00,coin00) # |00><00|
C0001 = np.outer(coin00,coin01) # |00><01|
C0010 = np.outer(coin00,coin10) # |00><10|
C0011 = np.outer(coin00,coin11) # |00><11|
C0100 = np.outer(coin01,coin00) # |01><00|
C0101 = np.outer(coin01,coin01) # |01><01|
C0110 = np.outer(coin01,coin10) # |01><10|
C0111 = np.outer(coin01,coin11) # |01><11|
C1000 = np.outer(coin10,coin00) # |10><00|
C1001 = np.outer(coin10,coin01) # |10><01|
C1010 = np.outer(coin10,coin10) # |10><10|
C1011 = np.outer(coin10,coin11) # |10><11|
C1100 = np.outer(coin11,coin00) # |11><00|
C1101 = np.outer(coin11,coin01) # |11><01|
C1110 = np.outer(coin11,coin10) # |11><10|
C1111 = np.outer(coin11,coin11) # |11><11|

C_hat = (C0000 + C0001 + C0010 + C0011 + C0100 - C0101 + C0110 - C0111 + C1000 + C1001 - C1010 - C1011 + C1100 - C1101 - C1110 + C1111)/2.0

ShiftLeft  = np.kron(np.eye(P),np.roll(np.eye(P), 1, axis=0))
ShiftRight = np.kron(np.eye(P),np.roll(np.eye(P), -1, axis=0))
ShiftDown  = np.kron(np.roll(np.eye(P), -1, axis=0),np.eye(P))
ShiftUp    = np.kron(np.roll(np.eye(P), 1, axis=0),np.eye(P))

S_hat = np.kron(ShiftRight, C0000) + np.kron(ShiftUp, C0101) + np.kron(ShiftLeft, C1010) + np.kron(ShiftDown,C1111)

U = S_hat.dot(np.kron(np.eye(P*P),C_hat))

posn0 = np.zeros(P*P)
posn0[N*P+N] = 1 # np.array indexing starts from (0) so index (N*P+N) is the central position
initialCoin=(coin00+coin01+coin10+coin11)/2.0
psi0 = np.kron(posn0,initialCoin) #Initial spin 

for frame in range(N+1):
    psiN = np.linalg.matrix_power(U, frame).dot(psi0)

    s = np.kron(np.eye(P**2),np.array((1,1,1,1)))
    prob3d = np.reshape((s.dot((psiN.conjugate()*psiN).real)),(P,P))

    fig = figure()
    matplotlib.pyplot.imshow(prob3d.real, cmap='hot', interpolation='nearest')
    savefig('HeatMapShouldBeSymmetric_{:03d}.png'.format(frame))

# show()

