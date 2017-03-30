from numpy import *
from matplotlib.pyplot import *
import pylab as p
#import matplotlib.axes3d as p3
# import mpl_toolkits.mplot3d.axes3d as p3
# from mpl_toolkits.mplot3d import Axes3D

N = 8 #number of steps
P = 2*N+1 #number of positions in each direction

coin00 = array([1,0,0,0]) # |00>
coin01 = array([0,1,0,0]) # |01>
coin10 = array([0,0,1,0]) # |10>
coin11 = array([0,0,0,1]) # |11>

C0000 = outer(coin00,coin00) # |00><00|
C0001 = outer(coin00,coin01) # |00><01|
C0010 = outer(coin00,coin10) # |00><10|
C0011 = outer(coin00,coin11) # |00><11|
C0100 = outer(coin01,coin00) # |01><00|
C0101 = outer(coin01,coin01) # |01><01|
C0110 = outer(coin01,coin10) # |01><10|
C0111 = outer(coin01,coin11) # |01><11|
C1000 = outer(coin10,coin00) # |10><00|
C1001 = outer(coin10,coin01) # |10><01|
C1010 = outer(coin10,coin10) # |10><10|
C1011 = outer(coin10,coin11) # |10><11|
C1100 = outer(coin11,coin00) # |11><00|
C1101 = outer(coin11,coin01) # |11><01|
C1110 = outer(coin11,coin10) # |11><10|
C1111 = outer(coin11,coin11) # |11><11|

C_hat = (C0000 + C0001 + C0010 + C0011 + C0100 - C0101 + C0110 - C0111 + C1000 + C1001 - C1010 - C1011 + C1100 - C1101 - C1110 + C1111)/2.0

ShiftLeft  = kron(eye(P),roll(eye(P), 1, axis=0))
ShiftRight = kron(eye(P),roll(eye(P), -1, axis=0))
ShiftDown  = kron(roll(eye(P), -1, axis=0),eye(P))
ShiftUp    = kron(roll(eye(P), 1, axis=0),eye(P))

S_hat = kron(ShiftRight, C0000) + kron(ShiftUp, C0101) + kron(ShiftLeft, C1010) + kron(ShiftDown,C1111)

U = S_hat.dot(kron(eye(P*P),C_hat))

posn0 = zeros(P*P)
posn0[N*P+N] = 1 # array indexing starts from (0) so index (N*P+N) is the central position
psi0 = kron(posn0,(coin00+coin01*1j+coin10*1j -coin11)/2) #Initial spin 


psiN = linalg.matrix_power(U, N).dot(psi0)

prob = empty(P*P)
z = zeros((P,P))
for k in range(P):
    for m in range(P):
        posn = zeros(P*P)
        posn[k*P+m] = 1
        M_hat_km = kron( outer(posn,posn), eye(4)) #the 4x4 identity since we have a 4-regular graph
        proj = M_hat_km.dot(psiN)
        prob[k*P+m] = proj.dot(proj.conjugate()).real
        z[k,m] = prob[k*P+m]

#try to reshape the array to size P by P
prob3d = reshape(prob,(P,P))


fig = figure()
# ax = fig.gca(projection='3d')
#ax = fig.add_subplot(111)
x = arange(P)
y = arange(P)

print(z)
matplotlib.pyplot.imshow(z, cmap='hot', interpolation='nearest')
#plot(arange(P), prob, 'o')
#loc = range (0, P, 20) #location of ticks
#xticks(loc)
#xlim(0, P)
#ax.set_xticklabels(range (-N,N+1, 20))
show()
