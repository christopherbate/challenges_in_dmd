"""
file: plot_test.py

  Unit test for plotting functions
  
(c) 2020,2021 Ziyou Wu

Work on this code was funded by ARO MURI W911NF-17-1-0306 "From Data-Driven
Operator Theoretic Schemes to Prediction, Inference, and Control of Systems",
and government usage rights are reserved as per that funding agreement.

All other usage is governed by the GPL 3.0 license as specified below.

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from numpy import log, asarray, real, imag, sqrt, sin, cos, append, count_nonzero
from numpy.linalg import inv
from scipy.linalg import expm
from matplotlib.pyplot import figure, plot, xlim, ylim, show, axes
from dmd import MUL
from plot_util import genTraj, doDMD, visSortedEig
from mpl_toolkits import mplot3d

if __name__!="__main__":
  raise RuntimeError("Run this as a script")
  
dt = 0.05 #time stepsize
repeat = 300
N=1 #order of observable monomials
mag = 0.05 #standard deviation of system and observation noise
Ntraj=10 #number of trajectories
L=10 #trajectory length

#Construct system evolution matrix A (discrete time)
theta = 0. #in [0, pi/2)
phi = 0. #in [0, pi/2)
s = 1.0 #in (0,1]
lams = log(asarray([0.5*(sqrt(3)/2+1j/2.), 0.5*(sqrt(3)/2-1j/2.), 0.8]))/dt #eigenvalues
Q = asarray([[1, 0 ,sin(theta)*cos(phi)], [0,1,sin(theta)*sin(phi)], [0,0,cos(theta)]])
S = asarray([[s,0,0],[0,1,0],[0,0,1]])
Sinv = asarray([[1/s,0,0],[0,1,0],[0,0,1]])
Lam = expm(asarray([[real(lams[0]), imag(lams[0]) ,0],\
                  [imag(lams[1]), real(lams[1]),0],\
                  [0,0,lams[2]]])*dt)
A = real(MUL(Q, S, Lam, Sinv, inv(Q)))

Traj, DEv_true = genTraj(A, N, dt, lams, repeat=repeat, mag=mag, magsys=mag, Ntraj=Ntraj, L=L)

if 1:
    figure()
    ax = axes(projection='3d')
    for i in range(Ntraj):
        ax.plot3D(Traj.X[i][:,0,0], Traj.X[i][:,1,0], Traj.X[i][:,2,0], '*-')
    show()

figure()
vl, bl,_ = doDMD(Traj, Ntraj, L, method='exact') #Method choices: 'exact', 'fb', 'tls'
if N==1:
    cmaps = ['Purples','Reds','Purples']
elif N==2:
    cmaps = ['Purples','Greys','Greens','Blues','Reds', 'Oranges', 'Greens','Greys','Purples' ]

v_true = Traj.DEv_true
realNum = count_nonzero(v_true.imag==0)
imagNum = int((len(v_true) - realNum)/2)
lst = append(range(-1, -imagNum-1, -1), range(imagNum))
lst = append(lst, range(imagNum, imagNum+realNum))
visSortedEig(vl, bl, lst, cmaps, rlabels=[0.5,1.0])
plot(v_true.real, v_true.imag, 'x', c='r', markersize=10)
xlim([-0.1, 1.1])
ylim([-0.6, 0.6])
show()
