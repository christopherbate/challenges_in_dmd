"""
file: plot_appendix.py

  Generate plots from the appendix
  
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
from numpy import (
    log, asarray, real, imag, sqrt, append, identity, exp, zeros, count_nonzero
)
from scipy.linalg import expm
from matplotlib.pyplot import (
  axes, subplots,subplots_adjust, xticks, yticks, text, savefig
  )
#from plot_util import *
from dmdc.plot_eigDensity import system_matrix_A
from dmdc.plot_util import genTraj, doDMD, visSortedEig, dooptDMD
from sys import stdout


#Constant parameters
dt = 0.05
sl = [3,9]
mag = 0.05
Ll = [50,10,2]
Ntrajl = [2,10,50]
lb = [0.5,1.0]
repeat = 500

def plot_nonresonant():
    "Non resonant 9d plot"
    print('Figure 9')
    #!!! klt = [[4, 10], [5, 5], [10, 3]]
    lamsl = [log(asarray([0.43-0.25j, 0.124-0.22j, 0.087-0.05j, 0.04, 0.21, 0.25, \
                     0.087+0.05j, 0.124+0.22j, 0.43+0.25j]))/dt ,\
        log(asarray([0.43-0.25j, 0.124-0.22j, 0.22-0.125j,\
                    0.25, 0.24, 0.5, 0.22+0.125j, 0.124+0.22j, 0.43+0.25j      ]))/dt,\
        log(asarray([0.43-0.25j, 0.124-0.22j, 0.35-0.2j,\
                    0.25, 0.643, 0.82, 0.35+0.2j, 0.124+0.22j, 0.43+0.25j      ]))/dt]
    Trajl = []

    for lams,j in zip(lamsl, range(3)):
        A = zeros((9,9))
        i=0
        A[0:4, 0:4] = asarray([[real(lams[i]), imag(lams[i]), 1, 0], [imag(lams[-i-1]), real(lams[-i-1]), 0,1], \
                        [0,0, real(lams[i+1]), imag(lams[i+1])], [0,0, imag(lams[-i-2]), real(lams[-i-2])]])
        i=2
        A[4:7, 4:7] = real(asarray([[real(lams[i]), imag(lams[i]), 1], [imag(lams[-i-1]), real(lams[-i-1]), 0], \
                        [0,0, lams[i+1]]]))

        for i in [4,5]:
            A[i+3, i+3] = real(lams[i])
        A = real(expm(A*dt))
        Traj, _ = genTraj(A, 1, dt, lams, repeat=repeat, mag=mag, Ntraj=max(Ntrajl), L=max(Ll), sums=identity(9))
        Trajl.append(Traj)
        print('Generating trajectories: %i/3' % (j+1), end='\r')
        stdout.flush()
    print()

    cmaps = ['Purples','Greys','Greens','Blues','Reds', 'Oranges', \
                          'Greens','Greys','Purples' ]

    fig,ax = subplots(3,3, figsize=(12,12), num='Figure 9')
    subplots_adjust(wspace=0.05, hspace=0.05)

    for j, L, Ntraj in zip(range(3), Ll, Ntrajl):
        for i, Traj in zip(range(3), Trajl):
            DEv_e, b_e,_  = doDMD(Traj, Ntraj, L)
            axes(ax[j, i])
            v_true = Traj.DEv_true
            realNum = count_nonzero(v_true.imag==0)
            imagNum = int((len(v_true) - realNum)/2)
            lst = append(range(-1, -imagNum-1, -1), range(imagNum))
            lst = append(lst, range(imagNum, imagNum+realNum))
            visSortedEig(DEv_e, b_e, lst, cmaps, rlabels=lb)
            ax[j,i].grid(1, linestyle='--')
            ax[j, i].set_xlim([-0.45, 1.15])
            ax[j, i].set_ylim([-0.8, 0.8])
            if j!=2:
                ax[j, i].set_xticklabels([])
            if i!=0:
                ax[j, i].set_yticklabels([])
            ax[j, i].plot(v_true.real, v_true.imag, 'x', c='r', markersize=10)
            yticks(fontsize=24)
            xticks(fontsize=24)
            print('Doing DMD: %i/9' % (j*3+i+1), end='\r')
            stdout.flush()
        ax[j,0].set_ylabel('L=%i'%L, fontsize=26)
    print()
    text(-4.1, 4.2, 'Im', fontsize=24, weight='bold')
    text(1.2, -1.1, 'Re', fontsize=24, weight='bold')
    savefig('nonresonant.pdf')


def plot_otherMethods(N, theta, s, figName):
    print(figName)
    k = 0.8
    lams = log(asarray([0.5*(sqrt(3)/2+1j/2.), 0.5*(sqrt(3)/2-1j/2.), k]))/dt
    phi=0
    A = system_matrix_A(lams, theta, phi, s)
    Traj, DEv_true = genTraj(real(A), N, dt, lams, mag=0.05, \
                              repeat=repeat, Ntraj=max(Ntrajl), L=max(Ll), sums=[])
    print('Generating trajectories: 1/1')
    if N==1:
        cmaps = ['Purples','Reds','Purples']
    elif N==2:
        cmaps = ['Purples','Greys','Greens','Blues','Reds', 'Oranges', \
                          'Greens','Greys','Purples' ]
    fig,ax = subplots(3,4, figsize=(16,12), num=figName)
    subplots_adjust(wspace=0.05, hspace=0.05)

    for j, L, Ntraj in zip(range(3), Ll, Ntrajl):
        for i, m, ml in zip(range(4), ['exact', 'fb', 'tls', 'opt'], ['Exact','FB','TLS','Optimized']):
            if i==3 and j!=2:
                DEv_e, b_e  = dooptDMD(Traj, 1, L, sl[N-1])
                DEv_e = exp(DEv_e*dt)
            elif i==3 and j==2:
                pass
            else:
                DEv_e, b_e,_  = doDMD(Traj, Ntraj, L, method=m)

            ax[0,i].set_title(ml, fontsize=20)
            axes(ax[j, i])

            v_true = Traj.DEv_true
            realNum = count_nonzero(v_true.imag==0)
            imagNum = int((len(v_true) - realNum)/2)
            lst = append(range(-1, -imagNum-1, -1), range(imagNum))
            lst = append(lst, range(imagNum, imagNum+realNum))
            visSortedEig(DEv_e, b_e, lst, cmaps, rlabels=lb)
            ax[j,i].grid(1, linestyle='--')
            if N==1:
                ax[j, i].set_xlim([-0.1, 1.1])
                ax[j, i].set_ylim([-0.6, 0.6])
            if N==2:
                ax[j, i].set_xlim([-0.45, 1.15])
                ax[j, i].set_ylim([-0.8, 0.8])
            if j!=2:
                ax[j, i].set_xticklabels([])
            if i!=0:
                ax[j, i].set_yticklabels([])
            ax[j, i].plot(v_true.real, v_true.imag, 'x', c='r', markersize=10)

            if i==3 and j==2:
                ax[j,i].cla()
                ax[j,i].set_yticklabels([])
                ax[j,i].set_xticklabels([])
            yticks(fontsize=24)
            xticks(fontsize=24)
            print('Doing DMD: %i/12' % (j*4+i+1), end='\r')
            stdout.flush()
        ax[j,0].set_ylabel('L=%i'%L,fontsize=26)
    print()
    text(-3.3, 3.2, 'Im', fontsize=24, weight='bold')
    text(1.0, -0.1, 'Re', fontsize=24, weight='bold')
    savefig('appendix-order'+str(N)+'s'+str(s)+'.pdf')
    #show()

def plot_5d():
    print('Figure 14')
    N = 1
    k = 0.8
    lams = log(asarray([0.5*(sqrt(3)/2+1j/2.), (1+0.1j)/sqrt(1+0.1**2), k, (1-0.1j)/sqrt(1+0.1**2), 0.5*(sqrt(3)/2-1j/2.)]))/dt
    A = zeros((5,5))
    A[0:4, 0:4] = asarray([[real(lams[0]), imag(lams[0]), 1, 0], [imag(lams[-1]), real(lams[-1]), 0,1], \
                    [0,0, real(lams[1]), imag(lams[1])], [0,0, imag(lams[-2]), real(lams[-2])]])
    A[4,4] = real(lams[2])
    A = real(expm(A*dt))
    Traj, DEv_true = genTraj(real(A), N, dt, lams, mag=0.05, \
                              repeat=repeat, Ntraj=max(Ntrajl), L=max(Ll))
    print('Generating trajectories: 1/1')

    cmaps = ['Purples','Oranges','Blues','Oranges','Purples' ]
    fig,ax = subplots(3,4, figsize=(16,12), num='Figure 14')
    subplots_adjust(wspace=0.05, hspace=0.05)

    for j, L, Ntraj in zip(range(3), Ll, Ntrajl):
        for i, m, ml in zip(range(4), ['exact', 'fb', 'tls', 'opt'], ['Exact','FB','TLS','Optimized']):
            if i==3 and j!=2:
                DEv_e, b_e  = dooptDMD(Traj, 1, L, 5)
                DEv_e = exp(DEv_e*dt)
            elif i==3 and j==2:
                pass
            else:
                DEv_e, b_e,_  = doDMD(Traj, Ntraj, L, method=m)

            ax[0,i].set_title(ml, fontsize=20)
            axes(ax[j, i])

            v_true = Traj.DEv_true
            realNum = count_nonzero(v_true.imag==0)
            imagNum = int((len(v_true) - realNum)/2)
            lst = append(range(-1, -imagNum-1, -1), range(imagNum))
            lst = append(lst, range(imagNum, imagNum+realNum))
            visSortedEig(DEv_e, b_e, lst, cmaps, rlabels=lb)
            ax[j,i].grid(1, linestyle='--')
            ax[j, i].set_xlim([-0.1, 1.1])
            ax[j, i].set_ylim([-0.6, 0.6])
            if j!=2:
                ax[j, i].set_xticklabels([])
            if i!=0:
                ax[j, i].set_yticklabels([])
            ax[j, i].plot(v_true.real, v_true.imag, 'x', c='r', markersize=10)

            if i==3 and j==2:
                ax[j,i].cla()
                ax[j,i].set_yticklabels([])
                ax[j,i].set_xticklabels([])
            yticks(fontsize=24)
            xticks(fontsize=24)
            print('Doing DMD: %i/12' % (j*4+i+1), end='\r')
            stdout.flush()
        ax[j,0].set_ylabel('L=%i'%L,fontsize=26)
    print()
    text(-3.3, 3.2, 'Im', fontsize=24, weight='bold')
    text(1.0, -0.1, 'Re', fontsize=24, weight='bold')
    savefig('appendix-5d.pdf')

if __name__=="__main__":
    from numpy.random import seed
    seed(2333)
    plot_nonresonant()
    #show()
    plot_otherMethods(1, 0, 1, 'Figure 10')
    plot_otherMethods(2, 0, 1, 'Figure 11')
    plot_otherMethods(1, 1.4, 0.1, 'Figure 12')
    plot_otherMethods(2, 1.4, 0.1, 'Figure 13')
    plot_5d()
    #show()
