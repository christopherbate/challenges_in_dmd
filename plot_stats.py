"""
file: plot_stats.py

  Generate statistics plots
  
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
    log, asarray, real, sqrt, append, linspace, pi, std, where,
    argsort, shape
    )
from numpy.linalg import cond, norm
from matplotlib.pyplot import (
  subplots,subplots_adjust, xticks, yticks, savefig
  )
from plot_eigDensity import system_matrix_A
from plot_util import genTraj, doDMD
from sys import stdout

mag = 0.05 #1/SNR
Ntrajl = [2,10,50]
Ll = [50,10,2]
dt = 0.05
repeat = 300
phil = [0, pi/4, pi/2-0.01]
sl = [1,0.5,0.1]
thetal = [0, 1.1, 1.2, 1.3, 1.5,1.52,1.53,1.557,1.56]
kl = append(linspace(0.001,0.1,10), linspace(0.1,1,35))

def DEv_helper(N, sl, phil, thetal, kl):
    Trajl = []
    Ac = []
    DEvl = []
    i = 0
    Num = len(kl)*len(sl)*len(phil)*len(thetal)
    for k in kl:
        lams = log(asarray([0.5*(sqrt(3)/2+1j/2.), 0.5*(sqrt(3)/2-1j/2.), k]))/dt
        for s in sl:
            for phi in phil:
                for theta in thetal:
                    A = system_matrix_A(lams, theta, phi, s)
                    Traj, DEv_true = genTraj(real(A), N, dt, lams, repeat=repeat, mag=mag, Ntraj=max(Ntrajl), L=max(Ll), sums=[])
                    DEvl.append(Traj.DEv_true)
                    Trajl.append(Traj)
                    Ac.append(cond(A))
                    i+=1
                    print('Generating trajectories: %i/%i' %(i, Num), end='\r')
                    stdout.flush()
    print()
    Ac = asarray(Ac)
    DEv_ell = []
    b_ell = []
    i = 0
    for Ntraj, L in zip(Ntrajl, Ll):
        DEv_el = []
        b_el = []
        for Traj in Trajl:
            DEv_e, b_e,_ = doDMD(Traj, Ntraj, L)
            DEv_el.append(DEv_e)
            b_el.append(b_e)
            i+=1
            print('Doing DMD: %i/%i' %(i, Num*3), end='\r')
            stdout.flush()
        DEv_ell.append(DEv_el)
        b_ell.append(b_el)
    print()
    return DEv_ell, Ac, Trajl


def plot_eig1():
    #Figure 7
    print('Figure 7')
    DEv_ell, Ac, Trajl = DEv_helper(1, [1.0], [0.], [0.], kl)
    print('Generating plots.')
    fig,ax = subplots(1,3, figsize=(12,4), num='Figure 7')
    subplots_adjust(wspace=0.05, hspace=0.1)

    DEv_stdl = []
    DEv_eml = []

    for DEv_el in DEv_ell:
        DEv_std = []
        DEv_em = []
        for i, Traj in zip(range(len(DEv_el)), Trajl):
            DEv_std.append(std(DEv_el[i], axis=0))
            DEv_em.append(norm(DEv_el[i] - Traj.DEv_true, axis=0)/repeat)
        DEv_stdl.append(asarray(DEv_std))
        DEv_eml.append(asarray(DEv_em))

    mks = ['^', 'o', 's']
    for DEv_std, DEv_em,mk in zip(DEv_stdl,DEv_eml, mks):
        ax[1].plot(kl,DEv_std[:,0], '*m',marker=mk, linestyle='None', markerfacecolor='None' )
        ax[1].plot(kl,DEv_std[:,2], '.m',marker=mk, linestyle='None', markerfacecolor='None' )
        ax[0].plot(kl,DEv_std[:,1], '.r',marker=mk, linestyle='None', markerfacecolor='None' )
        ax[0].tick_params(axis='y')

        ax[0].set_ylim([0,0.2])
        ax[1].set_ylim([0,0.2])
        ax[1].set_yticks([])
    ax[0].set_xlabel(r'$\lambda_1$',fontsize=14)
    ax[1].set_xlabel(r'$\lambda_1$',fontsize=14)
    ax[0].set_title(r'$\lambda_1$',fontsize=14)
    ax[0].legend(['L=50', 'L=10', 'L=2'],loc=2,fontsize=12)
    ax[1].set_title(r'$\lambda_2, \bar{\lambda}_2$',fontsize=14)
    ax[0].set_ylabel('Std',fontsize=14)

    ax3 = ax[2]
    gar_shape = []
    for DEv_el in DEv_ell:
        gar_shape.append([shape(DEv_eli)[0] for DEv_eli in DEv_el])
    gar_shape = asarray(gar_shape)
    for i, mk in zip(range(3), mks):
        ax3.plot(kl, (repeat-asarray(gar_shape[i]))/float(repeat) *100. , marker=mk, linestyle='None', markerfacecolor='None')
    ax3.set_ylabel('Discarded percentage [%]',fontsize=14)
    ax3.set_xlabel('$\lambda_1$',fontsize=14)
    ax3.yaxis.tick_right()
    ax3.yaxis.set_label_position("right")
    savefig('eig-std-order1.pdf')


def plot_eig2():
    #Figure 8
    print('Figure 8')
    DEv_ell, Ac, Trajl = DEv_helper(2, [1.0], [0.], [0.], kl)
    fig,ax = subplots(2,4, figsize=(16,8), num='Figure 8')
    subplots_adjust(wspace=0.05, hspace=0.15)
    print('Generating plots.')
    DEv_stdl = []
    DEv_eml = []
    arg2l = []
    arg22bl = []
    for DEv_el in DEv_ell:
        DEv_std = []
        DEv_em = []
        arg2 = []
        arg22b = []
        for i, Traj in zip(range(len(DEv_el)), Trajl):
            DEv_std.append(std(DEv_el[i], axis=0))
            DEv_em.append(norm(DEv_el[i] - Traj.DEv_true, axis=0)/repeat)
            arg = [abs(v-0.25)>0.0001 for v in Traj.DEv_true]
            arg2.append(where(arg)[0])
            arg = [abs(v-0.25)<0.0001 for v in Traj.DEv_true]
            arg22b.append(where(arg)[0])

        DEv_stdl.append(asarray(DEv_std))
        DEv_eml.append(asarray(DEv_em))
        arg2l.append(asarray(arg2))
        arg22bl.append(asarray(arg22b))
    mks = ['^', 'o', 's']

    colors = ['m', 'grey', 'g', 'b', 'r', 'orange']

    names = [r'$\bar{\lambda}_2$', r'$\bar{\lambda}_2^2$', r'$\lambda_1\bar{\lambda}_2$',\
             r'$\lambda_1^2$',r'$\lambda_1$',\
            r'$\lambda_1\lambda_2$',r'$\lambda_2^2$', r'$\lambda_2$',r'$\lambda_2\bar{\lambda_2}$']


    for DEv_std, DEv_em,mk, arg2 in zip(DEv_stdl,DEv_eml, mks, arg2l):
        for i in range(2):
            ax[0,i].plot(kl, DEv_std[range(45),arg2[:,3+i]],c=colors[i+3], marker=mk, linestyle='None', markerfacecolor='None')
            ax[0,i].set_title(names[3+i], fontsize=20)
            ax[0,i].set_ylim([0,0.36])
            ax[0,i].set_xticks([])
            if i!=0:
                ax[0,i].set_yticks([])
        for i, cc in zip(range(3), colors):
            if i==2:
                ax[0,-1].plot(kl, DEv_std[:,i], c=cc, marker=mk, linestyle='None', markerfacecolor='None' )
                ax[0,-1].set_title(names[i]+', '+names[-i-2], fontsize=20)
                ax[0,-1].set_ylim([0,0.36])
                ax[0,-1].set_yticks([])
                ax[0,-1].set_xticks([])

            else:
                ax[1,i].plot(kl, DEv_std[:,i], c=cc, marker=mk, linestyle='None', markerfacecolor='None' )
                #ax[1,i].plot(kl, DEv_std[:,-(i+1)], '.', c=cc, linestyle=l)
                ax[1,i].set_title(names[i]+', '+names[-i-2], fontsize=20)
                ax[1,i].set_ylim([0,0.36])

                if i!=0:
                    ax[1,i].set_yticks([])

        ax[0,-2].plot(kl, DEv_std[range(45), arg22b[0]], c=colors[-1], marker=mk, linestyle='None', markerfacecolor='None' )
        ax[0,-2].set_title(names[-1], fontsize=20)
        ax[0,-2].set_ylim([0,0.36])
        ax[0,-2].set_yticks([])
        ax[0,-2].set_xticks([])

        ax[0,0].tick_params(axis='y')
        ax[1,0].tick_params(axis='y')
        ax[0,0].set_ylabel('Std', fontsize=20)
        ax[1,0].set_ylabel('Std', fontsize=20)
        yticks(fontsize=20)
        ax[0,0].legend(['L=50', 'L=10', 'L=2'], loc=2, fontsize=16)
        ax[-1,0].set_xlabel(r'$\lambda_1$', fontsize=20)
        ax[-1,1].set_xlabel(r'$\lambda_1$', fontsize=20)
        ax[-1,2].set_xlabel(r'$\lambda_1$', fontsize=20)
    ax[0,0].tick_params(axis='both', which='major', labelsize=16)
    ax[1,0].tick_params(axis='both', which='major', labelsize=16)
    ax[1,1].tick_params(axis='both', which='major', labelsize=16)

    gs = ax[1, 2].get_gridspec()
    for ax in ax[1, 2:]:
        ax.remove()
    ax3 = fig.add_subplot(gs[1, 2:])

    gar_shape = []
    for DEv_el in DEv_ell:
        gar_shape.append([shape(DEv_eli)[0] for DEv_eli in DEv_el])
    gar_shape = asarray(gar_shape)
    for i, mk in zip(range(3), mks):
        ax3.plot(kl, (repeat-asarray(gar_shape[i]))/float(repeat) *100. , marker=mk, linestyle='None', markerfacecolor='None')
    ax3.set_ylabel('Discarded percentage [%]', fontsize=20)
    ax3.set_xlabel('$\lambda_1$', fontsize=20)
    ax3.yaxis.tick_right()
    yticks(fontsize=16)
    xticks(fontsize=16)
    ax3.yaxis.set_label_position("right")
    savefig('eig-std-order2.pdf')


def plot_cond1():
    #Figure 3
    print('Figure 3')
    DEv_ell, Ac, Trajl = DEv_helper(1, sl, phil, thetal, [0.8])
    print('Generating plots.')
    fig,ax = subplots(1,3, figsize=(12,4), num='Figure 3')
    subplots_adjust(wspace=0.05, hspace=0.1)
    DEv_stdl = []
    DEv_eml = []
    for DEv_el in DEv_ell:
        DEv_std = []
        DEv_em = []
        for i, Traj in zip(range(len(DEv_el)), Trajl):
            DEv_std.append(std(DEv_el[i], axis=0))
            DEv_em.append(norm(DEv_el[i] - Traj.DEv_true, axis=0)/repeat)
        DEv_stdl.append(asarray(DEv_std))
        DEv_eml.append(asarray(DEv_em))
    mks = ['^', 'o', 's']
    subplots_adjust(wspace=0.05, hspace=0.05)
    for DEv_std, DEv_em,mk in zip(DEv_stdl,DEv_eml, mks):
        arg = argsort(Ac)[:-5]
        ax[1].plot(Ac[arg],DEv_std[arg,2], 'm', marker=mk, linestyle='None', markerfacecolor='None')
        ax[0].plot(Ac[arg],DEv_std[arg,1], 'r', marker=mk,linestyle = 'None',markerfacecolor='None')
        ax[0].tick_params(axis='y')
        ax[0].set_ylim([0,0.8])
        ax[1].set_ylim([0,0.8])
        ax[1].set_yticks([])
        ax[0].semilogx()
        ax[1].semilogx()

    ax[0].set_xlabel('condition number', fontsize=14)
    ax[1].set_xlabel('condition number', fontsize=14)
    ax[0].set_title(r'$\lambda_1$', fontsize=14)
    ax[0].legend(['L=50', 'L=10', 'L=2'], loc=2, fontsize=12)
    ax[1].set_title(r'$\lambda_2, \bar{\lambda}_2$', fontsize=14)
    ax[0].set_ylabel('Std', fontsize=14)
    ax3 = ax[2]
    gar_shape = []
    for DEv_el in DEv_ell:
        gar_shape.append([shape(DEv_eli)[0] for DEv_eli in DEv_el])
    gar_shape = asarray(gar_shape)
    for i, mk in zip(range(3), mks):
        ax3.plot(Ac[arg], (repeat-asarray(gar_shape[i][arg]))/float(repeat) *100. , marker=mk, linestyle='None', markerfacecolor='None')
    ax3.set_ylabel('Discarded percentage [%]', fontsize=14)
    ax3.set_xlabel('condition number', fontsize=14)
    ax3.semilogx()
    ax3.yaxis.set_label_position("right")
    ax3.yaxis.tick_right()
    savefig('cond-std-order1.pdf')


def plot_cond2():
    #Figure 5
    print('Figure 5')
    DEv_ell, Ac, Trajl = DEv_helper(2, sl, phil, thetal, [0.8])
    print('Generating plots.')
    fig,ax = subplots(2,4, figsize=(16,8), num="Figure 5")
    subplots_adjust(wspace=0.05, hspace=0.15)
    DEv_stdl = []
    DEv_eml = []
    arg2l = []
    arg22bl = []
    for DEv_el in DEv_ell:
        DEv_std = []
        DEv_em = []
        arg2 = []
        arg22b = []
        for i, Traj in zip(range(len(DEv_el)), Trajl):
            DEv_std.append(std(DEv_el[i], axis=0))
            DEv_em.append(norm(DEv_el[i] - Traj.DEv_true, axis=0)/repeat)
            arg = [abs(v-0.25)>0.0001 for v in Traj.DEv_true]
            arg2.append(where(arg)[0])
            arg = [abs(v-0.25)<0.0001 for v in Traj.DEv_true]
            arg22b.append(where(arg)[0])
        DEv_stdl.append(asarray(DEv_std))
        DEv_eml.append(asarray(DEv_em))
        arg2l.append(asarray(arg2))
        arg22bl.append(asarray(arg22b))

    mks = ['^', 'o', 's']
    colors = ['m', 'grey', 'g', 'b', 'r', 'orange']
    names = [r'$\bar{\lambda}_2$', r'$\bar{\lambda}_2^2$', r'$\lambda_1\bar{\lambda}_2$',\
             r'$\lambda_1^2$',r'$\lambda_1$',\
            r'$\lambda_1\lambda_2$',r'$\lambda_2^2$', r'$\lambda_2$',r'$\lambda_2\bar{\lambda_2}$']

    subplots_adjust(wspace=0.05, hspace=0.15)
    for DEv_std, DEv_em,mk, arg2 in zip(DEv_stdl,DEv_eml, mks, arg2l):
        for i in range(2):
            arg = argsort(Ac)[:-5]
            ax[0,i].plot(Ac[arg], DEv_std[arg,arg2[0,3+i]], c=colors[i+3], marker=mk, linestyle='None', markerfacecolor='None')
            ax[0,i].set_title(names[3+i], fontsize=20)
            ax[0,i].set_ylim([0,1.5])
            ax[0,i].semilogx()
            if i!=0:
                ax[0,i].set_yticks([])

        for i, cc in zip(range(3), colors):
            if i==2:
                ax[0,-1].plot(Ac[arg], DEv_std[arg,i], c=cc, marker=mk, linestyle='None', markerfacecolor='None' )
                ax[0,-1].set_title(names[i]+', '+names[-i-2], fontsize=20)
                ax[0,-1].set_ylim([0,0.36])
                ax[0,-1].set_yticks([])
                ax[0,-1].semilogx()
                ax[0,-1].set_xticklabels([])
            else:
                ax[1,i].plot(Ac[arg], DEv_std[arg,-(i+1)], c=cc, marker=mk, linestyle='None', markerfacecolor='None')
                ax[1,i].set_title(names[i]+', '+names[-i-2], fontsize=20)
                ax[1,i].set_ylim([0,1.5])
                ax[1,i].semilogx()
                if i!=0:
                    ax[1,i].set_yticks([])
                ax[0,i].set_xticklabels([])

        ax[0,-2].plot(Ac[arg], DEv_std[arg, arg22b[0]], c=colors[-1], marker=mk, linestyle='None', markerfacecolor='None')
        ax[0,-2].set_title(names[-1], fontsize=20)
        ax[0,-2].set_ylim([0,1.5])
        ax[0,-2].set_yticks([])
        ax[0,-2].semilogx()
        ax[0,-2].set_xticklabels([])
        ax[0,0].tick_params(axis='y')
        ax[1,0].tick_params(axis='y')
        ax[0,0].set_ylabel('Std', fontsize=20)
        ax[1,0].set_ylabel('Std', fontsize=20)
        ax[0,0].legend(['L=50', 'L=10', 'L=2'], loc=2, fontsize=16)
        ax[-1,0].set_xlabel(r'condition number', fontsize=20)
        ax[-1,1].set_xlabel(r'condition number', fontsize=20)
        ax[-1,2].set_xlabel(r'condition number', fontsize=20)
    ax[0,0].tick_params(axis='both', which='major', labelsize=16)
    ax[1,0].tick_params(axis='both', which='major', labelsize=16)
    ax[1,1].tick_params(axis='both', which='major', labelsize=16)

    gs = ax[1, 2].get_gridspec()
    for ax in ax[1, 2:]:
        ax.remove()
    ax3 = fig.add_subplot(gs[1, 2:])
    gar_shape = []
    for DEv_el in DEv_ell:
        gar_shape.append([shape(DEv_eli)[0] for DEv_eli in DEv_el])
    gar_shape = asarray(gar_shape)
    for i, mk in zip(range(3), mks):
        ax3.plot(Ac[arg], (repeat-asarray(gar_shape[i][arg]))/float(repeat) *100. , marker=mk, linestyle='None', markerfacecolor='None')
    ax3.set_ylabel('Discarded percentage [%]', fontsize=20)
    ax3.set_xlabel('condition number', fontsize=20)
    ax3.yaxis.tick_right()
    ax3.semilogx()
    ax3.tick_params(axis='both', which='major', labelsize=16)
    ax3.yaxis.set_label_position("right")
    savefig('cond-std-order2.pdf')


if __name__=="__main__":
    from numpy.random import seed
    seed(2333)
    plot_cond1()
    #show()
    plot_cond2()
    #show()
    plot_eig1()
    #show()
    plot_eig2()
    #show()
