"""
file: plot_eigDensity.py

  Generate eigenvalue density plots.
  
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
    log,
    asarray,
    real,
    imag,
    sqrt,
    sin,
    cos,
    mean,
    append,
    diag,
    mod,
    count_nonzero,
)
from numpy.linalg import inv, cond
from scipy.linalg import expm
from matplotlib.pyplot import (
    axes,
    subplots,
    subplots_adjust,
    xticks,
    yticks,
    text,
    savefig,
)
from dmdc.dmd import MUL
from dmdc.plot_util import genTraj, doDMD, visSortedEig
from sys import stdout

# Constant parameters
dt = 0.05
sl = [3, 9]
mag = 0.05
Ll = [50, 10, 2]
Ntrajl = [2, 10, 50]
lb = [0.5, 1.0]
repeat = 300


def system_matrix_A(lams, theta, phi, s):
    Q = asarray(
        [
            [1, 0, sin(theta) * cos(phi)],
            [0, 1, sin(theta) * sin(phi)],
            [0, 0, cos(theta)],
        ]
    )
    S = asarray([[s, 0, 0], [0, 1, 0], [0, 0, 1]])
    Sinv = asarray([[1 / s, 0, 0], [0, 1, 0], [0, 0, 1]])
    Lam = expm(
        asarray(
            [
                [real(lams[0]), imag(lams[0]), 0],
                [imag(lams[1]), real(lams[1]), 0],
                [0, 0, lams[2]],
            ]
        )
        * dt
    )
    A = MUL(Q, S, Lam, Sinv, inv(Q))
    return real(A)


def plot_cond(N, figName):
    print(figName)
    lams = (
        log(
            asarray(
                [0.5 * (sqrt(3) / 2 + 1j / 2.0), 0.5 * (sqrt(3) / 2 - 1j / 2.0), 0.8]
            )
        )
        / dt
    )
    thetal = [0.0, 0.0, 1.4, 1.4, 1.56]
    sl = [1.0, 0.1, 1.0, 0.1, 1.0]
    phi = thetal[0]
    Trajl = []
    Ac = []
    for theta, s, i in zip(thetal, sl, range(5)):
        A = system_matrix_A(lams, theta, phi, s)
        Traj, DEv_true = genTraj(
            A, N, dt, lams, repeat=repeat, mag=mag, Ntraj=max(Ntrajl), L=100, sums=[]
        )        
        Trajl.append(Traj)
        Ac.append(cond(A))
        print("Generating trajectories: %i/5" % (i + 1), end="\r")
        stdout.flush()
    print()

    if N == 1:
        cmaps = ["Purples", "Reds", "Purples"]
    elif N == 2:
        cmaps = [
            "Purples",
            "Greys",
            "Greens",
            "Blues",
            "Reds",
            "Oranges",
            "Greens",
            "Greys",
            "Purples",
        ]
    fig, ax = subplots(3, 5, figsize=(20, 12), num=figName)
    subplots_adjust(wspace=0.05, hspace=0.05)

    for j, L, Ntraj in zip(range(3), Ll, Ntrajl):
        for Traj, i, Aci in zip(Trajl, range(5), Ac):
            DEv_e, b_e, DModes_e = doDMD(Traj, Ntraj, L)
            cond_obs = mean(
                [cond(MUL(m, diag(v), inv(m))) for v, m in zip(DEv_e, DModes_e)]
            )
            ax[0, i].set_title(
                "$\\theta={:.2f}$, s={:.2f} \n $\kappa(\mathbf{{A}}) = {:.0e}$\n $\kappa(\\tilde \mathbf{{A}})={:.0e}$".format(
                    thetal[int(i / 2)], sl[mod(i, 2)], Aci, cond_obs
                ),
                fontsize=26,
            )
            axes(ax[j, i])
            v_true = Traj.DEv_true
            realNum = count_nonzero(v_true.imag == 0)
            imagNum = int((len(v_true) - realNum) / 2)
            lst = append(range(-1, -imagNum - 1, -1), range(imagNum))
            lst = append(lst, range(imagNum, imagNum + realNum))
            ax[j, i].grid(1, linestyle="--")
            visSortedEig(DEv_e, b_e, lst, cmaps, rlabels=lb)
            yticks(fontsize=24)
            xticks(fontsize=24)
            if N == 1:
                ax[j, i].set_xlim([-0.1, 1.1])
                ax[j, i].set_ylim([-0.6, 0.6])
            if N == 2:
                ax[j, i].set_xlim([-0.45, 1.15])
                ax[j, i].set_ylim([-0.8, 0.8])
            if j != 2:
                ax[j, i].set_xticklabels([])
            if i != 0:
                ax[j, i].set_yticklabels([])
            ax[j, i].plot(v_true.real, v_true.imag, "x", c="r", markersize=10)
            print("Doing DMD: %i/15" % (j * 5 + i + 1), end="\r")
            stdout.flush()
        ax[j, 0].set_ylabel("L=%i" % L, fontsize=26)
    print()
    if N == 1:
        text(-5.4, 3.2, "Im", fontsize=24, weight="bold")
        text(1.2, -0.8, "Re", fontsize=24, weight="bold")
    if N == 2:
        text(-7.5, 4.3, "Im", fontsize=24, weight="bold")
        text(1.2, -1.1, "Re", fontsize=24, weight="bold")
    print("Show plots.")
    savefig("cond-density-order" + str(N) + ".pdf")


def plot_eig():
    """Density contour plots"""
    print("Figure 6")
    N = 1
    klt = [0.2, 0.5, 0.8]
    Trajl = []
    for k, i in zip(klt, range(3)):
        lams = (
            log(
                asarray(
                    [0.5 * (sqrt(3) / 2 + 1j / 2.0), 0.5 * (sqrt(3) / 2 - 1j / 2.0), k]
                )
            )
            / dt
        )
        A = system_matrix_A(lams, 0, 0, 1)
        Traj, DEv_true = genTraj(
            A,
            N,
            dt,
            lams,
            repeat=repeat,
            mag=mag,
            Ntraj=max(Ntrajl),
            L=max(Ll),
            sums=[],
        )
        Trajl.append(Traj)
        print("Generating trajectories: %i/3" % (i + 1), end="\r")
        stdout.flush()
    print()
    cmaps = ["Purples", "Reds", "Purples"]
    fig, ax = subplots(3, 6, figsize=(24, 12), num="Figure 6")
    subplots_adjust(wspace=0.05, hspace=0.05)

    for j, L, Ntraj in zip(range(3), Ll, Ntrajl):
        for k, i, Traj in zip(klt, range(4), Trajl):
            print(Traj, Ntraj, L)
            DEv_e, b_e, _ = doDMD(Traj, Ntraj, L)
            ax[0, i].set_title(r"$\lambda_1=%.1f$" % k, fontsize=26)
            axes(ax[j, i])

            v_true = Traj.DEv_true
            realNum = count_nonzero(v_true.imag == 0)
            imagNum = int((len(v_true) - realNum) / 2)
            lst = append(range(-1, -imagNum - 1, -1), range(imagNum))
            lst = append(lst, range(imagNum, imagNum + realNum))
            visSortedEig(DEv_e, b_e, lst, cmaps, rlabels=lb)
            ax[j, i].grid(1, linestyle="--")
            if N == 1:
                ax[j, i].set_xlim([-0.1, 1.1])
                ax[j, i].set_ylim([-0.6, 0.6])
            if N == 2:
                ax[j, i].set_xlim([-0.5, 1.1])
                ax[j, i].set_ylim([-0.8, 0.8])
            if j != 2:
                ax[j, i].set_xticklabels([])
            if i != 0:
                ax[j, i].set_yticklabels([])
            ax[j, i].plot(v_true.real, v_true.imag, "x", c="r", markersize=10)
            yticks(fontsize=24)
            xticks(fontsize=24)
            print("Doing DMD: %i/9" % (j * 3 + i + 1), end="\r")
            stdout.flush()
        ax[j, 0].set_ylabel("L=%i" % L, fontsize=26)
    print()

    N = 2
    Trajl = []
    cmaps = [
        "Purples",
        "Greys",
        "Greens",
        "Blues",
        "Reds",
        "Oranges",
        "Greens",
        "Greys",
        "Purples",
    ]
    for k, i in zip(klt, range(3)):
        lams = (
            log(
                asarray(
                    [0.5 * (sqrt(3) / 2 + 1j / 2.0), 0.5 * (sqrt(3) / 2 - 1j / 2.0), k]
                )
            )
            / dt
        )
        A = system_matrix_A(lams, 0, 0, 1)
        Traj, DEv_true = genTraj(
            real(A),
            N,
            dt,
            lams,
            repeat=repeat,
            mag=mag,
            Ntraj=max(Ntrajl),
            L=max(Ll),
            sums=[],
        )
        Trajl.append(Traj)
        print("Generating trajectories: %i/3" % (i + 1), end="\r")
        stdout.flush()
    print()
    for j, L, Ntraj in zip(range(3), Ll, Ntrajl):
        for k, i, Traj in zip(klt, range(3, 6), Trajl):
            DEv_e, b_e, _ = doDMD(Traj, Ntraj, L)
            ax[0, i].set_title(r"$\lambda_1=%.1f$" % k, fontsize=26)
            axes(ax[j, i])
            v_true = Traj.DEv_true
            realNum = count_nonzero(v_true.imag == 0)
            imagNum = int((len(v_true) - realNum) / 2)
            lst = append(range(-1, -imagNum - 1, -1), range(imagNum))
            lst = append(lst, range(imagNum, imagNum + realNum))
            visSortedEig(DEv_e, b_e, lst, cmaps, rlabels=lb)
            ax[j, i].grid(1, linestyle="--")
            ax[j, i].set_xlim([-0.45, 1.15])
            ax[j, i].set_ylim([-0.8, 0.8])
            if j != 2:
                ax[j, i].set_xticklabels([])

            if i != 5:
                ax[j, i].set_yticklabels([])
            else:
                ax[j, i].yaxis.tick_right()
            ax[j, i].plot(v_true.real, v_true.imag, "x", c="r", markersize=10)
            yticks(fontsize=24)
            xticks(fontsize=24)
            print("Doing DMD: %i/9" % (j * 3 + i - 2), end="\r")
            stdout.flush()
        ax[j, 0].set_ylabel("L=%i" % L, fontsize=26)
    print()
    text(-9.2, 4.3, "Im", fontsize=24, weight="bold")
    text(1.3, -1.1, "Re", fontsize=24, weight="bold")
    print("Show plots.")
    savefig("eig-density.pdf")


if __name__ == "__main__":
    from numpy.random import seed

    seed(2333)
    plot_cond(1, "Figure 2")
    # show()
    plot_cond(2, "Figure 4")
    # show()
    plot_eig()
    # show()
