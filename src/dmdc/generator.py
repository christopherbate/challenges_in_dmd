""" 
File: generator.py
Generate a linear system and get its monomial observations.

(c) 2020,2021 Ziyou Wu

Work on this code was funded by ARO MURI W911NF-17-1-0306 "From Data-Driven
Operator Theoretic Schemes to Prediction, Inference, and Control of Systems",
and government usage rights are reserved as per that funding agreement.

All other usage is governed by the GPL 3.0 license as specified below.

generator.py is free software: you can redistribute it and/or modify
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
    asarray,
    exp,
    dot,
    empty,
    concatenate,
    array,
    linspace,
    argsort,
    where,
    imag,
)
from numpy.random import normal
from itertools import combinations


def findPairs(lst, K, N):
    return [pair for pair in combinations(lst, N) if sum(pair) == K]


class trajectory:
    def __init__(self, A, N, dt, lams, repeat=100, sys_noise=0, obs_noise=0, sums=[]):
        """
        Repeatedly generates trajectories for linear system with monomial observables

        INPUT:
            A -- discrete linear system dynamics matrix
            N -- observables are monomials up to order N
            dt -- discrete time step
            lams -- eigenvalues of the linear
            repeat -- number of trajectories
            sys_noise -- system noise
            obs_noise -- observation noise
        """

        self.A = A
        self.sys_noise = sys_noise
        self.obs_noise = obs_noise
        self.dt = dt
        self.N = N
        D = len(lams)
        lst = list(range(0, D)) * D
        if len(sums) == 0:
            for n in range(N):
                for ss in findPairs(lst, n + 1, D):
                    if ss not in sums and sum(ss) != 0:
                        sums.append(ss)
        print("sums:")                        
        self.sums = sums  # order N, dim D
        self.L = []
        self.X = []  # state list: NUM trajectroies x L x D x repeat
        self.Y = []  # observation list: NUM trajectroies x L x D_obs x repeat
        self.ts = []
        self.Xeigvals = lams
        self.Yeigvals = self.true_eigs()  # eigenvalues in lifted observation space
        self.repeat = repeat
        self.Yamp = []
        self.DEv_true = self.sortTrueEig(exp(dt * self.Yeigvals))
        # self.A.extend(A for i in range(self.repeat))
        # self.A = asarray(self.A)

    def sortTrueEig(self, v):
        """First sort by imag, then by real"""

        arg = argsort(v.imag)
        v = v[arg]
        vv = v[where(imag(v) == 0)]
        v[where(imag(v) == 0)] = vv[argsort(v[where(imag(v) == 0)])]
        return v

    def add_trajectory(self, L, x0):
        """
        Generate linear trajectory with specified parameters
        INPUT:
            N -- dimension of the system
            L -- length of the trajectory
            A -- flow matrix
            x0 -- initial point
            b -- amplitude
        OUTPUT:
            trajectory in LxN array
        """
        # x0 random initialization
        # print("add trajectory: repeat ", self.repeat)
        xt = []
        xt.extend(list(x0) for i in range(self.repeat))
        xt = asarray(xt).T
        Yt = []
        #  xt (3, 300)
        
        # Trajectory of the linear system
        # (100, 3, 300)
        x = empty((L, len(self.Xeigvals), self.repeat))
        x[0, :, :] = xt

        Yt.append([self.observ(x[0], noise=self.obs_noise)])
        for i in range(L - 1):            
            xt = dot(self.A, xt) + normal(0, self.sys_noise, xt.shape)            
            Yt.append([self.observ(x[i + 1], noise=self.obs_noise)])
        self.X.append(x)
        self.Y.append(concatenate(Yt, axis=0))
        self.L.append(L)
        self.ts.append(linspace(0, self.dt * (L - 1), L))
        # self.Yamp.append(self.observ(b))

    def observ(self, x, noise=0):
        y = []
        y.append(self.extend(x, noise))
        print("objs", y[-1].shape)
        return concatenate(y)

    def extend(self, x, noise):
        y = []
        for s in self.sums:
            yy = 1            
            for xx, ss in zip(x, s):
                print("xx", xx.shape)
                yy *= xx**ss
            y.append(yy)
        y = array(y)
        y += normal(0, noise, y.shape)
        return y

    def true_eigs(self):
        y = []
        for s in self.sums:
            if sum(s) == 0:
                pass
            else:
                yy = 0
                for xx, ss in zip(self.Xeigvals, s):
                    yy += xx * ss
                y.append(yy)
        return asarray(y)
