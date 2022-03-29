"""
File: plot_util.py

Utility functions for plotting.

(c) 2008,2009,2018-2021 Shai Revzen
(c) 2020,2021 Ziyou Wu

This code is a subset of the plotmisc library published by BIRDS Lab, 
University of Michigan, and is not maintained.  If you want the current,
maintained code, it is available via:
git clone https://www.birds.eecs.umich.edu/plotmisc.git

Some of the plotting code in plot_util.py goes back to 2008, and was developed
as part of Revzen's thesis. This older code was released under the GPL 3.0
licence.

Work on new code was funded by ARO MURI W911NF-17-1-0306 "From Data-Driven
Operator Theoretic Schemes to Prediction, Inference, and Control of Systems",
and government usage rights are reserved as per that funding agreement.

All other usage is governed by the GPL 3.0 license as specified below.

plot_util.py is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, version 3 of the License.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from generator import trajectory
from dmd import dmd
from numpy import (
    zeros, ones, asarray, histogram, histogram2d, log, exp, shape, nansum,
    convolve, linspace, concatenate, pi, count_nonzero, where, argsort, seterr
    )
from numpy.linalg import norm, pinv
from numpy.random import uniform
from scipy.signal import windows
from matplotlib.pyplot import gca, gcf, plot, imshow
from optimalDMD import optdmd

class VisEig:
    """VisEig is a tool for visualizing distributions of complex numbers
    that are, generally speaking, around 0. It is typically used to show
    eigenvalue distributions of ensembles of matrices.

    Typical useage:
    >>> vi = VisEig()
    >>> [ vi.add(eigvals(M)) for M in listOfMatrices ]
    >>> vi.vis()
    """
    def __init__(self, N=63, rng=[[-2.0,2.0],[-2.0,2.0]], fishEye=True):
        self.N = N
        self.H = zeros((self.N,self.N))
        self.Hr = zeros(self.N)
        self.scl = 1
        self.fishy = fishEye
        self.rng = rng
        self.rB = None

    def clear(self):
        self.H[:] = 0
        self.Hr[:] = 0

    def fishEye( self, z ):
        z = asarray(z).copy().flatten() * self.scl
        if self.fishy:
            r = abs(z)
            b = r>1
            z[b]= z[b]/r[b]*(2-1/r[b])
        return z.real, z.imag

    def add( self, ev, wgt=1. ):
        x,y = self.fishEye(ev.flatten())
        H, self.iB,self.rB = histogram2d( y,x,self.N, range=[self.rng[1],self.rng[0]])
        if any(y==0):
            Hr = histogram( x[y==0], bins=self.N, range=self.rng[0]  )[0]
            self.Hr = self.Hr + wgt * Hr
        self.H = self.H + wgt * H

    def plt( self ):
        """
        Plot marginal density of real parts, and marginal density
        of real parts of numbers which were real to begin with

        return plot object handles
        """
        x = (self.rB[1:]+self.rB[:-1])/2.0
        tot = sum(self.H.flat)
        h = plot(x,sum(self.H,0)/tot,color=[0,0,1],linewidth=3)
        h.extend(plot(x,self.Hr/tot,color=[0.5,0.5,0.5],linewidth=2))
        return h

    def vis( self, ax=None, rlabels = [0.5,1], N=None):
        "Visualize density with color-coded bitmap"
        fig = gcf()
        vis = fig.get_visible()
        fig.set_visible(False)
        if ax == None:
            ax = gca()
        h = [imshow(log(1+self.H),interpolation='nearest',
            extent=[self.rB[0],self.rB[-1],self.iB[0],self.iB[-1]]
        )]
        self._decorate( ax, rlabels, dict(color=[1,1,1],linestyle='--') )
        fig.set_visible(vis)
        return h

    def vis1( self, rlabels = [0.5,1], N=8, Wr=None, Wi=None, **kw):
        """
        Visualize density with contours

        INPUTS:
            rlabels -- list of positive reals -- radii to plot
            N -- int -- number of contours
            Wr -- N -- window for convolving with the real axis of histogram
                    Default: no convolution
            Wi -- M -- window for convolving with the imaginary axis of histogram
                    Default: same window as Wr
        """
        ax = gca()
        if (Wr != None).all():
            Wr = asarray(Wr)
            assert Wr.ndim == 1
            if Wi == None:
                Wi = Wr
        fig = gcf()
        vis = fig.get_visible()
        fig.set_visible(False)
        if (Wr == None).any():
            H = self.H
        else:
            H = asarray([convolve(r,Wr,'same') for r in self.H])
            H = asarray([convolve(c,Wi,'same') for c in H.T]).T

        z = log( 1 + H )
        h = [ax.contour( z, N, extent=[self.rB[0],self.rB[-1],self.iB[0],self.iB[-1]], **kw)]

        self._decorate(rlabels, dict(color=[0,0,0],linestyle='-',linewidth=0.3) )
        fig.set_visible(vis)
        return h[0]

    def _decorate( self, rlabels, lt,  ):
        ax = gca()
        rlabels = asarray(rlabels).flatten()
        t = linspace(-3.1415,3.1415,self.N)
        t[-1]=t[0]
        d = exp(1j*pi/10)
        self.h_lbl = []
        self.h_circ = []
        for r in rlabels:
            x, y = self.fishEye( exp(1j * t)*r )
            self.h_circ.append(ax.plot( x, y, **lt )[0])
            x, y = self.fishEye( r*d )
        ax.plot(linspace(-2,2, 100), zeros(100), '-k', linewidth=0.3)
        # Prepare labels and label positions
        l = concatenate([-rlabels[::-1],[0],rlabels])
        v0 = self.fishEye( rlabels )[0]
        v = concatenate([-v0[::-1],[0],v0])
        ax.set(xticks=v,xticklabels=l,yticks=v,yticklabels=l)


def genTraj(A, N, dt, lams, Ntraj=50, L=50, repeat=200, mag=0.05, magsys=None, sums=[]):
    print(f"genTraj L={L}")
    D = len(lams)
    if magsys == None:
        magsys=mag
    Traj = trajectory(A, N, dt, lams, \
                repeat=repeat, sys_noise=magsys, obs_noise=mag, sums=sums)
    x0 = uniform(-1,1,(Ntraj,D))
    # print("x0 shape", x0.shape)
    for x0i,j in zip(x0,range(Ntraj)):
        x0i/=norm(x0i)
        Traj.add_trajectory(L, x0i)
    return Traj, Traj.DEv_true


def doDMD(Traj, Ntraj, L, method='exact'):
    DEv_l  = []
    b_l = []
    DModes_l = []
    for i in range(Traj.repeat):
        print(f"Traj.Y={len(Traj.Y)} Traj.Y[0]={Traj.Y[0].shape}")
        Y0 = [yi[:L-1,:,i].T for yi in Traj.Y[:Ntraj]]
        Y1 = [yi[1:L,:,i].T for yi in Traj.Y[:Ntraj]]
        print(f"Y0={Y0[0].shape}")
        Y0 = concatenate(Y0, axis=-1)
        Y1 = concatenate(Y1, axis=-1)
        print(f"Y0={Y0.shape}")
        DModes, DEv = dmd(Y0, Y1, -len(Traj.sums), method=method)
        print(DModes, DEv)
        b = pinv(DModes).dot(Traj.Y[0][0,:,i])
        inUse, DEv, b  = eigValSort(DEv, b, Traj.DEv_true)
        print(inUse, DEv, b)
        if inUse:
            DEv_l.append(DEv)
            b_l.append(b)
            DModes_l.append(DModes)
    return asarray(DEv_l), asarray(b_l), asarray(DModes_l)


def dooptDMD(Traj, Ntraj, L, r):
    DEv_el  = []
    b_el = []
    for i in range(Traj.repeat):
        Y = [yi[:L,:,i].T for yi in Traj.Y[:Ntraj]]
        t = [ti[:L] for ti in Traj.ts[:Ntraj]]
        Y = concatenate(Y, axis=-1)
        t = concatenate(t, axis=-1)[None,:]
        try:
            _, DEv_e, b_e = optdmd(Y,t,r,0)
            inUse, DEv_e, b_e  = eigValSort(DEv_e, b_e, Traj.DEv_true)
            if inUse:
                DEv_el.append(DEv_e)
                b_el.append(b_e)
        except:
            "opt dmd fail to converge"
            pass
    return asarray(DEv_el),asarray(b_el)


def eigValSort(v, b, v_true, thresh=1e-5):
    realNum = count_nonzero(v_true.imag==0)
    imagNum = int((len(v_true) - realNum)/2)
    if count_nonzero(abs(v.imag)<=thresh)>realNum:
        inUse = False
    elif count_nonzero(abs(v)<=0.01)>1:
        inUse = False
    elif abs(sum(v).imag)>0.01:
        inUse = False
    else:
        inUse = True
        arg = argsort(v.imag)
        v = v[arg]
        b = b[arg]
        vv = v[where(abs(v.imag)<=thresh)]
        bb = b[where(abs(v.imag)<=thresh)]
        v[where(abs(v.imag)<=thresh)] = vv[argsort(v[where(abs(v.imag)<=thresh)])]
        b[where(abs(v.imag)<=thresh)] = bb[argsort(v[where(abs(v.imag)<=thresh)])]
        for i in range(realNum - count_nonzero(abs(v.imag)<=thresh)):
            v[imagNum+i-1], v[imagNum+realNum-i] = (v[imagNum+i-1] + v[imagNum+realNum-i])/2.,  (v[imagNum+i-1] + v[imagNum+realNum-i])/2.
            b[imagNum+i-1], b[imagNum+realNum-i] = (b[imagNum+i-1] + b[imagNum+realNum-i])/2.,  (b[imagNum+i-1] + b[imagNum+realNum-i])/2.
        for i in range(imagNum-1):
            if norm(v[i] - v_true[i])**2 + norm(v[i+1] - v_true[i+1])**2 > norm(v[i] - v_true[i+1])**2 + norm(v[i+1] - v_true[i])**2:
                v[i], v[i+1], v[-i-1], v[-i-2] = v[i+1], v[i], v[-i-2], v[-i-1]
                b[i], b[i+1], b[-i-1], b[-i-2] = b[i+1], b[i], b[-i-2], b[-i-1]
    if count_nonzero(abs(v.real)<=thresh)>imagNum:
        inUse = False
    return inUse, v, b


def visSortedEig(vl, bl, lst, cmaps, rlabels=[0.5, 1.0]):
    assert shape(vl)[1] == len(cmaps)
    for i in lst:
        vi = VisEig(N=512,fishEye=False, rng=[[-1.5,1.5],[-1.5,1.5]])
        arg = argsort(vl[:,i].real)
        H1 = ones(shape(vi.H))
        seterr(invalid='ignore', divide = 'ignore')
        cnt = 0
        for v, b, cnt in zip(vl[arg,i], bl[arg,i], range(len(vl[arg,i]))):
            H1 = vi.H
            vi.add(asarray(v), wgt=abs(b))
            if cnt>len(vl[arg,i])*0.9 and nansum(log(H1/sum(H1) * sum(vi.H)/vi.H) * H1/sum(H1)) < 1e-3 : #KL-divergence
                h = vi.vis1(Wr=windows.gaussian(50, 5), N=6,  cmap=cmaps[i], alpha=0.6, rlabels = rlabels)
                h.collections[0].remove()
                break
        else:
            print("Warning: eigenvalue density distribution did not converge, try again.")
        vi.clear()
