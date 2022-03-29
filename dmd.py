"""
file: dmd.py

An implementation of Schmidt DMD, based on matlab code provided courtesy
of UCSB

(c) 2018 Shai Revzen

An extension to forward and backward (fb) DMD and total least square (tls) DMD, based on book
J. Nathan Kutz, Steven L. Brunton, Bingni W. Brunton, and Joshua L. Proctor. 2016.
Dynamic Mode Decomposition: Data-Driven Modeling of Complex Systems.
SIAM-Society for Industrial and Applied Mathematics, Philadelphia, PA, USA.
(c) 2020,2021 Ziyou Wu

Work on this code was funded by ARO MURI W911NF-17-1-0306 "From Data-Driven
Operator Theoretic Schemes to Prediction, Inference, and Control of Systems",
and government usage rights are reserved as per that funding agreement.

All other usage is governed by the GPL 3.0 license as specified below.

dmd.py is free software: you can redistribute it and/or modify
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
    dot, sum, asarray, sqrt, newaxis, argsort, append, 
    shape, concatenate, allclose
    )
from numpy.linalg import eig, svd, pinv
'''
from numpy import dot, allclose, sum, asarray, sqrt, newaxis, argsort, append, shape, diag
from numpy.linalg import eig, svd, pinv
'''
from scipy.linalg import sqrtm


def MUL( *lst ):
  """
  Matrix multiply multiple matrices
  If matrices are not aligned, gives a more informative error
  message than dot(), with shapes of all matrices

  >>> a = ones((2,3)); b = ones((3,4))
  >>> MUL(a,b,b.T)
     array([[ 12.,  12.,  12.],
       [ 12.,  12.,  12.]])
  >>> MUL(a,b,b)
  ... ValueError: matrices are not aligned for [(2, 3), (3, 4), (3, 4)]
  """
  seq = list(lst)
  try:
    res = seq.pop(0)
    while seq:
        res=dot(res,seq.pop(0))
    return res
  except ValueError as msg:
    sz = [ asarray(x).shape for x in lst ]
    raise ValueError("%s for %s" % (msg,sz))


def Aug(X,s,step=1):
    """
    Construct a time delay matrix, with s delay coordinates augmented
    [x1, .., xk;
     x2, .. x(k+1);
      ...;
     xs, ... x(k+s-1)]
    INPUT:
      X -- data matrix
      s -- number of delay states
      step -- time interval
    OUTPUT:
      Augmented X matrix
    """
    L = concatenate( [X[k:-s+k,...,newaxis] for k in range(0,s,step)], axis=1)
    L.shape = L.shape[0],L.shape[1]*L.shape[2]
    return L


def compute_svd(X, relTol):
    """
    Compute SVD of data matrix X
    """
    U0,S0,V0 = svd(X,full_matrices=False)
    if relTol>0:
        r = (S0>S0[0]*relTol).nonzero()[0][-1]+1
    else:
        r = int(-relTol)
    #print('subspace dimension:', r)
    U = U0[:,:r]
    S = S0[:r]
    V = V0[:r,:]
    assert allclose(dot(U,S[:,newaxis]*V),X), "SVD error not within tolerance."
    return U, S, V


def dmd(X,Y,relTol=1e-12,withRes=False,method='exact'):
    """
    Compute a Schmidt DMD mapping states X (in rows) to states Y
    INPUT:
      X,Y -- NxD -- data matrices
      method -- 'exact', 'fb', 'tls'
      relTol -- real>0 -- tolerance for accepting modes as significant
      withRes -- bool -- add additional results variables
    OUTPUT:
      DModes, DEv, [relPower,relativeError,Q]

      DModes -- NxM -- modes representing the data
      DEv -- eigenvalues of modes
      relPower -- M -- relative power in each mode
      relativeError -- residual norm of DMD representation
      Q -- NxD -- reconstruction of the data using the modes
    """
    U,S,V = compute_svd(X, relTol)
    sq = sqrt(S)

    if method == 'exact':
        #Exact DMD
        Ahat = MUL(U.T,Y,V.T)/sq[:,newaxis]/sq[newaxis,:]
        DEv,what = eig(Ahat)

    elif method == 'fb':
        #Forward and backward DMD
        U1,S1,V1 = compute_svd(Y, relTol)
        sq1 = sqrt(S1)
        fAhat = MUL(U.T,Y,V.T)/sq[:,newaxis]/sq[newaxis,:]
        bAhat = MUL(U1.T,X,V1.T)/sq1[:,newaxis]/sq1[newaxis,:]
        Ahat2 = pinv(bAhat).dot(fAhat)
        DEv,what = eig(Ahat2)
        DEv = sqrt(DEv)
        Ahat = sqrtm(Ahat2)

    elif method == 'tls':
        #Total least square DMD
        Z = append(X,Y,axis=0)
        U,_,_ = svd(Z,full_matrices=False)
        n = shape(X)[0]
        Ahat = U[n:, :n].dot(pinv(U[:n, :n]))
        DEv,what = eig(Ahat)
    else:
        raise ValueError("DMD method not defined")

    w = sq[:,newaxis]*what
    DModes = MUL(Y,V.T,w/S[:,newaxis])

    normsSquared = sum(DModes*DModes.conj(),0)
    Index = argsort(normsSquared)[::-1] #[Power,Index]=sort(normsSquared,'descend');
    Power = normsSquared[Index]
    DEv = DEv[Index]
    DModes = DModes[:,Index] / Power[newaxis,:] #DModes = DModes./repmat(sqrt(Power'),size(DModes,1),1);
    if not withRes:
        return DModes, DEv
    relPower = (Power/sum(Power)).real
    if method == 'exact':
        Q = MUL(DModes*DEv[newaxis,:], pinv(DModes), X)
    else:
        Q = MUL(Ahat, X)
    relativeError = sum((Q-Y)*(Q-Y).conj())/sum(Y*Y.conj())
    return DModes,DEv,relPower,relativeError,Q


if __name__=="__main__":
    from numpy.random import randn, seed
    from numpy import abs
    from matplotlib.pyplot import figure, semilogy, title, show, subplots
    # Reduced order dynamic
    seed(100)
    while True:
        #Generate a random matrix with eigenvalue specturm in right half plane
        A0 = randn(2,2)
        e, v = eig(A0)
        if (e.real>0).all():
            break

    while True:
        scl = abs(eig(A0)[0][0])
        # Lifting matrix
        L = svd(randn(2,2))[0][:len(A0)]
        # Lifted dynamic
        A = dot(dot(L.T,A0*0.98/scl),L).real
        if abs(eig(A)[0][0])<1:
            break

    ap = [A]
    for k in range(8):
        ap.append(dot(ap[-1],ap[-1]))

    X = [randn(A.shape[1])]
    for k in range(1, 2**len(ap)):
        Ak = 1
        for kk,apk in zip(bin(k)[::-1][:-2],ap):
            if kk == '1':
                Ak = dot(Ak,apk)
        X.append(dot(Ak,X[0]))
    X = asarray(X)
    # Noisy result
    #X += randn(*X.shape) * 1e-5
    Y = X[1:,...]
    X = X[:-1,...]
    relTol = -shape(X)[1]

    fig, ax = subplots()
    for method, marker in zip(['exact', 'fb', 'tls'], ['o', '*', '^']):
        DModes,DEv,relPower,relativeError,Q = dmd(X.T,Y.T,relTol,withRes=True, method=method)
        try:
            figure()
            semilogy(abs(Y),'r')
            semilogy(abs(Y-Q.T),'b')
            title(method)
            show()
            ax.plot(DEv.real, DEv.imag, marker, label=method)
        except NameError:
            print("Cannot produce plots for method ",method)
        print("relative error is ",relativeError,' for method: ', method)
    et, _ = eig(A)
    ax.plot(et.real, et.imag, 'xr', label='ground truth')
    ax.legend()
    ax.set_xlim(0., 1.0)
    ax.set_ylim(-1.0, 1.0)
