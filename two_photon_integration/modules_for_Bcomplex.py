#!/usr/bin/env python
# coding: utf-8

# In[1]:


# get_ipython().run_line_magic('pylab', 'inline')
import numpy as np  #
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
import cmath
from numpy.typing import NDArray


# In[2]:


def getgaussiangrid(t, nknot, kord, mult):

    dt = np.complex128
    ig = kord + 1
    print("this is the gaussian grid, with", ig, " points between each  knot point")
    outr = np.zeros((nknot - 2 * kord + 1 - mult) * ig, dtype=dt)
    outw = np.zeros((nknot - 2 * kord + 1 - mult) * ig, dtype=dt)
    absc, weights = np.polynomial.legendre.leggauss(ig)
    i = kord - 1
    k = 0
    while i < nknot - kord:
        j = 0
        if abs(t[i + 1] - t[i]) > 0.00000000001:
            while j < ig:
                outr[k] = (t[i] + t[i + 1]) / 2.0 + absc[j] * (t[i + 1] - t[i]) / 2.0
                outw[k] = weights[j] * (t[i + 1] - t[i]) / 2.0
                k = k + 1
                j = j + 1
        i = i + 1
    return outr, outw, ig


# In[3]:


def lineargrid(kord, Rot, rmax, fi, h, plus1, mult):
    dt = np.complex128
    theta = fi / 180.0 * np.pi

    nknot = 2 * (kord - 1) + int((rmax + 0.0000000001) / h) + 1 + 3 + plus1 + mult
    print(
        " number of knotpoints=",
        nknot,
        " rotation from",
        Rot,
        " theta=",
        theta,
        " radians",
    )
    t = np.zeros(nknot, dtype=dt)
    i = 0
    while i < kord:
        t[i] = complex(0.0, 0.0)
        t[nknot - i - 1 - mult] = complex((1.0 + plus1) * rmax, 0.0)
        i = i + 1
    i = kord
    hsmall = h / 4.0
    while i < kord + 4:
        t[i] = t[i - 1] + hsmall
        i = i + 1
    i = kord + 4
    while i < nknot - kord - mult:
        t[i] = t[i - 1] + h
        i = i + 1
    i = 0
    ij = complex(0.0, 1.0)
    iclose = 10**10
    while i < nknot:
        if np.real(t[i]) > Rot:
            t[i] = Rot + (t[i] - Rot) * np.exp(ij * theta)
            if i < iclose:
                iclose = i
        #        print(i,t[i])
        i = i + 1
    print(
        "rotation from knot nr", iclose - 1, "where there are ", mult, " extra points"
    )
    i = nknot - 1
    while i > iclose - 1:
        t[i] = t[i - mult]
        i = i - 1
    i = iclose
    while i < iclose + mult:
        t[i] = t[iclose - 1]
        i = i + 1
    return nknot, t, iclose - 1


# In[4]:


def getgaussiangridold(t, nknot, kord):

    dt = np.complex128
    ig = kord + 1
    print("this is the gaussian grid, with", ig, " points between each  knot point")
    outr = np.zeros((nknot - 2 * kord + 1) * ig, dtype=dt)
    outw = np.zeros((nknot - 2 * kord + 1) * ig, dtype=dt)
    absc, weights = np.polynomial.legendre.leggauss(ig)
    i = kord - 1
    k = 0
    while i < nknot - kord:
        j = 0
        if abs(t[i + 1] - t[i]) > 0.00000000001:

            while j < ig:
                outr[k] = (t[i] + t[i + 1]) / 2.0 + absc[j] * (t[i + 1] - t[i]) / 2.0
                outw[k] = weights[j] * (t[i + 1] - t[i]) / 2.0
                k = k + 1
                j = j + 1
        i = i + 1
    return outr, outw, ig


# In[5]:


def bsplvb_cmplx(t, kord, x, left):
    dt = np.complex128
    sp = np.zeros(kord, dtype=dt)
    deltar = np.zeros(kord, dtype=dt)
    deltal = np.zeros(kord, dtype=dt)

    j = 0
    sp[j] = complex(1.0, 0.0)
    while j < kord - 1:
        jp1 = j + 1
        deltar[j] = t[left + j + 1] - x
        deltal[j] = x - t[left + 1 - j - 1]
        #       print(j,t[left+1-j-1],t[left+j+1],deltal[j],deltar[j])
        saved = complex(0.0, 0.0)
        i = 0
        while i <= j:
            #            print(' check',deltar[i], deltal[jp1-i-1] )
            term = sp[i] / (deltar[i] + deltal[jp1 - i - 1])
            sp[i] = saved + deltar[i] * term
            saved = deltal[jp1 - i - 1] * term
            #            print(i,j,jp1,jp1-i-1,'term',term,saved,sp[i])
            i = i + 1

        sp[jp1] = saved
        #        print('sp in',jp1,sp[jp1])
        j = jp1

    return sp


# In[6]:


def splinder(t, kord, rr, left):
    sp = bsplvb_cmplx(t, kord - 1, rr, left)
    dt = np.complex128
    out = np.zeros(kord, dtype=dt)
    i = 0
    while i < kord:
        #       index for the splines of order k
        indexi = left - kord + i + 1
        if i == 0:
            out[i] = -sp[i] / (t[indexi + kord] - t[indexi + 1]) * (kord - 1.0)
        elif i == kord - 1:
            out[i] = sp[i - 1] / (t[indexi + kord - 1] - t[indexi]) * (kord - 1.0)
        else:
            out[i] = (
                sp[i - 1] / (t[indexi + kord - 1] - t[indexi])
                - sp[i] / (t[indexi + kord] - t[indexi + 1])
            ) * (kord - 1.0)
        i = i + 1
    return out


# In[7]:


def getfunc(nknot, kord, t, vec, ilast, rr):
    # ilast =0 skip the last B-spline. Assumed that the first vec[0] is for the second spline
    out = complex(0.0, 0.0)

    if abs(rr - t[t.size - 1]) < 10 ** (-10):
        if ilast == 1:

            out = complex(1.0, 0.0)
        return out
    if np.real(rr) > np.real(t[t.size - 1]):
        print("warning rr=", rr, " while tlast is", t[t.size - 1])
        return out
    i = 0
    left = -1
    left = getleft(rr, t)
    #    while i< nknot-kord:
    #        if(np.real(rr) >= np.real(t[i])):
    #            left=i
    #        i=i+1
    if left < 0:
        print("warning left=", -1)
    sp = bsplvb_cmplx(t, kord, rr, left)

    i = 0
    while i < kord:
        #       index for the splines of order k
        indexi = left - kord + i + 1
        if indexi > 0:
            if indexi < nknot - kord + ilast - 1:

                out = out + sp[i] * vec[indexi - 1]

        i = i + 1
    return out


# In[8]:


def getderivative(nknot, kord, t, vec, ilast, rr):
    # ilast =0 skip the last B-spline
    out = complex(0.0, 0.0)
    if abs(rr - t[t.size - 1]) < 10 ** (-10):
        if ilast == 1:
            print("warning the derivativ in the very last point not fixed")

        return out
    if np.real(rr) > np.real(t[t.size - 1]):

        return out
    i = 0
    left = -1
    left = getleft(rr, t)
    #    while i< nknot-kord:
    #        if(np.real(rr) >= np.real(t[i])):
    #            left=i
    #        i=i+1
    if left < 0:
        print("warning left=", -1)
    sp = bsplvb_cmplx(t, kord - 1, rr, left)

    i = 0
    while i < kord:
        #       index for the splines of order k
        indexi = left - kord + i + 1
        if i == 0:
            deri = -sp[i] / (t[indexi + kord] - t[indexi + 1]) * (kord - 1.0)
        elif i == kord - 1:
            deri = sp[i - 1] / (t[indexi + kord - 1] - t[indexi]) * (kord - 1.0)
        else:
            deri = (
                sp[i - 1] / (t[indexi + kord - 1] - t[indexi])
                - sp[i] / (t[indexi + kord] - t[indexi + 1])
            ) * (kord - 1.0)

        if indexi > 0:
            if indexi < nknot - kord + ilast - 1:
                out = out + deri * vec[indexi - 1]
        i = i + 1
    return out


# In[9]:


def getfunc_in_r(nknot, kord, t, vec, rr):
    dt = np.complex128
    f = np.zeros(rr.size, dtype=dt)
    i = 0
    while i < rr.size:
        f[i] = getfunc(nknot, kord, t, vec, 0, rr[i])
        i = i + 1
    return f


# In[1]:


def getder_in_r(nknot, kord, t, vec, rr):
    dt = np.complex128
    fder = np.zeros(rr.size, dtype=dt)
    i = 0
    while i < rr.size:
        fder[i] = getderivative(nknot, kord, t, vec, 0, rr[i])
        i = i + 1
    return fder


# In[10]:


def getrate_in_r(nknot, kord, t, vec, rr):
    dt = np.complex128
    f = np.zeros(rr.size, dtype=dt)
    fder = np.zeros(rr.size, dtype=dt)
    i = 0
    while i < rr.size:
        f[i] = getfunc(nknot, kord, t, vec, 0, rr[i])
        fder[i] = getderivative(nknot, kord, t, vec, 0, rr[i])
        i = i + 1
    rate = -np.real(
        (complex(0.0, 1.0) * (np.conjugate(f) * fder - f * np.conjugate(fder)) / 2.0)
    )
    return rate


# In[11]:


def setuph_nrel(rr, weights, t, nknot, kord, Z, ll):
    b, bz, bl, bder, bdip = setupparts_nrel(rr, weights, t, nknot, kord)
    ndim = nknot - kord - 2
    dt = np.complex128
    h = np.zeros((ndim, ndim), dtype=dt)
    h = bder / 2.0 - Z * bz + ll * (ll + 1) * bl / 2.0
    j = 0

    alpha, beta, vecl, vecr, work, info = scipy.linalg.lapack.zggev(h, b)
    i = 0
    print(" result for zggev", info)
    #    while i <ndim:
    #        print(i,alpha[i],beta[i])
    #        i=i+1
    EE = alpha / beta
    #   the  vectors are not normalized: fix this
    i = 0
    while i < ndim:
        v = np.matmul(b, vecr[:, i])
        a = np.dot(v, vecr[:, i])
        aa = 1.0
        if vecr[0 + ll, i] < 0:
            aa = -aa
        vecr[:, i] = aa * vecr[:, i] / np.sqrt(a)
        i = i + 1
    return EE, vecr, b, bdip


# In[12]:


def perturbedwf(fin, ein, vec, eig, omega, bdip):
    dt = np.complex128
    out = np.zeros(fin.size, dtype=dt)
    v = np.matmul(bdip, fin)
    i = 0
    while i < eig.size:
        matelem = np.dot(vec[:, i], v) / (ein + omega - eig[i])
        out = out + vec[:, i] * matelem
        i = i + 1
    return out


# In[13]:


def getleft(x, t):
    i = t.size - 1
    while np.real(t[i]) > np.real(x):
        i = i - 1
    return i


# In[14]:


def setupparts_nrel(rr, weights, t, nknot, kord):
    ig = kord + 1
    ndim = nknot - kord - 2
    print(" Constructing matricies with dimension", ndim)
    dt = np.complex128
    b = np.zeros((ndim, ndim), dtype=dt)
    bz = np.zeros((ndim, ndim), dtype=dt)
    bl = np.zeros((ndim, ndim), dtype=dt)
    bder = np.zeros((ndim, ndim), dtype=dt)
    bdip = np.zeros((ndim, ndim), dtype=dt)
    i = kord - 1
    k = 0
    while k < rr.size:
        left = getleft(rr[k], t)
        sp = bsplvb_cmplx(t, kord, rr[k], left)
        der = splinder(t, kord, rr[k], left)
        ii = 0
        while ii < kord:
            indexi = left - kord + ii + 1
            jj = 0
            if np.logical_and(indexi > 0, indexi < ndim + 1):
                while jj < kord:
                    indexj = left - kord + jj + 1
                    if np.logical_and(indexj > 0, indexj < ndim + 1):
                        b[indexi - 1, indexj - 1] = (
                            b[indexi - 1, indexj - 1] + sp[ii] * sp[jj] * weights[k]
                        )
                        bz[indexi - 1, indexj - 1] = (
                            bz[indexi - 1, indexj - 1]
                            + sp[ii] * sp[jj] * weights[k] / rr[k]
                        )
                        bl[indexi - 1, indexj - 1] = (
                            bl[indexi - 1, indexj - 1]
                            + sp[ii] * sp[jj] * weights[k] / rr[k] ** 2
                        )
                        bder[indexi - 1, indexj - 1] = (
                            bder[indexi - 1, indexj - 1]
                            + der[ii] * der[jj] * weights[k]
                        )
                        bdip[indexi - 1, indexj - 1] = (
                            bdip[indexi - 1, indexj - 1]
                            + sp[ii] * sp[jj] * weights[k] * rr[k]
                        )
                    jj = jj + 1
            ii = ii + 1
        k = k + 1

    #        i=i+1
    return b, bz, bl, bder, bdip


# In[15]:


def setupparts_nrelold(rr, weights, t, nknot, kord):
    ig = kord + 1
    ndim = nknot - kord - 2
    print(" Constructing matricies with dimension", ndim)
    dt = np.complex128
    b = np.zeros((ndim, ndim), dtype=dt)
    bz = np.zeros((ndim, ndim), dtype=dt)
    bl = np.zeros((ndim, ndim), dtype=dt)
    bder = np.zeros((ndim, ndim), dtype=dt)
    bdip = np.zeros((ndim, ndim), dtype=dt)
    i = kord - 1
    k = 0
    while i < nknot - kord:
        #   In the region between t[i] and t[i+1] kord Bsplines are non-zero
        #        vec=np.zeros((ndim,kord),dtype=dt)
        left = getleft(rr[k], t)
        j = 0
        while j < ig:
            sp = bsplvb_cmplx(t, kord, rr[k], left)
            der = splinder(t, kord, rr[k], left)
            ii = 0
            while ii < kord:
                indexi = left - kord + ii + 1
                jj = 0
                if np.logical_and(indexi > 0, indexi < ndim + 1):

                    while jj < kord:
                        indexj = left - kord + jj + 1
                        if np.logical_and(indexj > 0, indexj < ndim + 1):

                            b[indexi - 1, indexj - 1] = (
                                b[indexi - 1, indexj - 1] + sp[ii] * sp[jj] * weights[k]
                            )
                            bz[indexi - 1, indexj - 1] = (
                                bz[indexi - 1, indexj - 1]
                                + sp[ii] * sp[jj] * weights[k] / rr[k]
                            )
                            bl[indexi - 1, indexj - 1] = (
                                bl[indexi - 1, indexj - 1]
                                + sp[ii] * sp[jj] * weights[k] / rr[k] ** 2
                            )
                            bder[indexi - 1, indexj - 1] = (
                                bder[indexi - 1, indexj - 1]
                                + der[ii] * der[jj] * weights[k]
                            )
                            bdip[indexi - 1, indexj - 1] = (
                                bdip[indexi - 1, indexj - 1]
                                + sp[ii] * sp[jj] * weights[k] * rr[k]
                            )
                        jj = jj + 1
                ii = ii + 1
            k = k + 1
            j = j + 1
        i = i + 1
    return b, bz, bl, bder, bdip


# In[16]:


def printtofile(x, y, ipoints, folder, filename):
    fullname = folder + filename
    #    np.savetxt(fullname, (x,y), fmt='%.18e', delimiter=' ', newline='\n')
    np.savetxt(fullname, np.transpose([x, y]), fmt="%.18e", delimiter=" ", newline="\n")
    return


# In[ ]:


def preduced(lb, la, bdivr, bder, nknot, kord):
    #   -i<lb || p || la>
    #   see Maartensson&Salomonson J. PhysB 15 2115(1982)
    lmax = lb
    if la > lb:
        lmax = la
    ang1 = -(lb - la) * np.sqrt(lmax)
    ang2 = lmax * np.sqrt(lmax)
    dt = np.complex128
    ndim = nknot - kord - 2
    pred = np.zeros((ndim, ndim), dtype=dt)
    if abs(lb - la) == 1:
        pred = ang1 * bder + ang2 * bdivr
    #    pred2=np.zeros((ndim,ndim),dtype=dt)
    #    i=0
    #    while i<nknot-kord-2:
    #        j=0
    #        while j<nknot-kord-2:
    #           pred2[i,j]=pred[j,i]
    #            j=j+1
    #        i=i+1
    return pred


# In[ ]:


def setbder(rr, weights, t, nknot, kord):
    ig = kord + 1
    ndim = nknot - kord - 2
    print(" Constructing matricies with dimension", ndim)
    dt = np.complex128

    bder = np.zeros((ndim, ndim), dtype=dt)
    bdivr = np.zeros((ndim, ndim), dtype=dt)
    i = kord - 1
    k = 0
    while k < rr.size:
        left = getleft(rr[k], t)
        sp = bsplvb_cmplx(t, kord, rr[k], left)
        der = splinder(t, kord, rr[k], left)
        ii = 0
        while ii < kord:
            indexi = left - kord + ii + 1
            jj = 0
            if np.logical_and(indexi > 0, indexi < ndim + 1):
                while jj < kord:
                    indexj = left - kord + jj + 1
                    if np.logical_and(indexj > 0, indexj < ndim + 1):

                        bder[indexi - 1, indexj - 1] = (
                            bder[indexi - 1, indexj - 1] + sp[ii] * der[jj] * weights[k]
                        )
                        bdivr[indexi - 1, indexj - 1] = (
                            bdivr[indexi - 1, indexj - 1]
                            + sp[ii] * sp[jj] * weights[k] / rr[k]
                        )

                    jj = jj + 1
            ii = ii + 1
        k = k + 1

    #        i=i+1
    return bder, bdivr


# In[ ]:


def coulphase(ell, eps, Z):
    #   This is the definition of the phase of the Coulomb function
    #   both the angular momentum part and the so-called Coulomb phase
    #   eps should be entered in atomic units
    x = Z / np.sqrt(2 * eps)
    zz = ell + 1.0 - complex(0.0, 1.0) * x
    #    print " this is the argument of the function=",zz
    #    print " this its phase", cmath.phase(zz)
    a = special.gamma(zz)
    #    b=cmath.phase(a)
    b = np.angle(a)
    #    print " gamma",a," def coulphase(ell,eps,Z)phase",b

    coulphase = -ell * np.pi / 2.0 + b
    return coulphase


# In[ ]:


def honrho(b, vec, eig, rho):
    dt = np.complex128
    out = np.zeros(eig.size, dtype=dt)
    v = np.matmul(b, rho)
    i = 0
    while i < eig.size:
        v2 = eig[i] * np.matmul(vec[:, i], v)
        out = out + v2 * vec[:, i]
        i = i + 1
    return out


# In[ ]:


def fakultet(n):
    prod = 1
    for k in range(2, n + 1):
        prod *= k

    return prod


# In[ ]:


def my3j(j1, j2, j3):
    jsum = j1 + j2 + j3
    even = jsum % 2
    if even != 0:
        print(" the sum has to be even")
        result = 0.0
    if even == 0:
        jhalf = int(jsum / 2)
        f1 = fakultet(j1 + j2 - j3)
        f2 = fakultet(j1 + j3 - j2)
        f3 = fakultet(j2 + j3 - j1)
        f4 = fakultet(j1 + j2 + j3 + 1)
        ftot1 = fakultet(jhalf)
        ftot2 = fakultet(jhalf - j1)
        ftot3 = fakultet(jhalf - j2)
        ftot4 = fakultet(jhalf - j3)
        result = (
            ((-1) ** (jhalf))
            * np.sqrt(f1 * f2 * f3 / f4)
            * ftot1
            / (ftot2 * ftot3 * ftot4)
        )
    return result


# In[ ]:


def redmat(lut, k, lin):
    result = (
        (-1) ** lut * np.sqrt((2.0 * lut + 1.0) * (2.0 * lin + 1.0)) * my3j(lut, k, lin)
    )
    return result


def read_knot_points(file: str) -> NDArray[np.complex128]:
    """
    Reads the knot point sequence from the given file.

    Args:
        file - path to the file.

    Returns:
        knots - array with the knot points.
    """

    knots_raw = np.loadtxt(file)

    knots = np.zeros(knots_raw.shape[0], dtype=np.complex128)

    knots = knots_raw[:, 0] + 1j * knots_raw[:, 1]

    return knots


def get_knots_f_g(
    k_order_f: int, k_order_g: int, knots: NDArray[np.complex128]
) -> tuple[NDArray[np.complex128], NDArray[np.complex128]]:
    """
    Takes a sequence of knot points and extarcts (separates) the knot point for the large and small
    components.

    Args:
        k_order_f - order of the B-splines for the large component.
        k_order_g - order of the B-splines for the small component.
        knots - a sequence of the knot points from Relcode.

    Returns:
        knots_f, knots_g - knot points for the large and small components respectively.
    """

    if k_order_f < k_order_g:
        knots_g = knots.copy()
        k_diff = k_order_g - k_order_f
        knots_f = knots[k_diff:-k_diff].copy()

    elif k_order_f > k_order_g:
        knots_f = knots.copy()
        k_diff = k_order_f - k_order_g
        knots_g = knots[k_diff:-k_diff].copy()

    else:  # k_order_f == k_order_g
        knots_f = knots.copy()
        knots_g = knots.copy()

    return knots_f, knots_g


def read_q_state_coeffs(file: str) -> NDArray[np.complex128]:
    """
    Reads the B-spline coefficients for the continuum state from the given file.

    Args:
        file - path to the file.

    Returns:
        coeffs - array with the continuum state coefficients.
    """

    coeffs_raw = np.loadtxt(file)
    coeffs_raw = coeffs_raw.T
    coeffs = np.zeros(coeffs_raw.shape[0], dtype=np.complex128)
    coeffs = coeffs_raw[:, 0] + 1j * coeffs_raw[:, 1]

    return coeffs
