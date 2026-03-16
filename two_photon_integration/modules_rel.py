#!/usr/bin/env python
# coding: utf-8

# In[5]:


# get_ipython().run_line_magic('pylab', 'inline')
import numpy as np  #
import scipy
from scipy.interpolate import InterpolatedUnivariateSpline
import cmath
from modules_for_Bcomplex import *


# In[6]:


def setuph_rel(
    rr,
    weights,
    tf,
    tg,
    nknotf,
    nknotg,
    kordf,
    kordg,
    Z,
    ll,
    jj,
    ig,
    cpeed,
    ilastf,
    ilastg,
):
    # usual is with ilastf=ilastg=0
    if ll < jj:
        kappa = -(ll + 1)
    else:
        kappa = ll
    blrg, bzlrg, bdiplrg = setupparts_rel_diag(
        rr, weights, tf, nknotf, kordf, ig, ilastf
    )
    bsml, bzsml, bdipsml = setupparts_rel_diag(
        rr, weights, tg, nknotg, kordg, ig, ilastg
    )
    bkappafg, bfderg = setupparts_rel_nondiag(
        rr, weights, tf, nknotf, kordf, tg, nknotg, kordg, ig, ilastf, ilastg
    )
    ndim = nknotf - kordf - 2 + nknotg - kordg - 2 + ilastf + ilastg
    dt = np.complex128
    h = np.zeros((ndim, ndim), dtype=dt)
    b = np.zeros((ndim, ndim), dtype=dt)
    bdip = np.zeros((ndim, ndim), dtype=dt)

    print("the Dirac matrix for kappa=", kappa, " has dimension", ndim)
    hup = -Z * bzlrg
    hdown = -Z * bzsml - 2.0 * cpeed**2 * bsml
    hnondiag = -bfderg * cpeed + bkappafg * kappa * cpeed
    i = 0
    while i < nknotf - kordf - 2 + ilastf:
        j = 0
        while j < nknotf - kordf - 2 + ilastf:
            h[i, j] = h[i, j] + hup[i, j]
            b[i, j] = b[i, j] + blrg[i, j]
            bdip[i, j] = bdip[i, j] + bdiplrg[i, j]
            j = j + 1
        j = 0
        while j < nknotg - kordg - 2 + ilastg:
            h[i, j + nknotf - kordf - 2 + ilastf] = (
                h[i, j + nknotf - kordf - 2 + ilastf] + hnondiag[i, j]
            )
            j = j + 1
        i = i + 1
    i = 0
    while i < nknotg - kordg - 2 + ilastg:
        j = 0
        while j < nknotf - kordf - 2 + ilastf:
            h[i + nknotf - kordf - 2 + ilastf, j] = (
                h[i + nknotf - kordf - 2 + ilastf, j] + hnondiag[j, i]
            )
            j = j + 1
        j = 0
        while j < nknotg - kordg - 2 + ilastg:
            h[i + nknotf - kordf - 2 + ilastf, j + nknotf - kordf - 2 + ilastf] = hdown[
                i, j
            ]
            b[i + nknotf - kordf - 2 + ilastf, j + nknotf - kordf - 2 + ilastf] = bsml[
                i, j
            ]
            bdip[i + nknotf - kordf - 2 + ilastf, j + nknotf - kordf - 2 + ilastf] = (
                bdip[i + nknotf - kordf - 2 + ilastf, j + nknotf - kordf - 2 + ilastf]
                + bdipsml[i, j]
            )
            j = j + 1
        i = i + 1

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


# In[7]:


def setupparts_rel_diagold(rr, weights, t, nknot, kord, ig, ilast):

    ndim = nknot - kord - 2
    print(
        " Relativistic case: constructing matricies with dimension",
        ndim + ilast,
        " ig=",
        ig,
        " ilast=",
        ilast,
    )
    dt = np.complex128
    b = np.zeros((ndim + ilast, ndim + ilast), dtype=dt)
    bz = np.zeros((ndim + ilast, ndim + ilast), dtype=dt)
    bdip = np.zeros((ndim + ilast, ndim + ilast), dtype=dt)
    i = kord - 1
    k = 0
    while i < nknot - kord:
        #   In the region between t[i] and t[i+1] kord Bsplines are non-zero
        #        vec=np.zeros((ndim,kord),dtype=dt)
        #        left=i
        left = getleft(rr[k], t)
        j = 0
        while j < ig:
            sp = bsplvb_cmplx(t, kord, rr[k], left)
            der = splinder(t, kord, rr[k], left)
            ii = 0
            while ii < kord:
                indexi = left - kord + ii + 1
                jj = 0
                if np.logical_and(indexi > 0, indexi < ndim + 1 + ilast):

                    while jj < kord:
                        indexj = left - kord + jj + 1
                        if np.logical_and(indexj > 0, indexj < ndim + 1 + ilast):

                            b[indexi - 1, indexj - 1] = (
                                b[indexi - 1, indexj - 1] + sp[ii] * sp[jj] * weights[k]
                            )
                            bz[indexi - 1, indexj - 1] = (
                                bz[indexi - 1, indexj - 1]
                                + sp[ii] * sp[jj] * weights[k] / rr[k]
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
    return b, bz, bdip


# In[1]:


def setupparts_rel_diag(rr, weights, t, nknot, kord, ig, ilast):

    ndim = nknot - kord - 2
    print(
        " Relativistic case: constructing matricies with dimension",
        ndim + ilast,
        " ig=",
        ig,
        " ilast=",
        ilast,
    )
    dt = np.complex128
    b = np.zeros((ndim + ilast, ndim + ilast), dtype=dt)
    bz = np.zeros((ndim + ilast, ndim + ilast), dtype=dt)
    bdip = np.zeros((ndim + ilast, ndim + ilast), dtype=dt)

    k = 0
    while k < rr.size:
        left = getleft(rr[k], t)
        sp = bsplvb_cmplx(t, kord, rr[k], left)
        der = splinder(t, kord, rr[k], left)
        ii = 0
        while ii < kord:
            indexi = left - kord + ii + 1
            jj = 0
            if np.logical_and(indexi > 0, indexi < ndim + 1 + ilast):

                while jj < kord:
                    indexj = left - kord + jj + 1
                    if np.logical_and(indexj > 0, indexj < ndim + 1 + ilast):

                        b[indexi - 1, indexj - 1] = (
                            b[indexi - 1, indexj - 1] + sp[ii] * sp[jj] * weights[k]
                        )
                        bz[indexi - 1, indexj - 1] = (
                            bz[indexi - 1, indexj - 1]
                            + sp[ii] * sp[jj] * weights[k] / rr[k]
                        )
                        bdip[indexi - 1, indexj - 1] = (
                            bdip[indexi - 1, indexj - 1]
                            + sp[ii] * sp[jj] * weights[k] * rr[k]
                        )
                    jj = jj + 1
            ii = ii + 1
        k = k + 1

    return b, bz, bdip


# In[4]:


def setupparts_rel_nondiagold(
    rr, weights, tf, nknotf, kordf, tg, nknotg, kordg, ig, ilastf, ilastg
):

    ndimf = nknotf - kordf - 2 + ilastf
    ndimg = nknotg - kordg - 2 + ilastg
    print(
        " Relativistic case: constructing non-diagonal matricies with dimension",
        ndimf,
        " x ",
        ndimg,
        " ilastf=",
        ilastf,
        " ilastg=",
        ilastg,
    )
    dt = np.complex128
    bkappafg = np.zeros((ndimf, ndimg), dtype=dt)
    bder = np.zeros((ndimf, ndimg), dtype=dt)

    k = 0

    while i < nknotf - kordf:

        #   In the region between t[i] and t[i+1] kord Bsplines are non-zero
        #        vec=np.zeros((ndim,kord),dtype=dt)
        #        leftf=i
        #        leftg=leftf+kordg-kordf
        leftf = getleft(rr[k], tf)
        leftg = getleft(rr[k], tg)
        j = 0
        while j < ig:
            spf = bsplvb_cmplx(tf, kordf, rr[k], leftf)
            derf = splinder(tf, kordf, rr[k], leftf)
            spg = bsplvb_cmplx(tg, kordg, rr[k], leftg)
            derg = splinder(tg, kordg, rr[k], leftg)
            ii = 0
            while ii < kordf:
                indexi = leftf - kordf + ii + 1
                jj = 0
                if np.logical_and(indexi > 0, indexi < ndimf + 1 + ilastf):

                    while jj < kordg:
                        indexj = leftg - kordg + jj + 1
                        if np.logical_and(indexj > 0, indexj < ndimg + 1 + ilastg):

                            bkappafg[indexi - 1, indexj - 1] = (
                                bkappafg[indexi - 1, indexj - 1]
                                + spf[ii] * spg[jj] * weights[k] / rr[k]
                            )

                            bder[indexi - 1, indexj - 1] = (
                                bder[indexi - 1, indexj - 1]
                                + spf[ii] * derg[jj] * weights[k]
                            )
                        #                           checked that these give the same
                        #                           bder[indexi-1,indexj-1]=bder[indexi-1,indexj-1] -derf[ii]*spg[jj]*weights[k]
                        jj = jj + 1
                ii = ii + 1
            k = k + 1
            j = j + 1
        i = i + 1
    return bkappafg, bder


# In[5]:


def setupparts_rel_nondiag(
    rr, weights, tf, nknotf, kordf, tg, nknotg, kordg, ig, ilastf, ilastg
):

    ndimf = nknotf - kordf - 2 + ilastf
    ndimg = nknotg - kordg - 2 + ilastg
    print(
        " Relativistic case: constructing non-diagonal matricies with dimension",
        ndimf,
        " x ",
        ndimg,
        " ilastf=",
        ilastf,
        " ilastg=",
        ilastg,
    )
    dt = np.complex128
    bkappafg = np.zeros((ndimf, ndimg), dtype=dt)
    bder = np.zeros((ndimf, ndimg), dtype=dt)

    k = 0
    while k < rr.size:

        #   In the region between t[i] and t[i+1] kord Bsplines are non-zero
        #        vec=np.zeros((ndim,kord),dtype=dt)
        #        leftf=i
        #        leftg=leftf+kordg-kordf
        leftf = getleft(rr[k], tf)
        leftg = getleft(rr[k], tg)
        j = 0

        spf = bsplvb_cmplx(tf, kordf, rr[k], leftf)
        derf = splinder(tf, kordf, rr[k], leftf)
        spg = bsplvb_cmplx(tg, kordg, rr[k], leftg)
        derg = splinder(tg, kordg, rr[k], leftg)
        ii = 0
        while ii < kordf:
            indexi = leftf - kordf + ii + 1
            jj = 0
            if np.logical_and(indexi > 0, indexi < ndimf + 1 + ilastf):

                while jj < kordg:
                    indexj = leftg - kordg + jj + 1
                    if np.logical_and(indexj > 0, indexj < ndimg + 1 + ilastg):

                        bkappafg[indexi - 1, indexj - 1] = (
                            bkappafg[indexi - 1, indexj - 1]
                            + spf[ii] * spg[jj] * weights[k] / rr[k]
                        )

                        bder[indexi - 1, indexj - 1] = (
                            bder[indexi - 1, indexj - 1]
                            + spf[ii] * derg[jj] * weights[k]
                        )
                    #                           checked that these give the same
                    #                           bder[indexi-1,indexj-1]=bder[indexi-1,indexj-1] -derf[ii]*spg[jj]*weights[k]
                    jj = jj + 1
            ii = ii + 1
        k = k + 1

    return bkappafg, bder


# In[9]:


def getfunc_in_r_rel(nknotf, kordf, tf, nknotg, kordg, tg, vec, rr, ilastf):
    dt = np.complex128
    f = np.zeros(rr.size, dtype=dt)
    g = np.zeros(rr.size, dtype=dt)
    vecf = vec[0 : nknotf - kordf - 2 + ilastf]
    vecg = vec[nknotf - kordf - 2 + ilastf : vec.size]
    #    print(vecf.size,vecg.size)
    i = 0
    while i < rr.size:
        f[i] = getfunc(nknotf, kordf, tf, vecf, 0, rr[i])
        g[i] = getfunc(nknotg, kordg, tg, vecg, 0, rr[i])
        i = i + 1
    return f, g


# In[10]:


def separate_pos_neg(nknotf, kordf, tf, nknotg, kordg, tg, vec, ee, ilastf, ilastg):
    dt = np.complex128
    epos = np.zeros(nknotf - kordf - 2 + ilastf, dtype=dt)
    eneg = np.zeros(nknotg - kordg - 2 + ilastg, dtype=dt)

    ilast = ilastf + ilastg
    vec_pos = np.zeros(
        (nknotf - kordf - 2 + nknotg - kordg - 2 + ilast, nknotf - kordf - 2 + ilastf),
        dtype=dt,
    )
    vec_neg = np.zeros(
        (nknotf - kordf - 2 + nknotg - kordg - 2 + ilast, nknotg - kordg - 2 + ilastg),
        dtype=dt,
    )
    eesort = np.argsort(ee)
    i = 0
    while i < nknotg - kordg - 2 + ilastg:
        vec_neg[:, i] = vec[:, eesort[i]]
        eneg[i] = ee[eesort[i]]
        i = i + 1
    i = 0
    while i < nknotf - kordf - 2 + ilastf:
        vec_pos[:, i] = vec[:, eesort[i + nknotg - kordg - 2 + ilastg]]
        epos[i] = ee[eesort[i + nknotg - kordg - 2 + ilastg]]
        i = i + 1
    return epos, vec_pos, eneg, vec_neg


# In[11]:


def getrate_rel(nknotf, kordf, tf, nknotg, kordg, tg, vec, rr, cpeed, ilastf):
    f, g = getfunc_in_r_rel(nknotf, kordf, tf, nknotg, kordg, tg, vec, rr, ilastf)
    rate = (
        -np.real(complex(0.0, 1.0) * (np.conjugate(f) * g - f * np.conjugate(g)))
        * cpeed
    )
    return rate


# In[12]:


def getgfromf(
    nknotf, kordf, tf, nknotg, kordg, tg, vec, rr, kappa, cpeed, energy, Z, ilastf
):
    dt = np.complex128
    f = np.zeros(rr.size, dtype=dt)
    g = np.zeros(rr.size, dtype=dt)
    fder = np.zeros(rr.size, dtype=dt)
    gnew = np.zeros(rr.size, dtype=dt)
    vecf = vec[0 : nknotf - kordf - 2 + ilastf]
    vecg = vec[nknotf - kordf - 2 + ilastf : vec.size]
    #    print(vecf.size,vecg.size)
    i = 0
    while i < rr.size:
        f[i] = getfunc(nknotf, kordf, tf, vecf, 0, rr[i])
        g[i] = getfunc(nknotg, kordg, tg, vecg, 0, rr[i])
        fder[i] = getderivative(nknotf, kordf, tf, vecf, 0, rr[i])
        konst = (energy + 2.0 * cpeed**2 + Z / rr[i]) / cpeed
        gnew[i] = gnew[i] + (fder[i] + kappa / rr[i] * f[i]) / konst
        i = i + 1
    return g, gnew


# In[ ]:


def setuph_rel_noovlp(
    rr,
    weights,
    tf,
    tg,
    nknotf,
    nknotg,
    kordf,
    kordg,
    Z,
    ll,
    jj,
    ig,
    cpeed,
    ilastf,
    ilastg,
):
    # usual is with ilastf=ilastg=0
    if ll < jj:
        kappa = -(ll + 1)
    else:
        kappa = ll
    blrgorg, bzlrg, bdiplrgorg = setupparts_rel_diag(
        rr, weights, tf, nknotf, kordf, ig, ilastf
    )
    bsmlorg, bzsml, bdipsmlorg = setupparts_rel_diag(
        rr, weights, tg, nknotg, kordg, ig, ilastg
    )
    bkappafg, bfderg = setupparts_rel_nondiag(
        rr, weights, tf, nknotf, kordf, tg, nknotg, kordg, ig, ilastf, ilastg
    )

    ndim = nknotf - kordf - 2 + nknotg - kordg - 2 + ilastf + ilastg
    ndimf = nknotf - kordf - 2 + ilastf
    ndimg = nknotg - kordg - 2 + ilastg
    ijdiff = 1
    dt = np.complex128
    h = np.zeros((ndim - 2 * ijdiff, ndim - 2 * ijdiff), dtype=dt)
    b = np.zeros((ndim - 2 * ijdiff, ndim - 2 * ijdiff), dtype=dt)
    bdip = np.zeros((ndim - 2 * ijdiff, ndim - 2 * ijdiff), dtype=dt)

    vnew = np.zeros((ndim, ndim - 2 * ijdiff), dtype=dt)

    #################################################
    j = 0
    while np.abs(np.imag(tf[j])) < 10 ** (-12):
        j = j + 1
    irotf = j - 1 - kordf
    print("first large component spline starting after the rotation", irotf)
    j = 0
    while np.abs(np.imag(tg[j])) < 10 ** (-12):
        j = j + 1
    irotg = j - 1 - kordg
    print("first small component spline starting after the rotation", irotg)
    #################################################################################

    print("the Dirac matrix for kappa=", kappa, " has dimension", ndim)
    huporg = -Z * bzlrg
    hdownorg = -Z * bzsml - 2.0 * cpeed**2 * bsmlorg
    hnondiagorg = -bfderg * cpeed + bkappafg * kappa * cpeed
    hup = fixfornoovlp(huporg, ndimf, ndimf, irotf, irotf)
    hdown = fixfornoovlp(hdownorg, ndimg, ndimg, irotg, irotg)
    hnondiag = fixfornoovlp(hnondiagorg, ndimf, ndimg, irotf, irotg)
    blrg = fixfornoovlp(blrgorg, ndimf, ndimf, irotf, irotf)
    bsml = fixfornoovlp(bsmlorg, ndimg, ndimg, irotg, irotg)
    #    bdiplrg=fixfornoovlp(bdiplrgorg,ndimf,ndimf,irotf,irotf)
    #    bdipsml=fixfornoovlp(bdipsmlorg,ndimg,ndimg,irotg,irotg)

    i = 0
    while i < nknotf - kordf - 2 + ilastf - ijdiff:
        j = 0
        while j < nknotf - kordf - 2 + ilastf - ijdiff:
            h[i, j] = h[i, j] + hup[i, j]
            b[i, j] = b[i, j] + blrg[i, j]
            #            bdip[i,j]=bdip[i,j] + bdiplrg[i,j]
            j = j + 1
        j = 0
        while j < nknotg - kordg - 2 + ilastg - ijdiff:
            h[i, j + nknotf - kordf - 2 + ilastf - ijdiff] = (
                h[i, j + nknotf - kordf - 2 + ilastf - ijdiff] + hnondiag[i, j]
            )
            j = j + 1
        i = i + 1
    i = 0
    while i < nknotg - kordg - 2 + ilastg - ijdiff:
        j = 0
        while j < nknotf - kordf - 2 + ilastf - ijdiff:
            h[i + nknotf - kordf - 2 + ilastf - ijdiff, j] = (
                h[i + nknotf - kordf - 2 + ilastf - ijdiff, j] + hnondiag[j, i]
            )
            j = j + 1
        j = 0
        while j < nknotg - kordg - 2 + ilastg - ijdiff:
            h[
                i + nknotf - kordf - 2 + ilastf - ijdiff,
                j + nknotf - kordf - 2 + ilastf - ijdiff,
            ] = hdown[i, j]
            b[
                i + nknotf - kordf - 2 + ilastf - ijdiff,
                j + nknotf - kordf - 2 + ilastf - ijdiff,
            ] = bsml[i, j]
            #            bdip[i+nknotf-kordf-2+ilastf-ijdiff,j+nknotf-kordf-2+ilastf-ijdiff]=\
            #            bdip[i+nknotf-kordf-2+ilastf-ijdiff,j+nknotf-kordf-2+ilastf-ijdiff] + bdipsml[i,j]
            j = j + 1
        i = i + 1

    alpha, beta, vecl, vecr, work, info = scipy.linalg.lapack.zggev(h, b)

    j = 0
    while j < ndim - 2 * ijdiff:
        i = 0
        while i < irotf:
            vnew[i, j] = vecr[i, j]
            i = i + 1
        vnew[irotf, j] = vnew[irotf - 1, j]
        i = irotf + 1
        while i < ndimf + irotg:
            vnew[i, j] = vecr[i - 1, j]
            i = i + 1
        vnew[irotg + ndimf, j] = vnew[ndimf + irotg - 1, j]
        i = ndimf + irotg + 1
        while i < ndim:
            vnew[i, j] = vecr[i - 2, j]
            i = i + 1
        j = j + 1

    b = originalb(blrgorg, bsmlorg, ndimf, ndimg)
    bdip = originalb(bdiplrgorg, bdipsmlorg, ndimf, ndimg)

    print(" result for zggev", info)
    #    while i <ndim:
    #        print(i,alpha[i],beta[i])
    #        i=i+1
    EE = alpha / beta
    #   the  vectors are not normalized: fix this
    i = 0
    while i < ndim - 2 * ijdiff:
        v = np.matmul(b, vnew[:, i])
        a = np.dot(v, vnew[:, i])
        aa = 1.0
        if vnew[0 + ll, i] < 0:
            aa = -aa
        vnew[:, i] = aa * vnew[:, i] / np.sqrt(a)
        i = i + 1
    test = vnew.T[0]
    print("returnes", EE.size, "states", " expressed in", test.size, " Bsplines")
    test1 = b[0]
    test = b.T[0]
    print("returned B is", test1.size, "x", test.size)
    test1 = bdip[0]
    test = bdip.T[0]
    print("returned Bdip is", test1.size, "x", test.size)
    return EE, vnew, b, bdip


# In[ ]:


def originalb(blrg, bsml, ndimf, ndimg):
    dt = np.complex128
    b = np.zeros((ndimf + ndimg, ndimf + ndimg), dtype=dt)
    i = 0
    while i < ndimf:
        j = 0
        while j < ndimf:
            b[i, j] = blrg[i, j]
            j = j + 1
        i = i + 1
    i = 0
    while i < ndimg:
        j = 0
        while j < ndimg:
            b[i + ndimf, j + ndimf] = bsml[i, j]
            j = j + 1
        i = i + 1
    return b


# In[ ]:


def fixfornoovlp(h, ndimi, ndimj, irot, jrot):
    dt = np.complex128
    hnew = np.zeros((ndimi - 1, ndimj - 1), dtype=dt)
    j = 0
    while j < jrot:
        i = 0
        while i < irot:
            hnew[i, j] = h[i, j]

            i = i + 1
        j = j + 1
    j = jrot - 1
    while j < ndimj - 1:
        i = irot - 1
        while i < ndimi - 1:
            hnew[i, j] = hnew[i, j] + h[i + 1, j + 1]

            i = i + 1
        j = j + 1
    return hnew


# In[ ]:


def setuph_nrel_nooverlap(rr, weights, t, nknot, kord, Z, ll):
    b, bz, bl, bder, bdip = setupparts_nrel(rr, weights, t, nknot, kord)
    ndim = nknot - kord - 2
    dt = np.complex128
    h = np.zeros((ndim, ndim), dtype=dt)
    h = bder / 2.0 - Z * bz + ll * (ll + 1) * bl / 2.0
    hnew = np.zeros((ndim - 1, ndim - 1), dtype=dt)
    bnew = np.zeros((ndim - 1, ndim - 1), dtype=dt)
    #   here we have ndim-1 solutions but ndim B-splines sist the
    #   two Bsplines going to zero at the rotation point has to be the same
    vnew = np.zeros((ndim, ndim - 1), dtype=dt)
    j = 0
    while np.abs(np.imag(t[j])) < 10 ** (-12):
        j = j + 1
    irot = j - 1 - kord
    print("first spline starting after the rotation", irot)
    j = 0
    while j < irot:
        i = 0
        while i < irot:
            hnew[i, j] = h[i, j]
            bnew[i, j] = b[i, j]
            i = i + 1
        j = j + 1
    j = irot - 1
    while j < ndim - 1:
        i = irot - 1
        while i < ndim - 1:
            hnew[i, j] = hnew[i, j] + h[i + 1, j + 1]
            bnew[i, j] = bnew[i, j] + b[i + 1, j + 1]
            i = i + 1
        j = j + 1

    alpha, beta, vecl, vecr, work, info = scipy.linalg.lapack.zggev(hnew, bnew)

    j = 0
    while j < ndim - 1:
        i = 0
        while i < irot:
            vnew[i, j] = vecr[i, j]
            i = i + 1
        vnew[irot, j] = vnew[irot - 1, j]
        i = irot + 1
        while i < ndim:
            vnew[i, j] = vecr[i - 1, j]
            i = i + 1
        j = j + 1

    print(" result for zggev", info)
    #    while i <ndim:
    #        print(i,alpha[i],beta[i])
    #        i=i+1
    EE = alpha / beta
    #   the  vectors are not normalized: fix this
    i = 0
    while i < ndim - 1:
        v = np.matmul(b, vnew[:, i])
        a = np.dot(v, vnew[:, i])
        aa = 1.0
        if vnew[0 + ll, i] < 0:
            aa = -aa
        vnew[:, i] = aa * vnew[:, i] / np.sqrt(a)
        i = i + 1
    test = vnew.T[0]
    print("returnes", EE.size, "states", " expressed in", test.size, " Bsplines")
    test1 = b[0]
    test = b.T[0]
    print("returned B is", test1.size, "x", test.size)
    test1 = bdip[0]
    test = bdip.T[0]
    print("returned Bdip is", test1.size, "x", test.size)
    return EE, vnew, b, bdip
