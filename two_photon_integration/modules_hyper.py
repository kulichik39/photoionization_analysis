#!/usr/bin/env python
# coding: utf-8

# In[3]:


# get_ipython().run_line_magic('pylab', 'inline')
import numpy as np  #
from mpmath import hyperu
from scipy import special
import cmath
from modules_for_Bcomplex import *


# In[4]:


def Hplmn(sign, l, eps, r):
    ij = complex(0.0, 1.0)
    k = np.sqrt(2.0 * eps)
    eta = 1.0 / k
    a = special.gamma(l + 1.0 - ij * eta)
    sigma = np.angle(a)
    #   sign=+1 gives H+, sign=-1 gives H-
    H = (
        -sign
        * 2.0
        * ij
        * (-2.0) ** l
        * np.exp(-np.pi * eta / 2.0)
        * np.exp(sign * ij * sigma)
        * (k * r) ** (l + 1.0)
        * np.exp(sign * ij * k * r)
        * hyperu(l + 1.0 - sign * eta * ij, 2 * l + 2.0, -sign * 2.0 * ij * k * r)
    )
    Hut = np.array(H, dtype=np.complex128)
    return Hut


# In[5]:


def fixH(sign, l, eps, r):
    i = 0
    dt = np.complex128
    hut = np.zeros(r.size, dtype=dt)
    while i < r.size:
        hut[i] = Hplmn(sign, l, eps, r[i])
        i = i + 1
    return hut


# In[1]:


def contwB(q_state, rr, weights, t, nknot, kord, l, eps, tcut, pw, ig):
    # ig=kord+1
    ndim = nknot - kord - 2
    dt = np.complex128
    bdip = np.zeros(ndim, dtype=dt)
    ij = complex(0.0, 1.0)
    i = kord - 1
    k = 0
    indx = 0
    norm = np.sqrt(2.0 / np.pi / np.sqrt(2.0 * eps))
    while np.real(rr[k]) < np.real(tcut):
        # reg = (
        #     (Hplmn(1.0, l, eps, rr[k]) - Hplmn(-1.0, l, eps, rr[k])) / (2.0 * ij) * norm
        # )  # TODO: replace this with numerical continuum from the program
        reg = q_state[k] * norm
        left = getleft(rr[k], t)
        sp = bsplvb_cmplx(t, kord, rr[k], left)
        ii = 0
        while ii < kord:
            indexi = left - kord + ii + 1
            if indexi > indx:
                indx = indexi
            if np.logical_and(indexi > 0, indexi < ndim + 1):
                bdip[indexi - 1] = (
                    bdip[indexi - 1] + sp[ii] * weights[k] * (rr[k] ** pw) * reg
                )

            ii = ii + 1
        k = k + 1

    #        i=i+1
    #    print('for tlast=',tcut,'last spline=',indx)
    return bdip


# In[2]:


def numericalint(rho, q_state, rr, weights, t, nknot, kord, l, eps, tcut, pw, ig):

    # TODO: change this function to accept numerical continuum
    bdip = contwB(q_state, rr, weights, t, nknot, kord, l, eps, tcut, pw, ig)

    integral = complex(0.0, 0.0)
    integral = np.dot(rho, bdip)

    return integral


# In[3]:


def getamp_and_phase(nknot, kord, t, pwf, eps, l, rtest):
    rho = getfunc(nknot, kord, t, pwf, 0, rtest)
    ana = -Hplmn(1.0, l, eps, rtest)
    # this sign is to fix how we are doing in the fortran code
    # has to be be matched in analytint!
    amp = np.abs(rho) / np.abs(ana)
    ph = np.angle(rho) - np.angle(ana)
    #    print(np.abs(rho),np.abs(ana),np.angle(rho),np.angle(ana))
    return amp, ph


# In[4]:


def analytint(eps1, eps2, tcut, l1, l2, amp, ph, pw):
    k1 = np.real(np.sqrt(2.0 * eps1))
    k2 = np.real(np.sqrt(2.0 * eps2))
    lmb = 0.1 * np.pi / np.sqrt(k1**2 + k2**2)
    ig = 7
    ij = complex(0.0, 1.0)
    integral = complex(0.0, 0.0)
    integral1 = complex(0.0, 0.0)
    integral2 = complex(0.0, 0.0)
    norm2 = np.sqrt(2.0 / np.pi / np.sqrt(2.0 * eps2))

    tol = 0.000001
    absc, weights = np.polynomial.legendre.leggauss(ig)
    r = tcut
    last = 100.0
    nmax = 1000
    n = 0
    while abs(last) > tol and n < nmax:
        rp = tcut + n * lmb * ij
        rm = tcut - n * lmb * ij
        i = 0
        delta1 = complex(0.0, 0.0)
        delta2 = complex(0.0, 0.0)
        while i < ig:
            r = absc[i] * lmb * ij / 2.0 + rp + lmb * ij / 2.0
            #            if(n<2):
            #                print(n,rp,r)
            hpl1 = -Hplmn(1.0, l1, eps1, r)
            # this sign is to fix how we are doing in the fortran code
            hpl2 = Hplmn(1.0, l2, eps2, r)
            delta1 = delta1 + weights[i] * hpl1 * hpl2 * (r**pw) * (lmb * ij) / 2.0
            #            print(r,hpl1,hpl2,delta)
            if k1 > k2:
                hmn2 = Hplmn(-1.0, l2, eps2, r)
                delta2 = delta2 - weights[i] * hpl1 * hmn2 * (r**pw) * (lmb * ij) / 2.0

            else:
                r = -absc[i] * lmb * ij / 2.0 + rm - lmb * ij / 2.0
                hmn2 = Hplmn(-1.0, l2, eps2, r)
                hpl1 = -Hplmn(1.0, l1, eps1, r)
                # this sign is to fix how we are doing in the fortran code
                delta2 = delta2 - weights[i] * hpl1 * hmn2 * (r**pw) * (-lmb * ij) / 2.0
            #                print(r,hpl1,hmn2,delta)
            i = i + 1

        delta1 = delta1 / (2.0 * ij) * amp * np.exp(ij * ph) * norm2
        delta2 = delta2 / (2.0 * ij) * amp * np.exp(ij * ph) * norm2
        #        print('multiplied with',amp,norm2,' =',amp*norm2)

        d1 = "{:.2e}".format(delta1)
        d2 = "{:.2e}".format(delta2)

        delta = delta1 + delta2
        integral = integral + delta1 + delta2
        integral1 = integral1 + delta1
        integral2 = integral2 + delta2
        #        print('r=',rp,'n=',n,'integral=',integral,'delta=',delta)
        n = n + 1
        if n > 10:
            last = delta1 / integral1 + delta2 / integral2
    #    print('ana int from',np.real(tcut),'1st',np.round(integral1,3),\
    #          '+2nd',np.round(integral2,3),'=',np.round(integral,3))

    #    print(' after',n,'steps','integral1', integral1,\
    #          'integral2',integral2,' rp,rm=',rp,rm)
    return integral


# In[5]:


def ordered_closure(vec, eig, eomega, rr, weights, t, nknot, kord, vin, bdip):

    ndim = nknot - kord - 2
    ij = complex(0.0, 1.0)
    print(" Constructing matricies with dimension", ndim, " and", rr.size)
    dt = np.complex128
    b = np.zeros((ndim, ndim, rr.size), dtype=dt)
    ccr = np.zeros((ndim, rr.size), dtype=dt)
    cci = np.zeros((ndim, rr.size), dtype=dt)
    cccrr = np.zeros((ndim, rr.size), dtype=dt)
    cccri = np.zeros((ndim, rr.size), dtype=dt)
    cccir = np.zeros((ndim, rr.size), dtype=dt)
    cccii = np.zeros((ndim, rr.size), dtype=dt)

    c = np.zeros((ndim, ndim), dtype=dt)

    outr = np.zeros((rr.size), dtype=dt)
    outi = np.zeros((rr.size), dtype=dt)
    testr = np.zeros((ndim), dtype=dt)
    testkr = np.zeros((ndim, rr.size), dtype=dt)
    testi = np.zeros((ndim), dtype=dt)
    testki = np.zeros((ndim, rr.size), dtype=dt)

    print("b ", ndim, " x ", rr.size)
    print("bout ", rr.size)
    print("vsum ", rr.size, " x ", ndim)

    evec = 1.0 / (eomega - eig)
    i = 0
    while i < ndim:
        c[i, :] = np.multiply(vecd[i, :], evec)
        i = i + 1
    green = np.matmul(c, np.transpose(vec))
    #   this is |r><r| /(e0+omega-er)

    k = 0
    while k < rr.size:
        left = getleft(rr[k], t)
        sp = bsplvb_cmplx(t, kord, rr[k], left)

        ii = 0
        while ii < kord:
            indexi = left - kord + ii + 1
            jj = 0
            if np.logical_and(indexi > 0, indexi < ndim + 1):
                while jj < kord:
                    indexj = left - kord + jj + 1
                    if np.logical_and(indexj > 0, indexj < ndim + 1):
                        b[indexi - 1, indexj - 1, k] = (
                            b[indexi - 1, indexj - 1, k]
                            + sp[ii] * sp[jj] * weights[k] * rr[k]
                        )
                    jj = jj + 1
            ii = ii + 1
        k = k + 1
    ########################################
    print("ready with first part")

    k = 0
    while k < rr.size:
        ccr[:, k] = np.matmul(np.real(b[:, :, k]), vin)
        cci[:, k] = np.matmul(np.imag(b[:, :, k]), vin)
        cccrr[:, k] = np.matmul(np.real(green), ccr[:, k])
        cccri[:, k] = np.matmul(np.real(green), cci[:, k]) * ij
        cccir[:, k] = np.matmul(np.imag(green), ccr[:, k]) * ij
        cccii[:, k] = np.matmul(np.imag(green), cci[:, k]) * (-1)
        k = k + 1

    ##################################################
    print("ready with second part")

    i = 0
    while i < ndim:
        k = 0
        testr[i] = testr[i] + cccrr[i, k] + cccii[i, k]
        testkr[i, k] = testkr[i, k] + cccrr[i, k] + cccii[i, k]
        testi[i] = testi[i] + cccri[i, k] + cccii[i, k]
        testki[i, k] = testki[i, k] + cccri[i, k] + cccir[i, k]
        k = 1
        while k < rr.size:
            testr[i] = testr[i] + cccrr[i, k] + cccii[i, k]
            testi[i] = testi[i] + cccri[i, k] + cccir[i, k]
            testkr[i, k] = testkr[i, k - 1] + cccrr[i, k] + cccii[i, k]
            testki[i, k] = testki[i, k - 1] + cccri[i, k] + cccir[i, k]
            k = k + 1
        i = i + 1
    newr = getfunc_in_r(nknot, kord, t, testr, rr)
    newi = getfunc_in_r(nknot, kord, t, testi, rr)
    k1 = 0
    while k1 < rr.size:

        outr[k1] = getfunc(nknot, kord, t, testkr[:, k1], 0, rr[k1])
        outi[k1] = getfunc(nknot, kord, t, testki[:, k1], 0, rr[k1])
        k1 = k1 + 1

    return outr, outi, newr, newi


# In[11]:


def perturbedwf_mod2(fin, ein, vec, eig, omega, bdip):
    dt = np.complex128
    ndim = eig.size
    v = np.matmul(bdip, fin)
    gr = np.zeros((ndim), dtype=dt)
    gi = np.zeros((ndim), dtype=dt)
    k = 0
    ij = complex(0.0, 1.0)
    while k < eig.size:

        j = 0
        while j < eig.size:
            i = 0
            while i < eig.size:
                gr[i] = (
                    gr[i]
                    + np.real(vec[i, k] * vec[j, k] / (ein + omega - eig[k])) * v[j]
                )
                gi[i] = (
                    gi[i]
                    + np.imag(vec[i, k] * vec[j, k] / (ein + omega - eig[k])) * v[j]
                )
                i = i + 1
            j = j + 1
        k = k + 1
    return gr, gi


# In[12]:


def perturbedwf_mod3(fin, ein, vec, eig, omega, bdip):
    dt = np.complex128
    ndim = eig.size
    v = np.matmul(bdip, fin)
    frrr = np.zeros((ndim), dtype=dt)
    firr = np.zeros((ndim), dtype=dt)
    frir = np.zeros((ndim), dtype=dt)
    fiir = np.zeros((ndim), dtype=dt)
    frri = np.zeros((ndim), dtype=dt)
    firi = np.zeros((ndim), dtype=dt)
    frii = np.zeros((ndim), dtype=dt)
    fiii = np.zeros((ndim), dtype=dt)
    k = 0
    ij = complex(0.0, 1.0)
    while k < eig.size:

        j = 0
        den = 1.0 / (ein + omega - eig[k])
        while j < eig.size:
            i = 0
            aj = vec[j, k]
            while i < eig.size:
                ai = vec[i, k]
                frrr[i] = frrr[i] + np.real(ai) * np.real(aj) * np.real(den) * v[j]
                firr[i] = firr[i] + np.imag(ai) * np.real(aj) * np.real(den) * v[j]
                frir[i] = frir[i] + np.real(ai) * np.imag(aj) * np.real(den) * v[j]
                fiir[i] = fiir[i] + np.imag(ai) * np.imag(aj) * np.real(den) * v[j]
                frri[i] = frri[i] + np.real(ai) * np.real(aj) * np.imag(den) * v[j]
                firi[i] = firi[i] + np.imag(ai) * np.real(aj) * np.imag(den) * v[j]
                frii[i] = frii[i] + np.real(ai) * np.imag(aj) * np.imag(den) * v[j]
                fiii[i] = fiii[i] + np.imag(ai) * np.imag(aj) * np.imag(den) * v[j]

                i = i + 1
            j = j + 1
        k = k + 1
    return frrr, firr, frir, fiir, frri, firi, frii, fiii


# In[13]:


def perturbedwf_mod(fin, ein, vec, eig, omega, bdip):
    dt = np.complex128
    outr = np.zeros(fin.size, dtype=dt)
    outi = np.zeros(fin.size, dtype=dt)
    v = np.matmul(bdip, fin)
    ij = complex(0.0, 1.0)
    i = 0
    # r and i refer here to the real and imaginary part of the
    # Greens function
    while i < eig.size:
        matelemr = np.dot(np.real(vec[:, i]), v)
        matelemi = ij * np.dot(np.imag(vec[:, i]), v)
        Er = np.real(1.0 / (ein + omega - eig[i]))
        Ei = ij * np.imag(1.0 / (ein + omega - eig[i]))
        fr = np.real(vec[:, i])
        fi = ij * np.imag(vec[:, i])
        outr = (
            outr
            + fr * matelemr * Er
            + fr * matelemi * Ei
            + fi * matelemr * Ei
            + fi * matelemi * Er
        )

        outi = (
            outi
            + fi * matelemi * Ei
            + fr * matelemr * Ei
            + fr * matelemi * Er
            + fi * matelemr * Er
        )

        i = i + 1
    return outr, outi


# In[14]:


def contproj(rr, weights, t, nknot, kord, l, eps, tcut):
    ig = kord + 1
    ndim = nknot - kord - 2
    dt = np.complex128
    brep = np.zeros(ndim, dtype=dt)
    brepplus = np.zeros(ndim, dtype=dt)
    ij = complex(0.0, 1.0)
    i = kord - 1
    k = 0

    norm = np.sqrt(2.0 / np.pi / np.sqrt(2.0 * eps))
    while k < rr.size:
        if np.real(rr[k]) < np.real(tcut):
            reg = (
                (Hplmn(1.0, l, eps, rr[k]) - Hplmn(-1.0, l, eps, rr[k]))
                / (2.0 * ij)
                * norm
            )
            hplus = Hplmn(1.0, l, eps, rr[k])
            left = getleft(rr[k], t)
            sp = bsplvb_cmplx(t, kord, rr[k], left)
            ii = 0
            while ii < kord:
                indexi = left - kord + ii + 1

                if np.logical_and(indexi > 0, indexi < ndim + 1):
                    brep[indexi - 1] = brep[indexi - 1] + sp[ii] * weights[k] * reg
                    brepplus[indexi - 1] = (
                        brepplus[indexi - 1] + sp[ii] * weights[k] * hplus
                    )
                ii = ii + 1
        k = k + 1

    #        i=i+1
    #    print('for tlast=',tcut,'last spline=',indx)
    return brep, brepplus


# In[ ]:


def rhs(fin, vec, eig, bdip):
    dt = np.complex128
    out = np.zeros(fin.size, dtype=dt)
    v = np.matmul(bdip, fin)
    i = 0
    while i < eig.size:
        matelem = np.dot(vec[:, i], v)
        out = out + vec[:, i] * matelem
        i = i + 1
    return out


# In[ ]:


def rhsold(fin, ein, vec, eig, omega, bdip):
    dt = np.complex128
    out = np.zeros(fin.size, dtype=dt)
    temp = np.zeros(fin.size, dtype=dt)
    temp2 = np.zeros(fin.size, dtype=dt)
    i = 0
    while i < eig.size:
        j = 0
        while j < eig.size:
            temp[i] = temp[i] + bdip[i, j] * fin[j]
            j = j + 1
        i = i + 1
    i = 0
    while i < eig.size:
        j = 0
        while j < eig.size:
            temp2[i] = temp2[i] + vec[j, i] * temp[j] / (ein + 2.0 * omega - eig[i])
            j = j + 1
        i = i + 1
    i = 0
    while i < eig.size:
        j = 0
        while j < eig.size:
            out[i] = out[i] + vec[i, j] * temp2[j]
            j = j + 1
        i = i + 1
    return out
