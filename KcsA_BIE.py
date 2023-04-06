"""
1D model to simulate 3D PNP for KcsA ion channel from the paper:
    Gardner, C.L., Nonner, W. and Eisenberg, R.S., 2004.
    Electrodiffusion model simulation of ionic channels: 1D simulations.
    Journal of Computational Electronics, 3(1), pp.25-31.
By Zhen, Oct. 6, 2022 at UM
"""
import matplotlib.pyplot as plt
import numpy as np
eps = 1e-10
from time import perf_counter

class BEM4PNP1D:
    def __init__(self, Ls, lengthChannel, lengthBath, numIntervalInBath, DiffB, DiffC, QB, QC, ZB, ZC, epsB, epsC,
                     phiB, ciB, h, Zs, RB, RC, chi1, chi2, omega, maxiter, tol):
        self.Ls = Ls  # the left endpoint
        self.lengthChannel = lengthChannel  # this is a list including the length of each subinterval in the channel
        self.lengthBath = lengthBath
        self.numIntervalInBath = numIntervalInBath
        self.DiffB = DiffB
        self.DiffC = DiffC
        self.QB = QB
        self.QC = QC
        self.ZB = ZB
        self.ZC = ZC
        self.epsB = epsB
        self.epsC = epsC
        self.phiB = phiB
        self.ciB = ciB
        self.h = h
        self.Zs = Zs
        self.RB = RB
        self.RC = RC
        self.chi1 = chi1
        self.chi2 = chi2
        self.omega = omega
        self.maxiter = maxiter
        self.tol = tol
        self.NA = 6.02214076e23  # Avogadro number
        self.e = 1.60217663e-19  # elementary charge and measured in coulomb

    def generateMesh(self, Ls, hs):
        """
        Generate mesh grid
        :param Ls: the left end point, fox example, x =-5
        :param lls: the length of each subdomain, it is a list
        :param hs: the mesh size of each subdomain, it is a list
        :return: xs: it is a numpy array
        """
        numIntervalInBath = self.numIntervalInBath
        lengthBath = self.lengthBath
        lengthChannel = self.lengthChannel
        ns = [] # number of subintervals in each subdomain, here is a list
        xs = []
        lls = []
        h = np.array(lengthBath)/ np.array(numIntervalInBath)
        hLeftBath = [h[0] for i in range(numIntervalInBath[0])]
        hRightBath = [h[1] for i in range(numIntervalInBath[1])]
        lls.extend(hLeftBath)
        lls.extend(lengthChannel)
        lls.extend(hRightBath)
        for i, (length, meshsize) in enumerate(zip(lls, hs)):
            # number of subintervals in each subdomain
            num_grid = int(round(length / float("{:.5f}".format(meshsize))))
            x = np.linspace(Ls + sum(lls[:i]), Ls + sum(lls[:i + 1]), num_grid + 1)
            ns.append(num_grid)
            xs.append(x)

        return xs, ns, lls

    def calculateG(self, xs):
        m = self.num_Domain
        ns = self.ns
        G = []
        Gx = []
        Gy = []
        for k in range(m):
            xs_k = xs[k]
            ns_k = ns[k]
            # Calculate G(x, y)
            G_k = np.zeros((ns_k, ns_k+1))
            Gx_k = np.zeros((ns_k, ns_k+1))
            Gy_k = np.zeros((ns_k, ns_k+1))
            for j in range(ns_k + 1):  # y
                for i in range(ns_k):  # x
                    ave_i = 0.5 * (xs_k[i] + xs_k[i + 1])
                    G_k[i, j] = -0.5 * abs(ave_i - xs_k[j])
                    if ave_i >= xs_k[j]:
                        Gx_k[i, j] = -0.5
                        Gy_k[i, j] = 0.5
                    else:
                        Gx_k[i, j] = 0.5
                        Gy_k[i, j] = -0.5
            G.append(G_k)
            Gx.append(Gx_k)
            Gy.append(Gy_k)
        return G, Gx, Gy

    def doPreparation(self):
        Ls = self.Ls
        numIntervalInBath = self.numIntervalInBath
        numIntervalInChannel = len(self.lengthChannel)
        hs = []
        h = self.h
        hs.extend([h for i in range(numIntervalInBath[0])])
        hs.extend([h for i in range(numIntervalInChannel)])
        hs.extend([h for i in range(numIntervalInBath[1])])
        self.hs = hs
        xs, ns, lls = self.generateMesh(Ls, hs)
        self.xs = xs
        self.ns = ns
        self.lls = lls

        # Generate Qs: a list of fixed charge in each subinterval
        QB = self.QB
        QC = self.QC
        Qs = []
        QsL = [QB[0] for i in range(numIntervalInBath[0])]
        QsR = [QB[1] for i in range(numIntervalInBath[1])]
        Qs.extend(QsL)
        Qs.extend(QC)
        Qs.extend(QsR)
        self.Qs = Qs

        ZB = self.ZB
        ZC = self.ZC
        Zf = []
        ZfL = [ZB[0] for i in range(numIntervalInBath[0])]
        ZfR = [ZB[1] for i in range(numIntervalInBath[1])]
        Zf.extend(ZfL)
        Zf.extend(ZC)
        Zf.extend(ZfR)
        self.Zf = Zf

        # Generate eps
        epsB = self.epsB
        epsC = self.epsC
        eps = []
        epsL = [epsB[0] for i in range(numIntervalInBath[0])]
        epsR = [epsB[1] for i in range(numIntervalInBath[1])]
        eps.extend(epsL)
        eps.extend(epsC)
        eps.extend(epsR)
        self.eps = eps

        # Generate eps
        DiffB = self.DiffB
        DiffC = self.DiffC
        Diffs = []
        DiffL = [DiffB[0] for i in range(numIntervalInBath[0])]
        DiffR = [DiffB[1] for i in range(numIntervalInBath[1])]
        Diffs.extend(DiffL)
        Diffs.extend(DiffC)
        Diffs.extend(DiffR)
        self.Diffs = Diffs

        NA = self.NA
        self.num_Domain = len(lls)
        # generate mesh grids

        self.G, self.Gx, self.Gy = self.calculateG(xs)

        # Generate radius: a list of fixed charge in each subinterval
        RB = self.RB
        RC = self.RC
        lengthBath = self.lengthBath
        hBath = np.array(lengthBath)/np.array(numIntervalInBath)
        Rs = []
        RsL = [RB - i *hBath[0] for i in range(numIntervalInBath[0])]
        RsR = [RC + i * hBath[0] for i in range(1, numIntervalInBath[0]+1)]
        Rs.extend(RsL)
        Rs.extend([RC for i in range(numIntervalInChannel)])
        Rs.extend(RsR)
        self.Rs = Rs
        # calculate fixed charge density and measured in molar, 1e24: dm^3 to nm^3, it is a list
        rho = np.array(Qs) / (np.array(lls) * np.pi * np.array(Rs) ** 2) * 1e24 / NA
        self.As = np.pi * (np.array(Rs) ** 2)
        self.rho = rho

    def chargeDensity4Plot(self):
        # define fixed charge density function and measured in log10(#/10^21)/cm^3
        rho = self.rho  # measured in molar
        m = self.num_Domain
        xs = self.xs
        NA = self.NA
        chargeDen = []
        for i in range(m):
            if abs(rho[i]) <= 1.0e-10:
                TEMP = [0 for j in range(len(xs[i]))]
                chargeDen.append(np.array(TEMP))
            else:
                TEMP = [np.log10(np.abs(rho[i] * NA / (1.0e24))) for j in range(len(xs[i]))]
                chargeDen.append(np.array(TEMP))
        return chargeDen

    def subLinearSystem4Poisson(self, ll_k, lr_k, cs_k, rho_k, eps_k, Zs, Zf_k, chi2, hs_k, xs_k):
        """
        Generate sublinear system of the poisson equation on the k-th subdomain
        :param ll_k: the left endpoint of the k-th subdomain
        :param lr_k: the right endpoint of the k-th subdomain
        :param cs_k: [c1, c2, ....] values on the k-th subdomain
        :param rho_k: the fixed charge density on the k-th subdomain
        :param eps_k: the permittivity of the k-th subdomain
        :param Zs: the valence of ions
        :param Zf: the valence of the fixed charge
        :param chi2: fixed parameter for poisson equation
        :return: Ak : submatrix
        :return: bk : the related right-hand side
        """
        Ak = np.array([[-2, 2, ll_k - lr_k, ll_k - lr_k],
                       [0,  0, -1,          1]])
        # calculate the total concentration and moment over the k-th subdomain by using the trapezoidal rule
        totalCon_k = 0
        totalMon_k = 0
        for zi, ci in zip(Zs, cs_k):
            totalCon_k += zi * hs_k * 0.5 * sum(ci[:-1] + ci[1:])  # sum zi * ai
            totalMon_k += zi * hs_k * 0.5 * sum(xs_k[:-1] * ci[:-1] + xs_k[1:] * ci[1:])  # sum zi * bi
        # add  Zf * af_k and Zf * bf_k
        totalCon_k += Zf_k * rho_k * (lr_k - ll_k)  # int Zf * af_k, here rho_k is a constant
        totalMon_k += Zf_k * rho_k * 0.5 * (lr_k ** 2 - ll_k ** 2)  # int Zf * bf_k, here rho_k is a constant
        bk = np.array([-chi2/eps_k * ((lr_k + ll_k) * totalCon_k - 2.0 * totalMon_k),
                       -chi2/eps_k * totalCon_k])
        return bk, Ak

    def solveP(self, cs):
        """
        solve the poisson equation with Dirichlet boundary conditions
        :param cs: a list, each element is a numpy array
        :return: phi, Dphi
        """
        Zs = self.Zs
        Zf = self.Zf
        rho = self.rho
        xs = self.xs
        Ls = self.Ls
        lls = self.lls
        hs = self.hs
        As = self.As
        eps = self.eps
        m = self.num_Domain
        Nd = 4 * m
        AP = np.zeros((Nd, Nd))
        bP = np.zeros(Nd)
        # Generate the linear system
        for k, (rho_k, eps_k, hs_k, xs_k, Zf_k) in enumerate(zip(rho, eps, hs, xs, Zf)):
            ll_k = Ls + sum(lls[:k])
            lr_k = Ls + sum(lls[:k+1])
            cs_k = []
            for i in range(len(Zs)):
                cs_k.append(cs[i][k])
            bk, Ak = self.subLinearSystem4Poisson(ll_k, lr_k, cs_k, rho_k, eps_k, Zs, Zf_k, chi2, hs_k, xs_k)
            AP[4 * k + 1:4 * k + 3, 4 * k:4 * k + 4] = Ak
            bP[4 * k + 1:4 * k + 3] = bk
            # insert the interface conditions
            if k < m-1:
                AP[4 * k + 3, 4 * k + 3:4 * k + 7] = [eps[k]*As[k], 0, 0, -eps[k+1]*As[k+1]]  # interface 1
                AP[4 * k + 4, 4 * k + 1:4 * k + 5] = [1, 0, 0, -1]  # interface 2
        # set the boundary conditions
        AP[0, 0] = 1
        AP[-1, -3] = 1
        phiB = self.phiB
        bP[0] = phiB[0]
        bP[-1] = phiB[1]
        # solve linear system
        X = np.linalg.solve(AP, bP)
        phi = []
        Dphi = []
        # Calculate potential and potential gradient at the interior points
        G = self.G
        Gy = self.Gy
        ns = self.ns  # number of subintervals
        for k in range(m):
            # k-th subdomain
            cs_k = []
            for i in range(len(Zs)):
                cs_k.append(cs[i][k])
            xs_k = xs[k]
            eps_k = eps[k]
            h_k = hs[k]
            G_k = G[k]
            Gy_k = Gy[k]
            rho_k = rho[k]
            ns_k = ns[k]
            Zf_k = Zf[k]
            phi_k = np.zeros(ns_k+1)
            Dphi_k = np.zeros(ns_k+1)
            phi_k[0] = X[4*k]  # left end point of the k-subinterval
            phi_k[-1] = X[4*k+1]  # right end point of the k-subinterval
            Dphi_k[0] = X[4*k+2]
            Dphi_k[-1] = X[4*k+3]
            for j in range(1, ns_k):
                source = 0
                sourceG = 0
                for zi, ci in zip(Zs, cs_k):
                    source += zi * sum(G_k[:, j] * 0.5 * (ci[:-1] + ci[1:]) * h_k)
                    sourceG += zi * sum(Gy_k[:, j] * 0.5 * (ci[:-1] + ci[1:]) * h_k)
                # add the volume integral of the fixed point charge
                source += Zf_k * rho_k * h_k * sum(G_k[:, j])
                sourceG += Zf_k * rho_k * h_k * sum(Gy_k[:, j])
                phi_k[j] = (-0.5* abs(xs_k[-1] - xs_k[j]) * Dphi_k[-1] - (-0.5) * phi_k[-1]) - \
                           (-0.5 * abs(xs_k[0] - xs_k[j]) * Dphi_k[0] - 0.5 * phi_k[0]) + chi2 / eps_k * source
                Dphi_k[j] = 0.5 * Dphi_k[-1] - (-0.5) * Dphi_k[0] + chi2 / eps_k * sourceG
            Dphi_k[1: ns_k-1] = (phi_k[2:ns_k] - phi_k[0:ns_k-2]) / (2*h_k)
            phi.append(phi_k)
            Dphi.append(Dphi_k)
        return np.array(phi, dtype=object), np.array(Dphi, dtype=object)

    def subLinearSystem4NP(self, ll_k, lr_k, cs_k, zi, chi1, hs_k, Dphi_k):
        """
        Generate sublinear system of the poisson equation on the k-th subdomain
        :param ll_k: the left endpoint of the k-th subdomain
        :param lr_k: the right endpoint of the k-th subdomain
        :param cs_k: [c1, c2, ....] values on the k-th subdomain
        :param rho_k: the fixed charge density on the k-th subdomain
        :param eps_k: the permittivity of the k-th subdomain
        :param Zs: the valence of ions
        :param Zf: the valence of the fixed charge
        :param chi2: fixed parameter for poisson equation
        :return: Ak : submatrix
        :return: bk : the related right-hand side
        """
        Ak = np.array([[-2, 2, ll_k - lr_k, ll_k - lr_k],
                       [0, 0, -1, 1]])
        # calculate the volume integral over the k-th subdomain by using the trapezoidal rule
        vol_integral = 0.5 * sum(cs_k[:-1] * Dphi_k[:-1] + cs_k[1:] * Dphi_k[1:]) * hs_k
        bk = np.array([-chi1 * zi * (2.0 * vol_integral - (lr_k - ll_k)*(cs_k[0] * Dphi_k[0] + cs_k[-1] * Dphi_k[-1])),
                       -chi1 * zi * (cs_k[-1] * Dphi_k[-1] - cs_k[0] * Dphi_k[0])])
        return bk, Ak

    def solveNP(self, Dphi, ci0, zi):
        m = self.num_Domain
        chi1 = self.chi1
        Diffs = self.Diffs
        xs = self.xs
        ns = self.ns
        hs = self.hs
        lls = self.lls
        Nd = 4 * m
        As = self.As
        ANP = np.zeros((Nd, Nd))
        bNP = np.zeros(Nd)
        # Generate the linear system
        for k, hs_k in enumerate(hs):
            ll_k = Ls + sum(lls[:k])
            lr_k = Ls + sum(lls[:k + 1])
            bk, Ak = self.subLinearSystem4NP(ll_k, lr_k, ci0[k], zi, chi1, hs_k, Dphi[k])
            ANP[4 * k + 1:4 * k + 3, 4 * k:4 * k + 4] = Ak
            bNP[4 * k + 1:4 * k + 3] = bk
            # insert the interface conditions
            if k < m - 1:
                ANP[4 * k + 3, 4 * k + 3:4 * k + 7] = [Diffs[k] * As[k], 0, 0, -Diffs[k + 1] * As[k + 1]]
                # ANP[4 * k + 3, 4 * k + 1] = zi * chi1 * Diffs[k] * As[k] * Dphi[k][-1]  # interface 1
                # ANP[4 * k + 3, 4 * k + 4] = -zi * chi1 * Diffs[k+1] * As[k+1] * Dphi[k+1][0]
                ANP[4 * k + 4, 4 * k + 1:4 * k + 5] = [1, 0, 0, -1]  # interface 2
                bNP[4 * k + 3] = -zi * chi1 * (Diffs[k] * As[k] * ci0[k][-1] * Dphi[k][-1] - Diffs[k+1] * As[k+1] * ci0[k+1][0] * Dphi[k+1][0])
        # set the boundary conditions
        ANP[0, 0] = 1
        ANP[-1, -3] = 1
        ciBB = self.ciB
        bNP[0] = ciBB[0]
        bNP[-1] = ciBB[1]
        self.ANP = ANP
        self.bNP = bNP
        # solve linear system
        Y = np.linalg.solve(ANP, bNP)
        # N = sum(self.ns) + 1
        G = self.G
        Gy = self.Gy
        hs = self.hs
        ci = []
        Dci = []
        for k in range(m):
            # k-th subdomain
            h_k = hs[k]
            G_k = G[k]
            Gy_k = Gy[k]
            ci0_k = ci0[k]
            xs_k = xs[k]
            ns_k = ns[k]
            ci_k = np.zeros(ns_k+1)
            Dci_k = np.zeros(ns_k+1)
            ci_k[0] = Y[4*k]
            ci_k[-1] = Y[4*k+1]
            Dci_k[0] = Y[4*k+2]
            Dci_k[-1] = Y[4*k+3]
            Dphi_k = Dphi[k]
            for j in range(1, ns_k):
                source = sum(G_k[:, j] * (ci0_k[1:] * Dphi_k[1:] - ci0_k[:-1] * Dphi_k[:-1]))
                sourceG = sum(Gy_k[:, j] * (ci0_k[1:] * Dphi_k[1:] - ci0_k[:-1] * Dphi_k[:-1]))
                ci_k[j] = (-0.5 *  abs(xs_k[-1] - xs_k[j]) * Dci_k[-1] - (-0.5) * ci_k[-1]) - \
                           (-0.5 * abs(xs_k[0] - xs_k[j]) * Dci_k[0] - 0.5 * ci_k[0]) + chi1 * zi * source
                Dci_k[j] = 0.5 * Dci_k[-1] - (-0.5) * Dci_k[0] + chi1 * zi * sourceG
            Dci_k[1: ns_k - 1] = (ci_k[2:ns_k] - ci_k[0:ns_k - 2]) / (2 * h_k)
            ci.append(ci_k)
            Dci.append(Dci_k)
        return np.array(ci, dtype=object), np.array(Dci, dtype=object)

    def iterative(self, cs0, phi0, Dphi0):
        # Do the Gummel iteration
        num_domain = self.num_Domain
        omega = np.zeros(num_domain)
        omega[:] = self.omega
        Iter = 0
        resi = 1000.0
        sor_tol = self.tol
        sor_max_iter = self.maxiter
        m = len(self.hs)
        print("\n     Start the iterative scheme")
        print("\n      Ite.      Error\n")
        print("        0      ")
        while resi > sor_tol and Iter < sor_max_iter:
            phi, Dphi = self.solveP(cs0)
            phi1 = omega * phi + (1 - omega) * phi0
            Dphi1 = omega * Dphi + (1 - omega) * Dphi0
            resiU = []
            for i in range(m):
                # resiU.append(np.linalg.norm(phi[i] - phi0[i]))
                resiU.append(np.linalg.norm(Dphi[i] - Dphi0[i]))
            resiU = max(resiU)
            phi0 = phi1
            Dphi0 = Dphi1
            errorC_norm = []
            cs = []
            Dcs = []
            for zi, ci0 in zip(Zs, cs0):
                 ci, Dci = self.solveNP(Dphi1, ci0, zi)
                 err = []
                 for i in range(m):
                    err.append(np.linalg.norm(ci[i] - ci0[i]))
                 errorC_norm.append(max(err))
                 Dcs.append(Dci)
                 cs.append(omega * ci + (1 - omega) * ci0)
            self.cs = cs
            errorC = max(errorC_norm)
            resi = max(resiU, errorC)
            cs0 = cs
            Iter += 1
            print(('     %4d       %10.4e    ' % (Iter, resi)))
        return phi, cs, Dphi, Dcs

    def plotFigure(self, phi, Dphi, cs, Dcs, fileName):
        # plt.rcParams['text.usetex'] = True
        plt.figure(num=4)
        fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, 1, figsize=(20, 60))
        m = self.num_Domain
        xs = self.xs
        I = self.calculateCurrent(cs, Dcs, Dphi)
        Ls = self.Ls
        lls = self.lls
        xlabels = []

        for i in range(m+1):
            xlabels.append(float("{:.2f}".format(Ls + sum(lls[:i]))))
        ax1.plot(xs[0], phi[0] * 1000, linewidth=8)
        ax2.plot(xs[0], Dphi[0], linewidth=8)
        for i in range(1, m-2):
            ax1.plot(xs[i], phi[i]*1000,   linewidth=8)
            ax2.plot(xs[i], Dphi[i],   linewidth=8)
            ciNew = np.log10(np.abs(cs[0][i] * self.NA / (1.0e24)))
            ax3.plot(xs[i], ciNew,  linewidth=8)
            ciNew = np.log10(np.abs(cs[1][i] * self.NA / (1.0e24)))
            ax3.plot(xs[i], ciNew,   linewidth=8)
            ax4.plot(xs[i], Dcs[0][i],  linewidth=8)
            ax4.plot(xs[i], Dcs[1][i],   linewidth=8)
            ax5.yaxis.offsetText.set_fontsize(60)
            ax5.plot(xs[i], I[0][i],   linewidth=8)
            ax5.plot(xs[i], I[1][i],   linewidth=8)

        ax1.plot(xs[-2], phi[-2]*1000,  linewidth=8)
        ax2.plot(xs[-2], Dphi[-2],   linewidth=8)
        ax1.plot(xs[-1], phi[-1]*1000, label=r'$\phi$', linewidth=8)
        ax2.plot(xs[-1], Dphi[-1], label=r'$\phi^\prime$', linewidth=8)
        ciNew = np.log10(np.abs(cs[0][-2] * self.NA / (1.0e24)))
        ax3.plot(xs[-2], ciNew, label=r'Cl$^-$', linewidth=8)
        ciNew = np.log10(np.abs(cs[1][-2] * self.NA / (1.0e24)))
        ax3.plot(xs[-2], ciNew, label=r'K$^+$', linewidth=8)
        ax4.plot(xs[-2], Dcs[0][-2], label=r'Cl$^-$', linewidth=8)
        ax4.plot(xs[-2], Dcs[1][-2], label=r'K$^+$', linewidth=8)
        ax5.yaxis.offsetText.set_fontsize(60)
        ax5.plot(xs[-2], I[0][-2], label=r'$I^-$', linewidth=8)
        ax5.plot(xs[-2], I[1][-2], label=r'$I^+$', linewidth=8)

        fontsize = 40
        ax1.yaxis.offsetText.set_fontsize(fontsize)
        ax2.yaxis.offsetText.set_fontsize(fontsize)
        ax3.yaxis.offsetText.set_fontsize(fontsize)
        ax4.yaxis.offsetText.set_fontsize(fontsize)
        ax5.yaxis.offsetText.set_fontsize(fontsize)
        ax1.xaxis.set_tick_params(labelsize=60)
        ax1.yaxis.set_tick_params(labelsize=60)
        ax1.set_xticks(xlabels)
        ax1.set_xticklabels(xlabels)
        ax2.xaxis.set_tick_params(labelsize=60)
        ax2.yaxis.set_tick_params(labelsize=60)
        ax2.set_xticks(xlabels)
        ax2.set_xticklabels(xlabels)
        ax3.xaxis.set_tick_params(labelsize=60)
        ax3.yaxis.set_tick_params(labelsize=60)
        ax3.set_xticks(xlabels[1:-1])
        ax3.set_xticklabels(xlabels[1:-1])

        ax4.xaxis.set_tick_params(labelsize=60)
        ax4.yaxis.set_tick_params(labelsize=60)
        ax4.set_xticks(xlabels[1:-1])
        ax4.set_xticklabels(xlabels[1:-1])
        ax5.set_xticks(xlabels[1:-1])
        ax5.set_xticklabels(xlabels[1:-1])
        ax5.xaxis.set_tick_params(labelsize=60)
        ax5.yaxis.set_tick_params(labelsize=60)

        ax1.set_xlabel(r'$x$', fontsize=60)
        ax2.set_xlabel(r'$x$', fontsize=60)
        ax3.set_xlabel(r'$x$', fontsize=60)
        ax4.set_xlabel(r'$x$', fontsize=60)
        ax5.set_xlabel(r'$x$', fontsize=60)

        ax1.set_ylabel(r' $\phi(x)$', fontsize=60)
        ax2.set_ylabel(r' $\phi^\prime(x)$', fontsize=60)
        # ax2.set_ylabel(r' $\phi^\prime$', fontsize=40)
        ax3.set_ylabel(r'$\log(c_i(x))$', fontsize=60)
        ax4.set_ylabel(r' $c_i^\prime(x)$', fontsize=60)
        ax5.set_ylabel('Current ' + r'$I(x)$', fontsize=60)

        ax1.set_title('(a) Electrostatic potential', fontsize=60, pad=60)
        ax2.set_title('(b) Gradient of potential', fontsize=60, pad=60)
        ax3.set_title('(c) Concentration', fontsize=60, pad=60)
        ax4.set_title('(d) Gradient of Concentration', fontsize=60, pad=60)
        ax5.set_title('(e) Current', fontsize=60, pad=60)

        ax1.legend(fontsize=60)
        ax2.legend(fontsize=60)
        ax3.legend(fontsize=60)
        ax4.legend(fontsize=60)
        ax5.legend(fontsize=60)
        # plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.4, hspace=0.2)
        fig.tight_layout()
        plt.savefig(fileName)

    def plotFigure4Paper1(self, phi, cs):
        # plt.rcParams['text.usetex'] = True
        plt.rcParams['axes.linewidth'] = 3
        m = self.num_Domain
        xs = self.xs
        Ls = self.Ls
        lls = self.lls
        xlabels = []
        linewidth = 4
        for i in range(m + 1):
            xlabels.append(float("{:.2f}".format(Ls + sum(lls[:i]))))
        plt.figure(num=10)
        plt.figure(figsize=(15, 6))
        for i in range(m):
            plt.plot(xs[i], phi[i] * 1000, linewidth=linewidth, color='k')
        fontsize = 25
        plt.xlabel(r'$x$' + ' [nm]', fontsize=fontsize + 5)
        plt.ylabel(r'$\phi$' + ' [mV]', fontsize=fontsize + 5)
        plt.tick_params(which='both', length=7, width=3, direction="in")
        # plt.xticks([-5,  0, 1.3, 2.3, 3.5, 8.5], ['-5', '0', '1.3', '2.3', '3.5', '8.5'], fontsize=fontsize)
        plt.xticks([-5, -4, -3, -2, -1, 0, 1, 1.3, 2, 2.3, 3, 3.5, 4, 5, 6, 7, 8, 8.5],
                   ['-5', ' ', ' ', ' ', ' ', '0', ' ', '1.3', '', '2.3', ' ', '3.5', '', '', '', '', '', '8.5'],
                   fontsize=fontsize)
        plt.yticks((0, -25, -50, -75, -100, -125, -150), fontsize=fontsize)
        plt.xlim([-5, 8.5])
        plt.ylim([-160, 5])
        plt.tight_layout()
        # plt.legend(loc='upper left', bbox_to_anchor=(0.03, 1.03), fontsize=18, framealpha=0.0)
        plt.savefig('KcsApotential.png')
        plt.figure(num=31)
        plt.figure(figsize=(14, 6))
        chargeDen = self.chargeDensity4Plot()

        for i in range(0, m - 1):
            # c2new = np.log10(np.abs(cs[1][i] * self.NA / (1.0e24)))
            c2new = np.abs(cs[1][i] * self.NA / (1.0e24))
            plt.semilogy(xs[i], c2new, linewidth=4, color='r')
            # c1new = np.log10(np.abs(cs[0][i] * self.NA / (1.0e24)))
            c1new = np.abs(cs[0][i] * self.NA / (1.0e24))
            plt.semilogy(xs[i], c1new, linewidth=4, color='b')

        #
        mm =self.numIntervalInBath[0]-2
        # c2new = np.log10(np.abs(cs[1][i] * self.NA / (1.0e24)))
        c2new = np.abs(cs[1][mm] * self.NA / (1.0e24))
        plt.semilogy(xs[mm], c2new, linewidth=4, color='r',  label=r'K$^+$',
                     marker='s', markevery=[0], markersize=20, markerfacecolor='None', markeredgewidth=2)
        c1new = np.abs(cs[0][mm] * self.NA / (1.0e24))
        plt.semilogy(xs[mm], c1new, linewidth=4, color='b', label=r'Cl$^-$',
                     marker='o',  markevery=[0], markersize=20, markerfacecolor='None', markeredgewidth=2)

        c1new = np.abs(cs[0][m - 1] * self.NA / (1.0e24))
        plt.semilogy(xs[m - 1], c1new, linewidth=4, color='b')
        # c2new = np.log10(np.abs(cs[1][i] * self.NA / (1.0e24)))
        c2new = np.abs(cs[1][m - 1] * self.NA / (1.0e24))
        plt.semilogy(xs[m - 1], c2new, linewidth=4, color='r')

        plt.xlabel(r'$x$' + ' [nm]', fontsize=fontsize + 5)
        # plt.ylabel(r'$c_i(x)/(10^{21} \rm{{c m}}^{-3})$', fontsize=fontsize+5)
        plt.ylabel(r'$c_i / (10^{21}cm^{-3})$', fontsize=fontsize + 5)
        # plt.xticks([-5,  0, 1.3, 2.3, 3.5, 8.5], ['-5', '0', '1.3', '2.3', '3.5', '8.5'], fontsize=fontsize)
        plt.xticks([-5, -4, -3, -2, -1, 0, 1, 1.3, 2, 2.3, 3, 3.5, 4, 5, 6, 7, 8, 8.5],
                   ['-5', ' ', ' ', ' ', ' ', '0', ' ', '1.3', '', '2.3', ' ', '3.5', '', '', '', '', '', '8.5'],
                   fontsize=fontsize)
        # plt.yticks([-3, -2, -1, 0, 1], fontsize=fontsize)
        mB = self.numIntervalInBath[0]
        plt.vlines(x=0, ymin=10 ** (-3.3), ymax=10 ** (chargeDen[mB][0] - 0.02), colors='g', linewidth=linewidth,
                   linestyle='--', )

        plt.hlines(y=10 ** (chargeDen[mB][0] - 0.02), xmin=0, xmax=0.2, colors='g', linewidth=linewidth,
                   linestyle='--', )
        plt.vlines(x=0.2, ymin=10 ** (-3.3), ymax=10 ** (chargeDen[mB][0] - 0.02), colors='g', linewidth=linewidth,
                   linestyle='--', )

        plt.vlines(x=1.3, ymin=10 ** (-3.3), ymax=10 ** (chargeDen[mB + 2][0] + 0.02), colors='g', linewidth=linewidth,
                   linestyle='--', )
        plt.hlines(y=10 ** (chargeDen[mB + 2][0] + 0.02), xmin=1.3, xmax=2.3, colors='g', linewidth=linewidth,
                   linestyle='--', )
        # plt.vlines(x=2.3, ymin=chargeDen[mB+2][0] + 0.02, ymax=chargeDen[mB+3][0]+0.02, colors='g', linewidth=2,
        #            linestyle='--', )
        plt.vlines(x=2.3, ymin=10 ** (-3.3), ymax=10 ** (chargeDen[mB + 3][0] + 0.02), colors='g', linewidth=linewidth,
                   linestyle='--', )
        plt.hlines(y=10 ** (chargeDen[mB + 3][0] + 0.02), xmin=2.3, xmax=3.5, colors='g', linewidth=linewidth,
                   linestyle='--', label= r'$\rho_n$')
        plt.vlines(x=3.5, ymin=10 ** (-3.3), ymax=10 ** (chargeDen[mB + 3][0] + 0.02), colors='g', linewidth=linewidth,
                   linestyle='--', )
        # plt.tick_params(left=False)
        plt.ylim([10 ** (-3.3), 10 ** 1.8])
        # plt.xlim([-3, 6.5])
        plt.xlim([-5, 8.5])
        plt.tick_params(axis='x', length=7, width=3, direction="in")
        plt.yticks([10 ** (-3), 10 ** (-2), 10 ** (-1), 10 ** (0), 10 ** (1)],
                   [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'], fontsize=fontsize)
        plt.tick_params(axis='y', length=7, width=3, direction="in")
        plt.tight_layout()
        plt.legend(fontsize=22)
        plt.savefig('KcsAconcentration.png')

    def calculateCurrent(self, cs, Dcs, Dphi):
        Zs = self.Zs
        NA = self.NA
        e = self.e
        Diffs = self.Diffs
        num_domain = self.num_Domain
        current1 = []
        current2 = []
        As = self.As
        for i in range(num_domain):
            I1 = -e*NA/1000 * As[i] * Diffs[i] * Zs[0] * (Dcs[0][i] + Zs[0] * chi1 * cs[0][i] * Dphi[i])
            I2 = -e*NA/1000 * As[i] * Diffs[i] * Zs[1] * (Dcs[1][i] + Zs[1] * chi1 * cs[1][i] * Dphi[i])
            current1.append(I1)
            current2.append(I2)
        return [current1, current2]

    def plotIVcurve(self):
        plt.figure(num=2)
        plt.rcParams['axes.linewidth'] = 1.5
        xs = [0, 20, 40, 60, 80, 100]
        # No charge
        # ys = [0, 0.03809, 0.0733, 0.1037, 0.1288, 0.1495, 0.1667, 0.1815, 0.1947]
        ys =[0, 3.5, 7.1, 11.2, 15.3, 19.4]
        plt.scatter(xs, ys, facecolors="none", edgecolors='k', s=100)
        plt.tick_params(which='both', length=6, width=3, direction="in")
        plt.plot(xs, ys, '-', linewidth=3, color='k')
        plt.xlabel('applied voltage, ' + r'$V_{app}$'+ ' [mV]', fontsize=20)
        plt.ylabel('current, ' + r'$I$' + ' [pA]',  fontsize=20)
        plt.yticks([0, 5,10,15,20], ['0', '5', '10', '15', '20'],fontsize=20)
        plt.xticks([0,20,40,60,80, 100], ['0', '20', '40', '60','80', '100'], fontsize=20)
        # for x, y in zip(xs, ys):
        #     label = "{:.3f}".format(y)
        #     plt.annotate(label,  # this is the text
        #                  (x, y),  # this is the point to label
        #                  textcoords="offset points",  # how to position the text
        #                  xytext=(0, 10),  # distance from text to points (x,y)
        #                  ha='center')  # horizontal alignment can be left, right or center
        # # plt.suptitle(title, fontsize=20)
        plt.tight_layout()
        plt.savefig('IV.png')

    def initialConcentration(self):
        xs = self.xs
        ciB = self.ciB
        Zs = self.Zs
        cs0 = []
        m = self.num_Domain
        for i in range(len(Zs)):
            ci = []
            for k in range(m):
                xs_k = xs[k]
                ci_k = np.array([ciB[1] for x in xs_k])
                ci.append(ci_k)
            cs0.append(ci)
        cs0 = np.array(cs0, dtype=object)
        return cs0

    def initialPotential(self):
        xs = self.xs
        phiB = self.phiB
        phi0 = []
        m = self.num_Domain
        for k in range(m):
            xs_k = xs[k]
            phi0_k = np.array([phiB[0] for x in xs_k])
            phi0.append(phi0_k)
        phi0 =np.array(phi0, dtype=object)
        # phi0[0][0] = phiB[0]
        return phi0

    def DPotential(self):
        xs = self.xs
        Dphi0 = []
        m = self.num_Domain
        for k in range(m):
            xs_k = xs[k]
            Dphi0_k = np.array([1 for x in xs_k])
            Dphi0.append(Dphi0_k)
        return np.array(Dphi0, dtype=object)

def G(x, y):
    """
     Evaluate the value of the green function
     g(x, xbar) = -0.5*|x-y|
    """
    return -0.5*np.abs(x-y)

def Gx(x, y):
    if x>=y:
        gx = -0.5
    else:
        gx = 0.5
    return gx

def Gy(x, y):
    if x>=y:
        gy = 0.5
    else:
        gy = -0.5
    return gy


if __name__ == "__main__":
    # Define parameters of the PNP model,
    Ls = -5  # the left endpoint of the domain: unit nm
    # the length of each subinterval and measured in nm
    lengthChannel = [0.2, 1.1, 1.0, 1.2]  #[-4e group, nonpolar, cavity,filter ]
    lengthBath = [5,  5]   # [left bath, right bath]
    numIntervalInBath = [20, 20]  # [left bath, right bath]
    # Diffusion coefficient of each subinterval and measured in 10^{-5} * cm^2/s
    DiffB = [1.5, 1.5]  # [left bath, right bath]
    DiffC = [0.4, 0.4, 0.4, 0.5]
    # the fixed charges in channel, and measured in e
    QB = [0, 0]  # [left bath, right bath]
    QC = [4,   0,   0.5, 1.5]
    # the valence of fixed charges in channel
    ZB = [0, 0]  # [left bath, right bath]
    ZC = [-1,   0,   -1,  -1]
    # the relative permittivity in subintervals, no unit
    epsB = [80, 80]  # [left bath, right bath]
    epsC = [80,  4,   30,   30]
    # Boundary values of potential phi, [left, right], unit volt
    phiB = [0, -0.1]
    # Boundary value of c_i, [left, right], unit molar = mol/L
    ciB = [0.15, 0.15]
    Zs = [-1, 1]  # valence of the ionic species
    # radius of the cross section in the channel
    RC = 0.5
    # radius of the cross section in the bath
    RB = 5.5
    # parameter in NP equation
    chi1 = 1
    # parameter in P equation
    chi2 = 10.8971
    # # --------------------------------------------------------------
    omega = 0.9  # the relaxation parameter of Gummel iteration
    maxiter = 2000
    tol = 1.0e-6
    h = 0.0025
    # the mesh size
    t1 = perf_counter()
    BEM = BEM4PNP1D(Ls, lengthChannel, lengthBath, numIntervalInBath, DiffB, DiffC, QB, QC, ZB, ZC, epsB, epsC,
                    phiB, ciB, h, Zs, RB, RC, chi1, chi2, omega, maxiter, tol)
    BEM.doPreparation()
    # initial guess of the concentration
    cs0 = BEM.initialConcentration()
    # initial guess of the potential
    phi0 = BEM.initialPotential()
    # initial guess of the gradient of the potential
    Dphi0 = BEM.DPotential()
    phi, cs, Dphi, Dcs = BEM.iterative(cs0, phi0, Dphi0)
    for chi1, omega in zip([10,  20, 40], [0.4, 0.26, 0.18]):
        BEM = BEM4PNP1D(Ls, lengthChannel, lengthBath, numIntervalInBath, DiffB, DiffC, QB, QC, ZB, ZC, epsB, epsC,
                    phiB, ciB, h, Zs, RB, RC, chi1, chi2, omega, maxiter, tol)
        BEM.doPreparation()
        print('chi1 is {}'.format(chi1))
        phi, cs, Dphi, Dcs = BEM.iterative(cs, phi, Dphi)
    t2 = perf_counter()
    print('Total CPU time is {}'.format(t2 - t1))
    fileName = 'test4morePaper100_100.png'
    # BEM.plotFigure(phi, Dphi, cs, Dcs, fileName)
    BEM.plotFigure4Paper1(phi, cs)
    I = BEM.calculateCurrent(cs, Dcs, Dphi)
    print('Current passes through the channel is {}'.format(I[0][5][0]+I[1][5][0]))

