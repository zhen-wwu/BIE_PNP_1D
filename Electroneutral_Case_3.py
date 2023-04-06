"""
1D case: RK's new idea
Solve PNP equations with Dirichlet Boundary conditions over [-1,1]
-epsilon * div (grad (phi)) = chi2 *(Z1*c1+Z2*c2)
div (grad(c1) +chi1 * Z1* c1 *grad(phi)) = 0
div (grad(c2) +chi1 * Z2* c2* grad(phi)) = 0
phi(-1) = phiB[0], phi(1) = phiB[0]
ci(-1) = ciB[0], ci(1) = ciB[1]
By Zhen, August 10, 2022 at UM
"""
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
class BEM4PNP1D:

    def __init__(self, domain, phiB, ciB, Zs, N, chi1, chi2, epsilon, eta, omega, tol, maxIter, Di, ai, meshMethod):
        self.domain = domain  # the domain of the model
        self.phiB = phiB  # the boundary values of potential phi
        self.ciB = ciB    # the boundary condition of the concentration ci
        self.Zs = Zs      # valence of the ionic, i.e., [-1, 1]
        self.N = N        # number of subintervals over the whole domain
        self.chi1 = chi1  # a parameter in NP equation
        self.chi2 = chi2  # a parameter in P equation
        self.epsilon = epsilon  # the related permittivity
        self.eta = eta
        self.omega = omega  # parameter for Gummel iteration method
        self.tol = tol      # the stopping criterion of the Gummel iteration
        self.maxIter = maxIter
        self.NA = 6.02214076e23  # Avogadro number
        self.e = 1.60217663e-19  # elementary charge and measured in coulomb
        self.Di = Di
        self.ai = ai
        self.meshMethod = meshMethod

    def generate_mesh(self, domain, N, meshMethod):
        """
        Generate mesh grids over the whole domain, then save some data
        :param domain: [-1, 1]
        :param N: the number of subintervals over [-1, 1]
        :return:
        """
        a = domain[0]
        b = domain[1]
        if meshMethod == 'uniform':
            xs = np.linspace(a, b, N+1)
        elif meshMethod == 'cheb':
            xs0 = np.polynomial.chebyshev.chebpts1(N - 1)
            xs = np.insert(xs0, 0, -1, axis=0)
            xs = np.append(xs, [1])
        else:
            print('please give the mesh generation method')
        numInterval = len(xs) - 1
        hs = np.abs(xs[1:numInterval + 1] - xs[0:numInterval])  # the length of each subinterval
        return hs, xs

    def calculateVolumeIntegral(self, xs, hs):
        """
           Calculate the volume integrals over the domain
               IG[k, j] = int_{x^j}^{x^j+1} g(x, x^k)dx
               IGy[k, j] = int_{x^j}^{x^j+1} g_y(x, x^k)dx
           and function values
               G[k, j] = g(0.5*(x^j+x^j+1), x^k)
               Gy[k, j] = g_y(0.5*(x^j+x^j+1), x^k)
           where
               g(x, y) = -1/2 |x- y|
           is the green function of the Laplace equation
           :param xs
           :param hs
           :return IG, IGy, G, Gy
        """
        N = len(xs) - 1  # number of sub-intervals over the domain
        # define two numpy arrays to save the volume integrals
        IG = np.zeros((N + 1, N))
        IGy = np.zeros((N + 1, N))
        # define two numpy arrays to save the function values
        G = np.zeros((N + 1, N))
        Gy = np.zeros((N + 1, N))
        for k in range(N + 1):
            for j in range(N):
                G[k, j] = -0.5 * abs(0.5 * (xs[j] + xs[j + 1]) - xs[k])
                IG[k, j] = G[k, j] * hs[j]
                if k <= j:
                    IGy[k, j] = 0.5 * hs[j]
                    Gy[k, j] = 0.5
                else:
                    IGy[k, j] = -0.5 * hs[j]
                    Gy[k, j] = -0.5
        return IG, IGy, G, Gy

    def solvePoisson(self, cs, Zs, phiB):
        """
        Solve the poisson equation with Dirichlet boundary condition over [-1, 1]
        """
        chi2 = self.chi2
        epsilon = self.epsilon
        N = self.N
        xs = self.xs
        eta = self.eta
        hs = self.hs
        taub = chi2 / epsilon
        # Solve the potential gradient on the boundary
        sourceT = 0.0  # total concentration:  int_{-1}^1 z1 * c1(x)dx + int_{-1}^1 z2 * c2(x)dx + int_{-1}^1 zf * rho(x)dx
        sourceM = 0.0  # moment: int_{-1}^1 z1 * x*  c1(x)dx + int_{-1}^1 z2 * x * c2(x)dx + int_{-1}^1 zf * x * rho(x)dx
        for zi, ci in zip(Zs, cs):
            # For the calculation of the potential: ai*zi
            temp = 0.5 * zi * sum((ci[:-1] + ci[1:]) * hs[:])
            sourceT = sourceT + temp
            # for the calculation of the potential gradient
            temp = 0.5 * zi * sum((xs[:-1]*ci[:-1] + xs[1:]*ci[1:]) * hs[:])
            sourceM = sourceM + temp
        # Solve linear system of Poisson equation
        AP = np.array([[1, 0, eta,      0],
                       [0, 1, 0,     -eta],
                       [0, 0, 1,       -1],
                       [1, -1, -1,     -1]])
        bP = np.array([phiB[1],
                       phiB[0],
                       -taub * sourceT,
                        taub * sourceM])
        x = np.linalg.solve(AP, bP)
        # Define zero numpy array to save potential and its gradient
        phi = np.zeros(N + 1)
        Dphi = np.zeros(N + 1)
        phi[N] =  x[0]  # Dphi(1)
        phi[0] =  x[1]  # phi(-1)
        Dphi[N] = x[2]  # Dphi(1)
        Dphi[0] = x[3]  # Dphi(-1)

        # 2. Calculate potential and potential gradient at the interior points
        IG = self.IG  # F[k, j] = int_{x^j}^{x^j+1} g(x, x^k)dx
        IGy = self.IGy  # H[k, j] = int_{x^j}^{x^j+1} g_y(x, x^k)dx
        source = np.zeros(N + 1)  # chi2/epsilon * sum_i=1^2 zi * int_{-1}^1  g(x, y) * ci(x) dx + fix charge
        sourceG = np.zeros(N + 1)  # chi2/epsilon * sum_i=1^2 zi * int_{-1}^1  g_y(x, y) * ci(x) dx
        for k in range(N + 1):
            for zi, ci in zip(Zs, cs):
                # for potential
                temp = 0.5 * zi * sum((ci[:-1] + ci[1:]) * IG[k, :])
                source[k] = source[k] + temp
                # for derivative of potential
                temp = 0.5 * zi * sum((ci[:-1] + ci[1:]) * IGy[k, :])
                sourceG[k] = sourceG[k] + temp
            # Add the fixed point charge
            source[k] =  taub * source[k]
            sourceG[k] = taub * sourceG[k]

        # Calculate the potential values and  potential gradient at
        # the interior points, k = 1,2, ... , N-1 by using the BIE
        for k in range(1, N):
            phi[k] = (-0.5 * abs(xs[N]-xs[k]) * Dphi[N] - phi[N] * (-0.5)) \
                    -(-0.5 * abs(xs[0]-xs[k]) * Dphi[0] - phi[0] * ( 0.5)) + source[k]
            Dphi[k] = (0.5 * Dphi[N] - (-0.5) * Dphi[0]) + sourceG[k]
        Dphi[1:N] = (phi[2:N + 1] - phi[0:N - 1]) / (hs[0:N - 1] + hs[1:N])
        return phi, Dphi

    def solveNP(self, Dphi, ci0, zi, ai):
        """
        Solve the nernst planck equation, suppose the derivative of phi is given.
        This one does not use the fundamental theory of calculus
        """
        chi1 = self.chi1
        N = self.N
        xs = self.xs
        hs = self.hs
        # integral ci(x) phi'(x) over [-1,1]
        # source = 0.5 * sum((ci0[0:N] + ci0[1:N+1]) * hs[:])
        sourceM1  = 0.5 * sum((ci0[0:N]*Dphi[0:N] + ci0[1:N+1] * Dphi[1:N+1]) * hs[:])
        sourceM2 = 0.5 * sum((xs[0:N] * ci0[0:N] * Dphi[0:N] + xs[1:N+1] * ci0[1:N + 1] * Dphi[1:N + 1]) * hs[:])

        # Calculate the concentration and concentration gradient on the boundary
        ANP = np.array([[chi1 * zi * Dphi[0], 1, 0,                   0],
                        [0,                   0, chi1 * zi * Dphi[N], 1],
                        [-1,                  0, 1,                   0],
                        [1,                   0, 1,                   0]])
        bNP = np.array([0,
                        0,
                        -chi1 * zi * sourceM1,
                        ai - chi1 * zi * sourceM2])
        y = np.linalg.solve(ANP, bNP)
        ci = np.zeros(N+1)
        Dci = np.zeros(N + 1)
        ci[0] = y[0]   # ci(-1)
        Dci[0] = y[1]  # Dci(-1)
        ci[N] = y[2]   # ci(1)
        Dci[N] = y[3]  # Dci(1)
        sourcePhi = np.zeros(N + 1)
        sourceGPhi = np.zeros(N + 1)
        Gy = self.Gy  # Gy[k, j] = g_y(0.5 * (x ^ j + x ^ j + 1), x ^ k)
        G = self.G    # G[k, j] = g(0.5 * (x ^ j + x ^ j + 1), x ^ k)
        for k in range(N + 1):
            # for the concentration: int_{-1}^1 g(x, x^k) * (ci(x) * Dphi(x))' dx
            temp = (ci0[1] * Dphi[1] - ci[0] * Dphi[0]) * G[k, 0] +\
                sum((ci0[2:N] * Dphi[2:N] - ci0[1:N-1] * Dphi[1:N-1]) * G[k, 1:N-1]) + \
                   (ci[N] * Dphi[N] - ci0[N-1] * Dphi[N-1]) * G[k, N-1]
            sourcePhi[k] = sourcePhi[k] + temp
            # for derivative of concentration: int_{-1}^1 g_y(x, x^k) * (ci(x) * Dphi(x))' dx
            # temp = sum((ci0[1:] * Dphi[1:] - ci0[:-1] * Dphi[:-1]) * G[k, :])
            temp = (ci0[1] * Dphi[1] - ci[0] * Dphi[0]) * Gy[k, 0] +\
                sum((ci0[2:N] * Dphi[2:N] - ci0[1:N-1] * Dphi[1:N-1]) * Gy[k, 1:N-1]) + \
                   (ci[N] * Dphi[N] - ci0[N-1] * Dphi[N-1]) * Gy[k, N-1]
            sourceGPhi[k] = sourceGPhi[k] + temp

        # Calculate the concentration values  and  derivative of potential at the interior points, k = 1,2, ... , N-1
        # Calculate the gradient of the concentration by using the BIE
        for k in range(1, N):
            ci[k] = (-0.5 * abs(xs[N]-xs[k]) * Dci[N] - ci[N] * (-0.5)) \
                   -(-0.5 * abs(xs[0]-xs[k]) * Dci[0] - ci[0] * ( 0.5)) + chi1 * zi * sourcePhi[k]
            Dci[k] = (0.5 * Dci[N] - (-0.5) * Dci[0]) + chi1 * zi * sourceGPhi[k]
        # Dci[1:N] = (ci[2:N + 1] - ci[0:N - 1]) / (hs[0:N - 1] + hs[1:N])
        return ci, Dci

    def iterative(self):
        domain = self.domain
        N = self.N
        meshMethod = self.meshMethod
        # Generate the mesh data
        hs, xs = self.generate_mesh(domain, N, meshMethod)
        self.xs = xs
        self.hs = hs
        # calculate the volume integrals and function values for preparations
        IG, IGy, G, Gy = self.calculateVolumeIntegral(xs, hs)
        self.IG = IG
        self.IGy = IGy
        self.G = G
        self.Gy = Gy
        Zs = self.Zs
        ai = self.ai
        # -------------------------------
        # Do the Gummel iteration
        omega = self.omega
        Iter = 0
        resi = 1000.0
        sor_tol = self.tol
        sor_max_iter = self.maxIter
        # initial guess of the concentration
        cs0 = self.initialConcentration()
        # Dcs0 = [np.zeros(N+1), np.zeros(N+1)]
        # initial guess of the potential
        phi0 = self.initialPotential()
        # initial guess of the gradient of the potential
        Dphi0 = self.DPotential()
        # Boundary condition of the potential
        phiB = self.phiB
        print("\n     Start the iterative scheme")
        print("\n      Ite.      Error\n")
        print("        0      ")
        while resi >= sor_tol and Iter < sor_max_iter:
            phi, Dphi = self.solvePoisson(cs0, Zs, phiB)
            phi1 = omega * phi + (1 - omega) * phi0
            Dphi1 = omega * Dphi + (1 - omega) * Dphi0
            resiU = np.linalg.norm(Dphi - Dphi0)
            phi0 = phi1
            Dphi0 = Dphi1
            errorC_norm = []
            cs = []
            Dcs = []
            for zi, ci0,  a in zip(Zs,  cs0, ai):
                 # ci00 = ci0
                 # for i in range(40):
                 ci, Dci = self.solveNP(Dphi1,  ci0, zi, a)
                     # print(np.linalg.norm(ci - ci0))
                     # ci0 = omega *ci + (1- omega) * ci0
                 err = np.linalg.norm(ci - ci0)
                 # err = np.linalg.norm(Dci - Dci0)
                 errorC_norm.append(err)
                 Dcs.append(Dci)
                 cs.append(omega * ci + (1 - omega) * ci0)
            errorC = max(errorC_norm)
            resi = max(resiU, errorC)
            cs0 = cs
            Iter += 1
            print(('     %4d       %10.4e    ' % (Iter, resi)))
        return phi, cs, Dphi, Dcs

    def initialConcentration(self):
        xs = self.xs
        domain = self.domain
        ciB = self.ciB
        Zs = self.Zs
        ai = self.ai
        cs0 = []
        for i in range(len(Zs)):
            ciB[1] = ai[i]/2.0
            ciB[0] = ai[i]/2.0
            ci0 = [ciB[1] + (ciB[1] - ciB[0]) / (domain[1] - domain[0]) * (x - domain[1]) for x in xs]
            ci0 = np.array(ci0)
            cs0.append(ci0)
        return cs0

    def initialPotential(self):
        xs = self.xs
        phiB = self.phiB
        domain = self.domain
        phi0 = [phiB[1] + (phiB[1] - phiB[0]) / (domain[1] - domain[0]) * (x - domain[1]) for x in xs]
        phi0 = np.array(phi0)
        return phi0

    def DPotential(self):
        xs = self.xs
        phiB = self.phiB
        domain = self.domain
        Dphi0 = [(phiB[1] - phiB[0]) / (domain[1] - domain[0]) for x in xs]
        Dphi0 = np.array(Dphi0)
        return Dphi0

    def calculateCurrent(self, cs, Dcs, Dphi):
        Zs = self.Zs
        NA = self.NA
        e = self.e
        Di = self.Di
        I1 = -e*NA/1000 * Di * Zs[0] * (Dcs[0] + Zs[0] * chi1 * cs[0] * Dphi)
        I2 = -e*NA/1000 * Di * Zs[1] * (Dcs[1] + Zs[1] * chi1 * cs[1] * Dphi)
        current = [I1, I2]
        return current

    def printResult4Paper(self, cs, phi, fileName):
        xs = self.xs
        # plt.rcParams['text.usetex'] = True
        plt.figure(num=2)
        fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(25, 12))
        linewidth = 4
        ax1.tick_params(which='both', length=10, width=4, direction="in")
        ax1.plot(xs, phi,  linewidth=linewidth, color='b')
        ax3.tick_params(which='both', length=10, width=4, direction="in")
        ax3.plot(xs, cs[0],  linewidth=linewidth, color='b')
        # style = 'no'
        # if style == 'plain':
        #     ax1.ticklabel_format(style=style)
        #     ax3.ticklabel_format(style=style)
        fontsize = 50
        for axis in ['top', 'bottom', 'left', 'right']:
            ax1.spines[axis].set_linewidth(4)
            ax3.spines[axis].set_linewidth(4)
        ax1.xaxis.set_tick_params(labelsize=fontsize)
        ax1.yaxis.set_tick_params(labelsize=fontsize)
        # ax2.xaxis.set_tick_params(labelsize=24)
        # ax2.yaxis.set_tick_params(labelsize=24)
        ax3.xaxis.set_tick_params(labelsize=fontsize)
        ax3.yaxis.set_tick_params(labelsize=fontsize)
        # ax4.xaxis.set_tick_params(labelsize=24)
        # ax4.yaxis.set_tick_params(labelsize=24)
        ax1.set_xlabel(r'$x$', fontsize=fontsize)
        # ax2.set_xlabel('x', fontsize=24)
        ax3.set_xlabel(r'$x$', fontsize=fontsize)
        # ax4.set_xlabel('x', fontsize=24)
        # ax1.set_ylabel(r' $\phi(x)$', fontsize=40)
        ax1.set_title('(a) potential' + r' $\phi$', fontsize=fontsize, pad=20)
        ax3.set_title('(b) anion concentration' + r' $c_1$', fontsize=fontsize, pad=20)
        # ax2.set_ylabel(r' $\phi^\prime$', fontsize=30)
        # ax3.set_ylabel(r' $c_1(x)$', fontsize=40)
        plt.savefig(fileName, dpi=50)
        plt.tight_layout(pad=4)
        plt.show()  # --------------------------------------------------------------


    def printResult(self, cs, Dcs, phi, Dphi, current, fileName):
        xs = self.xs
        plt.figure(num=2)
        fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(48, 48))
        epsilon = self.epsilon
        chi1 = self.chi1
        chi2 = self.chi2
        title = r'$\chi_1=$ {}, $\chi_2=$ {}, $\epsilon=$ {}'.format(chi1, chi2, epsilon)
        linewidth = 8
        ax1.plot(xs, phi,  linewidth=linewidth)
        ax2.plot(xs, Dphi,   linewidth=linewidth)
        ax3.plot(xs, cs[0],  label='Anion', linewidth=linewidth, color='green')
        ax3.plot(xs, cs[1], label='Cation', linewidth=linewidth, color='red')
        ax4.plot(xs, Dcs[0], label='Anion', linewidth=linewidth, color='green')
        ax4.plot(xs, Dcs[1], label='Cation', linewidth=linewidth, color='red')
        ax5.plot(xs, current[0], label='Anion',  linewidth=linewidth, color='green')
        ax5.plot(xs, current[1], label='Cation', linewidth=linewidth, color='red')
        ax5.plot(xs, sum(current), label='Total', linewidth=linewidth, color='black')
        ax6.plot(xs, sum(current), label='Total', linewidth=linewidth, color='black')
        style = 'no'
        if style == 'plain':
            ax1.ticklabel_format(style=style)
            ax2.ticklabel_format(style=style)
            ax3.ticklabel_format(style=style)
            ax4.ticklabel_format(style=style)
            ax5.ticklabel_format(style=style)
            ax6.ticklabel_format(style=style)
        fontsize = 60
        ax1.yaxis.offsetText.set_fontsize(fontsize)
        ax2.yaxis.offsetText.set_fontsize(fontsize)
        ax3.yaxis.offsetText.set_fontsize(fontsize)
        ax4.yaxis.offsetText.set_fontsize(fontsize)
        ax5.yaxis.offsetText.set_fontsize(fontsize)
        ax6.yaxis.offsetText.set_fontsize(fontsize)
        ax1.xaxis.set_tick_params(labelsize=fontsize)
        ax1.yaxis.set_tick_params(labelsize=fontsize)
        ax2.xaxis.set_tick_params(labelsize=fontsize)
        ax2.yaxis.set_tick_params(labelsize=fontsize)
        ax3.xaxis.set_tick_params(labelsize=fontsize)
        ax3.yaxis.set_tick_params(labelsize=fontsize)
        ax4.xaxis.set_tick_params(labelsize=fontsize)
        ax4.yaxis.set_tick_params(labelsize=fontsize)
        ax5.xaxis.set_tick_params(labelsize=fontsize)
        ax5.yaxis.set_tick_params(labelsize=fontsize)
        ax6.xaxis.set_tick_params(labelsize=fontsize)
        ax6.yaxis.set_tick_params(labelsize=fontsize)
        ax1.set_xlabel('x', fontsize=40)
        ax2.set_xlabel('x', fontsize=40)
        ax3.set_xlabel('x', fontsize=40)
        ax4.set_xlabel('x', fontsize=40)
        ax5.set_xlabel('x', fontsize=40)
        ax6.set_xlabel('x', fontsize=40)
        ax1.set_ylabel('Potential ' + r' $\phi$', fontsize=60)
        ax2.set_ylabel(r' $\phi^\prime$', fontsize=60)
        ax3.set_ylabel('Concentration ' + r' $c_i$', fontsize=60)
        ax4.set_ylabel(r' $c_i^\prime$', fontsize=60)
        ax5.set_ylabel('Current ' + r' $I_i$', fontsize=60)
        ax6.set_ylabel('Current ', fontsize=60)
        # ax1.legend(fontsize=25)
        # ax2.legend(fontsize=30)
        ax3.legend(fontsize=60)
        ax4.legend(fontsize=60)
        ax5.legend(fontsize=60)
        ax6.legend(fontsize=60)
        plt.suptitle(title,fontsize =80)
        plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.4, hspace=0.5)
        plt.savefig(fileName, dpi=50)
        plt.show()  # --------------------------------------------------------------

    def plotIVcurve(self):
        epsilon = self.epsilon
        chi1 = self.chi1
        chi2 = self.chi2
        title = r'$\chi_1=$ {}, $\chi_2=$ {}, $\epsilon=$ {}'.format(chi1, chi2, epsilon)
        plt.figure(num=2)
        xs = [0, 0.2, 0.4, 0.6, 0.8, 1.0]
        ys = [0, 0.32, 0.63, 0.95, 1.26, 1.56]
        plt.plot(xs, ys, 'o-')
        plt.xlabel('Voltage (V)')
        plt.ylabel('Current (pA)')
        for x, y in zip(xs, ys):
            label = "{:.2f}".format(y)
            plt.annotate(label,  # this is the text
                         (x, y),  # this is the point to label
                         textcoords="offset points",  # how to position the text
                         xytext=(0, 10),  # distance from text to points (x,y)
                         ha='center')  # horizontal alignment can be left, right or center
        plt.suptitle(title, fontsize=20)
        plt.savefig('IV.png')
if __name__ == "__main__":
    # Domain of PNP model
    domain = [-1, 1]
    # Boundary values of potenrial phi, [left, right]
    phiB = [1, -1]
    # Boundary value of c_i, here, we set all ionic species have the same boundary value conditions [left, right]
    ciB = [1, 1]
    # valence of the ionic species
    Zs = [-1, 1]
    ai = [2, 2]
    epsilon = 1
    eta = 4.63e-5
    Di = 0.4e-3 # unit is nm^3/ps
    # chi1, the physical value is 40
    meshMethod = 'cheb'
    chi1 = 3.1

    # chi2, the physical value is 10.8971
    chi2 = 125.4
    tol = 1.0e-6
    maxIter = 10000
    # case 1.1: omega=0.1; 1.2: omega=0.08, 3.2: omega=0.01, 4.2: omega=0.1
    omega = 0.09
    # Test Poisson Nernst Planck equation
    N = 100
    fileName = 'Electroneutral_Case_3.png'
    t1 = perf_counter()
    BEM = BEM4PNP1D(domain, phiB, ciB, Zs, N, chi1, chi2, epsilon, eta, omega, tol, maxIter, Di, ai, meshMethod)
    phi, cs, Dphi, Dcs = BEM.iterative()
    current = BEM.calculateCurrent(cs, Dcs, Dphi)
    BEM.printResult4Paper(cs,  phi, fileName)
    t2 = perf_counter()
    print('Total CPU time is {}'.format(t2-t1))