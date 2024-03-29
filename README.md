# BIE_PNP_1D
An integral equation method is presented for the 1D steady-state Poisson-Nernst-Planck equations modeling ion transport through membrane channels. The differential equations are recast as integral equations using Green's 3rd identity yielding a fixed-point problem  for the electric potential gradient and ion concentrations. The integrals are discretized by a combination of midpoint and trapezoid rules and the resulting algebraic equations are solved by Gummel iteration. Numerical tests for electroneutral and non-electroneutral systems demonstrate the method's 2nd order accuracy and ability to resolve sharp boundary layers. The method is applied to a 1D model of the potassium ion channel with a fixed charge density that ensures cation selectivity. In these tests, the proposed integral equation method yields potential and concentration profiles in good agreement with published results. More details can be found at https://link.springer.com/article/10.1007/s10825-023-02092-y.

Authors: Zhen Chao (zhench@umich.edu), Weihua Geng (wgeng@smu.edu), Robert Krasny (krasny@umich.edu)

# Compile and run
After downloading and unzipping the current repository, navigate to the library directory where the following five Python codes are found.
Electroneutral_Case_1.py, Electroneutral_Case_2.py, Electroneutral_Case_3.py, Nonelectroneutral_Case_4.py, KcsA_BIE.py

# License
Copyright © 2019-2021, The Regents of the University of Michigan. Released under the MIT License.

# Support
This material is based upon work supported by the National Science Foundation under grants DMS-1819094/1819193 and DMS-2110767/2110869. Any opinions, findings, conclusions or recommendations expressed in this material are those of the author(s) and do not necessarily reflect the views of the National Science Foundation.
