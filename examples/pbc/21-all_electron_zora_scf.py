#!/usr/bin/env python

'''
Atomic ZORA Hartree-Fock/DFT

See also examples/x2c/04-zora.py
'''

import pyscf
from pyscf import gto
from pyscf.pbc.x2c.zora import sfzora

cell = pyscf.M(
    atom='''
C 0.000000000000   0.000000000000   0.000000000000
C 1.685068664391   1.685068664391   1.685068664391
''',
    basis='6-31g',
    a='''
0.000000000, 3.370137329, 3.370137329
3.370137329, 0.000000000, 3.370137329
3.370137329, 3.370137329, 0.000000000''',
    unit = 'B',
    verbose = 5)

# Use a larger basis set to expand ZORA operator in matrix form
xbasis = (
    'unc-ano', gto.etbs([(0, 8, 1e7, 2.5),   # s-function
                         (1, 5, 5e4, 2.5),   # p-function
                         (2, 2, 1e3, 2.5)]))

mf = cell.RKS(xc='pbe').density_fit()
mf = sfzora(mf)
mf.with_x2c.basis = xbasis
mf.run()
mf.kernel()

nk = [2,2,2]  # 2 k-poins for each axis, 2^3=8 kpts in total
kpts = cell.make_kpts(nk)

#
# Spin-free ZORA HF/DFT for the scalar relativistic effects
#
mf = cell.KRHF(kpts=kpts).density_fit()
mf = sfzora(mf)
mf.with_x2c.basis = xbasis
mf.run()
mf.kernel()

mf = cell.KRKS(xc='pbe', kpts=kpts).density_fit()
mf = sfzora(mf)
mf.with_x2c.basis = xbasis
mf.run()
mf.kernel()
