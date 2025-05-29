'''
Approximate ZORA using X2C matrix method
'''

import pyscf
from pyscf import gto
from pyscf.x2c.zora import sfzora

mol = pyscf.M(
    atom='''O  0  0.     0
            H  0  -0.757 0.587
            H  0  0.757  0.587''',
    basis='ccpvdz-dk',
)

mf_zora = sfzora(mol.RHF())
# ZORA Hamiltonian is constructed in its matrix representation, derived from the
# X2C matrix formuliasm. A larger basis set is required to expand ZORA operator
# in matrix form. In the CBS limit, the matrix-ZORA is equivalent to the ZORA
# Hamiltonian in operator form.
mf_zora.with_x2c.basis = (
    'unc-ccpvdz-dk', gto.etbs([(0, 10, 1e4, 2.),   # s-function
                               (1, 5 , 2e1, 2.),   # p-function
                              ]))
mf_zora.run()

mf_x2c = mol.RHF().x2c()
mf_x2c.with_x2c.basis = (
    'unc-ccpvdz-dk', gto.etbs([(0, 10, 1e4, 2.),   # s-function
                               (1, 5 , 2e1, 2.),   # p-function
                              ]))
mf_x2c.run()

print(f'E(ZORA) = {mf_zora.e_tot}  E(X2C) = {mf_x2c.e_tot}')
