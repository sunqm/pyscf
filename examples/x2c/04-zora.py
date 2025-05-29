'''
Approximate ZORA using X2C matrix method
'''

import pyscf
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
    'unc-ano', gto.etbs([(0, 8, 1e7, 2.5),   # s-function
                         (1, 5, 5e4, 2.5),   # p-function
                         (2, 2, 1e3, 2.5)]))
mf.run()

mf_x2c = mol.RHF().x2c().run()

print(f'E(ZORA) = {mf_zora.e_tot}  E(X2C) = {mf_x2c.e_tot}')
