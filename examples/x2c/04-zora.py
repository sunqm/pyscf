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

mf_zora = sfzora(mol.RHF()).run()

mf_x2c = mol.RHF().x2c().run()

print(f'E(ZORA) = {mf_zora.e_tot}  E(X2C) = {mf_x2c.e_tot}')
