import pyscf
mol = pyscf.M(atom='H 0 0 0; H 0 0 1', basis='unc-ccpvdz-dk', verbose=4)

print('Dirac-Coulomb')
method = mol.DHF()
method.kernel()

method.local_dcb = 'A1N'
method.kernel()

method.local_dcb = 'A2N'
method.kernel()

print('with Gaunt')
method.with_gaunt = True
method.local_dcb = None
method.kernel()

method.local_dcb = 'A1N'
method.kernel()

method.local_dcb = 'A2N'
method.kernel()

print('with Breit')
method.with_gaunt = False
method.with_breit = True
method.local_dcb = None
method.kernel()

method.local_dcb = 'A1N'
method.kernel()

method.local_dcb = 'A2N'
method.kernel()
