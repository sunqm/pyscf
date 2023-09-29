import unittest
import numpy as np
from pyscf import lib
from pyscf import gto
from pyscf.scf import hf
from pyscf.lrdf import lrdf


class KnownValues(unittest.TestCase):
    def test_get_jk(self):
        mol = gto.M(atom = [["O" , (0. , 0.     , 0.)],
                            [1   , (0. , -0.757 , 0.587)],
                            [1   , (0. , 0.757  , 0.587)]],
                    basis = '6-31g')
        nao = mol.nao
        dm = np.random.seed(1)
        dm = np.random.rand(nao,nao)
        dm = dm.T.dot(dm)
        jref, kref = hf.get_jk(mol, dm)
        with_df = lrdf.LRDF(mol)
        with_df.lr_auxbasis = [[0, [1., 1]], [1, [1., 1]], [2, [1., 1]]]
        with_df.omega = 0.1
        with_df.lr_thresh = 1e-5
        vj, vk = with_df.get_jk(dm)
        self.assertAlmostEqual(abs(jref - vj).max(), 0, 5)
        self.assertAlmostEqual(abs(kref - vk).max(), 0, 5)

        with_df = lrdf.LRDF(mol)
        with_df.lr_thresh = 1e-6
        #with_df.grids = (5, 26)
        with_df.grids = (5, 100)
        with_df.build()
        vj, vk = with_df.get_jk(dm)
        self.assertAlmostEqual(abs(jref - vj).max(), 0, 5)
        self.assertAlmostEqual(abs(kref - vk).max(), 0, 5)

    def test_get_jk_sr(self):
        mol = gto.M(atom='''
C 0.0000 0.0000 0.0000
C 1.4468 0.0000 0.0000
C 3.2878 -0.0000 1.6664
C 3.7825 0.0002 3.0260
C 5.3079 0.0002 3.0260
''',
basis='cc-pvdz')
        with_df = lrdf.LRDF(mol)
        with_df.omega = 0.15
        with_df.lr_auxbasis = {'default': [[0, [1., 1]]]}
        with_df.build()

        np.random.seed(2)
        nao = mol.nao
        dm = np.eye(nao) + np.random.rand(nao,nao) * .1
        vj, vk = with_df._get_jk_sr(dm[None])
        with mol.with_range_coulomb(-0.15):
            jref, kref = hf.get_jk(mol, dm)
        self.assertAlmostEqual(abs(jref-vj[0]).max(), 0, 11)
        self.assertAlmostEqual(abs(kref-vk[0]).max(), 0, 11)

if __name__ == "__main__":
    print('Full Tests for LRDF')
    unittest.main()
