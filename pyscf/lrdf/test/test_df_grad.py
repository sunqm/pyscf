# Copyright 2023 The PySCF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Author: Qiming Sun <osirpt.sun@gmail.com>
#

import unittest
from pyscf import lib
from pyscf import gto
from pyscf.lrdf import lrdf
from pyscf.lrdf.grad import rhf as rhf_grad

def setUpModule():
    global mol
    mol = gto.Mole()
    mol.build(
        verbose = 6,
        atom = '''O     0    0.       0.
                  1     0    -0.757   0.587
                  1     0    0.757    0.587''',
        basis = '6-31g',
        output = '/dev/null'
    )

def tearDownModule():
    global mol
    del mol


class KnownValues(unittest.TestCase):
    def test_rhf_grad(self):
        mf = mol.RHF().density_fit()
        mf.with_df = lrdf.LRDF(mol)
        mf.with_df.omega = 0.2
        mf.with_df.lr_thresh = 1e-4
        mf.run()
        g1 = rhf_grad.Gradients(mf).kernel()
        ref = mol.RHF().run().Gradients().kernel()
        self.assertAlmostEqual(abs(g1-ref).max(), 0, 5)

if __name__ == "__main__":
    print("Full Tests for lrdf.grad")
    unittest.main()
