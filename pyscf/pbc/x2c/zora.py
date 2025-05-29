#!/usr/bin/env python
# Copyright 2025 The PySCF Developers. All Rights Reserved.
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

'''
Zeroth Order Regular Approximation (ZORA)
'''


from functools import reduce
import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.x2c import x2c
from pyscf.pbc.scf import hf, ghf
from pyscf.pbc.x2c.sfx2c1e import PBCX2CHelper


def sfzora(mf):
    '''Spin-free ZORA
    '''
    if isinstance(mf, SFZORA_SCF):
        if mf.with_x2c is None:
            mf.with_x2c = SpinFreeZORAHelper(mf.mol)
            return mf
        elif not isinstance(mf.with_x2c, SpinFreeZORAHelper):
            raise NotImplementedError
        else:
            return mf

    return lib.set_class(SFZORA_SCF(mf), (SFZORA_SCF, mf.__class__))

class SFZORA_SCF(x2c._X2C_SCF):
    __name_mixin__ = 'sfZORA'

    _keys = {'with_x2c'}

    def __init__(self, mf):
        self.__dict__.update(mf.__dict__)
        self.with_x2c = SpinFreeZORAHelper(mf.mol)

    def undo_zora(self):
        obj = lib.view(self, lib.drop_class(self.__class__, SFZORA_SCF))
        del obj.with_x2c
        return obj

    def get_hcore(self, mol=None):
        if self.with_x2c:
            hcore = self.with_x2c.get_hcore(mol)
            if isinstance(self, ghf.GHF):
                hcore = scipy.linalg.block_diag(hcore, hcore)
            return hcore
        else:
            return super(x2c._X2C_SCF, self).get_hcore(mol)

class SpinFreeZORAHelper(PBCX2CHelper):
    approx = 'atom'

    def dump_flags(self, verbose=None):
        if self.approx:
            log = logger.new_logger(self, verbose)
            log.info('zora.approx = %s',    self.approx)
        return self

    def get_hcore(self, cell=None, kpts=None):
        from pyscf.pbc.df import df
        if cell is None: cell = self.cell
        if kpts is None:
            kpts_lst = np.zeros((1,3))
        else:
            kpts_lst = np.reshape(kpts, (-1,3))
        # By default, we use uncontracted cell.basis plus additional steep orbital for modified Dirac equation.
        xcell, contr_coeff = self.get_xmol(cell)
        c = lib.param.LIGHT_SPEED

        assert 'ATOM' in self.approx.upper()
        atom_slices = xcell.offset_nr_by_atom()
        nao = xcell.nao_nr()
        x = np.zeros((nao,nao))
        for ia in range(xcell.natm):
            ish0, ish1, p0, p1 = atom_slices[ia]
            shls_slice = (ish0, ish1, ish0, ish1)
            tloc = xcell.intor('int1e_kin', shls_slice=shls_slice)
            with xcell.with_rinv_at_nucleus(ia):
                z = -xcell.atom_charge(ia)
                w = z * xcell.intor('int1e_prinvp', shls_slice=shls_slice)
            x[p0:p1,p0:p1] = np.linalg.solve(tloc - .25/c**2 * w, tloc)

        with_df = df.DF(cell)
        if cell.pseudo:
            raise NotImplementedError
        else:
            v = np.asarray(with_df.get_nuc(kpts_lst))

        t = xcell.pbc_intor('int1e_kin', 1, lib.HERMITIAN, kpts_lst)
        if self.xuncontract and contr_coeff is not None:
            t = [contr_coeff.T.dot(tk).dot(x).dot(contr_coeff) for tk in t]

        h_zora = [vk+tk for vk, tk in zip(v, t)]

        if kpts is None or np.shape(kpts) == (3,):
            h_zora = h_zora[0]
        return np.asarray(h_zora)
