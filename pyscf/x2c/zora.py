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

Four-component equation:
    [V T       ] [A] = [S 0     ] [A] E
    [T W/4c^2-T] [B]   [0 T/2c^2] [B]
Let B = XA, two equations are derived from the four-component equation
    VA + TXA = SAE
    TA + (W/4c^2-T)XA = T/2c^2 XAE

Let E=0, X_ZORA is obtained from the second equation
    T + (W/4c^2-T) X_ZORA = 0
Substituting X_ZORA into the first equation results in the H_ZORA in matrix form
    H_ZORA = V + TX_ZORA
'''

import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf.scf import hf, ghf
from pyscf.x2c import x2c

class SpinFreeZORAHelper(x2c.X2CHelperBase):
    approx = '' # can be set to "atom" for atomic approximation

    def dump_flags(self, verbose=None):
        if self.approx:
            log = logger.new_logger(self, verbose)
            log.info('zora.approx = %s',    self.approx)
        return self

    def get_hcore(self, mol=None):
        '''1-component X2c Foldy-Wouthuysen (FW Hamiltonian  (spin-free part only)
        '''
        if mol is None: mol = self.mol
        if mol.has_ecp():
            raise NotImplementedError

        xmol, contr_coeff = self.get_xmol(mol)
        c = lib.param.LIGHT_SPEED
        t = xmol.intor_symmetric('int1e_kin')

        if 'ATOM' in self.approx.upper():
            atom_slices = xmol.offset_nr_by_atom()
            nao = xmol.nao_nr()
            x = np.zeros((nao,nao))
            for ia in range(xmol.natm):
                ish0, ish1, p0, p1 = atom_slices[ia]
                shls_slice = (ish0, ish1, ish0, ish1)
                tloc = t[p0:p1,p0:p1]
                with xmol.with_rinv_at_nucleus(ia):
                    z = -xmol.atom_charge(ia)
                    w = z * xmol.intor('int1e_prinvp', shls_slice=shls_slice)
                x[p0:p1,p0:p1] = np.linalg.solve(tloc - .25/c**2 * w, tloc)
        else:
            w = xmol.intor_symmetric('int1e_pnucp')
            x = np.linalg.solve(t - .25/c**2 * w, t)

        v = mol.intor_symmetric('int1e_nuc')
        if self.xuncontract and contr_coeff is not None:
            t = contr_coeff.T.dot(t).dot(x).dot(contr_coeff)
        h_zora = v + t
        return h_zora

def sfzora(mf):
    '''Spin-free ZORA
    '''
    assert isinstance(mf, hf.SCF)
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

    def _transfer_attrs_(self, dst):
        if self.with_x2c and not hasattr(dst, 'with_x2c'):
            logger.warn(self, 'Destination object of to_hf/to_ks method is not '
                        'an ZORA object. Convert dst to ZORA object.')
            dst = sfzora(dst)
        return hf.SCF._transfer_attrs_(self, dst)

    def to_gpu(self):
        raise NotImplementedError
