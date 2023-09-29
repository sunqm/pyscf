#!/usr/bin/env python
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

import numpy as np
from pyscf import lib
from pyscf.scf import _vhf
from pyscf.lib import logger
from pyscf.lrdf import lrdf
from pyscf.grad import rhf as rhf_grad
from pyscf.df.grad import rhf as df_rhf_grad


class Gradients(rhf_grad.Gradients):
    def __init__(self, mf):
        assert isinstance(mf.with_df, lrdf.LRDensityFitting)
        # Whether to include the response of DF auxiliary basis when computing
        # nuclear gradients of J/K matrices
        self.auxbasis_response = False
        rhf_grad.Gradients.__init__(self, mf)

    def get_jk(self, mol=None, dm=None, hermi=0, omega=None):
        if omega is not None:
            return rhf_grad.Gradients.get_jk(self, mol, dm, hermi, omega)

        lrdf_obj = self.base.with_df
        omega = lrdf_obj.omega
        # TODO: initialize q_cond with CVHFgrad_jk_direct_scf
        #vhfopt = lrdf._VHFOpt(mol, 'int2e_ip1',
        #                      prescreen='CVHFgrad_jk_prescreen', omega=omega)
        vhfopt = lrdf._VHFOpt(mol, 'int2e_ip1', omega=omega)
        vhfopt._this.q_cond = lrdf_obj._vhfopt._this.q_cond
        vhfopt._this.dm_cond = lrdf_obj._vhfopt._this.dm_cond

        with mol.with_short_range_coulomb(omega):
            intor = mol._add_suffix('int2e_ip1')
            vj, vk = _vhf.direct_mapdm(intor,  # (nabla i,j|k,l)
                                       's2kl', # ip1_sph has k>=l,
                                       ('lk->s1ij', 'jk->s1il'),
                                       dm, 3, # xyz, 3 components
                                       mol._atm, mol._bas, mol._env, vhfopt=vhfopt,
                                       optimize_sr=True)

        with lrdf_obj.range_coulomb(omega):
            with lib.temporary_env(lrdf_obj, auxmol=lrdf_obj.lr_auxmol):
                vj1, vk1 = df_rhf_grad.get_jk(self, mol, dm, hermi,
                                              decompose_j2c='ED',
                                              lindep=lrdf_obj.lr_thresh)
        vj = vj1 - np.asarray(vj)
        vk = vk1 - np.asarray(vk)
        if self.auxbasis_response:
            vj = lib.tag_array(vj, aux=vj1.aux)
            vk = lib.tag_array(vk, aux=vk1.aux)
        return vj, vk

    def get_j(self, mol=None, dm=None, hermi=0, omega=None):
        raise NotImplementedError

    def extra_force(self, atom_id, envs):
        if self.auxbasis_response:
            return envs['vhf'].aux[atom_id]
        else:
            return 0

    get_veff = df_rhf_grad.Gradients.get_veff
