import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.scf import dhf, ghf, hf, _vhf
from pyscf.x2c import x2c

FULL_HCORE = True
# When RS Hcore is set, Vnuc not

class GHF(ghf.GHF):

    coulomb_rs_omega = 1.
    # SS is highly local. (SS|SS) can be safely ignored.
    # For SR Coulomb, the J and K for SSSS are mostly self-interactions and
    # largely canclled out.
    with_ssss = False

    def build(self, mol=None):
        if mol is None: mol = self.mol
        # X is evaluated in j-adapted spinors and then transformed to spherical
        # GTO representation
        with_x2c = x2c.SpinorX2CHelper(mol)
        c = lib.param.LIGHT_SPEED
        xmol, contr_coeff_nr = with_x2c.get_xmol(mol)
        self._xmol = xmol
        omega = self.coulomb_rs_omega

        s = xmol.intor_symmetric('int1e_ovlp_spinor')
        t = xmol.intor_symmetric('int1e_spsp_spinor') * .5
        v = xmol.intor_symmetric('int1e_nuc_spinor')
        w = xmol.intor_symmetric('int1e_spnucsp_spinor')
        if FULL_HCORE:
            self._X = X = x2c._x2c1e_xmatrix(t, v, w, s, c)
        else:
            nuc_model_backup = xmol._atm[:,2].copy()
            # Set gaussian nuc-model, which enables the LR Coulomb for nuclear attraction
            xmol._atm[:,2] = 2
            xmol._env[xmol._atm[:,3]] = omega**2
            vnuc_lr = xmol.intor_symmetric('int1e_nuc_spinor')
            wnuc_lr = xmol.intor_symmetric('int1e_spnucsp_spinor')
            v_sr = v - vnuc_lr
            w_sr = w - wnuc_lr
            xmol._atm[:,2] = nuc_model_backup
            self._X = X = x2c._x2c1e_xmatrix(t, v_sr, w_sr, s, c)

        snesc = s + .5/c**2 * X.conj().T.dot(t).dot(X)
        self._ovlp = snesc
        R = x2c._get_r(s, snesc)

        nprim, nc = contr_coeff_nr.shape
        contr_coeff = np.zeros((nprim*2,nc*2))
        contr_coeff[0::2,0::2] = contr_coeff_nr
        contr_coeff[1::2,1::2] = contr_coeff_nr
        R = R.dot(contr_coeff)

        # GHF is represented in spherical GTOs
        ca, cb = mol.sph2spinor_coeff()
        u = np.vstack([ca, cb])
        self._R = R = np.linalg.solve(u.T, R.T).T

        if FULL_HCORE:
            tx = t.dot(X)
            h1 = v + tx + tx.conj().T
            h1 += .25/c**2 * X.conj().T.dot(w).dot(X) - X.conj().T.dot(tx)
            self._hcore = R.conj().T.dot(h1).dot(R)
        else:
            tx = t.dot(X)
            h1 = v_sr + tx + tx.conj().T
            h1 += .25/c**2 * X.conj().T.dot(w_sr).dot(X) - X.conj().T.dot(tx)
            h1 = R.conj().T.dot(h1).dot(R)

            mol._atm[:,2] = 2
            mol._env[mol._atm[:,3]] = omega**2
            vnuc_lr = mol.intor_symmetric('int1e_nuc')
            h1 += scipy.linalg.block_diag(vnuc_lr, vnuc_lr)
            mol._atm[:,2] = nuc_model_backup
            self._hcore = h1

        with xmol.with_range_coulomb(-omega):
            opt_llll = dhf._VHFOpt(
                xmol, 'int2e_spinor', 'CVHFrkbllll_prescreen',
                'CVHFrkb_q_cond', 'CVHFrkb_dm_cond',
                direct_scf_tol=self.direct_scf_tol)
            opt_llll._this.r_vkscreen = _vhf._fpointer('CVHFrkbllll_vkscreen')

            c1 = .5 / c
            opt_ssss = dhf._VHFOpt(
                xmol, 'int2e_spsp1spsp2_spinor', 'CVHFrkbllll_prescreen',
                'CVHFrkb_q_cond', 'CVHFrkb_dm_cond',
                direct_scf_tol=self.direct_scf_tol/c1**4)
            opt_ssss.direct_scf_tol = self.direct_scf_tol
            opt_ssss.q_cond *= c1**2
            opt_ssss._this.r_vkscreen = _vhf._fpointer('CVHFrkbllll_vkscreen')

            opt_ssll = dhf._VHFOpt(
                xmol, 'int2e_spsp1_spinor', 'CVHFrkbssll_prescreen',
                dmcondname='CVHFrkbssll_dm_cond',
                direct_scf_tol=self.direct_scf_tol)
            opt_ssll.q_cond = np.array([opt_llll.q_cond, opt_ssss.q_cond])
            opt_ssll._this.r_vkscreen = _vhf._fpointer('CVHFrkbssll_vkscreen')
            sr_opt = (opt_llll, opt_ssll, opt_ssss)
            logger.debug(self, 'SR DHF q_cond initialization for omega = %g', -omega)

        with mol.with_range_coulomb(omega):
            lr_opt = _vhf._VHFOpt(
                mol, 'int2e', 'CVHFnrs8_prescreen', 'CVHFnr_int2e_q_cond',
                'CVHFnr_dm_cond', self.direct_scf_tol)
            logger.debug(self, 'LR RHF q_cond initialization for omega = %g', omega)

        self._opt[omega] = (sr_opt, lr_opt)
        return self

    def get_hcore(self, mol=None):
        return self._hcore

    def get_veff(self, mol, dm, dm_last=None, vhf_last=None, hermi=1):
        if dm_last is not None:
            dm = dm - dm_last

        omega = self.coulomb_rs_omega
        sr_opt, lr_opt = self._opt[omega]
        opt_llll, opt_ssll, opt_ssss = sr_opt

        logger.debug(self, 'Computing DHF SR get_jk')
        c1 = .5 / lib.param.LIGHT_SPEED
        R = self._R
        X = self._X
        dm_LL = R.dot(dm).dot(R.conj().T)
        dm_LS = dm_LL.dot(X.conj().T)
        dm_SS = X.dot(dm_LS)
        dm_4c = np.block([[dm_LL, dm_LS],
                          [dm_LS.conj().T, dm_SS]])
        if self.with_ssss:
            coulomb_level = 'SSSS'
        else:
            coulomb_level = 'SSLL'
        xmol = self._xmol
        with xmol.with_range_coulomb(-omega):
            vj, vk = dhf.get_jk_coulomb(
                xmol, dm_4c, hermi, coulomb_level, opt_llll, opt_ssll, opt_ssss, -omega)

        n2c = X.shape[0]
        vhf_4c = vj  - vk
        vhf_LL = vhf_4c[:n2c,:n2c]
        vhf_LS = vhf_4c[:n2c,n2c:].dot(X)
        vhf_SS = X.conj().T.dot(vhf_4c[n2c:,n2c:]).dot(X)
        vhf_2c = vhf_LL + vhf_LS + vhf_LS.conj().T + vhf_SS
        vhf_2c = R.conj().T.dot(vhf_2c).dot(R)

        # GHF LR JK matrices for complex density matrices
        logger.debug(self, 'Computing RHF LR get_jk')
        def jkbuild(mol, dm, hermi, with_j, with_k, omega=None):
            return hf.get_jk(mol, dm, hermi, lr_opt, with_j, with_k, omega=omega)
        vj, vk = ghf.get_jk(mol, dm, hermi, jkbuild=jkbuild, omega=omega)

        vhf_2c += vj - vk
        if vhf_last is not None:
            vhf_2c += vhf_last
        return vhf_2c

if __name__ == '__main__':
    basis = 'unc-ccpvdzdk'
    lib.param.LIGHT_SPEED = 3
    mol1 = gto.M(atom='He 0 0 0; He 0 0 1', basis=basis)
    #mol1.verbose = 4
    mf = mol1.DHF().run()
    mf = GHF(mol1).run()
    #mf = x2c.UHF(mol1).run()
    #mf = mol1.GHF().x2c()
