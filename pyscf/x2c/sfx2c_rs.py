import numpy as np
import scipy.linalg
from pyscf import lib
from pyscf.lib import logger
from pyscf import gto
from pyscf.scf import hf, _vhf, jk
from pyscf.x2c import sfx2c1e, x2c

FULL_HCORE = True
# When RS Hcore is set, Vnuc not

class RHF(hf.RHF):

    coulomb_rs_omega = 1.
    # SS is highly local. (SS|SS) can be safely ignored.
    # For SR Coulomb, the J and K for SSSS are mostly self-interactions and
    # largely canclled out.
    with_ssss = False

    def build(self, mol=None):
        if mol is None: mol = self.mol
        # X is evaluated in j-adapted spinors and then transformed to spherical
        # GTO representation
        with_x2c = sfx2c1e.SpinFreeX2CHelper(mol)
        c = lib.param.LIGHT_SPEED
        xmol, contr_coeff = with_x2c.get_xmol(mol)
        self._xmol = xmol
        omega = self.coulomb_rs_omega

        s = xmol.intor_symmetric('int1e_ovlp')
        t = xmol.intor_symmetric('int1e_kin')
        v = xmol.intor_symmetric('int1e_nuc')
        w = xmol.intor_symmetric('int1e_pnucp')
        if FULL_HCORE:
            self._X = X = x2c._x2c1e_xmatrix(t, v, w, s, c)
        else:
            nuc_model_backup = xmol._atm[:,2].copy()
            # Set gaussian nuc-model, which enables the LR Coulomb for nuclear attraction
            xmol._atm[:,2] = 2
            xmol._env[xmol._atm[:,3]] = omega**2
            vnuc_lr = xmol.intor_symmetric('int1e_nuc')
            wnuc_lr = xmol.intor_symmetric('int1e_pnucp')
            v_sr = v - vnuc_lr
            w_sr = w - wnuc_lr
            xmol._atm[:,2] = nuc_model_backup
            self._X = X = x2c._x2c1e_xmatrix(t, v_sr, w_sr, s, c)

        snesc = s + .5/c**2 * X.conj().T.dot(t).dot(X)
        self._ovlp = snesc
        R = x2c._get_r(s, snesc)
        self._R = R = R.dot(contr_coeff)

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
            opt_llll = _vhf._VHFOpt(
                xmol, 'int2e', 'CVHFnrs8_prescreen', 'CVHFnr_int2e_q_cond',
                'CVHFnr_dm_cond', self.direct_scf_tol*1e-2)
            opt_ssll = _vhf._VHFOpt(
                xmol, 'int2e', 'CVHFnrs8_prescreen', 'CVHFnr_int2e_q_cond',
                'CVHFnr_dm_cond', self.direct_scf_tol*1e-2)
            opt_ssll._intor = 'int2e_pp1_sph'
            opt_ssll._cintopt = _vhf.make_cintopt(xmol._atm, xmol._bas, xmol._env,
                                                  opt_ssll._intor)
            sr_opt = opt_llll, opt_ssll
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
        opt_llll, opt_ssll = sr_opt

        logger.debug(self, 'Computing DHF SR get_jk')
        c1 = .5 / lib.param.LIGHT_SPEED
        R = self._R
        X = self._X
        dm_LL = R.dot(dm).dot(R.conj().T)
        dm_LS = dm_LL.dot(X.conj().T)
        dm_SS = X.dot(dm_LS)
        with_j = with_k = True
        assert not self.with_ssss
        xmol = self._xmol
        with xmol.with_range_coulomb(-omega):
            vjLL, vkLL = hf.get_jk(xmol, dm_LL, hermi, opt_llll, with_j, with_k)
            dms = [dm_SS, dm_LL, dm_LS]
            scripts = ['ijkl,ji->s2kl',
                       'ijkl,lk->s2ij',
                       'ijkl,li->s1kj',
                      ]
            vs = jk.get_jk(xmol, dms, scripts, 'int2e_pp1_sph', aosym='s4',
                           vhfopt=opt_ssll)
            lib.hermi_triu(vs[0], inplace=True)
            lib.hermi_triu(vs[1], inplace=True)

        nao = X.shape[0]
        vhf_4c = np.zeros((nao*2, nao*2))
        vhf_LL = vjLL - vkLL * .5 + vs[0] * c1**2
        vhf_LS = (-.5 * c1**2 * vs[2]).dot(X)
        vhf_SS = X.conj().T.dot(vs[1] * c1**2).dot(X)
        vhf_2c = vhf_LL + vhf_LS + vhf_LS.conj().T + vhf_SS
        vhf_2c = R.conj().T.dot(vhf_2c).dot(R)

        # GHF LR JK matrices for complex density matrices
        logger.debug(self, 'Computing RHF LR get_jk')
        vj, vk = hf.get_jk(mol, dm, hermi, lr_opt, with_j, with_k, omega=omega)

        vhf_2c += vj - vk * .5
        if vhf_last is not None:
            vhf_2c += vhf_last
        return vhf_2c

if __name__ == '__main__':
    basis = 'unc-ccpvdzdk'
    lib.param.LIGHT_SPEED = 3
    mol1 = gto.M(atom='He 0 0 0; He 0 0 1', basis=basis)
    #mol1.verbose = 4
    mf = mol1.DHF().run()
    mf = RHF(mol1).run()
    mf = mol1.RHF().sfx2c1e().run()
