#!/usr/bin/env python

'''
Range separation JK builder
'''

import ctypes
import tempfile
import numpy as np
import scipy.linalg
from pyscf import gto
from pyscf import lib
from pyscf.lib import logger
from pyscf.df import df, df_jk, outcore, addons
from pyscf.gto import ft_ao
from pyscf.dft.gen_grid import LEBEDEV_NGRID, libdft
from pyscf.gto.moleintor import make_cintopt
from pyscf.pbc.df.incore import libpbc
from pyscf.scf._vhf import libcvhf, _fpointer

MIN_CUTOFF = 1e-44
AUXBASIS = {
    #'H': [[0, [1., 1]]],
    'default': [[0, [1., 1]], [1, [1., 1]], [2, [1., 1]]]
}

class _CVHFOpt(ctypes.Structure):
    _fields_ = [('nbas', ctypes.c_int),
                ('ngrids', ctypes.c_int),
                ('log_cutoff', ctypes.c_double),
                ('logq_cond', ctypes.c_void_p),
                ('dm_cond', ctypes.c_void_p),
                ('fprescreen', ctypes.c_void_p),
                ('r_vkscreen', ctypes.c_void_p)]

class LRDensityFitting(df.DF):

    omega = 0.1
    grids = None
    direct_scf_tol = 1e-12
    lr_thresh = 1e-4
    lr_auxbasis = AUXBASIS
    lr_dfj = True

    def __init__(self, mol, auxbasis=None):
        self._intor = 'int2e'
        self._cintopt = None
        self.q_cond = None
        self.lr_auxmol = None
        self.wcoulG = None
        self.Gv = None
        self._last_vs = (0, 0, 0)
        df.DF.__init__(self, mol, auxbasis)

    def reset(self, mol=None):
        self.q_cond = None
        self._cintopt = None
        df.DF.reset(self, mol)

    def dump_flags(self, verbose=None):
        log = logger.new_logger(self, verbose)
        log.info('******** %s ********', self.__class__)
        log.info('direct_scf_tol = %s', self.direct_scf_tol)
        log.info('omega = %s', self.omega)
        log.info('lr_thresh = %s', self.lr_thresh)
        if self.grids is not None:
            log.info('grids = %s', self.grids)
        elif self.lr_auxmol is None:
            log.info('lr_auxbasis = %s', self.lr_auxbasis)
        else:
            log.info('lr_auxbasis = lr_auxmol.basis = %s', self.lr_auxmol.basis)
        if self.lr_dfj:
            if self.auxmol is None:
                log.info('auxbasis = %s', self.auxbasis)
            else:
                log.info('auxbasis = auxmol.basis = %s', self.auxmol.basis)
        return self

    def build(self):
        cpu0 = logger.process_clock(), logger.perf_counter()
        log = logger.Logger(self.stdout, self.verbose)
        self.dump_flags()

        mol = self.mol
        nbas = mol.nbas
        self.q_cond = np.empty((6,nbas,nbas), dtype=np.float32)
        ao_loc = mol.ao_loc
        omega = self.omega
        assert omega > 0

        with mol.with_short_range_coulomb(omega):
            self._cintopt = make_cintopt(
                mol._atm, mol._bas, mol._env, self._intor)

            with mol.with_integral_screen(self.direct_scf_tol**2):
                libpbc.CVHFsetnr_sr_direct_scf(
                    libpbc.int2e_sph, self._cintopt,
                    self.q_cond.ctypes.data_as(ctypes.c_void_p),
                    ao_loc.ctypes.data_as(ctypes.c_void_p),
                    mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                    mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                    mol._env.ctypes.data_as(ctypes.c_void_p))
        cpu0 = log.timer('initializing q_cond', *cpu0)

        if self.lr_auxmol is None:
            self.lr_auxmol = addons.make_auxmol(mol, self.lr_auxbasis)
            self.lr_auxmol.omega = omega

        if self.grids is None:
            if self._cderi_to_save is None:
                self._cderi_to_save = tempfile.NamedTemporaryFile(dir=lib.param.TMPDIR)
            self._cderi = cderi = self._cderi_to_save
            mol = self.mol
            max_memory = self.max_memory - lib.current_memory()[0]
            with mol.with_long_range_coulomb(self.omega):
                outcore.cholesky_eri_b(
                    mol, cderi, auxmol=self.lr_auxmol, max_memory=max_memory,
                    decompose_j2c='eig', lindep=self.lr_thresh, verbose=log)
        else:
            n_rad, n_ang = self.grids
            Gv, weights = _non_uniform_Gv(n_rad, n_ang, omega, self.lr_thresh)
            self.wcoulG = weights * _get_coulG(Gv, omega)
            self.Gv = Gv
            # TODO: utilize symmetry between Gv and -Gv
            log.debug('grids = (%d, %d)  Gv size = %d',
                      n_rad, n_ang, self.wcoulG.size)
        return self

    def get_jk(self, dm, hermi=1, with_j=True, with_k=True,
               direct_scf_tol=None, omega=None):
        assert omega is None
        mo_coeff = getattr(dm, 'mo_coeff', None)
        #mo_occ = getattr(dm, 'mo_occ', None)
        dm = np.asarray(dm)
        dm_shape = dm.shape
        if dm.ndim == 2:
            dm = dm[np.newaxis]

        regular_dfj = with_j and not self.lr_dfj
        with_j = with_j and self.lr_dfj
        # Using direct-SCF algorithm if dm is obtained from SCF make_rdm1
        if mo_coeff is not None:
            last_dm, last_vj, last_vk = self._last_vs
            vj, vk = self._get_jk_sr(dm-last_dm, hermi, with_j, with_k)
            if with_j:
                vj += last_vj
                last_vj = vj.copy()
            if with_k:
                vk += last_vk
                last_vk = vk.copy()
            self._last_vs = (dm, last_vj, last_vk)
        else:
            vj, vk = self._get_jk_sr(dm, hermi, with_j, with_k)

        vj1, vk1 = self._get_jk_lr(dm, hermi, with_j, with_k)
        if with_j:
            vj += vj1
            vj = vj.reshape(dm_shape)
        if with_k:
            vk += vk1
            vk = vk.reshape(dm_shape)

        if regular_dfj:
            vj = df_jk.get_j(self, dm, hermi, self.direct_scf_tol)
            vj = vj.reshape(dm_shape)
        return vj, vk

    def _get_jk_sr(self, dm, hermi=1, with_j=True, with_k=True):
        if self.q_cond is None:
            self.build()

        assert hermi == 1
        assert dm.ndim == 3
        cpu0 = logger.process_clock(), logger.perf_counter()
        mol = self.mol
        n_dm, nao = dm.shape[:2]

        dm_cond = _make_dm_cond(mol, dm, self.direct_scf_tol)
        vhfopt = _CVHFOpt()
        vhfopt.dm_cond = dm_cond.ctypes.data_as(ctypes.c_void_p)
        vhfopt.logq_cond = self.q_cond.ctypes.data_as(ctypes.c_void_p)
        vhfopt.log_cutoff = np.log(self.direct_scf_tol)

        intor = mol._add_suffix(self._intor)
        cintor = getattr(libcvhf, intor)
        fdot = getattr(libcvhf, 'CVHFdot_nr_sr_s8')

        vj = vk = None
        dmsptr = []
        vjkptr = []
        fjk = []

        if with_j:
            fvj = _fpointer('CVHFnrs8_ji_s2kl')
            vj = np.empty((n_dm,nao,nao))
            for i in range(n_dm):
                dmsptr.append(dm[i].ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vj[i].ctypes.data_as(ctypes.c_void_p))
                fjk.append(fvj)

        if with_k:
            fvk = _fpointer('CVHFnrs8_li_s2kj')
            vk = np.empty((n_dm,nao,nao))
            for i in range(n_dm):
                dmsptr.append(dm[i].ctypes.data_as(ctypes.c_void_p))
                vjkptr.append(vk[i].ctypes.data_as(ctypes.c_void_p))
                fjk.append(fvk)

        shls_slice = (ctypes.c_int*8)(*([0, mol.nbas]*4))
        ao_loc = mol.ao_loc
        n_ops = len(dmsptr)
        comp = 1

        with mol.with_short_range_coulomb(self.omega):
            libcvhf.CVHFnr_sr_direct_drv(
                cintor, fdot, (ctypes.c_void_p*n_ops)(*fjk),
                (ctypes.c_void_p*n_ops)(*dmsptr),
                (ctypes.c_void_p*n_ops)(*vjkptr),
                ctypes.c_int(n_ops), ctypes.c_int(comp),
                shls_slice, ao_loc.ctypes.data_as(ctypes.c_void_p),
                self._cintopt, ctypes.byref(vhfopt),
                mol._atm.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.natm),
                mol._bas.ctypes.data_as(ctypes.c_void_p), ctypes.c_int(mol.nbas),
                mol._env.ctypes.data_as(ctypes.c_void_p))

        if with_j:
            for i in range(n_dm):
                lib.hermi_triu(vj[i], 1, inplace=True)
        if with_k:
            if hermi != 0:
                for i in range(n_dm):
                    lib.hermi_triu(vk[i], hermi, inplace=True)
        logger.timer(mol, 'short range part vj and vk', *cpu0)
        return vj, vk

    def _get_jk_lr(self, dm, hermi=1, with_j=True, with_k=True):
        mol = self.mol
        if self.grids is None:
            with mol.with_long_range_coulomb(self.omega):
                return df_jk.get_jk(self, dm, hermi, with_j, with_k)

        cpu0 = logger.process_clock(), logger.perf_counter()
        dm_shape = dm.shape
        nao = dm_shape[-1]
        dm_factor = None
        if hasattr(dm, 'mo_coeff'):
            mo_coeff = dm.mo_coeff
            mo_occ = dm.mo_occ
            if dm.ndim == 2:
                mo_coeff = mo_coeff[np.new_axis]
                mo_occ = mo_occ[np.new_axis]
            dm_factor = [_factorize_dm(c, occ) for c, occ in zip(mo_coeff, mo_occ)]
        dm = dm.reshape(-1,nao,nao)
        n_dm = dm.shape[0]
        vj = np.zeros((n_dm,nao,nao))
        vk = np.zeros((n_dm,nao,nao))

        ngrids = self.wcoulG.size
        max_memory = self.max_memory - lib.current_memory()[0]
        Gblksize = max(8, int(max_memory*1e6/16/nao**2/2))
        logger.debug1(mol, 'Gblksize = %d', Gblksize)
        for p0, p1 in lib.prange(0, ngrids, Gblksize):
            # TODO: Optimize ft_aopair GpqR, GpqI = Gpq
            Gpq = ft_ao.ft_aopair(mol, self.Gv[p0:p1])
            pLqR = np.asarray(Gpq.real.transpose(1,0,2))
            pLqI = np.asarray(Gpq.imag.transpose(1,0,2))
            for i_dm in range(n_dm):
                wcoulG = self.wcoulG[p0:p1]
                if with_j:
                    rhoG = np.einsum('Gpq,qp->G', Gpq, dm[i_dm])
                    vG = rhoG * wcoulG
                    vj[i_dm] += np.einsum('Gpq,G->pq', Gpq.real, vG.real)
                    vj[i_dm] += np.einsum('Gpq,G->pq', Gpq.imag, vG.imag)

                if not with_k:
                    continue

                if dm_factor is None:
                    pLiR = lib.einsum('pGq,qr->pGr', pLqR, dm[i_dm])
                    pLiI = lib.einsum('pGq,qr->pGr', pLqI, dm[i_dm])
                    pLiR *= wcoulG[:,None]
                    pLiI *= wcoulG[:,None]
                    vk[i_dm] += lib.einsum('pGi,qGi->pq', pLiR, pLqR)
                    vk[i_dm] += lib.einsum('pGi,qGi->pq', pLiI, pLqI)
                else:
                    pLiR = lib.dot(pLqR, dm_factor[i_dm])
                    pLiI = lib.dot(pLqI, dm_factor[i_dm])
                    vk[i_dm] += lib.einsum('pGi,qGi->pq', pLiR*wcoulG[:,None], pLiR)
                    vk[i_dm] += lib.einsum('pGi,qGi->pq', pLiI*wcoulG[:,None], pLiI)

        vj = vj.reshape(dm_shape)
        vk = vk.reshape(dm_shape)
        logger.timer(mol, 'long range part vj and vk', *cpu0)
        return vj, vk

LRDF = LRDensityFitting

def _make_dm_cond(mol, dm, direct_scf_tol):
    assert dm.ndim == 3
    ao_loc = mol.ao_loc
    dm_cond = [lib.condense('NP_absmax', d, ao_loc, ao_loc) for d in dm]
    dm_cond = np.max(dm_cond, axis=0)
    dm_cond += MIN_CUTOFF  # to remove divide-by-zero error
    return np.asarray(dm_cond, order='C', dtype=np.float32)

def _quadrature_roots(n, omega):
    rs, ws = scipy.special.roots_hermite(n*2)
    rs = rs[n:]
    ws = ws[n:]
    ws *= np.exp(rs**2)
    rs *= 2*omega
    ws *= 2*omega
    return rs, ws

def _angular_grids_Lebedev(n):
    n = LEBEDEV_NGRID[np.searchsorted(LEBEDEV_NGRID, n)]
    grid = np.empty((n, 4))
    libdft.MakeAngularGrid(grid.ctypes.data_as(ctypes.c_void_p),
                           ctypes.c_int(n))
    return grid[:,:3], 4*np.pi * grid[:,3]

def _angular_grids_legendre2d(n):
    m = max(int(n**.5), 1)
    xs, wt = scipy.special.roots_legendre(m)
    phi, wp = scipy.special.roots_legendre(m)
    theta = np.arccos(xs)
    phi *= np.pi
    phi += np.pi
    wp *= np.pi
    x = np.sin(theta[:,None]) * np.cos(phi)
    y = np.sin(theta[:,None]) * np.sin(phi)
    z = np.cos(theta[:,None]) * np.ones_like(phi)
    rs = np.array((x.ravel(), y.ravel(), z.ravel())).T
    ws = (wt[:,None] * wp).ravel()
    return rs, ws

def _get_coulG(Gv, omega):
    G2 = np.einsum('gx,gx->g', Gv, Gv)
    coulG = 4*np.pi/G2
    if omega > 0:
        coulG *= np.exp((-.25/omega**2)*G2)
    return coulG

def _non_uniform_Gv(n_rad, n_ang, omega, thresh=1e-4):
    assert omega > 0
    rs, ws = _quadrature_roots(n_rad, omega)
    ang_r, ang_w = _angular_grids_legendre2d(n_ang)
    if 1:
        Gv = np.einsum('i,jk->jik', rs, ang_r).reshape(-1,3)
        weights = np.einsum('i,j->ji', ws, ang_w).ravel()
    else:
        ang_r1, ang_w1 = _angular_grids_Lebedev(n_ang/16)
        ang_r2, ang_w2 = _angular_grids_Lebedev(n_ang/256)
        seg0 = rs < 2./3
        seg1 = (rs < 0.8) & ~seg0
        seg2 = ~(seg0 | seg1)
        #print(seg0.sum(), seg1.sum(), seg2.sum())
        Gv = np.vstack((
            np.einsum('i,jk->jik', rs[seg0], ang_r).reshape(-1,3),
            np.einsum('i,jk->jik', rs[seg1], ang_r1).reshape(-1,3),
            np.einsum('i,jk->jik', rs[seg2], ang_r2).reshape(-1,3),
        ))

        weights = np.hstack((
            np.einsum('i,j->ji', ws[seg0], ang_w).ravel(),
            np.einsum('i,j->ji', ws[seg1], ang_w1).ravel(),
            np.einsum('i,j->ji', ws[seg2], ang_w2).ravel(),
        ))
    pw_ints_norm = 1/(2*np.pi)**3
    factor_sph_coords_dV = np.einsum('gx,gx->g', Gv, Gv)
    weights *= pw_ints_norm * factor_sph_coords_dV

    wcoulG = _get_coulG(Gv, omega) * weights
    screening = wcoulG > .1*thresh / (n_rad * n_ang)
    Gv = Gv[screening]
    weights = weights[screening]
    return Gv, weights

def _factorize_dm(mo, occ):
    occ_mask = occ > 0
    dm_factor = mo[:,occ_mask]
    dm_factor *= occ[occ_mask]**.5
    return dm_factor
