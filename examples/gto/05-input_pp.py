#!/usr/bin/env python

'''
Using pseudo potential in molecule
'''

import numpy
from pyscf import gto

mol = gto.M(
    atom = 'Mg 0 0 0; Mg 0 0 1.5',
    basis = 'gth-szv',
    pseudo = 'gth-lda')
