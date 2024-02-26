'''
Conversion from the Basis Set Exchange format to PySCF format

17 Nov 2021 Susi Lehtola
'''

try:
    import basis_set_exchange
    from basis_set_exchange import lut, manip, sort
except ImportError:
    basis_set_exchange = None


def _orbital_basis(basis):
    '''Extracts the orbital basis from the BSE format in PySCF format'''

    r = {}

    #basis = manip.make_general(basis, False, True)
    basis = sort.sort_basis(basis, False)

    # Elements for which we have electron basis
    electron_elements = [k for k, v in basis['elements'].items() if 'electron_shells' in v]

    # List of references in the used basis
    reference_list = []

    # Electron Basis
    for z in electron_elements:
        data = basis['elements'][z]

        sym = lut.element_sym_from_Z(z, True)

        # List of shells
        atom_shells = []
        sp_contraction = False
        for shell in data['electron_shells']:
            exponents = shell['exponents']
            coefficients = shell['coefficients']
            ncontr = len(coefficients)
            am = shell['angular_momentum']
            if len(am) == 1:
                shell_data = [am[0]]
                for e_c in zip(exponents, *coefficients):
                    shell_data.append([float(x) for x in e_c])
                atom_shells.append(shell_data)
            else:
                sp_contraction = True  # such as Pople basis
                for am in shell['angular_momentum']:
                    shell_data = [am]
                    for e, c in zip(exponents, coefficients[am]):
                        shell_data.append([float(e), float(c)])
                    atom_shells.append(shell_data)
        if sp_contraction:
            atom_shells = sorted(atom_shells, key=lambda x: x[0])
        r[sym] = atom_shells

        # Collect the literature references
        for ref in data['references']:
            for key in ref['reference_keys']:
                reference_list.append(key)

    return r, list(set(reference_list))


def _ecp_basis(basis):
    '''Extracts the ECP from the BSE format in PySCF format'''

    r = {}

    #basis = manip.make_general(basis, False, True)
    basis = sort.sort_basis(basis, False)

    # Elements for which we have ECP
    ecp_elements = [k for k, v in basis['elements'].items() if 'ecp_potentials' in v]

    # Electron Basis
    for z in ecp_elements:
        data = basis['elements'][z]
        sym = lut.element_sym_from_Z(z, True)

        # Sort lowest->highest
        ecp_list = sorted(data['ecp_potentials'], key=lambda x: x['angular_momentum'])

        # List of ECP
        atom_ecp = []
        for ir, pot in enumerate(ecp_list):
            rexponents = pot['r_exponents']
            gexponents = pot['gaussian_exponents']
            coefficients = pot['coefficients']
            am = pot['angular_momentum']
            assert len(am) == 1

            # PySCF wants the data in order of rexp=0, 1, 2, ..
            rcontr = [[] for _ in range(max(rexponents) + 1)]
            for r_i, e, c in zip(rexponents, gexponents, coefficients[0]):
                rcontr[r_i].append([float(e), float(c)])
            shell_data = [am[0], rcontr]
            atom_ecp.append(shell_data)
        r[sym] = [data['ecp_electrons'], atom_ecp]

    return r

def _print_basis_information(basis):
    name = basis['name']
    version = basis['version']
    revision_description = basis['revision_description']
    revision_date = basis['revision_date']
    print('{} basis set, version {}'.format(name, version))
    print('Last revised on {}'.format(revision_date))
    print('Revision description: {}'.format(revision_description))

if __name__ == '__main__':
    from basis_set_exchange import api, references

    # Get reference data
    reference_data = api.get_reference_data()
    #print(references)

    o631gbas = api.get_basis('6-31g', elements='O')
    #print('O 6-31G basis, BSE format\n{}'.format(o631gbas))
    _print_basis_information(o631gbas)
    o631gorb, o631gref = _orbital_basis(o631gbas)
    print('O 6-31G orbital basis, PySCF format\n{}'.format(o631gorb))
    print('Literature references')
    for ref in o631gref:
        print(references.reference_text(ref, reference_data[ref]))
    print('')

    nalanl2dzbas = api.get_basis('lanl2dz', elements='Na')
    #print('Na LANL2DZ basis, BSE format\n{}'.format(nalanl2dzbas))
    _print_basis_information(nalanl2dzbas)
    nalanl2dzorb, nalanl2dzref = _orbital_basis(nalanl2dzbas)
    print('Na LANL2DZ orbital basis, PySCF format\n{}'.format(nalanl2dzorb))
    nalanl2dzecp = _ecp_basis(nalanl2dzbas)
    print('Na LANL2DZ ECP basis, PySCF format\n{}'.format(nalanl2dzecp))
    print('Literature references')
    for ref in nalanl2dzref:
        print(references.reference_text(ref, reference_data[ref]))
    print('')
