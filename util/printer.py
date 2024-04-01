def fgroup(df):
    """Prints group assignments in human readable format given a list of molecules in a dataframe"""
    df = df.iloc[:, 1:425]
    bt = df.apply(lambda x: x > 0)
    print(bt.apply(lambda x: list(df.columns[x.values] + 1), axis=1))


def pretty_format_molecule_set(self, properties, full_valence_valid,  order, full_vec=np.zeros((220,)), constr=None, ms=None,):
    """Converts the found solutions to human readable format. Adds molecules only if they meet all
    constraints (property, group, valency)

    Args:
        `self`: set to the MolecularSearchEnv obj if called during the RL run; None otherwise.
        `properties`: the properties of the correspoding molecule, polled from the property models.
        `full_valence_valid`: a boolean that indicates whether a zero-valency integral structure
                            has been found.
        `building_blocks`: is the set of building blocks to consider printing.
        `full_vec`: a full representation of the state vector, if called outside the RL run.
        `constraints`: CAMD constraints, if called outside the RL run.
        `molecule set`: list of found molecules, if called outside the RL run.

    Returns: a tuple representing the set of functional groups in a discovered solution, formatted
             as `(NUM GROUP_NAME, NUM2 GROUP_NAME2, ..)`

    Example: (1,1,0,0,0,0) --> (1 CH3, 1 CH2)"""
    if (order == 1):
        found_molecules = full_vec
        molecule_set = ms
        constraints = constr
        if (full_vec.any()):
            building_blocks = found_molecules.nonzero()
        if (not (full_vec.any())):
            constraints = self.constraints()
            building_blocks = self._building_blocks
            found_molecules = np.zeros((220,))
            found_molecules[building_blocks] = self._state['observation']
            molecule_set = self._molecule_set
        if group_constraints_check(found_molecules, constraints) and full_valence_valid:
            names = first_order_group_names[building_blocks]
            idx = found_molecules[building_blocks].nonzero()
            found_molecules = np.char.mod(
                "%d ", found_molecules[building_blocks])
            if show_prop:
                molecule_set.add((
                    tuple(np.char.add(found_molecules[idx], names[idx])), tuple(properties[:, 0].flatten()), tuple(properties[:, 1].flatten())))
            else:
                molecule_set.add((
                    tuple(np.char.add(found_molecules[idx], names[idx]))))
