def default_atomic_models():
    """
    Returns the default set of atomic models to be used, when not otherwise specified.
    """
    from lightweaver.rh_atoms import (
        Al_atom,
        C_atom,
        CaII_atom,
        Fe_atom,
        H_6_atom,
        He_9_atom,
        MgII_atom,
        N_atom,
        Na_atom,
        O_atom,
        S_atom,
        Si_atom,
    )

    atomic_models = [
        H_6_atom(),
        C_atom(),
        O_atom(),
        Si_atom(),
        Al_atom(),
        CaII_atom(),
        Fe_atom(),
        He_9_atom(),
        MgII_atom(),
        N_atom(),
        Na_atom(),
        S_atom(),
    ]
    return atomic_models
