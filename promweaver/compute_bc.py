from typing import List, Optional

import lightweaver as lw
from lightweaver.atomic_model import AtomicModel
from lightweaver.fal import Falc82
import numpy as np

from .utils import default_atomic_models


def compute_falc_bc_ctx(active_atoms : List[str], atomic_models : Optional[List[AtomicModel]]=None,
                        prd : bool=False, vz : Optional[float]=None, Nthreads: int=1,
                        quiet: bool=False, ctx_kwargs: Optional[dict]=None) -> lw.Context:
    """
    Configures and iterates a lightweaver Context with a FALC atmosphere, the
    selected atomic models, and active atoms. This can then be used with a
    DynamicContextPromBcProvider.
    """

    if atomic_models is None:
        atomic_models = default_atomic_models()
    if ctx_kwargs is None:
        ctx_kwargs = {}

    atmos = Falc82()
    if vz is not None:
        atmos.vlos[:] = vz
        atmos.quadrature(5)
    else:
        atmos.quadrature(3)

    rad_set = lw.RadiativeSet(atomic_models)
    rad_set.set_active(*active_atoms)
    eq_pops = rad_set.compute_eq_pops(atmos)
    spect = rad_set.compute_wavelength_grid()

    hprd = prd and (vz is not None)
    ctx = lw.Context(atmos, spect, eq_pops, Nthreads=Nthreads, hprd=hprd, **ctx_kwargs)
    lw.iterate_ctx_se(ctx, prd=prd, quiet=quiet)
    return ctx

def tabulate_bc(ctx: lw.Context, wavelength: Optional[np.ndarray]=None, 
                mu_grid: Optional[np.ndarray]=None, 
                compute_rays_kwargs: Optional[dict]=None):
    """
    Computes the necessary data for a `TabulatedPromBcProvider` from a Context
    (e.g. from `compute_falc_bc_ctx`).

    Parameters
    ----------
    ctx : lw.Context
        The Context that will be used to compute the boundary condition (should
        already be converged, as no iteration occurs here).
    wavelength : array-like, optional
        The wavelength grid to use for the boundary condition table. Default:
        the one used in the Context.
    mu_grid : array-like, optional
        The grid of viewing angles (in the form of mu) to be used for the
        boundary condition table. Default: `np.linspace(0.01, 1.0, 100)`.
    compute_rays_kwargs : dict, optional
        Any extra kwargs to be passed to `compute_rays` (e.g. `refinePrd`).

    Returns
    -------
    data : dict
        Dict with keys 'wavelength', 'mu_grid' and 'I' that can be splatted
        directly into a `TabulatedPromBcProvider` or pickled.
    """
    if wavelength is None:
        wavelength = ctx.spect.wavelength

    if mu_grid is None:
        mu_grid = np.linspace(0.01, 1.0, 100)

    if compute_rays_kwargs is None:
        compute_rays_kwargs = {}
    
    Igrid = ctx.compute_rays(wavelengths=wavelength, mus=mu_grid, **compute_rays_kwargs)
    return {'wavelength': wavelength, 'mu_grid': mu_grid, 'I': Igrid}
