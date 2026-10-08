"""
Shared fixtures and helpers for the Promweaver smoke tests.

The incident radiation for the models is tabulated once per session from a
converged FAL-C solution (H, Ca II and Mg II active), and shared by all of the
tests. The models are kept small (few rays, few atoms, PRD only where needed) so
the suite runs in under three minutes; they are smoke tests of the workflows,
not production setups.
"""

import os

import lightweaver as lw
import numpy as np
import pytest
from references import ReferenceStore

import promweaver as pw

# NOTE(cmo): GitHub's standard Linux runners have 4 cores.
Nthreads = 4
Nrays = 3
MAX_ITER = 2000
BC_ATOMS = ["H", "Ca", "Mg"]

# Optionally force a particular iteration scheme (scalar, SSE2, AVX2FMA, AVX512).
if "LW_TEST_SIMD" in os.environ:
    lw.ConfigDict["SimdImpl"] = os.environ["LW_TEST_SIMD"]


def tabulated_falc_bc(active_atoms=BC_ATOMS, prd=False, vz=None):
    """
    Converge FAL-C with the given atoms active and tabulate its outgoing
    radiation as a `TabulatedPromBcProvider`.
    """
    bc_ctx = pw.compute_falc_bc_ctx(active_atoms, prd=prd, vz=vz, Nthreads=Nthreads, quiet=True)
    return pw.TabulatedPromBcProvider(**pw.tabulate_bc(bc_ctx))


@pytest.fixture(scope="session")
def bc_provider():
    """
    FAL-C (H, Ca II, Mg II) incident radiation. Do not modify.
    """
    return tabulated_falc_bc()


@pytest.fixture(scope="session")
def reference():
    store = ReferenceStore()
    yield store
    store.save()


def iterate(model, **kwargs):
    """
    Iterate a model to statistical equilibrium, checking that it converged.
    """
    Niter = model.iterate_se(quiet=True, max_iter=MAX_ITER, **kwargs)
    assert Niter < MAX_ITER - 1
    return Niter


def check_pops(model):
    """
    The populations of every active atom should be finite and positive, and sum
    to the total population of the species.
    """
    for atom in model.ctx.active_atoms:
        pops = model.eq_pops.atomic_pops[atom.element]
        assert np.all(np.isfinite(pops.n))
        assert np.all(pops.n > 0.0)
        np.testing.assert_allclose(pops.n.sum(axis=0), pops.n_total, rtol=1e-10)


def vertical_intensity(model, name, reference):
    """
    Synthesise the vertical (mu = 1) emergent intensity, check it is sane and
    compare it against the stored reference `name`.
    """
    I = model.compute_rays(mus=1.0)
    assert np.all(np.isfinite(I))
    assert np.all(I > 0.0)
    reference.check(name, I)
    return I


def gas_pressure(model):
    atmos = model.atmos
    N = lw.DefaultAtomicAbundance.total_abundance * atmos.nh_tot + atmos.ne
    return N * lw.KBoltzmann * atmos.temperature


def line_core_idx(model, lambda0):
    """
    The index of the wavelength closest to `lambda0` [nm] in the model's grid.
    """
    return np.argmin(np.abs(model.ctx.spect.wavelength - lambda0))


# Vacuum line centres [nm] from the rh_atoms models.
Halpha = 656.469
CaK = 393.478
MgK = 279.635


def pytest_report_header(config):
    return (
        f"promweaver {pw.__version__}, lightweaver {lw.__version__}, "
        f"SimdImpl: {lw.ConfigDict['SimdImpl']}"
    )
