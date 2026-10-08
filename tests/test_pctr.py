"""
Prominence models with a prominence-corona transition region (`PctrPromModel`),
with both boundary condition types, and with a radial velocity (Doppler-shifted
incident radiation).
"""

import lightweaver as lw
import numpy as np
from conftest import Nrays, Nthreads, check_pops, iterate, tabulated_falc_bc, vertical_intensity

import promweaver as pw


def pctr_model(bc_provider, **kwargs):
    kwargs.setdefault("active_atoms", ["H", "Ca"])
    return pw.PctrPromModel(
        "prominence",
        cen_temperature=8000,
        tr_temperature=1e5,
        cen_pressure=0.05,
        tr_pressure=0.001,
        max_cmass=5e-4,
        vturb=5e3,
        altitude=10e6,
        gamma=2,
        Nthreads=Nthreads,
        Nrays=Nrays,
        bc_provider=bc_provider,
        **kwargs,
    )


def check_structure(model):
    """
    The temperature should rise from the cool core to the transition region on
    both sides, and the height scale (updated during the iterations to follow
    the column mass) should stay monotonic.
    """
    T = model.atmos.temperature
    centre = T.shape[0] // 2
    assert T[centre] == np.min(T)
    assert np.all(np.diff(T[: centre + 1]) <= 0.0)
    assert np.all(np.diff(T[centre:]) >= 0.0)
    assert np.all(np.diff(model.atmos.z) < 0.0)


def test_prominence_cone_bc(bc_provider, reference):
    model = pctr_model(bc_provider, BcType=pw.ConePromBc)
    iterate(model)
    check_pops(model)
    check_structure(model)
    vertical_intensity(model, "pctr_prominence_cone", reference)


def test_prominence_uniform_j(bc_provider, reference):
    model = pctr_model(bc_provider, BcType=pw.UniformJPromBc)
    iterate(model)
    check_pops(model)
    check_structure(model)
    vertical_intensity(model, "pctr_prominence_uniform_j", reference)


def test_vrad(reference):
    vrad = 20e3
    active_atoms = ["H", "Ca"]
    model = pctr_model(
        tabulated_falc_bc(active_atoms, vz=vrad),
        active_atoms=active_atoms,
        BcType=pw.ConePromBc,
        vrad=vrad,
        ctx_kwargs={"ng_options": lw.NgOptions(Norder=3, Nperiod=5, Ndelay=5, threshold=3e-2)},
    )
    iterate(model)
    check_pops(model)
    check_structure(model)
    vertical_intensity(model, "pctr_vrad", reference)
