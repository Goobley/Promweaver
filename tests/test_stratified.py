"""
A stratified filament (`StratifiedPromModel`) with a velocity gradient: a CRD
solution is used to seed the populations of a PRD one, as is often needed for
convergence (e.g. with Mg II).
"""

import numpy as np
from conftest import (
    Nrays,
    Nthreads,
    check_pops,
    gas_pressure,
    iterate,
    vertical_intensity,
)

import promweaver as pw

ACTIVE_ATOMS = ["H", "Ca"]


def stratification(Npoints=60):
    """
    A synthetic slab 3 Mm thick: a cool core at 8 kK, rising smoothly to 30 kK
    at both surfaces, at uniform pressure, with line-of-sight velocities from
    -5 to 5 km/s across it. Returns z [m] (decreasing), T [K], vlos [m/s] and
    pressure [Pa].
    """
    z = np.linspace(3e6, 0.0, Npoints)
    distance_from_centre = np.abs(z - 1.5e6)
    T = 8e3 + 22e3 * 0.5 * (1.0 + np.tanh((distance_from_centre - 1.2e6) / 1e5))
    vlos = np.linspace(-5e3, 5e3, Npoints)
    pressure = np.full(Npoints, 0.05)
    return z, T, vlos, pressure


def filament_model(bc_provider, **kwargs):
    z, T, vlos, pressure = stratification()
    return pw.StratifiedPromModel(
        "filament",
        z=z,
        temperature=T,
        vlos=vlos,
        vturb=np.full_like(z, 5e3),
        altitude=10e6,
        pressure=pressure,
        initial_ionisation_fraction=0.5,
        active_atoms=ACTIVE_ATOMS,
        Nthreads=Nthreads,
        Nrays=Nrays,
        bc_provider=bc_provider,
        **kwargs,
    )


def test_crd_seeding_prd(bc_provider, reference):
    model_crd = filament_model(bc_provider, prd=False)
    iterate(model_crd)
    check_pops(model_crd)
    np.testing.assert_allclose(gas_pressure(model_crd), model_crd.pressure, rtol=1e-3)
    vertical_intensity(model_crd, "stratified_crd", reference)

    model = filament_model(bc_provider, BcType=pw.ConePromBc, prd=True)
    for a in ACTIVE_ATOMS:
        model.eq_pops[a][...] = model_crd.eq_pops[a]
    model.ne[:] = model_crd.ne
    model.nh_tot[:] = model_crd.nh_tot
    iterate(model, max_prd_sub_iter=5)
    check_pops(model)
    np.testing.assert_allclose(gas_pressure(model), model.pressure, rtol=1e-3)
    vertical_intensity(model, "stratified_prd", reference)
