"""
Isothermal, isobaric slab models (`IsoPromModel`), seen as a prominence (off
limb) and a filament (on disk), with both boundary condition types.
"""

import numpy as np
from conftest import (
    Halpha,
    Nrays,
    Nthreads,
    check_pops,
    gas_pressure,
    iterate,
    line_core_idx,
    vertical_intensity,
)

import promweaver as pw

PRESSURE = 0.05  # [Pa]


def iso_model(projection, bc_provider, **kwargs):
    kwargs.setdefault("active_atoms", ["H", "Ca"])
    return pw.IsoPromModel(
        projection,
        temperature=8000,
        pressure=PRESSURE,
        thickness=10e6,
        vturb=5e3,
        altitude=10e6,
        Nthreads=Nthreads,
        Nrays=Nrays,
        bc_provider=bc_provider,
        **kwargs,
    )


def test_prominence_cone_bc(bc_provider, reference):
    model = iso_model("prominence", bc_provider, BcType=pw.ConePromBc)
    iterate(model)
    check_pops(model)
    # The pressure updates hold the slab at its prescribed pressure.
    np.testing.assert_allclose(gas_pressure(model), PRESSURE, rtol=1e-3)
    vertical_intensity(model, "iso_prominence_cone", reference)


def test_prominence_uniform_j_prd(bc_provider, reference):
    model = iso_model("prominence", bc_provider, prd=True)
    iterate(model)
    check_pops(model)
    np.testing.assert_allclose(gas_pressure(model), PRESSURE, rtol=1e-3)
    vertical_intensity(model, "iso_prominence_uniform_j_prd", reference)


def test_filament(bc_provider, reference):
    model = iso_model("filament", bc_provider)
    iterate(model)
    check_pops(model)
    I = vertical_intensity(model, "iso_filament", reference)

    # A filament is dark in the core of H alpha against the disk behind it.
    idx = line_core_idx(model, Halpha)
    wave = model.ctx.spect.wavelength[idx : idx + 1]
    I_disk = bc_provider.compute_I(wave, 1.0)
    assert I[idx] < 0.9 * I_disk[0]
