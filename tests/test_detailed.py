"""
Detailed atoms: atoms whose populations are held fixed (from a previous NLTE
solution, or LTE) while the radiation field is still computed from all of their
transitions. Also covers removing transitions from the atomic models inside the
prominence.
"""

import lightweaver as lw
import numpy as np
import pytest
from conftest import (
    BC_ATOMS,
    Nrays,
    Nthreads,
    check_pops,
    iterate,
    vertical_intensity,
)
from lightweaver.rh_atoms import H_atom, MgII_atom

import promweaver as pw

ISO_PARAMS = dict(temperature=8000, pressure=0.05, thickness=0.5e6, vturb=5e3)
PCTR_PARAMS = dict(
    cen_temperature=8000,
    tr_temperature=1e5,
    cen_pressure=0.05,
    tr_pressure=0.001,
    max_cmass=5e-4,
    vturb=5e3,
    gamma=2,
)


def iso_model(bc_provider, **kwargs):
    kwargs.setdefault("active_atoms", BC_ATOMS)
    return pw.IsoPromModel(
        "prominence",
        **ISO_PARAMS,
        altitude=10e6,
        Nthreads=Nthreads,
        Nrays=Nrays,
        BcType=pw.ConePromBc,
        bc_provider=bc_provider,
        prd=True,
        **kwargs,
    )


def pctr_model(bc_provider, **kwargs):
    kwargs.setdefault("active_atoms", BC_ATOMS)
    return pw.PctrPromModel(
        "prominence",
        **PCTR_PARAMS,
        altitude=10e6,
        Nthreads=Nthreads,
        Nrays=Nrays,
        BcType=pw.ConePromBc,
        bc_provider=bc_provider,
        prd=True,
        **kwargs,
    )


@pytest.fixture(scope="module")
def iso_nlte(bc_provider):
    """
    Full NLTE solution of the isothermal prominence. Do not modify.
    """
    model = iso_model(bc_provider)
    iterate(model)
    return model


@pytest.fixture(scope="module")
def pctr_nlte(bc_provider):
    """
    Full NLTE solution of the PCTR prominence. Do not modify.
    """
    model = pctr_model(bc_provider)
    iterate(model)
    return model


def test_iso_nlte(iso_nlte, reference):
    check_pops(iso_nlte)
    vertical_intensity(iso_nlte, "detailed_iso_nlte", reference)


def test_iso_detailed_nlte_h_crd(iso_nlte, bc_provider, reference):
    # NOTE(cmo): Detailed H with PRD (the default) is covered by the PCTR tests.
    model = iso_model(
        bc_provider,
        detailed_atoms=["H"],
        detailed_pops={"H": np.copy(iso_nlte.eq_pops["H"])},
        ctx_kwargs={"detailed_atom_prd": False},
    )
    iterate(model)
    check_pops(model)
    np.testing.assert_array_equal(model.eq_pops["H"], iso_nlte.eq_pops["H"])
    vertical_intensity(model, "detailed_iso_nlte_h_crd", reference)


def test_iso_detailed_nlte_mg(iso_nlte, bc_provider, reference):
    # NOTE(cmo): Detailed Mg needs a sensible NLTE starting point for the other
    # atoms; seed it from an H and Ca only solution.
    model_h_ca = iso_model(bc_provider, active_atoms=["H", "Ca"])
    iterate(model_h_ca)

    model = iso_model(
        bc_provider,
        detailed_atoms=["Mg"],
        detailed_pops={"Mg": np.copy(iso_nlte.eq_pops["Mg"])},
    )
    model.eq_pops["H"][...] = model_h_ca.eq_pops["H"]
    model.eq_pops["Ca"][...] = model_h_ca.eq_pops["Ca"]
    model.atmos.ne[:] = model_h_ca.atmos.ne
    model.atmos.nh_tot[:] = model_h_ca.atmos.nh_tot
    iterate(model)
    check_pops(model)
    vertical_intensity(model, "detailed_iso_nlte_mg", reference)


def test_iso_detailed_lte_h(iso_nlte, bc_provider, reference):
    model = iso_model(bc_provider, detailed_atoms=["H"])
    iterate(model)
    check_pops(model)
    I = vertical_intensity(model, "detailed_iso_lte_h", reference)
    # LTE hydrogen in a prominence is a poor approximation.
    assert not np.allclose(I, iso_nlte.compute_rays(mus=1.0), rtol=0.1)


def test_pctr_nlte(pctr_nlte, reference):
    check_pops(pctr_nlte)
    vertical_intensity(pctr_nlte, "detailed_pctr_nlte", reference)


def without_ground_transitions(atom):
    """
    Remove every transition from the ground level of `atom`.
    """
    atom.lines[:] = [l for l in atom.lines if l.i != 0]
    atom.continua[:] = [c for c in atom.continua if c.i != 0]
    lw.reconfigure_atom(atom)
    return atom


def atomic_models_replacing(element, atom):
    atomic_models = pw.default_atomic_models()
    for i, a in enumerate(atomic_models):
        if a.element == lw.PeriodicTable[element]:
            atomic_models[i] = atom
    return atomic_models


@pytest.mark.parametrize(
    "element, atom, name",
    [("H", H_atom, "h_no_lyman"), ("Mg", MgII_atom, "mg_no_resonance")],
    ids=["h_no_lyman", "mg_no_resonance"],
)
def test_pctr_removing_lines(pctr_nlte, bc_provider, reference, element, atom, name):
    atomic_models = atomic_models_replacing(element, without_ground_transitions(atom()))
    model = pctr_model(
        bc_provider,
        atomic_models=atomic_models,
        detailed_atoms=["H"],
        detailed_pops={"H": np.copy(pctr_nlte.eq_pops["H"])},
    )
    iterate(model)
    check_pops(model)
    element_model = model.rad_set[element]
    assert all(t.i != 0 for t in element_model.lines + element_model.continua)
    vertical_intensity(model, f"detailed_pctr_{name}", reference)
