"""
Tests for the pre-Lightweaver 1.0 (camelCase) names still accepted by
Promweaver, and for `python -m promweaver.migrate`.
"""

import pytest
from conftest import Nthreads
from lightweaver.deprecation import LightweaverDeprecationWarning
from lightweaver.migrate import build_kwarg_registry, migrate_source

import promweaver as pw
import promweaver.migrate  # noqa: F401
from promweaver.prom_model import PromModel

OLD_KWARGS = {"NmaxIter": 3, "popsTol": 1e-3, "returnFinalConvergence": False}


@pytest.fixture(scope="module")
def models(bc_provider):
    kwargs = dict(active_atoms=["H"], Nthreads=Nthreads, bc_provider=bc_provider)
    return {
        "iso": pw.IsoPromModel("prominence", 8000, 0.05, 10e6, 5e3, 10e6, **kwargs),
        "pctr": pw.PctrPromModel(
            "prominence", 8000, 1e5, 0.05, 0.001, 5e-4, 5e3, 10e6, 2, **kwargs
        ),
    }


@pytest.mark.parametrize("name", ["iso", "pctr"])
def test_iterate_se_old_kwargs(models, name):
    with pytest.warns(LightweaverDeprecationWarning) as record:
        models[name].iterate_se(quiet=True, **OLD_KWARGS)
    assert sorted(str(w.message).split("`")[1] for w in record) == sorted(OLD_KWARGS)
    # The warnings point at the caller, not inside Promweaver.
    assert all(w.filename == __file__ for w in record)


def test_iterate_se_old_and_new_kwarg(models):
    with pytest.raises(TypeError, match="popsTol"):
        PromModel.iterate_se(models["iso"], popsTol=1e-3, pops_tol=1e-3)


def test_migrate_registry():
    registry = build_kwarg_registry()
    assert set(PromModel.iterate_se.__lw_renamed_kwargs__) <= registry["iterate_se"]


def test_migrate_source():
    old = (
        "model = pw.IsoPromModel('prominence', 8000, 0.05, 10e6, 5e3, 10e6, active_atoms=['H'])\n"
        "model.iterate_se(popsTol=1e-4, prdIterTol=1e-3, NmaxIter=100)\n"
        "print(model.atmos.nHTot)\n"
    )
    new = (
        "model = pw.IsoPromModel('prominence', 8000, 0.05, 10e6, 5e3, 10e6, active_atoms=['H'])\n"
        "model.iterate_se(pops_tol=1e-4, prd_iter_tol=1e-3, max_iter=100)\n"
        "print(model.atmos.nh_tot)\n"
    )
    migrated, _, _ = migrate_source(old, build_kwarg_registry())
    assert migrated == new
