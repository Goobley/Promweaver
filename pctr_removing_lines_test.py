import lightweaver as lw
from lightweaver.rh_atoms import H_6_atom, MgII_atom
import matplotlib.pyplot as plt
import numpy as np
import promweaver as pw
import psutil

process = psutil.Process()

Nthreads = 8

active_atoms = ["H", "Ca", "Mg"]

bc_ctx = pw.compute_falc_bc_ctx(active_atoms, prd=True, Nthreads=Nthreads)
bc_table = pw.tabulate_bc(bc_ctx, max_rays=10)
bc_provider = pw.TabulatedPromBcProvider(**bc_table)

print("---------")

model = pw.PctrPromModel(
    "prominence",
    cen_temperature=8000,
    tr_temperature=1e5,
    cen_pressure=0.05,
    tr_pressure=0.001,
    max_cmass=5e-4,
    vturb=5e3,
    altitude=10e6,
    gamma=2,
    active_atoms=active_atoms,
    Nthreads=Nthreads,
    Nrays=10,
    BcType=pw.ConePromBc,
    bc_provider=bc_provider,
    prd=True,
)

model.iterate_se()

detailed_atoms = ["H"]
detailed_pops = {"H": np.copy(model.eq_pops["H"])}


def H_no_lyman():
    h = H_6_atom()
    idxs_to_delete = [i for i, l in enumerate(h.lines) if l.i == 0]
    for i in reversed(idxs_to_delete):
        del h.lines[i]

    idxs_to_delete = [i for i, l in enumerate(h.continua) if l.i == 0]
    for i in reversed(idxs_to_delete):
        del h.continua[i]
    lw.reconfigure_atom(h)
    return h


atomic_models = pw.default_atomic_models()
for i, atom in enumerate(atomic_models):
    if atom.element == lw.PeriodicTable["H"]:
        atomic_models[i] = H_no_lyman()

model_detailed_H_no_lyman = pw.PctrPromModel(
    "prominence",
    cen_temperature=8000,
    tr_temperature=1e5,
    cen_pressure=0.05,
    tr_pressure=0.001,
    max_cmass=5e-4,
    vturb=5e3,
    altitude=10e6,
    gamma=2,
    active_atoms=active_atoms,
    atomic_models=atomic_models,
    detailed_atoms=detailed_atoms,
    detailed_pops=detailed_pops,
    Nthreads=Nthreads,
    Nrays=10,
    BcType=pw.ConePromBc,
    bc_provider=bc_provider,
    prd=True,
)
model_detailed_H_no_lyman.iterate_se()


def MgII_no_resonance():
    mg = MgII_atom()
    idxs_to_delete = [i for i, l in enumerate(mg.lines) if l.i == 0]
    for i in reversed(idxs_to_delete):
        del mg.lines[i]

    idxs_to_delete = [i for i, l in enumerate(mg.continua) if l.i == 0]
    for i in reversed(idxs_to_delete):
        del mg.continua[i]
    lw.reconfigure_atom(mg)
    return mg


atomic_models = pw.default_atomic_models()
for i, atom in enumerate(atomic_models):
    if atom.element == lw.PeriodicTable["Mg"]:
        atomic_models[i] = MgII_no_resonance()
model_detailed_Mg_no_resonance = pw.PctrPromModel(
    "prominence",
    cen_temperature=8000,
    tr_temperature=1e5,
    cen_pressure=0.05,
    tr_pressure=0.001,
    max_cmass=5e-4,
    vturb=5e3,
    altitude=10e6,
    gamma=2,
    active_atoms=active_atoms,
    atomic_models=atomic_models,
    detailed_atoms=detailed_atoms,
    detailed_pops=detailed_pops,
    Nthreads=Nthreads,
    Nrays=10,
    BcType=pw.ConePromBc,
    bc_provider=bc_provider,
    prd=True,
)
model_detailed_Mg_no_resonance.iterate_se()

Ivert = model.compute_rays(wavelengths=model.ctx.spect.wavelength, mus=1.0)
Ivert_detailed_H_no_lyman = model_detailed_H_no_lyman.compute_rays(mus=1.0)
Ivert_detailed_Mg_no_resonance = model_detailed_Mg_no_resonance.compute_rays(mus=1.0)

plt.ion()
plt.plot(model.ctx.spect.wavelength, Ivert, label="Full NLTE")
plt.plot(
    model_detailed_H_no_lyman.ctx.spect.wavelength,
    Ivert_detailed_H_no_lyman,
    label="No Lyman internal to prominence",
)
plt.plot(
    model_detailed_Mg_no_resonance.ctx.spect.wavelength,
    Ivert_detailed_Mg_no_resonance,
    label="No Mg resonance internal to prominence",
)
plt.legend()
