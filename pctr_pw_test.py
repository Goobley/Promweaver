import lightweaver as lw
import matplotlib.pyplot as plt
import numpy as np
import promweaver as pw

Nthreads = 8

active_atoms = ["H", "Ca"]

bc_ctx = pw.compute_falc_bc_ctx(active_atoms, Nthreads=Nthreads)
bc_table = pw.tabulate_bc(bc_ctx)
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
)

modelJ = pw.PctrPromModel(
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
    bc_provider=bc_provider,
    BcType=pw.UniformJPromBc,
    Nrays=3,
)

plt.ion()

model.iterate_se()
modelJ.iterate_se()
Ivert = model.compute_rays(wavelengths=model.ctx.spect.wavelength, mus=1.0)
IvertJ = modelJ.compute_rays(wavelengths=modelJ.ctx.spect.wavelength, mus=1.0)

plt.plot(model.ctx.spect.wavelength, Ivert)
plt.plot(model.ctx.spect.wavelength, IvertJ)
