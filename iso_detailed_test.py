import lightweaver as lw
import matplotlib.pyplot as plt
import numpy as np
import promweaver as pw
import psutil

process = psutil.Process()


def print_mem():
    print(f"Mem: {process.memory_info().rss / 1024 / 1024} MB")


Nthreads = 8

active_atoms = ["H", "Ca", "Mg"]

print_mem()
bc_ctx = pw.compute_falc_bc_ctx(active_atoms, prd=True, Nthreads=Nthreads)
print_mem()
bc_table = pw.tabulate_bc(bc_ctx, max_rays=10)
print_mem()
bc_provider = pw.TabulatedPromBcProvider(**bc_table)

print("---------")

print_mem()
model = pw.IsoPromModel(
    "prominence",
    temperature=8000,
    pressure=0.05,
    thickness=0.5e6,
    vturb=5e3,
    altitude=10e6,
    active_atoms=active_atoms,
    Nthreads=Nthreads,
    Nrays=10,
    BcType=pw.ConePromBc,
    bc_provider=bc_provider,
    prd=True,
)

print_mem()

model.iterate_se()

detailed_atoms = ["H"]
detailed_pops = {"H": np.copy(model.eq_pops["H"])}
model_detailed_nlte_H = pw.IsoPromModel(
    "prominence",
    temperature=8000,
    pressure=0.05,
    thickness=0.5e6,
    vturb=5e3,
    altitude=10e6,
    active_atoms=active_atoms,
    detailed_atoms=detailed_atoms,
    detailed_pops=detailed_pops,
    Nthreads=Nthreads,
    Nrays=10,
    BcType=pw.ConePromBc,
    bc_provider=bc_provider,
    prd=True,
)
print_mem()
model_detailed_nlte_H.iterate_se()

detailed_atoms = ["H"]
detailed_pops = {"H": np.copy(model.eq_pops["H"])}
model_detailed_nlte_H_crd = pw.IsoPromModel(
    "prominence",
    temperature=8000,
    pressure=0.05,
    thickness=0.5e6,
    vturb=5e3,
    altitude=10e6,
    active_atoms=active_atoms,
    detailed_atoms=detailed_atoms,
    detailed_pops=detailed_pops,
    Nthreads=Nthreads,
    Nrays=10,
    BcType=pw.ConePromBc,
    bc_provider=bc_provider,
    prd=True,
    ctx_kwargs={"detailedAtomPrd": False},
)
print_mem()
model_detailed_nlte_H_crd.iterate_se()

# NOTE(cmo): Doing detailed Mg from the start seems to need some help... it doesn't seem to start in a sensible position. Use an H and Ca only to seed it to a good NLTE start.
model_H_Ca = pw.IsoPromModel(
    "prominence",
    temperature=8000,
    pressure=0.05,
    thickness=0.5e6,
    vturb=5e3,
    altitude=10e6,
    active_atoms=["H", "Ca"],
    Nthreads=Nthreads,
    Nrays=10,
    BcType=pw.ConePromBc,
    bc_provider=bc_provider,
    prd=True,
)
model_H_Ca.iterate_se()

detailed_atoms = ["Mg"]
detailed_pops = {"Mg": np.copy(model.eq_pops["Mg"])}
model_detailed_nlte_Mg = pw.IsoPromModel(
    "prominence",
    temperature=8000,
    pressure=0.05,
    thickness=0.5e6,
    vturb=5e3,
    altitude=10e6,
    active_atoms=active_atoms,
    detailed_atoms=detailed_atoms,
    detailed_pops=detailed_pops,
    Nthreads=Nthreads,
    Nrays=10,
    BcType=pw.ConePromBc,
    bc_provider=bc_provider,
    prd=True,
)
print_mem()
model_detailed_nlte_Mg.eq_pops["H"][...] = model_H_Ca.eq_pops["H"]
model_detailed_nlte_Mg.eq_pops["Ca"][...] = model_H_Ca.eq_pops["Ca"]
model_detailed_nlte_Mg.atmos.ne[:] = model_H_Ca.atmos.ne
model_detailed_nlte_Mg.atmos.nHTot[:] = model_H_Ca.atmos.nHTot
model_detailed_nlte_Mg.iterate_se()

detailed_atoms = ["H"]
model_detailed_lte_H = pw.IsoPromModel(
    "prominence",
    temperature=8000,
    pressure=0.05,
    thickness=0.5e6,
    vturb=5e3,
    altitude=10e6,
    active_atoms=active_atoms,
    detailed_atoms=detailed_atoms,
    Nthreads=Nthreads,
    Nrays=10,
    BcType=pw.ConePromBc,
    bc_provider=bc_provider,
    prd=True,
)
print_mem()
model_detailed_lte_H.iterate_se()

Ivert = model.compute_rays(wavelengths=model.ctx.spect.wavelength, mus=1.0)
Ivert_detailed_nlte_H = model_detailed_nlte_H.compute_rays(mus=1.0)
Ivert_detailed_nlte_H_crd = model_detailed_nlte_H_crd.compute_rays(mus=1.0)
Ivert_detailed_nlte_Mg = model_detailed_nlte_Mg.compute_rays(mus=1.0)
Ivert_detailed_lte_H = model_detailed_lte_H.compute_rays(mus=1.0)

plt.ion()
plt.plot(model.ctx.spect.wavelength, Ivert)
plt.plot(model_detailed_nlte_H.ctx.spect.wavelength, Ivert_detailed_nlte_H)
plt.plot(model_detailed_nlte_H.ctx.spect.wavelength, Ivert_detailed_nlte_H_crd)
plt.plot(model_detailed_nlte_Mg.ctx.spect.wavelength, Ivert_detailed_nlte_Mg)
plt.plot(model_detailed_lte_H.ctx.spect.wavelength, Ivert_detailed_lte_H)
