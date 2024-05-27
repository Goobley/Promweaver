import pickle
from os import path

import lightweaver as lw
import matplotlib.pyplot as plt
import numpy as np
import promweaver as pw

active_atoms = ["H", "Ca", "Mg"]
Nthreads = 8
Prd = True

active_atoms_id = f'{"".join(sorted(active_atoms))}_{Prd}'
data_path = f"FalcTabulatedBcData_{active_atoms_id}.pickle"
if path.isfile(data_path):
    with open(data_path, "rb") as pkl:
        bc_table = pickle.load(pkl)
else:
    bc_ctx = pw.compute_falc_bc_ctx(active_atoms, Nthreads=Nthreads, prd=Prd)
    bc_table = pw.tabulate_bc(bc_ctx)

    with open(data_path, "wb") as pkl:
        pickle.dump(bc_table, pkl)

    print(f"Made BC data and written to {data_path}")

bc_provider = pw.TabulatedPromBcProvider(**bc_table)

atmos = np.flip(np.loadtxt("stratification.dat"), axis=0)
z = np.ascontiguousarray((atmos[:, 0] - atmos[-1, 0]) / 100)
temperature = np.ascontiguousarray(atmos[:, 1])
vlos = np.ascontiguousarray(atmos[:, 4] / 100)
pressure = np.ascontiguousarray(atmos[:, 2] / 10)
# ne = np.ascontiguousarray(atmos[:, 5] / 1e-6)

model_crd = pw.StratifiedPromModel(
    "filament",
    z=z,
    temperature=temperature,
    vlos=vlos,
    vturb=np.ones_like(z) * 5e3,
    altitude=10e6,
    pressure=pressure,
    initial_ionisation_fraction=0.5,
    active_atoms=active_atoms,
    Nthreads=Nthreads,
    Nrays=10,
    bc_provider=bc_provider,
    prd=False,
)
model_crd.iterate_se(
)
model = pw.StratifiedPromModel(
    "filament",
    z=z,
    temperature=temperature,
    vlos=vlos,
    vturb=np.ones_like(z) * 5e3,
    altitude=10e6,
    pressure=pressure,
    initial_ionisation_fraction=0.5,
    active_atoms=active_atoms,
    Nthreads=Nthreads,
    Nrays=10,
    BcType=pw.ConePromBc,
    bc_provider=bc_provider,
    prd=Prd,
)
for a in active_atoms:
    model.eq_pops[a][...] = model_crd.eq_pops[a][...]
model.ne[:] = model_crd.ne
model.nh_tot[:] = model_crd.nh_tot
model.iterate_se(
    maxPrdSubIter=5
)
Is = model.compute_rays()
wave = model.ctx.spect.wavelength
plt.ion()
plt.plot(wave, bc_table["I"][:, -1], label="FAL C")
plt.plot(wave, Is, label="Filament")
plt.legend()
