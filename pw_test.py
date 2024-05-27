import pickle
from os import path

import lightweaver as lw
import matplotlib.pyplot as plt
import numpy as np
import promweaver as pw

active_atoms = ["H", "Ca"]
Nthreads = 8
# NOTE(cmo): String hashes are not stable across python runs. Huh.
# active_atoms_hash = sum(hash(a) for a in active_atoms)
active_atoms_hash = "".join(sorted(active_atoms))

data_path = f"FalcTabulatedBcData_{active_atoms_hash}.pickle"
if path.isfile(data_path):
    with open(data_path, "rb") as pkl:
        bc_table = pickle.load(pkl)
else:
    bc_ctx = pw.compute_falc_bc_ctx(active_atoms, Nthreads=Nthreads)
    bc_table = pw.tabulate_bc(bc_ctx)

    with open(data_path, "wb") as pkl:
        pickle.dump(bc_table, pkl)

    print(f"Made BC data and written to {data_path}")

bc_provider = pw.TabulatedPromBcProvider(**bc_table)

model = pw.IsoPromModel(
    "prominence",
    8000,
    0.05,
    10e6,
    5e3,
    1e7,
    active_atoms=active_atoms,
    Nthreads=Nthreads,
    Nrays=10,
    BcType=pw.ConePromBc,
    bc_provider=bc_provider,
    ctx_kwargs={"ngOptions": lw.NgOptions(2, 5, 10)},
)

modelJ = pw.IsoPromModel(
    "prominence",
    8000,
    0.05,
    10e6,
    5e3,
    1e7,
    active_atoms=active_atoms,
    Nthreads=Nthreads,
    ctx_kwargs={"ngOptions": lw.NgOptions(2, 5, 10)},
    prd=True,
)

plt.ion()

modelJ.iterate_se()
model.iterate_se()
IvertJ = modelJ.compute_rays(mus=1.0)
IvertCtx = model.compute_rays(mus=1.0)

wave = model.ctx.spect.wavelength
plt.plot(wave, IvertJ)
plt.plot(wave, IvertCtx, "--")
