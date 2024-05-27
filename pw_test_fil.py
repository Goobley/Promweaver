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
    "filament",
    8000,
    0.05,
    10e6,
    5e3,
    1e7,
    active_atoms=active_atoms,
    Nthreads=12,
    bc_provider=bc_provider,
    ctx_kwargs={"ngOptions": lw.NgOptions(2, 5, 10)},
)
model.iterate_se()
Is = model.compute_rays()
wave = model.ctx.spect.wavelength
plt.ion()
plt.plot(wave, Is)
