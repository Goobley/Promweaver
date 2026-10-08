# Promweaver
**C. Osborne (University of Glasgow) 2022**

**MIT License**

A simple package demonstrating possible techniques for building on top of [_Lightweaver_](https://github.com/Goobley/Lightweaver), but also helping to bridge the gap between chromospheric NLTE Radiative Transfer codes and those designed for isolated irradiated structures, such as prominences and filaments.
Two of the main models: `IsoPromModel` and `PctrPromModel` are built on the ideas presented in the PROM code, but supporting the modern "Cone" boundary conditions described in Jenkins, Osborne & Keppens (2023). `StratifiedPromModel` provides the infrastructure to do the column-by-column (1.5D) modelling described therein.

### Documentation

The documentation for Promweaver can be found [here](https://goobley.github.io/Promweaver).

### Migrating to Lightweaver 1.0

Promweaver now requires Lightweaver 1.0, which renames its public API to snake_case. Promweaver's `iterate_se` arguments follow suit (e.g. `popsTol` → `pops_tol`, `NmaxIter` → `max_iter`).
The old names still work for now, but emit a `LightweaverDeprecationWarning`.
Scripts can be updated automatically with:

```
python -m promweaver.migrate --inplace my_script.py my_analysis_dir/
```

(omit `--inplace` to preview the changes as a diff). The warnings can be silenced with `lw.silence_deprecations()`.

### Tests

The smoke tests run with `pytest tests` (a few minutes). If a change to the results is intended, regenerate the stored references with `PW_REGEN_REFERENCES=1 pytest tests` and commit `tests/data/references.npz`.

### Acknowledgements

Many thanks for Jack Jenkins and Nic Labrosse for discussing the ideas and implementation of the techniques used in this package.
