"""
Update scripts to the snake_case names introduced in Lightweaver 1.0, including
the matching renames in Promweaver (e.g. ``model.iterate_se(popsTol=1e-3)``
becomes ``model.iterate_se(pops_tol=1e-3)``).

Usage::

    python -m promweaver.migrate [--inplace] PATH [PATH ...]

This is `lightweaver.migrate` with Promweaver's renamed keyword arguments
registered too; see its documentation for details. By default a unified diff of
the proposed changes is printed and nothing is modified; ``--inplace`` modifies
the files in place.
"""

import sys

from lightweaver.migrate import main

# NOTE(cmo): Importing promweaver registers its renamed kwargs with lightweaver.migrate.
import promweaver  # noqa: F401

if __name__ == "__main__":
    sys.exit(main())
