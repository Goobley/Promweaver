.. Promweaver documentation master file, created by
   sphinx-quickstart on Fri May 13 12:34:47 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Promweaver's documentation!
=======================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   promweaver-api

Introduction
------------

Promweaver is a package built on the Lightweaver_ framework for modelling the
the non-LTE spectra of filaments and prominences. The ideas are primarily based
on those presented in the PROM [#]_ codes, but the implementation is unique,
leveraging *Lightweaver's* capabilities, allowing for modular boundary
conditions, and general support for multiple atoms, PRD and overlapping
transitions, all from flexible python code.

.. _Lightweaver: https://github.com/Goobley/Lightweaver
.. [#] https://idoc.ias.u-psud.fr/MEDOC/Radiative%20transfer%20codes

Installation
------------

Installation should be simple through PyPI, and the package is pure python, so
should be supported wherever *Lightweaver* is supported.

.. code-block:: bash

   python -m pip install promweaver


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
