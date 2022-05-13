from typing import Optional, Type

import lightweaver as lw
import lightweaver.wittmann as witt
import numpy as np

from .bc_provider import DynamicContextPromBcProvider
from .compute_bc import compute_falc_bc_ctx
from .j_prom_bc import UniformJPromBc
from .prom_bc import PromBc
from .prom_model import PromModel
from .utils import default_atomic_models

class PctrPromModel(PromModel):
    r"""
    Class for prominence with prominence-corona transition region (PCTR)
    simulations. Uses the stratifications first described by Anzer & Heinzel
    (1999), and later adopted by Labrosse et al. in PROM.

    As a function of column mass :math:`m`, with maximum value along the line of
    sight :math:`M`, the pressure and temperature stratifications in the first
    half of the slab :math:`m \in [0, M/2]` are given by

    .. math::
        p(m) = 4 * (p_{cen} - p_{tr}) \frac{m}{M} \left(1 - \frac{m}{M}\right) + p_{tr}\\
        T(m) = T_{cen} + (T_{tr} - T_{cen}) * \left(1 - 4 \frac{m}{M} \left(1 - \frac{m}{M}\right)\right)^\gamma

    This is then reflected in the second half of the slab.

    Parameters
    ----------
    projection : str
        Whether the object is to be treated as a "filament" or a "prominence".
    cen_temperature : float
        The central temperature of the prominence [K].
    cen_pressure : float
        The central pressure of the prominence [Pa].
    tr_temperature : float
        The transition-region temperature of the prominence [K].
    tr_pressure : float
        The transition-region pressure of the prominence [Pa].
    max_cmass : float
        The maximum column mass of the prominence model [kg/m2].
    vturb : float
        The microturbulent velocity inside the prominence [m/s].
    altitude : float
        The altitude of the prominence above the solar surface [m].
    gamma : float
        The slope of the temperature gradient in the PCTR (must be >=2.0).
        Higher values are steeper.
    active_atoms : list of str
        The element names to make "active" i.e. consider in non-LTE.
    atomic_models : list of `lw.AtomicModels`, optional
        The atomic models to use, a default set will be chosen if none are
        specified.
    Ncmass_decades : float, optional
        The number of column mass decades to span in each half of the slab, i.e.
        the value of :math:`log10(M/2) - log10(m_0)`. Default: 6.
    Nhalf_points : int, optional
        The number of points in half of the slab. Default: 100. Total slab size
        will be 2N-1 for symmetry reasons.
    Nrays : int, optional
        The number of Gauss-Legendre angular quadrature rays to use in the
        model. Default: 3. This number will need to be set higher (e.g. 10) if
        using the `ConePromBc`.
    Nthreads : int, optional
        The number of CPU threads to use when solving the radiative transfer
        equations. Default: 1.
    prd : bool, optional
        Whether to consider the effects of partial frequency redistribution.
        Default: False.
    vlos : float, optional
        The z-projected velocity to apply to the prominence model. Default:
        None, i.e. 0.
    vrad : float, optional
        The Doppler-dimming radial velocity to apply. Note that for filaments
        this is the same as `vlos` and that should be used instead. Not fully
        supported in boundary conditions yet, (i.e. you should interpolate the
        wavelength grid first). Default: None, i.e. 0.
    ctx_kwargs : dict, optional
        Extra kwargs to be passed when constructing the Context.
    BcType : Constructor for a type of PromBc, optional
        The base type to be used for constructing the boundary conditions.
        Default: UniformJPromBc.
    bc_kwargs : dict, optional
        Extra kwargs to be passed to the construction of the boundary conditions.
    bc_provider : PromBcProvider, optional
        The provider to use for computing the radiation in the boundary
        conditions. Default: `DynamicContextPromBcProvider` using an
        `lw.Context` configured to match the current model. Note that the
        default is note very performant, but is convenient for experimenting.
        When running a grid of models, consider creating a
        `TabulatedPromBcProvider` using `compute_falc_bc_ctx` and `tabulate_bc`,
        since the default performs quite a few extra RT calculations.
    """
    def __init__(self, projection, cen_temperature, tr_temperature,
                 cen_pressure, tr_pressure, max_cmass, vturb, altitude,
                 gamma,
                 active_atoms, atomic_models=None, Nhalf_points=100,
                 Ncmass_decades=6.0, Nrays=3, prd=False,
                 vlos: Optional[float]=None, vrad: Optional[float]=None,
                 Nthreads=1, ctx_kwargs=None, BcType: Optional[Type[PromBc]]=None,
                 bc_kwargs=None, bc_provider=None):

        self.projection = projection
        if projection not in ["prominence", "filament"]:
            raise ValueError(f"Expected projection ({projection}), to be 'prominence' or 'filament'")

        self.cen_temperature = cen_temperature
        self.tr_temperature = tr_temperature
        self.cen_pressure = cen_pressure
        self.tr_pressure = tr_pressure
        self.max_cmass = max_cmass
        self.vturb = vturb
        self.altitude = altitude
        self.gamma = gamma
        self.Nrays = Nrays
        self.prd = prd
        self.vlos = vlos
        self.vrad = vrad

        if gamma < 2.0:
            raise ValueError("gamma must be >= 2.0")

        if projection == "filament" and vrad is not None and vlos is not None:
            raise ValueError("Cannot set both vrad and vlos for a filament. (Just set one of the two).")

        if projection == "filament" and vrad is not None and vlos is None:
            vlos = vrad
            self.vlos = vrad

        if ctx_kwargs is None:
            ctx_kwargs = {}
        if BcType is None:
            BcType = UniformJPromBc
        if bc_kwargs is None:
            bc_kwargs = {}

        if atomic_models is None:
            atomic_models = default_atomic_models()

        if bc_provider is None:
            vz = None
            if projection == "prominence" and vrad is not None:
                vz = self.vrad

            ctx = compute_falc_bc_ctx(active_atoms=active_atoms, atomic_models=atomic_models, 
                                      prd=self.prd, vz=vz, Nthreads=Nthreads)
            bc_provider = DynamicContextPromBcProvider(ctx)

        log_cmass_half = np.log10(0.5 * max_cmass)
        log_cmass_start = log_cmass_half - Ncmass_decades
        log_cmass_half = np.log10(0.5 * 10**log_cmass_start + 0.5 * max_cmass)
        cmass_half = np.logspace(log_cmass_start, log_cmass_half, Nhalf_points)
        cmass_step = (cmass_half[1:] - cmass_half[:-1])[::-1]
        cmass = np.zeros(2 * Nhalf_points - 1)
        cmass[:Nhalf_points] = cmass_half
        for i in range(Nhalf_points - 1):
            j = i + Nhalf_points
            cmass[j] = cmass[j-1] + cmass_step[i]

        pc = self.cen_pressure - self.tr_pressure
        pressure_half = (4.0 * pc * cmass_half / max_cmass
                         * (1.0 - cmass_half / max_cmass)
                         + self.tr_pressure)
        temperature_half = (self.cen_temperature
                            + (self.tr_temperature - self.cen_temperature)
                              * (1.0 - 4.0 * cmass_half / max_cmass
                                       * (1.0 - cmass_half / max_cmass)
                                )**gamma)
        self.pressure = np.concatenate([pressure_half, pressure_half[:-1][::-1]])
        self.temperature = np.concatenate([temperature_half, temperature_half[:-1][::-1]])
        self.cmass = cmass

        # NOTE(cmo): CGS Starts Here
        pressure_cgs = self.pressure * 10
        eos = witt.Wittmann()
        rho = np.zeros_like(self.temperature)
        ne = np.zeros_like(self.temperature)
        for k in range(rho.shape[0]):
            rho[k] = eos.rho_from_pg(self.temperature[k], pressure_cgs[k])
            ne[k] = eos.pe_from_pg(self.temperature[k], pressure_cgs[k]) / (witt.BK * self.temperature[k])

        rho /= (lw.CM_TO_M**3 / lw.G_TO_KG)
        nHTot = rho / (lw.Amu * lw.DefaultAtomicAbundance.massPerH)
        ne /= lw.CM_TO_M**3
        # NOTE(cmo): CGS Ends Here

        starting_z = np.zeros_like(cmass)
        for k in range(1, starting_z.shape[0]):
            starting_z[k] = starting_z[k-1] + 2.0 * (cmass[k] - cmass[k-1]) / (rho[k] + rho[k-1])

        lower_bc = BcType(projection, bc_provider, altitude, "lower", **bc_kwargs)
        if projection == "prominence":
            upper_bc : Optional[PromBc] = BcType(projection, bc_provider, altitude, "upper", **bc_kwargs)
        else:
            upper_bc = None

        vel = np.zeros_like(cmass) if vlos is None else np.ones_like(cmass) * vlos
        self.atmos = lw.Atmosphere.make_1d(lw.ScaleType.Geometric, depthScale=starting_z,
                                           temperature=self.temperature,
                                           vlos=vel,
                                           vturb=np.ones_like(cmass) * vturb,
                                           ne=ne,
                                           nHTot=nHTot,
                                           lowerBc=lower_bc,
                                           upperBc=upper_bc
                                          )
        atmos = self.atmos
        atmos.quadrature(Nrays)
        self.rad_set = lw.RadiativeSet(atomic_models)
        self.rad_set.set_active(*active_atoms)
        self.eq_pops = self.rad_set.iterate_lte_ne_eq_pops(atmos)

        self.spect = self.rad_set.compute_wavelength_grid()
        hprd = (self.prd and self.vlos is not None)
        if hprd and hprd not in ctx_kwargs:
            ctx_kwargs['hprd'] = hprd
        ctx = lw.Context(self.atmos, self.spect, self.eq_pops,
                         Nthreads=Nthreads, conserveCharge=True,
                         **ctx_kwargs
                         )
        super().__init__(ctx)


    def iterate_se(self, *args, update_model_kwargs:  Optional[dict]=None, **kwargs):
        if update_model_kwargs is None:
            update_model_kwargs = {}

        if self.prd and 'prd' not in kwargs:
            kwargs['prd'] = self.prd

        def update_model(self, printNow, **kwargs):
            # NOTE(cmo): Fix pressure throughout the atmosphere.
            N = (lw.DefaultAtomicAbundance.totalAbundance * self.atmos.nHTot + self.atmos.ne)
            NError = self.pressure / (lw.KBoltzmann * self.temperature) - N
            nHTotCorrection = NError / (lw.DefaultAtomicAbundance.totalAbundance
                                        + self.eq_pops['H'][-1] / self.atmos.nHTot)
            if printNow:
                print(f'    nHTotError: {np.max(np.abs(nHTotCorrection / self.atmos.nHTot))}')
            self.atmos.ne[:] += nHTotCorrection * self.eq_pops['H'][-1] / self.atmos.nHTot
            prevnHTot = np.copy(self.atmos.nHTot)
            self.atmos.nHTot[:] += nHTotCorrection
            if np.any(self.atmos.nHTot < 0.0):
                raise lw.ConvergenceError("nHTot driven negative!")
            nHTotRatio = self.atmos.nHTot / prevnHTot

            for atom in self.rad_set.activeAtoms:
                p = self.eq_pops[atom.element]
                p[...] *= nHTotRatio[None, :]

            # NOTE(cmo): The only iteration difference from the Iso case
            rho = self.atmos.nHTot * lw.DefaultAtomicAbundance.massPerH * lw.Amu
            for k in range(1, rho.shape[0]):
                self.atmos.z[k] = self.atmos.z[k-1] + 2 * ((self.cmass[k] - self.cmass[k-1])
                                                           / (rho[k] + rho[k-1]))
            # to here.
            self.ctx.update_deps(vlos=False, background=False)

        return super().iterate_se(*args, update_model=update_model, update_model_kwargs=update_model_kwargs, **kwargs)
