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
    def __init__(self, projection, cen_temperature, tr_temperature,
                 cen_pressure, tr_pressure, max_cmass, vturb, altitude,
                 gamma,
                 active_atoms, atomic_models=None, Nhalf_points=100,
                 Ncmass_decades=6, Nrays=3, prd=False,
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

        return super().iterate_se(*args, **kwargs, update_model=update_model, update_model_kwargs=update_model_kwargs)
