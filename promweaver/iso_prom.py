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


class IsoPromModel(PromModel):
    """
    Class for "Iso" prominence simulations. Iso implies isothermal and isobaric.

    Parameters
    ----------
    projection : str
        Whether the object is to be treated as a "filament" or a "prominence".
    temperature : float
        The temperature of the prominence [K].
    pressure : float
        The pressure of the prominence [Pa].
    thickness : float
        The thickness of the prominence [m].
    vturb : float
        The microturbulent velocity inside the prominence [m/s].
    altitude : float
        The altitude of the prominence above the solar surface [m].
    active_atoms : list of str
        The element names to make "active" i.e. consider in non-LTE.
    atomic_models : list of `lw.AtomicModels`, optional
        The atomic models to use, a default set will be chosen if none are
        specified.
    Nhalf_points : int, optional
        The number of points in half of the slab. Default: 45.
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
    def __init__(self, projection, temperature, pressure, thickness, vturb, altitude,
                 active_atoms, atomic_models=None, Nhalf_points=45, Nrays=3, Nthreads=1, prd=False,
                 vlos: Optional[float]=None, vrad: Optional[float]=None,
                 ctx_kwargs=None, BcType: Optional[Type[PromBc]]=None, bc_kwargs=None, bc_provider=None):

        self.projection = projection
        if projection not in ["prominence", "filament"]:
            raise ValueError(f"Expected projection ({projection}), to be 'prominence' or 'filament'")

        self.temperature = temperature
        self.pressure = pressure
        self.pressure_cgs = pressure * 10
        self.thickness = thickness
        self.vturb = vturb
        self.altitude = altitude
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

        xmod = np.concatenate([(1e-30,), np.logspace(-8, 0, Nhalf_points)])
        z1 = thickness - xmod * 0.5 * thickness
        z2 = xmod[::-1] * 0.5 * thickness
        z = np.concatenate((z1, z2[1:]))
        temp = np.ones_like(z) * temperature

        # NOTE(cmo): CGS Starts Here
        eos = witt.Wittmann()
        rho = np.zeros_like(temp)
        ne = np.zeros_like(temp)
        for k in range(rho.shape[0]):
            rho[k] = eos.rho_from_pg(temperature, self.pressure_cgs)
            ne[k] = eos.pe_from_pg(temperature, self.pressure_cgs) / (witt.BK * temperature)
        nHTot = rho / (lw.CM_TO_M**3 / lw.G_TO_KG) / (lw.Amu * lw.DefaultAtomicAbundance.massPerH)
        ne /= lw.CM_TO_M**3
        # NOTE(cmo): CGS Ends Here

        lower_bc = BcType(projection, bc_provider, altitude, "lower", **bc_kwargs)
        if projection == "prominence":
            upper_bc : Optional[PromBc] = BcType(projection, bc_provider, altitude, "upper", **bc_kwargs)
        else:
            upper_bc = None

        vel = np.zeros_like(z) if vlos is None else np.ones_like(z) * vlos
        self.atmos = lw.Atmosphere.make_1d(lw.ScaleType.Geometric, depthScale=z,
                                           temperature=temp, vlos=vel,
                                           vturb=np.ones_like(z) * vturb,
                                           ne=ne,
                                           nHTot=nHTot,
                                           lowerBc=lower_bc,
                                           upperBc=upper_bc
                                          )
        self.atmos.quadrature(Nrays)

        self.rad_set = lw.RadiativeSet(atomic_models)
        self.rad_set.set_active(*active_atoms)
        self.eq_pops = self.rad_set.iterate_lte_ne_eq_pops(self.atmos)

        self.spect = self.rad_set.compute_wavelength_grid()
        self.Nthreads = Nthreads
        hprd = (self.prd and self.vlos is not None)
        if hprd and hprd not in ctx_kwargs:
            ctx_kwargs['hprd'] = hprd
        ctx = lw.Context(self.atmos, self.spect, self.eq_pops,
                         Nthreads=Nthreads, conserveCharge=True,
                         **ctx_kwargs
                        )
        super().__init__(ctx)


    def iterate_se(self, *args, update_model_kwargs: Optional[dict]=None, **kwargs):
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

            # TODO(cmo): Add condition to not always re-evaluate the line profiles. Maybe on ne change?
            self.ctx.update_deps(vlos=False, background=False)

        return super().iterate_se(*args, update_model=update_model, update_model_kwargs=update_model_kwargs, **kwargs)
