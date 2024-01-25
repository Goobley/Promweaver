from typing import Optional

import numpy as np
from numpy.polynomial.legendre import leggauss

from .bc_provider import PromBcProvider
from .limb_darkening import outgoing_chromo_ray_mu
from .prom_bc import PromBc


class UniformJPromBc(PromBc):
    r"""
    Simple boundary condition that supplies :math:`J_\nu` at each incoming ray
    and frequency for the boundary condition. This value is uniform over each
    ray. Putting the boundary condition into `final_synthesis` mode by
    requesting a single `compute_rays` from the model, or setting
    `final_synthesis` to true on the `lw.Atmosphere` used in the model samples
    the radiation directly along the ray path, as the converged populations have
    been determined.

    Parameters
    ----------
    projection : str
        filament or prominence
    bc_provider : PromBcProvider
        The provider to be used for computing the necessary intensity values.
    altitude_m : float
        The altitude of the prominence above the solar surface [m]
    prom_upper_lower : str, optional
        Whether this is the 'upper' or 'lower' z-boundary for a prominence (not
        used in filament cases).
    Nrays : int, optional
        The number of Gauss-Legendre rays to be used to compute J. Default: 50.
    """

    def __init__(
        self,
        projection: str,
        bc_provider: PromBcProvider,
        altitude_m: float,
        prom_upper_lower: Optional[str] = None,
        Nrays: int = 50,
    ):
        self.provider = bc_provider
        self.projection = projection
        self.altitude = altitude_m

        if projection not in ["prominence", "filament"]:
            raise ValueError(
                f"Expected projection ({projection}), to be 'prominence' or 'filament'"
            )
        self.projection = projection

        self.bc_type = prom_upper_lower
        self.Nrays = Nrays
        Jmuz, Jwmu = leggauss(Nrays)
        Jmuz = 0.5 * Jmuz + 0.5
        Jwmu *= 0.5
        self.Jmuz = Jmuz
        self.Jwmu = Jwmu

        self.final_synthesis = False

    def update_bc(self, atmos, spect):
        Nmu = atmos.muz.shape[0]
        Nwave = spect.wavelength.shape[0]

        # NOTE(cmo): Final output variable
        Iinterp = np.zeros((Nwave, Nmu, 1))

        final_synth = self.final_synthesis if Nmu != 1 else True
        is_prom = self.projection == "prominence"
        lower = self.bc_type == "lower"
        multi_I = self.provider.specialised_multi_I

        if final_synth:
            mu_ins = []
            for mu_idx in range(Nmu):
                if is_prom:
                    mu_out = atmos.mux[mu_idx]
                else:
                    mu_out = atmos.muz[mu_idx]

                if is_prom:
                    if (lower and mu_out >= 0.0) or ((not lower) and mu_out <= 0.0):
                        # NOTE(cmo): From corona, so zero
                        mu_ins.append(None)
                        continue

                mu_out = abs(mu_out)

                mu_in = outgoing_chromo_ray_mu(mu_out, self.altitude)
                mu_ins.append(mu_in)
                if not multi_I:
                    Iinterp[:, mu_idx, 0] = self.provider.compute_I(
                        spect.wavelength, mu_in
                    )
            if multi_I:
                Iinterp[:, :, 0] = self.provider.compute_multi_I(
                    spect.wavelength, mu_ins
                )
        else:
            J = np.zeros(Nwave)
            if multi_I:
                mu_ins = [outgoing_chromo_ray_mu(mu, self.altitude) for mu in self.Jmuz]
                Is = self.provider.compute_multi_I(spect.wavelength, mu_ins)
                for idx, wmu in enumerate(self.Jwmu):
                    J += Is[:, idx] * wmu
            else:
                for mu, wmu in zip(self.Jmuz, self.Jwmu):
                    mu_in = outgoing_chromo_ray_mu(mu, self.altitude)
                    J += self.provider.compute_I(spect.wavelength, mu_in) * wmu

            Iinterp[:, :, 0] = J[:, None]
            # NOTE(jmj): 0.5 multiplication to account for flipping in radial
            # axis i.e., we don't explicitly consider coronal and so the
            # chromospheric input would be copy-pasted there
            if is_prom:
                Iinterp[:, :, 0] *= 0.5

        self.Iinterp = Iinterp
        self.muz_computed = np.copy(self.muz)
        self.wavelength_computed = np.copy(spect.wavelength)

    def compute_bc(self, atmos, spect):
        any_param_change = False
        try:
            if self.final_synthesis != atmos.final_synthesis:
                self.final_synthesis = atmos.final_synthesis
                any_param_change = True
        except AttributeError:
            if self.final_synthesis:
                self.final_synthesis = False
                any_param_change = True

        try:
            if (
                self.muz.shape[0] != self.muz_computed.shape[0]
                or np.any(self.muz != self.muz_computed)
                or self.wavelength_computed.shape[0] != spect.wavelength.shape[0]
                or np.any(self.wavelength_computed != spect.wavelength)
            ):
                any_param_change = True
        except AttributeError:
            any_param_change = True

        if any_param_change:
            self.update_bc(atmos, spect)

        return self.Iinterp
