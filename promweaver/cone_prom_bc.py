from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.polynomial.legendre import leggauss

from .bc_provider import PromBcProvider
from .limb_darkening import outgoing_chromo_ray_mu
from .prom_bc import PromBc


@dataclass
class ConeRadParams:
    """
    Dataclass to compact the parameters to `compute_I_cone` which were getting out of hand.
    """

    wavelength: np.ndarray
    z_mu_cone: np.ndarray
    w_mu_cone: np.ndarray
    phi_nodes: np.ndarray
    w_phi_nodes: np.ndarray
    alt: float
    projection: str
    provider: PromBcProvider


def compute_I_cone(p: ConeRadParams):
    # NOTE(cmo): Plenty of enhancements could be made here. It may be a good
    # idea to hoist the multi_I out a level above, but it could also be become a
    # silly number of rays to ask the Context for in the DynamicBc case.
    # More of the maths could be vectorised too.
    is_prom = p.projection == "prominence"
    multi_I = p.provider.specialised_multi_I
    Iin = np.zeros_like(p.wavelength)
    Icone = np.zeros_like(p.wavelength)

    if multi_I:
        mu_ins = []
        for cone_idx, zmu_c in enumerate(p.z_mu_cone):
            sinTheta = np.sqrt(1.0 - zmu_c**2)  # projection of muz onto the y-x plane
            for phi_ray_idx, phi in enumerate(p.phi_nodes):
                if is_prom:
                    mu_out = (
                        np.cos(phi) * sinTheta
                    )  # Projection of the muOut from the y-x plane to the x axis in the prominence case
                else:
                    mu_out = zmu_c  # Raw zmu in the filament case

                mu_in = outgoing_chromo_ray_mu(
                    mu_out, p.alt
                )  # Find the input ray from the chromosphere based on this muOut
                mu_ins.append(mu_in)

        Is = p.provider.compute_multi_I(p.wavelength, mu_ins)
        for cone_idx, zmu_c in enumerate(p.z_mu_cone):
            Icone[:] = 0.0
            for phi_ray_idx, phi in enumerate(p.phi_nodes):
                Icone += Is[:, phi_ray_idx] * p.w_phi_nodes[phi_ray_idx]

            Iin += Icone * p.w_mu_cone[cone_idx]
            # NOTE(cmo): Trim the batch of Is we just integrated for the next loop iteration
            if Is.shape[1] > p.phi_nodes.shape[0]:
                Is = Is[:, p.phi_nodes.shape[0] :]
        return Iin

    # NOTE(cmo): Non-multi-I case; the default
    for cone_idx, zmu_c in enumerate(p.z_mu_cone):
        sinTheta = np.sqrt(1.0 - zmu_c**2)  # projection of muz onto the y-x plane
        Icone[:] = 0.0
        for phi_ray_idx, phi in enumerate(p.phi_nodes):
            if is_prom:
                mu_out = (
                    np.cos(phi) * sinTheta
                )  # Projection of the muOut from the y-x plane to the x axis in the prominence case
            else:
                mu_out = zmu_c  # Raw zmu in the filament case

            mu_in = outgoing_chromo_ray_mu(
                mu_out, p.alt
            )  # Find the input ray from the chromosphere based on this muOut
            Icone += (
                p.provider.compute_I(p.wavelength, mu_in) * p.w_phi_nodes[phi_ray_idx]
            )

        Iin += (
            Icone * p.w_mu_cone[cone_idx]
        )  # Once the cone has been summed, multiply it by the cone weight from the original GL quadrature set within the prominence model
    return Iin


class ConePromBc(PromBc):
    r"""
    Complex boundary condition that supplies an averaged :math:`I_\nu` distinct
    for each ray of the boundary condition. The primary use case for this is
    better handling complex velocity fields. In the filament view, this is
    simple as mu_z for the filament and the underlying atmosphere are aligned.
    For the prominence case, we instead need to average over 1/4 of a cone (in
    the sense of a polar angle :math:`phi` in the prominence frame) and its
    angles of intersection with the solar surface. This slightly breaks the
    axisymmetric assumptions of one-dimensional models, but the averaging
    ensures that it's correct from an energy standpoint. In `final_synthesis`
    mode (by requesting a single `compute_rays` from the model, or setting
    `final_synthesis` to true on the `lw.Atmosphere` used in the model),
    radiation is directly sampled along the ray path, as the converged
    populations have been determined.

    This style of boundary condition is similar to the Gouttebroze 2005, and
    based on those described in Jenkins, Osborne & Keppens 2022 (in prep.).

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
    Nphi : int, optional
        The number of trapezoidal phi rays to be used in each cone. Default: 50.
    Ncone_rays : int, optional
        The number of Gauss-Legendre rays per cone to supersample the range of
        mus associated with this cone. This prevents sharp changes in intensity
        when one ray (out of few) starts to miss the Sun with increasing
        altitude. Default: 10.
    """

    def __init__(
        self,
        projection: str,
        bc_provider: PromBcProvider,
        altitude_m: float,
        prom_upper_lower: Optional[str] = None,
        Nphi_nodes: int = 50,
        Ncone_rays: int = 10,
    ):
        self.provider = bc_provider
        self.projection = projection
        self.altitude = altitude_m
        if projection not in ["prominence", "filament"]:
            raise ValueError(
                f"Expected projection ({projection}), to be 'prominence' or 'filament'"
            )
        self.projection = projection

        if prom_upper_lower is not None and projection == "prominence":
            if prom_upper_lower not in ["upper", "lower"]:
                raise ValueError(
                    f"Expected prom_upper_lower ({prom_upper_lower}), to be 'upper' or 'lower'"
                )
            self.bc_type = prom_upper_lower
        else:
            self.bc_type = "lower"

        self.Nphi_nodes = Nphi_nodes
        if self.projection == "filament":
            self.Nphi_nodes = 1

        # These are the samplings for the cones i.e., 10 additional rays around the prom GL quadrature
        self.Ncone_rays = Ncone_rays
        self.x_cone, self.w_cone = leggauss(Ncone_rays)
        self.w_cone *= 0.5  # LG weights always sum to two so normalise to 1
        # NOTE(cmo): Final synthesis indicates that we're not worried about the correct J in the boundary condition,
        # and instead need for the ray to sample the atmosphere directly.
        self.final_synthesis = False

        # TODO(cmo): Warn if Nrays is small? Maybe compute an error on J by default?

    def update_bc(self, atmos, spect):
        Nmu = atmos.muz.shape[0]
        Nwave = spect.wavelength.shape[0]

        # NOTE(cmo): Final output variable
        Iinterp = np.zeros((Nwave, Nmu, 1))

        # NOTE(cmo): Trapezoidal integration
        phi_nodes = np.linspace(0, np.pi / 2, self.Nphi_nodes)
        w_phi_nodes = np.ones_like(phi_nodes)
        w_phi_nodes[0] = w_phi_nodes[-1] = 0.5
        w_phi_nodes /= np.sum(w_phi_nodes)

        final_synth = self.final_synthesis if Nmu != 1 else True
        is_prom = self.projection == "prominence"
        lower = self.bc_type == "lower"
        multi_I = self.provider.specialised_multi_I

        mu_ins = []
        for mu_idx in range(Nmu):
            if final_synth:
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
            else:
                # NOTE(cmo): This case is about getting correct J, won't handle negative mux
                # NOTE(jmj): Construct cone quadrature around a single quadrature ray from prominence model
                lower_mu = (
                    0.0
                    if mu_idx == 0
                    else 0.5 * (self.muz[mu_idx - 1] + self.muz[mu_idx])
                )
                upper_mu = (
                    1.0
                    if mu_idx == self.muz.shape[0] - 1
                    else 0.5 * (self.muz[mu_idx] + self.muz[mu_idx + 1])
                )
                cone_half_width = 0.5 * (upper_mu - lower_mu)
                cone_mid = 0.5 * (lower_mu + upper_mu)
                cone_zmu = (
                    cone_half_width * self.x_cone + cone_mid
                )  # xCone only represents x sampling for order NconeRays GL quadrature - then shifted based on bounds of cone
                cone_wmu = self.w_cone  # Weights correspond to the quadrature order

                # NOTE(jmj): Compute the intensity integrated within the cone
                params = ConeRadParams(
                    spect.wavelength,
                    cone_zmu,
                    cone_wmu,
                    phi_nodes,
                    w_phi_nodes,
                    self.altitude,
                    self.projection,
                    self.provider,
                )
                Iinterp[:, mu_idx, 0] = compute_I_cone(params)

                # NOTE(jmj): 0.5 multiplication to account for flipping in radial axis i.e., we don't explicitly consider coronal and so the chromospheric input would be copy-pasted there
                if is_prom:
                    Iinterp[:, mu_idx, 0] *= 0.5

        if final_synth and multi_I:
            Iinterp[:, :, 0] = self.provider.compute_multi_I(spect.wavelength, mu_ins)

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
