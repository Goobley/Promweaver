from typing import Optional, List

import lightweaver as lw
import numpy as np
from weno4 import weno4


class PromBcProvider:
    """
    Abstract base class for providing the intensity to prominence BCs.

    Parameters
    ----------
    specialised_multi_I : bool
        A bool that can be checked by boundary conditions to check if it is
        worth calling compute_multi_I (i.e. there are significant optimisations
        to doing so).
    """

    def __init__(self, specialised_multi_I: bool = False):
        self.specialised_multi_I = specialised_multi_I

    def compute_I(self, wavelength: np.ndarray, mu: Optional[float]):
        """
        Return the intensity at the requested wavelengths and mu. If mu is None,
        return zeros.
        """
        raise NotImplementedError()

    def compute_multi_I(self, wavelength: np.ndarray, mus: List[Optional[float]]):
        """
        Method to `compute_I` for multiple mus. It can be helpful to override
        this if your method has a significant overhead on `compute_I` that can
        be reduced by batching (e.g. this is very much the case for
        `DynamicContextPromBcProvider`).

        Returns
        -------
        I : np.ndarray
            The requested intensity, shape: [wavelength, mu]
        """
        result = np.zeros((wavelength.shape[0], len(mus)))

        for i, mu in enumerate(mus):
            result[:, i] = self.compute_I(wavelength, mu)
        return result


class TabulatedPromBcProvider(PromBcProvider):
    """
    Class for tabulated incident radiation for prominence BC.
    Mu grids are interpolated linearly, wavelength grids using WENO4.

    Parameters
    ----------
    wavelength : array-like
        The wavelengths at which the intensity is defined [nm]
    mu_grid : array-like
        The mus at which the intensity is defined (typically 0.01 - 1.0)
    I : array-like (2D)
        The intensity associated with the above 2 grids [wavelength, mu], SI
        units.
    interp : interpolation function, optional
        The interpolation function to use over mu at each wavelength, if None,
        linear. If a function is provided then it must have the same signature
        as np.interp/weno4
    """

    def __init__(self, wavelength, mu_grid, I, interp=None):
        self.wavelength = np.asarray(wavelength)
        self.mu_grid = np.asarray(mu_grid)
        self.I = np.asarray(I)
        self.interp = interp

        specialised_multi_I = interp is not None
        super().__init__(specialised_multi_I=specialised_multi_I)

    def compute_I(self, wavelength: np.ndarray, mu: Optional[float]):
        if mu is None:
            return np.zeros_like(wavelength)

        if self.interp is not None:
            interp = self.interp
            Igrid = np.zeros(self.I.shape[0])
            for la in range(self.wavelength.shape[0]):
                Igrid[la] = interp(mu, self.mu_grid, self.I[la])
        else:
            frac_mu_idx = weno4(mu, self.mu_grid, np.arange(self.mu_grid.shape[0]))
            int_part = int(frac_mu_idx)
            alpha = frac_mu_idx - int_part

            if alpha <= 0.0:
                Igrid = self.I[:, int_part]
            else:
                Igrid = (1.0 - alpha) * self.I[:, int_part] + alpha * self.I[
                    :, int_part + 1
                ]

        return weno4(wavelength, self.wavelength, Igrid)

    def compute_multi_I(self, wavelength: np.ndarray, mus: List[Optional[float]]):
        if self.interp is None:
            return super().compute_multi_I(wavelength, mus)

        valid_mu_idxs = np.array(
            [i for i, mu in enumerate(mus) if mu is not None], dtype=np.int64
        )
        valid_mus = np.array([mu for mu in mus if mu is not None])

        mu_interp = np.zeros((self.wavelength.shape[0], len(mus)))
        interp = self.interp
        for la in range(self.wavelength.shape[0]):
            mu_interp[la, valid_mu_idxs] = interp(valid_mus, self.mu_grid, self.I[la])

        result = np.zeros((wavelength.shape[0], len(mus)))
        for idx in valid_mu_idxs:
            result[:, idx] = weno4(wavelength, self.wavelength, mu_interp[:, idx])

        return result


class DynamicContextPromBcProvider(PromBcProvider):
    """
    Class for computing the incident radiation from a Lightweaver Context

    Parameters
    ----------
    ctx :  lw.Context
        The Lightweaver Context representing the chromospheric model to take
        into account, ready for `compute_rays` to be called.
    """

    def __init__(self, ctx: lw.Context):
        self.ctx = ctx
        super().__init__(specialised_multi_I=True)

    def compute_I(self, wavelength: np.ndarray, mu: Optional[float]):
        # TODO(cmo): Look at caching this.
        if mu is None:
            return np.zeros_like(wavelength)

        return self.ctx.compute_rays(wavelengths=wavelength, mus=mu)

    def compute_multi_I(self, wavelength: np.ndarray, mus: List[Optional[float]]):
        valid_mu_idxs = np.array(
            [i for i, mu in enumerate(mus) if mu is not None], dtype=np.int64
        )
        valid_mus = np.array([mu for mu in mus if mu is not None])

        result = np.zeros((wavelength.shape[0], len(mus)))
        if valid_mus.shape[0] > 0:
            I = self.ctx.compute_rays(wavelengths=wavelength, mus=valid_mus)
            result[:, valid_mu_idxs] = I
        return result
