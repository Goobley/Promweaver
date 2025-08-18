from dataclasses import dataclass
from typing import Generator, List, Optional, Tuple

import astropy.units as u
import lightweaver as lw
import numpy as np

from lightweaver.atomic_model import AtomicModel
from lightweaver.rh_atoms import H_atom
from numpy.polynomial import Polynomial

def consume_txt_file(path) -> Generator[str]:
    """
    Yields a file line-by-line. The lines are stripped and empty lines are
    skipped.
    """
    with open(path, "r") as f:
        data = [l.strip() for l in f.readlines() if l.strip() != ""]
    for d in data:
        yield d

def line_index_to_ij(idx: int) -> Tuple[int, int]:
    """
    Converts from "column major" single line index to i, j. Assumes 0 based
    indexing. Raises ValueError if line index not found.
    """
    max_level = 30
    for l in range(1, max_level):
        lower_bracket = (l - 1) * l / 2
        upper_bracket = l * (l + 1) / 2
        if lower_bracket <= idx < upper_bracket:
            j = l
            i = int(idx - lower_bracket)
            return i, j
    raise ValueError(f"No matching line index found. Searched all combinations < n={max_level}")

@dataclass
class LineIncidentIntensityData:
    """Stores the result of loading a line \"intinc.dat\". By default, this will contain a half-profile."""
    i: int
    """Lower transition index"""
    j: int
    """Upper transition index"""
    limb_darkening: Polynomial
    """Polynomial for limb darkening as a function of mu. Ignore if `limb_darkening.coef[0] == 0.0`."""
    delta_lambda: np.ndarray
    """Wavelengths of the intensity samples [nm]"""
    intensity: np.ndarray
    """Intensity samples [W/m2/Hz/sr]"""


def parse_intinc(path: Optional[str]=None) -> List[LineIncidentIntensityData]:
    """
    Load an `intinc.dat` file in the form used by MALI (P. Heinzel) and CYMA2DV
    (P. Gouttebroze), returning a list of the lines described therein.

    Parameters
    ----------
    path: str, optional
        The location to load the intinc.dat. If not provided `"intinc.dat"` will be tried.

    Returns
    -------
    line_data: list of LineIncidentIntensityData
        A list of all the lines and their associated incident intensities loaded from the file.
    """
    if path is None:
        path = "intinc.dat"
    data = consume_txt_file(path)
    num_lines_in_file = int(next(data))

    line_data = []
    for kr in range(num_lines_in_file):
        line_index = int(next(data))
        d = next(data).split()
        num_freq, int_coeff = int(d[0]), float(d[1])
        poly_coeffs = np.array([float(d) for d in next(data).split()])
        wave_offset = np.zeros(num_freq)
        intensity = np.zeros(num_freq)
        for f in range(num_freq):
            wave_offset[f], intensity[f] = [float(d) for d in next(data).split()]
        i, j = line_index_to_ij(line_index - 1)
        line_data.append(
            LineIncidentIntensityData(
                i=i,
                j=j,
                limb_darkening=Polynomial(poly_coeffs),
                delta_lambda=wave_offset * 0.1,
                intensity=((int_coeff * intensity) << u.Unit("erg/(cm2 s Hz sr)")).to("W/(m2 Hz sr)").value,
            )
        )
    return line_data

@dataclass
class BrightnessTemperatures:
    """
    Storage for the result of the `tembri.dat` file.
    """
    wavelength: np.ndarray
    """nm"""
    temperature: np.ndarray
    """K"""

def parse_tembri(path=None):
    """
    Load a `tembri.dat` file in the form used by MALI (P. Heinzel) and CYMA2DV
    (P. Gouttebroze), returning a wavelength varying set of brightness
    temperatures for the continuum.

    Parameters
    ----------
    path: str, optional
        The location to load the tembri.dat. If not provided `"tembri.dat"` will be tried.

    Returns
    -------
    tembri: BrightnessTemperatures
        Wavelength-varying brightness temperatures for continuum emission.
    """
    if path is None:
        path = "tembri.dat"
    tembri = np.genfromtxt(path)
    return BrightnessTemperatures(
        wavelength=1e3 * np.ascontiguousarray(tembri[:, 0]),
        temperature=np.ascontiguousarray(tembri[:, 1])
    )

def tabulate_intinc_boundary(
        wavelength: np.ndarray,
        H_model: Optional[AtomicModel]=None,
        mu_grid: Optional[np.ndarray]=None,
        intinc_path: Optional[str]=None,
        tembri_path: Optional[str]=None,
    ) -> dict:
    """
    Provide input for `TabulatedPromBcProvider` using data in the form consumed
    by MALI (P. Heinzel), and CYMA2DV (P. Gouttebroze). Currently only
    compatible with H.

    N.B. Due to licensing, the necessary data files can't be distributed with
    Promweaver. The data files are distributed here under GPLv3:
    <https://idoc.osups.universite-paris-saclay.fr/medoc/tools/radiative-transfer-codes/cyma2dv/>
    (doi: 10.48326/idoc.medoc.radtransfer.cyma2dv). The download does not
    currently produce a usable archive (there is a null byte prepended to the
    file), but this can be fixed:

    .. code-block::

        tail -c +2 CYMA2DV_Licence_GPL.tgz > cyma2dv_fixed.tar.gz
        tar xvzf cyma2dv_fixed.tar.gz

    gzip will complain about an unexpected end to the file, but it will have
    successfully extracted the data.


    Parameters
    ----------
    wavelength : array-like
        The wavelength grid to use for the boundary condition table
    H_model: AtomicModel, optional
        The model H atom to use. If not provided, the 9-level lightweaver
        `H_atom` is used. It is important that this match the model employed in
        your simulation or the irradiation could be slightly shifted in
        wavelength, causing Doppler dimming and brightening effects.
    mu_grid : array-like, optional
        The grid of viewing angles (in the form of mu) to be used for the
        boundary condition table. Default: `np.linspace(0.01, 1.0, 100)`.
    intinc_path: str, optional
        Path to `intinc.dat`. Default "intinc.dat".
    tembri_path: str, optional
        Path to `tembri.dat`. Default "tembri.dat".

    Returns
    -------
    data : dict
        Dict with keys 'wavelength', 'mu_grid' and 'I' that can be splatted
        directly into a `TabulatedPromBcProvider` or pickled.
    """
    if mu_grid is None:
        mu_grid = np.linspace(0.01, 1.0, 100)
    if H_model is None:
        H_model = H_atom

    H = H_model()

    def line_in_model(i, j):
        for line in H.lines:
            if line.i == i and line.j == j:
                return True
        return False

    def mirror_line(l):
        return LineIncidentIntensityData(
            i=l.i,
            j=l.j,
            limb_darkening=l.limb_darkening,
            delta_lambda=np.concatenate([-l.delta_lambda[::-1], l.delta_lambda]),
            intensity=np.concatenate([l.intensity[::-1], l.intensity]),
        )

    def lambda0(i, j):
        for line in H.lines:
            if line.i == i and line.j == j:
                return line.lambda0
        return 0.0


    lines = parse_intinc(path=intinc_path)
    tembri = parse_tembri(path=tembri_path)

    lines = [mirror_line(l) for l in lines if line_in_model(l.i, l.j)]
    lambda0s = [lambda0(l.i, l.j) for l in lines]
    line_ranges = [(centre + l.delta_lambda[0], centre+l.delta_lambda[-1]) for centre, l in zip(lambda0s, lines)]

    def in_range(wave):
        for i, (rs, re) in enumerate(line_ranges):
            if rs <= wave <= re:
                return i
        raise ValueError("Not in lines")


    result = np.zeros((wavelength.shape[0], mu_grid.shape[0]))
    for la, wave in enumerate(wavelength):
        try:
            line_idx = in_range(wave)
            line = lines[line_idx]
            if line.limb_darkening.coef[0] == 0.0:
                # NOTE(cmo): No variation with angle
                limb_darkening = 1.0
            else:
                limb_darkening = line.limb_darkening(mu_grid)

            wavelength_sample = np.interp(wave, lambda0s[line_idx] + line.delta_lambda, line.intensity)
            result[la, :] = wavelength_sample * limb_darkening
        except ValueError:
            temperature = np.interp(wave, tembri.wavelength, tembri.temperature)
            result[la, :] = lw.planck(temperature, wave)
    return {
        "wavelength": wavelength,
        "mu_grid": mu_grid,
        "I": result,
    }
