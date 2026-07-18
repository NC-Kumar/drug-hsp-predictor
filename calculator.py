"""
core.calculator
================

Thermodynamic engine that turns a :class:`~core.fragmenter.FragmentationResult`
into Hansen Solubility Parameters.

This module is the second half of the "Model" layer: it never touches
RDKit directly and never imports Streamlit. It consumes the structural
decomposition produced by :class:`core.fragmenter.MoleculeFragmenter` and
applies the standard Van Krevelen / Hoftyzer group-contribution relations:

.. math::

    \\delta_d = \\frac{\\sum_i F_{di}}{V_m}
    \\qquad
    \\delta_p = \\frac{\\sqrt{\\sum_i F_{pi}^{2}}}{V_m}
    \\qquad
    \\delta_h = \\sqrt{\\frac{\\sum_i E_{hi}}{V_m}}
    \\qquad
    \\delta_t = \\sqrt{\\delta_d^{2} + \\delta_p^{2} + \\delta_h^{2}}

where :math:`V_m` is the molar volume, obtained by summing every group's
volume contribution :math:`V_i` (first-order groups) plus any second-order
volumetric corrections.

The euclidean distance between the target molecule's HSP coordinate and a
reference solvent/polymer's HSP coordinate in "Hansen space" is exposed via
:meth:`HSPCalculator.hansen_distance`, using the Hansen-weighted metric

.. math::

    R_a = \\sqrt{4(\\delta_{d,1}-\\delta_{d,2})^2 +
                 (\\delta_{p,1}-\\delta_{p,2})^2 +
                 (\\delta_{h,1}-\\delta_{h,2})^2}

which is the standard distance used to judge solvent/polymer miscibility
against a material's Hansen solubility sphere.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from core.fragmenter import FragmentationResult, GroupDefinition, MatchedGroup


class CalculationError(RuntimeError):
    """Raised when HSP quantities cannot be computed from a fragmentation result."""


@dataclass(frozen=True)
class GroupContributionRow:
    """
    One row of the fragment contribution table shown in the UI.

    Attributes
    ----------
    group_name : str
        Display name of the matched group.
    smarts : str
        SMARTS pattern that produced the match.
    order : int
        ``1`` for first-order groups, ``2`` for second-order corrections.
    frequency : int
        Number of non-overlapping occurrences found in the molecule.
    f_d_total : float
        Frequency-weighted dispersion contribution, :math:`n_i \\cdot F_{di}`.
    f_p_total : float
        Frequency-weighted polar contribution, :math:`n_i \\cdot F_{pi}`.
    e_h_total : float
        Frequency-weighted hydrogen-bonding contribution, :math:`n_i \\cdot E_{hi}`.
    v_total : float
        Frequency-weighted molar volume contribution, :math:`n_i \\cdot V_i`.
    """

    group_name: str
    smarts: str
    order: int
    frequency: int
    f_d_total: float
    f_p_total: float
    e_h_total: float
    v_total: float


@dataclass(frozen=True)
class HSPResult:
    """
    Final computed Hansen Solubility Parameters for a molecule.

    Attributes
    ----------
    delta_d : float
        Dispersion parameter, MPa\\ :sup:`1/2`.
    delta_p : float
        Polar parameter, MPa\\ :sup:`1/2`.
    delta_h : float
        Hydrogen-bonding parameter, MPa\\ :sup:`1/2`.
    delta_t : float
        Total (overall) Hansen parameter, MPa\\ :sup:`1/2`.
    molar_volume : float
        Estimated molar volume, cm\\ :sup:`3`/mol.
    contribution_table : List[GroupContributionRow]
        Row-per-group breakdown used to render the interactive fragment
        table in the GUI.
    sum_f_d : float
        Raw sum :math:`\\sum F_{di}` before dividing by :math:`V_m`.
    sum_f_p_sq : float
        Raw sum :math:`\\sum F_{pi}^2` before the square root / division.
    sum_e_h : float
        Raw sum :math:`\\sum E_{hi}` before the square root / division.
    is_mass_balanced : bool
        Passthrough of the fragmenter's atom/hydrogen balance check; when
        ``False`` the HSP estimate is scientifically provisional and the
        GUI must surface a warning.
    """

    delta_d: float
    delta_p: float
    delta_h: float
    delta_t: float
    molar_volume: float
    contribution_table: List[GroupContributionRow]
    sum_f_d: float
    sum_f_p_sq: float
    sum_e_h: float
    is_mass_balanced: bool


# --------------------------------------------------------------------------- #
# Reference Hansen-space coordinates for common pharmaceutical solvents/
# polymers, used by the GUI's 3D Hansen-space scatter plot. Values follow
# the widely tabulated Hansen (2007) *Hansen Solubility Parameters: A
# User's Handbook* reference set.
# --------------------------------------------------------------------------- #
REFERENCE_SOLVENTS: Dict[str, Tuple[float, float, float]] = {
    "Water": (15.5, 16.0, 42.3),
    "Ethanol": (15.8, 8.8, 19.4),
    "Methanol": (14.7, 12.3, 22.3),
    "Acetone": (15.5, 10.4, 7.0),
    "Ethyl acetate": (15.8, 5.3, 7.2),
    "Dimethyl sulfoxide (DMSO)": (18.4, 16.4, 10.2),
    "Polyethylene glycol (PEG)": (17.0, 9.0, 14.0),
    "Polyvinylpyrrolidone (PVP)": (17.5, 8.0, 12.0),
    "Chloroform": (17.8, 3.1, 5.7),
    "Dichloromethane (DCM)": (18.2, 6.3, 6.1),
    "n-Hexane": (14.9, 0.0, 0.0),
    "Propylene glycol": (16.8, 9.4, 23.3),
}


class HSPCalculator:
    """
    Computes Hansen Solubility Parameters from a fragmentation result.

    Examples
    --------
    >>> from core.fragmenter import MoleculeFragmenter
    >>> fragmenter = MoleculeFragmenter()
    >>> mol = fragmenter.parse_smiles("CC(=O)Oc1ccccc1C(=O)O")
    >>> result = fragmenter.fragment(mol)
    >>> calculator = HSPCalculator()
    >>> hsp = calculator.calculate(result)
    >>> round(hsp.delta_t, 1) > 0
    True
    """

    #: Minimum physically sensible molar volume (cm^3/mol) used as a guard
    #: rail against degenerate/erroneous group-contribution sums.
    _MIN_MOLAR_VOLUME: float = 1.0

    def calculate(self, fragmentation: FragmentationResult) -> HSPResult:
        """
        Compute :math:`\\delta_d, \\delta_p, \\delta_h, \\delta_t` and
        :math:`V_m` from a fragmentation result.

        Parameters
        ----------
        fragmentation : FragmentationResult
            Output of :meth:`core.fragmenter.MoleculeFragmenter.fragment`.

        Returns
        -------
        HSPResult
            Fully populated result object, including the per-group
            contribution table for the GUI's data table / bar chart.

        Raises
        ------
        CalculationError
            If no groups were matched at all (nothing to sum), or if the
            resulting molar volume is non-physical (``<= 0``).
        """
        if not fragmentation.matched_groups:
            raise CalculationError(
                "No structural groups were matched; HSP values cannot be "
                "computed for an empty fragmentation."
            )

        contribution_table: List[GroupContributionRow] = []
        sum_f_d = 0.0
        sum_f_p_sq = 0.0
        sum_e_h = 0.0
        sum_v = 0.0

        for group in fragmentation.matched_groups:
            gdef: GroupDefinition = group.definition
            n = group.frequency

            f_d_total = n * gdef.f_d
            f_p_total = n * gdef.f_p
            e_h_total = n * gdef.e_h
            v_total = n * gdef.v_contrib

            sum_f_d += f_d_total
            sum_f_p_sq += n * (gdef.f_p**2)
            sum_e_h += e_h_total
            sum_v += v_total

            contribution_table.append(
                GroupContributionRow(
                    group_name=gdef.name,
                    smarts=gdef.smarts,
                    order=gdef.order,
                    frequency=n,
                    f_d_total=f_d_total,
                    f_p_total=f_p_total,
                    e_h_total=e_h_total,
                    v_total=v_total,
                )
            )

        molar_volume = sum_v
        if molar_volume <= 0:
            raise CalculationError(
                f"Computed molar volume ({molar_volume:.2f} cm^3/mol) is "
                "non-physical (<= 0). Check group volume contributions or "
                "the input structure."
            )
        molar_volume = max(molar_volume, self._MIN_MOLAR_VOLUME)

        delta_d = sum_f_d / molar_volume
        delta_p = math.sqrt(max(sum_f_p_sq, 0.0)) / molar_volume
        delta_h = math.sqrt(max(sum_e_h, 0.0) / molar_volume)
        delta_t = math.sqrt(delta_d**2 + delta_p**2 + delta_h**2)

        return HSPResult(
            delta_d=delta_d,
            delta_p=delta_p,
            delta_h=delta_h,
            delta_t=delta_t,
            molar_volume=molar_volume,
            contribution_table=contribution_table,
            sum_f_d=sum_f_d,
            sum_f_p_sq=sum_f_p_sq,
            sum_e_h=sum_e_h,
            is_mass_balanced=fragmentation.is_balanced,
        )

    # ------------------------------------------------------------------- #
    # Hansen-space comparisons
    # ------------------------------------------------------------------- #
    @staticmethod
    def hansen_distance(
        hsp_a: Tuple[float, float, float], hsp_b: Tuple[float, float, float]
    ) -> float:
        """
        Compute the Hansen-weighted distance :math:`R_a` between two HSP
        coordinates in (:math:`\\delta_d, \\delta_p, \\delta_h`) space.

        Parameters
        ----------
        hsp_a : Tuple[float, float, float]
            ``(delta_d, delta_p, delta_h)`` for the first material.
        hsp_b : Tuple[float, float, float]
            ``(delta_d, delta_p, delta_h)`` for the second material.

        Returns
        -------
        float
            The Hansen distance :math:`R_a`, in MPa\\ :sup:`1/2`. The
            dispersion axis is conventionally weighted by a factor of 4 to
            reflect its different physical origin and scale relative to
            the polar and hydrogen-bonding axes.
        """
        d1, p1, h1 = hsp_a
        d2, p2, h2 = hsp_b
        return math.sqrt(4.0 * (d1 - d2) ** 2 + (p1 - p2) ** 2 + (h1 - h2) ** 2)

    @classmethod
    def rank_reference_solvents(
        cls,
        target: HSPResult,
        reference_solvents: Optional[Dict[str, Tuple[float, float, float]]] = None,
    ) -> List[Tuple[str, float, Tuple[float, float, float]]]:
        """
        Rank reference solvents/polymers by Hansen distance to a target.

        Parameters
        ----------
        target : HSPResult
            The computed HSP of the drug molecule of interest.
        reference_solvents : Dict[str, Tuple[float, float, float]], optional
            Mapping of solvent/polymer name to its ``(delta_d, delta_p,
            delta_h)`` coordinate. Defaults to :data:`REFERENCE_SOLVENTS`.

        Returns
        -------
        List[Tuple[str, float, Tuple[float, float, float]]]
            ``(name, hansen_distance, coordinate)`` tuples sorted from most
            to least compatible (ascending distance).
        """
        solvents = reference_solvents or REFERENCE_SOLVENTS
        target_coord = (target.delta_d, target.delta_p, target.delta_h)
        ranked = [
            (name, cls.hansen_distance(target_coord, coord), coord)
            for name, coord in solvents.items()
        ]
        ranked.sort(key=lambda row: row[1])
        return ranked
