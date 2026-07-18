"""
core.fragmenter
================

Cheminformatics engine responsible for turning a small-molecule structure
into a set of Hansen Solubility Parameter (HSP) group contributions.

This module is intentionally free of any UI / Streamlit imports. It is the
"Model" layer in the MVC split used by this application: it owns RDKit
parsing, salt stripping, SMARTS-based substructure decomposition and the
atom-balance validation that guarantees every heavy atom in the parent
molecule has been accounted for exactly once.

The group-contribution constants used here follow the spirit of the
Van Krevelen / Hoftyzer scheme (first-order structural groups) extended
with second-order corrections in the style of Stefanis & Panayiotou
(2008), *"A new expanded solubility parameter approach..."*. The numeric
constants shipped in :data:`FIRST_ORDER_GROUPS` and
:data:`SECOND_ORDER_GROUPS` are literature-representative values intended
for a transparent, auditable reference implementation. They are stored as
plain, swappable data so a validated proprietary or extended database can
be substituted without touching any matching logic.

Notes
-----
Rather than hard-coding an atom/hydrogen "recipe" for every group (which
is brittle and a common source of silent accounting bugs), every group's
elemental composition is computed dynamically from the RDKit atoms that
were actually matched. This makes the mass/atom balance check a genuine
structural invariant instead of a second, independently-maintained table
that can drift out of sync with the SMARTS definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Set, Tuple

from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.SaltRemover import SaltRemover


# --------------------------------------------------------------------------- #
# Exceptions
# --------------------------------------------------------------------------- #
class InvalidSMILESError(ValueError):
    """Raised when an input string cannot be parsed into a valid RDKit molecule."""


class HypervalentAtomError(ValueError):
    """Raised when RDKit sanitization flags a chemically invalid (hypervalent) atom."""


class FragmentationError(RuntimeError):
    """Raised when the fragmentation engine cannot complete a structural decomposition."""


# --------------------------------------------------------------------------- #
# Data model
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class GroupDefinition:
    """
    Static definition of a single Hansen group-contribution fragment.

    Parameters
    ----------
    name : str
        Human-readable group label shown in the UI (e.g. ``"Aromatic CH"``).
    smarts : str
        SMARTS pattern used to locate the group in a target molecule. The
        pattern's matched atom indices are treated as the atoms "consumed"
        by this group instance.
    order : int
        ``1`` for first-order (Van Krevelen / Hoftyzer) structural groups,
        ``2`` for second-order (Stefanis-Panayiotou style) corrections such
        as ring or conjugation effects.
    f_d : float
        Dispersion contribution, :math:`F_{di}`, in
        :math:`\\mathrm{MPa^{1/2}\\, cm^{3}\\, mol^{-1}}`.
    f_p : float
        Polar contribution, :math:`F_{pi}`, in
        :math:`\\mathrm{MPa^{1/2}\\, cm^{3}\\, mol^{-1}}`.
    e_h : float
        Hydrogen-bonding cohesive-energy contribution, :math:`E_{hi}`, in
        :math:`\\mathrm{J\\, mol^{-1}}`.
    v_contrib : float
        Molar volume contribution, :math:`V_i`, in
        :math:`\\mathrm{cm^{3}\\, mol^{-1}}`.
    consumes_atoms : bool
        ``True`` if a match "claims" its matched heavy atoms (first-order
        structural groups). ``False`` for second-order corrections, which
        are corrective energy/volume terms layered on top of an
        already-complete first-order decomposition and must **not**
        participate in atom-balance accounting.
    priority : int
        Lower values are matched first. Multi-atom functional groups
        (carboxylic acid, ester, nitrile, ...) must outrank the generic
        aliphatic/aromatic skeleton groups so their atoms are claimed
        before a looser pattern can grab them.
    """

    name: str
    smarts: str
    order: int
    f_d: float
    f_p: float
    e_h: float
    v_contrib: float
    consumes_atoms: bool = True
    priority: int = 100


@dataclass
class MatchedGroup:
    """
    A single realised instance (or aggregated count) of a matched group.

    Attributes
    ----------
    definition : GroupDefinition
        The static group this match corresponds to.
    frequency : int
        Number of non-overlapping times this group was found in the molecule.
    atom_indices : List[Tuple[int, ...]]
        The RDKit atom index tuples claimed by each occurrence.
    heavy_atoms_consumed : int
        Total number of heavy (non-hydrogen) atoms claimed across all
        occurrences of this group.
    hydrogens_consumed : int
        Total number of hydrogens (implicit + explicit) attributed to this
        group across all occurrences.
    """

    definition: GroupDefinition
    frequency: int = 0
    atom_indices: List[Tuple[int, ...]] = field(default_factory=list)
    heavy_atoms_consumed: int = 0
    hydrogens_consumed: int = 0


@dataclass
class FragmentationResult:
    """
    Full output of a :meth:`MoleculeFragmenter.fragment` call.

    Attributes
    ----------
    mol : Chem.Mol
        The (desalted) molecule that was fragmented.
    matched_groups : List[MatchedGroup]
        Every group definition that matched at least once, in priority order.
    unassigned_atom_indices : List[int]
        Heavy-atom indices that no SMARTS pattern was able to claim. A
        non-empty list means the decomposition is scientifically incomplete
        for this structure and results should be treated as provisional.
    total_heavy_atoms : int
        Heavy atom count of the parent molecule (``mol.GetNumHeavyAtoms()``).
    total_hydrogens : int
        Total hydrogen count (implicit + explicit) of the parent molecule.
    assigned_heavy_atoms : int
        Sum of heavy atoms claimed across all matched groups.
    assigned_hydrogens : int
        Sum of hydrogens attributed across all matched groups.
    molecular_formula : str
        RDKit-computed Hill formula of the parent molecule, used as the
        ground truth for the mass/atom balance check.
    is_balanced : bool
        ``True`` if, and only if, every heavy atom and every hydrogen in
        the parent molecule was claimed by exactly one group.
    """

    mol: Chem.Mol
    matched_groups: List[MatchedGroup]
    unassigned_atom_indices: List[int]
    total_heavy_atoms: int
    total_hydrogens: int
    assigned_heavy_atoms: int
    assigned_hydrogens: int
    molecular_formula: str
    is_balanced: bool


# --------------------------------------------------------------------------- #
# First-order structural groups (Van Krevelen / Hoftyzer style)
# --------------------------------------------------------------------------- #
# Fd, Fp : MPa^1/2 . cm3 / mol       Eh : J / mol       V : cm3 / mol
FIRST_ORDER_GROUPS: List[GroupDefinition] = [
    # --- Carboxylic acid / ester / carbonyl family (must precede ether/OH) ---
    # NOTE: Ketone/ester/amide/aldehyde carbonyl carbons use a recursive
    # SMARTS constraint, $(...), to *test* the identity of neighboring
    # atoms without pulling those neighbors into the match tuple. This
    # keeps each functional-group match limited to its own heteroatoms
    # (plus the carbonyl carbon), so a flanking -CH3/-CH2-/aromatic-C is
    # left free to be independently classified as its own skeletal group
    # rather than being silently absorbed into the functional-group match.
    GroupDefinition(
        "Carboxylic acid (-COOH)",
        "[#6X3](=[OX1])[OX2H1]",
        1,
        530.0,
        420.0,
        10_000.0,
        28.5,
        priority=1,
    ),
    GroupDefinition(
        "Carboxylate anion (-COO\u207b)",
        "[#6X3](=[OX1])[OX1-]",
        1,
        530.0,
        420.0,
        9_000.0,
        27.0,
        priority=1,
    ),
    GroupDefinition(
        "Ester (-COO-)",
        "[#6X3;$([#6](=O)[OX2][#6])](=[OX1])[OX2;$([OX2][#6])]",
        1,
        390.0,
        490.0,
        7_000.0,
        18.0,
        priority=2,
    ),
    GroupDefinition(
        "Amide (-CONH-)",
        "[#6X3;$([#6](=O)[#7])]=[OX1]",
        1,
        340.0,
        500.0,
        12_100.0,
        20.0,
        priority=2,
    ),
    GroupDefinition(
        "Amide N-H (aliphatic amide/anilide nitrogen)",
        "[NX3;$([NX3]-[#6]=[OX1])]",
        1,
        20.0,
        0.0,
        0.0,
        4.5,
        priority=2,
    ),
    GroupDefinition(
        "Ketone (>C=O)",
        "[#6X3;$([#6](=O)[#6])]=[OX1]",
        1,
        290.0,
        770.0,
        2_000.0,
        10.8,
        priority=3,
    ),
    GroupDefinition(
        "Aldehyde (-CHO)",
        "[#6X3H1;$([#6H1](=O)[#6])]=[OX1]",
        1,
        470.0,
        800.0,
        4_500.0,
        22.3,
        priority=3,
    ),
    # --- Nitrogen / nitrile / nitro ---
    GroupDefinition(
        "Nitrile (-C#N)", "[CX2]#[NX1]", 1, 430.0, 1_100.0, 2_500.0, 24.0, priority=4
    ),
    GroupDefinition(
        "Nitro (-NO2)",
        "[NX3](=[OX1])=[OX1]",
        1,
        500.0,
        1_070.0,
        1_500.0,
        24.0,
        priority=4,
    ),
    GroupDefinition(
        "Amidine/guanidine carbon (>C=N-)",
        "[#6X3;$([#6](=[NX2])[#7])]=[NX2]",
        1,
        200.0,
        400.0,
        8_000.0,
        15.0,
        priority=4,
    ),
    GroupDefinition(
        "Sulfoxide (>S=O)", "[SX3](=[OX1])", 1, 350.0, 700.0, 4_500.0, 25.0, priority=4
    ),
    GroupDefinition(
        "Sulfone (-SO2-)",
        "[SX4](=[OX1])(=[OX1])",
        1,
        400.0,
        1_100.0,
        3_500.0,
        32.0,
        priority=4,
    ),
    GroupDefinition(
        "Primary amine (-NH2)",
        "[NX3H2;!$(NC=O)]",
        1,
        280.0,
        0.0,
        8_400.0,
        19.2,
        priority=5,
    ),
    GroupDefinition(
        "Secondary amine (>NH)",
        "[NX3H1;!$(NC=O);!R]",
        1,
        160.0,
        210.0,
        3_100.0,
        4.5,
        priority=5,
    ),
    GroupDefinition(
        "Tertiary amine (>N-)",
        "[NX3H0;!$(NC=O);!R]",
        1,
        20.0,
        800.0,
        5_000.0,
        -9.0,
        priority=5,
    ),
    GroupDefinition(
        "Cyclic amine N (ring >N-)",
        "[NX3;R;!$(NC=O)]",
        1,
        50.0,
        600.0,
        4_000.0,
        4.0,
        priority=5,
    ),
    # --- Oxygen: acid/ester patterns already claimed the relevant O atoms ---
    GroupDefinition(
        "Phenolic -OH", "[OX2H1][c]", 1, 350.0, 480.0, 21_000.0, 10.0, priority=6
    ),
    GroupDefinition(
        "Alcohol -OH", "[OX2H1][CX4]", 1, 210.0, 500.0, 20_000.0, 10.0, priority=6
    ),
    GroupDefinition(
        "Ether -O-",
        "[OX2;!$([OX2H1]);!$([OX2][CX3]=[OX1])]",
        1,
        100.0,
        400.0,
        3_000.0,
        3.8,
        priority=7,
    ),
    # --- Halogens ---
    GroupDefinition("Fluorine (-F)", "[FX1]", 1, 220.0, 150.0, 0.0, 18.0, priority=8),
    GroupDefinition(
        "Chlorine (-Cl)", "[ClX1]", 1, 450.0, 550.0, 400.0, 24.0, priority=8
    ),
    GroupDefinition("Bromine (-Br)", "[BrX1]", 1, 550.0, 0.0, 0.0, 30.0, priority=8),
    GroupDefinition("Iodine (-I)", "[IX1]", 1, 425.0, 0.0, 0.0, 31.5, priority=8),
    # --- Sulfur ---
    GroupDefinition(
        "Thioether (-S-)", "[SX2;!$([SX2]=O)]", 1, 440.0, 0.0, 1_500.0, 12.0, priority=8
    ),
    # --- Aromatic carbon skeleton ---
    GroupDefinition("Aromatic CH", "[cH1]", 1, 200.0, 0.0, 0.0, 13.5, priority=10),
    GroupDefinition(
        "Aromatic C (substituted/ring-fusion)",
        "[c;H0]",
        1,
        175.0,
        0.0,
        0.0,
        8.5,
        priority=11,
    ),
    GroupDefinition(
        "Aromatic N (pyridine-type)",
        "[n;H0]",
        1,
        200.0,
        550.0,
        2_500.0,
        10.0,
        priority=10,
    ),
    GroupDefinition(
        "Aromatic N-H (pyrrole-type)",
        "[nH]",
        1,
        200.0,
        300.0,
        13_500.0,
        9.0,
        priority=10,
    ),
    GroupDefinition(
        "Aromatic O (furan-type)", "[o]", 1, 100.0, 400.0, 3_000.0, 7.0, priority=10
    ),
    GroupDefinition(
        "Aromatic S (thiophene-type)", "[s]", 1, 350.0, 200.0, 1_000.0, 9.0, priority=10
    ),
    # --- Aliphatic unsaturation ---
    GroupDefinition(
        "=CH2 (terminal vinylidene)",
        "[CX3H2]=[#6]",
        1,
        400.0,
        0.0,
        0.0,
        28.5,
        priority=12,
    ),
    GroupDefinition(
        "=CH- (vinylene)", "[CX3H1]=[#6]", 1, 200.0, 0.0, 0.0, 13.5, priority=13
    ),
    GroupDefinition(
        "=C< (trisubstituted alkene)",
        "[CX3H0]=[#6]",
        1,
        70.0,
        0.0,
        0.0,
        -5.5,
        priority=14,
    ),
    # --- Aliphatic saturated skeleton (broadest patterns, matched last) ---
    GroupDefinition("-CH3 (methyl)", "[CX4H3]", 1, 420.0, 0.0, 0.0, 33.5, priority=20),
    GroupDefinition(
        "-CH2- (ring methylene)", "[CX4H2;R]", 1, 270.0, 0.0, 0.0, 16.1, priority=21
    ),
    GroupDefinition(
        "-CH2- (methylene)", "[CX4H2;!R]", 1, 270.0, 0.0, 0.0, 16.1, priority=21
    ),
    GroupDefinition(
        ">CH- (ring methine)", "[CX4H1;R]", 1, 80.0, 0.0, 0.0, 10.0, priority=22
    ),
    GroupDefinition(
        ">CH- (methine)", "[CX4H1;!R]", 1, 80.0, 0.0, 0.0, -1.0, priority=22
    ),
    GroupDefinition(
        ">C< (ring quaternary)", "[CX4H0;R]", 1, -70.0, 0.0, 0.0, -19.2, priority=23
    ),
    GroupDefinition(
        ">C< (quaternary)", "[CX4H0;!R]", 1, -70.0, 0.0, 0.0, -19.2, priority=23
    ),
]

# --------------------------------------------------------------------------- #
# Second-order corrections (Stefanis-Panayiotou style, non atom-consuming)
# --------------------------------------------------------------------------- #
SECOND_ORDER_GROUPS: List[GroupDefinition] = [
    GroupDefinition(
        "Ring correction (aliphatic ring closure)",
        "[R]",
        2,
        15.0,
        5.0,
        100.0,
        4.0,
        consumes_atoms=False,
        priority=90,
    ),
    GroupDefinition(
        "Conjugation correction (C=C-C=C)",
        "[#6]=[#6][#6]=[#6]",
        2,
        25.0,
        10.0,
        0.0,
        0.0,
        consumes_atoms=False,
        priority=91,
    ),
    GroupDefinition(
        "Ortho-proximity correction (adjacent polar aromatic substituents)",
        "[c](-[OX2H1,NX3])[c](-[OX2H1,NX3])",
        2,
        0.0,
        -30.0,
        -1_500.0,
        0.0,
        consumes_atoms=False,
        priority=92,
    ),
]


# --------------------------------------------------------------------------- #
# Fragmenter
# --------------------------------------------------------------------------- #
class MoleculeFragmenter:
    """
    Decomposes a parent (salt-stripped) small molecule into HSP groups.

    Parameters
    ----------
    first_order_groups : Sequence[GroupDefinition], optional
        Ordered catalogue of first-order structural groups to attempt to
        match, defaults to :data:`FIRST_ORDER_GROUPS`.
    second_order_groups : Sequence[GroupDefinition], optional
        Catalogue of second-order corrections, defaults to
        :data:`SECOND_ORDER_GROUPS`.

    Examples
    --------
    >>> fragmenter = MoleculeFragmenter()
    >>> mol = fragmenter.parse_smiles("CC(=O)Oc1ccccc1C(=O)O")  # aspirin
    >>> result = fragmenter.fragment(mol)
    >>> result.is_balanced
    True
    """

    def __init__(
        self,
        first_order_groups: Optional[Sequence[GroupDefinition]] = None,
        second_order_groups: Optional[Sequence[GroupDefinition]] = None,
    ) -> None:
        self._first_order = sorted(
            first_order_groups or FIRST_ORDER_GROUPS, key=lambda g: g.priority
        )
        self._second_order = sorted(
            second_order_groups or SECOND_ORDER_GROUPS, key=lambda g: g.priority
        )
        self._salt_remover = SaltRemover()
        # Pre-compile SMARTS patterns once for performance.
        self._compiled_first: List[Tuple[GroupDefinition, Chem.Mol]] = [
            (g, self._compile_smarts(g)) for g in self._first_order
        ]
        self._compiled_second: List[Tuple[GroupDefinition, Chem.Mol]] = [
            (g, self._compile_smarts(g)) for g in self._second_order
        ]

    @staticmethod
    def _compile_smarts(group: GroupDefinition) -> Chem.Mol:
        """Compile a group's SMARTS string into an RDKit query molecule."""
        query = Chem.MolFromSmarts(group.smarts)
        if query is None:
            raise FragmentationError(
                f"Invalid SMARTS pattern for group '{group.name}': {group.smarts!r}"
            )
        return query

    # ------------------------------------------------------------------- #
    # Parsing / preprocessing
    # ------------------------------------------------------------------- #
    def parse_smiles(self, smiles: str, strip_salts: bool = True) -> Chem.Mol:
        """
        Parse a SMILES string into a sanitized, salt-stripped RDKit molecule.

        Parameters
        ----------
        smiles : str
            Raw input SMILES, possibly containing counter-ions/salts
            (e.g. ``"CC(=O)Oc1ccccc1C(=O)O.[Na]"``).
        strip_salts : bool, default True
            If ``True``, remove common salt/solvate fragments and keep the
            largest remaining organic fragment as the parent API.

        Returns
        -------
        Chem.Mol
            The sanitized parent molecule with explicit valence and
            aromaticity perception applied.

        Raises
        ------
        InvalidSMILESError
            If ``smiles`` is empty, malformed, or cannot be parsed by RDKit.
        HypervalentAtomError
            If RDKit's sanitizer rejects the molecule due to an impossible
            valence (e.g. pentavalent carbon).
        """
        if not smiles or not smiles.strip():
            raise InvalidSMILESError("Empty SMILES string was provided.")

        raw_mol = Chem.MolFromSmiles(smiles.strip(), sanitize=False)
        if raw_mol is None:
            raise InvalidSMILESError(f"RDKit could not parse SMILES: {smiles!r}")

        try:
            Chem.SanitizeMol(raw_mol)
        except Chem.rdchem.AtomValenceException as exc:
            raise HypervalentAtomError(
                f"Molecule contains an atom with an invalid (hypervalent) "
                f"valence: {exc}"
            ) from exc
        except Chem.rdchem.KekulizeException as exc:
            raise InvalidSMILESError(
                f"Failed to kekulize aromatic system: {exc}"
            ) from exc
        except Exception as exc:  # pragma: no cover - defensive catch-all
            raise InvalidSMILESError(f"Sanitization failed: {exc}") from exc

        mol = raw_mol
        if strip_salts:
            mol = self._strip_salts(mol)

        if mol.GetNumAtoms() == 0:
            raise InvalidSMILESError(
                "Molecule is empty after salt stripping; no organic parent "
                "fragment remained."
            )

        return mol

    def _strip_salts(self, mol: Chem.Mol) -> Chem.Mol:
        """
        Remove common counter-ions/solvates and keep the largest fragment.

        Combines RDKit's :class:`SaltRemover` (dictionary-based stripping)
        with a fragment-size fallback so that unusual counter-ions not in
        the salt dictionary are still excluded, keeping only the parent
        active pharmaceutical ingredient.

        Parameters
        ----------
        mol : Chem.Mol
            Sanitized, potentially multi-fragment molecule.

        Returns
        -------
        Chem.Mol
            The single largest organic fragment by heavy-atom count.
        """
        try:
            stripped = self._salt_remover.StripMol(mol, dontRemoveEverything=True)
        except Exception:  # pragma: no cover - defensive
            stripped = mol

        fragments = Chem.GetMolFrags(stripped, asMols=True, sanitizeFrags=True)
        if not fragments:
            return stripped
        largest = max(fragments, key=lambda frag: frag.GetNumHeavyAtoms())
        return largest

    # ------------------------------------------------------------------- #
    # Fragmentation
    # ------------------------------------------------------------------- #
    def fragment(self, mol: Chem.Mol) -> FragmentationResult:
        """
        Decompose ``mol`` into non-overlapping first-order groups, then
        layer on second-order corrections.

        Parameters
        ----------
        mol : Chem.Mol
            A sanitized molecule, typically the output of
            :meth:`parse_smiles`.

        Returns
        -------
        FragmentationResult
            Structured decomposition including the atom/hydrogen balance
            check described in :attr:`FragmentationResult.is_balanced`.
        """
        molH = Chem.AddHs(mol)  # noqa: N806 - only used transiently for H accounting
        # We keep indices consistent with the heavy-atom-only `mol` object;
        # AddHs does not renumber existing heavy atoms, so indices carry over.
        claimed: Set[int] = set()
        matched_groups: List[MatchedGroup] = []

        for definition, query in self._compiled_first:
            matches = mol.GetSubstructMatches(query, uniquify=True)
            instance = MatchedGroup(definition=definition)
            for match in matches:
                match_set = set(match)
                if match_set & claimed:
                    continue  # one or more atoms already claimed by a higher-priority group
                claimed |= match_set
                instance.atom_indices.append(match)
                instance.frequency += 1
                heavy_count = len(match_set)
                hydrogen_count = sum(
                    mol.GetAtomWithIdx(idx).GetTotalNumHs() for idx in match_set
                )
                instance.heavy_atoms_consumed += heavy_count
                instance.hydrogens_consumed += hydrogen_count
            if instance.frequency > 0:
                matched_groups.append(instance)

        # Second-order corrections do not claim atoms; they simply count
        # how many times a structural motif (ring, conjugation, proximity)
        # occurs, contributing corrective energy/volume terms.
        for definition, query in self._compiled_second:
            matches = mol.GetSubstructMatches(query, uniquify=True)
            if not matches:
                continue
            instance = MatchedGroup(definition=definition)
            instance.frequency = len(matches)
            instance.atom_indices = list(matches)
            matched_groups.append(instance)

        all_heavy_indices = {atom.GetIdx() for atom in mol.GetAtoms()}
        unassigned = sorted(all_heavy_indices - claimed)

        total_heavy = mol.GetNumHeavyAtoms()
        total_hydrogens = sum(atom.GetTotalNumHs() for atom in mol.GetAtoms())
        assigned_heavy = sum(
            g.heavy_atoms_consumed
            for g in matched_groups
            if g.definition.consumes_atoms
        )
        assigned_hydrogens = sum(
            g.hydrogens_consumed for g in matched_groups if g.definition.consumes_atoms
        )

        is_balanced = (assigned_heavy == total_heavy) and (
            assigned_hydrogens == total_hydrogens
        )

        return FragmentationResult(
            mol=mol,
            matched_groups=matched_groups,
            unassigned_atom_indices=unassigned,
            total_heavy_atoms=total_heavy,
            total_hydrogens=total_hydrogens,
            assigned_heavy_atoms=assigned_heavy,
            assigned_hydrogens=assigned_hydrogens,
            molecular_formula=rdMolDescriptors.CalcMolFormula(mol),
            is_balanced=is_balanced,
        )

    # ------------------------------------------------------------------- #
    # Validation helpers
    # ------------------------------------------------------------------- #
    @staticmethod
    def describe_balance(result: FragmentationResult) -> str:
        """
        Produce a short, human-readable mass-balance summary string.

        Parameters
        ----------
        result : FragmentationResult
            Output of :meth:`fragment`.

        Returns
        -------
        str
            A one-line status message suitable for display in the GUI,
            e.g. ``"Balanced: 21/21 heavy atoms, 21/21 hydrogens assigned."``
            or a warning listing unassigned atom indices.
        """
        if result.is_balanced:
            return (
                f"Balanced: {result.assigned_heavy_atoms}/{result.total_heavy_atoms} "
                f"heavy atoms, {result.assigned_hydrogens}/{result.total_hydrogens} "
                f"hydrogens assigned (formula {result.molecular_formula})."
            )
        return (
            f"UNBALANCED: {result.assigned_heavy_atoms}/{result.total_heavy_atoms} heavy "
            f"atoms and {result.assigned_hydrogens}/{result.total_hydrogens} hydrogens "
            f"assigned. Unassigned atom indices: {result.unassigned_atom_indices}. "
            f"Formula: {result.molecular_formula}."
        )
