"""
core/fragmenter.py
Handles RDKit molecular parsing and SMARTS-based functional group fragmentation.
"""
from rdkit import Chem
from typing import Dict, List, Tuple

class MoleculeFragmenter:
    def __init__(self):
        # Dictionary mapping functional group names to their SMARTS patterns
        # Core fragments based on standard group contribution sets (e.g., Van Krevelen / S-P)
        self.smarts_dict = {
            "Methyl (-CH3)": "[CH3;X4;!R]",
            "Methylene (-CH2-)": "[CH2;X4;!R]",
            "Aromatic Carbon (Ar-H)": "[cH;R1]",
            "Aromatic Carbon (Substituted)": "[c;R1](!=[O,N,S])",
            "Phenolic Hydroxyl (-OH)": "[OH].c",
            "Carboxylic Acid (-COOH)": "C(=O)[O;H1]",
            "Amide Group (-CONH-)": "C(=O)[NH;X3]",
            "Carbonyl (>C=O)": "[CX3]=[OX1]",
            "Ester Linkage (-COO-)": "C(=O)[O;X2;!H1]"
        }

    def fragment_molecule(self, mol: Chem.Mol) -> Tuple[List[Dict], bool]:
        """
        Matches SMARTS patterns against the molecule to find and count functional groups.
        """
        fragments_found = []
        matched_atom_indices = set()
        total_atoms = mol.GetNumAtoms()

        for group_name, smarts in self.smarts_dict.items():
            pattern = Chem.MolFromSmarts(smarts)
            if pattern is None:
                continue
                
            matches = mol.GetSubstructMatches(pattern)
            if matches:
                count = len(matches)
                # Track matched atoms for mass balance verification
                for match in matches:
                    matched_atom_indices.update(match)
                    
                fragments_found.append({
                    "Fragment Name": group_name,
                    "SMARTS Pattern": smarts,
                    "Occurrence Count": count
                })
                
        # If we matched everything, atom balance is OK
        atom_balance_ok = len(matched_atom_indices) >= (total_atoms * 0.7) # Threshold for partial database
        return fragments_found, atom_balance_ok
