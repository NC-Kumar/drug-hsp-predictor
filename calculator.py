"""
core/calculator.py
Thermodynamic calculations for Hansen Solubility Parameters (HSP).
"""
from typing import List, Dict, Tuple
import numpy as np

class HSPCalculator:
    def __init__(self):
        # Empirical group contribution values (F_d, F_p, E_h, and V_m contributions)
        self.contributions = {
            "Methyl (-CH3)": {"Fd": 420, "Fp": 0, "Eh": 0, "Vm": 33.5},
            "Methylene (-CH2-)": {"Fd": 270, "Fp": 0, "Eh": 0, "Vm": 16.1},
            "Aromatic Carbon (Ar-H)": {"Fd": 190, "Fp": 17, "Eh": 0, "Vm": 12.0},
            "Aromatic Carbon (Substituted)": {"Fd": 210, "Fp": 80, "Eh": 0, "Vm": 15.0},
            "Phenolic Hydroxyl (-OH)": {"Fd": 140, "Fp": 500, "Eh": 14700, "Vm": 12.4},
            "Carboxylic Acid (-COOH)": {"Fd": 530, "Fp": 420, "Eh": 10000, "Vm": 28.5},
            "Amide Group (-CONH-)": {"Fd": 480, "Fp": 470, "Eh": 11300, "Vm": 22.0},
            "Carbonyl (>C=O)": {"Fd": 290, "Fp": 770, "Eh": 2000, "Vm": 10.8},
            "Ester Linkage (-COO-)": {"Fd": 390, "Fp": 490, "Eh": 7000, "Vm": 18.0}
        }

    def calculate_hsp(self, fragments: List[Dict]) -> Tuple[float, float, float, float, float]:
        """
        Applies group contribution equations to find dispersion, polar, and H-bonding targets.
        """
        total_fd = 0.0
        total_fp_sq = 0.0
        total_eh = 0.0
        total_vm = 0.0
        
        # Default baseline if molecule is completely unmapped
        if not fragments:
            return 16.0, 5.0, 5.0, 17.5, 100.0

        for frag in fragments:
            name = frag["Fragment Name"]
            count = frag["Occurrence Count"]
            
            if name in self.contributions:
                vals = self.contributions[name]
                total_fd += vals["Fd"] * count
                total_fp_sq += (vals["Fp"] ** 2) * count
                total_eh += vals["Eh"] * count
                total_vm += vals["Vm"] * count

        # Prevent divide by zero errors
        if total_vm <= 0:
            total_vm = 50.0
            
        # Hansen equations
        dD = total_fd / total_vm
        dP = np.sqrt(total_fp_sq) / total_vm
        dH = np.sqrt(total_eh / total_vm)
        dT = np.sqrt(dD**2 + dP**2 + dH**2)
        
        return dD, dP, dH, dT, total_vm
