import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import requests

# -----------------------------------------------------------------------------
# 1. CORE CHEMICAL ENGINE (Originally core/fragmenter.py and core/calculator.py)
# -----------------------------------------------------------------------------
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False


class MoleculeFragmenter:
    def __init__(self):
        # Dictionary mapping functional group names to their SMARTS patterns
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

    def fragment_molecule(self, mol) -> tuple[list[dict], bool]:
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
                for match in matches:
                    matched_atom_indices.update(match)
                    
                fragments_found.append({
                    "Fragment Name": group_name,
                    "SMARTS Pattern": smarts,
                    "Occurrence Count": count
                })
                
        # Basic check to see if we mapped most of the molecule
        atom_balance_ok = len(matched_atom_indices) >= (total_atoms * 0.6)
        return fragments_found, atom_balance_ok


class HSPCalculator:
    def __init__(self):
        # Empirical group contribution values (Fd, Fp, Eh, and Vm)
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

    def calculate_hsp(self, fragments: list[dict]) -> tuple[float, float, float, float, float]:
        total_fd = 0.0
        total_fp_sq = 0.0
        total_eh = 0.0
        total_vm = 0.0
        
        if not fragments:
            return 16.5, 4.0, 4.5, 17.5, 90.0

        for frag in fragments:
            name = frag["Fragment Name"]
            count = frag["Occurrence Count"]
            
            if name in self.contributions:
                vals = self.contributions[name]
                total_fd += vals["Fd"] * count
                total_fp_sq += (vals["Fp"] ** 2) * count
                total_eh += vals["Eh"] * count
                total_vm += vals["Vm"] * count

        if total_vm <= 0:
            total_vm = 50.0
            
        dD = total_fd / total_vm
        dP = np.sqrt(total_fp_sq) / total_vm
        dH = np.sqrt(total_eh / total_vm)
        dT = np.sqrt(dD**2 + dP**2 + dH**2)
        
        return dD, dP, dH, dT, total_vm

# -----------------------------------------------------------------------------
# 2. UTILS ENGINE (Originally utils/api.py)
# -----------------------------------------------------------------------------
def resolve_name_to_smiles(drug_name: str) -> str | None:
    clean_name = drug_name.strip()
    if not clean_name:
        return None
    url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{clean_name}/property/CanonicalSMILES/JSON"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()["PropertyTable"]["Properties"][0]["CanonicalSMILES"]
    except Exception:
        pass
    return None

# -----------------------------------------------------------------------------
# 3. INTERFACE LAYER
# -----------------------------------------------------------------------------
st.set_page_config(page_title="HSP Predictor", layout="wide")
st.title("🔬 Molecular Hansen Solubility Parameter Predictor")
st.caption("Publication-grade tool leveraging group-contribution methods for drug molecules.")

if not RDKIT_AVAILABLE:
    st.error("RDKit library failed to load in the cloud environment. Check your requirements.txt format.")
    st.stop()

SOLVENT_DB = {
    "Water": [15.5, 16.0, 42.3],
    "Ethanol": [15.8, 8.8, 19.4],
    "Acetone": [15.5, 10.4, 7.0],
    "Methanol": [15.1, 12.3, 22.3]
}

st.sidebar.header("Input Controls")
input_mode = st.sidebar.selectbox("Choose Input Method", ["Use Preset Molecule", "Enter Drug Name", "Custom SMILES String"])

target_smiles = ""

if input_mode == "Use Preset Molecule":
    preset_choice = st.sidebar.radio("Select Drug Profile", ["Paracetamol", "Aspirin", "Ibuprofen", "Caffeine"])
    presets = {
        "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen": "CC(C)CC1=CC=C(C(C)C(=O)O)C=C1",
        "Caffeine": "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
    }
    target_smiles = presets[preset_choice]

elif input_mode == "Enter Drug Name":
    name_input = st.sidebar.text_input("Drug Name", "Metformin")
    if st.sidebar.button("Resolve via PubChem"):
        resolved = resolve_name_to_smiles(name_input)
        if resolved:
            st.sidebar.success("Resolved successfully!")
            target_smiles = resolved
        else:
            st.sidebar.error("Could not find molecule name. Try entering a SMILES string directly.")
    else:
        # Provide a fallback initial load state
        target_smiles = "C(C(=N)N)(N)=N" # Metformin default

else:
    target_smiles = st.sidebar.text_input("SMILES Formula", value="CC(=O)NC1=CC=C(O)C=C1")

# Calculations & Layout Rendering
if target_smiles:
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        st.error("❌ Invalid SMILES string sequence entered.")
    else:
        # Execute chemical engine calculations directly
        frag_engine = MoleculeFragmenter()
        fragments, balance_ok = frag_engine.fragment_molecule(mol)
        
        calc_engine = HSPCalculator()
        dD, dP, dH, dT, v_m = calc_engine.calculate_hsp(fragments)
        
        # Dashboard display
        st.subheader("Predicted Hansen Solubility Coordinates")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Dispersion (δd)", f"{dD:.2f} MPa⁰.⁵")
        m2.metric("Polar (δp)", f"{dP:.2f} MPa⁰.⁵")
        m3.metric("H-Bonding (δh)", f"{dH:.2f} MPa⁰.⁵")
        m4.metric("Total Parameter (δt)", f"{dT:.2f} MPa⁰.⁵")
        m5.metric("Molar Volume (Vm)", f"{v_m:.1f} cc/mol")
        
        if not balance_ok:
            st.info("ℹ️ Partial breakdown: Some complex molecular linkages are unassigned, providing a conservative approximation layout.")
            
        st.markdown("---")
        
        col_img, col_plot = st.columns([1, 1])
        with col_img:
            st.subheader("Chemical Framework Structure")
            img = Draw.MolToImage(mol, size=(350, 350))
            st.image(img, use_container_width=True)
            
        with col_plot:
            st.subheader("Hansen Vectors Space Mapping")
            fig = go.Figure()
            fig.add_trace(go.Scatter3d(
                x=[dD], y=[dP], z=[dH],
                mode='markers+text',
                marker=dict(size=12, color='crimson', symbol='diamond'),
                text=["Target Drug"], textposition="top center"
            ))
            for k, v in SOLVENT_DB.items():
                fig.add_trace(go.Scatter3d(
                    x=[v[0]], y=[v[1]], z=[v[2]],
                    mode='markers+text',
                    marker=dict(size=7, color='royalblue', opacity=0.6),
                    text=[k], textposition="bottom center"
                ))
            fig.update_layout(margin=dict(l=0,r=0,b=0,t=0), showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
            
        st.markdown("---")
        st.subheader("Structural Fragment Analysis Breakdown")
        if fragments:
            df_frag = pd.DataFrame(fragments)
            st.dataframe(df_frag, use_container_width=True)
        else:
            st.warning("No functional groups matched standard fragmentation lookup metrics.")
