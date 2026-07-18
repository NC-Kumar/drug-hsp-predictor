import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# Safely handle RDKit imports to prevent deployment silent drops
try:
    from rdkit import Chem
    from rdkit.Chem import Draw
    RDKIT_AVAILABLE = True
except ImportError:
    RDKIT_AVAILABLE = False

# Absolute imports from your core architecture
try:
    from core.fragmenter import MoleculeFragmenter
    from core.calculator import HSPCalculator
    from utils.api import resolve_name_to_smiles
except ImportError:
    st.error("Engine directory structure missing core files. Please verify 'core' and 'utils' folders exist on GitHub.")

# Page config
st.set_page_config(page_title="HSP Predictor", layout="wide")

st.title("🔬 Molecular Hansen Solubility Parameter Predictor")

if not RDKIT_AVAILABLE:
    st.error("RDKit library failed to load in the cloud environment. Please check your requirements.txt.")
    st.stop()

# Database setup
SOLVENT_DB = {
    "Water": [15.5, 16.0, 42.3],
    "Ethanol": [15.8, 8.8, 19.4],
    "Acetone": [15.5, 10.4, 7.0],
    "Methanol": [15.1, 12.3, 22.3]
}

# Default initial value so the app NEVER starts completely blank
default_smiles = "CC(=O)NC1=CC=C(O)C=C1" # Paracetamol

# Input layout elements
st.sidebar.header("Input Controls")
input_mode = st.sidebar.selectbox("Choose Input Method", ["Use Preset Molecule", "Custom SMILES String"])

if input_mode == "Use Preset Molecule":
    preset_choice = st.sidebar.radio("Select Drug Profile", ["Paracetamol", "Aspirin", "Ibuprofen"])
    presets = {
        "Paracetamol": "CC(=O)NC1=CC=C(O)C=C1",
        "Aspirin": "CC(=O)OC1=CC=CC=C1C(=O)O",
        "Ibuprofen": "CC(C)CC1=CC=C(C(C)C(=O)O)C=C1"
    }
    target_smiles = presets[preset_choice]
else:
    target_smiles = st.sidebar.text_input("SMILES Formula", value=default_smiles)

# Calculation Execution Guard
if target_smiles:
    mol = Chem.MolFromSmiles(target_smiles)
    if mol is None:
        st.sidebar.error("❌ Invalid SMILES sequence entered.")
    else:
        # Run calculations
        frag_engine = MoleculeFragmenter()
        fragments, balance_ok = frag_engine.fragment_molecule(mol)
        
        calc_engine = HSPCalculator()
        dD, dP, dH, dT, v_m = calc_engine.calculate_hsp(fragments)
        
        # Display Row 1 Metrics
        st.subheader("Predicted Hansen Solubility Vector Coordinates")
        m1, m2, m3, m4, m5 = st.columns(5)
        m1.metric("Dispersion (δd)", f"{dD:.2f} MPa⁰.⁵")
        m2.metric("Polar (δp)", f"{dP:.2f} MPa⁰.⁵")
        m3.metric("H-Bonding (δh)", f"{dH:.2f} MPa⁰.⁵")
        m4.metric("Total Parameter (δt)", f"{dT:.2f} MPa⁰.⁵")
        m5.metric("Molar Volume (Vm)", f"{v_m:.1f} cc/mol")
        
        st.markdown("---")
        
        # Display Row 2 Visualization split
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
                marker=dict(size=12, color='crimson'),
                text=["Drug Target"], textposition="top center"
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
