# Molecular Hansen Solubility Parameter (HSP) Predictor

An interactive, open-source pharmacinformatics tool designed to predict the Hansen Solubility Parameters (HSP) of active pharmaceutical ingredients (APIs) and drug candidates using group-contribution methods.

Developed to facilitate rational formulation design, solid-state dispersion screenings, and thermodynamic compatibility mappings in pharmaceutical development pipelines.

## 🔬 Core Scientific Methodology

The application breaks down target drug molecules based on structural functional groups using RDKit-driven SMARTS substructure searches. It applies empirical contribution values derived from literature benchmarks (e.g., Stefanis-Panayiotou and Van Krevelen methodologies) to evaluate cohesive energy densities divided into three distinct vectors:

- **Dispersion Forces ($\delta_d$):** Representing non-polar, atomic refractivity vectors.
- **Polar Forces ($\delta_p$):** Representing permanent molecular dipole-dipole moments.
- **Hydrogen-Bonding Energy ($\delta_h$):** Characterizing specific proton donor/acceptor mechanics.

The total solubility parameter is calculated via the geometric relationship:
$$\delta_t = \sqrt{\delta_d^2 + \delta_p^2 + \delta_h^2}$$

## 🛠️ Features & Architecture
- **Multi-Modal Input:** Accepts canonical SMILES syntax or automated remote API resolution of drug names via PubChem query fallbacks.
- **Visual Vector Analytics:** Generates interactive 3D interactive Plotly diagrams charting the target molecule coordinates within the Hansen Space relative to common pharmaceutical solvents and matrices.
- **Audit Logging:** Outputs an itemized structural fragment data frame detailing matched molecular sub-components for transparency and mass balance accounting.

## 🚀 Quick Start & Deployment

### Cloud Execution
Access the live, managed web user interface directly via Streamlit Cloud at: `[PASTE_YOUR_LIVE_STREAMLIT_URL_HERE]`

### Local Installation
To run the web console locally on your workstation, configure a clean Python virtual environment and run the following deployment commands:

```bash
# Clone the repository
git clone [https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git](https://github.com/YOUR_GITHUB_USERNAME/YOUR_REPO_NAME.git)
cd YOUR_REPO_NAME

# Install system and chemical libraries
pip install -r requirements.txt

# Launch local GUI instance
streamlit run app.py
