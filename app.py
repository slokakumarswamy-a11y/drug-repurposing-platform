import streamlit as st

st.set_page_config(
    page_title="Drug Repurposing Intelligence Platform",
    page_icon="🧬",
    layout="wide"
)

# =========================================================
# HEADER
# =========================================================

st.title("🧬 Drug Repurposing Intelligence Platform")

st.markdown("""
### AI-powered discovery of new disease opportunities for existing drugs

This platform combines:

- Protein target overlap
- Biological network proximity
- Disease-gene relationships
- AI scientific reasoning

to identify high-potential drug repurposing opportunities.
""")

st.divider()

# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.header("Analysis Settings")

selected_drug = st.sidebar.selectbox(
    "Select Drug",
    [
        "Metformin",
        "Baricitinib",
        "Hydroxychloroquine",
        "Methotrexate",
        "Tocilizumab"
    ]
)

target_weight = st.sidebar.slider(
    "Target Overlap Weight",
    0.0,
    1.0,
    0.35
)

network_weight = st.sidebar.slider(
    "Network Proximity Weight",
    0.0,
    1.0,
    0.40
)

expression_weight = st.sidebar.slider(
    "Expression Reversal Weight",
    0.0,
    1.0,
    0.25
)

run_analysis = st.sidebar.button("Run Repurposing Analysis")

# =========================================================
# MAIN CONTENT
# =========================================================

if run_analysis:

    st.success(f"Analysis complete for {selected_drug}")

    st.subheader("Top Repurposing Opportunities")

    sample_results = [
        {
            "Disease": "Colorectal Cancer",
            "Score": 0.82,
            "Confidence": "HIGH",
            "Evidence": "Phase III"
        },
        {
            "Disease": "Alzheimer's Disease",
            "Score": 0.67,
            "Confidence": "MEDIUM",
            "Evidence": "Observational"
        },
        {
            "Disease": "Type 1 Diabetes",
            "Score": 0.74,
            "Confidence": "HIGH",
            "Evidence": "Phase II"
        }
    ]

    st.dataframe(sample_results, use_container_width=True)

    st.subheader("AI Scientific Rationale")

    st.info(
        f"""
        {selected_drug} demonstrates pathway-level
        biological overlap with multiple inflammatory
        and metabolic diseases.

        Network analysis suggests strong proximity between
        the drug targets and disease-associated genes.
        """
    )

else:
    st.warning("Select a drug and run analysis.")