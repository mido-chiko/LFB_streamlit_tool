import streamlit as st
import pandas as pd

# -----------------------------------------------------------------------------
# 1. Page Configuration & Logo
# -----------------------------------------------------------------------------
st.set_page_config(page_title="LFB Data Dictionary", layout="wide")
#st.logo("assets/lfb_logo.png")

st.title("Data Dictionary & Feature Definitions")
st.markdown("A comprehensive guide to the operational metrics, spatial coordinates, and engineered features used by the LightGBM predictive model.")
st.divider()

# -----------------------------------------------------------------------------
# 2. Data Dictionary Definition
# -----------------------------------------------------------------------------
@st.cache_data
def load_data_dictionary():
    # Defining the core features used throughout the pipeline
    data = {
        "Feature Name": [
            "Average_AttendanceTimeSeconds",
            "distance",
            "HourOfCall",
            "CalYear / Month / DayOfWeek",
            "DelayCodeId",
            "Easting_m / Northing_m",
            "IncidentGroup",
            "StopCodeDescription",
            "SpecialServiceType",
            "USRN",
            "UPRN",
            "IncGeo_BoroughName"
        ],
        "Type": [
            "Target (Numeric)",
            "Engineered (Numeric)",
            "Temporal (Numeric)",
            "Temporal (Categorical)",
            "Logistical (Categorical)",
            "Geospatial (Numeric)",
            "Categorical",
            "Categorical",
            "Categorical",
            "Geospatial (Categorical)",
            "Geospatial (Categorical)",
            "Categorical"
        ],
        "Definition / Business Logic": [
            "The average time (in seconds) taken for the fire brigade resources to arrive at the scene. This is the primary target variable the machine learning models attempt to predict.",
            "Calculated straight-line distance between the responding Fire Station's coordinates and the Incident's coordinates.",
            "The specific hour of the day (0-23) the emergency call was received. Crucial for capturing rush-hour traffic delays.",
            "Temporal components extracted from the DateOfCall to capture yearly trends, seasonal monthly shifts, and weekend vs. weekday variations.",
            "A specific code indicating if and why a fire engine was delayed en route (e.g., traffic, road closures, difficulty finding the address).",
            "British National Grid (OSGB36) coordinates pinpointing the exact location of the incident.",
            "High-level classification of the emergency. Primarily grouped into: Fire, Special Service, or False Alarm.",
            "A more granular description of the emergency (e.g., 'Secondary Fire', 'AFA', 'Road Traffic Collision').",
            "If the incident is a 'Special Service', this details the specific type of service (e.g., Flooding, Lift Release, Animal Rescue).",
            "Unique Street Reference Number. Identifies the specific street/road of the incident in the UK.",
            "Unique Property Reference Number. Identifies the specific building or property.",
            "The specific London Borough where the incident occurred (e.g., Westminster, Camden, Croydon)."
        ]
    }
    return pd.DataFrame(data)

df_dict = load_data_dictionary()

# -----------------------------------------------------------------------------
# 3. Layout & Presentation
# -----------------------------------------------------------------------------
st.markdown("Core Features & Target Variable")
st.markdown("The following table outlines the most impactful features retained after our high-cardinality dimensionality reduction and correlation filtering.")

# Display the dataframe with custom column widths
st.dataframe(
    df_dict,
    width="stretch",
    hide_index=True,
    column_config={
        "Feature Name": st.column_config.TextColumn("Feature Name", width="medium"),
        "Type": st.column_config.TextColumn("Data Type", width="small"),
        "Definition / Business Logic": st.column_config.TextColumn("Definition / Business Logic", width="large")
    }
)

st.divider()

# -----------------------------------------------------------------------------
# 4. Feature Engineering Context
# -----------------------------------------------------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("Dimensionality Reduction")
    st.info("""
    **Handling High Cardinality:** The raw dataset contained exceptionally dense categorical variables (like specific addresses and ward codes) totaling over **534 raw features** after initial One-Hot Encoding.

    To maintain memory efficiency and prevent the tree-based models from overfitting, a custom preprocessor was built. It evaluated the **Pearson Correlation** of each category against the target response time, retaining only the top statistically significant categories and bundling the rest into an 'Other' classification. This reduced the feature matrix to a highly optimized **269 features**.
    """)

with col2:
    st.markdown("Target Leakage Prevention")
    st.warning("""
    **Post-Incident Variables:** During Phase 1, features such as `PumpMinutesRounded` and `NumPumpsAttending` were removed from the training matrix.

    Because these values represent the *total effort* applied after the engines have already arrived and resolved the fire, including them would cause "Target Leakage"—allowing the model to artificially cheat by looking into the future. Removing them ensures the LightGBM model relies solely on data known *at the exact moment the 999 call is placed*.
    """)
