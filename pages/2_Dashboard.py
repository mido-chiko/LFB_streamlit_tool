import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk
import colorsys

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(page_title="LFB Dashboard", page_icon="🚒", layout="wide")
#st.sidebar.image("assets/lfb_logo.png", width=100)
#st.logo("assets/lfb_logo.png")
#st.image("assets/lfb_logo.png", width=300)
st.title("London Fire Brigade Incident Records")
st.markdown("Interactive dashboard showing London Fire Brigade incident data.")

# -----------------------------------------------------------------------------
# 2. Data Loading & Preprocessing (Cached for performance)
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_preprocess_data():
    # Load data using pyarrow/fastparquet
    df = pd.read_parquet('models/dashboard_deploy.parquet', engine='fastparquet')

    # Simpler preprocessing that avoids categorical mapping issues
    df_processed = df.copy()

    # Convert date columns
    if 'DateOfCall' in df_processed.columns:
        df_processed['DateOfCall'] = pd.to_datetime(df_processed['DateOfCall'], errors='coerce')
        df_processed['Month'] = df_processed['DateOfCall'].dt.month
        df_processed['DayOfWeek'] = df_processed['DateOfCall'].dt.day_name()

    # Convert potential categorical columns to string to avoid issues
    categorical_cols = ['IncidentGroup', 'StopCodeDescription', 'IncGeo_BoroughName', 'SpecialServiceType', 'DelayCodeId']

    for col in categorical_cols:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype(str)
            df_processed.loc[df_processed[col] == 'nan', col] = 'Unknown'
            df_processed[col] = df_processed[col].fillna('Unknown')

    # Convert British National Grid (Easting/Northing) to Lat/Lon for Mapping
    try:
        from pyproj import Transformer
        # EPSG:27700 is British National Grid, EPSG:4326 is standard WGS84 Lat/Lon
        transformer = Transformer.from_crs("EPSG:27700", "EPSG:4326", always_xy=True)

        # Ensure coordinate columns are numeric
        coords_cols = ['Easting_rounded', 'Northing_rounded', 'Easting', 'Northing']
        for col in coords_cols:
            if col in df_processed.columns:
                df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')

        # Incident Coordinates
        if 'Easting_rounded' in df_processed.columns and 'Northing_rounded' in df_processed.columns:
            df_processed['Inc_Lon'], df_processed['Inc_Lat'] = transformer.transform(
                df_processed['Easting_rounded'].values,
                df_processed['Northing_rounded'].values
            )

        # Station Coordinates
        if 'Easting' in df_processed.columns and 'Northing' in df_processed.columns:
            df_processed['Station_Lon'], df_processed['Station_Lat'] = transformer.transform(
                df_processed['Easting'].values,
                df_processed['Northing'].values
            )
    except ImportError:
        st.warning("Please install 'pyproj' to enable map coordinate transformations.")

    return df_processed

# Load the data
df_processed = load_and_preprocess_data()

# -----------------------------------------------------------------------------
# 3. Sidebar Filters
# -----------------------------------------------------------------------------
st.sidebar.header("Dashboard Filters")

# Extract unique values for filters (excluding 'Unknown')
available_years = sorted(df_processed['CalYear'].dropna().unique())
boroughs = sorted([b for b in df_processed['IncGeo_BoroughName'].unique() if b != 'Unknown'])
incident_groups = sorted([ig for ig in df_processed['IncidentGroup'].unique() if ig != 'Unknown'])

# Year Range Slider
min_year, max_year = int(available_years[0]), int(available_years[-1])
selected_years = st.sidebar.slider(
    "Year Range",
    min_value=min_year,
    max_value=max_year,
    value=(max(min_year, max_year - 3), max_year)
)

# Borough Dropdown
selected_borough = st.sidebar.selectbox("Borough", options=["All Boroughs"] + boroughs)

# Incident Type Dropdown
selected_incident_type = st.sidebar.selectbox("Incident Type", options=["All Types"] + incident_groups)

# Time Period Dropdown
selected_time_period = st.sidebar.selectbox(
    "Time Period",
    options=["All Time", "Last 7 Days", "Last 30 Days", "Last 90 Days", "Last Year"]
)

# -----------------------------------------------------------------------------
# 4. Filter Application Logic
# -----------------------------------------------------------------------------
filtered_df = df_processed.copy()

# Apply Year filter
filtered_df = filtered_df[
    (filtered_df['CalYear'] >= selected_years[0]) &
    (filtered_df['CalYear'] <= selected_years[1])
]

# Apply Borough filter
if selected_borough != "All Boroughs":
    filtered_df = filtered_df[filtered_df['IncGeo_BoroughName'] == selected_borough]

# Apply Incident Type filter
if selected_incident_type != "All Types":
    filtered_df = filtered_df[filtered_df['IncidentGroup'] == selected_incident_type]

# Apply Time Period filter
if selected_time_period != "All Time" and 'DateOfCall' in filtered_df.columns:
    today = pd.Timestamp.now()
    if selected_time_period == "Last 7 Days":
        cutoff_date = today - pd.Timedelta(days=7)
    elif selected_time_period == "Last 30 Days":
        cutoff_date = today - pd.Timedelta(days=30)
    elif selected_time_period == "Last 90 Days":
        cutoff_date = today - pd.Timedelta(days=90)
    elif selected_time_period == "Last Year":
        cutoff_date = today - pd.Timedelta(days=365)

    filtered_df = filtered_df[pd.to_datetime(filtered_df['DateOfCall']) >= cutoff_date]

# -----------------------------------------------------------------------------
# 5. Key Metrics Layout
# -----------------------------------------------------------------------------
total_incidents = len(filtered_df)
fires_count = len(filtered_df[filtered_df['IncidentGroup'] == 'Fire'])
special_services_count = len(filtered_df[filtered_df['IncidentGroup'] == 'Special Service'])
false_alarms_count = len(filtered_df[filtered_df['IncidentGroup'] == 'False Alarm'])

col1, col2, col3, col4 = st.columns(4)
col1.metric("TOTAL INCIDENTS", f"{total_incidents:,}")
col2.metric("FIRES", f"{fires_count:,}")
col3.metric("SPECIAL SERVICES", f"{special_services_count:,}")
col4.metric("FALSE ALARMS", f"{false_alarms_count:,}")

st.divider()

# -----------------------------------------------------------------------------
# 6. Charts Section
# -----------------------------------------------------------------------------
# Row 1 Charts
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("<h5 style='text-align: center; color: #1a3f6c;'>Incidents by Type</h5>", unsafe_allow_html=True)
    incident_type_data = filtered_df['IncidentGroup'].value_counts()
    fig_pie = px.pie(
        values=incident_type_data.values,
        names=incident_type_data.index,
        color=incident_type_data.index,
        color_discrete_map={'Fire': '#dc3545', 'Special Service': '#28a745', 'False Alarm': '#ffc107'}
    )
    fig_pie.update_layout(showlegend=True, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_pie, use_container_width=True)

with c2:
    st.markdown("<h5 style='text-align: center; color: #1a3f6c;'>Incidents by Hour of Day</h5>", unsafe_allow_html=True)
    hourly_data = filtered_df['HourOfCall'].value_counts().sort_index()
    fig_hourly = go.Figure(go.Bar(x=hourly_data.index, y=hourly_data.values, marker_color='#1a3f6c', opacity=0.8))
    fig_hourly.update_layout(xaxis_title='Hour of Day', yaxis_title='Count', margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_hourly, use_container_width=True)

with c3:
    st.markdown("<h5 style='text-align: center; color: #1a3f6c;'>Monthly Trend</h5>", unsafe_allow_html=True)
    monthly_trend = filtered_df.groupby(['CalYear', 'IncidentGroup']).size().reset_index(name='Count')
    fig_trend = px.line(
        monthly_trend,
        x='CalYear',
        y='Count',
        color='IncidentGroup',
        color_discrete_map={'Fire': '#dc3545', 'Special Service': '#28a745', 'False Alarm': '#ffc107'}
    )
    fig_trend.update_layout(xaxis_title='Year', yaxis_title='Count', margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_trend, use_container_width=True)

# Row 2 Charts
c4, c5 = st.columns(2)

with c4:
    st.markdown("<h5 style='text-align: center; color: #1a3f6c;'>Top Boroughs</h5>", unsafe_allow_html=True)
    borough_data = filtered_df['IncGeo_BoroughName'].value_counts().head(10)
    fig_borough = px.bar(
        x=borough_data.values,
        y=borough_data.index,
        orientation='h',
        color=borough_data.values,
        color_continuous_scale='Blues'
    )
    fig_borough.update_layout(xaxis_title='Number of Incidents', yaxis_title='Borough', showlegend=False, margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_borough, use_container_width=True)

with c5:
    st.markdown("<h5 style='text-align: center; color: #1a3f6c;'>Incident Response Times</h5>", unsafe_allow_html=True)
    fig_response = go.Figure()

    if 'FirstPumpArriving_AttendanceTime' in filtered_df.columns:
        filtered_df['ResponseTime'] = pd.to_numeric(filtered_df['FirstPumpArriving_AttendanceTime'], errors='coerce')
        response_times = filtered_df.groupby('IncidentGroup')['ResponseTime'].mean().dropna()

        if not response_times.empty:
            fig_response.add_trace(go.Bar(
                x=response_times.index,
                y=response_times.values,
                marker_color=['#dc3545', '#28a745', '#ffc107']
            ))
        else:
            fig_response.add_annotation(text="No response time data available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)
    else:
        fig_response.add_annotation(text="Response time data not available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False)

    fig_response.update_layout(xaxis_title='Incident Type', yaxis_title='Avg Seconds', margin=dict(t=30, b=0, l=0, r=0))
    st.plotly_chart(fig_response, use_container_width=True)

st.divider()
# -----------------------------------------------------------------------------
# 7. Map Section (Station to Incident Arcs)
# -----------------------------------------------------------------------------
st.markdown("### 🗺️ Incident Response Map")
st.markdown("Visualizing the response path from Stations (Large Blue Dots) to Incidents. Paths are color-coded by **Delay Code**.")

# Check for our transformed coordinate columns
required_map_cols = ['Inc_Lat', 'Inc_Lon', 'Station_Lat', 'Station_Lon']
has_map_cols = all(col in filtered_df.columns for col in required_map_cols)

if has_map_cols:
    # Drop rows with missing coordinates to prevent map rendering errors
    map_data = filtered_df.dropna(subset=required_map_cols).copy()

    if not map_data.empty:
        # Standardize the Delay Code and Distance column names
        delay_col = 'DelayCodeId' if 'DelayCodeId' in map_data.columns else ('delay_code_ID' if 'delay_code_ID' in map_data.columns else None)
        dist_col = 'distance' if 'distance' in map_data.columns else None

        if not delay_col:
            map_data['DelayCodeId'] = 'Unknown'
            delay_col = 'DelayCodeId'

        if not dist_col:
            map_data['distance'] = 0
            dist_col = 'distance'

        # --- NEW DYNAMIC MAP POINT CONTROL ---
        total_map_points = len(map_data)

        # Add a slider right above the map to control performance
        c_slider, c_info = st.columns([1, 2])
        with c_slider:
            max_map_points = st.slider(
                "Max Map Points",
                min_value=100,
                max_value=15000,
                value=5000,
                step=100,
                help="Lower this number if the map feels slow or laggy."
            )

        with c_info:
            # Display information about the current sample size
            st.write("") # Spacer to align with slider
            st.write("")
            if total_map_points > max_map_points:
                st.info(f"Showing a random sample of {max_map_points:,} incidents (out of {total_map_points:,} available).")
                map_data = map_data.sample(max_map_points, random_state=42)
            else:
                st.success(f"Showing all {total_map_points:,} available incidents.")
        # -------------------------------------

        # Dynamically assign colors to unique Delay Codes
        unique_delays = map_data[delay_col].unique()

        def get_line_color(index, total):
            # High transparency for lines (alpha = 60 out of 255)
            h = index / max(total, 1)
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.9)
            return [int(r * 255), int(g * 255), int(b * 255), 60]

        def get_point_color(index, total):
            # Solid color for points (alpha = 255)
            h = index / max(total, 1)
            r, g, b = colorsys.hsv_to_rgb(h, 0.85, 0.9)
            return [int(r * 255), int(g * 255), int(b * 255), 255]

        line_color_map = {code: get_line_color(i, len(unique_delays)) for i, code in enumerate(unique_delays)}
        point_color_map = {code: get_point_color(i, len(unique_delays)) for i, code in enumerate(unique_delays)}

        map_data['line_color'] = map_data[delay_col].map(line_color_map)
        map_data['point_color'] = map_data[delay_col].map(point_color_map)

        # Set the initial view over London
        view_state = pdk.ViewState(
            latitude=map_data['Inc_Lat'].mean(),
            longitude=map_data['Inc_Lon'].mean(),
            zoom=10.5,
            pitch=45,
            bearing=15
        )

        # Layer 1: Bottom Layer - 3D Arcs (Thinner and highly transparent)
        arc_layer = pdk.Layer(
            "ArcLayer",
            map_data,
            get_source_position=["Station_Lon", "Station_Lat"],
            get_target_position=["Inc_Lon", "Inc_Lat"],
            get_source_color="line_color",
            get_target_color="line_color",
            get_width=2,
            pickable=True,
            auto_highlight=True
        )

        # Layer 2: Middle Layer - Incidents
        incident_layer = pdk.Layer(
            "ScatterplotLayer",
            map_data,
            get_position=["Inc_Lon", "Inc_Lat"],
            get_color="point_color",
            get_radius=80,
            radius_min_pixels=3,
            pickable=True
        )

        # Extract unique stations so they don't over-render and become visually heavy
        station_data = map_data[['Station_Lon', 'Station_Lat']].drop_duplicates()

        # Layer 3: Top Layer - Stations (Distinct dark blue styling)
        station_layer = pdk.Layer(
            "ScatterplotLayer",
            station_data,
            get_position=["Station_Lon", "Station_Lat"],
            get_fill_color=[26, 63, 108, 255], # LFB Theme dark blue
            get_line_color=[255, 255, 255, 255], # White border
            stroked=True,
            line_width_min_pixels=2,
            get_radius=200,
            radius_min_pixels=6,
            pickable=False
        )

        # Map Tooltip
        tooltip = {
            "html": f"<b>Delay Code:</b> {{{delay_col}}}<br/><b>Distance:</b> {{{dist_col}}} meters",
            "style": {
                "backgroundColor": "#1a3f6c",
                "color": "white",
                "fontFamily": "sans-serif",
                "fontSize": "13px"
            }
        }

        # Render Map: Notice the layers array order -> [Bottom, Middle, Top]
        r = pdk.Deck(
            layers=[arc_layer, incident_layer, station_layer],
            initial_view_state=view_state,
            tooltip=tooltip,
            map_style='https://basemaps.cartocdn.com/gl/positron-gl-style/style.json'
            #map_style='road'
            #map_style='light'
        )

        st.pydeck_chart(r, use_container_width=True)
    else:
        st.warning("No coordinate data available for the currently selected filters.")
else:
    st.warning("Coordinate columns are missing or the BNG to Lat/Lon transformation failed. Map cannot be rendered.")

st.divider()
# -----------------------------------------------------------------------------
# 8. Data Table
# -----------------------------------------------------------------------------
st.markdown("### Incident Details")

table_columns = ['IncidentNumber', 'DateOfCall', 'IncidentGroup', 'StopCodeDescription', 'IncGeo_BoroughName']
available_columns = [col for col in table_columns if col in filtered_df.columns]

# Streamlit handles pagination natively
st.dataframe(filtered_df[available_columns], use_container_width=True)
