import streamlit as st
import pandas as pd
from scipy.spatial import cKDTree
import numpy as np
import geopandas as gpd
import folium
from streamlit_folium import folium_static
from shapely.geometry import Point

# Define required columns
required_columns = [
    "sn", "unique_code", "state_name", "lga_name", "ward_name", "take_off_point",
    "settlement_name", "primary_settlement_name", "alternate_name", "latitude", "longitude",
    "security_compromised", "accessibility_status", "reasons_for_inaccessibility",
    "habitational_status", "set_population", "set_target", "number_of_household",
    "noncompliant_household", "team_code", "day_of_activity", "urban", "rural",
    "scattered", "highrisk", "slums", "densely_populated", "hard2reach", "border",
    "nomadic", "riverine", "fulani", "source", "comments", "globalid", "validation_status"
]

# ✅ Summary of Predefined Categorical Values
validations = {
    "security_compromised": ["Y", "N"],
    "habitational_status": ["Inhabited", "Partially Inhabited", "Abandoned", "Migrated"],
    "accessibility_status": ["Fully Accessible", "Inaccessible", "Partially Inaccessible"],
    "validation_status": ["Validated", "Not validated", "Validation ongoing"],
    "slums": ["Y", "N"], "highrisk": ["Y", "N"], "urban": ["Y", "N"], "rural": ["Y", "N"],
    "scattered": ["Y", "N"], "fulani": ["Y", "N"], "riverine": ["Y", "N"], "nomadic": ["Y", "N"],
    "border": ["Y", "N"], "hard2reach": ["Y", "N"], "densely_populated": ["Y", "N"]
}

# Numeric validation rules
numeric_columns = ["set_population", "set_target", "number_of_household", "noncompliant_household"]

# Streamlit UI Config
st.set_page_config(page_title="✅MLOS Quality Check Dashboard", layout="wide")


st.markdown("""
    <center>
        <h1>✅ MLOS Quality Check Dashboard</h1>
        <p>📌 Perform automated data validation for MLOS datasets.</p>
        <hr>
    </center>
""", unsafe_allow_html=True)

# ✅ Initialize df before checking file upload to avoid 'NameError'
df = pd.DataFrame()

# File Upload Section
st.markdown("#### 📂 Upload Your MLOS File Below:")
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=["csv", "xlsx"])

if uploaded_file:
    st.success("✅ Upload successful!")

    # Read file
    file_extension = uploaded_file.name.split(".")[-1]
    df = pd.read_csv(uploaded_file, encoding="latin1") if file_extension == "csv" else pd.read_excel(uploaded_file, engine="openpyxl")
    # Convert numeric columns silently
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    # Preview dataset
    st.markdown("### 🔎 Data Preview")
    st.dataframe(df.head())
    st.write("---")

    # Schema Validation
    st.markdown("### 🔎 Schema Validation")
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        st.error(f"❌ {len(missing_columns)} Required Columns are Missing!")
        st.info("🛠️ **Fix:** Ensure all required columns are included in the dataset.")
        st.write(", ".join(missing_columns))
    else:
        st.success("✅ All required columns are present.")

    st.write("---")

    # ✅ Ensure required columns exist before applying QC checks
    total_rows = df.shape[0]
    total_columns = df.shape[1]

    # ✅ Check for duplicate unique identifiers
    uniquecode_duplicates = 0  # Initialize default value

    if "unique_code" in df.columns:
        uniquecode_duplicates = df.duplicated(subset=["unique_code"], keep=False).sum()

    # ✅ Exclude rows where both latitude and longitude are exactly 0
    stacked_geolocations = 0
    if "latitude" in df.columns and "longitude" in df.columns:
        filtered_df = df[~((df["latitude"] == 0) & (df["longitude"] == 0))]
        stacked_geolocations = filtered_df.duplicated(subset=["latitude", "longitude"], keep=False).sum()

    # ✅ Validation status errors
    validation_issues = 0

    if "validation_status" in df.columns:
        validation_issues = df[~df["validation_status"].isin(validations["validation_status"])].shape[0]

    # ✅ Missing values in source and globalid
    missing_source = df["source"].isnull().sum() if "source" in df.columns else 0
    missing_globalid = df["globalid"].isnull().sum() if "globalid" in df.columns else 0

    # ✅ Check for duplicate unique identifiers
    globalid_duplicates = 0  # Initialize default value

    if "globalid_code" in df.columns:
        globalid_duplicates = df.duplicated(subset=["globalid_code"], keep=False).sum()

    # ✅ Percentage correctness (Corrected to count valid cells)
    total_cells = df.size  # Total number of cells (rows * columns)

    # Create a mask to identify valid cells
    valid_mask = pd.DataFrame(True, index=df.index, columns=df.columns)

    # 1. Check for missing values
    valid_mask = valid_mask & ~df.isnull()

    # 2. Check for negative numeric values
    for col in numeric_columns:
        if col in df.columns:
            valid_mask[col] = valid_mask[col] & (df[col] >= 0)

    # 3. Check for categorical validation
    for col, valid_values in validations.items():
        if col in df.columns:
            valid_mask[col] = valid_mask[col] & df[col].isin(valid_values)

    # 4. Check for geolocation validity (excluding (0,0))
    if "latitude" in df.columns and "longitude" in df.columns:
        valid_mask["latitude"] = valid_mask["latitude"] & (df["latitude"] != 0)
        valid_mask["longitude"] = valid_mask["longitude"] & (df["longitude"] != 0)

    # 5. Ensure 'set_target' is not greater than 'set_population'
    if "set_population" in df.columns and "set_target" in df.columns:
        valid_mask["set_target"] = valid_mask["set_target"] & (df["set_target"] <= df["set_population"])

    # 6. Check for duplicate geolocations (excluding (0,0))
    if "latitude" in df.columns and "longitude" in df.columns:
        non_zero_geos = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
        duplicate_mask = non_zero_geos.duplicated(subset=["latitude", "longitude"], keep=False)
        valid_mask.loc[non_zero_geos.index, "latitude"] = valid_mask.loc[non_zero_geos.index, "latitude"] & ~duplicate_mask
        valid_mask.loc[non_zero_geos.index, "longitude"] = valid_mask.loc[non_zero_geos.index, "longitude"] & ~duplicate_mask

    # Count valid cells
    valid_cells = valid_mask.sum().sum()

    # Calculate percentage correctness
    percentage_correctness = (valid_cells / total_cells) * 100 if total_cells > 0 else 0

    # ✅ Display the metric
    st.metric("✅ Data Correctness Percentage", f"{percentage_correctness:.2f}%")

    # ✅ Summary Section
    st.markdown("### 📊 Data Summary")
    col1, col2, col3, col4 = st.columns(4)
    col5, col6, col7, col8 = st.columns(4)

    col1.metric("Total Rows", total_rows)
    col2.metric("Total Columns", total_columns)
    col3.metric("Uniquecode Duplicates", uniquecode_duplicates)
    col4.metric("Stacked Geolocation", stacked_geolocations)
    col5.metric("Wrong Validation Status", validation_issues)
    col6.metric("Null values in 'Source'", missing_source)
    col7.metric("Null values in 'GlobalID'", missing_globalid)
    col8.metric("Duplicates GlobalID", globalid_duplicates)

    st.write("---")

    # ✅ Categorical Validation
    st.markdown("### 🔎 Categorical Validation")
    for col, valid_values in validations.items():
        if col in df.columns:
            invalid_count = (~df[col].isin(valid_values)).sum()
            if invalid_count > 0:
                st.error(f"❌ `{col}` has {invalid_count} invalid values!")
                st.info(f"🛠️ **Fix:** Ensure '{col}' values match predefined valid options: {valid_values}")
            else:
                st.success(f"✅ `{col}` values are valid.")

# ✅ Numeric Validation (Runs Only If Data Exists)
if not df.empty:  # Ensure data is uploaded before running validation
    st.markdown("### 🔢 Numeric Validation")

    # 1️⃣ Check for Null and Negative Values
    if any(col in df.columns for col in numeric_columns):
        null_values = df[numeric_columns].isnull().sum()
        negative_values = df[numeric_columns].lt(0).sum()

        if null_values.sum() > 0:
            st.error("❌ Some numeric columns contain missing (null) values!")
            st.write(null_values)
            st.info("🛠 **Fix:** Ensure all numeric values are provided.")

        if negative_values.sum() > 0:
            st.error("❌ Some numeric columns contain negative values!")
            st.write(negative_values)
            st.info("🛠 **Fix:** Ensure all numeric values are non-negative.")

        if null_values.sum() == 0 and negative_values.sum() == 0:
            st.success("✅ No missing or negative values found in numeric columns(set_target,set_population,number_of_household,noncompliant_household)")

    else:
        st.warning("⚠️ No valid numeric columns found in the dataset!")
        st.info("🛠 **Fix:** Ensure numeric columns are included before running validation.")

    # 2️⃣ Validate `set_population` and `set_target` Constraints
    if "set_population" in df.columns and "set_target" in df.columns:
        invalid_pop_target = df[df["set_target"] > df["set_population"]]
        if not invalid_pop_target.empty:
            st.error(f"❌ {len(invalid_pop_target)} rows have 'set_target' greater than 'set_population'.")
            st.info("🛠 **Fix:** Ensure 'set_target' is less than or equal to 'set_population'.")
            st.dataframe(invalid_pop_target)
        else:
            st.success("✅ 'set_population' and 'set_target' constraints are valid.")
    else:
        st.warning("⚠️ The columns 'set_population' and 'set_target' are missing from the dataset.")
        st.info("🛠 **Fix:** Ensure these columns exist before performing constraint validation.")

st.write("---")


# ✅ Ensure latitude and longitude are unique (excluding (0,0))
if "latitude" in df.columns and "longitude" in df.columns:
    df_filtered = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()

    # Convert 'latitude' and 'longitude' to numeric to avoid errors
    df_filtered["latitude"] = pd.to_numeric(df_filtered["latitude"], errors="coerce")
    df_filtered["longitude"] = pd.to_numeric(df_filtered["longitude"], errors="coerce")

    # Drop rows with NaN values after conversion
    df_filtered = df_filtered.dropna(subset=["latitude", "longitude"])

    # Round lat/lon to 6 decimal places to prevent floating-point precision issues
    df_filtered["lat_lon"] = df_filtered.apply(lambda row: (round(row["latitude"], 6), round(row["longitude"], 6)), axis=1)

    # Find duplicate lat/lon pairs
    duplicate_lat_lon = df_filtered[df_filtered.duplicated("lat_lon", keep=False)]

    if not duplicate_lat_lon.empty:
        st.error(f"❌ {len(duplicate_lat_lon)} duplicate latitude/longitude pairs found (excluding (0,0))!")
        st.info("🛠️ **Fix:** Ensure each location has a unique latitude and longitude.")

        # Show full table with all columns
        st.dataframe(duplicate_lat_lon)
    else:
        st.success("✅ All latitude/longitude pairs are unique (excluding (0,0)).")


# Create two columns for side-by-side display
col1, col2 = st.columns(2)

if uploaded_file:
    st.success("✅ Upload successful!")
    # ... your file reading and processing code ...

    # Create two columns for side-by-side display
    col1, col2 = st.columns(2)

    # 📍 Geolocation Proximity Validation (Left Side)
with col1:
    if uploaded_file and not df.empty:  # ✅ Ensures this runs only if a file is uploaded
        st.markdown("### 📍 Geolocation Proximity Validation")
        
        if "latitude" in df.columns and "longitude" in df.columns:
            df_filtered = df[(df["latitude"] != 0) & (df["longitude"] != 0)].copy()

            # Convert 'latitude' and 'longitude' to numeric before calculations
            df_filtered["latitude"] = pd.to_numeric(df_filtered["latitude"], errors="coerce")
            df_filtered["longitude"] = pd.to_numeric(df_filtered["longitude"], errors="coerce")

            # Drop NaN values after conversion
            df_filtered = df_filtered.dropna(subset=["latitude", "longitude"])
            df_filtered = df_filtered.replace([np.inf, -np.inf], np.nan).dropna(subset=["latitude", "longitude"])

            if df_filtered.empty:
                st.error("🚨 No valid geolocations found after filtering NaN/Inf values.")
            else:
                # Convert latitude/longitude to radians
                coords = np.radians(df_filtered[["latitude", "longitude"]].values)
                tree = cKDTree(coords)

                def count_neighbors(tree, coords, distance_threshold):
                    """Returns an array with the count of neighbors within a given distance."""
                    radius = distance_threshold / 6371000  # Convert meters to radians
                    counts = [len(c) - 1 for c in tree.query_ball_point(coords, radius)]  # Exclude self
                    return np.array(counts)

                # Compute proximity counts
                df_filtered["count_within_10m"] = count_neighbors(tree, coords, 10)
                df_filtered["count_within_30m"] = count_neighbors(tree, coords, 30)
                df_filtered["count_within_50m"] = count_neighbors(tree, coords, 50)

                # ✅ Fix: Ensure only valid indexes are used
                valid_index = df_filtered.index.intersection(df.index)
                df_filtered = df_filtered.loc[valid_index]

                # Filter only locations that have at least one nearby point
                df_filtered = df_filtered[
                    (df_filtered["count_within_10m"] > 0) | 
                    (df_filtered["count_within_30m"] > 0) | 
                    (df_filtered["count_within_50m"] > 0)
                ]

                if df_filtered.empty:
                    st.success("✅ No geolocations found within 10m, 30m, or 50m proximity.")
                else:
                    st.dataframe(df_filtered[["latitude", "longitude", "count_within_10m", "count_within_30m", "count_within_50m"]])

# 📊 Null Values Summary (Right Side)
with col2:
    if uploaded_file and not df.empty:  # ✅ Ensures this runs only if a file is uploaded
        st.markdown("### 📊 Null Values Summary")

        schema_columns = df.columns  # Replace with actual validated columns if needed
        null_counts = df[schema_columns].isnull().sum()
        total_rows = len(df)
        null_percentage = (null_counts / total_rows) * 100

        null_summary = pd.DataFrame({
            "Column": schema_columns,
            "Null Count": null_counts,
            "Null Percentage (%)": null_percentage,
        }).reset_index(drop=True)

        # ✅ Function to style the DataFrame
        def highlight_nulls(val):
            """Highlight cells: Green if 0 nulls, Red otherwise."""
            color = "green" if val == 0 else "red"
            return f"background-color: {color}; color: white;"

        # Apply styling to the 'Null Count' column
        styled_df = null_summary.style.applymap(highlight_nulls, subset=["Null Count"])

        # Display styled dataframe
        st.dataframe(styled_df)




if uploaded_file and not df.empty:
    st.markdown("### 🔧 Validation Status Check for 'Validated' Rows")
    
    # Ensure 'validation_status' column exists before processing
    if "validation_status" in df.columns:
        validated_rows = df[df["validation_status"] == "Validated"].copy()
        
        # Precompute duplicate geolocation mask for rows with valid geolocations (excluding (0,0))
        valid_geo = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
        dup_geo = valid_geo.duplicated(subset=["latitude", "longitude"], keep=False)
        
        # Ensure dup_geo aligns with validated_rows to avoid KeyError
        dup_geo = dup_geo.reindex(validated_rows.index, fill_value=False)
        
        # Define validation conditions (rows must meet all checks)
        failed_rows = validated_rows[
            (validated_rows["latitude"].isnull() | validated_rows["longitude"].isnull()) |  # Missing geolocation
            (dup_geo) |  # Duplicate geolocation for valid geos
            (validated_rows["set_target"] > validated_rows["set_population"]) |  # Population constraint
            (validated_rows[numeric_columns].lt(0).any(axis=1)) |  # Negative numeric values
            (~validated_rows[list(validations.keys())].isin(validations).all(axis=1))  # Categorical validation failure
        ]
        
        failed_count = failed_rows.shape[0]
        
        if failed_count > 0:
            st.error(f"❌ {failed_count} 'Validated' rows failed one or more QC checks!")
            st.markdown(f"### 🚨 Failed Validated Rows (Count: {failed_count})")
            failed_rows["Failed Checks"] = ""
            
            # Identify failed checks for each row
            for index, row in failed_rows.iterrows():
                failed_checks = []
                
                if pd.isnull(row["latitude"]) or pd.isnull(row["longitude"]):
                    failed_checks.append("Missing Geolocation")
                
                if dup_geo.loc[index]:
                    failed_checks.append("Duplicate Geolocation")
                
                if "set_population" in row and "set_target" in row and row["set_target"] > row["set_population"]:
                    failed_checks.append("'set_target' > 'set_population'")
                
                if any(row[col] < 0 for col in numeric_columns if col in df.columns):
                    failed_checks.append("Negative Numeric Value")
                
                for col, valid_values in validations.items():
                    if col in df.columns and row[col] not in valid_values:
                        failed_checks.append(f"Invalid {col}")
                
                failed_rows.at[index, "Failed Checks"] = ", ".join(failed_checks)
            
            # Show the failed rows table
            st.dataframe(failed_rows[["validation_status", "latitude", "longitude", "set_population", 
                                      "set_target", "Failed Checks"]])
        else:
            st.success("✅ All 'Validated' rows passed QC checks!")
    else:
        st.warning("⚠️ The column 'validation_status' is missing. Cannot perform validation check.")


# ✅ Initialize error tracking columns
if "latitude" in df.columns and "longitude" in df.columns:
    df["error_duplicate"] = df.duplicated(subset=["unique_code", "globalid"], keep=False).map(
        lambda x: "Duplicate ID" if x else ""
    )
    df["error_missing_geolocation"] = df.apply(
        lambda row: "Missing or invalid geolocation"
        if pd.isnull(row["latitude"]) or pd.isnull(row["longitude"]) or row["latitude"] == 0 or row["longitude"] == 0 
        else "", axis=1
    )


for col, valid_values in validations.items():
    if col in df.columns:
        df[f"error_invalid_{col}"] = df[col].apply(lambda x: f"Invalid {col}: {x}" if x not in valid_values else "")

if "set_population" in df.columns and "set_target" in df.columns:
    df["error_invalid_set_target"] = df.apply(lambda row: "set_target > set_population" 
        if row["set_target"] > row["set_population"] else "", axis=1)
        
# Drop initial boolean error columns (those that contain only True/False)
bool_error_columns = [col for col in df.columns if col.startswith("error_") and df[col].dtype == bool]
df = df.drop(columns=bool_error_columns)


# Allow file download with full dataset + error columns
st.markdown("### 📥 Download QC Report")
csv = df.to_csv(index=False).encode('utf-8')

st.download_button(
label="Download Full QC Report",
data=csv,
file_name="Full_QC_Report.csv",
mime="text/csv"
)


st.markdown("""
    <center>
        <h7>"Developed by **GIS Data Analytics Department** 🚀"</h7>
        <hr>
    </center>
""", unsafe_allow_html=True)       