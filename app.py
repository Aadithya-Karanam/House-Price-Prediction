
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("xgb_model.pkl")


neighborhoods = {
    "Blmngtn": ("Bloomington Heights", "Bloomington Heights neighborhood."),
    "Blueste": ("Bluestem", "Bluestem neighborhood."),
    "BrDale": ("Briardale", "Briardale neighborhood."),
    "BrkSide": ("Brookside", "Brookside neighborhood."),
    "ClearCr": ("Clear Creek", "Clear Creek neighborhood."),
    "CollgCr": ("College Creek", "College Creek neighborhood."),
    "Crawfor": ("Crawford", "Crawford neighborhood."),
    "Edwards": ("Edwards", "Edwards neighborhood."),
    "Gilbert": ("Gilbert", "Gilbert neighborhood."),
    "IDOTRR": ("Iowa DOT and Railroad", "Area identified by the Iowa DOT and Railroad label."),
    "MeadowV": ("Meadow Village", "Meadow Village neighborhood."),
    "Mitchel": ("Mitchell", "Mitchell neighborhood."),
    "Names": ("North Ames", "North Ames neighborhood."),
    "NoRidge": ("Northridge", "Northridge neighborhood."),
    "NPkVill": ("Northpark Villa", "Northpark Villa neighborhood."),
    "NridgHt": ("Northridge Heights", "Northridge Heights neighborhood."),
    "NWAmes": ("Northwest Ames", "Northwest Ames neighborhood."),
    "OldTown": ("Old Town", "Old Town neighborhood."),
    "SWISU": ("South and West of Iowa State University", "Area south and west of Iowa State University."),
    "Sawyer": ("Sawyer", "Sawyer neighborhood."),
    "SawyerW": ("Sawyer West", "Sawyer West neighborhood."),
    "Somerst": ("Somerset", "Somerset neighborhood."),
    "StoneBr": ("Stone Brook", "Stone Brook neighborhood."),
    "Timber": ("Timberland", "Timberland neighborhood."),
    "Veenker": ("Veenker", "Veenker neighborhood.")
}



st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)


st.markdown("""
<style>

.main {
    padding-top: 2rem;
}

.title {
    font-size: 38px;
    font-weight: 700;
}

.subtitle {
    color: #9CA3AF;
    font-size: 16px;
}

.prediction {
    background-color: rgba(22, 163, 74, 0.10);
    padding: 25px;
    border-radius: 12px;
    text-align: center;
}

</style>
""", unsafe_allow_html=True)



st.title("🏠 House Price Prediction")

st.markdown(
    '<p class="subtitle">Estimate property prices using Machine Learning and XGBoost.</p>',
    unsafe_allow_html=True
)

st.divider()


st.subheader("🏡 Property Information")

st.write(
    "Enter the details of the property to estimate its market value."
)

col1, col2 = st.columns(2)

with col1:

    st.markdown("#### Property Quality")

    overall_qual = st.slider(
        "Overall Quality",
        min_value=1,
        max_value=10,
        value=5,
        help="Rates the overall material and finish of the house. 1 = Very Poor, 10 = Very Excellent."
    )

    gr_liv_area = st.number_input(
        "Living Area (sq ft)",
        min_value=100,
        max_value=10000,
        value=1500,
        help="Above-ground living area in square feet. Basement area is not included."
    )

    first_flr_sf = st.number_input(
        "First Floor Area (sq ft)",
        min_value=100,
        max_value=5000,
        value=1000,
        help="Total area of the first floor in square feet."
    )

    total_bsmt_sf = st.number_input(
        "Basement Area (sq ft)",
        min_value=0,
        max_value=6000,
        value=800,
        help="Total basement area in square feet. Enter 0 if there is no basement."
    )


with col2:

    st.markdown("#### Construction Details")

    garage_cars = st.selectbox(
        "Garage Capacity",
        [0, 1, 2, 3, 4],
        index=2,
        help="Number of cars that can fit in the garage."
    )

    full_bath = st.selectbox(
        "Full Bathrooms",
        [0, 1, 2, 3, 4],
        index=2,
        help="Number of full bathrooms in the house."
    )

    year_built = st.number_input(
        "Year Built",
        min_value=1870,
        max_value=2026,
        value=2000,
        help="The original construction year of the property."
    )


st.divider()


st.subheader("📍 Property Location")

st.write(
    "Select the neighborhood where the property is located."
)

st.info(
    "Neighborhood names represent geographical areas in Ames, Iowa. "
    "They are not quality ratings or rankings. The model learns "
    "their relationship with historical house prices."
)

selected_neighborhood = st.selectbox(
    "Choose Neighborhood",
    options=sorted(neighborhoods.keys()),
    format_func=lambda code: (
        f"{neighborhoods[code][0]} ({code})"
    ),
    help="Choose the property's neighborhood. The original dataset code is passed to the ML model."
)

st.caption(
    neighborhoods[selected_neighborhood][1]
)

st.divider()


st.subheader("💰 Price Estimation")

if st.button(
    "Predict House Price",
    type="primary",
    use_container_width=True
):

    if first_flr_sf > gr_liv_area:
        st.error(
            "First floor area cannot exceed the total above-ground living area."
        )

    else:

        input_data = pd.DataFrame([{
            "OverallQual": overall_qual,
            "GrLivArea": gr_liv_area,
            "GarageCars": garage_cars,
            "TotalBsmtSF": total_bsmt_sf,
            "1stFlrSF": first_flr_sf,
            "FullBath": full_bath,
            "YearBuilt": year_built,
            "Neighborhood": selected_neighborhood
        }])

        prediction = model.predict(input_data)[0]

        st.success("Prediction generated successfully!")

        st.markdown(
            '<div class="prediction">',
            unsafe_allow_html=True
        )

        st.markdown("### Estimated House Price")

        st.markdown(
            f"## ${prediction:,.2f}"
        )

        st.markdown(
            "</div>",
            unsafe_allow_html=True
        )

        st.divider()

        st.subheader("📋 Property Summary")

        result_col1, result_col2, result_col3 = st.columns(3)

        result_col1.metric(
            "Living Area",
            f"{gr_liv_area:,} sq ft"
        )

        result_col2.metric(
            "Overall Quality",
            f"{overall_qual}/10"
        )

        result_col3.metric(
            "Garage Capacity",
            f"{garage_cars} cars"
        )

        st.write(
            "**Neighborhood:**",
            neighborhoods[selected_neighborhood][0]
        )

        st.write(
            "**Construction Year:**",
            year_built
        )

        st.info(
            "This prediction is generated by an XGBoost regression model "
            "trained on the Ames Housing dataset. It is an educational "
            "estimate and not a professional property valuation."
        )



st.divider()

with st.expander("ℹ️ About This Project"):

    st.write("""
    **Algorithm:** XGBoost Regressor

    **Problem Type:** Regression

    **Dataset:** Ames Housing Dataset

    **Input Features:** 8 selected features

    **Preprocessing:**
    - Missing value imputation
    - One-Hot Encoding
    - Numerical and categorical feature processing

    **Evaluation Metrics:**
    - Mean Absolute Error (MAE)
    - Mean Squared Error (MSE)
    - Root Mean Squared Error (RMSE)
    - R-squared (R²)
    """)

    st.write(
        "Developed as a Machine Learning project."
    )
