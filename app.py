import streamlit as st
import plotly.graph_objects as go
from prophet.serialize import model_from_json


# -------------------------------
# Page configuration
# -------------------------------

st.set_page_config(
    page_title="AirSense | Humidity Forecast",
    page_icon="🌿",
    layout="wide"
)


# -------------------------------
# Custom professional design
# -------------------------------

st.markdown(
    """
    <style>
        .stApp {
            background-color: #f4f8fb;
        }

        .block-container {
            max-width: 1200px;
            padding-top: 2rem;
            padding-bottom: 2rem;
        }

        .hero {
            background: linear-gradient(135deg, #073b4c, #118ab2);
            padding: 35px;
            border-radius: 24px;
            color: white;
            margin-bottom: 25px;
        }

        .hero h1 {
            font-size: 42px;
            margin-bottom: 8px;
        }

        .hero p {
            font-size: 18px;
            margin-bottom: 0;
        }

        .info-card {
            background-color: white;
            padding: 22px;
            border-radius: 18px;
            box-shadow: 0 4px 18px rgba(0, 0, 0, 0.08);
            margin-bottom: 20px;
        }

        .small-label {
            color: #607d8b;
            font-size: 14px;
        }

        .large-value {
            color: #073b4c;
            font-size: 30px;
            font-weight: bold;
        }

        .footer {
            text-align: center;
            color: #607d8b;
            padding-top: 20px;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# -------------------------------
# Load the saved Prophet model
# -------------------------------

@st.cache_resource
def load_model():
    with open("prophet_rh_model.json", "r") as file:
        return model_from_json(file.read())


model = load_model()


# -------------------------------
# Header
# -------------------------------

st.markdown(
    """
    <div class="hero">
        <h1>🌿 AirSense</h1>
        <p>
            Intelligent relative-humidity forecasting using the Prophet model
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------------------
# About section
# -------------------------------

st.markdown(
    """
    <div class="info-card">
        <h3>About this project</h3>
        <p>
            AirSense uses historical environmental observations to forecast
            future relative humidity. Select a forecast duration and generate
            an interactive prediction.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)


# -------------------------------
# Sidebar controls
# -------------------------------

st.sidebar.title("⚙️ Forecast Controls")

st.sidebar.write(
    "Choose how many future hours you want to forecast."
)

hours = st.sidebar.slider(
    "Forecast duration",
    min_value=24,
    max_value=720,
    value=168,
    step=24
)

st.sidebar.write(f"Selected duration: **{hours} hours**")

generate_forecast = st.sidebar.button(
    "🚀 Generate Forecast",
    use_container_width=True
)


# -------------------------------
# Generate forecast
# -------------------------------

if generate_forecast:

    with st.spinner("Generating forecast..."):

        future = model.make_future_dataframe(
            periods=hours,
            freq="h"
        )

        forecast = model.predict(future)

        result = forecast[
            ["ds", "yhat", "yhat_lower", "yhat_upper"]
        ].tail(hours).copy()


    # ---------------------------
    # Calculate summary values
    # ---------------------------

    average_rh = result["yhat"].mean()
    minimum_rh = result["yhat"].min()
    maximum_rh = result["yhat"].max()


    # ---------------------------
    # Summary cards
    # ---------------------------

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="small-label">
                    Average Relative Humidity
                </div>
                <div class="large-value">
                    {average_rh:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="small-label">
                    Minimum Relative Humidity
                </div>
                <div class="large-value">
                    {minimum_rh:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            f"""
            <div class="info-card">
                <div class="small-label">
                    Maximum Relative Humidity
                </div>
                <div class="large-value">
                    {maximum_rh:.1f}%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )


    # ---------------------------
    # Interactive forecast chart
    # ---------------------------

    st.subheader("📈 Relative Humidity Forecast")

    chart = go.Figure()

    # Upper prediction boundary
    chart.add_trace(
        go.Scatter(
            x=result["ds"],
            y=result["yhat_upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False
        )
    )

    # Lower prediction boundary and shaded area
    chart.add_trace(
        go.Scatter(
            x=result["ds"],
            y=result["yhat_lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(17, 138, 178, 0.18)",
            line=dict(width=0),
            name="Prediction range"
        )
    )

    # Main prediction line
    chart.add_trace(
        go.Scatter(
            x=result["ds"],
            y=result["yhat"],
            mode="lines",
            line=dict(
                color="#118ab2",
                width=3
            ),
            name="Predicted RH"
        )
    )

    chart.update_layout(
        height=500,
        template="plotly_white",
        hovermode="x unified",
        xaxis_title="Date and Time",
        yaxis_title="Relative Humidity (%)",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )

    st.plotly_chart(
        chart,
        use_container_width=True
    )


    # ---------------------------
    # Forecast table
    # ---------------------------

    st.subheader("📋 Forecast Details")

    display_result = result.rename(
        columns={
            "ds": "Date and Time",
            "yhat": "Predicted RH",
            "yhat_lower": "Lower Estimate",
            "yhat_upper": "Upper Estimate"
        }
    )

    st.dataframe(
        display_result.style.format(
            {
                "Predicted RH": "{:.2f}",
                "Lower Estimate": "{:.2f}",
                "Upper Estimate": "{:.2f}"
            }
        ),
        use_container_width=True,
        hide_index=True
    )


    # ---------------------------
    # Download forecast
    # ---------------------------

    csv_data = display_result.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="⬇️ Download Forecast CSV",
        data=csv_data,
        file_name="relative_humidity_forecast.csv",
        mime="text/csv"
    )


else:
    st.info(
        "Choose a forecast duration from the left sidebar, "
        "then click 'Generate Forecast'."
    )


# -------------------------------
# Footer
# -------------------------------

st.markdown(
    """
    <div class="footer">
        Built with Python • Streamlit • Prophet
    </div>
    """,
    unsafe_allow_html=True
)