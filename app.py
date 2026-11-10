import os
import time
import joblib
import pandas as pd
import streamlit as st

# 1. Konfigurasi Halaman
st.set_page_config(
    page_title="Food Delivery Time Prediction",
    page_icon="🍔",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# 2. Custom CSS - Solid Flat Design + Layout Dual Panel Card
st.markdown(
    """
    <style>
    /* Keyframe Animations */
    @keyframes popIn {
        0% { opacity: 0; transform: scale(0.95) translateY(10px); }
        100% { opacity: 1; transform: scale(1) translateY(0); }
    }

    @keyframes bounceIcon {
        0%, 100% { transform: translateY(0); }
        50% { transform: translateY(-6px); }
    }

    @keyframes motorRide {
        0% { transform: translateX(0px); }
        50% { transform: translateX(15px); }
        100% { transform: translateX(0px); }
    }

    /* Container Utama */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 980px;
    }

    /* Hero Header Banner */
    .hero-card {
        background-color: #FF5722;
        padding: 2rem;
        border-radius: 16px;
        color: #FFFFFF;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(255, 87, 34, 0.2);
    }

    .hero-title {
        display: flex;
        align-items: center;
        gap: 12px;
    }

    .hero-icon {
        font-size: 2.2rem;
        display: inline-block;
        animation: bounceIcon 2.5s infinite ease-in-out;
    }

    .hero-card h1 {
        color: #FFFFFF !important;
        font-weight: 700;
        margin: 0 !important;
        font-size: 2.1rem;
    }

    .hero-card p {
        color: #FFEBE5;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        margin-bottom: 0;
    }

    .model-badge {
        display: inline-block;
        background-color: #E64A19;
        color: #FFFFFF;
        padding: 0.3rem 0.85rem;
        border-radius: 8px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    /* Motor Loading Widget */
    .motor-loading-box {
        text-align: center;
        padding: 1.5rem;
        background-color: #FFF5F2;
        border: 2px dashed #FFAB91;
        border-radius: 12px;
        margin-top: 1rem;
    }

    .motor-animation {
        font-size: 2.5rem;
        display: inline-block;
        animation: motorRide 1.2s infinite ease-in-out;
    }

    /* Container Card Hasil Prediksi (Split Layout) */
    .result-card-solid {
        animation: popIn 0.4s ease-out forwards;
        background-color: #FFFFFF;
        border: 2px solid #E2E8F0;
        border-top: 6px solid #FF5722;
        border-radius: 16px;
        padding: 1.8rem 2rem;
        margin-top: 1.5rem;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.05);
    }

    .result-left-panel {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
        text-align: center;
        border-right: 1px solid #F1F5F9;
        padding-right: 1.5rem;
    }

    .result-right-panel {
        display: flex;
        flex-direction: column;
        justify-content: center;
        padding-left: 1.5rem;
        height: 100%;
    }

    .result-motor-icon {
        font-size: 2rem;
        margin-bottom: 0.2rem;
        display: inline-block;
        animation: motorRide 2s infinite ease-in-out;
    }

    .result-label {
        color: #64748B;
        font-size: 0.82rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
        margin-bottom: 0.2rem;
    }

    .result-number-solid {
        color: #FF5722;
        font-size: 4rem;
        font-weight: 800;
        line-height: 1;
        display: inline-block;
    }

    .result-unit {
        font-size: 1.2rem;
        font-weight: 700;
        color: #334155;
        margin-left: 4px;
    }

    /* Status Pill */
    .status-pill {
        display: inline-flex;
        align-items: center;
        justify-content: center;
        gap: 8px;
        padding: 0.55rem 1.2rem;
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.88rem;
        margin-top: 0.8rem;
        width: 100%;
        text-align: center;
    }

    .status-fast { background-color: #DCFCE7; color: #166534; border: 1px solid #BBF7D0; }
    .status-normal { background-color: #E0F2FE; color: #0369A1; border: 1px solid #BAE6FD; }
    .status-slow { background-color: #FEF3C7; color: #92400E; border: 1px solid #FDE68A; }

    /* Right Panel Elements (Progress Card) */
    .progress-title {
        font-size: 0.9rem;
        font-weight: 700;
        color: #1E293B;
        margin-bottom: 0.4rem;
    }

    .progress-subtext {
        font-size: 0.82rem;
        color: #64748B;
        margin-bottom: 0.8rem;
    }

    .progress-bar-bg {
        background-color: #F1F5F9;
        border-radius: 10px;
        height: 14px;
        width: 100%;
        overflow: hidden;
        border: 1px solid #E2E8F0;
    }

    .progress-bar-fill {
        background-color: #FF5722;
        height: 100%;
        border-radius: 10px;
        transition: width 0.6s ease-in-out;
    }

    .progress-meta {
        display: flex;
        justify-content: space-between;
        font-size: 0.78rem;
        font-weight: 600;
        color: #94A3B8;
        margin-top: 0.4rem;
    }

    /* Footer Styling */
    .footer-text {
        text-align: center;
        color: #94A3B8;
        font-size: 0.85rem;
        margin-top: 3.5rem;
        padding-top: 1.5rem;
        border-top: 1px solid #E2E8F0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = "best_delivery_model.joblib"
CONFIG_PATH = "feature_config.joblib"


@st.cache_resource
def load_artifacts():
    if not os.path.exists(MODEL_PATH) or not os.path.exists(CONFIG_PATH):
        return None, None
    model = joblib.load(MODEL_PATH)
    config = joblib.load(CONFIG_PATH)
    return model, config


def engineer_features(
    raw_df: pd.DataFrame, feature_cols_num, feature_cols_cat, feature_cols_binary
):
    df_out = raw_df.copy()
    df_out["Distance_Prep_Ratio"] = df_out["Distance_km"] / (
        df_out["Preparation_Time_min"] + 1
    )
    df_out["Is_Peak_Hour"] = (
        df_out["Time_of_Day"].isin(["Morning", "Evening"]).astype(int)
    )
    bad_weather = ["Rainy", "Snowy", "Foggy"]
    df_out["Bad_Condition"] = (
        (df_out["Traffic_Level"] == "High") | (df_out["Weather"].isin(bad_weather))
    ).astype(int)
    return df_out[feature_cols_num + feature_cols_cat + feature_cols_binary]


model, config = load_artifacts()

# Banner Header Utama
model_name = config["best_model_name"] if config else "N/A"
st.markdown(
    f"""
    <div class="hero-card">
        <div class="hero-title">
            <span class="hero-icon">🍔</span>
            <h1>Food Delivery Time Prediction</h1>
        </div>
        <p>Estimasi waktu pengantaran makanan presisi berbasis Machine Learning berdasarkan kondisi lalu lintas, jarak, cuaca, dan profil kurir.</p>
        <div class="model-badge">⚡ Model Aktif: {model_name}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

if model is None:
    st.error(
        "⚠️ File model tidak ditemukan. Pastikan `best_delivery_model.joblib` dan "
        "`feature_config.joblib` berada di folder yang sama dengan `app.py`."
    )
    st.stop()

# Form Input Interaktif
with st.form("prediction_form"):
    st.subheader("📋 Input Parameter Pesanan")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📍 Rincian Pengiriman")
        distance_km = st.number_input(
            "Jarak Pengantaran (km)",
            min_value=0.1,
            max_value=50.0,
            value=6.5,
            step=0.1,
            help="Jarak lokasi restoran ke tujuan pengiriman.",
        )
        weather = st.selectbox(
            "Kondisi Cuaca",
            options=config["weather_options"],
            index=config["weather_options"].index("Rainy")
            if "Rainy" in config["weather_options"]
            else 0,
        )
        traffic_level = st.selectbox(
            "Kondisi Lalu Lintas",
            options=config["traffic_options"],
            index=min(2, len(config["traffic_options"]) - 1),
        )
        time_of_day = st.selectbox(
            "Waktu Pemesanan",
            options=config["time_of_day_options"],
            index=min(2, len(config["time_of_day_options"]) - 1),
        )

    with col2:
        st.markdown("#### 🛵 Rincian Kurir & Makanan")
        vehicle_type = st.selectbox(
            "Jenis Kendaraan Kurir", options=config["vehicle_options"]
        )
        preparation_time = st.number_input(
            "Waktu Persiapan Makanan (menit)",
            min_value=1,
            max_value=120,
            value=15,
            step=1,
            help="Perkiraan waktu yang dibutuhkan restoran untuk mengemas makanan.",
        )
        courier_experience = st.number_input(
            "Pengalaman Kerja Kurir (tahun)",
            min_value=0.0,
            max_value=30.0,
            value=2.0,
            step=0.5,
        )

    st.markdown("---")
    submit_btn = st.form_submit_button(
        "🔎 Hitung Estimasi Waktu", use_container_width=True, type="primary"
    )

# Eksekusi Prediksi Saat Tombol Ditekan
if submit_btn:
    # 1. Animasi Loading Kurir Motor
    loading_placeholder = st.empty()
    loading_placeholder.markdown(
        """
        <div class="motor-loading-box">
            <div class="motor-animation">🛵💨</div>
            <div style="color: #FF5722; font-weight: 700; margin-top: 8px;">Memproses Rute & Mengkalkulasi Estimasi Waktu...</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    time.sleep(1.2)
    loading_placeholder.empty()

    # 2. Proses Prediksi Model ML
    input_df = pd.DataFrame(
        [
            {
                "Distance_km": distance_km,
                "Weather": weather,
                "Traffic_Level": traffic_level,
                "Time_of_Day": time_of_day,
                "Vehicle_Type": vehicle_type,
                "Preparation_Time_min": preparation_time,
                "Courier_Experience_yrs": courier_experience,
            }
        ]
    )

    processed = engineer_features(
        input_df,
        config["feature_cols_num"],
        config["feature_cols_cat"],
        config["feature_cols_binary"],
    )

    prediction = model.predict(processed)[0]
    pred_val = max(1, int(round(prediction)))
    progress_pct = min(100, int((pred_val / 90.0) * 100))

    # Penentuan Kategori Status
    if pred_val <= 30:
        status_class = "status-fast"
        status_text = "🚀 Pengiriman Cepat: Estimasi waktu singkat."
        st.balloons()
    elif pred_val <= 60:
        status_class = "status-normal"
        status_text = "⏱️ Pengiriman Normal: Waktu standar pengantaran."
    else:
        status_class = "status-slow"
        status_text = "⚠️ Potensi Keterlambatan: Dipengaruhi cuaca/lalu lintas."

    # Visual Card Berisi Estimasi Waktu & Progress Bar Samping
    st.markdown(
        f"""
        <div class="result-card-solid">
            <div style="display: flex; flex-wrap: wrap; align-items: center;">
                <div class="result-left-panel" style="flex: 1; min-width: 280px;">
                    <div class="result-motor-icon">🛵💨</div>
                    <div class="result-label">Estimasi Waktu Pengiriman</div>
                    <div>
                        <span class="result-number-solid">{pred_val}</span>
                        <span class="result-unit">Menit</span>
                    </div>
                    <div class="status-pill {status_class}">
                        {status_text}
                    </div>
                </div>
                <div class="result-right-panel" style="flex: 1; min-width: 280px;">
                    <div class="progress-title">📊 Tingkat Durasi Pengantaran</div>
                    <div class="progress-subtext">
                        <b>{pred_val} Menit</b> dari batas standar acuan (90 menit)
                    </div>
                    <div class="progress-bar-bg">
                        <div class="progress-bar-fill" style="width: {progress_pct}%;"></div>
                    </div>
                    <div class="progress-meta">
                        <span>0m</span>
                        <span>{progress_pct}% dari acuan</span>
                        <span>90m</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write("")

    # Menampilkan Detail Input
    with st.expander("📊 Lihat Detail Parameter Input", expanded=False):
        st.dataframe(input_df, use_container_width=True, hide_index=True)

# Footer
st.markdown(
    """
    <div class="footer-text">
        <p>Catatan: Model dilatih menggunakan data historis pengiriman makanan. Hasil prediksi merupakan estimasi berbasis statistik.</p>
        Crafted with ❤️ by <b>Muhammad Dafa Sirajudin</b>
    </div>
    """,
    unsafe_allow_html=True,
)