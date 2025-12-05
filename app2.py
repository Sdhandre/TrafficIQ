import streamlit as st
import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostRegressor
import folium
from streamlit_folium import st_folium

# ⚠️ MUST BE FIRST
st.set_page_config(
    page_title="ETA Predictor | Indore",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ======================================================
# 🎨 PROFESSIONAL CSS WITH NAVBAR & FOOTER
# ======================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

/* Dark professional background */
.stApp {
    background: linear-gradient(180deg, #0a0e1a 0%, #111827 50%, #0a0e1a 100%);
}

/* Remove default padding */
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* NAVBAR */
.navbar {
    background: rgba(15, 23, 42, 0.95);
    backdrop-filter: blur(20px);
    border-bottom: 1px solid rgba(148, 163, 184, 0.2);
    padding: 16px 0;
    position: sticky;
    top: 0;
    z-index: 1000;
    box-shadow: 0 4px 20px rgba(0,0,0,0.5);
}

.navbar-content {
    max-width: 1200px;
    margin: 0 auto;
    padding: 0 24px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.navbar-brand {
    font-size: 1.4rem;
    font-weight: 700;
    color: #22c55e;
    display: flex;
    align-items: center;
    gap: 10px;
}

.navbar-links {
    display: flex;
    gap: 24px;
    align-items: center;
}

.nav-link {
    color: #9ca3af;
    text-decoration: none;
    font-size: 0.9rem;
    font-weight: 500;
    transition: color 0.2s;
}

.nav-link:hover {
    color: #22c55e;
}

/* MAIN CONTENT WRAPPER */
.main-content {
    max-width: 1200px;
    margin: 40px auto;
    padding: 0 24px;
}

/* GLASS CARD - PROPER CONTAINER */
.glass-card {
    background: rgba(15, 23, 42, 0.85);
    backdrop-filter: blur(30px);
    border-radius: 20px;
    border: 1px solid rgba(148, 163, 184, 0.25);
    box-shadow: 0 25px 70px rgba(0, 0, 0, 0.6);
    padding: 32px;
    margin-bottom: 28px;
}

/* HERO SECTION */
.hero-section {
    text-align: center;
    padding: 48px 32px;
    background: linear-gradient(135deg, rgba(8, 145, 178, 0.12), rgba(34, 197, 94, 0.12));
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.hero-title {
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #0ea5e9, #22c55e);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 12px;
    letter-spacing: -0.02em;
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 1.05rem;
    margin-bottom: 24px;
}

.badge-container {
    display: flex;
    gap: 12px;
    justify-content: center;
    flex-wrap: wrap;
}

.feature-badge {
    background: rgba(15, 23, 42, 0.9);
    color: #e5e7eb;
    padding: 8px 18px;
    border-radius: 999px;
    font-size: 0.85rem;
    font-weight: 600;
    border: 1px solid rgba(148, 163, 184, 0.3);
    display: inline-flex;
    align-items: center;
    gap: 6px;
}

/* SECTION TITLES */
.section-title {
    color: #f1f5f9;
    font-size: 1.3rem;
    font-weight: 700;
    margin-bottom: 6px;
    display: flex;
    align-items: center;
    gap: 10px;
}

.section-divider {
    height: 2px;
    background: linear-gradient(90deg, rgba(34,197,94,0.6), rgba(14,165,233,0.3), transparent);
    margin-bottom: 24px;
    border-radius: 2px;
}

/* FORM INPUTS */
.stSelectbox label, .stTimeInput label {
    color: #cbd5e1 !important;
    font-weight: 500 !important;
    font-size: 0.9rem !important;
}

.stSelectbox > div > div,
.stTimeInput > div > div {
    background: rgba(15, 23, 42, 0.95) !important;
    border: 1px solid rgba(75, 85, 99, 0.8) !important;
    border-radius: 12px !important;
    color: #e5e7eb !important;
}

/* BUTTON */
.stButton > button {
    background: linear-gradient(135deg, #0ea5e9, #22c55e) !important;
    color: #0a0e1a !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 16px 32px !important;
    font-size: 1rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-transform: uppercase !important;
    box-shadow: 0 20px 50px rgba(34, 197, 94, 0.4) !important;
    transition: all 0.3s ease !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 25px 60px rgba(34, 197, 94, 0.5) !important;
}

/* RESULT BOX */
.result-box {
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), rgba(14, 165, 233, 0.1));
    border: 2px solid rgba(34, 197, 94, 0.5);
    border-radius: 18px;
    padding: 32px;
    text-align: center;
    margin: 20px 0;
}

.result-label {
    color: #94a3b8;
    font-size: 0.9rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 8px;
}

.result-value {
    color: #22c55e;
    font-size: 3.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 12px;
}

.result-range {
    color: #cbd5e1;
    font-size: 0.95rem;
}

/* TRAFFIC STATUS */
.traffic-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 10px 20px;
    border-radius: 999px;
    font-weight: 600;
    font-size: 0.95rem;
    background: rgba(15, 23, 42, 0.9);
    border: 1px solid rgba(148, 163, 184, 0.4);
    color: #e5e7eb;
}

/* TRIP DETAILS */
.details-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 16px;
    margin-top: 20px;
}

.detail-item {
    background: rgba(15, 23, 42, 0.6);
    padding: 16px;
    border-radius: 12px;
    border: 1px solid rgba(75, 85, 99, 0.5);
}

.detail-label {
    color: #94a3b8;
    font-size: 0.85rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 6px;
}

.detail-value {
    color: #f1f5f9;
    font-size: 1.1rem;
    font-weight: 600;
}

/* FOOTER */
.footer {
    background: rgba(15, 23, 42, 0.95);
    border-top: 1px solid rgba(148, 163, 184, 0.2);
    padding: 32px 24px;
    margin-top: 60px;
    text-align: center;
}

.footer-content {
    max-width: 1200px;
    margin: 0 auto;
}

.footer-text {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-bottom: 12px;
}

.footer-links {
    display: flex;
    gap: 20px;
    justify-content: center;
    margin-bottom: 16px;
}

.footer-link {
    color: #cbd5e1;
    text-decoration: none;
    font-size: 0.85rem;
}

.footer-link:hover {
    color: #22c55e;
}

/* HIDE STREAMLIT CHROME */
#MainMenu {visibility: visible;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* COLUMN GAP */
.row-widget {
    gap: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# ======================================================
# 📌 LOAD MODEL & DATA
# ======================================================

@st.cache_resource
def load_model_and_data():
    model = CatBoostRegressor()
    model.load_model("catboost_eta_model.cbm")
    feature_cols =         feature_cols = [
            'origin_lat', 'origin_lng',
            'dest_lat', 'dest_lng',
            'delta_lat', 'delta_lng',
            'haversine_km',
            'bearing_deg',
            'distance_km',
            'hour',
            'day_enc',
            'weekend', 'peak_hour',
            'temperature', 'humidity', 'visibility', 'rain'
        ]
    from sklearn.preprocessing import LabelEncoder
    le_day = LabelEncoder()
    le_day.classes_ = np.array(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                                     'Friday', 'Saturday', 'Sunday'])
    coords = pd.read_csv("place_coordinates.csv")
    coord_dict = {row["place"]: (row["lat"], row["lng"]) for _, row in coords.iterrows()}
    places = list(coord_dict.keys())
    return model, feature_cols, le_day, coord_dict, places

model, feature_cols, le_day, coord_dict, places = load_model_and_data()

# ======================================================
# 🌍 HELPER FUNCTIONS
# ======================================================

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat/2)**2 + np.cos(lat1)*np.cos(lat2)*np.sin(dlon/2)**2
    return R * 2 * np.arcsin(np.sqrt(a))

def calculate_bearing(lat1, lon1, lat2, lon2):
    lat1, lat2 = np.radians(lat1), np.radians(lat2)
    dlon = np.radians(lon2 - lon1)
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1)*np.sin(lat2) - np.sin(lat1)*np.cos(lat2)*np.cos(dlon)
    return (np.degrees(np.arctan2(x, y)) + 360) % 360

def is_weekend(day):
    return 1 if day in ["Saturday", "Sunday"] else 0

def is_peak(hour):
    return 1 if (9 <= hour <= 11) or (17 <= hour <= 20) else 0

def traffic_status(eta, distance_km):
    if eta == 0:
        return "⚠️ INVALID"
    speed = distance_km / (eta/60)
    if speed >= 19:
        return "🟢 FAST TRAFFIC"
    elif speed >= 15:
        return "🟡 MODERATE TRAFFIC"
    else:
        return "🔴 SLOW TRAFFIC"

# ======================================================
# 🔝 NAVBAR
# ======================================================

st.markdown("""
<div class="navbar">
    <div class="navbar-content">
        <div class="navbar-brand">
            🚗 TrafficIQ
        </div>
        <div class="navbar-links">
            <a href="#" class="nav-link">Home</a>
            <a href="#" class="nav-link">About</a>
            <a href="#" class="nav-link">Features</a>
            <a href="#" class="nav-link">Contact</a>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ======================================================
# 📄 MAIN CONTENT
# ======================================================

st.markdown('<div class="main-content">', unsafe_allow_html=True)

# HERO SECTION
st.markdown("""
<div class="glass-card hero-section">
    <div class="hero-title">🚗 TrafficIQ-AI Powered ETA Predictor</div>
    <div class="hero-subtitle">Predict future travel times in Indore using Machine Learning • 28,000+ Data Points • CatBoost AI</div>
    <div class="badge-container">
        <span class="feature-badge">🔥 No Internet Required</span>
        <span class="feature-badge">🎯 High Accuracy</span>
        <span class="feature-badge">⚡ Instant Prediction</span>
        <span class="feature-badge">📊 Historical Data</span>
    </div>
</div>
""", unsafe_allow_html=True)

# INPUT SECTION

st.markdown('<div class="section-title">📍 Route Selection</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    origin = st.selectbox("🔵 Origin Point", places, key="origin")
with col2:
    destination = st.selectbox("🔴 Destination Point", places, key="dest")

st.markdown('<br>', unsafe_allow_html=True)
st.markdown('<div class="section-title">⏰ Time & Day Selection</div>', unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

col3, col4 = st.columns(2)
with col3:
    time_input = st.time_input("🕐 Travel Time")
with col4:
    selected_day = st.selectbox("📅 Day of Week", le_day.classes_)

st.markdown('<br>', unsafe_allow_html=True)

# Create 3 columns: empty | button | empty for centering
col_left, col_center, col_right = st.columns([2, 3, 2])
with col_center:
    clicked = st.button("🔮 PREDICT ETA", use_container_width=True)

st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# 🧠 PREDICTION LOGIC
# ======================================================

if clicked:
    if origin == destination:
        st.error("❌ Origin and Destination cannot be the same!")
        st.stop()
    
    lat1, lng1 = coord_dict[origin]
    lat2, lng2 = coord_dict[destination]
    delta_lat = lat2 - lat1
    delta_lng = lng2 - lng1
    hav_km = haversine(lat1, lng1, lat2, lng2)
    bearing = calculate_bearing(lat1, lng1, lat2, lng2)
    hour = time_input.hour
    day_encoded = le_day.transform([selected_day])[0]
    weekend_flag = is_weekend(selected_day)
    peak_flag = is_peak(hour)
    
    row = pd.DataFrame([[lat1, lng1, lat2, lng2, delta_lat, delta_lng, hav_km, bearing, hav_km,
                         hour, day_encoded, weekend_flag, peak_flag, 25, 60, 5000, 0]], 
                       columns=feature_cols)
    
    eta = model.predict(row)[0]
    
    st.session_state["result"] = {
        "eta": eta, "eta_low": eta * 0.9, "eta_high": eta * 1.1,
        "lat1": lat1, "lng1": lng1, "lat2": lat2, "lng2": lng2,
        "hav_km": hav_km, "weekend": weekend_flag, "peak": peak_flag,
        "origin": origin, "destination": destination, "hour": hour, "day": selected_day
    }

# ======================================================
# 📊 RESULTS
# ======================================================

if "result" in st.session_state:
    r = st.session_state["result"]
    
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🎯 Prediction Results</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="result-box">
        <div class="result-label">Estimated Time of Arrival</div>
        <div class="result-value">{r['eta']:.1f} min</div>
        <div class="result-range">Range: {r['eta_low']:.1f} – {r['eta_high']:.1f} minutes</div>
    </div>
    """, unsafe_allow_html=True)
    
    status = traffic_status(r["eta"], r["hav_km"])
    st.markdown(f'<div style="text-align:center;margin:20px 0;"><span class="traffic-badge">{status}</span></div>', 
                unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">📊 Trip Details</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="details-grid">
        <div class="detail-item">
            <div class="detail-label">From</div>
            <div class="detail-value">{r['origin']}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">To</div>
            <div class="detail-value">{r['destination']}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Day</div>
            <div class="detail-value">{r['day']}</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Time</div>
            <div class="detail-value">{r['hour']}:00</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Distance</div>
            <div class="detail-value">{r['hav_km']:.2f} km</div>
        </div>
        <div class="detail-item">
            <div class="detail-label">Peak Hour</div>
            <div class="detail-value">{'Yes ⚠️' if r['peak'] else 'No ✅'}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    # MAP
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">🗺 Route Visualization</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    center_lat = (r["lat1"] + r["lat2"]) / 2
    center_lng = (r["lng1"] + r["lng2"]) / 2
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13)
    folium.Marker([r["lat1"], r["lng1"]], tooltip=r["origin"], icon=folium.Icon(color='blue')).add_to(m)
    folium.Marker([r["lat2"], r["lng2"]], tooltip=r["destination"], icon=folium.Icon(color='red')).add_to(m)
    folium.PolyLine([[r["lat1"], r["lng1"]], [r["lat2"], r["lng2"]]], color="#22c55e", weight=4).add_to(m)
    st_folium(m, width=None, height=450)
    st.markdown('</div>', unsafe_allow_html=True)

# PROBLEM STATEMENT

st.markdown("### 🎯 What Problem Does This App Solve?")
st.markdown("""
<div style='color: #e0d4f7; font-size: 16px; line-height: 1.8;'>
Google Maps only gives <strong>live ETA</strong> based on current traffic. But it cannot answer:

<ul style='margin-top: 15px;'>
<li>❌ "What will be the travel time at 8 PM tonight from Vijay Nagar to Rajwada?"</li>
<li>❌ "If I start at 7 AM tomorrow, how much time will it take?"</li>
<li>❌ "What is the typical ETA on weekends vs weekdays?"</li>
<li>❌ "Is this route usually slow at 6 PM even if right now it looks empty?"</li>
</ul>

<strong style='color: #f093fb; font-size: 18px; display: block; margin-top: 20px;'>My app predicts ETA for a FUTURE time, based on:</strong>

<ul style='margin-top: 10px;'>
<li>✅ Historical traffic patterns</li>
<li>✅ Past congestion data</li>
<li>✅ Peak hour behavior</li>
<li>✅ Route-specific congestion patterns</li>
<li>✅ Day-of-week effects</li>
<li>✅ Movement patterns (bearing, geo distance, hotspots)</li>
</ul>

<div style='background: rgba(139, 92, 246, 0.2); padding: 15px; border-radius: 10px; margin-top: 20px; border-left: 4px solid #8b5cf6;'>
<strong>🚀 Key Advantage:</strong> Google Maps cannot forecast future ETA without live sensor network data. 
This app provides intelligent ETA predictions for Indore using ML and works <strong>completely offline</strong> once installed!
</div>
</div>
""", unsafe_allow_html=True)
st.markdown("</div>", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)  # close main-content

# ======================================================
# 🔻 FOOTER
# ======================================================

st.markdown("""
<div class="footer">
    <div class="footer-content">
        <div class="footer-text">© 2025 ETA Predictor. Built with ❤️ for Indore.</div>
        <div class="footer-links">
            <a href="#" class="footer-link">Privacy Policy</a>
            <a href="#" class="footer-link">Terms of Service</a>
            <a href="#" class="footer-link">GitHub</a>
            <a href="#" class="footer-link">Support</a>
        </div>
        <div class="footer-text" style='font-size:0.8rem;opacity:0.7;'>Powered by CatBoost ML • Streamlit</div>
    </div>
</div>
""", unsafe_allow_html=True)
