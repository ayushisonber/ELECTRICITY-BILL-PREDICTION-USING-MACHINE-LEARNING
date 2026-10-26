import streamlit as st
import pandas as pd
import joblib
 
st.set_page_config(page_title="Electricity Bill Predictor", page_icon="⚡", layout="wide")
 
# ---------- Fincheck-style colour theme ----------
st.markdown("""
<style>
/* page background: soft lavender */
.stApp { background: #EEECF8; }
/* white sidebar */
[data-testid="stSidebar"] { background: #FFFFFF; }
[data-testid="stHeader"] { background: transparent; }
h1, h2, h3, h4, p, label, span, li, [data-testid="stMarkdownContainer"] { color: #2D3050 !important; }
.card .title, .card .value { color: white !important; }
 
/* dropdowns: white box, dark text */
[data-baseweb="select"] > div { background: #FFFFFF !important; border: 1px solid #D6D3EE; }
[data-baseweb="select"] div, [data-baseweb="select"] span { color: #2D3050 !important; }
[data-baseweb="popover"] li, [data-baseweb="popover"] ul { background: #FFFFFF !important; color: #2D3050 !important; }
 
/* slider: blue instead of red */
[data-baseweb="slider"] div[role="slider"] { background-color: #6C9CF0 !important; box-shadow: none !important; }
[data-testid="stSliderThumbValue"], [data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: #4F86E8 !important; }
 
/* progress track light */
.stProgress > div > div { background-color: #D9D6F0 !important; }
 
/* gradient cards */
.card {
    border-radius: 14px; padding: 18px 20px; color: white;
    box-shadow: 0 8px 18px rgba(120, 120, 180, 0.18);
}
.card .title { font-size: 14px; opacity: 0.95; color: white; }
.card .value { font-size: 28px; font-weight: 700; margin-top: 6px; color: white; }
.teal   { background: linear-gradient(135deg, #4FC9C0, #8BE0D4); }
.blue   { background: linear-gradient(135deg, #6C9CF0, #B9D0FA); }
.pink   { background: linear-gradient(135deg, #EE97BB, #F9CBDD); }
.purple { background: linear-gradient(135deg, #AE9BF0, #D9CEFA); }
 
/* white panel */
.panel {
    background: white; border-radius: 14px; padding: 18px 22px;
    box-shadow: 0 4px 14px rgba(120, 120, 180, 0.12); margin-top: 18px;
}
 
/* blue button */
.stButton > button {
    background: #6C9CF0; color: white; border: none; border-radius: 8px;
    padding: 8px 26px; font-weight: 600;
}
.stButton > button:hover { background: #4F86E8; color: white; }
 
/* blue progress bar */
.stProgress > div > div > div > div { background: #6C9CF0; }
</style>
""", unsafe_allow_html=True)
 
# ---------- Load saved files ----------
model = joblib.load("best_model.pkl")
scaler = joblib.load("scaler.pkl")
columns = joblib.load("columns.pkl")
 
# ---------- Sidebar inputs ----------
st.sidebar.markdown("## ⚡ BillCheck")
st.sidebar.markdown("Enter household details")
 
location = st.sidebar.selectbox("Location", ["Barishal", "Chattogram", "Dhaka", "Khulna",
                                             "Mymensingh", "Rajshahi", "Rangpur", "Savar", "Sylhet"])
area_type = st.sidebar.selectbox("Area Type", ["Urban", "Semi-Urban", "Rural"])
household_size = st.sidebar.slider("Household Size", 1, 12, 5)
rooms = st.sidebar.slider("Rooms", 1, 10, 4)
ac_units = st.sidebar.slider("AC Units", 0, 6, 1)
fan_count = st.sidebar.slider("Fans", 0, 15, 7)
light_count = st.sidebar.slider("Lights", 0, 25, 10)
tv_count = st.sidebar.slider("TVs", 0, 6, 1)
refrigerator = st.sidebar.radio("Refrigerator", ["Yes", "No"], horizontal=True)
predict = st.sidebar.button("Predict Bill")
 
# ---------- Main page ----------
st.title("Electricity Bill Predictor")
st.write("Fill the details on the left and click **Predict Bill**.")
 
if predict:
    row = pd.DataFrame([{
        "Household_Size": household_size,
        "Rooms": rooms,
        "AC_Units": ac_units,
        "Fan_Count": fan_count,
        "Light_Count": light_count,
        "Refrigerator": 1 if refrigerator == "Yes" else 0,
        "TV_Count": tv_count,
        "Location": location,
        "Area_Type": area_type,
    }])
    row = pd.get_dummies(row, columns=["Location", "Area_Type"])
    row = row.reindex(columns=columns, fill_value=0)   # same order as training
 
    bill = max(0.0, float(model.predict(scaler.transform(row))[0]))
    per_person = bill / household_size
    per_room = bill / rooms
    appliances = ac_units + fan_count + light_count + tv_count + (1 if refrigerator == "Yes" else 0)
 
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f'<div class="card teal"><div class="title">Predicted Bill</div><div class="value">৳ {bill:,.0f}</div></div>', unsafe_allow_html=True)
    c2.markdown(f'<div class="card blue"><div class="title">Per Person</div><div class="value">৳ {per_person:,.0f}</div></div>', unsafe_allow_html=True)
    c3.markdown(f'<div class="card pink"><div class="title">Per Room</div><div class="value">৳ {per_room:,.0f}</div></div>', unsafe_allow_html=True)
    c4.markdown(f'<div class="card purple"><div class="title">Total Appliances</div><div class="value">{appliances}</div></div>', unsafe_allow_html=True)
 
    st.markdown('<div class="panel"><b>Bill level</b> (out of ৳ 8,000)</div>', unsafe_allow_html=True)
    st.progress(min(int(bill / 8000 * 100), 100))
    st.caption("Note: the model was trained on a simulated bill column.")
 