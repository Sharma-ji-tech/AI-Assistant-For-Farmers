import streamlit as st
import requests
from groq import Groq
import pickle
import numpy as np
import base64
from PIL import Image
import io

from dotenv import load_dotenv
import os

load_dotenv()

WEATHER_API_KEY = os.getenv("WEATHER_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

st.set_page_config(
    page_title="Farmer Assistant ML & AI",
    page_icon="🌾",
    layout="wide"
)

# ── PREMIUM CSS STYLING ─────────────────────────────────────────
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;800&display=swap');
    
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
    }
    
    /* Main Background & Glassmorphism */
    .stApp {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.8) !important;
        backdrop-filter: blur(10px);
        box-shadow: 2px 0 10px rgba(0,0,0,0.1);
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(90deg, #11998e 0%, #38ef7d 100%);
        color: white;
        border: none;
        border-radius: 30px;
        font-weight: 600;
        padding: 0.5rem 1rem;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        color: white;
    }
    
    /* Text Inputs and Selectboxes */
    .stTextInput>div>div>input, .stSelectbox>div>div>div, .stNumberInput>div>div>input, .stTextArea>div>div>textarea {
        border-radius: 15px;
        border: 1px solid #e0e0e0;
        padding: 10px;
        background: white;
    }
    
    /* Form Background */
    [data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.6);
        border-radius: 20px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.8);
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.1);
        backdrop-filter: blur(4px);
    }
</style>
""", unsafe_allow_html=True)

# ── LANGUAGE CONFIG ────────────────────────────────────────────
LANGUAGES = {
    "English":  "en",
    "Hindi":    "hi",
    "Marathi":  "mr",
    "Gujarati": "gu",
    "Telugu":   "te",
    "Tamil":    "ta",
    "Punjabi":  "pa",
}

LANG_NAMES = {
    "en": "English", "hi": "Hindi", "mr": "Marathi",
    "gu": "Gujarati", "te": "Telugu", "ta": "Tamil", "pa": "Punjabi",
}

# ── WEATHER ────────────────────────────────────────────────────
def get_weather(city):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        data = requests.get(url).json()
        temp     = round(data["main"]["temp"], 1)
        feels    = round(data["main"]["feels_like"], 1)
        humidity = data["main"]["humidity"]
        condition= data["weather"][0]["description"].capitalize()
        return f"Weather in {city.capitalize()}: {temp}°C (feels like {feels}°C), {condition}, Humidity: {humidity}%"
    except:
        return f"Could not fetch weather for {city}."

def get_city_from_query(q):
    cities = [
        "mumbai","pune","nagpur","nashik","aurangabad","delhi","kolhapur",
        "solapur","amravati","latur","satara","sangli","jalgaon","akola",
        "nanded","hyderabad","chennai","bangalore","ahmedabad","chandigarh",
        "surat","jaipur","lucknow","patna","bhopal","indore"
    ]
    q_lower = q.lower()
    for city in cities:
        if city in q_lower:
            return city.capitalize()
    return None

# ── AI MODELS & API ────────────────────────────────────────────
def get_ai_response(user_message, chat_history, lang_code):
    client = Groq(api_key=GROQ_API_KEY)
    lang_name = LANG_NAMES.get(lang_code, "English")

    system_prompt = f"""You are an expert Farmer Assistant Chatbot for Indian farmers, especially in Maharashtra and across India.

Your role:
- Answer ALL farming-related questions with accurate, practical advice
- Topics: crops, fertilizers, soil, irrigation, pest control, diseases, government schemes, market prices, weather interpretation, sowing/harvest timing, organic farming, equipment, storage, and more
- You understand Indian farming context — Maharashtra regions, Indian government schemes, Indian crops, and local farming practices

Language rules:
- ALWAYS respond in {lang_name} language
- Match the language the user is writing in, but default to {lang_name}

Response style:
- Be warm, helpful, and simple — farmers may not be highly educated
- Give practical, actionable advice
- Use bullet points for clarity
- Keep responses concise but complete
- If asked about weather, use the weather data provided
- Remember previous messages in the conversation for follow-up questions
"""

    messages = []
    history_to_send = chat_history[-20:] if len(chat_history) > 20 else chat_history
    for msg in history_to_send:
        if msg["role"] == "user":
            messages.append({"role": "user", "content": msg["text"]})
        elif msg["role"] == "bot":
            messages.append({"role": "assistant", "content": msg["text"]})
            
    messages.append({"role": "user", "content": user_message})
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            max_tokens=1024,
            messages=[{"role": "system", "content": system_prompt}] + messages
        )
        return response.choices[0].message.content
    except Exception as e:
        return f" Error: {str(e)}"

def process_query(user_message, chat_history, lang_code):
    q = user_message.lower()
    weather_keywords = ["weather","temperature","humidity","rain","forecast",
                        "मौसम","तापमान","हवामान","வானிலை","వాతావరణం","ਮੌਸਮ"]
    if any(w in q for w in weather_keywords):
        city = get_city_from_query(q)
        if city:
            weather_data = get_weather(city)
            enriched = f"{user_message}\n\n[Live weather data: {weather_data}]"
            return get_ai_response(enriched, chat_history, lang_code)
    return get_ai_response(user_message, chat_history, lang_code)

def get_symptom_analysis(symptom_text, crop_name, lang_code):
    client = Groq(api_key=GROQ_API_KEY)
    lang_name = LANG_NAMES.get(lang_code, "English")
    
    system_prompt = f"""You are an expert plant pathologist and agricultural advisor for Indian farmers. 
A farmer will describe the symptoms of a diseased crop. 
Analyze the symptoms and provide:
1. Identifying the likely disease or pest.
2. Immediate practical treatments (chemical, organic, or cultural).
3. Preventive measures to avoid it in the future.
ALWAYS respond in {lang_name} language. Be supportive and clear."""

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Crop: {crop_name}\nSymptoms: {symptom_text}"}
            ],
            temperature=0.5,
            max_tokens=1024
        )
        return response.choices[0].message.content
    except Exception as e:
        return f" Error diagnosing symptoms: {str(e)}"

# ── ML MODELS ────────────────────────────────────────────
@st.cache_resource
def load_ml_models():
    try:
        with open('crop_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('soil_encoder.pkl', 'rb') as f:
            soil_enc = pickle.load(f)
        with open('rain_encoder.pkl', 'rb') as f:
            rain_enc = pickle.load(f)
            
        with open('yield_model.pkl', 'rb') as f:
            ymodel = pickle.load(f)
        with open('yield_encoder.pkl', 'rb') as f:
            yenc = pickle.load(f)
            
        return model, soil_enc, rain_enc, ymodel, yenc
    except Exception as e:
        return None, None, None, None, None

rf_model, soil_encoder, rain_encoder, yield_model, yield_encoder = load_ml_models()

def predict_crop_ml(city, soil_type, rainfall):
    try:
        url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={WEATHER_API_KEY}&units=metric"
        response = requests.get(url)
        data = response.json()
        if response.status_code != 200:
            return None, "City not found. Please check spelling."
        temp = data["main"]["temp"]
        humidity = data["main"]["humidity"]
    except:
        return None, "Error connecting to Weather API"
        
    try:
        soil_enc = soil_encoder.transform([soil_type])[0]
        rain_enc = rain_encoder.transform([rainfall])[0]
        features = np.array([[soil_enc, temp, humidity, rain_enc]])
        prediction = rf_model.predict(features)[0]
        return prediction, {"temp": temp, "humidity": humidity}
    except Exception as e:
        return None, f"Error in Machine Learning Prediction: {e}"

# ── QUICK QUESTIONS ────────────────────────────────────────────
QUICK_QS = {
    "en": ["Weather in Pune","Best kharif crops","PM Kisan scheme details","Drip irrigation subsidy","How to do soil test","Cotton pest control"],
    "hi": ["पुणे का मौसम","खरीफ फसलें","पीएम किसान योजना","ड्रिप सिंचाई सब्सिडी","मिट्टी परीक्षण","कपास में कीट"],
    "mr": ["पुण्याचे हवामान","खरीप पिके","पीएम किसान","ठिबक सिंचन अनुदान","माती परीक्षण","कापूस कीड"]
}
CATEGORIES = [
    ("🌾", "Crops"),("🌦️", "Weather"),("🧪", "Fertilizer"),
    ("💧", "Irrigation"),("🐛", "Pest Control"),("🏛️", "Gov Schemes")
]

# ── UI ─────────────────────────────────────────────────────────

st.sidebar.title("🌾 Farmer Portal")
page = st.sidebar.radio("Navigate Apps", [
    "💬 AI Chatbot",
    "🌱 ML Crop Recommendation",
    "📈 Crop Yield Estimator",
    "🩺 AI Symptom Checker"
])
st.sidebar.markdown("---")
st.sidebar.subheader("⚙️ Global Settings")
selected_lang_name = st.sidebar.selectbox("🌐 Language", list(LANGUAGES.keys()), index=0)
selected_lang = LANGUAGES[selected_lang_name]

if page == "💬 AI Chatbot":
    col1, col2 = st.columns([1, 3])
    with col1:
        st.markdown("### 📂 Categories")
        for icon, label in CATEGORIES:
            if st.button(f"{icon} {label}", use_container_width=True, key=f"cat_{label}"):
                st.session_state.quick_input = f"Tell me about {label} in farming"

        st.markdown("---")
        st.markdown("**Quick Questions**")
        for q in QUICK_QS.get(selected_lang, QUICK_QS["en"]):
            if st.button(q, key=f"qq_{q}", use_container_width=True):
                st.session_state.quick_input = q

        st.markdown("---")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.session_state.quick_input = ""
            st.rerun()

    with col2:
        st.title("🌾 Farmer Assistant Chatbot ")
        st.caption("Powered by ABHISHEK SHARMA")
        st.markdown("---")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
        if "quick_input" not in st.session_state:
            st.session_state.quick_input = ""

        if not st.session_state.chat_history:
            greetings = {
                "en": "Hello! I am your AI-powered Farmer Assistant. Ask me anything about farming!",
                "hi": "नमस्ते! मैं आपका AI कृषि सहायक हूँ। खेती से जुड़ा कोई भी सवाल पूछें!",
                "mr": "नमस्कार! मी तुमचा AI शेती सहाय्यक आहे। शेतीसंबंधी कोणताही प्रश्न विचारा!"
            }
            st.session_state.chat_history.append({
                "role": "bot",
                "text": greetings.get(selected_lang, greetings["en"])
            })

        for chat in st.session_state.chat_history:
            if chat["role"] == "user":
                with st.chat_message("user"):
                    st.markdown(chat["text"])
            else:
                with st.chat_message("assistant", avatar="🌾"):
                    st.markdown(chat["text"])

        user_input = st.chat_input("Ask any farming question in any language...")

        if st.session_state.quick_input:
            user_input = st.session_state.quick_input
            st.session_state.quick_input = ""

        if user_input:
            st.session_state.chat_history.append({"role": "user", "text": user_input})

            with st.spinner("🌾 Thinking..."):
                response = process_query(
                    user_input,
                    st.session_state.chat_history[:-1], 
                    selected_lang
                )

            st.session_state.chat_history.append({"role": "bot", "text": response})
            st.rerun()

elif page == "🌱 ML Crop Recommendation":
    st.title("🌱 Smart Crop Recommendation (Machine Learning)")
    st.markdown("We use a **Random Forest Classifier** trained on agricultural data to predict the best crop for your farm. We automatically fetch current Weather data based on your city.")
    
    if not rf_model:
        st.error("⚠️ The Machine Learning models are not trained yet! Please run `python train_model.py` first.")
    else:
        with st.form("ml_form"):
            colA, colB = st.columns(2)
            with colA:
                city = st.text_input("🏙️ Enter your City (e.g., Pune, Nagpur):", "Pune")
            with colB:
                soil_type = st.selectbox("🌍 Select your Soil Type:", ["Black", "Red", "Alluvial", "Loamy", "Sandy"])
            
            rainfall = st.selectbox("🌧️ Expected Rainfall Level:", ["Low", "Moderate", "Heavy"])
            submit_ml = st.form_submit_button("Predict Best Crop Using ML 🚀", type="primary")

        if submit_ml:
            with st.spinner("🤖 Running Machine Learning Model..."):
                crop, info = predict_crop_ml(city, soil_type, rainfall)
                if crop is None:
                    st.error(info)
                else:
                    st.success(f"### 🎉 Recommended Crop to Plant: **{crop}**")
                    st.info(f"**ML Input Features:**\n- 🌍 Soil: {soil_type}\n- 🌧️ Rainfall: {rainfall}\n- 🌡️ Temperature: {info['temp']}°C (Auto-Fetched)\n- 💧 Humidity: {info['humidity']}% (Auto-Fetched)")

elif page == "🩺 AI Symptom Checker":
    st.title("🩺 Plant Disease Symptom Checker (AI)")
    st.markdown("Identify crop diseases instantly! Describe the symptoms you see on your plants, and our AI will diagnose the issue and provide practical treatments.")
    
    with st.form("symptom_form"):
        crop_name = st.text_input("🌾 Which crop is affected? (e.g., Tomato, Cotton, Wheat, Rice)")
        symptoms = st.text_area("✍️ Describe the symptoms you see:", 
                               placeholder="e.g., Yellowing leaves with brown spots, white powder on the stems, wilting during daytime...")
        submit_symptoms = st.form_submit_button("Diagnose Disease 🩺", type="primary")
        
    if submit_symptoms:
        if not crop_name or not symptoms:
            st.error("⚠️ Please provide both the crop name and a description of the symptoms.")
        else:
            with st.spinner("🔬 AI is analyzing the symptoms..."):
                result = get_symptom_analysis(symptoms, crop_name, selected_lang)
                st.markdown("### 📋 AI Diagnosis Report:")
                st.write(result)

elif page == "📈 Crop Yield Estimator":
    st.title("📈 Crop Yield Estimator")
    st.markdown("We use a **Linear Regression** model to predict how many Tonnes of crop you will harvest based on farm size and weather.")
    
    if not yield_model:
        st.error("⚠️ The Regression model is not trained yet! Please run `python train_yield_model.py`.")
    else:
        with st.form("yield_form"):
            col1, col2 = st.columns(2)
            with col1:
                crop_select = st.selectbox("🌾 Select Crop:", ["Wheat", "Rice", "Cotton", "Sugarcane", "Soybean"])
                farm_size = st.number_input("📏 Farm Size (Acres):", min_value=1.0, value=5.0)
            with col2:
                rainfall_mm = st.number_input("🌧️ Expected Rainfall (mm/year):", min_value=100.0, value=800.0)
                temp_c = st.number_input("🌡️ Average Temperature (°C):", min_value=10.0, value=25.0)
            
            submit_yield = st.form_submit_button("Predict Yield 🚜", type="primary")
            
        if submit_yield:
            with st.spinner("🤖 Running Regression Model..."):
                try:
                    c_enc = yield_encoder.transform([crop_select])[0]
                    features = np.array([[c_enc, farm_size, rainfall_mm, temp_c]])
                    pred_yield = yield_model.predict(features)[0]
                    st.success(f"### 🚜 Estimated Harvest: **{pred_yield:.2f} Tonnes**")
                    st.info(f"*(This prediction uses Scikit-Learn Linear Regression based on your {farm_size} acres of {crop_select}.)*")
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
