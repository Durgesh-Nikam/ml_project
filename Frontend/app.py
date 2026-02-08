import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import json
from PIL import Image
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier
from keras.models import load_model


# # -------- Disease Model Loading --------
@st.cache_resource
def load_cnn_model():
    try:
        model = load_model(
            r"../Models/Disease/plant_disease_model.keras",
            compile=False,
            safe_mode=False
        )

        with open("../Models/Disease/class_indices.json", "r") as f:
            class_indices = json.load(f)

        inv_class_indices = {v: k for k, v in class_indices.items()}
        return model, inv_class_indices

    except Exception as e:
        st.error("❌ ERROR LOADING CNN MODEL")
        st.exception(e)
        return None, None
    
model, inv_class_indices = load_cnn_model()





# -------- Crop Model Loading --------
@st.cache_resource
def load_crop_model():
    model = pickle.load(
        open("../Models/Crop_Recommandation/crop_model.pkl", "rb")
    )

    with open("../Models/Crop_Recommandation/crop_mapping.json", "r") as f:
        crop_mapping = json.load(f)

    crop_mapping = {int(k): v for k, v in crop_mapping.items()}
    return model, crop_mapping

crop_model, crop_mapping = load_crop_model()




# -------- Fertilizer Model Loading --------
@st.cache_resource
def load_fertilizer_model():
    model = pickle.load(open("../Models/All_Trained_Models/fertilizer_recommendation.pkl", "rb"))
    with open("../Models/Fertilizer_Recommandation/Model/fert_mapping.json") as f:
        fert_mapping = json.load(f)
    fert_mapping = {int(k): v for k, v in fert_mapping.items()}
    return model, fert_mapping

fert_model, fert_mapping = load_fertilizer_model()





# -------- Regional Dataset Loading --------
@st.cache_data
def load_regional_data():
    return pd.read_csv("../Models/Crop_Recommandation/Dataset/Regional_Data.csv") 

regional_df = load_regional_data()








# -------- Disease Info Function --------
# groq API
# 

from groq import Groq

client = Groq(api_key="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")  # API key read from env variable


# ===== Chat session state =====
if "messages" not in st.session_state:
    st.session_state.messages = [
        {"role": "system", "content": "You are an expert plant disease consultant."}
    ]

if "auto_explained" not in st.session_state:
    st.session_state.auto_explained = False

def get_disease_info_chat(user_input, language="English"):
    language_system_prompt = {
        "English": "You must answer ONLY in English.",
        "Hindi": "आपको उत्तर केवल हिंदी में देना है।",
        "Marathi": "तुम्हाला उत्तर फक्त मराठीत द्यायचे आहे."
    }

    # 🔹 Update system message dynamically
    st.session_state.messages[0] = {
        "role": "system",
        "content": (
            "You are an expert plant disease consultant.\n"
            + language_system_prompt.get(language)
        )
    }

    # 🔹 Add user message WITHOUT language instruction
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=st.session_state.messages,
            temperature=0.3,
            max_tokens=700
        )
        reply = response.choices[0].message.content
    except Exception:
        reply = "⚠️ AI service temporarily unavailable. Please try again later."

    st.session_state.messages.append({
        "role": "assistant",
        "content": reply
    })

    return reply











# -------- UI Layout --------
st.title("🌾 SmartAgroCare AI – Intelligent Crop Monitoring")

tab1, tab2, tab3 = st.tabs(["Crop Disease Prediction", "Crop Recommendation", "Fertilizer Suggestion"])

# -------- Tab 1: Disease Prediction --------
with tab1:
    st.header("Crop Disease Prediction")
    language = st.selectbox("Select language for disease details", ["English", "Hindi", "Marathi"])
    uploaded_file = st.file_uploader("Upload an image of the plant leaf", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image_pil = Image.open(uploaded_file)
        st.image(image_pil, caption="Uploaded Image", width=250)
        img = image_pil.resize((128, 128))
        img_array = image.img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        if st.button("Predict Disease"):

            if model is None:
                st.error("❌ CNN model failed to load. Cannot make prediction.")
            else:
                prediction = model.predict(img_array)
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = inv_class_indices[predicted_class_index]
                confidence = np.max(prediction)

                st.success(f"🧠 Predicted Disease: **{predicted_class_name}**")
                st.info(f"Confidence: {confidence:.2f}")
                with st.spinner("🔍 Getting disease info from expert..."):
                    if not st.session_state.auto_explained:
                        auto_prompt = (
                            f"Explain the plant disease {predicted_class_name} "
                            "with causes, symptoms, treatment and prevention."
                        )
                        get_disease_info_chat(auto_prompt, language)
                        st.session_state.auto_explained = True


    # ================= CHAT UI =================
    st.subheader("💬 Crop AI Assistant")

     # NOW render chat history
    for msg in st.session_state.messages:
        if msg["role"] == "system":
            continue
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


    language = st.selectbox(
        "Language",
        ["English", "Hindi", "Marathi"],
        key="chat_language"
    )        
    
    # Chat input FIRST
    user_input = st.chat_input("Ask a question about crop or disease...")
    
    if user_input:
        with st.spinner("🤖 Thinking..."):
            get_disease_info_chat(user_input, language)
            st.rerun()   # VERY IMPORTANT
   
    # ==========================================

                




# -------- Tab 2: Crop Recommendation --------
with tab2:
    st.header("Crop Recommendation")

    st.subheader("📍 Regional Inputs")
    state = st.selectbox("Select State", regional_df["State_Name"].unique())
    district = st.selectbox("Select District", regional_df[regional_df["State_Name"] == state]["District_Name"].unique())
    season = st.selectbox("Select Season", regional_df["Season"].unique())

    st.subheader("🌱 Soil Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        N = st.number_input("Nitrogen (N)", 0, 200, 90)
        temperature = st.number_input("Temperature (°C)", 0.0, 50.0, 25.0)
        ph = st.number_input("pH", 0.0, 14.0, 6.5)
    with col2:
        P = st.number_input("Phosphorous (P)", 0, 200, 42)
        humidity = st.number_input("Humidity (%)", 0.0, 100.0, 80.0)
    with col3:
        K = st.number_input("Potassium (K)", 0, 200, 43)
        rainfall = st.number_input("Rainfall (mm)", 0.0, 400.0, 200.0)

    if st.button("Recommend Crops"):
        soil_input = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
        pred_probs = crop_model.predict_proba(soil_input)[0]
        top_indices = pred_probs.argsort()[-3:][::-1]
        ml_crops = [crop_mapping[i] for i in top_indices]

        filtered = regional_df[
            (regional_df["State_Name"] == state) &
            (regional_df["District_Name"] == district) &
            (regional_df["Season"] == season)
        ]
        regional_crops = sorted(filtered["Crop"].unique().tolist())

        st.success("✅ Recommended Crops Based on Soil (ML Prediction)")
        for i, crop in enumerate(ml_crops, 1):
            st.markdown(f"**{i}. {crop}**")

        st.success("📍 Regionally Suitable Crops")
        if regional_crops:
            for i, crop in enumerate(regional_crops, 1):
                st.markdown(f"**{i}. {crop}**")
        else:
            st.warning("No regional data found for the selected district and season.")






# -------- Tab 3: Fertilizer Suggestion --------
with tab3:
    st.header("Fertilizer Suggestion")

    st.subheader("🧪 Soil and Weather Inputs")
    col1, col2, col3 = st.columns(3)
    with col1:
        temp = st.number_input("Temperature (°C)", 0, 50, 30)
        nitrogen = st.number_input("Nitrogen", 0, 100, 25)
    with col2:
        humidity = st.number_input("Humidity (%)", 0, 100, 60)
        potassium = st.number_input("Potassium", 0, 100, 20)
    with col3:
        moisture = st.number_input("Moisture", 0, 100, 40)
        phosphorous = st.number_input("Phosphorous", 0, 100, 30)

    soil_type = st.selectbox("Soil Type", ["Sandy", "Loamy", "Black", "Red", "Clayey"])

    if st.button("Recommend Fertilizer"):
        soil_map = {"Sandy": 0, "Loamy": 1, "Black": 2, "Red": 3, "Clayey": 4}
        soil_encoded = soil_map[soil_type]

        fert_input = np.array([[temp, humidity, moisture, soil_encoded, nitrogen, potassium, phosphorous]])
        fert_pred = fert_model.predict(fert_input)[0]
        fertilizer_name = fert_mapping.get(fert_pred, "Unknown")

        st.success(f"🧪 Recommended Fertilizer: **{fertilizer_name}**")
