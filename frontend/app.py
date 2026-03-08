#fronted/app.py
import streamlit as st
import requests
import os
from dotenv import load_dotenv

load_dotenv()

st.title("🧠 AI Duygu Analizi")
st.write("Metnin duygu durumunu analiz edin")

text = st.text_area(
    "Metni girin:",
    height=150
)

API_URL = "http://127.0.0.1:8000/predict"
API_KEY = os.getenv("API_KEY")

if st.button("Analiz Et"):
    if not text:
        st.warning("Lütfen bir metin girin.")
    else:
        headers = {"x-api-key": API_KEY}
        try:
            response = requests.post(
                API_URL,
                json={"text": text},
                headers=headers
            )

            if response.status_code == 200:
                result = response.json()
                st.success("Analiz tamamlandı")

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Pozitif", f"{result['positive']:.2%}")
                with col2:
                    st.metric("Negatif", f"{result['negative']:.2%}")

                if result['positive'] > result['negative']:
                    st.info("Bu metin genel olarak **Pozitif** bir duygu taşıyor.")
                else:
                    st.info("Bu metin genel olarak **Negatif** bir duygu taşıyor.")
            elif response.status_code == 401:
                st.error("API Anahtarı hatası (Unauthorized)")
            else:
                st.error(f"API hatası: {response.status_code}")
        except Exception as e:
            st.error(f"Sunucuya bağlanılamadı. Lütfen backend'in çalıştığından emin olun. Hata: {e}")


