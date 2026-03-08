import streamlit as st
import google.generativeai as genai
import os
import json
import re
import pandas as pd
import plotly.express as px
import io
from dotenv import load_dotenv

# Load environment variables (for local testing)
load_dotenv(override=True)

# Set Page Config
st.set_page_config(
    page_title="AI Duygu Analizi",
    page_icon="🧠",
    layout="centered"
)

# API Configuration
# Priority: 1. .env (local), 2. st.secrets (cloud)
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    try:
        # Check if we are running on Streamlit Cloud by checking if secrets are available
        API_KEY = st.secrets.get("API_KEY")
    except Exception:
        # Fallback to a plain None if secrets are not accessible
        API_KEY = None

if API_KEY:
    genai.configure(api_key=API_KEY)
    HAS_GEMINI = True
else:
    HAS_GEMINI = False

# Header Design
st.title("🧠 AI Sentiment Analysis (Duygu Analizi)")
st.markdown("""
Bu uygulama, girdiğiniz metnin duygu durumunu (Pozitif/Negatif) yapay zeka kullanarak analiz eder.
Google Gemini AI desteği ile güçlendirilmiştir.
""")

# Analysis Mode Toggle
is_bulk = st.checkbox("Toplu Analiz Modu (Birden fazla yorumu alt alta yapıştırın)", help="Her satırı ayrı bir yorum olarak değerlendirir.")

# Input Section
placeholder = "Örn: Bugün hava çok güzel!\nHarika bir uygulama, bayıldım.\nKargo çok geç geldi, hiç beğenmedim." if is_bulk else "Örn: Bugün hava çok güzel, kendimi harika hissediyorum!"
text_input = st.text_area(
    "Analiz edilecek metni girin:",
    height=200 if is_bulk else 150,
    placeholder=placeholder
)

def get_gemini_sentiment(text):
    if not HAS_GEMINI:
        return None
    try:
        # Using a more stable and widely available model name
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        Analyze the sentiment of the following Turkish text with EXTREME PRECISION.
        
        CRITICAL RULES:
        1. If the text describes app crashes, freezes, or bugs (e.g. 'donuyor', 'kasıyor', 'açılmıyor'), it is HEAVILY NEGATIVE.
        2. Return ONLY a JSON response in this exact format:
           {{"positive": score, "negative": score, "neutral": score}}
        3. Use high-precision floats. The SUM must be EXACTLY 1.0.
        
        Text: "{text}"
        """
        response = model.generate_content(prompt)
        content = response.text
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            p = float(data.get("positive", 0))
            n = float(data.get("negative", 0))
            neu = float(data.get("neutral", 0))
            total = p + n + neu
            if total > 0:
                return {"positive": p/total, "negative": n/total, "neutral": neu/total}
            return data
    except Exception:
        return None
    return None

def heuristic_analysis(text):
    text_lower = text.lower()
    # Expanded word list for more granularity
    pos_list = [
        "iyi", "güzel", "harika", "sevindim", "mutlu", "aşk", "seviyorum", 
        "başarılı", "hoş", "muhteşem", "süper", "kaliteli", "mükemmel", 
        "beğendim", "hızlı", "basit", "kolay", "memnun"
    ]
    neg_list = [
        "kötü", "berbat", "üzgün", "nefret", "başarısız", "korkunç", "çirkin", 
        "zayıf", "yavaş", "bozuk", "rezalet", "donuyor", "kasıyor", "açılmıyor", 
        "hata", "sorun", "çalışmıyor", "berbat", "iğrenç", "beğenmedim", "sil", 
        "çöp", "vakit kaybı", "donma", "kasılma"
    ]
    
    pos_count = sum(text_lower.count(word) for word in pos_list)
    neg_count = sum(text_lower.count(word) for word in neg_list)
    
    total_words = len(text_lower.split())
    if total_words == 0:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0, "method": "Heuristic"}
    
    # Calculate impact weights
    if pos_count > neg_count:
        positive = 0.7 + (pos_count / (pos_count + neg_count + 1)) * 0.2
        negative = 0.05
        neutral = 1.0 - positive - negative
    elif neg_count > pos_count:
        negative = 0.7 + (neg_count / (pos_count + neg_count + 1)) * 0.2
        positive = 0.05
        neutral = 1.0 - negative - positive
    else:
        # If no keywords found, favor Neutral
        positive = 0.15
        negative = 0.15
        neutral = 0.7
    
    return {"positive": positive, "negative": negative, "neutral": neutral, "method": "Heuristic"}

# Analysis Trigger
# Analysis Trigger
if st.button("Duygu Durumunu Analiz Et", use_container_width=True):
    if not text_input.strip():
        st.warning("Lütfen bir metin girin.")
    else:
        if is_bulk:
            raw_lines = text_input.split('\n')
            comments = [line.strip() for line in raw_lines if len(line.strip()) > 2]
            
            if not comments:
                st.error("Analiz edilecek geçerli yorum bulunamadı.")
            else:
                bulk_results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, comment in enumerate(comments):
                    status_text.text(f"Analiz ediliyor ({i+1}/{len(comments)})...")
                    res = get_gemini_sentiment(comment) or heuristic_analysis(comment)
                    scores = {"Pozitif": res['positive'], "Negatif": res['negative'], "Nötr": res['neutral']}
                    verdict = max(scores, key=scores.get)
                    
                    bulk_results.append({
                        "No": i + 1, "Yorum": comment, "Baskın Duygu": verdict,
                        "Pozitif %": f"{res['positive']:.2%}", "Nötrlük %": f"{res['neutral']:.2%}", "Negatiflik %": f"{res['negative']:.2%}"
                    })
                    progress_bar.progress((i + 1) / len(comments))
                
                st.session_state.bulk_results = bulk_results
                status_text.success("Analiz Başarıyla Tamamlandı!")
        else:
            # Single analysis logic preserved
            with st.spinner("Yapay Zeka analiz ediyor..."):
                result = get_gemini_sentiment(text_input) or heuristic_analysis(text_input)
                st.session_state.single_result = result

# --- Persistent Results Display ---
if is_bulk and "bulk_results" in st.session_state:
    df = pd.DataFrame(st.session_state.bulk_results)
    counts = df["Baskın Duygu"].value_counts()
    
    st.divider()
    st.subheader("📊 Etkileşimli Analiz Paneli")
    
    # Styling
    st.markdown("""
    <style>
    .neon-pos { border: 2px solid #2ecc71; box-shadow: 0 0 10px #2ecc71; padding: 12px; border-radius: 8px; margin: 8px 0; background: rgba(46, 204, 113, 0.1); }
    .neon-neg { border: 2px solid #e74c3c; box-shadow: 0 0 10px #e74c3c; padding: 12px; border-radius: 8px; margin: 8px 0; background: rgba(231, 76, 60, 0.1); }
    .neon-neu { border: 2px solid #3498db; box-shadow: 0 0 10px #3498db; padding: 12px; border-radius: 8px; margin: 8px 0; background: rgba(52, 152, 219, 0.1); }
    .normal-card { border: 1px solid #444; padding: 12px; border-radius: 8px; margin: 8px 0; background: #1e1e1e; opacity: 0.8; }
    </style>
    """, unsafe_allow_html=True)

    col_stats, col_pie = st.columns([1, 1])
    with col_stats:
        filter_choice = st.radio(
            "Highlight Edilecek Duygu Seçin:",
            ["Hiçbiri (Hepsi)", "Pozitif", "Nötr", "Negatif"],
            key="bulk_filter",
            horizontal=True
        )
        st.write(f"✅ **Pozitif:** {counts.get('Pozitif', 0)} | 😐 **Nötr:** {counts.get('Nötr', 0)} | ❌ **Negatif:** {counts.get('Negatif', 0)}")
    
    with col_pie:
        pie_data = pd.DataFrame({"Duygu": counts.index, "Sayı": counts.values})
        fig = px.pie(pie_data, values='Sayı', names='Duygu', hole=0.4,
                     color='Duygu', color_discrete_map={'Pozitif':'#2ecc71', 'Negatif':'#e74c3c', 'Nötr':'#3498db'})
        fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=180, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Comments List
    st.write(f"### 💬 Yorum Listesi ({filter_choice})")
    for _, row in df.iterrows():
        sentiment = row["Baskın Duygu"]
        # In 'Hiçbiri' mode, everyone gets normal-card. Otherwise, only match gets neon.
        if filter_choice != "Hiçbiri (Hepsi)" and sentiment == filter_choice:
            cls = "neon-pos" if sentiment == "Pozitif" else ("neon-neg" if sentiment == "Negatif" else "neon-neu")
        else:
            cls = "normal-card"
        
        st.markdown(f"""
        <div class="{cls}">
            <span style="font-size: 0.8em; color: #aaa;">#{row['No']} | {sentiment}</span><br>
            {row['Yorum']}
        </div>
        """, unsafe_allow_html=True)

    # Excel Download
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Analiz Sonuçları')
        st.download_button(label="📥 Sonuçları Excel Olarak İndir", data=output.getvalue(), file_name="analiz.xlsx", key="bulk_dl")
    except: pass

elif not is_bulk and "single_result" in st.session_state:
    result = st.session_state.single_result
    st.divider()
    st.success("Analiz Tamamlandı")
    c1, c2, c3 = st.columns(3)
    c1.metric("Pozitiflik", f"{result['positive']:.4f}")
    c2.metric("Nötrlük", f"{result['neutral']:.4f}")
    c3.metric("Negatiflik", f"{result['negative']:.4f}")
    
    scores = {"Pozitif": result['positive'], "Negatif": result['negative'], "Nötr": result['neutral']}
    verdict = max(scores, key=scores.get)
    st.subheader(f"Sonuç: {verdict}")
    
    if verdict == "Pozitif":
        st.info(f"Bu metin genel olarak **Pozitif** bir duygu taşıyor (%{result['positive']:.2%}). 😊")
    elif verdict == "Negatif":
        st.info(f"Bu metin genel olarak **Negatif** bir duygu taşıyor (%{result['negative']:.2%}). 😔")
    else:
        st.info(f"Bu metin **Nötr** bir duruş sergiliyor (%{result['neutral']:.2%}). 😐")

# Footer
st.divider()
st.caption("Geliştiren: Cem Evecen | Streamlit Cloud Deployment Ready")

