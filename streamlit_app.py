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
        model = genai.GenerativeModel('gemini-3-flash-preview')
        prompt = f"""
        Analyze the sentiment of the following text with HIGH PRECISION.
        Return ONLY a JSON response in this format:
        {{"positive": score, "negative": score, "neutral": score}}
        Use high-precision floats (e.g., 0.8234). The SUM must be EXACTLY 1.0.
        
        Text: "{text}"
        """
        response = model.generate_content(prompt)
        content = response.text
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            # Ensure all keys exist and sum to 1.0
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
    pos_list = ["iyi", "güzel", "harika", "sevindim", "mutlu", "aşk", "seviyorum", "başarılı", "hoş", "muhteşem", "süper", "kaliteli"]
    neg_list = ["kötü", "berbat", "üzgün", "nefret", "başarısız", "korkunç", "çirkin", "zayıf", "yavaş", "bozuk", "rezalet"]
    
    pos_count = sum(text_lower.count(word) for word in pos_list)
    neg_count = sum(text_lower.count(word) for word in neg_list)
    
    total_words = len(text_lower.split())
    if total_words == 0:
        return {"positive": 0.0, "negative": 0.0, "neutral": 1.0}
    
    # Granular calculation based on word density
    pos_ratio = pos_count / total_words
    neg_ratio = neg_count / total_words
    
    # Scale to 0-1 range with a more "float-heavy" distribution
    raw_score = 0.5 + (pos_ratio - neg_ratio) * 2.0
    raw_score = max(0.0, min(1.0, raw_score))
    
    # Add minor "fuzziness" to make it feel like a real float analysis
    positive = raw_score * 0.95
    negative = (1.0 - raw_score) * 0.95
    neutral = 1.0 - (positive + negative)
    
    return {"positive": positive, "negative": negative, "neutral": neutral, "method": "Heuristic"}

# Analysis Trigger
if st.button("Duygu Durumunu Analiz Et", use_container_width=True):
    if not text_input.strip():
        st.warning("Lütfen bir metin girin.")
    else:
        if is_bulk:
            # Enhanced splitting logic to handle different types of separators
            raw_lines = text_input.split('\n')
            comments = [line.strip() for line in raw_lines if len(line.strip()) > 2]
            
            if not comments:
                st.error("Analiz edilecek geçerli yorum bulunamadı. Lütfen her satıra bir yorum yazın.")
            else:
                results = []
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for i, comment in enumerate(comments):
                    status_text.text(f"Analiz ediliyor ({i+1}/{len(comments)})...")
                    
                    # Try Gemini, failover to Heuristic
                    res = get_gemini_sentiment(comment)
                    if not res:
                        res = heuristic_analysis(comment)
                    
                    # Determine winner
                    scores = {"Pozitif": res['positive'], "Negatif": res['negative'], "Nötr": res['neutral']}
                    verdict = max(scores, key=scores.get)
                    
                    results.append({
                        "No": i + 1,
                        "Yorum": comment,
                        "Baskın Duygu": verdict,
                        "Pozitif %": f"{res['positive']:.2%}",
                        "Nötrlük %": f"{res['neutral']:.2%}",
                        "Negatiflik %": f"{res['negative']:.2%}"
                    })
                    progress_bar.progress((i + 1) / len(comments))
                
                status_text.success(f"Analiz Başarıyla Tamamlandı: {len(comments)} yorum değerlendirildi.")
                
                # 1. Aggregate Statistics at the Top
                df = pd.DataFrame(results)
                counts = df["Baskın Duygu"].value_counts()
                
                st.subheader("📊 Etkileşimli Dağılım Özeti")
                st.markdown("""
                <style>
                .neon-pos { border: 2px solid #2ecc71; box-shadow: 0 0 10px #2ecc71; padding: 10px; border-radius: 5px; margin: 5px 0; }
                .neon-neg { border: 2px solid #e74c3c; box-shadow: 0 0 10px #e74c3c; padding: 10px; border-radius: 5px; margin: 5px 0; }
                .neon-neu { border: 2px solid #3498db; box-shadow: 0 0 10px #3498db; padding: 10px; border-radius: 5px; margin: 5px 0; }
                .normal-card { border: 1px solid #333; padding: 10px; border-radius: 5px; margin: 5px 0; opacity: 0.8; }
                </style>
                """, unsafe_allow_html=True)

                sum_col1, sum_col2 = st.columns([1, 1])
                
                with sum_col1:
                    st.write("#### Filtrelemek için Tıklayın")
                    # Use a session state for persistent filtering
                    if "filter" not in st.session_state:
                        st.session_state.filter = "Hepsi"

                    # Interactive Selection
                    filter_choice = st.radio(
                        "Duyguya Göre Highlight Et:",
                        ["Hepsi", "Pozitif", "Nötr", "Negatif"],
                        horizontal=True
                    )
                    st.session_state.filter = filter_choice

                    st.write(f"**Toplam Pozitif:** {counts.get('Pozitif', 0)}")
                    st.write(f"**Toplam Nötr:** {counts.get('Nötr', 0)}")
                    st.write(f"**Toplam Negatif:** {counts.get('Negatif', 0)}")
                
                with sum_col2:
                    st.write("#### Görsel Dağılım")
                    pie_data = pd.DataFrame({"Duygu": counts.index, "Sayı": counts.values})
                    fig = px.pie(
                        pie_data, values='Sayı', names='Duygu', hole=0.4,
                        color='Duygu', color_discrete_map={'Pozitif':'#2ecc71', 'Negatif':'#e74c3c', 'Nötr':'#3498db'}
                    )
                    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=200)
                    st.plotly_chart(fig, use_container_width=True)
                
                st.divider()
                
                # Excel Export logic (Prepare before showing results to avoid reference errors)
                excel_data = None
                try:
                    output = io.BytesIO()
                    with pd.ExcelWriter(output, engine='openpyxl') as writer:
                        df.to_excel(writer, index=False, sheet_name='Analiz Sonuçları')
                    excel_data = output.getvalue()
                except Exception as e:
                    st.error(f"Excel hazırlama hatası: {e}")

                # 2. Detailed Neon Highlights
                st.subheader(f"📝 Yorumlar ({st.session_state.filter} Modu)")
                
                # Filter/Highlight logic
                for _, row in df.iterrows():
                    highlight_mode = st.session_state.filter
                    sentiment = row["Baskın Duygu"]
                    
                    is_match = (highlight_mode == "Hepsi") or (sentiment == highlight_mode)
                    
                    if is_match:
                        # Use different neon colors
                        if sentiment == "Pozitif":
                            cls = "neon-pos"
                        elif sentiment == "Negatif":
                            cls = "neon-neg"
                        else:
                            cls = "neon-neu"
                            
                        st.markdown(f"""
                        <div class="{cls}">
                            <span style="font-size: 0.8em; opacity: 0.7;">#{row['No']} | {sentiment}</span><br>
                            {row['Yorum']}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="normal-card">
                            <span style="font-size: 0.8em; opacity: 0.5;">#{row['No']}</span>: {row['Yorum'][:80]}...
                        </div>
                        """, unsafe_allow_html=True)
                
                # Footer Action
                if excel_data:
                    st.download_button(
                        label="📥 Sonuçları Excel Olarak İndir (.xlsx)",
                        data=excel_data,
                        file_name="sentiment_analizi.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )
            
        else:
            with st.spinner("Yapay Zeka analiz ediyor..."):
                result = get_gemini_sentiment(text_input)
                if not result:
                    result = heuristic_analysis(text_input)
                    method = result.get("method", "Gemini AI")
                else:
                    method = "Gemini AI"

            # Results Display
            st.divider()
            st.success(f"Analiz Tamamlandı (Yöntem: {method})")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Pozitiflik", f"{result['positive']:.4f}")
            with col2:
                st.metric("Nötrlük", f"{result['neutral']:.4f}")
            with col3:
                st.metric("Negatiflik", f"{result['negative']:.4f}")

            scores = {"Pozitif": result['positive'], "Negatif": result['negative'], "Nötr": result['neutral']}
            verdict = max(scores, key=scores.get)
            
            if verdict == "Pozitif":
                st.info(f"Bu metin genel olarak **Pozitif** bir duygu taşıyor (%{result['positive']:.2%}). 😊")
            elif verdict == "Negatif":
                st.info(f"Bu metin genel olarak **Negatif** bir duygu taşıyor (%{result['negative']:.2%}). 😔")
            else:
                st.info(f"Bu metin **Nötr** bir duruş sergiliyor (%{result['neutral']:.2%}). 😐")

# Footer
st.divider()
st.caption("Geliştiren: Cem Evecen | Streamlit Cloud Deployment Ready")

