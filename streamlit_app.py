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
comments_to_analyze = []

if is_bulk:
    tab1, tab2 = st.tabs(["✍️ Metin Girişi", "📁 Dosya Yükle (CSV/Excel)"])
    
    with tab1:
        text_input = st.text_area(
            "Yorumları alt alta girin:",
            height=200,
            placeholder="Örn: Harika uygulama!\nKötü performans..."
        )
        if text_input.strip():
            raw_lines = text_input.split('\n')
            comments_to_analyze = [line.strip() for line in raw_lines if len(line.strip()) > 2]
            
    with tab2:
        uploaded_files = st.file_uploader("CSV veya Excel dosyaları yükleyin", type=["csv", "xlsx"], accept_multiple_files=True)
        if uploaded_files:
            all_comments = []
            for uploaded_file in uploaded_files:
                df_upload = None
                try:
                    if uploaded_file.name.endswith('.csv'):
                        for encoding in ['utf-8', 'utf-16', 'latin-1', 'cp1252']:
                            try:
                                uploaded_file.seek(0)
                                # Automatic delimiter detection with fallback to common ones
                                df_upload = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
                                if len(df_upload.columns) <= 1: # Delimiter detection failed
                                    uploaded_file.seek(0)
                                    df_upload = pd.read_csv(uploaded_file, encoding=encoding, sep=';')
                                break
                            except Exception: continue
                    else:
                        df_upload = pd.read_excel(uploaded_file)
                    
                    if df_upload is not None:
                        st.write(f"📂 **Dosya İşleniyor:** {uploaded_file.name}")
                        
                        # --- Enhanced Smart Column Detection ---
                        # 1. Target keywords
                        target_keys = ["review text", "yorum metni", "yorum", "review", "text", "metin", "content", "body"]
                        # 2. Avoidance keywords
                        avoid_keys = ["language", "dil", "id", "name", "isim", "code", "vers", "date", "tarih", "star", "rating", "device", "millis", "epoch", "app"]
                        
                        scores = []
                        for col in df_upload.columns:
                            col_lower = col.lower()
                            score = 0
                            # Keyword matching
                            if any(tk in col_lower for tk in target_keys): score += 10
                            if any(ak in col_lower for ak in avoid_keys): score -= 15
                            
                            # Content estimation (Check first 5 rows)
                            sample = df_upload[col].head(5).astype(str).tolist()
                            avg_len = sum(len(s) for s in sample) / len(sample) if sample else 0
                            if avg_len > 15: score += 10 # Likely a real sentence/comment
                            elif avg_len < 5: score -= 10 # Likely short codes or names
                            
                            scores.append((score, col))
                        
                        # Get the column with the highest score
                        scores.sort(key=lambda x: x[0], reverse=True)
                        best_col = scores[0][1] if scores else df_upload.columns[0]
                        default_index = list(df_upload.columns).index(best_col)

                        col_name = st.selectbox(
                            f"Analiz edilecek sütun ({uploaded_file.name}):", 
                            df_upload.columns, 
                            index=default_index,
                            key=f"col_{uploaded_file.name}"
                        )
                        
                        if col_name:
                            def is_valid_comment(text):
                                text = str(text).strip()
                                # 1. Too short
                                if len(text) < 4: return False
                                # 2. Common metadata/null values
                                if text.lower() in ['nan', 'null', 'none', 'tr', 'en']: return False
                                # 3. ISO Timestamps (e.g. 2026-03-01T...)
                                if re.match(r'^\d{4}-\d{2}-\d{2}.*', text): return False
                                # 4. Pure numeric or ID-like values
                                if text.replace('.', '').replace('-', '').isdigit(): return False
                                # 5. Simple date patterns (e.g. 01.01.2024 or 2024/01/01)
                                if re.match(r'^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$', text): return False
                                
                                return True

                            valid_comments = [
                                str(c).strip() for c in df_upload[col_name].dropna() 
                                if is_valid_comment(c)
                            ]
                            all_comments.extend(valid_comments)
                            with st.expander(f"👀 {uploaded_file.name} Önizleme (Seçilen: {col_name})"):
                                st.write(valid_comments[:5])
                except Exception as e:
                    st.error(f"⚠️ {uploaded_file.name} okuma hatası: {e}")
            
            if all_comments:
                comments_to_analyze = all_comments
                st.success(f"📋 Toplam **{len(comments_to_analyze)}** gerçek yorum analiz için hazır!")
else:
    text_input = st.text_input("Analiz edilecek metni girin:", placeholder="Örn: Bugün harika bir gün!")
    if text_input:
        comments_to_analyze = [text_input]

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
if st.button("Duygu Durumunu Analiz Et", use_container_width=True):
    if not comments_to_analyze:
        st.warning("Lütfen analiz edilecek bir metin girin veya dosya yükleyin.")
    else:
        if is_bulk:
            bulk_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, comment in enumerate(comments_to_analyze):
                status_text.text(f"Analiz ediliyor ({i+1}/{len(comments_to_analyze)})...")
                res = get_gemini_sentiment(comment) or heuristic_analysis(comment)
                scores = {"Pozitif": res['positive'], "Negatif": res['negative'], "Nötr": res['neutral']}
                verdict = max(scores, key=scores.get)
                
                bulk_results.append({
                    "No": i + 1, "Yorum": comment, "Baskın Duygu": verdict,
                    "Pozitif %": f"{res['positive']:.2%}", "Nötrlük %": f"{res['neutral']:.2%}", "Negatiflik %": f"{res['negative']:.2%}"
                })
                progress_bar.progress((i + 1) / len(comments_to_analyze))
            
            st.session_state.bulk_results = bulk_results
            status_text.success("Analiz Başarıyla Tamamlandı!")
        else:
            # Single analysis logic
            with st.spinner("Analiz ediliyor..."):
                comment = comments_to_analyze[0]
                result = get_gemini_sentiment(comment) or heuristic_analysis(comment)
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
    .neon-pos { border: 1px solid #2ecc71; box-shadow: 0 0 3px #2ecc71; padding: 12px; border-radius: 8px; margin: 8px 0; background: rgba(46, 204, 113, 0.05); }
    .neon-neg { border: 1px solid #e74c3c; box-shadow: 0 0 3px #e74c3c; padding: 12px; border-radius: 8px; margin: 8px 0; background: rgba(231, 76, 60, 0.05); }
    .neon-neu { border: 1px solid #3498db; box-shadow: 0 0 3px #3498db; padding: 12px; border-radius: 8px; margin: 8px 0; background: rgba(52, 152, 219, 0.05); }
    .normal-card { border: 1px solid #333; padding: 12px; border-radius: 8px; margin: 8px 0; background: #1e1e1e; opacity: 0.9; }
    </style>
    """, unsafe_allow_html=True)

    _, col_pie, _ = st.columns([1, 2, 1])
    with col_pie:
        pie_data = pd.DataFrame({"Duygu": counts.index, "Sayı": counts.values})
        fig = px.pie(pie_data, values='Sayı', names='Duygu', hole=0.5,
                     color='Duygu', color_discrete_map={'Pozitif':'#2ecc71', 'Negatif':'#e74c3c', 'Nötr':'#3498db'})
        fig.update_traces(pull=[0.05, 0.05, 0.05], textinfo='percent+label')
        fig.update_layout(margin=dict(t=10, b=10, l=10, r=10), height=250, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # Comments List
    st.write("### 💬 Yorum Listesi")
    
    t_pos = counts.get('Pozitif', 0)
    t_neg = counts.get('Negatif', 0)
    t_neu = counts.get('Nötr', 0)
    t_all = len(df)

    tab_all, tab_pos, tab_neg, tab_neu = st.tabs([
        f"🌐 Tümü ({t_all})", 
        f"🟢 Pozitif ({t_pos})", 
        f"🔴 Negatif ({t_neg})", 
        f"🔵 Nötr ({t_neu})"
    ])
    
    def display_comments(filtered_df, highlight=True):
        if filtered_df.empty:
            st.info("Bu kategoride henüz yorum bulunmuyor.")
            return
        for _, row in filtered_df.iterrows():
            sentiment = row["Baskın Duygu"]
            cls = "normal-card"
            if highlight:
                cls = "neon-pos" if sentiment == "Pozitif" else ("neon-neg" if sentiment == "Negatif" else "neon-neu")
            
            st.markdown(f"""
            <div class="{cls}">
                <span style="font-size: 0.8em; color: #aaa;">#{row['No']} | {sentiment}</span><br>
                {row['Yorum']}
            </div>
            """, unsafe_allow_html=True)

    with tab_all:
        display_comments(df, highlight=False)
    
    with tab_pos:
        display_comments(df[df["Baskın Duygu"] == "Pozitif"])
        
    with tab_neg:
        display_comments(df[df["Baskın Duygu"] == "Negatif"])
        
    with tab_neu:
        display_comments(df[df["Baskın Duygu"] == "Nötr"])

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

