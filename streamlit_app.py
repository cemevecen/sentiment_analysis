import streamlit as st
import streamlit.components.v1 as components
import google.generativeai as genai
import os
import json
import re
import time
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

# --- PREMIUM STYLING (GLASSMORPHISM) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700&display=swap');

    /* Global Overrides - Light Blue Sweep */
    html, body, .stApp {
        background-color: #F0F9FF !important;
    }
    
    p, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText, .stButton, .stTextInput, .stTextArea {
        font-family: 'Poppins', sans-serif !important;
        color: #1E293B !important;
    }
    
    /* Corrected Global Icon Protection */
    [data-testid="stIcon"], [class*="st-emotion-cache-"], [class*="stIcon"], svg, span[aria-hidden="true"] {
        font-family: inherit !important;
    }

    /* Fix Selectbox & Inputs to Light Blue with DARK TEXT */
    .stSelectbox div[data-baseweb="select"], .stTextInput input, .stTextArea textarea {
        background-color: #F0F9FF !important;
        color: #1E293B !important;
    }
    
    /* Selectbox Popover (Dropdown) fixes */
    [data-baseweb="popover"], [data-baseweb="menu"], [data-baseweb="list"] {
        background-color: #F0F9FF !important;
    }
    [data-baseweb="popover"] * {
        color: #1E293B !important;
        background-color: #F0F9FF !important;
    }
    
    .stSelectbox div[data-baseweb="select"] > div {
        background-color: #F0F9FF !important;
        color: #1E293B !important;
        border-color: #FFE4D6 !important;
    }
    
    /* Ensure no dark boxes in Expanders - Aggressive Fix */
    [data-testid="stExpander"], 
    [data-testid="stExpander"] summary, 
    [data-testid="stExpander"] section {
        background-color: #F0F9FF !important;
        background: #F0F9FF !important;
        border-color: #FFE4D6 !important;
    }
    
    [data-testid="stExpander"] summary {
        border-radius: 12px !important;
        padding: 5px 15px !important;
    }
    
    [data-testid="stExpander"] summary p, 
    [data-testid="stExpander"] summary span:not([data-testid="stIcon"]) {
        font-family: 'Poppins', sans-serif !important;
        color: #1E293B !important;
        background-color: transparent !important;
    }

    /* Target Streamlit's internal header containers for expanders */
    .st-emotion-cache-p5mtransition, .st-emotion-cache-1vt4y6f {
        background-color: #F0F9FF !important;
        color: #1E293B !important;
    }
    
    /* Header Card - Light Blue */
    .header-container {
        background-color: #F0F9FF !important;
        border: 2px solid #E0F2FE;
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 40px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .header-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        color: #6366F1;
        margin-bottom: 15px;
    }
    .header-desc {
        color: #64748b;
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto;
    }
    
    /* File Uploader - Light Blue */
    [data-testid="stFileUploader"] {
        background-color: #F0F9FF !important;
        border: 2px dashed #FFD1B3 !important;
        border-radius: 16px;
        padding: 20px;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: #F0F9FF !important;
        border: none !important;
        border-radius: 12px;
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent !important;
    }
    
    /* Info/Alert boxes - Light Blue */
    .stAlert {
        background-color: #F0F9FF !important;
        color: #1E293B !important;
        border: 2px solid #FFD1B3 !important;
        border-radius: 12px !important;
    }
    
    /* Force checkbox to be light */
    .stCheckbox [data-testid="stCheckbox"] > div:first-child {
        background-color: white !important;
        border: 2px solid #E2E8F0 !important;
    }
    
    .stCheckbox label, .stRadio label {
        color: #475569 !important;
        font-weight: 500 !important;
    }
    
    /* Buttons - Restored */
    .stButton>button {
        background-color: #FFB067 !important;
        border: none !important;
        color: white !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 10px 24px !important;
        transition: transform 0.2s ease !important;
        font-size: 1rem !important;
    }
    .stButton>button:hover {
        background-color: #FB923C !important;
        transform: scale(1.02);
    }
    
    /* File Uploader Button - Restored & Refined */
    [data-testid="stFileUploader"] button[kind="secondary"] {
        background-color: #FFB067 !important;
        color: white !important;
        border: none !important;
        font-size: 0px !important;
        border-radius: 8px !important;
    }
    [data-testid="stFileUploader"] button[kind="secondary"]::after {
        content: "Dosya Yukle";
        font-size: 14px !important;
        font-weight: 600;
        visibility: visible;
    }
    
    /* Target the 'Clear' buttons in the file list */
    [data-testid="stFileUploaderDeleteBtn"] {
        width: auto !important;
        padding: 0 10px !important;
        background-color: #FFF5F5 !important;
        border: 1px solid #F87171 !important;
        border-radius: 6px !important;
        margin-left: 10px !important;
    }
    [data-testid="stFileUploaderDeleteBtn"]::after {
        content: "Dosyayi Cikar";
        font-size: 11px !important;
        margin-left: 5px !important;
        color: #B91C1C !important;
        font-weight: 600 !important;
    }
    
    /* Custom divider */
    .fancy-divider {
        height: 3px;
        background-color: #E2E8F0;
        margin: 40px 0;
    }

    /* Captions */
    .time-caption {
        color: #6366f1;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Result Cards */
    .result-card {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 12px;
        border-left: 5px solid #CBD5E1;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    .pos-border { border-left-color: #34D399 !important; }
    .neg-border { border-left-color: #F87171 !important; }
    .neu-border { border-left-color: #60A5FA !important; }

    /* Custom Metric Cards */
    .metric-container {
        display: flex;
        gap: 15px;
        margin-bottom: 30px;
        flex-wrap: wrap;
    }
    .metric-card {
        flex: 1;
        min-width: 140px;
        background-color: #F1F5F9;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 2px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
</style>
""", unsafe_allow_html=True)

# Custom Header
st.markdown(f"""
    <div class="header-container">
        <div class="header-title">AI Yorum Analizi</div>
        <div class="header-desc">
            En gelişmiş yapay zeka modelleri ile yorumlarınızdaki derin duyguları ve istekleri saniyeler içinde çözümler.
        </div>
    </div>
""", unsafe_allow_html=True)

# --- Input Section ---
comments_to_analyze = []

tab1, tab2 = st.tabs(["✍️ Metin Girişi", "📁 Dosya Yükle (CSV/Excel)"])

with tab1:
    text_input = st.text_area(
        "Yorumları alt alta girin:",
        height=200,
        placeholder="Örn: Harika uygulama!\nKötü performans..."
    )
    if text_input.strip():
        raw_lines = text_input.split('\n')
        comments_to_analyze = [{"text": line.strip()} for line in raw_lines if len(line.strip()) > 2]
        
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
                            df_upload = pd.read_csv(uploaded_file, encoding=encoding, sep=None, engine='python')
                            if len(df_upload.columns) <= 1:
                                uploaded_file.seek(0)
                                df_upload = pd.read_csv(uploaded_file, encoding=encoding, sep=';')
                            break
                        except Exception: continue
                else:
                    df_upload = pd.read_excel(uploaded_file)
                
                if df_upload is not None:
                    # Replace Expander with Container
                    st.markdown(f"#### 📄 {uploaded_file.name}")
                    with st.container(border=True):
                        st.info(f"Dosya okundu: {len(df_upload)} satir")
                        
                        # Date & Rating Detection
                        # Priority: 1. Review Last Update, 2. General date keys (excluding 'submit')
                        date_keys = ["date", "time", "tarih", "saat"]
                        rate_keys = ["rating", "star", "puan", "yildiz", "skor", "score"]
                        
                        date_col = None
                        rate_col = None
                        
                        # Explicit check for Priority Column
                        for col in df_upload.columns:
                            if "Review Last Update Date and Time" in col:
                                date_col = col
                                break
                        
                        for col in df_upload.columns:
                            col_l = col.lower()
                            # Never use Submit date
                            if "Review Submit Date and Time" in col:
                                continue
                            if not date_col and any(dk in col_l for dk in date_keys): 
                                date_col = col
                            if not rate_col and any(rk in col_l for rk in rate_keys): 
                                rate_col = col

                        # Advanced Sentiment Column Scoring
                        scores = []
                        for col in df_upload.columns:
                            col_l = col.lower()
                            score = 0
                            
                            # Textual keywords
                            if any(k in col_l for k in ["review", "yorum", "text", "metin", "content", "mesaj"]): score += 20
                            
                            # Metadata keywords (Avoid for sentiment)
                            if any(k in col_l for k in ["id", "rating", "star", "puan", "date", "tarih"]): score -= 25
                            
                            # Content Analysis
                            sample = df_upload[col].dropna().head(10).astype(str).tolist()
                            if sample:
                                avg_len = sum(len(s) for s in sample) / len(sample)
                                if avg_len > 30: score += 15 # High narrative factor
                                if avg_len < 10: score -= 20 # Too short/numeric likely
                                
                                # Check for Turkish stop words/common words to confirm natural language
                                common_tr = [" bir ", " bu ", " çok ", " ve ", " ama ", " için "]
                                text_blobs = " ".join(sample).lower()
                                if any(w in text_blobs for w in common_tr): score += 15
                            
                            scores.append((score, col))
                        
                        scores.sort(key=lambda x: x[0], reverse=True)
                        col_name = scores[0][1] if scores else df_upload.columns[0]
                        st.caption(f"✨ **Otomatik Secilen Sutun:** `{col_name}`")
                        
                        if col_name:
                            # NEW FEATURE: Dosya Istatistikleri
                            st.markdown("---")
                            col_vals = df_upload[col_name].astype(str)
                            total_words = col_vals.apply(lambda x: len(x.split())).sum()
                            avg_len = col_vals.apply(len).mean()
                            
                            stat_col1, stat_col2, stat_col3 = st.columns(3)
                            with stat_col1:
                                st.metric("Toplam Kelime", f"{total_words:,}")
                            with stat_col2:
                                st.metric("Ort. Yorum Boyu", f"{int(avg_len)} Karakter")
                            with stat_col3:
                                meta_status = []
                                if date_col: meta_status.append("📅 Tarih")
                                if rate_col: meta_status.append("⭐ Puan")
                                st.write("**Bulunan Ek Veriler:**")
                                st.write(", ".join(meta_status) if meta_status else "Yok")

                            # Pre-filter logic
                            def is_valid_comment(text):
                                s = str(text).strip()
                                if len(s) < 4: return False
                                if s.lower() in ['nan', 'null', 'none']: return False
                                reply_patterns = ["merhaba", "tesekkur ederiz", "bilginize sunar", "iyi gunler dileriz"]
                                if any(rp in s.lower() for rp in reply_patterns): return False
                                return True

                            valid_in_file = 0
                            for _, row in df_upload.iterrows():
                                comment_text = str(row[col_name]).strip() if pd.notnull(row[col_name]) else ""
                                valid_text = is_valid_comment(comment_text)
                                has_rating = rate_col and pd.notnull(row[rate_col])
                                
                                if valid_text or has_rating:
                                    entry = {"text": comment_text, "is_valid": valid_text}
                                    
                                    # Capture Date
                                    if date_col and pd.notnull(row[date_col]):
                                        dt_val = row[date_col]
                                        if not isinstance(dt_val, (int, float)):
                                            parsed_date = pd.to_datetime(dt_val, errors='coerce', dayfirst=True)
                                            if pd.notnull(parsed_date) and parsed_date.tzinfo is not None:
                                                parsed_date = parsed_date.tz_localize(None)
                                            entry["date"] = parsed_date
                                    
                                    # Capture Rating
                                    if rate_col and pd.notnull(row[rate_col]):
                                        entry["rating"] = str(row[rate_col])
                                        
                                    all_comments.append(entry)
                                    if valid_text:
                                        valid_in_file += 1
                                    
                            st.caption(f"Bu dosyadan {valid_in_file} gecerli yorum eklendi.")
                            
            except Exception as e:
                st.error(f"⚠️ {uploaded_file.name} okuma hatası: {e}")
        
        if all_comments:
            comments_to_analyze = all_comments
            st.success(f"📋 Toplam **{len(comments_to_analyze)}** gerçek yorum analiz için hazır!")


# ── Analiz Yapılandırması ──────────────────────
if comments_to_analyze:
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("### ⚙️ Analiz Yapılandırması")
    
    n = len(comments_to_analyze)

    def fmt_time(secs):
        m, s = divmod(secs, 60)
        return f"{m} dakika {s} saniye" if m > 0 else f"{s} saniye"

    mode_idx = st.radio(
        "Analiz hızı ve doğruluk dengesini seçin:",
        options=[0, 1],
        format_func=lambda x: ["🚀 Hızlı", "🎯 Yavaş (Daha Tutarlı)"][x],
        captions=[
            f"Genel değerlendirmeler — tahmini {fmt_time(n * 2)}",
            f"Çok daha doğru sonuçlar — tahmini {fmt_time(n * 4)}"
        ],
        horizontal=True,
        key="analysis_mode"
    )





def get_gemini_sentiment(text, model_name='gemini-3.1-flash-lite-preview'):
    if not HAS_GEMINI:
        return None
    try:
        model = genai.GenerativeModel(model_name)
        prompt = f"""
Sen bir uygulama yorumu duygu analizi uzmanısın.
Aşağıdaki yorumu oku ve kullanıcının genel duygusunu belirle.

Kategoriler:
- olumlu: Kullanıcı memnun, teşekkür ediyor, övüyor. Örn: "harika uygulama", "5 yıldız", "teşekkürler"
- olumsuz: Kullanıcı şikayetçi, sorun var. Örn: "açılmıyor", "donuyor", "kapanıyor", "zor giriyor", "silinmiş", "yaramaz", "bozuk"
- istek_gorus: Tarafsız öneri veya soru. Örn: "şu özellik gelse", "reklamdan kurtulabilir miyiz?"

Önemli kurallar:
1. Yorumun SON cümlesi / son edit bölümü en belirleyicidir.
2. "Teşekkürler ama problem devam ediyor" → olumsuz
3. "Sorun vardı ama çözdüler, harika" → olumlu
4. "Şu özellik gelse iyi olur" → istek_gorus
5. Kısa ama net şikayetler (örn: "girilmiyor", "yaramaz", "kapanıyor") → olumsuz
6. Kısa ama net övgüler (örn: "iyi gidiyor", "güzel", "süper") → olumlu

SADECE JSON döndür:
{{"olumlu": puan, "olumsuz": puan, "istek_gorus": puan}}
Toplam 1.0 olmalı.

Yorum: "{text}"
"""
        response = model.generate_content(prompt)
        content = response.text
        match = re.search(r'\{.*\}', content, re.DOTALL)
        if match:
            data = json.loads(match.group())
            p = float(data.get("olumlu", 0))
            n = float(data.get("olumsuz", 0))
            neu = float(data.get("istek_gorus", 0))
            total = p + n + neu
            if total > 0:
                return {"olumlu": p/total, "olumsuz": n/total, "istek_gorus": neu/total}
    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "quota" in err_str.lower():
            # Track count silently, let the caller display the message
            st.session_state['_quota_hits'] = st.session_state.get('_quota_hits', 0) + 1
        else:
            st.warning(f"⚠️ Gemini API hatası: {err_str[:120]}")
        return None
    return None




def heuristic_analysis(text):
    """Fallback — used when Gemini API is unavailable."""
    t = text.lower()
    pos_words = [
        "teşekkür", "harika", "başarılı", "mükemmel", "güzel", "iyi", "memnun",
        "sev", "süper", "5 yıldız", "sağolun", "devam etsin", "devam eder"
    ]
    neg_words = [
        "kötü", "berbat", "bozuk", "bozuldu", "yaramaz", "rezalet", "rezil",
        "açılmıyor", "açılmı", "zor açıl", "girilmiyor", "giremiyorum",
        "donuyor", "dondu", "kasıyor", "kasıldı", "kapanıyor", "kapandı",
        "çöküyor", "çöktü", "durduruldu", "hatası", "hata veriyor",
        "yavaş", "silinmiş", "gitmiş", "kayboldu", "çalışmıyor",
        "sorun", "problem", "mahvoldu", "batık", "mağdur"
    ]
    neu_words = ["keşke", "gelse", "olsa", "olurdu", "gelebilir", "eklense", "mı?", "mi?", "nasıl"]

    pos = sum(1 for w in pos_words if w in t)
    neg = sum(1 for w in neg_words if w in t)
    neu = sum(1 for w in neu_words if w in t)

    if neg > pos:   return {"olumlu": 0.05, "olumsuz": 0.90, "istek_gorus": 0.05, "method": "Heuristic"}
    if pos > neg:   return {"olumlu": 0.90, "olumsuz": 0.05, "istek_gorus": 0.05, "method": "Heuristic"}
    if neu > 0:     return {"olumlu": 0.10, "olumsuz": 0.10, "istek_gorus": 0.80, "method": "Heuristic"}
    # True default: balanced, slight lean to neutral
    return {"olumlu": 0.30, "olumsuz": 0.30, "istek_gorus": 0.40, "method": "Heuristic"}



# Analysis Trigger
ticker_placeholder = st.empty()
if st.button("Analizini Yap", use_container_width=True):
    if not comments_to_analyze:
        st.warning("Lütfen analiz edilecek bir metin girin veya dosya yükleyin.")
    else:
        bulk_results = []
        time_display = st.empty()  # MOVED: Immediately below button
        progress_bar = st.progress(0)
        status_text = st.empty()
        quota_info = st.empty()
        st.warning("🔴 Analiz süresince bu sayfayı kapatmayın veya yenilemeyin. Verileriniz kaybolabilir.")
        st.session_state['_quota_hits'] = 0
            
        # Seçilen moda göre model ve bekleme süresi (0=Hızlı, 1=Yavaş)
        mode_idx = st.session_state.get("analysis_mode", 0)
        if mode_idx == 0:  # Hızlı
            ANALYSIS_MODEL = 'gemini-2.0-flash-lite'
            DELAY_SECS = 2
            RPM_LIMIT = 30
        else:  # Yavaş
            ANALYSIS_MODEL = 'gemini-3.1-flash-lite-preview'
            DELAY_SECS = 4
            RPM_LIMIT = 15



        start_time = time.time()
        total_items = len(comments_to_analyze)
        est_total_secs = total_items * DELAY_SECS

        # JavaScript: sayfadan ayrılmaya karşı uyarı
        components.html(f"""
        <script>
        (function() {{
            var totalSecs = {est_total_secs};
            window.parent.onbeforeunload = function(e) {{
                var m = Math.floor(totalSecs / 60);
                var s = totalSecs % 60;
                var timeStr = (m > 0 ? m + ' dakika ' : '') + s + ' saniye';
                var msg = '⚠️ Analiz henüz tamamlanmadı! Tahmini kalan süre: ' + timeStr + '. Çıkarsanız verileriniz kaybolacak!';
                e.preventDefault();
                e.returnValue = msg;
                return msg;
            }};
        }})();
        </script>
        """, height=0)

        def update_time(done, total, start):
            elapsed = int(time.time() - start)
            el_m, el_s = divmod(elapsed, 60)
            el_str = f"{el_m} dk {el_s} sn" if el_m > 0 else f"{el_s} sn"
            if done > 0:
                avg = (time.time() - start) / done
                rem_secs = int(avg * (total - done))
                rem_m, rem_s = divmod(rem_secs, 60)
                rem_str = f"{rem_m} dk {rem_s} sn" if rem_m > 0 else f"{rem_s} sn"
            else:
                rem_str = "—"
            time_display.markdown(
                f"⏱ **Geçen süre:** {el_str} &nbsp;&nbsp;&nbsp; ⏳ **Tahmini kalan:** {rem_str}"
            )


        for i, entry in enumerate(comments_to_analyze):
            comment = entry["text"]
            date = entry.get("date")
            is_valid = entry.get("is_valid", True)
            
            status_text.text(f"Analiz ediliyor: {i+1} / {len(comments_to_analyze)}")
            update_time(i, len(comments_to_analyze), start_time)

            if is_valid and comment:
                res = get_gemini_sentiment(comment, model_name=ANALYSIS_MODEL) or heuristic_analysis(comment)
                scores = {"Olumlu": res['olumlu'], "Olumsuz": res['olumsuz'], "İstek/Görüş": res['istek_gorus']}
                verdict = max(scores, key=scores.get)
                # Update quota warning placeholder
                q = st.session_state.get('_quota_hits', 0)
                if q == 1:
                    quota_info.info(f"ℹ️ Gemini kota aşıldı. Bu yorum yerel motorla değerlendirildi. (Model: dakikada en fazla {RPM_LIMIT} istek)")
                elif q > 1:
                    quota_info.info(f"ℹ️ Toplam **{q} yorum** kota nedeniyle yerel motorla değerlendirildi.")
            else:
                verdict = "—"
                res = {'olumlu': 0, 'olumsuz': 0, 'istek_gorus': 0}

            # Update Ticker
            ticker_date = ""
            if date:
                try: ticker_date = f" | 📅 {date.strftime('%d-%m-%Y')}"
                except: pass

            ticker_color = "#34D399" if verdict == "Olumlu" else ("#F87171" if verdict == "Olumsuz" else "#60A5FA")
            ticker_placeholder.markdown(f"""
            <div style="border: 2px solid {ticker_color}; padding: 15px; border-radius: 12px; background: #FFFFFF; margin: 10px 0;">
                <div style="font-size: 0.85em; color: #64748b; margin-bottom: 5px;">⚡ ŞU AN ANALİZ EDİLİYOR (#{i+1}{ticker_date})</div>
                <div style="font-weight: 600; color: #1E293B;">{comment[:250]}{'...' if len(comment)>250 else ''}</div>
                <div style="margin-top: 10px; display: inline-block; padding: 2px 8px; border-radius: 4px; background: {ticker_color}; color: white; font-size: 0.8em; font-weight: bold;">
                    {verdict.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            bulk_results.append({
                "No": i + 1, "Yorum": comment, "Baskın Duygu": verdict,
                "Olumlu %": f"{res['olumlu']:.2%}", "İstek/Görüş %": f"{res['istek_gorus']:.2%}", "Olumsuz %": f"{res['olumsuz']:.2%}",
                "Tarih": date,
                "Puan": entry.get('rating')
            })
            remaining = len(comments_to_analyze) - (i + 1)
            if remaining == 0:
                ticker_placeholder.markdown(f"""
                <div style="border: 2px solid #10B981; padding: 20px; border-radius: 12px; background: #ECFDF5; margin: 10px 0; text-align: center;">
                    <div style="font-size: 2em; margin-bottom: 10px;">✅</div>
                    <div style="font-weight: 700; color: #065F46; font-size: 1.2em;">ANALİZ TAMAMLANDI</div>
                    <div style="font-size: 0.9em; color: #047857; margin-top: 5px;">Toplam {len(comments_to_analyze)} satır başarıyla işlendi.</div>
                </div>
                """, unsafe_allow_html=True)
            progress_bar.progress((i + 1) / len(comments_to_analyze))
            if remaining > 0 and is_valid:
                time.sleep(DELAY_SECS)
                update_time(i + 1, len(comments_to_analyze), start_time)
            elif remaining > 0:
                 update_time(i + 1, len(comments_to_analyze), start_time)

        
        st.session_state.bulk_results = bulk_results
        status_text.success("✅ Analiz Başarıyla Tamamlandı!")
        # Sayfadan ayrılma uyarısını kaldır
        components.html("<script>window.parent.onbeforeunload = null;</script>", height=0)

# --- Persistent Results Display ---
if "bulk_results" in st.session_state:
    df = pd.DataFrame(st.session_state.bulk_results)
    counts = df["Baskın Duygu"].value_counts()
    
    st.markdown("""
<style>
/* Results Card Styling */
.neon-pos { border: 2px solid #34D399 !important; padding: 15px; border-radius: 12px; margin: 10px 0; background: #F0F9FF !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
.neon-neg { border: 2px solid #F87171 !important; padding: 15px; border-radius: 12px; margin: 10px 0; background: #F0F9FF !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
.neon-neu { border: 2px solid #60A5FA !important; padding: 15px; border-radius: 12px; margin: 10px 0; background: #F0F9FF !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
.normal-card { border: 1px solid #E2E8F0 !important; padding: 15px; border-radius: 12px; margin: 10px 0; background: #F0F9FF !important; }

.neon-pos *, .neon-neg *, .neon-neu *, .normal-card * {
    color: #1E293B !important;
}

.metric-container {
    display: flex;
    justify-content: space-around;
    gap: 1rem;
    margin-bottom: 2rem;
    flex-wrap: wrap;
}
.metric-card {
    background: #F0F9FF !important;
    border: 2px solid #FFE4D6 !important;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    flex: 1;
    min-width: 150px;
}
.metric-value { font-size: 2.5em; font-weight: bold; line-height: 1.2; }
.metric-label { font-size: 0.9em; color: #64748b !important; margin-top: 0.3rem; }

.glass-card {
    background: #F0F9FF !important;
    border: 2px solid #F1F5F9 !important;
    border-radius: 15px;
    padding: 20px;
    margin-bottom: 25px;
    color: #1E293B !important;
}

.glass-card * {
    color: #1E293B !important;
}

.sentiment-indicator {
    display: inline-flex;
    align-items: center;
    padding: 4px 8px;
    border-radius: 6px;
    font-weight: 600;
    font-size: 0.85em;
    margin-right: 8px;
}
.sentiment-indicator.positive { border: 1px solid #10b981; color: #10b981; }
.sentiment-indicator.negative { border: 1px solid #f43f5e; color: #f43f5e; }
.sentiment-indicator.neutral { border: 1px solid #3b82f6; color: #3b82f6; }
</style>
""", unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Analiz Özeti")
    
    analysis_df = df[df["Baskın Duygu"] != "—"].copy()
    counts = analysis_df["Baskın Duygu"].value_counts()
    
    # Custom Metric Cards
    m_olumlu = counts.get("Olumlu", 0)
    m_olumsuz = counts.get("Olumsuz", 0)
    m_istek = counts.get("İstek/Görüş", 0)
    
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-card">
            <div class="metric-value" style="color: #10b981;">{m_olumlu}</div>
            <div class="metric-label">Olumlu</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: #f43f5e;">{m_olumsuz}</div>
            <div class="metric-label">Olumsuz</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: #3b82f6;">{m_istek}</div>
            <div class="metric-label">İstek / Görüş</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: #a78bfa;">{len(df)}</div>
            <div class="metric-label">Toplam Analiz</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    col_pie, col_summary = st.columns([1, 1])
    
    with col_pie:
        pie_data = pd.DataFrame({"Duygu": counts.index, "Sayı": counts.values})
        fig_pie = px.pie(pie_data, values='Sayı', names='Duygu', hole=0.6,
                      color='Duygu', color_discrete_map={'Olumlu':'#10b981', 'Olumsuz':'#f43f5e', 'İstek/Görüş':'#3b82f6'})
        fig_pie.update_traces(textinfo='percent', textfont_size=14, marker=dict(line=dict(color='#0f172a', width=2)))
        fig_pie.update_layout(height=350, showlegend=True, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
                             font=dict(color='#94a3b8'),
                             legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5},
                             margin={"t": 0, "b": 0, "l": 0, "r": 0})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_summary:
        st.write("#### � Yapay Zeka Görüsü")
        if counts.idxmax() == "Olumlu":
            st.success(f"Topluluk genel olarak **Olumlu** bir tavır sergiliyor. ({m_olumlu} yorum)")
        elif counts.idxmax() == "Olumsuz":
            st.error(f"Dikkat çeken **Olumsuz** bir eğilim var. ({m_olumsuz} yorum)")
        else:
            st.info(f"Kullanıcılar yoğun şekilde **İstek ve Görüş** paylaşıyor. ({m_istek} yorum)")
            
        st.write("Aşağıdaki sekmelerden tüm yorumları tek tek inceleyebilir, grafik üzerinde tarihsel değişimleri takip edebilirsiniz.")

    # NEW: Star Rating Distribution Chart (Sütunlu ve Renkli)
    if "Puan" in df.columns and df["Puan"].notnull().any():
        st.markdown("---")
        
        # UI for Frequency Selection
        g_col1, g_col2 = st.columns([2, 1])
        with g_col1:
            st.write("#### 📊 Puan Dağılımı Trendi")
        with g_col2:
            freq = st.radio("Zaman Ölçeği:", ["Günlük", "Haftalık", "Aylık"], index=2, horizontal=True, key="puan_freq_sel")

        df_puan = df.dropna(subset=["Tarih", "Puan"]).copy()
        try:
            # Ensure ratings are integers 1-5 for clean legend
            df_puan["Puan_val"] = pd.to_numeric(df_puan["Puan"], errors='coerce').fillna(0).astype(int)
            df_puan = df_puan[(df_puan["Puan_val"] >= 1) & (df_puan["Puan_val"] <= 5)]
            
            if not df_puan.empty:
                df_puan["Tarih_dt"] = pd.to_datetime(df_puan["Tarih"])
                
                # Resample based on chosen frequency
                if freq == "Haftalık":
                    df_puan["Grup"] = df_puan["Tarih_dt"].dt.to_period('W').apply(lambda r: r.start_time)
                    title_txt = "Haftalık Puan Dağılımı"
                elif freq == "Aylık":
                    df_puan["Grup"] = df_puan["Tarih_dt"].dt.to_period('M').apply(lambda r: r.start_time)
                    title_txt = "Aylık Puan Dağılımı"
                else:
                    df_puan["Grup"] = df_puan["Tarih_dt"].dt.date
                    title_txt = "Günlük Puan Dağılımı"

                # Group by chosen period and rating value
                dist_trend = df_puan.groupby(["Grup", "Puan_val"]).size().reset_index(name='Oy Sayısı')
                dist_trend["Puan_Label"] = dist_trend["Puan_val"].apply(lambda x: f"{x} Yıldız")
                dist_trend = dist_trend.sort_values("Puan_val", ascending=True)

                fig_dist = px.bar(dist_trend, x="Grup", y="Oy Sayısı", color="Puan_Label",
                                 title=title_txt,
                                 color_discrete_map={
                                     "1 Yıldız": "#08306b",
                                     "2 Yıldız": "#08519c",
                                     "3 Yıldız": "#2171b5",
                                     "4 Yıldız": "#6baed6",
                                     "5 Yıldız": "#deebf7"
                                 },
                                 category_orders={"Puan_Label": ["1 Yıldız", "2 Yıldız", "3 Yıldız", "4 Yıldız", "5 Yıldız"]})
                
                fig_dist.update_layout(
                    height=450, 
                    margin={"t": 60, "b": 20, "l": 10, "r": 10},
                    xaxis_title="Zaman Dönemi",
                    yaxis_title="Yorum / Puan Sayısı",
                    legend_title="Puan",
                    barmode='stack',
                    bargap=0.1
                )
                
                # Dynamic X-axis formatting
                fig_dist.update_xaxes(
                    tickformat="%b %Y" if freq == "Aylık" else "%d %b %y",
                    tickangle=-45
                )
                st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Grafik oluşturma hatası: {e}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Chart & List Logic
    def render_trend_chart(filtered_df, key, title_suffix=""):
        df_dates = filtered_df.dropna(subset=["Tarih"]).copy()
        if not df_dates.empty:
            df_dates["Tarih"] = pd.to_datetime(df_dates["Tarih"])
            df_dates['Hafta'] = df_dates['Tarih'].dt.to_period('W').apply(lambda r: r.start_time)
            trend_data = df_dates.groupby(['Hafta', "Baskın Duygu"]).size().reset_index(name='Adet')
            
            # Custom data for robust selection processing (includes exact Hafta and Sentiment)
            trend_data['Hafta_str'] = trend_data['Hafta'].astype(str)
            fig_trend = px.bar(trend_data, x="Hafta", y="Adet", color="Baskın Duygu",
                               title=f"Haftalık Duygu Dağılımı {title_suffix}",
                               color_discrete_map={'Olumlu':'#2ecc71', 'Olumsuz':'#e74c3c', 'İstek/Görüş':'#3498db'},
                               barmode='group',
                               custom_data=["Hafta_str", "Baskın Duygu"])
            
            fig_trend.update_layout(height=280, margin={"t": 50, "b": 20, "l": 10, "r": 10},
                                   legend={"orientation": "h", "yanchor": "bottom", "y": -0.4, "xanchor": "center", "x": 0.5},
                                   xaxis_title="Hafta (Başlangıç)", yaxis_title="Yorum Sayısı",
                                   clickmode='event+select')
            
            # Use on_select for interactivity (Streamlit 1.35+)
            selection = st.plotly_chart(fig_trend, use_container_width=True, on_select="rerun", key=f"chart_{key}")
            
            if selection and "selection" in selection and selection["selection"]["points"]:
                point = selection["selection"]["points"][0]
                # Use Hafta directly from custom_data for 100% precision
                sel_week_str = point["customdata"][0]
                sel_week = pd.to_datetime(sel_week_str).tz_localize(None)
                sel_sentiment = str(point["customdata"][1]).strip()
                
                # Standardize database weeks for comparison
                df_dates['Hafta_compare'] = pd.to_datetime(df_dates['Hafta']).dt.tz_localize(None)
                
                final_filtered = df_dates[
                    (df_dates['Hafta_compare'] == sel_week) & 
                    (df_dates['Baskın Duygu'] == sel_sentiment)
                ]
                
                st.info(f"🔎 Filtrelendi: **{sel_week.strftime('%d.%m.%Y')}** haftası - **{sel_sentiment}** yorumlar")
                if st.button("Filtreyi Temizle", key=f"clear_{key}"):
                    st.rerun()
                return final_filtered
            
        return filtered_df

    def display_comments(filtered_df, highlight=True):
        if filtered_df.empty:
            st.info("Bu kategoride henüz yorum bulunmuyor.")
            return
        for _, row in filtered_df.iterrows():
            sentiment = row["Baskın Duygu"]
            cls = "normal-card"
            if highlight:
                cls = "neon-pos" if sentiment == "Olumlu" else ("neon-neg" if sentiment == "Olumsuz" else "neon-neu")
            
            # Format extra info (Date & Rating)
            extra_info = ""
            if "Tarih" in row and pd.notnull(row["Tarih"]):
                try:
                    d = pd.to_datetime(row["Tarih"])
                    extra_info += f" | 📅 {d.strftime('%d-%m-%Y')}"
                except: pass
            
            if "Puan" in row and pd.notnull(row["Puan"]):
                extra_info += f" | ⭐ {row['Puan']}"

            st.markdown(f"""
            <div class="{cls}">
                <span style="font-size: 0.8em; color: #aaa;">#{row['No']} | {sentiment}{extra_info}</span><br>
                {row['Yorum']}
            </div>
            """, unsafe_allow_html=True)

    # --- Tabs and Unified Display ---
    st.write("### 💬 Yorum Listesi")
    
    t_pos = counts.get('Olumlu', 0)
    t_neg = counts.get('Olumsuz', 0)
    t_neu = counts.get('İstek/Görüş', 0)
    t_all = len(analysis_df)

    tab_all, tab_pos, tab_neg, tab_neu = st.tabs([
        f"🌐 Analizler ({t_all})", 
        f"🟢 Olumlu ({t_pos})", 
        f"🔴 Olumsuz ({t_neg})", 
        f"🔵 İstek/Görüş ({t_neu})"
    ])

    with tab_all:
        f_df = render_trend_chart(analysis_df, "all", "(Genel)")
        display_comments(f_df, highlight=False)
    
    with tab_pos:
        pos_df = df[df["Baskın Duygu"] == "Olumlu"]
        f_df = render_trend_chart(pos_df, "pos", "(Olumlu)")
        display_comments(f_df)
        
    with tab_neg:
        neg_df = df[df["Baskın Duygu"] == "Olumsuz"]
        f_df = render_trend_chart(neg_df, "neg", "(Olumsuz)")
        display_comments(f_df)
        
    with tab_neu:
        neu_df = df[df["Baskın Duygu"] == "İstek/Görüş"]
        f_df = render_trend_chart(neu_df, "neu", "(İstek/Görüş)")
        display_comments(neu_df)

    # Excel Download
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Analiz Sonuçları')
        st.download_button(label="📥 Sonuçları Excel Olarak İndir", data=output.getvalue(), file_name="analiz.xlsx", key="bulk_dl")
    except: pass

# Footer
st.divider()
st.caption("Geliştiren: Cem Evecen")

