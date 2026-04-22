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

    /* Global Overrides */
    html, body, [class*="st-"], .stApp, *, .stButton>button, .stTextInput input, .stTextArea textarea {
        font-family: 'Poppins', sans-serif !important;
        color: #1E293B !important; /* Dark Slate for readability */
    }
    
    .stApp {
        background-color: #F8FAFC !important; /* Solid Light Pastel Background */
    }
    
    /* Header Card */
    .header-container {
        background-color: #E0F2FE; /* Solid Pastel Light Blue/Green */
        border: 2px solid #BAE6FD;
        border-radius: 20px;
        padding: 40px;
        margin-bottom: 40px;
        text-align: center;
    }
    .header-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        color: #6366F1; /* Solid Pastel Indigo */
        margin-bottom: 15px;
    }
    .header-desc {
        color: #94a3b8;
        font-size: 1.1rem;
        max-width: 600px;
        margin: 0 auto;
    }

    /* Custom Container */
    .glass-card {
        background-color: #FFFFFF;
        border: 2px solid #F1F5F9;
        border-radius: 16px;
        padding: 30px;
        margin-bottom: 25px;
    }

    /* Buttons */
    .stButton>button {
        background-color: #FFB067 !important; /* Lighter Pastel Orange */
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

    /* Inputs & Toggles */
    .stTextInput>div>div>input, .stTextArea>div>div>textarea {
        background-color: #FFFFFF !important;
        border: 2px solid #E2E8F0 !important;
        border-radius: 10px !important;
        color: #1E293B !important;
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
                        st.info(f"📁 **Dosya İşleniyor:** {uploaded_file.name}")
                        
                        # Smart Column Detection
                        target_keys = ["review", "yorum", "text", "metin", "content", "body"]
                        avoid_keys = ["id", "name", "isim", "name", "rating", "star", "vers", "date", "tarih", "saat"]
                        
                        scores = []
                        for col in df_upload.columns:
                            col_l = col.lower()
                            score = 0
                            if any(k in col_l for k in target_keys): score += 10
                            if any(k in col_l for k in avoid_keys): score -= 15
                            
                            sample = df_upload[col].head(5).astype(str).tolist()
                            avg_len = sum(len(s) for s in sample) / 5 if sample else 0
                            if avg_len > 15: score += 10
                            scores.append((score, col))
                        
                        scores.sort(key=lambda x: x[0], reverse=True)
                        best_col = scores[0][1] if scores else df_upload.columns[0]
                        
                        # Date Column Detection
                        date_keys = ["date", "time", "tarih", "saat", "submit"]
                        date_col = None
                        for col in df_upload.columns:
                            if any(dk in col.lower() for dk in date_keys):
                                date_col = col
                                break

                        col_name = st.selectbox(
                            f"Analiz edilecek sütun ({uploaded_file.name}):",
                            options=df_upload.columns,
                            index=list(df_upload.columns).index(best_col),
                            key=f"col_{uploaded_file.name}"
                        )
                        
                        if col_name:
                            # Pre-filter for valid comments in this specific file
                            def is_valid_comment(text):
                                s = str(text).strip()
                                if len(s) < 4: return False
                                if s.lower() in ['nan', 'null', 'none']: return False
                                
                                # 3. Developer/Owner Reply Patterns (Professional/Turkish Store Language)
                                reply_patterns = [
                                    "merhaba", "merhabalar", "teşekkür ederiz", "bilginize sunar", 
                                    "iyi günler dileriz", "rica etsek", "geri bildirimleriniz", 
                                    "değerlendirmenizi bekler", "saygılarımızla", "ekibimiz", 
                                    "talebini", "incelemelerimiz sonucunda", "güncellememizi",
                                    "tarafımıza iletmenizi", "yenilenmiş tasarımı", "uygulamamız yayında",
                                    "yaşamış olduğunuz", "aksaklık için üzgünüz", "memnun oluruz",
                                    "bizimle iletişime", "iletmenizi rica ederiz"
                                ]
                                
                                # Highly aggressive check: if any signature pattern exists, it's a dev reply
                                if any(rp in s.lower() for rp in reply_patterns):
                                    return False
                                
                                # Filter formal addresses "Ad Soyad Bey/Hanım,"
                                if re.search(r'[a-zçğıöşü]+\s+(bey|hanım),', s.lower()):
                                    return False
                                
                                # 4. Metadata/Numeric IDs
                                if re.match(r'^\d{4}-\d{2}-\d{2}.*', s): return False
                                if s.replace('.', '').replace('-', '').isdigit(): return False
                                if re.match(r'^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$', s): return False
                                
                                return True

                            for _, row in df_upload.iterrows():
                                if pd.notnull(row[col_name]) and is_valid_comment(row[col_name]):
                                    entry = {"text": str(row[col_name]).strip()}
                                    if date_col and pd.notnull(row[date_col]):
                                        # Robust parsing & Strict bounds
                                        dt_val = row[date_col]
                                        # Ensure we don't parse large integers as dates
                                        if isinstance(dt_val, (int, float)):
                                            parsed_date = pd.NaT
                                        else:
                                            # Convert to datetime and strip timezone for safe comparison
                                            parsed_date = pd.to_datetime(dt_val, errors='coerce', dayfirst=True)
                                            if pd.notnull(parsed_date) and parsed_date.tzinfo is not None:
                                                parsed_date = parsed_date.tz_localize(None)
                                        
                                        # Limit: Up to Today (Mar 8, 2026) - Naive for safe comparison
                                        today_limit = pd.Timestamp("2026-03-08").tz_localize(None)
                                        start_limit = pd.Timestamp("2025-11-01").tz_localize(None)
                                        
                                        if pd.notnull(parsed_date) and start_limit <= parsed_date <= today_limit:
                                            entry["date"] = parsed_date
                                    all_comments.append(entry)
                                    
                            with st.expander(f"👀 {uploaded_file.name} Önizleme (Seçilen: {col_name})"):
                                st.write([c["text"] for c in all_comments[-5:]] if all_comments else [])
                                
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
if st.button("Analizini Yap", use_container_width=True):
    if not comments_to_analyze:
        st.warning("Lütfen analiz edilecek bir metin girin veya dosya yükleyin.")
    else:
        bulk_results = []
        progress_bar = st.progress(0)
        status_text = st.empty()
        quota_info = st.empty()  # Single placeholder for quota warnings
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
            time_display = st.empty()  # For side-by-side time display
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
                status_text.text(f"Analiz ediliyor: {i+1} / {len(comments_to_analyze)}")
                update_time(i, len(comments_to_analyze), start_time)

                res = get_gemini_sentiment(comment, model_name=ANALYSIS_MODEL) or heuristic_analysis(comment)
                scores = {"Olumlu": res['olumlu'], "Olumsuz": res['olumsuz'], "İstek/Görüş": res['istek_gorus']}
                verdict = max(scores, key=scores.get)

                # Update quota warning placeholder
                q = st.session_state.get('_quota_hits', 0)
                if q == 1:
                    quota_info.info(f"ℹ️ Gemini kota aşıldı. Bu yorum yerel motorla değerlendirildi. (Model: dakikada en fazla {RPM_LIMIT} istek)")
                elif q > 1:
                    quota_info.info(f"ℹ️ Toplam **{q} yorum** kota nedeniyle yerel motorla değerlendirildi.")

                bulk_results.append({
                    "No": i + 1, "Yorum": comment, "Baskın Duygu": verdict,
                    "Olumlu %": f"{res['olumlu']:.2%}", "İstek/Görüş %": f"{res['istek_gorus']:.2%}", "Olumsuz %": f"{res['olumsuz']:.2%}",
                    "Tarih": date
                })
                progress_bar.progress((i + 1) / len(comments_to_analyze))

                remaining = len(comments_to_analyze) - (i + 1)
                if remaining > 0:
                    time.sleep(DELAY_SECS)
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
    .neon-pos { border: 1px solid #2ecc71; box-shadow: 0 0 3px #2ecc71; padding: 12px; border-radius: 8px; margin: 8px 0; background: rgba(46, 204, 113, 0.05); }
    .neon-neg { border: 1px solid #e74c3c; box-shadow: 0 0 3px #e74c3c; padding: 12px; border-radius: 8px; margin: 8px 0; background: rgba(231, 76, 60, 0.05); }
    .neon-neu { border: 1px solid #3498db; box-shadow: 0 0 3px #3498db; padding: 12px; border-radius: 8px; margin: 8px 0; background: rgba(52, 152, 219, 0.05); }
    .normal-card { border: 1px solid #333; padding: 12px; border-radius: 8px; margin: 8px 0; background: #1e1e1e; opacity: 0.9; }
    .fancy-divider {
        height: 2px;
        background: linear-gradient(to right, #3b82f6, #a78bfa, #f43f5e);
        border-radius: 1px;
        margin: 2rem 0;
    }
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 1rem;
        margin-bottom: 2rem;
        flex-wrap: wrap;
    }
    .metric-card {
        background: rgba(30, 41, 59, 0.7); /* slate-800 with transparency */
        border: 1px solid rgba(71, 85, 105, 0.5); /* slate-600 with transparency */
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        text-align: center;
        flex: 1;
        min-width: 150px;
        backdrop-filter: blur(10px);
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        transition: transform 0.2s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 6px 12px rgba(0, 0, 0, 0.15);
    }
    .metric-value {
        font-size: 2.5em;
        font-weight: bold;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.9em;
        color: #94a3b8; /* slate-400 */
        margin-top: 0.3rem;
    }
    .glass-card {
        background: rgba(30, 41, 59, 0.6); /* slate-800 with transparency */
        border: 1px solid rgba(71, 85, 105, 0.4); /* slate-600 with transparency */
        border-radius: 15px;
        padding: 20px;
        margin-bottom: 25px;
        backdrop-filter: blur(15px);
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
        color: #e2e8f0; /* slate-200 */
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
    .sentiment-indicator.positive {
        background-color: rgba(16, 185, 129, 0.2); /* emerald-500 */
        color: #10b981;
    }
    .sentiment-indicator.negative {
        background-color: rgba(244, 63, 94, 0.2); /* rose-500 */
        color: #f43f5e;
    }
    .sentiment-indicator.neutral {
        background-color: rgba(59, 130, 246, 0.2); /* blue-500 */
        color: #3b82f6;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("### 📊 Analiz Özeti")
    
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
            
            st.markdown(f"""
            <div class="{cls}">
                <span style="font-size: 0.8em; color: #aaa;">#{row['No']} | {sentiment}</span><br>
                {row['Yorum']}
            </div>
            """, unsafe_allow_html=True)

    # --- Tabs and Unified Display ---
    st.write("### 💬 Yorum Listesi")
    
    t_pos = counts.get('Olumlu', 0)
    t_neg = counts.get('Olumsuz', 0)
    t_neu = counts.get('İstek/Görüş', 0)
    t_all = len(df)

    tab_all, tab_pos, tab_neg, tab_neu = st.tabs([
        f"🌐 Tümü ({t_all})", 
        f"🟢 Olumlu ({t_pos})", 
        f"🔴 Olumsuz ({t_neg})", 
        f"🔵 İstek/Görüş ({t_neu})"
    ])

    with tab_all:
        f_df = render_trend_chart(df, "all", "(Genel)")
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

elif not is_bulk and "single_result" in st.session_state:
    result = st.session_state.single_result
    st.divider()
    st.success("Analiz Tamamlandı")
    c1, c2, c3 = st.columns(3)
    c1.metric("Olumlu", f"{result['olumlu']:.4f}")
    c2.metric("İstek/Görüş", f"{result['istek_gorus']:.4f}")
    c3.metric("Olumsuz", f"{result['olumsuz']:.4f}")
    
    scores = {"Olumlu": result['olumlu'], "Olumsuz": result['olumsuz'], "İstek/Görüş": result['istek_gorus']}
    verdict = max(scores, key=scores.get)
    st.subheader(f"Sonuç: {verdict}")
    
    if verdict == "Olumlu":
        st.info(f"Bu metin genel olarak **Olumlu** bir duygu taşıyor (%{result['olumlu']:.2%}). 😊")
    elif verdict == "Olumsuz":
        st.info(f"Bu metin genel olarak **Olumsuz** bir duygu taşıyor (%{result['olumsuz']:.2%}). 😔")
    else:
        st.info(f"Bu metin **İstek/Görüş** kategorisine giriyor (%{result['istek_gorus']:.2%}). 😐")

# Footer
st.divider()
st.caption("Geliştiren: Cem Evecen")

