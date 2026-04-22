import streamlit as st
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
from google import genai
from google.genai import types as genai_types
import os
import json
import re
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import io
import requests
from datetime import datetime, timedelta
from google_play_scraper import Sort, reviews as play_reviews
# Removed app-store-scraper due to dependency conflicts with streamlit
from dotenv import load_dotenv
if os.path.exists(".env"):
    load_dotenv(override=True)

# Set Page Config
st.set_page_config(
    page_title="AI Duygu Analizi",
    page_icon="💮",
    layout="centered"
)

# API Configuration: Optimized via Caching
@st.cache_resource
def setup_api():
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GEMINI_API_KEY") or st.secrets.get("API_KEY")
        except:
            api_key = None
    if api_key and str(api_key).strip():
        client = genai.Client(api_key=str(api_key).strip())
        return client
    return None

GEMINI_CLIENT = setup_api()
HAS_GEMINI = GEMINI_CLIENT is not None

# --- Lottie Loader ---
@st.cache_data(ttl=3600)
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_loading = load_lottieurl("https://lottie.host/81729486-455b-426d-8833-255e2a222857/YV77X3ZzPZ.json") # Updated to a working modern Lottie asset

# --- UTILS: Content Cleanup Filter ---
def is_valid_comment(text):
    """
    Sophisticated filter to remove metadata, developer responses, and garbage lines.
    Useful for App Store Connect / Play Store copy-pastes.
    """
    if not text: return False
    s = str(text).strip()
    sl = s.lower()
    
    # 1. Basic length check
    if len(s) < 3: return False
    
    # 2. Null values
    if sl in ['nan', 'null', 'none']: return False
    
    # 3. Metadata Keywords / Headers / UI Buttons
    meta_keywords = [
        "developer response", "geliştirici cevabı", "developer answer", 
        "customer review", "müşteri yorumu", "app store connect",
        "review details", "yorum detayları", "version:", "versiyon:",
        "report a concern", "rapor et", "reply", "cevapla", "edit response", "cevabı düzenle"
    ]
    if any(k in sl for k in meta_keywords):
        return False

    # 4. Version Stamps (e.g. "Version 1.2.3 - Turkey")
    if re.search(r"version\s+\d+(\.\d+)*", sl):
        return False
        
    # 5. Date Patterns & Store Headers (e.g. "Mar 2, 2026", "21 Feb 2026")
    # Also block lines starting with Month names (likely nickname/date rows in copy-paste)
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
              "ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"]
    
    first_word = sl.split()[0].replace('.', '').replace(',', '') if sl.split() else ""
    if first_word in months and len(s) < 60:
        return False

    if len(s) < 45:
        date_regex = r"(\d{1,4}[-./]\d{1,2}[-./]\d{1,4})|((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{1,2},?\s+\d{4})"
        if re.search(date_regex, s, re.IGNORECASE):
            return False

    # 6. Formal Developer Canned Replies (Aggressive)
    formal_patterns = [
        "aksaklık için üzgünüz", "yaşanan aksaklık için",
        "teşekkür ederiz. yaşadığınız", "teşekkürler. yaşadığınız",
        "good day, thank you for the feedback",
        "support team", "destek ekibi",
        "iletişime geçtiğiniz için teşekkür",
        "bize ulaştığınız için teşekkür",
        "ilgili birimlerimize iletiyoruz",
        "çözüm için çalışıyoruz",
        "güncelleme ile giderilmiştir",
        "versiyonda giderilmiştir",
        "sorununuz devam ediyorsa",
        "yeni versiyon yayınlandı",
        "yükleyebilmiş miydiniz",
        "yardıma ihtiyacınız olursa",
        "iyi günler dileriz"
    ]
    if any(fp in sl for fp in formal_patterns):
        return False
        
    # 7. Email Addresses (Common in support replies)
    if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", sl):
        return False

    return True

@st.cache_data(show_spinner=False, ttl=600)
def get_app_store_reviews(app_id, country='tr'):
    """Fetch reviews using App Store RSS Feed (Pagination)"""
    reviews = []
    try:
        for page in range(1, 11): # Up to 10 pages * 50 = 500 reviews
            url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/json"
            response = requests.get(url, timeout=10)
            if response.status_code != 200:
                break
                
            data = response.json()
            entries = data.get('feed', {}).get('entry', [])
            if not entries: break
                
            # If only one entry, it's not a list
            if isinstance(entries, dict): entries = [entries]
            
            # Entry[0] is often app metadata on first page, content label check filters it out automatically usually
            for entry in entries:
                content = entry.get('content', {}).get('label')
                if not content: continue
                
                updated = entry.get('updated', {}).get('label', '')
                try:
                    r_date = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                    if r_date.tzinfo is not None: r_date = r_date.replace(tzinfo=None)
                except: r_date = None
                
                rating = entry.get('im:rating', {}).get('label', '0')
                
                reviews.append({
                    "text": content,
                    "date": r_date,
                    "rating": str(rating)
                })
        return reviews
    except Exception as e:
        return reviews

@st.cache_data(show_spinner=False, ttl=600)
def fetch_google_play_reviews(app_id, days_limit):
    """Cached Google Play fetcher"""
    from google_play_scraper import Sort, reviews as play_reviews
    threshold_date = datetime.now() - timedelta(days=days_limit)
    if days_limit <= 30: fetch_limit = 2000
    elif days_limit <= 90: fetch_limit = 10000
    elif days_limit <= 180: fetch_limit = 25000
    else: fetch_limit = 50000
    
    try:
        fetched = []
        continuation_token = None
        max_requests = (fetch_limit // 199) + 1
        
        for _ in range(max_requests):
            result, continuation_token = play_reviews(
                app_id,
                lang='tr',
                country='tr',
                sort=Sort.NEWEST,
                count=199,
                continuation_token=continuation_token
            )
            if not result: break
            
            # Kapsama ve validasyon
            batch_dates = []
            for r in result:
                r_at = r.get('at')
                if r_at and r_at >= threshold_date:
                    content = str(r.get('content', ''))
                    if is_valid_comment(content):
                        fetched.append({
                            "text": content,
                            "date": r_at,
                            "rating": str(r.get('score', ''))
                        })
                if r_at: batch_dates.append(r_at)
            
            # Eğer bu partinin en eski yorumu istediğimiz tarihten eskiyse veya token bittiyse kes
            if batch_dates and min(batch_dates) < threshold_date:
                break
                
            if not continuation_token:
                break
            
            if len(fetched) >= fetch_limit:
                break
                
        return fetched
    except:
        return []

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
    
    /* Strict 5px spacing for headers and common text blocks */
    h1, h2, h3, h4, h5, h6, .stMarkdown div {
        margin-top: 0px !important;
        margin-bottom: 5px !important;
    }
    
    /* Global Icon and Text color enforcement */
    [data-testid="stIcon"], [class*="st-emotion-cache-"], [class*="stIcon"], svg, span[aria-hidden="true"] {
        font-family: inherit !important;
        color: #000000 !important;
        fill: #000000 !important;
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
        padding: 20px;
        margin-bottom: 5px;
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
    
    /* Precise 5px vertical spacing for the whole app */
    [data-testid="stVerticalBlock"] > div {
        margin-top: 0px !important;
        margin-bottom: 0px !important;
    }
    [data-testid="stVerticalBlock"] {
        gap: 5px !important;
    }
    
    .stMarkdown p {
        margin-bottom: 4px !important;
    }
    
    /* Column Badge Styling */
    .column-badge {
        display: inline-block;
        background-color: #64748b;
        color: white !important;
        padding: 2px 10px;
        border-radius: 6px;
        font-size: 0.85rem;
        font-weight: 500;
        margin-left: 5px;
    }
    
    /* Info/Alert boxes - Light Blue */
    .stAlert {
        background-color: #F0F9FF !important;
        color: #1E293B !important;
        border: 2px solid #FFD1B3 !important;
        border-radius: 12px !important;
        margin-bottom: 5px !important;
    }
    
    /* Caption spacing */
    .stCaption {
        margin-bottom: 5px !important;
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

    /* Excel Download Button - Specific Styling */
    .stDownloadButton > button {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #E2E8F0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
    }
    .stDownloadButton > button:hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        color: #000000 !important;
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
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        height: 32px !important; /* Fixed height for better alignment */
    }
    [data-testid="stFileUploaderDeleteBtn"]::after {
        content: "Dosyayı Çıkar";
        font-size: 11px !important;
        margin-left: 6px !important;
        color: #B91C1C !important;
        font-weight: 600 !important;
        line-height: 1 !important;
    }
    [data-testid="stFileUploaderDeleteBtn"] svg {
        color: #B91C1C !important;
        fill: #B91C1C !important;
    }
    
    /* Ensure File Names are readable (not white) */
    [data-testid="stFileUploaderFileName"] {
        color: #1E293B !important;
        font-weight: 500 !important;
    }
    
    /* Custom divider */
    .fancy-divider {
        height: 2px;
        background-color: #E2E8F0;
        margin: 5px 0;
    }
    
    /* Radio button tightening */
    [data-testid="stRadio"] > div {
        gap: 5px !important;
    }
    [data-testid="stRadio"] label {
        margin-bottom: 0px !important;
    }

    /* Captions & Legend Styling */
    .time-caption {
        color: #6366f1;
        font-weight: 600;
        letter-spacing: 0.5px;
    }

    /* Reduce font size of st.metric values (Toplam Kelime, Ort. Boy) */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem !important;
    }
    
    /* Plotly Legend Pagination (Text and Arrows) */
    .legendpaging text {
        fill: #000000 !important;
    }
    .legendpaging path {
        fill: #000000 !important;
        stroke: #000000 !important;
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
        <div class="header-title" style="margin-bottom: 0px;">AI Yorum Analizi</div>
    </div>
""", unsafe_allow_html=True)

# --- Input Section ---
if 'comments_to_analyze' not in st.session_state:
    st.session_state.comments_to_analyze = []

comments_to_analyze = [] # Reset local ref for tab logic

tab1, tab2, tab3 = st.tabs(["🔗 Mağaza Linki", "📁 Dosya Yükle (CSV/Excel)", "✍️ Metin Girişi"])

with tab1:
    with st.container(border=True):
        col_u, col_r = st.columns([2, 1])
        with col_u:
            store_url = st.text_input("Uygulama linki veya ID girin:", placeholder="Örn: com.whatsapp veya 1500198745")
        with col_r:
            time_range = st.selectbox(
                "Tarih Aralığı Seçin:",
                options=["Son 1 Ay", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl"],
                index=0
            )
        
        # Map range to days
        range_map = {"Son 1 Ay": 30, "Son 3 Ay": 90, "Son 6 Ay": 180, "Son 1 Yıl": 365}
        days_limit = range_map[time_range]
        st.caption("ℹ️ Apple: Mağaza linki veya ID (id...), Play Store: Link veya paket adı (com...) geçerlidir.")


    if store_url.strip():
        u = store_url.strip()
        platform = None
        app_id = None
        country = "tr" 
        
        # Logic Improvement: Flexible link and ID detection
        if "play.google.com" in u:
            platform = "google"
            match = re.search(r"id=([^&/]+)", u)
            if match: app_id = match.group(1)
        elif "apple.com" in u:
            platform = "apple"
            # Search for id followed by digits
            match = re.search(r"id(\d+)", u)
            if match: app_id = match.group(1)
            # Try to catch country code like apple.com/tr/app/...
            country_match = re.search(r"apple\.com/([a-z]{2,3})/", u)
            if country_match: country = country_match.group(1)
        else:
            # Direct ID or prefixed ID
            clean_u = u.lower()
            if clean_u.startswith("id") and clean_u[2:].isdigit():
                platform = "apple"
                app_id = clean_u[2:]
            elif clean_u.isdigit():
                platform = "apple"
                app_id = clean_u
            elif "." in u and re.match(r"^[a-zA-Z0-9._]+$", u):
                platform = "google"
                app_id = u

        if not platform or not app_id:
            if store_url.strip():
                st.warning("⚠️ Geçerli bir Play Store veya App Store linki bulunamadı.")
        else:
            with st.container():
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    st.markdown(f"#### 🚀 {time_range} yorumları mağazadan çekiliyor...")
                    if lottie_loading:
                        st_lottie(lottie_loading, height=150, key="fetch_loader")
                    else:
                        st.info("İşlem devam ediyor, lütfen bekleyin...")
                
                fetched_comments = []
                threshold_date = datetime.now() - timedelta(days=days_limit)
                
                try:
                    if platform == "google":
                        fetched_comments = fetch_google_play_reviews(app_id, days_limit)
                    elif platform == "apple":
                        results = get_app_store_reviews(app_id, country)
                        if not results:
                            alt_country = 'tr' if country != 'tr' else 'us'
                            results = get_app_store_reviews(app_id, alt_country)
                        
                        for r in results:
                            r_date = r.get('date')
                            if r_date and r_date >= threshold_date:
                                text = r.get('text', '')
                                if is_valid_comment(text):
                                    fetched_comments.append(r)

                    if fetched_comments:
                        # Clear loading animation
                        loading_placeholder.empty()
                        MAX_REVIEWS = 500
                        if len(fetched_comments) > MAX_REVIEWS:
                            total_found = len(fetched_comments)
                            # Sort by date (newest first) to ensure we always get the *most recent* ones up to threshold
                            fetched_comments.sort(key=lambda x: x['date'], reverse=True)
                            fetched_comments = fetched_comments[:MAX_REVIEWS]
                            
                            min_dt = min([r['date'] for r in fetched_comments if r.get('date')]).strftime('%d-%m-%Y')
                            max_dt = max([r['date'] for r in fetched_comments if r.get('date')]).strftime('%d-%m-%Y')
                            
                            st.warning(f"⚠️ Toplamda **{total_found}** yorum bulundu. Bu yorumların arasından en güncel olan **{MAX_REVIEWS}** tanesi analize dahil ediliyor. (Seçilen yorumların tarih aralığı: {min_dt} ile {max_dt} arasındadır)")
                        
                        st.session_state.comments_to_analyze = fetched_comments
                        st.success(f"✅ **{len(st.session_state.comments_to_analyze)}** adet {time_range} yorumu başarıyla çekildi!")
                    else:
                        st.info(f"ℹ️ {time_range} kriterine uygun yorum bulunamadı.")
                except Exception as e:
                    st.error(f"⚠️ Yorumlar çekilirken bir hata oluştu: {e}")
        
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
                            # Metadata keywords
                            if any(k in col_l for k in ["id", "rating", "star", "puan", "date", "tarih"]): score -= 25
                            # Content Analysis
                            sample = df_upload[col].dropna().head(10).astype(str).tolist()
                            if sample:
                                avg_len = sum(len(s) for s in sample) / len(sample)
                                if avg_len > 30: score += 15
                                if avg_len < 10: score -= 20
                                common_tr = [" bir ", " bu ", " çok ", " ve ", " ama ", " için "]
                                text_blobs = " ".join(sample).lower()
                                if any(w in text_blobs for w in common_tr): score += 15
                            scores.append((score, col))
                        
                        scores.sort(key=lambda x: x[0], reverse=True)
                        col_name = scores[0][1] if scores else df_upload.columns[0]

                        # Unified Status Row (File info + Auto column)
                        st.markdown(f"""
                        <div style="display: flex; justify-content: space-between; align-items: center; background-color: #F0F9FF; padding: 10px 15px; border-radius: 10px; border: 1px solid #E0F2FE; margin-bottom: 5px;">
                            <div style="color: #0369a1; font-weight: 600; font-size: 0.9rem;">
                                ℹ️ Dosya okundu: {len(df_upload)} satır
                            </div>
                            <div style="font-size: 0.9rem; font-weight: 600; color: #475569;">
                                ✨ Otomatik Seçilen Sütun: <span class="column-badge">{col_name}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        
                        if col_name:
                            # NEW FEATURE: Dosya Istatistikleri
                            # Removed line separator for tighter look
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
                                if date_col: meta_status.append("Tarih")
                                if rate_col: meta_status.append("Puan")
                                st.write("**Bulunan Ek Veriler:**")
                                st.write(", ".join(meta_status) if meta_status else "Yok")


                            # OPTIMIZED: Vectorized Data Processing for performance
                            valid_masks = df_upload[col_name].astype(str).apply(is_valid_comment)
                            
                            processed_df = pd.DataFrame({
                                "text": df_upload[col_name].astype(str).str.strip(),
                                "is_valid": valid_masks
                            })
                            
                            # Vectorized Date Capture
                            if date_col:
                                try:
                                    processed_df["date"] = pd.to_datetime(df_upload[date_col], errors='coerce')
                                    # If mostly failed, try dayfirst
                                    if processed_df["date"].isnull().sum() > len(processed_df) * 0.5:
                                        processed_df["date"] = pd.to_datetime(df_upload[date_col], errors='coerce', dayfirst=True)
                                    # Clean timezone info for st.session_state compatibility
                                    processed_df["date"] = processed_df["date"].apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) and hasattr(x, 'tzinfo') else x)
                                except:
                                    pass
                            
                            # Vectorized Rating Capture
                            if rate_col:
                                processed_df["rating"] = df_upload[rate_col].astype(str)
                            
                            # Final Filter: Keep rows that are either valid comments or have a rating
                            mask = processed_df["is_valid"] | (processed_df["rating"].notnull() if rate_col else False)
                            final_comments_df = processed_df[mask]
                            all_comments = final_comments_df.to_dict('records')
                            valid_in_file = int(final_comments_df["is_valid"].sum())

                            st.caption(f"Bu dosyadan {valid_in_file} gecerli yorum eklendi.")
                            
            except Exception as e:
                st.error(f"⚠️ {uploaded_file.name} okuma hatası: {e}")
        
        if all_comments:
            MAX_REVIEWS = 500
            if len(all_comments) > MAX_REVIEWS:
                st.warning(f"⚠️ Dosyadaki ilk {MAX_REVIEWS} yorum analize alınmıştır (Toplam: {len(all_comments)} satır).")
                all_comments = all_comments[:MAX_REVIEWS]
            st.session_state.comments_to_analyze = all_comments
            st.success(f"📋 Toplam **{len(st.session_state.comments_to_analyze)}** gerçek yorum analiz için hazır!")

with tab3:
    text_input = st.text_area(
        "Yorumları alt alta girin:",
        height=200,
        placeholder="Örn: Harika uygulama!\nKötü performans...",
        key="manual_text_input"
    )
    if text_input.strip():
        raw_lines = text_input.split('\n')
        processed_comments = []
        
        # Date regex for "Jan 23, 2026 - User"
        store_meta_regex = r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{1,2},?\s+\d{4}\s*-\s*.*$"
        
        skip_dev_block = False
        
        for line in raw_lines:
            l = line.strip()
            if not l: continue
            
            # Detect Store Metadata (Date - User) -> This starts a NEW user review block
            if re.search(store_meta_regex, l, re.IGNORECASE):
                skip_dev_block = False # Reset on new review
                # If we have a previous line in progress, it's likely a TITLE or NICKNAME. Remove it.
                if processed_comments and len(processed_comments[-1]["text"]) < 85:
                    processed_comments.pop()
                continue
                
            # Detect Developer Response Header -> Starts a block to IGNORE
            if any(k in l.lower() for k in ["developer response", "geliştirici cevabı"]):
                skip_dev_block = True
                continue
            
            if skip_dev_block:
                continue

            if is_valid_comment(l):
                processed_comments.append({"text": l})
                
        if processed_comments:
            MAX_REVIEWS = 500
            if len(processed_comments) > MAX_REVIEWS:
                st.warning(f"⚠️ En fazla {MAX_REVIEWS} adet yorum girilebilir. Fazlası kırpıldı.")
                processed_comments = processed_comments[:MAX_REVIEWS]
            st.session_state.comments_to_analyze = processed_comments
            st.success(f"✏️ Toplam **{len(st.session_state.comments_to_analyze)}** geçerli satır eklendi!")

# Update the main reference (Important for analysis button)
comments_to_analyze = st.session_state.comments_to_analyze


# ── Analiz Yapılandırması ──────────────────────
if comments_to_analyze:
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("### Analiz Yapılandırması")
    
    n = len(comments_to_analyze)

    def fmt_time(secs):
        m, s = divmod(secs, 60)
        return f"{m} dakika {s} saniye" if m > 0 else f"{s} saniye"

    mode_idx = st.radio(
        "Analiz hızı ve doğruluk dengesini seçin:",
        options=[0, 1],
        format_func=lambda x: ["🚀 Hızlı", "🎯 Yavaş (Daha Tutarlı)"][x],
        captions=[
            f"Genel değerlendirmeler — tahmini {fmt_time(n * 1)}",
            f"Çok daha doğru sonuçlar — tahmini {fmt_time(n * 2)}"
        ],
        horizontal=True,
        key="analysis_mode"
    )





def get_gemini_sentiment(text, model_name='gemini-2.5-flash'):
    if not HAS_GEMINI:
        return None
    
    # Fallback zinciri: yeni API key'lerde eski modeller çalışmayabilir
    fallback_chain = ['gemini-2.5-flash', 'gemini-2.5-pro']
    models_to_try = [model_name] + [m for m in fallback_chain if m != model_name]
    
    for current_model in models_to_try:
        try:
            prompt = f"""Sen çok dilli (Türkçe, İngilizce, Arapça vb.) bir uygulama mağaza yorumu duygu analizi uzmanısın.
Aşağıdaki yorumu hangi dilde olursa olsun analiz et ve 3 kategoriye puan ver. Toplam 1.0 olmalı.

KATEGORİLER:
- olumlu: Kullanıcı memnun, övüyor, teşekkür ediyor, tavsiye ediyor. (happy, great, love, ممتاز, احبه vb.)
- olumsuz: Kullanıcı şikayetçi, sorun yaşıyor, kızgın, hayal kırıklığı var. (bad, terrible, سيء, أسوأ vb.)
- istek_gorus: Tarafsız öneri, soru, beklenti. Olumlu da olumsuz da değil. (when will X?, please add Y vb.)

KARAR KURALLARI (önem sırasına göre):
1. Son cümle/edit baskındır. Başta şikayet, sonda çözüm varsa → olumlu.
2. Başta iltifat, sonda şikayet varsa → olumsuz.
3. Ironi/alaycılık → olumsuz.
4. Karışık ama net şikayet sonuçlanıyorsa → olumsuz.
5. Karışık ama net memnuniyet sonuçlanıyorsa → olumlu.

SOMUT ÖRNEKLER - TÜRKÇE - OLUMLU:
"harika uygulama teşekkürler" → {{"olumlu":0.95,"olumsuz":0.02,"istek_gorus":0.03}}
"çok güzel, mükemmel" → {{"olumlu":0.95,"olumsuz":0.02,"istek_gorus":0.03}}
"beş yıldız hak ediyor" → {{"olumlu":0.90,"olumsuz":0.05,"istek_gorus":0.05}}
"güncelleme sonrası düzeldi sağolun" → {{"olumlu":0.88,"olumsuz":0.07,"istek_gorus":0.05}}
"işe yarıyor, memnunum" → {{"olumlu":0.85,"olumsuz":0.08,"istek_gorus":0.07}}
"fena değil" → {{"olumlu":0.65,"olumsuz":0.15,"istek_gorus":0.20}}

SOMUT ÖRNEKLER - TÜRKÇE - OLUMSUZ:
"uygulama açılmıyor, düzeltin" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"donuyor ve kapanıyor, berbat" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"yaramaz bu uygulama" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"iyiydi ama son güncellemeden sonra bozuldu" → {{"olumlu":0.05,"olumsuz":0.90,"istek_gorus":0.05}}
"teşekkürler ama hâlâ açılmıyor" → {{"olumlu":0.05,"olumsuz":0.90,"istek_gorus":0.05}}
"helal olsun, yine çöktü" → {{"olumlu":0.03,"olumsuz":0.92,"istek_gorus":0.05}}
"para iade etmiyorlar, dolandırıcılık" → {{"olumlu":0.02,"olumsuz":0.96,"istek_gorus":0.02}}
"çok yavaş, kasıyor" → {{"olumlu":0.03,"olumsuz":0.94,"istek_gorus":0.03}}
"reklam çok fazla, rahatsız edici" → {{"olumlu":0.05,"olumsuz":0.85,"istek_gorus":0.10}}

SOMUT ÖRNEKLER - TÜRKÇE - İSTEK/GÖRÜŞ:
"şu özelliği ekleseniz çok iyi olur" → {{"olumlu":0.10,"olumsuz":0.05,"istek_gorus":0.85}}
"ne zaman karanlık mod gelecek?" → {{"olumlu":0.05,"olumsuz":0.05,"istek_gorus":0.90}}

SOMUT ÖRNEKLER - İNGİLİZCE - OLUMLU:
"Great app, love it!" → {{"olumlu":0.95,"olumsuz":0.02,"istek_gorus":0.03}}
"Good" → {{"olumlu":0.85,"olumsuz":0.05,"istek_gorus":0.10}}
"Amazing experience, highly recommend!" → {{"olumlu":0.95,"olumsuz":0.02,"istek_gorus":0.03}}
"Works perfectly, no issues" → {{"olumlu":0.93,"olumsuz":0.03,"istek_gorus":0.04}}
"Excellent customer service, got my refund quickly" → {{"olumlu":0.90,"olumsuz":0.05,"istek_gorus":0.05}}

SOMUT ÖRNEKLER - İNGİLİZCE - OLUMSUZ:
"Terrible app, keeps crashing" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"Not recommended" → {{"olumlu":0.03,"olumsuz":0.92,"istek_gorus":0.05}}
"Doesn't work at all" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"App crashes every time I open it" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"Can't login, stuck on loading screen" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"I never received my refund after contacting support many times!" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"Prices shown are different from what you actually pay" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"Received wrong item and no one is helping" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"Scam! Don't trust them" → {{"olumlu":0.02,"olumsuz":0.96,"istek_gorus":0.02}}

SOMUT ÖRNEKLER - İNGİLİZCE - İSTEK/GÖRÜŞ:
"When will the US be able to use this app?" → {{"olumlu":0.05,"olumsuz":0.05,"istek_gorus":0.90}}
"Please add dark mode" → {{"olumlu":0.10,"olumsuz":0.05,"istek_gorus":0.85}}

SOMUT ÖRNEKLER - ARAPÇA - OLUMLU:
"ممتاز" → {{"olumlu":0.92,"olumsuz":0.03,"istek_gorus":0.05}}
"احبه 💕🥰" → {{"olumlu":0.95,"olumsuz":0.02,"istek_gorus":0.03}}
"رائع جداً وسهل الاستخدام" → {{"olumlu":0.93,"olumsuz":0.03,"istek_gorus":0.04}}
"أفضل تطبيق، أنصح به الجميع" → {{"olumlu":0.95,"olumsuz":0.02,"istek_gorus":0.03}}

SOMUT ÖRNEKLER - ARAPÇA - OLUMSUZ:
"سيء جدا جدا جدا جدا أخيس تطبيق" → {{"olumlu":0.02,"olumsuz":0.96,"istek_gorus":0.02}}
"أخيس تطبيق" → {{"olumlu":0.02,"olumsuz":0.96,"istek_gorus":0.02}}
"اسوأ تعامل" → {{"olumlu":0.02,"olumsuz":0.96,"istek_gorus":0.02}}
"لا يعمل التطبيق" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"يتعطل باستمرار" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"لا أستطيع الدخول إلى حسابي" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"لم أستلم أموال الاسترجاع رغم تواصلي مرات عديدة" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"الموقع ممتاز لكن عند الاسترجاع لا تصلك الاموال" → {{"olumlu":0.05,"olumsuz":0.88,"istek_gorus":0.07}}
"سعر المنتج قبل اضافته للسله يختلف عن بعد الاضافة" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"يتم شحن الوان مختلفه واغراض غير اصليه بجودة رديئة" → {{"olumlu":0.02,"olumsuz":0.96,"istek_gorus":0.02}}

SOMUT ÖRNEKLER - ARAPÇA - İSTEK/GÖRÜŞ:
"أتمنى أن يضيفوا خاصية البحث بالصور" → {{"olumlu":0.10,"olumsuz":0.05,"istek_gorus":0.85}}
"متى سيكون التطبيق متاحاً في دولتي؟" → {{"olumlu":0.05,"olumsuz":0.05,"istek_gorus":0.90}}
"مو سامح لي اختار دولة" → {{"olumlu":0.05,"olumsuz":0.40,"istek_gorus":0.55}}

ÇIKTI KURALI: SADECE JSON döndür, başka hiçbir şey yazma.
{{"olumlu": X, "olumsuz": Y, "istek_gorus": Z}}

Yorum: "{text}"
"""
            response = GEMINI_CLIENT.models.generate_content(
                model=current_model,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0)
            )
            content = response.text
            match = re.search(r'\{.*?\}', content, re.DOTALL)
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
            if "404" in err_str and current_model != models_to_try[-1]:
                continue
            elif "429" in err_str or "quota" in err_str.lower():
                return {"_error": "quota"}
            else:
                return {"_error": f"⚠️ Gemini API hatası: {err_str[:120]}"}
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
        # Eski sonuçları hemen temizle — yeni analiz başlamadan önce
        if "bulk_results" in st.session_state:
            del st.session_state["bulk_results"]

        time_display = st.empty()  # MOVED: Immediately below button
        progress_bar = st.progress(0)
        status_text = st.empty()
        quota_info = st.empty()
        st.warning("🔴 Analiz süresince bu sayfayı kapatmayın veya yenilemeyin. Verileriniz kaybolabilir.")
        st.session_state['_quota_hits'] = 0
            
        # Seçilen moda göre model ve bekleme süresi (0=Hızlı, 1=Yavaş)
        mode_idx = st.session_state.get("analysis_mode", 0)
        if mode_idx == 0:  # Hızlı
            ANALYSIS_MODEL = 'gemini-2.5-flash'
            DELAY_SECS = 0
            RPM_LIMIT = 500
        else:  # Yavaş
            ANALYSIS_MODEL = 'gemini-2.5-pro'
            DELAY_SECS = 0  
            RPM_LIMIT = 300



        start_time = time.time()
        total_items = len(comments_to_analyze)
        est_total_secs = total_items * (1 if mode_idx == 0 else 2) # tahmin: hızlıda 1sn, yavaşta 2sn

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


        import concurrent.futures

        def fetch_sentiment_worker(args):
            idx, entry = args
            comment = entry["text"]
            is_valid = entry.get("is_valid", True)
            if not is_valid or not comment:
                return idx, entry, {"olumlu": 0, "olumsuz": 0, "istek_gorus": 0}, "—", None
            
            res_api = get_gemini_sentiment(comment, model_name=ANALYSIS_MODEL)
            err = None
            if res_api is None or "_error" in res_api:
                err = res_api["_error"] if res_api else "unknown"
                res_api = heuristic_analysis(comment)
            
            scores = {"Olumlu": res_api['olumlu'], "Olumsuz": res_api['olumsuz'], "İstek/Görüş": res_api['istek_gorus']}
            verdict = max(scores, key=scores.get)
            return idx, entry, res_api, verdict, err

        completed_count = 0
        workers = 10 if mode_idx == 0 else 6

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            tasks = [executor.submit(fetch_sentiment_worker, (i, e)) for i, e in enumerate(comments_to_analyze)]
            
            for future in concurrent.futures.as_completed(tasks):
                i, entry, res, verdict, err = future.result()
                completed_count += 1
                
                status_text.text(f"Analiz ediliyor: {completed_count} / {total_items}")
                update_time(completed_count - 1, total_items, start_time)
                
                if err == "quota":
                    q = st.session_state.get('_quota_hits', 0) + 1
                    st.session_state['_quota_hits'] = q
                    if q == 1:
                        quota_info.info(f"ℹ️ Gemini kota aşıldı. Bu yorum yerel motorla değerlendirildi. (Model: dakikada en fazla {RPM_LIMIT} istek)")
                    elif q > 1:
                        quota_info.info(f"ℹ️ Toplam **{q} yorum** kota nedeniyle yerel motorla değerlendirildi.")
                elif err:
                    st.warning(err)
                
                comment = entry["text"]
                date = entry.get("date")
                ticker_date = ""
                if date:
                    try: ticker_date = f"📅 {date.strftime('%d-%m-%Y')}"
                    except: pass

                ticker_color = "#34D399" if verdict == "Olumlu" else ("#F87171" if verdict == "Olumsuz" else "#60A5FA")
                ticker_placeholder.markdown(f"""
                <div style="border: 2px solid {ticker_color}; padding: 15px; border-radius: 12px; background: #FFFFFF; margin: 10px 0;">
                    <div style="display: flex; justify-content: space-between; font-size: 0.85em; color: #64748b; margin-bottom: 5px;">
                        <span>⚡ ŞU AN EKLENEN (#{i+1})</span>
                        <span>{ticker_date}</span>
                    </div>
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
                
                progress_bar.progress(completed_count / total_items)

        # Sonuçları orjinal sıraya geri diz
        bulk_results = sorted(bulk_results, key=lambda x: x["No"])

        ticker_placeholder.markdown(f"""
        <div style="border: 2px solid #10B981; padding: 20px; border-radius: 12px; background: #ECFDF5; margin: 10px 0; text-align: center;">
            <div style="font-size: 2em; margin-bottom: 10px;">✅</div>
            <div style="font-weight: 700; color: #065F46; font-size: 1.2em;">ANALİZ TAMAMLANDI</div>
            <div style="font-size: 0.9em; color: #047857; margin-top: 5px;">Toplam {total_items} satır başarıyla işlendi.</div>
        </div>
        """, unsafe_allow_html=True)

        
        st.session_state.bulk_results = bulk_results
        status_text.success("Analiz Başarıyla Tamamlandı!")
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
    margin-bottom: 1.25rem;
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
    padding: 15px;
    margin-bottom: 15px;
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
    st.markdown("### Analiz Özeti")
    
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
        
        # Renkleri kategoriye göre sabit harita ile ata (sıraya bağlı değil)
        color_map = {"Olumlu": "#10b981", "Olumsuz": "#f43f5e", "İstek/Görüş": "#3b82f6"}
        pie_colors = [color_map.get(d, "#94a3b8") for d in pie_data["Duygu"]]
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_data["Duygu"], 
            values=pie_data["Sayı"],
            hole=0.5,
            pull=[0.05, 0.05, 0.05],
            marker=dict(
                colors=pie_colors,
                line=dict(color='#FFFFFF', width=3)
            ),
            textinfo='percent+label',
            textfont=dict(color='#000000', size=12),
            insidetextorientation='radial'
        )])
        
        fig_pie.update_layout(
            height=380,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#000000', family="Poppins, sans-serif"),
            legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5, "font": {"color": "#000000"}},
            margin={"t": 30, "b": 30, "l": 0, "r": 0}
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_summary:
        st.write("#### Yapay Zeka Görüşü")
        if counts.idxmax() == "Olumlu":
            st.success(f"Topluluk genel olarak **Olumlu** bir tavır sergiliyor. ({m_olumlu} yorum) Genel olarak kullanıcı kitlesi, uygulamanın sunduğu temel hizmetlerden, arayüz tasarımından ve kullanım kolaylığından yüksek düzeyde memnuniyet duyuyor diyebiliriz. Özellikle düzenli kullanıcılar uygulamanın günlük hayattaki işlevselliğini olumlu bularak tavsiye etme eğiliminde. Sistem performansı, hız ve güvenilirlik beklentileri büyük ölçüde karşılanıyor. Son güncellemelerle birlikte gelen yenilikler pozitif karşılanmış gibi görünüyor. Kullanıcıların markaya olan güveni bu aşamada sağlam temeller üzerinde duruyor. Müşteri hizmetlerinin ve destek birimlerinin sorunlara hızlı reaksiyon göstermesi de bu olumlu havayı destekleyen ana etkenlerden biri olabilir. Yine de aralardaki küçük oranlı şikayetleri dikkatle ele alıp, bu %100'e yakın memnuniyet oranını koruyacak stratejik adımların devam ettirilmesi oldukça önemli.")
        elif counts.idxmax() == "Olumsuz":
            st.error(f"Dikkat çeken **Olumsuz** bir eğilim var. ({m_olumsuz} yorum) Analiz edilen veri setinde kullanıcıların çok ciddi hayal kırıklıkları ve sistemsel şikayetleri olduğu açıkça görülmektedir. Özellikle kilitlenme, yavaşlık veya beklenen özelliklerin çalışmaması gibi kronikleşmiş teknik problemler kullanıcı deneyimini ciddi oranda baltalıyor. İade sorunları, müşteri hizmetlerinin ulaşılamaz olması veya vaat edilenle karşılaşılan hizmetin uyuşmaması gibi temel şikayetler marka imajına an itibariyle zarar veriyor. Kullanıcılar uygulamanın temel fonksiyonlarını bile kullanırken pürüzlerle karşılaştıkları için platformu terk etme veya rakiplere yönelme potansiyeline sahipler. Acil ve agresif bir hata ayıklama (bug-fixing) sürecine gidilmeli, müşteri destek hattının kapasitesi artırılmalı ve kullanıcılardan gelen yapısal eleştiriler bir an önce yazılım geliştirme döngüsüne entegre edilmelidir.")
        else:
            st.info(f"Kullanıcılar yoğun şekilde **İstek ve Görüş** paylaşıyor. ({m_istek} yorum) Kullanıcı tabanı şu anda markaya veya uygulamaya karşı keskin bir öfke yahut aşırı bir coşku beslemek yerine, daha akılcı ve beklenti odaklı bir tutum içinde. Yorumların geneli, sistemin temel ihtiyaçları karşıladığını ancak modern standartlara veya rakiplere kıyasla eksik bazı ufak tefek özellikler veya yaşam kalitesi (QoL) güncellemeleri barındırdığına işaret ediyor. Kullanıcılar aslında uygulamanın potansiyelinin farkında ve bu potansiyeli maksimize edecek yenilikler (örneğin karanlık mod, daha geniş dil desteği, pratik menü tasarımları vb.) görmek istiyorlar. Bu grup sadık bir kitleye dönüşmeye oldukça yakın; geliştirici ekip eğer bu geri bildirimleri dikkate alıp istenen özellikleri sisteme entegre ederse, tarafsız duran bu kitle çok hızlı bir şekilde savunucu ve sadık kullanıcılara (olumlu) evrilecektir.")

    # NEW: Star Rating Distribution Chart (Sütunlu ve Renkli)
    if "Puan" in df.columns and df["Puan"].notnull().any():
        st.markdown("---")
        
        # UI for Frequency Selection
        g_col1, g_col2 = st.columns([2, 1])
        with g_col1:
            st.write("#### Puan Dağılımı Trendi")
        with g_col2:
            freq = st.radio("Zaman Ölçeği:", ["Günlük", "Haftalık", "Aylık"], index=2, horizontal=True, key="puan_freq_sel")

        df_puan = df.dropna(subset=["Tarih", "Puan"]).copy()
        try:
            # Ensure ratings are integers 1-5 for clean legend
            df_puan["Puan_val"] = pd.to_numeric(df_puan["Puan"], errors='coerce').fillna(0).astype(int)
            df_puan = df_puan[(df_puan["Puan_val"] >= 1) & (df_puan["Puan_val"] <= 5)]
            
            if not df_puan.empty:
                df_puan["Tarih_dt"] = pd.to_datetime(df_puan["Tarih"])
                
                # Show Date Range Info
                min_d = df_puan["Tarih_dt"].min().strftime('%d-%m-%Y')
                max_d = df_puan["Tarih_dt"].max().strftime('%d-%m-%Y')
                st.caption(f"📅 **Tespit Edilen Tarih Aralığı:** {min_d} ile {max_d}")

                # Month names map
                tr_months = {1:"Ocak", 2:"Şubat", 3:"Mart", 4:"Nisan", 5:"Mayıs", 6:"Haziran", 
                             7:"Temmuz", 8:"Ağustos", 9:"Eylül", 10:"Ekim", 11:"Kasım", 12:"Aralık"}

                # Resample based on choice
                if freq == "Haftalık":
                    df_puan["Grup"] = df_puan["Tarih_dt"].dt.to_period('W').apply(lambda r: r.start_time)
                    df_puan["Grup_Label"] = df_puan["Grup"].apply(lambda x: f"{x.day} {tr_months[x.month]} {x.year}")
                    title_txt = "Haftalık Puan Dağılımı"
                elif freq == "Aylık":
                    df_puan["Grup_Label"] = df_puan["Tarih_dt"].apply(lambda x: f"{tr_months[x.month]} {x.year}")
                    df_puan["Grup"] = df_puan["Tarih_dt"].dt.to_period('M').apply(lambda r: r.start_time)
                    title_txt = "Aylık Puan Dağılımı"
                else:
                    df_puan["Grup_Label"] = df_puan["Tarih_dt"].dt.strftime('%d-%m-%Y')
                    df_puan["Grup"] = df_puan["Tarih_dt"].dt.date
                    title_txt = "Günlük Puan Dağılımı"

                # Group and Sort
                dist_trend = df_puan.groupby(["Grup", "Grup_Label", "Puan_val"]).size().reset_index(name='Oy Sayısı')
                dist_trend["Puan_Label"] = dist_trend["Puan_val"].apply(lambda x: f"{x} Yıldız")
                dist_trend = dist_trend.sort_values(["Grup", "Puan_val"], ascending=[True, True])

                fig_dist = px.bar(dist_trend, x="Grup_Label", y="Oy Sayısı", color="Puan_Label",
                                 title=title_txt,
                                 color_discrete_map={
                                     "1 Yıldız": "#08306b",
                                     "2 Yıldız": "#08519c",
                                     "3 Yıldız": "#2171b5",
                                     "4 Yıldız": "#6baed6",
                                     "5 Yıldız": "#deebf7"
                                 },
                                 category_orders={"Puan_Label": ["1 Yıldız", "2 Yıldız", "3 Yıldız", "4 Yıldız", "5 Yıldız"]},
                                 labels={"Puan_Label": "", "Grup_Label": "Zaman", "Oy Sayısı": "Sayı"})
                
                fig_dist.update_layout(
                    height=450, 
                    margin={"t": 60, "b": 100, "l": 10, "r": 10},
                    xaxis_title="",
                    yaxis_title="Yorum / Puan Sayısı",
                    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"color": "#000000"}},
                    barmode='stack',
                    bargap=0.3,
                    template="plotly_white",
                    paper_bgcolor="#F0F9FF",
                    plot_bgcolor="#F0F9FF",
                    font={"color": "#000000", "family": "Poppins, sans-serif"},
                    title_font={"color": "#000000", "size": 18}
                )
                
                # Force categorical X axis and black ticks
                fig_dist.update_xaxes(type='category', tickangle=-45, tickfont={"color": "#000000"}, title_font={"color": "#000000"})
                fig_dist.update_yaxes(tickfont={"color": "#000000"}, title_font={"color": "#000000"})
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
                               labels={"Baskın Duygu": ""},
                               custom_data=["Hafta_str", "Baskın Duygu"])
            
            fig_trend.update_layout(height=350, margin={"t": 80, "b": 40, "l": 10, "r": 10},
                                   legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"color": "#000000"}},
                                   xaxis_title="Tarih (Haftalık)", yaxis_title="Yorum Sayısı",
                                   template="plotly_white",
                                   paper_bgcolor="#F0F9FF",
                                   plot_bgcolor="#F0F9FF",
                                   font={"color": "#000000", "family": "Poppins, sans-serif"},
                                   title_font={"color": "#000000", "size": 18},
                                   clickmode='event+select')
            
            fig_trend.update_xaxes(tickfont={"color": "#000000"}, title_font={"color": "#000000"})
            fig_trend.update_yaxes(tickfont={"color": "#000000"}, title_font={"color": "#000000"})
            
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
            
            # Format extra info (Rating)
            extra_info = ""
            if "Puan" in row and pd.notnull(row["Puan"]):
                extra_info += f" | ⭐ {row['Puan']}"
            
            date_tag = ""
            if "Tarih" in row and pd.notnull(row["Tarih"]):
                try: 
                    d = pd.to_datetime(row["Tarih"])
                    date_tag = f"Tarih: {d.strftime('%d-%m-%Y')}"
                except: pass

            # Map sentiment to color dot
            dot_colors = {"Olumlu": "#10b981", "Olumsuz": "#f43f5e", "İstek/Görüş": "#3b82f6"}
            s_color = dot_colors.get(sentiment, "#94a3b8")
            sentiment_indicator = f'<span style="display: inline-block; width: 10px; height: 10px; background-color: {s_color}; border-radius: 50%; margin: 0 4px; vertical-align: middle;"></span>'

            st.markdown(f"""
            <div class="{cls}">
                <div style="display: flex; justify-content: space-between; margin-bottom: 4px;">
                    <span style="font-size: 0.8em; color: #94a3b8; font-weight: 500;">#{row['No']} | {sentiment_indicator}{extra_info}</span>
                    <span style="font-size: 0.8em; color: #94a3b8;">{date_tag}</span>
                </div>
                <div style="color: #1E293B; line-height: 1.5;">{row['Yorum']}</div>
            </div>
            """, unsafe_allow_html=True)

    # --- Tabs and Unified Display ---
    st.write("### Yorum Listesi")
    
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
        st.download_button(label=" Sonuçları Excel Olarak İndir", data=output.getvalue(), file_name="analiz.xlsx", key="bulk_dl")
    except: pass

# Footer
st.divider()
st.caption("Geliştiren: Cem Evecen")

