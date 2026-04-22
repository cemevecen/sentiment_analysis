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
import urllib.parse
from typing import List, Dict, Any, Optional, Union, cast
from datetime import datetime, timedelta
from google_play_scraper import Sort, reviews as play_reviews
# Removed app-store-scraper due to dependency conflicts with streamlit
from dotenv import load_dotenv
import textwrap
if os.path.exists(".env"):
    load_dotenv(override=True)

# Set Page Config
st.set_page_config(
    page_title="AI Duygu Analizi",
    layout="centered"
)

# API Configuration: Optimized via Caching
@st.cache_resource(show_spinner="API yapılandırılıyor...")
def setup_api():
    # Priority: 1. GOOGLE_API_KEY (SDK standard), 2. GEMINI_API_KEY, 3. API_KEY
    keys_to_check = ["GOOGLE_API_KEY", "GEMINI_API_KEY", "API_KEY"]
    
    api_key = None
    
    # Check Environment Variables
    for k in keys_to_check:
        val = os.getenv(k)
        if val and str(val).strip():
            api_key = str(val).strip()
            break
            
    # Check Streamlit Secrets if still not found
    if not api_key:
        try:
            for k in keys_to_check:
                val = st.secrets.get(k)
                if val and str(val).strip():
                    api_key = str(val).strip()
                    break
        except:
            pass

    if api_key:
        try:
            # Initialize client with the found key
            client = genai.Client(api_key=api_key)
            return client
        except Exception as e:
            st.error(f"API Client başlatma hatası: {e}")
            return None
    return None

GEMINI_CLIENT = setup_api()
HAS_GEMINI = GEMINI_CLIENT is not None

# Special check for Streamlit Cloud users
if not HAS_GEMINI and "streamlit" in str(st.__file__).lower():
    st.sidebar.error("⚠️ Gemini API Key bulunamadı! Lütfen Streamlit Cloud 'Secrets' kısmına GOOGLE_API_KEY tanımlayın.")
    if st.sidebar.button("API'yi Yeniden Kontrol Et"):
        st.cache_resource.clear()
        st.rerun()
elif HAS_GEMINI and "GEMINI_CLIENT" in locals():
    # Optional: Test client connectivity if needed, but we'll stick to a simple success indicator
    pass

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
def is_valid_comment(text: Any) -> bool:
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

def get_app_store_reviews(app_id: str, _progress_callback: Any = None, _days_limit: int = 30) -> List[Dict[str, Any]]:
    """Massive Parallel App Store Fetcher (40+ Countries) to break all limits"""
    import concurrent.futures
    all_reviews_map: Dict[str, Dict[str, Any]] = {}
    now = datetime.now()
    threshold_dt = now - timedelta(days=_days_limit)
    
    # 40 target countries to find all possible Turkish reviews globally
    countries = [
        'tr', 'us', 'de', 'az', 'nl', 'fr', 'gb', 'at', 'be', 'ch', 'kz', 'uz', 'tm', 'kg', 'ru',
        'cy', 'gr', 'ro', 'bg', 'pl', 'hu', 'cz', 'se', 'no', 'dk', 'it', 'es', 'ca', 'au', 'sa',
        'ae', 'qa', 'kw', 'jo', 'lb', 'eg', 'ly', 'dz', 'ma', 'tn'
    ]
    
    def fetch_country_reviews(country: str):
        country_reviews = []
        for page in range(1, 11): # Max 10 pages per country (Apple limit)
            try:
                url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortBy=mostRecent/json"
                resp = requests.get(url, timeout=5)
                if resp.status_code != 200: break
                data = resp.json()
                entries = data.get('feed', {}).get('entry', [])
                if not entries: break
                if isinstance(entries, dict): entries = [entries]
                
                found_old = False
                for entry in entries:
                    content = entry.get('content', {}).get('label', '')
                    if not content or len(content.strip()) < 2: continue
                    
                    updated = entry.get('updated', {}).get('label', '')
                    try:
                        r_date = datetime.fromisoformat(updated.replace('Z', '+00:00'))
                        r_date = r_date.replace(tzinfo=None)
                    except: continue
                    
                    if r_date >= threshold_dt:
                        r_id = entry.get('id', {}).get('label', content)
                        rating = str(entry.get('im:rating', {}).get('label', '0'))
                        country_reviews.append({"id": r_id, "text": content, "date": r_date, "rating": rating})
                    else:
                        found_old = True
                
                if found_old: break
            except: break
        return country_reviews

    # Parallel execution for massive speed
    total_countries = len(countries)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_country = {executor.submit(fetch_country_reviews, c): c for c in countries}
        for future in concurrent.futures.as_completed(future_to_country):
            completed += 1
            if _progress_callback: _progress_callback(min(completed / total_countries, 0.99))
            res = future.result()
            for r in res:
                all_reviews_map[r['id']] = r
    
    if _progress_callback: _progress_callback(1.0)
    return list(all_reviews_map.values())

def fetch_google_play_reviews(app_id: str, days_limit: int, _progress_callback: Any = None) -> List[Dict[str, Any]]:
    """Massive Parallel Google Play Fetcher with Multi-Channel Depth"""
    import concurrent.futures
    from google_play_scraper import Sort, reviews as play_reviews
    
    all_fetched_map = {}
    now = datetime.now()
    threshold_date = now - timedelta(days=days_limit)
    
    # Combinations to bypass the 3k limit per sort
    sort_strategies = [Sort.NEWEST, Sort.MOST_RELEVANT]
    scores = [1, 2, 3, 4, 5]
    channels = []
    for s in sort_strategies:
        for sc in scores:
            channels.append((s, sc))
            
    def fetch_channel(sort_type, score):
        channel_data = []
        token = None
        # 30 batches = 6k reviews per channel
        for _ in range(30):
            try:
                result, token = play_reviews(
                    app_id, lang='tr', country='tr',
                    sort=sort_type, count=200,
                    filter_score_with=score,
                    continuation_token=token
                )
                if not result: break
                
                out_of_range = False
                for r in result:
                    r_at_raw = r.get('at')
                    if r_at_raw:
                        r_at = cast(datetime, r_at_raw)
                        if r_at.tzinfo: r_at = r_at.replace(tzinfo=None)
                        
                        if r_at >= threshold_date:
                            content = str(r.get('content', ''))
                            if content and len(content.strip()) >= 2:
                                r_id = r.get('reviewId', content)
                                channel_data.append({"id": r_id, "text": content, "date": r_at, "rating": str(score)})
                        else:
                            if sort_type == Sort.NEWEST: out_of_range = True
                
                if out_of_range or not token: break
            except: break
        return channel_data

    total_channels = len(channels)
    completed_channels = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        future_to_channel = {executor.submit(fetch_channel, s, sc): (s, sc) for s, sc in channels}
        for future in concurrent.futures.as_completed(future_to_channel):
            completed_channels += 1
            if _progress_callback: _progress_callback(min(completed_channels / total_channels, 0.99))
            res = future.result()
            for r in res:
                all_fetched_map[r['id']] = r
                
    if _progress_callback: _progress_callback(1.0)
    return list(all_fetched_map.values())

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
    
    /* Expand the main container width to exactly 707.2px as requested */
    [data-testid="stAppViewBlockContainer"] {
        max-width: 707.2px !important;
        padding-left: 0px !important;
        padding-right: 0px !important;
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

    .st-emotion-cache-p5mtransition, .st-emotion-cache-1vt4y6f {
        background-color: #FFFFFF !important;
        color: #000000 !important;
    }
    
    /* Header Card - White */
    .header-container {
        background-color: #FFFFFF !important;
        border: 2px solid #E2E8F0;
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
    
    /* File Uploader - White */
    [data-testid="stFileUploader"] {
        background-color: #FFFFFF !important;
        border: 2px dashed #E2E8F0 !important;
        border-radius: 16px;
        padding: 20px;
    }
    [data-testid="stFileUploadDropzone"] {
        background-color: #FFFFFF !important;
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
    
    /* Info/Alert boxes - White */
    .stAlert {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 2px solid #E2E8F0 !important;
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
    
    /* Global Button Styling - forcing White/Black as requested */
    .stButton>button, .stLinkButton>a {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #000000 !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 10px 12px !important;
        transition: all 0.2s ease !important;
        font-size: 0.95rem !important;
        text-decoration: none !important;
        display: inline-flex !important;
        align-items: center !important;
        justify-content: center !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05) !important;
        height: 50px !important; /* Fixed height for uniform size */
        width: 100% !important;
        text-align: center !important;
    }
    .stButton>button:hover, .stLinkButton>a:hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        transform: scale(1.02);
        color: #000000 !important;
    }

    /* Excel Download Button - Specific Styling */
    .stDownloadButton > button {
        background-color: #66BB6A !important; /* Pastel Green */
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        height: 50px !important;
        font-weight: 400 !important;
        font-size: 0.95rem !important;
    }
    .stDownloadButton > button * {
        color: #FFFFFF !important;
    }
    .stDownloadButton > button:hover {
        background-color: #81C784 !important;
        color: #FFFFFF !important;
        transform: scale(1.02);
    }
    
    /* Primary Analyze Button Styling - Perfectly aligned to header and inputs */
    [data-testid="stButton"] {
        width: 100% !important;
    }
    .stButton > button[kind="primary"] {
        background-color: #F4A261 !important; /* Pastel Orange */
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        height: 50px !important;
        font-weight: 600 !important;
        width: 100% !important;
        margin: 0 !important;
    }
    .stButton > button[kind="primary"] * {
        color: #FFFFFF !important;
    }
    .stButton > button[kind="primary"]:hover {
        background-color: #F8B478 !important;
        transform: scale(1.02);
    }

    /* File Uploader Button - Restored & Refined */
    [data-testid="stFileUploader"] button[kind="secondary"] {
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border: 1px solid #E2E8F0 !important;
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
        margin: 20px 0;
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
        background-color: #F8FAFC;
        border: 2px solid #FFD1B3;
        border-radius: 100px;
        padding: 25px 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
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

    /* Tab Styling - Chips/Pill Design */
    div[data-testid="stTabList"] {
        border-bottom: none !important;
        gap: 12px !important;
        margin-bottom: 10px !important;
    }
    button[data-testid="stTab"] {
        background-color: #FFFFFF !important;
        color: #1E293B !important; /* Dark text for readability on white */
        border-radius: 50px !important;
        padding: 8px 20px !important;
        border: 1px solid #E2E8F0 !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
        height: auto !important;
    }
    button[data-testid="stTab"]:hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
    }
    button[data-testid="stTab"][aria-selected="true"] {
        background-color: #818CF8 !important; /* Eflatun */
        color: white !important;
        border-color: #818CF8 !important;
        box-shadow: 0 4px 10px rgba(129, 140, 248, 0.3) !important;
        font-weight: 600 !important;
    }
    /* Hide the default Streamlit selector bar and highlight line */
    div[data-testid="stTabList"] > div:last-child,
    div[data-baseweb="tab-highlight"] {
        display: none !important;
        visibility: hidden !important;
        height: 0px !important;
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

tab1, tab2, tab3 = st.tabs(["Mağaza Linki", "Dosya Yükle (CSV/Excel)", "Metin Girişi"])

with tab1:
    with st.container(border=True):
        col_u, col_r = st.columns([2, 1])
        with col_u:
            store_url = st.text_input("Uygulama linki veya ID girin:", placeholder="Örn: com.whatsapp veya 1500198745")
            st.session_state.app_url = store_url # Sync for share report
        with col_r:
            time_range = st.selectbox(
                "Tarih Aralığı Seçin:",
                options=["Son 1 Ay", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl"],
                index=0
            )
        
        # Map range to days
        range_map = {"Son 1 Ay": 30, "Son 3 Ay": 90, "Son 6 Ay": 180, "Son 1 Yıl": 365}
        days_limit = range_map[time_range]
        st.markdown('<div style="margin-top: 6px; margin-bottom: 10px; font-size: 0.85rem; color: #64748b;">Apple: Mağaza linki veya ID (id...), Play Store: Link veya paket adı (com...) geçerlidir.</div>', unsafe_allow_html=True)


    if store_url.strip():
        u = store_url.strip()
        platform: Optional[str] = None
        app_id: str = ""
        country: str = "tr" 
        
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

        # Logic Improvement: Manual cache to avoid re-fetching on every UI interaction
        fetch_key = f"{platform}_{app_id}_{time_range}_{country}"
        
        if not platform or not app_id:
            if store_url.strip():
                st.warning("Geçerli bir Play Store veya App Store linki bulunamadı.")
        elif st.session_state.get("last_fetch_key") == fetch_key and st.session_state.get("all_fetched_pool"):
            # Already fetched, skip to results summary
            pass
        else:
            # Clear old results to keep UI in sync during new fetch
            if "bulk_results" in st.session_state:
                del st.session_state.bulk_results
            if "comments_to_analyze" in st.session_state:
                st.session_state.comments_to_analyze = []
            
            with st.container():
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    st.markdown(f"#### {time_range} yorumları mağazadan çekiliyor...")
                    if lottie_loading:
                        st_lottie(lottie_loading, height=130, key="fetch_loader")
                    p_bar = st.progress(0, text="Hazırlanıyor...")
                
                def update_fetch_progress(p: float) -> None:
                    # Clamp progress to 100% and avoid re-rendering issues
                    p_safe = min(max(float(p), 0.0), 1.0)
                    
                    # Store last progress for smooth transition if needed
                    last_p = float(st.session_state.get("_last_fetch_p", 0.0))
                    
                    # If it's a significant jump (especially at the end), animate slightly
                    if p_safe >= 1.0 and last_p < 0.95:
                        for i in range(1, 11):
                            smooth_p = last_p + (1.0 - last_p) * (i / 10.0)
                            p_bar.progress(smooth_p, text=f"Tamamlanıyor: %{int(smooth_p*100)}")
                            time.sleep(0.05)
                    else:
                        p_bar.progress(p_safe, text=f"Veriler indiriliyor: %{int(p_safe*100)}")
                    
                    st.session_state["_last_fetch_p"] = p_safe

                fetched_comments: List[Dict[str, Any]] = []
                threshold_date = datetime.now() - timedelta(days=days_limit)
                
                try:
                    # Fetch App Details first to get real name if possible
                    name_for_state = app_id
                    st_for_state = "Store"
                    if platform == "google":
                        try:
                            from google_play_scraper import app
                            info = app(app_id)
                            name_for_state = info.get('title', app_id)
                            st_for_state = "Google Play"
                        except: pass
                    elif platform == "apple":
                        st_for_state = "App Store"
                        # Try to get name from URL if it's a link
                        if "apple.com" in u and "/app/" in u:
                            try:
                                raw_name = u.split("/app/")[-1].split("/")[0].replace("-", " ")
                                name_for_state = urllib.parse.unquote(raw_name).title()
                            except: pass
                    
                    st.session_state.detected_app_name = name_for_state
                    st.session_state.detected_store_type = st_for_state

                    if platform == "google":
                        fetched_comments = fetch_google_play_reviews(app_id, days_limit, _progress_callback=update_fetch_progress)
                    elif platform == "apple":
                        def apple_cb(p: float) -> None: update_fetch_progress(p)
                        results = get_app_store_reviews(app_id, _progress_callback=apple_cb, _days_limit=days_limit)
                        
                        for r in results:
                            r_date = r.get('date')
                            if r_date is not None and isinstance(r_date, datetime) and r_date >= threshold_date:
                                text = r.get('text', '')
                                if is_valid_comment(text):
                                    fetched_comments.append(r)

                    if fetched_comments:
                        # Force 100% and a small sleep to ensure user sees completion
                        update_fetch_progress(1.0)
                        time.sleep(0.5)
                        
                        # Clear loading animation
                        loading_placeholder.empty()
                        
                        # Store all fetched comments for potential extended analysis
                        fetched_comments.sort(key=lambda x: x['date'], reverse=True)
                        st.session_state.all_fetched_pool = fetched_comments
                        st.session_state.last_fetch_key = fetch_key # Update manual cache key
                        
                        if len(fetched_comments) > 500:
                            total_found = len(fetched_comments)
                            # Use count-based take for maximum linter compatibility
                            limited_comments = []
                            for idx in range(500):
                                limited_comments.append(fetched_comments[idx])
                            
                            st.session_state.comments_to_analyze = limited_comments
                            
                            # Analyzed Range (Top 500)
                            v_dates_anal = [r.get('date') for r in limited_comments if r.get('date') is not None and isinstance(r.get('date'), datetime)]
                            # Total Range (Entire Pool)
                            v_dates_pool = [r.get('date') for r in fetched_comments if r.get('date') is not None and isinstance(r.get('date'), datetime)]
                            
                            if v_dates_anal and v_dates_pool:
                                pool_start = cast(datetime, min(v_dates_pool)).strftime('%d-%m-%Y')
                                pool_end = cast(datetime, max(v_dates_pool)).strftime('%d-%m-%Y')
                                anal_start = cast(datetime, min(v_dates_anal)).strftime('%d-%m-%Y')
                                anal_end = cast(datetime, max(v_dates_anal)).strftime('%d-%m-%Y')
                                
                                st.warning(f"""
                                    Toplamda **{total_found}** yorum bulundu (Tüm Aralık: {pool_start} - {pool_end}).
                                    Hızlı analiz için **en güncel 500 tanesi** seçildi (Analiz Aralığı: {anal_start} - {anal_end}).
                                """)
                        else:
                            st.session_state.comments_to_analyze = fetched_comments
                        
                        st.success(f"**{len(st.session_state.comments_to_analyze)}** adet {time_range} yorumu başarıyla çekildi!")
                    else:
                        loading_placeholder.empty()
                        st.info(f"{time_range} kriterine uygun yorum bulunamadı.")
                except Exception as e:
                    loading_placeholder.empty()
                    st.error(f"Yorumlar çekilirken bir hata oluştu: {e}")
        
with tab2:
    uploaded_files = st.file_uploader("CSV veya Excel dosyaları yükleyin", type=["csv", "xlsx"], accept_multiple_files=True)
    if uploaded_files:
        # Clear old state when new files are uploaded
        if "bulk_results" in st.session_state:
            del st.session_state.bulk_results
        
        all_comments: List[Dict[str, Any]] = []
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
                    st.markdown(f"#### Dosya: {uploaded_file.name}")
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
                                Dosya okundu: {len(df_upload)} satır
                            </div>
                            <div style="font-size: 0.9rem; font-weight: 600; color: #475569;">
                                Otomatik Seçilen Sütun: <span class="column-badge">{col_name}</span>
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
                st.error(f"{uploaded_file.name} okuma hatası: {e}")
        
        if all_comments:
            if len(all_comments) > 500:
                sliced_comments = []
                for idx in range(500):
                    sliced_comments.append(all_comments[idx])
                st.warning(f"Dosyadaki ilk 500 yorum analize alınmıştır (Toplam: {len(all_comments)} satır).")
                st.session_state.comments_to_analyze = sliced_comments
            else:
                st.session_state.comments_to_analyze = all_comments
            st.success(f"Toplam **{len(st.session_state.comments_to_analyze)}** gerçek yorum analiz için hazır!")

with tab3:
    text_input = st.text_area(
        "Yorumları alt alta girin:",
        height=200,
        placeholder="Örn: Harika uygulama!\nKötü performans...",
        key="manual_text_input"
    )
    if text_input.strip():
        # Clear old analysis when manual text is changed
        if "bulk_results" in st.session_state:
            del st.session_state.bulk_results
            
        raw_lines = text_input.split('\n')
        processed_comments: List[Dict[str, Any]] = []
        
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
                if processed_comments:
                    # Using indexing safely after checking truthiness
                    last_idx = len(processed_comments) - 1
                    if len(str(processed_comments[last_idx].get("text", ""))) < 85:
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
            if len(processed_comments) > 500:
                st.warning("En fazla 500 adet yorum girilebilir. Fazlası kırpıldı.")
                processed_final = []
                for idx in range(500):
                    processed_final.append(processed_comments[idx])
                st.session_state.comments_to_analyze = processed_final
            else:
                st.session_state.comments_to_analyze = processed_comments
            st.success(f"Toplam **{len(st.session_state.comments_to_analyze)}** geçerli satır eklendi!")

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
        format_func=lambda x: ["Hızlı", "Yavaş (Daha Tutarlı)"][x],
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
"احبه" → {{"olumlu":0.95,"olumsuz":0.02,"istek_gorus":0.03}}
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
                return {"_error": f"Gemini API hatası: {err_str[:120]}"}
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



# Analysis Logic Wrapper
def run_bulk_analysis(data_to_process, is_append=False):
    bulk_results = st.session_state.get("bulk_results", []) if is_append else []
    
    time_display = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    ticker_placeholder = st.empty() 
    quota_info = st.empty()
    st.warning("Analiz süresince bu sayfayı kapatmayın veya yenilemeyin. Verileriniz kaybolabilir.")
    st.session_state['_quota_hits'] = 0
            
    mode_idx = st.session_state.get("analysis_mode", 0)
    if mode_idx == 0:
        ANALYSIS_MODEL = 'gemini-2.5-flash'
        RPM_LIMIT = 500
    else:
        ANALYSIS_MODEL = 'gemini-2.5-pro'
        RPM_LIMIT = 300

    start_time = time.time()
    total_items = len(data_to_process)
    est_total_secs = total_items * (1 if mode_idx == 0 else 2)

    components.html(f"""
    <script>
    (function() {{
        var totalSecs = {est_total_secs};
        window.parent.onbeforeunload = function(e) {{
            var msg = 'Analiz henüz tamamlanmadı! Çıkarsanız verileriniz kaybolacak!';
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
        time_display.markdown(f"**Geçen süre:** {el_str} &nbsp;&nbsp;&nbsp; **Tahmini kalan:** {rem_str}")

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
    
    start_offset = len(bulk_results)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = [executor.submit(fetch_sentiment_worker, (i, e)) for i, e in enumerate(data_to_process)]
        
        for future in concurrent.futures.as_completed(tasks):
            i, entry, res, verdict, err = future.result()
            completed_count += 1
            
            status_text.text(f"Analiz ediliyor: {completed_count} / {total_items}")
            update_time(completed_count - 1, total_items, start_time)
            
            if err == "quota":
                q = st.session_state.get('_quota_hits', 0) + 1
                st.session_state['_quota_hits'] = q
            elif err:
                st.warning(err)
            
            comment = entry["text"]
            date = entry.get("date")
            ticker_date = f"{date.strftime('%d-%m-%Y')}" if date else ""

            ticker_color = "#34D399" if verdict == "Olumlu" else ("#F87171" if verdict == "Olumsuz" else "#60A5FA")
            ticker_placeholder.markdown(f"""
            <div class="header-container" style="border-color: {ticker_color}; text-align: left; margin: 10px 0; width: 100%; box-sizing: border-box;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85em; color: #64748b; margin-bottom: 8px;">
                    <span style="font-weight: 600;">ŞU AN EKLENEN (#{start_offset + i + 1})</span>
                    <span>{ticker_date}</span>
                </div>
                <div style="font-weight: 600; color: #1E293B; line-height: 1.6; font-size: 1.1rem;">
                    {comment[:350]}{'...' if len(comment)>350 else ''}
                </div>
                <div style="margin-top: 15px; display: inline-block; padding: 4px 12px; border-radius: 8px; background: {ticker_color}; color: white; font-size: 0.85em; font-weight: 700;">
                    {verdict.upper()}
                </div>
            </div>
            """, unsafe_allow_html=True)

            bulk_results.append({
                "No": start_offset + i + 1, "Yorum": comment, "Baskın Duygu": verdict,
                "Olumlu %": f"{res['olumlu']:.2%}", "İstek/Görüş %": f"{res['istek_gorus']:.2%}", "Olumsuz %": f"{res['olumsuz']:.2%}",
                "Tarih": date, "Puan": entry.get('rating')
            })
            progress_bar.progress(completed_count / total_items)

    st.session_state.bulk_results = sorted(bulk_results, key=lambda x: x["No"])
    status_text.success("Analiz Başarıyla Tamamlandı!")
    components.html("<script>window.parent.onbeforeunload = null;</script>", height=0)
    st.rerun()

if st.button("Analizini Yap", type="primary", use_container_width=True):
    if not comments_to_analyze:
        st.warning("Lütfen analiz edilecek bir metin girin veya dosya yükleyin.")
    else:
        run_bulk_analysis(comments_to_analyze)

# --- Persistent Results Display ---
if "bulk_results" in st.session_state:
    df = pd.DataFrame(st.session_state.bulk_results)
    counts = df["Baskın Duygu"].value_counts()
    
    st.markdown("""
<style>
/* Results Card Styling */
.neon-pos { border: 2px solid #34D399 !important; padding: 15px; border-radius: 12px; margin: 10px 0; background: #FFFFFF !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
.neon-neg { border: 2px solid #F87171 !important; padding: 15px; border-radius: 12px; margin: 10px 0; background: #FFFFFF !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
.neon-neu { border: 2px solid #60A5FA !important; padding: 15px; border-radius: 12px; margin: 10px 0; background: #FFFFFF !important; box-shadow: 0 4px 6px rgba(0,0,0,0.05); }
.normal-card { border: 1px solid #E2E8F0 !important; padding: 15px; border-radius: 12px; margin: 10px 0; background: #FFFFFF !important; }

.neon-pos *, .neon-neg *, .neon-neu *, .normal-card * {
    color: #000000 !important;
}

.metric-container {
    display: flex;
    justify-content: space-around;
    gap: 1rem;
    margin-bottom: 1.25rem;
    flex-wrap: wrap;
}
.metric-card {
    background: #FFFFFF !important;
    border: 2px solid #FFE4D6 !important;
    border-radius: 100px !important;
    padding: 1.5rem 1rem !important;
    text-align: center;
    flex: 1;
    min-width: 150px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.03) !important;
}
.metric-value { font-size: 2.5em; font-weight: bold; line-height: 1.2; }
.metric-label { font-size: 0.9em; color: #64748b !important; margin-top: 0.3rem; }

.glass-card {
    background: #FFFFFF !important;
    border: 2px solid #F1F5F9 !important;
    border-radius: 15px;
    padding: 15px;
    margin-bottom: 15px;
    color: #000000 !important;
}

.glass-card * {
    color: #000000 !important;
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
        # Calculate summary counts
        total_all = m_olumlu + m_olumsuz + m_istek
        diff_val = abs(m_olumlu - m_olumsuz)
        
        # Determine Colors and Summary Text based on dominant sentiment
        if total_all == 0:
            grad_bg = "#F8FAFC"
            border_c = "#E2E8F0"
            summary_title = "Henüz yeterli veri yok."
            summary_body = "Analiz edilecek yorumlar geldikçe burası güncellenecektir."
        elif total_all > 10 and (diff_val / total_all) < 0.15 and m_olumlu > 0 and m_olumsuz > 0: # Balanced/Mixed case
            grad_bg = "#fef9c3" # Light Yellow
            border_c = "#eab308" # Amber
            summary_title = f"Dengeli/Karmaşık bir kullanıcı deneyimi gözlendi. ({total_all} yorum)"
            summary_body = "Uygulama şu anda kullanıcı kitlesini neredeyse tam ortadan ikiye bölmüş durumda. Bir grup kullanıcı sunulan hizmetten, hızdan ve arayüzden son derece memnunken; diğer bir önemli grup ise teknik aksaklıklar, bağlantı sorunları veya beklenen özelliklerin eksikliği gibi konularda ciddi eleştiriler dile getiriyor. Marka imajı şu anda bir 'kritik eşik' evresinde; olumlu taraftaki kullanıcılar sadık kalmaya meyilliyken, olumsuz taraftakiler ise her an rakiplere yönelebilir. Bu bıçak sırtı dengeden kurtulmak için en acil şikayetlere (bug'lar, performans sorunları vb.) odaklanılmalı ve bu kitle hızlıca memnun edilmeli. Eğer bu karmaşık tablo doğru yönetilirse kitle olumlu yöne çekilebilir, aksi takdirde olumsuz sesler baskın hale gelecektir."
        elif counts.idxmax() == "Olumlu":
            grad_bg = "#dcfce7" # Light Green
            border_c = "#10b981" # Emerald
            summary_title = f"Topluluk genel olarak Olumlu bir tavır sergiliyor. ({m_olumlu} yorum)"
            summary_body = "Genel olarak kullanıcı kitlesi, uygulamanın sunduğu temel hizmetlerden, arayüz tasarımından ve kullanım kolaylığından yüksek düzeyde memnuniyet duyuyor diyebiliriz. Özellikle düzenli kullanıcılar uygulamanın günlük hayattaki işlevselliğini olumlu bularak tavsiye etme eğiliminde. Sistem performansı, hız ve güvenilirlik beklentileri büyük ölçüde karşılanıyor. Son güncellemelerle birlikte gelen yenilikler pozitif karşılanmış gibi görünüyor. Kullanıcıların markaya olan güveni bu aşamada sağlam temeller üzerinde duruyor. Müşteri hizmetlerinin ve destek birimlerinin sorunlara hızlı reaksiyon göstermesi de bu olumlu havayı destekleyen ana etkenlerden biri olabilir. Yine de aralardaki küçük oranlı şikayetleri dikkatle ele alıp, bu %100'e yakın memnuniyet oranını koruyacak stratejik adımların devam ettirilmesi oldukça önemli."
        elif counts.idxmax() == "Olumsuz":
            grad_bg = "#fee2e2" # Light Red
            border_c = "#f43f5e" # Rose
            summary_title = f"Dikkat çeken Olumsuz bir eğilim var. ({m_olumsuz} yorum)"
            summary_body = "Analiz edilen veri setinde kullanıcıların çok ciddi hayal kırıklıkları ve sistemsel şikayetleri olduğu açıkça görülmektedir. Özellikle kilitlenme, yavaşlık veya beklenen özelliklerin çalışmaması gibi kronikleşmiş teknik problemler kullanıcı deneyimini ciddi oranda baltalıyor. İade sorunları, müşteri hizmetlerinin ulaşılamaz olması veya vaat edilenle karşılaşılan hizmetin uyuşmaması gibi temel şikayetler marka imajına an itibariyle zarar veriyor. Kullanıcılar uygulamanın temel fonksiyonlarını bile kullanırken pürüzlerle karşılaştıkları için platformu terk etme veya rakiplere yönelme potansiyeline sahipler. Acil ve agresif bir hata ayıklama (bug-fixing) sürecine gidilmeli, müşteri destek hattının kapasitesi artırılmalı ve kullanıcılardan gelen yapısal eleştiriler bir an önce yazılım geliştirme döngüsüne entegre edilmelidir."
        else: # Neutral dominant
            grad_bg = "#dbeafe" # Light Blue
            border_c = "#3b82f6" # Blue
            summary_title = f"Kullanıcılar yoğun şekilde İstek ve Görüş paylaşıyor. ({m_istek} yorum)"
            summary_body = "Kullanıcı tabanı şu anda markaya veya uygulamaya karşı keskin bir öfke yahut aşırı bir coşku beslemek yerine, daha akılcı ve beklenti odaklı bir tutum içinde. Yorumların geneli, sistemin temel ihtiyaçları karşıladığını ancak modern standartlara veya rakiplere kıyasla eksik bazı ufak tefek özellikler veya yaşam kalitesi (QoL) güncellemeleri barındırdığına işaret ediyor. Kullanıcılar aslında uygulamanın potansiyelinin farkında ve bu potansiyeli maksimize edecek yenilikler (örneğin karanlık mod, daha geniş dil desteği, pratik menü tasarımları vb.) görmek istiyorlar. Bu grup sadık bir kitleye dönüşmeye oldukça yakın; geliştirici ekip eğer bu geri bildirimleri dikkate alıp istenen özellikleri sisteme entegre ederse, tarafsız duran bu kitle çok hızlı bir şekilde savunucu ve sadık kullanıcılara (olumlu) evrilecektir."
        
        # Save to session state for the report card
        st.session_state.ai_summary = summary_body

        st.markdown(f"""
        <div style="background: {grad_bg}; padding: 20px; border-radius: 12px; border: 2px solid {border_c}; color: #1e293b; line-height: 1.6;">
            <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 10px;">{summary_title}</div>
            <div style="font-size: 0.95rem; opacity: 0.9;">{summary_body}</div>
        </div>
        """, unsafe_allow_html=True)
        

    # NEW: Star Rating Distribution Chart (Sütunlu ve Renkli)
    if "Puan" in df.columns and df["Puan"].notnull().any():
        st.markdown("---")
        
        # UI for Frequency Selection
        st.write("#### Puan Dağılımı Trendi")
        freq = st.radio("Zaman Ölçeği:", ["Günlük", "Haftalık", "Aylık"], index=2, horizontal=True, key="puan_freq_sel", label_visibility="collapsed")
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

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
                st.caption(f"**Tespit Edilen Tarih Aralığı:** {min_d} ile {max_d}")

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
    
    # --- Extended Analysis Prompt ---
    all_pool = st.session_state.get("all_fetched_pool", [])
    if all_pool:
        # Indices of comments already analyzed
        analyzed_count = len(df)
        remaining_pool = all_pool[analyzed_count:]
        
        if remaining_pool:
            st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
            with st.container(border=True):
                st.markdown(f"""
                <div style="background-color: #F0F9FF; margin: -1rem; padding: 1.5rem; border-radius: 10px; margin-bottom: 1rem;">
                    <div style="color: #0369a1; font-weight: 600; text-align: center;">
                        Havuzda henüz analiz edilmemiş {len(remaining_pool)} yorum daha var.
                    </div>
                </div>
                """, unsafe_allow_html=True)
                take_next = min(len(remaining_pool), 500)
                if st.button(f"Sonraki {take_next} yorumu da analiz et ve sonuçlara ekle", use_container_width=True):
                    next_batch = remaining_pool[:take_next]
                    run_bulk_analysis(next_batch, is_append=True)

    # Chart & List Logic
    def render_trend_chart(filtered_df, key, title_suffix="", freq="Haftalık"):
        df_dates = filtered_df.dropna(subset=["Tarih"]).copy()
        if not df_dates.empty:
            df_dates["Tarih"] = pd.to_datetime(df_dates["Tarih"])
            
            if freq == "Haftalık":
                df_dates['Grup'] = df_dates['Tarih'].dt.to_period('W').apply(lambda r: r.start_time)
                xaxis_title = "Tarih (Haftalık)"
                chart_title_prefix = "Haftalık"
            elif freq == "Aylık":
                df_dates['Grup'] = df_dates['Tarih'].dt.to_period('M').apply(lambda r: r.start_time)
                xaxis_title = "Tarih (Aylık)"
                chart_title_prefix = "Aylık"
            else: # Günlük
                df_dates['Grup'] = df_dates['Tarih'].dt.date
                xaxis_title = "Tarih (Günlük)"
                chart_title_prefix = "Günlük"

            trend_data = df_dates.groupby(['Grup', "Baskın Duygu"]).size().reset_index(name='Adet')
            
            # Custom data for robust selection processing (includes exact Grup and Sentiment)
            trend_data['Grup_str'] = trend_data['Grup'].astype(str)
            fig_trend = px.bar(trend_data, x="Grup", y="Adet", color="Baskın Duygu",
                               title=f"{chart_title_prefix} Duygu Dağılımı {title_suffix}",
                               color_discrete_map={'Olumlu':'#2ecc71', 'Olumsuz':'#e74c3c', 'İstek/Görüş':'#3498db'},
                               barmode='group',
                               labels={"Baskın Duygu": ""},
                               custom_data=["Grup_str", "Baskın Duygu"])
            
            fig_trend.update_layout(height=400, margin={"t": 80, "b": 40, "l": 10, "r": 10},
                                   legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1, "font": {"color": "#000000"}},
                                   xaxis_title=xaxis_title, yaxis_title="Yorum Sayısı",
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
                # Use Grup directly from custom_data for 100% precision
                sel_grup_str = point["customdata"][0]
                sel_grup = pd.to_datetime(sel_grup_str).tz_localize(None)
                sel_sentiment = str(point["customdata"][1]).strip()
                
                # Standardize database groups for comparison
                df_dates['Grup_compare'] = pd.to_datetime(df_dates['Grup']).dt.tz_localize(None)
                
                final_filtered = df_dates[
                    (df_dates['Grup_compare'] == sel_grup) & 
                    (df_dates['Baskın Duygu'] == sel_sentiment)
                ]
                
                st.info(f"Filtrelendi: **{sel_grup.strftime('%d.%m.%Y')}** periyodu - **{sel_sentiment}** yorumlar")
                if st.button("Filtreyi Temizle", key=f"clear_{key}"):
                    st.rerun()
                return final_filtered
            
        return filtered_df

    def display_comments(filtered_df, tab_id, highlight=True):
        if filtered_df.empty:
            st.info("Bu kategoride henüz yorum bulunmuyor.")
            return
            
        show_all_key = f"show_all_{tab_id}"
        page_key = f"page_{tab_id}"
        
        if show_all_key not in st.session_state:
            st.session_state[show_all_key] = False
        if page_key not in st.session_state:
            st.session_state[page_key] = 1
            
        show_all = st.session_state[show_all_key]
        total_items = len(filtered_df)
        
        if not show_all:
            render_df = filtered_df.head(3)
        else:
            if total_items > 250:
                current_page = st.session_state[page_key]
                total_pages = (total_items - 1) // 250 + 1
                if current_page > total_pages: current_page = total_pages
                if current_page < 1: current_page = 1
                
                start_idx = (current_page - 1) * 250
                render_df = filtered_df.iloc[start_idx : start_idx + 250]
            else:
                render_df = filtered_df

        for _, row in render_df.iterrows():
            sentiment = row["Baskın Duygu"]
            cls = "normal-card"
            if highlight:
                cls = "neon-pos" if sentiment == "Olumlu" else ("neon-neg" if sentiment == "Olumsuz" else "neon-neu")
            
            # Format extra info (Rating)
            extra_info = ""
            if "Puan" in row and pd.notnull(row["Puan"]):
                extra_info += f" | Puan: {row['Puan']}"
            
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
                <div style="color: #000000; line-height: 1.5;">{row['Yorum']}</div>
            </div>
            """, unsafe_allow_html=True)

        if total_items > 3:
            if not show_all:
                if st.button("Tüm Yorumları Göster", key=f"btn_show_{tab_id}", use_container_width=True):
                    st.session_state[show_all_key] = True
                    st.rerun()
            else:
                if total_items > 250:
                    total_pages = (total_items - 1) // 250 + 1
                    current_page = st.session_state[page_key]
                    
                    nav_cols = st.columns([1, 2, 1])
                    with nav_cols[0]:
                        if st.button("⬅️ Önceki Sayfa", key=f"prev_{tab_id}", use_container_width=True, disabled=(current_page == 1)):
                            st.session_state[page_key] -= 1
                            st.rerun()
                    with nav_cols[1]:
                        st.markdown(f"<div style='text-align: center; margin-top: 10px; font-weight: bold; color: #64748B;'>Sayfa {current_page} / {total_pages}</div>", unsafe_allow_html=True)
                    with nav_cols[2]:
                        if st.button("Sonraki Sayfa ➡️", key=f"next_{tab_id}", use_container_width=True, disabled=(current_page == total_pages)):
                            st.session_state[page_key] += 1
                            st.rerun()
                            
                    st.markdown("<br>", unsafe_allow_html=True)

                if st.button("Daha Az Göster", key=f"btn_hide_{tab_id}", use_container_width=True):
                    st.session_state[show_all_key] = False
                    st.session_state[page_key] = 1
                    st.rerun()

    # --- Tabs and Unified Display ---
    st.write("### Yorum Listesi")
    
    # Frequency Selector for Trend Chart
    yorum_freq = st.radio("Zaman Ölçeği:", ["Günlük", "Haftalık", "Aylık"], index=1, horizontal=True, key="yorum_freq_sel", label_visibility="collapsed")
    st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

    t_pos = counts.get('Olumlu', 0)
    t_neg = counts.get('Olumsuz', 0)
    t_neu = counts.get('İstek/Görüş', 0)
    t_all = len(analysis_df)

    tab_all, tab_pos, tab_neg, tab_neu = st.tabs([
        f"Analizler ({t_all})", 
        f"Olumlu ({t_pos})", 
        f"Olumsuz ({t_neg})", 
        f"İstek/Görüş ({t_neu})"
    ])

    with tab_all:
        f_df = render_trend_chart(analysis_df, "all", "(Genel)", freq=yorum_freq)
        display_comments(f_df, "all", highlight=False)
    
    with tab_pos:
        pos_df = df[df["Baskın Duygu"] == "Olumlu"]
        f_df = render_trend_chart(pos_df, "pos", "(Olumlu)", freq=yorum_freq)
        display_comments(f_df, "pos")
        
    with tab_neg:
        neg_df = df[df["Baskın Duygu"] == "Olumsuz"]
        f_df = render_trend_chart(neg_df, "neg", "(Olumsuz)", freq=yorum_freq)
        display_comments(f_df, "neg")
        
    with tab_neu:
        neu_df = df[df["Baskın Duygu"] == "İstek/Görüş"]
        f_df = render_trend_chart(neu_df, "neu", "(İstek/Görüş)", freq=yorum_freq)
        display_comments(f_df, "neu")

    # Excel Download
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Analiz Sonuçları')
        
        # --- SHARE SECTION ---
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.subheader("Analiz Raporunu Paylaş")
        
        # --- PIE CHART & STATS PREPARATION ---
        total_q = len(df)
        pos_p = int((t_pos / total_q) * 100) if total_q > 0 else 0
        neg_p = int((t_neg / total_q) * 100) if total_q > 0 else 0
        neu_p = int((t_neu / total_q) * 100) if total_q > 0 else 0
        
        import math
        def get_svg_path(start_pct, end_pct):
            if end_pct - start_pct >= 99.9: 
                return "M 0 -1 A 1 1 0 1 1 -0.001 -1 Z"
            start_rad = math.radians(start_pct * 3.6 - 90)
            end_rad = math.radians(end_pct * 3.6 - 90)
            x1, y1 = math.cos(start_rad), math.sin(start_rad)
            x2, y2 = math.cos(end_rad), math.sin(end_rad)
            large_arc = 1 if (end_pct - start_pct) > 50 else 0
            return f"M 0 0 L {x1:.3f} {y1:.3f} A 1 1 0 {large_arc} 1 {x2:.3f} {y2:.3f} Z"

        p_path = get_svg_path(0, pos_p)
        n_path = get_svg_path(pos_p, pos_p + neg_p)
        u_path = get_svg_path(pos_p + neg_p, 100)

        # Priority: Use detected names
        app_name = urllib.parse.unquote(st.session_state.get('detected_app_name', "Uygulama"))
        store_type = st.session_state.get('detected_store_type', "STORE")
        report_title = f"{app_name.upper()} {store_type.upper()} ANALİZ RAPORU"

        # Text generation for sharing (Robust newlines)
        summary_text = (
            f"{app_name} Analiz Raporu (v3.0 Parallel Engine)\n"
            f"━━━━━━━━━━━━━━━━━━━━━\n"
            f"Toplam Veri: {total_q} Yorum\n"
            f"Olumlu Deneyim: %{pos_p}\n"
            f"Olumsuz Deneyim: %{neg_p}\n"
        )
        if st.session_state.get('ai_summary'):
            summary_text += f"Stratejik Tespit: {st.session_state.ai_summary[:150]}...\n"
        summary_text += "━━━━━━━━━━━━━━━━━━━━━\n"
        summary_text += f"Analizini yap: https://sentimentanalysis-aimode.streamlit.app/\n"
        summary_text += "#ivicin"
        
        encoded_text = urllib.parse.quote(summary_text)

        # --- DIGITAL REPORT CARD (SVG 3D) ---
        def clean_html(h):
            return "\n".join([line.strip() for line in h.split('\n') if line.strip()])
        
        # Sanitize summary for card display (remove formatting that might break HTML)
        display_summary = st.session_state.get('ai_summary', 'Analiz özeti hazırlanıyor...')
        display_summary = display_summary.replace("`", "").replace("*", "").replace("#", "")

        card_html = clean_html(f"""
            <div id="nlp-report-card" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 20px; padding: 35px; margin: 20px auto; box-shadow: 0 15px 35px rgba(0,0,0,0.08); font-family: 'Poppins', sans-serif; color: #1E293B; max-width: 600px; position: relative; overflow: hidden;">
                <div style="text-align: center; border-bottom: 2px solid #F1F5F9; padding-bottom: 15px; margin-bottom: 25px;">
                    <h2 style="margin: 0; color: #0F172A; font-size: 1.3rem; font-weight: 700;">{report_title}</h2>
                </div>
                
                <div style="display: flex; justify-content: space-between; margin-bottom: 35px; gap: 10px;">
                    <div style="text-align: center; flex: 1; background: #F8FAFC; padding: 12px; border-radius: 12px;">
                        <div style="font-size: 0.65rem; color: #64748B; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Analiz</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #334155;">{total_q}</div>
                    </div>
                    <div style="text-align: center; flex: 1; background: #ECFDF5; padding: 12px; border-radius: 12px; border: 1px solid #D1FAE5;">
                        <div style="font-size: 0.65rem; color: #059669; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Olumlu</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #059669;">{t_pos}</div>
                    </div>
                    <div style="text-align: center; flex: 1; background: #FEF2F2; padding: 12px; border-radius: 12px; border: 1px solid #FEE2E2;">
                        <div style="font-size: 0.65rem; color: #DC2626; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Olumsuz</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #DC2626;">{t_neg}</div>
                    </div>
                    <div style="text-align: center; flex: 1; background: #EFF6FF; padding: 12px; border-radius: 12px; border: 1px solid #DBEAFE;">
                        <div style="font-size: 0.65rem; color: #2563EB; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Görüş</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #2563EB;">{t_neu}</div>
                    </div>
                </div>

                <div style="display: flex; align-items: center; justify-content: space-around; background: #F8FAFC; border-radius: 20px; padding: 30px; margin-bottom: 25px;">
                    <div style="width: 140px; height: 140px; position: relative;">
                        <div style="position: absolute; width: 130px; height: 130px; background: #CBD5E1; border-radius: 50%; top: 10px; transform: scaleY(0.6);"></div>
                        <svg width="130" height="130" viewBox="-1.1 -1.1 2.2 2.2" style="position: absolute; top: 0; transform: scaleY(0.6); filter: drop-shadow(0 2px 4px rgba(0,0,0,0.1)); overflow: visible;">
                            <path d="{p_path}" fill="#10B981" stroke="#FFFFFF" stroke-width="0.02" />
                            <path d="{n_path}" fill="#EF4444" stroke="#FFFFFF" stroke-width="0.02" />
                            <path d="{u_path}" fill="#3B82F6" stroke="#FFFFFF" stroke-width="0.02" />
                        </svg>
                    </div>
                    <div style="display: flex; flex-direction: column; gap: 10px;">
                        <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 12px; height: 12px; background: #10B981; border-radius: 3px;"></div><span style="font-size: 0.85rem; font-weight: 600;">Olumlu %{pos_p}</span></div>
                        <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 12px; height: 12px; background: #EF4444; border-radius: 3px;"></div><span style="font-size: 0.85rem; font-weight: 600;">Olumsuz %{neg_p}</span></div>
                        <div style="display: flex; align-items: center; gap: 8px;"><div style="width: 12px; height: 12px; background: #3B82F6; border-radius: 3px;"></div><span style="font-size: 0.85rem; font-weight: 600;">Görüş %{neu_p}</span></div>
                    </div>
                </div>

                <div style="background: #FFFFFF; border-radius: 16px; padding: 20px; border: 1px solid #F1F5F9; border-left: 6px solid #6366F1; box-shadow: 0 4px 12px rgba(0,0,0,0.03);">
                    <div style="font-weight: 800; color: #1E293B; margin-bottom: 10px; font-size: 0.95rem; display: flex; align-items: center; gap: 8px;">
                        <span style="font-size: 1.2rem;">💡</span> Stratejik Özet
                    </div>
                    <div style="color: #475569; font-size: 0.9rem; line-height: 1.6; font-weight: 500;">
                        {display_summary}
                    </div>
                </div>
                <div style="margin-top: 30px; text-align: center; color: #94A3B8; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;">
                    📊 AI SENTIMENT INTELLIGENCE
                </div>
            </div>
        """)
        st.markdown(card_html, unsafe_allow_html=True)
        st.info("💡 Yukarıdaki kartı kopyalayabilir veya doğrudan paylaşabilirsiniz.")

        # --- GLOBAL UTILITY SCRIPTS (Parent Frame) ---
        import textwrap
        import json
        global_scripts = textwrap.dedent(f"""
            <div id="trayNotif" style="position: fixed; top: 20px; left: 50%; transform: translateX(-50%) translateY(-20px) scale(0.9); background: #10B981; color: white; padding: 14px 28px; border-radius: 12px; font-weight: 700; opacity: 0; transition: all 0.4s cubic-bezier(0.19, 1, 0.22, 1); z-index: 9999999; box-shadow: 0 15px 30px rgba(16, 185, 129, 0.4); display: flex; align-items: center; gap: 10px; pointer-events: none; font-family: 'Poppins', sans-serif; white-space: nowrap;">
                <span id="trayMsg">Kopyalandı!</span>
            </div>
            <script>
                window.pushNotif = function(msg) {{
                    const n = document.getElementById('trayNotif');
                    const m = document.getElementById('trayMsg');
                    if(!n || !m) return;
                    m.innerText = msg;
                    n.style.opacity = '1';
                    n.style.transform = 'translateX(-50%) translateY(0) scale(1)';
                    setTimeout(() => {{ 
                        n.style.opacity = '0';
                        n.style.transform = 'translateX(-50%) translateY(-20px) scale(0.9)';
                    }}, 3500);
                }};

                window.doCopyText = function(txt, plat) {{
                    navigator.clipboard.writeText(txt).then(() => {{
                        if(plat === 'Google Chat') {{
                            window.open('https://chat.google.com', '_blank');
                            window.pushNotif("Google Chat Açılıyor & Metin Kopyalandı! ✅");
                        }} else {{
                            window.pushNotif(plat + " Metni Kopyalandı! ✅");
                        }}
                    }}).catch(() => {{
                        const el = document.createElement('textarea');
                        el.value = txt; document.body.appendChild(el); el.select();
                        document.execCommand('copy'); document.body.removeChild(el);
                        if(plat === 'Google Chat') {{
                            window.open('https://chat.google.com', '_blank');
                            window.pushNotif("Google Chat Açılıyor & Metin Kopyalandı! ✅");
                        }} else {{
                            window.pushNotif(plat + " Metni Kopyalandı! ✅");
                        }}
                    }});
                }};

                window.doCopyCard = function() {{
                    const target = document.getElementById('nlp-report-card');
                    if(!target) return;
                    const h2c = window.html2canvas || (window.parent && window.parent.html2canvas);
                    if(!h2c) {{ window.pushNotif("Sistem Hazırlanıyor... ⏳"); return; }}
                    window.pushNotif("Görsel Hazırlanıyor... ⏳");
                    h2c(target, {{ scale: 2, useCORS: true, backgroundColor: '#FFFFFF', logging: false }}).then(canvas => {{
                        canvas.toBlob(blob => {{
                            try {{
                                const data = [new ClipboardItem({{ [blob.type]: blob }})];
                                navigator.clipboard.write(data).then(() => {{
                                    window.pushNotif("Görsel Kopyalandı! ✅");
                                }}).catch(() => {{ throw new Error(); }});
                            }} catch(e) {{
                                const url = canvas.toDataURL();
                                const link = document.createElement('a');
                                link.download = 'nlp-report-card.png';
                                link.href = url; link.click();
                                window.pushNotif("İndirme Başlatıldı ⬇️");
                            }}
                        }}, 'image/png');
                    }});
                }};

                window.doSocialImageShare = function(platUrl, platName) {{
                    const target = document.getElementById('nlp-report-card');
                    if(!target) return;
                    const h2c = window.html2canvas || (window.parent && window.parent.html2canvas);
                    if(!h2c) {{ window.pushNotif("Sistem Hazırlanıyor... ⏳"); window.open(platUrl, '_blank'); return; }}
                    window.pushNotif("Kart Kopyalanıyor & " + platName + " Açılıyor... ⏳");
                    h2c(target, {{ scale: 2, useCORS: true, backgroundColor: '#FFFFFF', logging: false }}).then(canvas => {{
                        canvas.toBlob(blob => {{
                            try {{
                                const data = [new ClipboardItem({{ [blob.type]: blob }})];
                                navigator.clipboard.write(data).then(() => {{
                                    window.open(platUrl, '_blank');
                                    window.pushNotif("Görsel Panoda! ✅ " + platName + "'da Yapıştırabilirsiniz (Ctrl+V)");
                                }}).catch(() => {{ throw new Error(); }});
                            }} catch(e) {{
                                window.open(platUrl, '_blank');
                                window.pushNotif(platName + " Açıldı. Lütfen Görseli Manuel Kopyalayın.");
                            }}
                        }}, 'image/png');
                    }});
                }};

                // --- ROBUST MESSAGE LISTENER ---
                window.addEventListener('message', function(event) {{
                    const data = event.data;
                    if (!data || !data.type) return;
                    
                    if (data.type === 'copyText') {{
                        window.doCopyText(data.text, data.platform);
                    }} else if (data.type === 'copyCard') {{
                        window.doCopyCard();
                    }} else if (data.type === 'socialShare') {{
                        window.doSocialImageShare(data.url, data.platform);
                    }}
                }});
            </script>
        """).strip()
        st.markdown(global_scripts, unsafe_allow_html=True)

        # --- PREMIUM SHARE TRAY (Iframe Component) ---
        import streamlit.components.v1 as components
        summary_escaped = json.dumps(summary_text)
        
        tray_html = textwrap.dedent(f"""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
            <style>
                body {{ margin: 0; padding: 0; display: flex; align-items: center; justify-content: center; overflow: hidden; font-family: sans-serif; background: transparent; }}
                .share-tray {{ display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; }}
                .share-btn {{
                    width: 48px; height: 48px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
                    display: flex; align-items: center; justify-content: center; font-size: 1.4rem; cursor: pointer;
                    transition: all 0.2s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-decoration: none !important;
                }}
                .share-btn:hover {{ transform: translateY(-3px); box-shadow: 0 8px 15px rgba(0,0,0,0.1); border-color: #CBD5E1; }}
                .btn-wa {{ color: #25D366; }} .btn-li {{ color: #0077B5; }} .btn-x {{ color: #000000; }}
                .btn-tg {{ color: #0088CC; }} .btn-fb {{ color: #1877F2; }} .btn-mail {{ color: #D44638; }}
                .btn-rd {{ color: #FF4500; }} .btn-sl {{ color: #4A154B; }} .btn-gc {{ color: #00897B; }}
                .btn-pic {{ color: #8B5CF6; }}
            </style>

            <div class="share-tray">
                <a href="https://api.whatsapp.com/send?text={encoded_text}" target="_blank" class="share-btn btn-wa"><i class="fa-brands fa-whatsapp"></i></a>
                <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://cem-evecen.com&summary={encoded_text}" target="_blank" class="share-btn btn-li"><i class="fa-brands fa-linkedin-in"></i></a>
                <a href="https://twitter.com/intent/tweet?text={encoded_text}" target="_blank" class="share-btn btn-x"><i class="fa-brands fa-x-twitter"></i></a>
                <a href="https://t.me/share/url?url=https://cem-evecen.com&text={encoded_text}" target="_blank" class="share-btn btn-tg"><i class="fa-brands fa-telegram"></i></a>
                <div id="btn-fb" class="share-btn btn-fb"><i class="fa-brands fa-facebook-f"></i></div>
                <a href="mailto:?subject=NLP Analiz Raporu&body={encoded_text}" class="share-btn btn-mail"><i class="fa-solid fa-envelope"></i></a>
                <a href="https://www.reddit.com/submit?title=NLP Raporu&text={encoded_text}" target="_blank" class="share-btn btn-rd"><i class="fa-brands fa-reddit-alien"></i></a>
                <a href="slack://share?text={encoded_text}" class="share-btn btn-sl"><i class="fa-brands fa-slack"></i></a>
                <div id="btn-gc" class="share-btn btn-gc"><i class="fa-solid fa-comment-dots"></i></div>
                <div id="btn-pic" class="share-btn btn-pic"><i class="fa-solid fa-camera"></i></div>
            </div>

            <script>
                const summaryTxt = {summary_escaped};
                const fbUrl = "https://www.facebook.com/sharer/sharer.php?u=https://cem-evecen.com&quote=" + encodeURIComponent(summaryTxt);
                
                function sendCmd(type, extra = {{}}) {{
                    window.parent.postMessage({{ type: type, ...extra }}, '*');
                }}

                document.getElementById('btn-gc').addEventListener('click', () => {{
                    sendCmd('copyText', {{ text: summaryTxt, platform: 'Google Chat' }});
                }});
                document.getElementById('btn-fb').addEventListener('click', () => {{
                    sendCmd('socialShare', {{ url: fbUrl, platform: 'Facebook' }});
                }});
                document.getElementById('btn-pic').addEventListener('click', () => {{
                    sendCmd('copyCard');
                }});
            </script>
        """).strip()
        components.html(tray_html, height=70)
        st.markdown("<br>", unsafe_allow_html=True)
        import streamlit.components.v1 as components
        components.html(f"""
        <style>body {{ margin: 0; padding: 0; overflow: hidden; display: flex; align-items: center; justify-content: center; font-family: sans-serif; }}</style>
        <button onclick="downloadPNG()" style='width: 100%; height: 50px; background: #6366F1; color: #FFFFFF; border: none; padding: 0; border-radius: 12px; cursor: pointer; font-size: 0.95rem; font-weight: 500; box-shadow: 0 4px 6px rgba(0,0,0,0.1); transition: transform 0.2s;'>
            📷 Kartı PNG Görseli Olarak İndir
        </button>
        <script>
            var pDoc = window.parent.document;
            if (!pDoc.getElementById('html2canvas-js')) {{
                var s = pDoc.createElement('script');
                s.id = 'html2canvas-js';
                s.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
                pDoc.head.appendChild(s);
            }}

            function downloadPNG() {{
                var target = pDoc.getElementById('nlp-report-card');
                if (target && window.parent.html2canvas) {{
                    var btn = document.querySelector('button');
                    var oldText = btn.innerText;
                    btn.innerText = "⏳ Hazırlanıyor...";
                    
                    window.parent.html2canvas(target, {{scale: 2, backgroundColor: '#FFFFFF', useCORS: true}}).then(canvas => {{
                        var link = pDoc.createElement('a');
                        link.download = '{f"{app_name} ai sentiment report.png".replace(" ", "_")}';
                        link.href = canvas.toDataURL("image/png");
                        link.click();
                        btn.innerText = oldText;
                    }}).catch(err => {{
                        alert("Görsel oluşturulurken hata oluştu.");
                        btn.innerText = oldText;
                    }});
                }} else {{
                    alert('Sistem hazırlanıyor... Lütfen 1-2 saniye bekleyip tekrar deneyin.');
                }}
            }}
        </script>
        """, height=53)
        
        # Actions Row: Excel and PDF triggers side-by-side
        st.markdown("<br>", unsafe_allow_html=True)
        action_cols = st.columns(2)
        
        with action_cols[0]:
            st.download_button(
                label="Sonuçları Excel Olarak İndir", 
                data=output.getvalue(), 
                file_name=f"{app_name} ai sentiment report.xlsx".replace(" ", "_"), 
                key="bulk_dl", 
                use_container_width=True
            )
            
        with action_cols[1]:
            import streamlit.components.v1 as components
            components.html("""
                <style>body { margin: 0; padding: 0; overflow: hidden; }</style>
                <div style='text-align: center; font-family: sans-serif; display: flex; align-items: center; justify-content: center; height: 50px;'>
                    <button onclick='window.parent.print()' style='width: 100%; height: 50px; background: #F4A261; color: #FFFFFF; border: none; padding: 0; border-radius: 12px; cursor: pointer; font-family: inherit; font-weight: 400; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-size: 0.95rem;'>
                        Raporu PDF Olarak İndir / Yazdır
                    </button>
                </div>
            """, height=50)

                    
    except Exception as e:
        st.error(f"Paylaşım aracı hazırlanırken hata: {e}")

# Footer
st.divider()
st.caption("Geliştiren: ivicin")

