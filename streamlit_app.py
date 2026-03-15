import streamlit as st
import streamlit.components.v1 as components
from streamlit_lottie import st_lottie
from google import genai
from google.genai import types as genai_types
try:
    from mistralai import Mistral
    HAS_MISTRAL_PKG = True
except ImportError:
    try:
        from mistralai.client import MistralClient as Mistral
        HAS_MISTRAL_PKG = True
    except ImportError:
        HAS_MISTRAL_PKG = False
        Mistral = None

try:
    from groq import Groq
    HAS_GROQ_PKG = True
except ImportError:
    HAS_GROQ_PKG = False
    Groq = None
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
from google_play_scraper import Sort, reviews as play_reviews, app as play_app

from dotenv import load_dotenv
import textwrap
if os.path.exists(".env"):
    load_dotenv(override=True)


st.set_page_config(
    page_title="AI Duygu Analizi",
    layout="centered"
)


@st.cache_resource(show_spinner="API yapılandırılıyor...")
def setup_api():
    
    keys_to_check = ["GOOGLE_API_KEY", "GEMINI_API_KEY", "API_KEY"]
    
    api_key = None
    
    
    for k in keys_to_check:
        val = os.getenv(k)
        if val and str(val).strip():
            api_key = str(val).strip()
            break
            
    
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
            
            client = genai.Client(api_key=api_key)
            return client
        except Exception as e:
            st.error(f"API Client başlatma hatası: {e}")
            return None
    return None

@st.cache_resource(show_spinner="Mistral API yapılandırılıyor...")
def setup_mistral():
    api_key = os.getenv("MISTRAL_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("MISTRAL_API_KEY")
        except:
            pass
    
    if api_key and Mistral is not None:
        try:
            client = Mistral(api_key=api_key)
            return client
        except Exception as e:
            st.error(f"Mistral API Client başlatma hatası: {e}")
            return None
    return None

@st.cache_resource(show_spinner="Groq API yapılandırılıyor...")
def setup_groq():
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("GROQ_API_KEY")
        except:
            pass
    
    if api_key and Groq is not None:
        try:
            client = Groq(api_key=api_key)
            return client
        except Exception as e:
            st.error(f"Groq API Client başlatma hatası: {e}")
            return None
    return None

GEMINI_CLIENT = setup_api()
HAS_GEMINI = GEMINI_CLIENT is not None

MISTRAL_CLIENT = setup_mistral()
HAS_MISTRAL = MISTRAL_CLIENT is not None

GROQ_CLIENT = setup_groq()
HAS_GROQ = GROQ_CLIENT is not None


API_TRACKER = {"cost_tl": 0.0}


if not HAS_GEMINI and not HAS_MISTRAL and "streamlit" in str(st.__file__).lower():
    st.sidebar.error("⚠️ AI API Key bulunamadı! Lütfen Streamlit Cloud 'Secrets' kısmına GOOGLE_API_KEY veya MISTRAL_API_KEY tanımlayın.")
    if st.sidebar.button("API'yi Yeniden Kontrol Et"):
        st.cache_resource.clear()
        st.rerun()
elif HAS_GEMINI and "GEMINI_CLIENT" in locals():
    
    pass

# Sidebar API Configuration
st.sidebar.title("🤖 AI Ayarları")
ai_provider = st.sidebar.selectbox(
    "AI Sağlayıcı:",
    options=["Google Gemini", "Mistral AI", "Groq AI"],
    index=0 if HAS_GEMINI else 1 if HAS_MISTRAL else 2 if HAS_GROQ else 0,
    key="ai_provider"
)

if ai_provider == "Google Gemini":
    if not HAS_GEMINI:
        st.sidebar.error("⚠️ Gemini API Key bulunamadı!")
    ai_model = st.sidebar.selectbox(
        "Model:",
        options=["gemini-2.0-flash-lite", "gemini-2.0-flash", "gemini-1.5-flash", "gemini-1.5-pro"],
        index=0,
        key="ai_model_gemini"
    )
    model_full_name = f"models/{ai_model}"
elif ai_provider == "Mistral AI":
    if not HAS_MISTRAL:
        st.sidebar.error("⚠️ Mistral API Key bulunamadı!")
    ai_model = st.sidebar.selectbox(
        "Model:",
        options=["mistral-tiny", "mistral-small-latest", "mistral-medium-latest", "mistral-large-latest", "open-mistral-7b", "open-mixtral-8x7b"],
        index=1,
        key="ai_model_mistral"
    )
    model_full_name = ai_model
else:
    if not HAS_GROQ:
        st.sidebar.error("⚠️ Groq API Key bulunamadı!")
    ai_model = st.sidebar.selectbox(
        "Model:",
        options=["llama-3.3-70b-versatile", "llama-3.1-8b-instant", "mixtral-8x7b-32768", "deepseek-r1-distill-llama-70b"],
        index=0,
        key="ai_model_groq"
    )
    model_full_name = ai_model

st.session_state.current_ai_provider = ai_provider
st.session_state.current_ai_model = model_full_name


@st.cache_data(ttl=3600)
def load_lottieurl(url: str):
    try:
        r = requests.get(url, timeout=5)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_loading = load_lottieurl("https://lottie.host/81729486-455b-426d-8833-255e2a222857/YV77X3ZzPZ.json") 


def is_valid_comment(text: Any) -> bool:
    """
    Sophisticated filter to remove metadata, developer responses, and garbage lines.
    Useful for App Store Connect / Play Store copy-pastes.
    """
    if not text: return False
    s = str(text).strip()
    sl = s.lower()
    
    
    if len(s) < 3: return False
    
    
    if sl in ['nan', 'null', 'none']: return False
    
    
    meta_keywords = [
        "developer response", "geliştirici cevabı", "developer answer", 
        "customer review", "müşteri yorumu", "app store connect",
        "review details", "yorum detayları", "version:", "versiyon:",
        "report a concern", "rapor et", "reply", "cevapla", "edit response", "cevabı düzenle"
    ]
    if any(k in sl for k in meta_keywords):
        return False

    
    if re.search(r"version\s+\d+(\.\d+)*", sl):
        return False
        
    
    
    months = ["jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec",
              "ocak", "şubat", "mart", "nisan", "mayıs", "haziran", "temmuz", "ağustos", "eylül", "ekim", "kasım", "aralık"]
    
    first_word = sl.split()[0].replace('.', '').replace(',', '') if sl.split() else ""
    if first_word in months and len(s) < 60:
        return False

    if len(s) < 45:
        date_regex = r"(\d{1,4}[-./]\d{1,2}[-./]\d{1,4})|((Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{1,2},?\s+\d{4})"
        if re.search(date_regex, s, re.IGNORECASE):
            return False

    
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
        
    
    if re.search(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}", sl):
        return False

    return True

def get_app_store_reviews(app_id: str, _progress_callback: Any = None, _days_limit: int = 30) -> List[Dict[str, Any]]:
    """Massive Parallel App Store Fetcher (40+ Countries) to break all limits"""
    import concurrent.futures
    all_reviews_map: Dict[str, Dict[str, Any]] = {}
    now = datetime.now()
    threshold_dt = now - timedelta(days=_days_limit)
    
    
    countries = [
        'tr', 'us', 'de', 'az', 'nl', 'fr', 'gb', 'at', 'be', 'ch', 'kz', 'uz', 'tm', 'kg', 'ru',
        'cy', 'gr', 'ro', 'bg', 'pl', 'hu', 'cz', 'se', 'no', 'dk', 'it', 'es', 'ca', 'au', 'sa',
        'ae', 'qa', 'kw', 'jo', 'lb', 'eg', 'ly', 'dz', 'ma', 'tn'
    ]
    
    def fetch_country_reviews(country: str):
        country_reviews = []
        for page in range(1, 11): 
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
    
    all_fetched_map = {}
    now = datetime.now()
    threshold_date = now - timedelta(days=days_limit)
    
    
    # Genişletilmiş dil/ülke listesi
    LANG_COUNTRY_PAIRS = [
        ('tr', 'tr'),
        ('en', 'us'),
        ('en', 'gb'),
        ('en', 'au'),
        ('en', 'ca'),
        ('ar', 'sa'),
        ('ar', 'ae'),
        ('de', 'de'),
        ('fr', 'fr'),
        ('ru', 'ru'),
        ('nl', 'nl'),
        ('es', 'es'),
        ('es', 'mx'),
        ('pt', 'br'),
        ('it', 'it'),
        ('pl', 'pl'),
        ('ro', 'ro'),
        ('bg', 'bg'),
        ('uk', 'ua'),
        ('kk', 'kz'),
    ]
    
    sort_strategies = [Sort.NEWEST, Sort.MOST_RELEVANT]
    scores = [1, 2, 3, 4, 5]
    channels = []
    for s in sort_strategies:
        for sc in scores:
            for lang, country in LANG_COUNTRY_PAIRS:
                channels.append((s, sc, lang, country))
            
    def fetch_channel(sort_type, score, lang, country):
        channel_data = []
        token = None
        
        for _ in range(30):
            try:
                result, token = play_reviews(
                    app_id, lang=lang, country=country,
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
                                # Append language info to ID to avoid collisions across regions
                                unique_id = f"{r_id}_{lang}_{country}"
                                channel_data.append({
                                    "id": unique_id, 
                                    "text": content, 
                                    "date": r_at, 
                                    "rating": str(score),
                                    "lang": lang
                                })
                        else:
                            if sort_type == Sort.NEWEST: out_of_range = True
                
                if out_of_range or not token: break
            except: break
        return channel_data

    total_channels = len(channels)
    completed_channels = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=15) as executor:
        future_to_channel = {executor.submit(fetch_channel, s, sc, l, c): (s, sc, l, c) for s, sc, l, c in channels}
        for future in concurrent.futures.as_completed(future_to_channel):
            completed_channels += 1
            if _progress_callback: _progress_callback(min(completed_channels / total_channels, 0.99))
            res = future.result()
            for r in res:
                all_fetched_map[r['id']] = r
                
    if _progress_callback: _progress_callback(1.0)
    return list(all_fetched_map.values())


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
    
    /* Responsive Container Control */
    [data-testid="stAppViewBlockContainer"] {
        max-width: 707.2px !important;
        padding-left: 1rem !important;
        padding-right: 1rem !important;
        width: 100% !important;
    }

    /* Mobile Specific Adjustments */
    @media (max-width: 768px) {
        [data-testid="stAppViewBlockContainer"] {
            max-width: 100% !important;
            padding-left: 0.5rem !important;
            padding-right: 0.5rem !important;
        }
        
        .header-title {
            font-size: 2rem !important;
        }
        
        .metric-card {
            min-width: 100% !important;
            padding: 15px !important;
        }

        /* Allow tabs to scroll horizontally on mobile instead of wrapping/breaking */
        div[data-testid="stTabList"] {
            display: flex !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            padding-bottom: 8px !important;
            scrollbar-width: none; /* Hide scrollbar for cleaner look */
        }
        div[data-testid="stTabList"]::-webkit-scrollbar {
            display: none;
        }
        
        button[data-testid="stTab"] {
            flex: 0 0 auto !important;
            white-space: nowrap !important;
        }

        /* Stack columns on mobile */
        [data-testid="column"] {
            width: 100% !important;
            flex: 1 1 100% !important;
        }
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
        background-color: #F8FAFC !important; /* Slightly whiter base */
        color: #1E293B !important;
        caret-color: #6366F1 !important; /* Indigo blinking cursor */
        border: 1px solid #E2E8F0 !important;
        transition: all 0.2s ease !important;
    }

    .stTextInput input:hover, .stTextArea textarea:hover {
        border-color: #CBD5E1 !important;
    }

    .stTextInput input:focus, .stTextArea textarea:focus {
        border-color: #6366F1 !important;
        box-shadow: 0 0 0 4px rgba(99, 102, 241, 0.15) !important;
        background-color: #FFFFFF !important;
        outline: none !important;
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
    
    /* Header Card - Neumorphic */
    .header-container {
        background-color: #F0F9FF !important;
        border: none !important;
        border-radius: 30px;
        padding: 25px;
        margin-top: 15px;
        margin-bottom: 25px;
        text-align: center;
        box-shadow: -10px -10px 20px #FFFFFF, 10px 10px 20px #D1E5F4 !important;
    }
    .header-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        color: #475569 !important;
        margin-bottom: 0px !important;
        letter-spacing: -0.5px;
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

    /* Localization: Hide 'CSV veya Excel dosyalarını buraya sürükleyin/yükleyin' and 'Limit: Her dosya için 200 MB' */
    /* Aggressive approach: Set font-size to 0 to hide original text while keeping container */
    [data-testid="stFileUploadDropzone"] section div {
        font-size: 0px !important;
    }
    [data-testid="stFileUploadDropzone"] section div::after {
        content: "CSV veya Excel dosyalarını buraya sürükleyin/yükleyin";
        font-size: 14px !important;
        font-weight: 500 !important;
        color: #475569 !important;
        display: block !important;
        visibility: visible !important;
        margin-top: 10px;
    }
    
    /* Hide the small text specifically if it's still visible elsewhere */
    [data-testid="stFileUploadDropzone"] small, 
    [data-testid="stFileUploadDropzone"] span {
        display: none !important;
        font-size: 0px !important;
    }

    /* Important: Re-show text for the 'Browse Files' button specifically */
    [data-testid="stFileUploadDropzone"] button span {
        display: inline-block !important;
        font-size: 14px !important;
        visibility: visible !important;
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
        position: absolute !important;
    }
</style>
""", unsafe_allow_html=True)


st.markdown(f"""
    <div class="header-container">
        <div class="header-title" style="margin-bottom: 0px;">AI Yorum Analizi</div>
    </div>
""", unsafe_allow_html=True)


if 'comments_to_analyze' not in st.session_state:
    st.session_state.comments_to_analyze = []

comments_to_analyze = [] 

tab1, tab2, tab3 = st.tabs(["Mağaza Linki", "Dosya Yükle (CSV/Excel)", "Metin Girişi"])

with tab1:
    with st.container(border=True):
        col_u, col_r = st.columns([2, 1])
        with col_u:
            store_url = st.text_input("Uygulama linki veya ID girin:", placeholder="Örn: com.instagram.android veya 1500198745", disabled=False)
            st.session_state.app_url = store_url 
        with col_r:
            time_range = st.selectbox(
                "Tarih Aralığı Seçin:",
                options=["Son 1 Ay", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl"],
                index=0
            )
        
        
        range_map = {"Son 1 Ay": 30, "Son 3 Ay": 90, "Son 6 Ay": 180, "Son 1 Yıl": 365}
        days_limit = range_map[time_range]
        st.markdown('<div style="margin-top: 6px; margin-bottom: 10px; font-size: 0.85rem; color: #64748b;">Apple: Mağaza linki veya ID (id...), Play Store: Link veya paket adı (com...) geçerlidir.</div>', unsafe_allow_html=True)


    if store_url.strip():
        u = store_url.strip()
        platform: Optional[str] = None
        app_id: str = ""
        country: str = "tr" 
        
        
        if "play.google.com" in u:
            platform = "google"
            match = re.search(r"id=([^&/]+)", u)
            if match: app_id = match.group(1)
        elif "apple.com" in u:
            platform = "apple"
            
            match = re.search(r"id(\d+)", u)
            if match: app_id = match.group(1)
            
            country_match = re.search(r"apple\.com/([a-z]{2,3})/", u)
            if country_match: country = country_match.group(1)
        else:
            
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

        
        fetch_key = f"{platform}_{app_id}_{time_range}_{country}"
        
        if not platform or not app_id:
            if store_url.strip():
                st.warning("Geçerli bir Play Store veya App Store linki bulunamadı.")
        elif st.session_state.get("last_fetch_key") == fetch_key and st.session_state.get("all_fetched_pool"):
             pass
        else:
            if "bulk_results" in st.session_state:
                del st.session_state.bulk_results
            if "comments_to_analyze" in st.session_state:
                st.session_state.comments_to_analyze = []

            name_for_state = app_id
            st_for_state = "Mağaza"
            if platform == "google": 
                st_for_state = "Google Play"
                try:
                    app_info = play_app(app_id, lang='tr', country='tr')
                    name_for_state = app_info.get('title', app_id)
                except: pass
            elif platform == "apple":
                st_for_state = "App Store"
                # Try iTunes API first for accurate name
                try:
                    resp = requests.get(f"https://itunes.apple.com/lookup?id={app_id}&country={country}", timeout=5)
                    if resp.status_code == 200:
                        data = resp.json()
                        if data.get('results'):
                            name_for_state = data['results'][0].get('trackCensoredName', app_id)
                        elif "apple.com" in u and "/app/" in u:
                            raw_name = u.split("/app/")[-1].split("/")[0].replace("-", " ")
                            name_for_state = urllib.parse.unquote(raw_name).title()
                except:
                    if "apple.com" in u and "/app/" in u:
                        try:
                            raw_name = u.split("/app/")[-1].split("/")[0].replace("-", " ")
                            name_for_state = urllib.parse.unquote(raw_name).title()
                        except: pass

            with st.container():
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    st.markdown(f"### 🔍 {name_for_state} Analizi")
                    if lottie_loading:
                        st_lottie(lottie_loading, height=130, key="fetch_loader")
                    p_bar = st.progress(0, text="Hazırlanıyor...")
                
                def update_fetch_progress(p: float) -> None:
                    
                    p_safe = min(max(float(p), 0.0), 1.0)
                    
                    
                    last_p = float(st.session_state.get("_last_fetch_p", 0.0))
                    
                    
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
                        # Az yorum uyarısı
                        if len(fetched_comments) < 50:
                            st.warning(
                                f"⚠️ Bu uygulama için yalnızca **{len(fetched_comments)}** yorum bulundu. "
                                f"Doğru uygulama ID'sini kullandığınızdan emin olun. "
                            )
                        
                        update_fetch_progress(1.0)
                        time.sleep(0.5)
                        
                        
                        loading_placeholder.empty()
                        
                        
                        fetched_comments.sort(key=lambda x: x['date'], reverse=True)
                        st.session_state.all_fetched_pool = fetched_comments
                        st.session_state.last_fetch_key = fetch_key 
                        
                        analysis_type_now = st.session_state.get("analysis_type", "Hızlı Analiz")
                        AI_LIMIT = 500

                        if analysis_type_now == "Zengin Analiz" and len(fetched_comments) > AI_LIMIT:
                            total_found = len(fetched_comments)
                            limited_comments = fetched_comments[:AI_LIMIT]
                            st.session_state.comments_to_analyze = limited_comments

                            v_dates_anal = [r.get('date') for r in limited_comments if isinstance(r.get('date'), datetime)]
                            v_dates_pool = [r.get('date') for r in fetched_comments if isinstance(r.get('date'), datetime)]
                            if v_dates_anal and v_dates_pool:
                                pool_start = cast(datetime, min(v_dates_pool)).strftime('%d-%m-%Y')
                                pool_end   = cast(datetime, max(v_dates_pool)).strftime('%d-%m-%Y')
                                anal_start = cast(datetime, min(v_dates_anal)).strftime('%d-%m-%Y')
                                anal_end   = cast(datetime, max(v_dates_anal)).strftime('%d-%m-%Y')
                                st.warning(f"""
                                    Toplamda **{total_found}** yorum bulundu (Tüm Aralık: {pool_start} - {pool_end}).
                                    Zengin Analiz kotası için **en güncel {AI_LIMIT} tanesi** seçildi 
                                    (Analiz Aralığı: {anal_start} - {anal_end}).
                                """)
                        else:
                            # Hızlı Analiz — limit yok, tamamı alınır
                            st.session_state.comments_to_analyze = fetched_comments
                            if len(fetched_comments) > AI_LIMIT:
                                st.info(f"Hızlı Analiz modunda tüm **{len(fetched_comments)}** yorum analiz edilecek.")
                        
                        st.success(f"**{len(st.session_state.comments_to_analyze)}** adet {time_range} yorumu başarıyla çekildi!")
                    else:
                        loading_placeholder.empty()
                        st.info(f"{time_range} kriterine uygun yorum bulunamadı.")
                except Exception as e:
                    loading_placeholder.empty()
                    st.error(f"Yorumlar çekilirken bir hata oluştu: {e}")
        
with tab2:
    uploaded_files = st.file_uploader("Dosya Yükle", type=["csv", "xlsx"], accept_multiple_files=True, label_visibility="collapsed")
    if uploaded_files:
        # Use a list of file info as a key to detect if files changed
        current_files_key = "_".join([f"{f.name}_{f.size}" for f in uploaded_files])
        if st.session_state.get("last_files_key") != current_files_key:
            if "bulk_results" in st.session_state:
                del st.session_state.bulk_results
            if "comments_to_analyze" in st.session_state:
                st.session_state.comments_to_analyze = []
            st.session_state.last_files_key = current_files_key
        
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
                    st.markdown(f"### 📄 {uploaded_file.name}")
                    with st.container(border=True):
                        
                        
                        
                        date_keys = ["date", "time", "tarih", "saat"]
                        rate_keys = ["rating", "star", "puan", "yildiz", "skor", "score"]
                        
                        date_col = None
                        rate_col = None
                        
                        
                        for col in df_upload.columns:
                            if "Review Last Update Date and Time" in col:
                                date_col = col
                                break
                        
                        for col in df_upload.columns:
                            col_l = col.lower()
                            
                            if "Review Submit Date and Time" in col:
                                continue
                            if not date_col and any(dk in col_l for dk in date_keys): 
                                date_col = col
                            if not rate_col and any(rk in col_l for rk in rate_keys): 
                                rate_col = col

                        
                        scores = []
                        for col in df_upload.columns:
                            col_l = col.lower()
                            score = 0
                            
                            if any(k in col_l for k in ["review", "yorum", "text", "metin", "content", "mesaj"]): score += 20
                            
                            if any(k in col_l for k in ["id", "rating", "star", "puan", "date", "tarih"]): score -= 25
                            
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


                            
                            valid_masks = df_upload[col_name].astype(str).apply(is_valid_comment)
                            
                            processed_df = pd.DataFrame({
                                "text": df_upload[col_name].astype(str).str.strip(),
                                "is_valid": valid_masks
                            })
                            
                            
                            if date_col:
                                try:
                                    processed_df["date"] = pd.to_datetime(df_upload[date_col], errors='coerce')
                                    
                                    if processed_df["date"].isnull().sum() > len(processed_df) * 0.5:
                                        processed_df["date"] = pd.to_datetime(df_upload[date_col], errors='coerce', dayfirst=True)
                                    
                                    processed_df["date"] = processed_df["date"].apply(lambda x: x.replace(tzinfo=None) if pd.notnull(x) and hasattr(x, 'tzinfo') else x)
                                except:
                                    pass
                            
                            
                            if rate_col:
                                processed_df["rating"] = df_upload[rate_col].astype(str)
                            
                            
                            mask = processed_df["is_valid"] | (processed_df["rating"].notnull() if rate_col else False)
                            final_comments_df = processed_df[mask]
                            all_comments.extend(final_comments_df.to_dict('records'))
                            valid_in_file = int(final_comments_df["is_valid"].sum())

                            st.caption(f"Bu dosyadan {valid_in_file} gecerli yorum eklendi.")
                            
            except Exception as e:
                st.error(f"{uploaded_file.name} okuma hatası: {e}")
        
        if all_comments:
            analysis_type_now = st.session_state.get("analysis_type", "Hızlı Analiz")
            AI_LIMIT = 500

            if analysis_type_now == "Zengin Analiz" and len(all_comments) > AI_LIMIT:
                st.warning(f"Zengin Analiz kotası: ilk {AI_LIMIT} yorum alındı (Toplam: {len(all_comments)}).")
                st.session_state.comments_to_analyze = all_comments[:AI_LIMIT]
            else:
                st.session_state.comments_to_analyze = all_comments
                if len(all_comments) > AI_LIMIT:
                    st.info(f"Hızlı Analiz: tüm **{len(all_comments)}** yorum işlenecek.")
            st.success(f"Toplam **{len(st.session_state.comments_to_analyze)}** gerçek yorum analiz için hazır!")

with tab3:
    text_input = st.text_area(
        "Yorumları alt alta girin:",
        height=200,
        placeholder="Örn: Harika uygulama!\nKötü performans...",
        key="manual_text_input"
    )
    if text_input.strip():
        
        current_text_hash = str(hash(text_input))
        if st.session_state.get("last_text_hash") != current_text_hash:
            if "bulk_results" in st.session_state:
                del st.session_state.bulk_results
            if "comments_to_analyze" in st.session_state:
                st.session_state.comments_to_analyze = []
            st.session_state.last_text_hash = current_text_hash
            
        raw_lines = text_input.split('\n')
        processed_comments: List[Dict[str, Any]] = []
        
        
        store_meta_regex = r"^(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|Ocak|Şubat|Mart|Nisan|Mayıs|Haziran|Temmuz|Ağustos|Eylül|Ekim|Kasım|Aralık)\s+\d{1,2},?\s+\d{4}\s*-\s*.*$"
        
        skip_dev_block = False
        
        for line in raw_lines:
            l = line.strip()
            if not l: continue
            
            
            if re.search(store_meta_regex, l, re.IGNORECASE):
                skip_dev_block = False 
                
                if processed_comments:
                    
                    last_idx = len(processed_comments) - 1
                    if len(str(processed_comments[last_idx].get("text", ""))) < 85:
                        processed_comments.pop()
                continue
                
            
            if any(k in l.lower() for k in ["developer response", "geliştirici cevabı"]):
                skip_dev_block = True
                continue
            
            if skip_dev_block:
                continue

            if is_valid_comment(l):
                processed_comments.append({"text": l})
                
        analysis_type_now = st.session_state.get("analysis_type", "Hızlı Analiz")
        AI_LIMIT = 500

        if analysis_type_now == "Zengin Analiz" and len(processed_comments) > AI_LIMIT:
            st.warning(f"Zengin Analiz kotası: ilk {AI_LIMIT} yorum alındı.")
            st.session_state.comments_to_analyze = processed_comments[:AI_LIMIT]
        else:
            st.session_state.comments_to_analyze = processed_comments
            st.success(f"Toplam **{len(st.session_state.comments_to_analyze)}** geçerli satır eklendi!")


comments_to_analyze = st.session_state.comments_to_analyze



if comments_to_analyze:
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    st.markdown("## Analiz Ayarları")
    
    n = len(comments_to_analyze)

    def fmt_time(secs):
        m, s = divmod(secs, 60)
        return f"{m} dakika {s} saniye" if m > 0 else f"{s} saniye"

    col_method, col_depth = st.columns([1, 1])
    
    with col_method:
        analysis_type = st.radio(
            "Yöntem:",
            options=["Hızlı Analiz", "Zengin Analiz"],
            index=0,
            key="analysis_type"
        )

    with col_depth:
        if analysis_type == "Zengin Analiz":
            mode_idx = st.radio(
                "Derinlik:",
                options=[0, 1],
                format_func=lambda x: ["Genel", "Derin"][x],
                captions=[
                    f"~ {fmt_time(n * 1)}",
                    f"~ {fmt_time(n * 2)}"
                ],
                key="analysis_mode"
            )
        else:
            
            st.session_state.analysis_mode = 0
            mode_idx = 0

    if analysis_type == "Zengin Analiz":
        st.info("Zengin Analiz: Sonuçlar yapay zeka tarafından derinlemesine taranır.")
    else:
        st.info("Hızlı Tarama: Kelime bazlı analiz yapar. Basit derinlikte sonuç üretir.")


def get_ai_sentiment(text, model_name=None, provider=None):
    if API_TRACKER["cost_tl"] >= 150.0:
        return {"_error": "cost_limit"}
    
    if provider is None:
        provider = st.session_state.get('current_ai_provider', 'Google Gemini')
    if model_name is None:
        model_name = st.session_state.get('current_ai_model', 'models/gemini-2.0-flash')

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
"لا أستطيع الدخول إلى حسابı" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"لم أستلم أمwal الاسترجاع رغم تواصلي مرات عديدة" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"الموقع ممتاز لكن عند الاسترجاع لا تصلك الامwal" → {{"olumlu":0.05,"olumsuz":0.88,"istek_gorus":0.07}}
"سعر المنتج قبل اضافته للسله يختلف عن بعد الاضافة" → {{"olumlu":0.02,"olumsuz":0.95,"istek_gorus":0.03}}
"yتم شحن الwal مختلفه واغراض غير اصليه بجودة رdiئة" → {{"olumlu":0.02,"olumsuz":0.96,"istek_gorus":0.02}}

SOMUT ÖRNEKLER - ARAPÇA - İSTEK/GÖRÜŞ:
"أتمنى أن يضيفوا خاصية البحث بالصور" → {{"olumlu":0.10,"olumsuz":0.05,"istek_gorus":0.85}}
"متى سيكون التطبيق mتاحاً في دولتي؟" → {{"olumlu":0.05,"olumsuz":0.05,"istek_gorus":0.90}}
"مو سامح لي اختar دولة" → {{"olumlu":0.05,"olumsuz":0.40,"istek_gorus":0.55}}

ÇIKTI KURALI: SADECE JSON döndür, başka hiçbir şey yazma.
{{"olumlu": X, "olumsuz": Y, "istek_gorus": Z}}

ZENGİN ANALİZ EK NOTU: Eğer kullanıcı "Zengin ve Derin" seçmişse, yorumdaki alt metinleri, imaları ve yapısal eleştirileri de dikkate al.

Yorum: "{text}"
"""

    content = ""
    try:
        if provider == "Google Gemini":
            response = GEMINI_CLIENT.models.generate_content(
                model=model_name,
                contents=prompt,
                config=genai_types.GenerateContentConfig(temperature=0)
            )
            meta = getattr(response, 'usage_metadata', None)
            if meta:
                prompt_tokens = getattr(meta, 'prompt_token_count', 0)
                cand_tokens = getattr(meta, 'candidates_token_count', 0)
                is_pro = 'pro' in model_name.lower()
                cost_in = prompt_tokens * (3.50 if is_pro else 0.075) / 1000000
                cost_out = cand_tokens * (10.50 if is_pro else 0.30) / 1000000
                API_TRACKER["cost_tl"] += (cost_in + cost_out) * 36.0 
            content = response.text
        elif provider == "Mistral AI":
            response = MISTRAL_CLIENT.chat.complete(
                model=model_name,
                messages=[{"role": "user", "content": prompt}]
            )
            usage = getattr(response, 'usage', None)
            if usage:
                p_t = getattr(usage, 'prompt_tokens', 0)
                c_t = getattr(usage, 'completion_tokens', 0)
                is_large = 'large' in model_name.lower()
                cost_in = p_t * (2.0 if is_large else 0.2) / 1000000
                cost_out = c_t * (6.0 if is_large else 0.6) / 1000000
                API_TRACKER["cost_tl"] += (cost_in + cost_out) * 36.0
            content = response.choices[0].message.content
        elif provider == "Groq AI":
            response = GROQ_CLIENT.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0
            )
            usage = getattr(response, 'usage', None)
            if usage:
                p_t = getattr(usage, 'prompt_tokens', 0)
                c_t = getattr(usage, 'completion_tokens', 0)
                # Groq pricing is much lower usually
                cost_in = p_t * 0.1 / 1000000
                cost_out = c_t * 0.4 / 1000000
                API_TRACKER["cost_tl"] += (cost_in + cost_out) * 36.0
            content = response.choices[0].message.content
    except Exception as e:
        return {"_error": f"{provider} hatası: {str(e)[:100]}"}

    match = re.search(r'\{.*?\}', content, re.DOTALL)
    if match:
        try:
            data = json.loads(match.group())
            p = float(data.get("olumlu", 0))
            n = float(data.get("olumsuz", 0))
            neu = float(data.get("istek_gorus", 0))
            total = p + n + neu
            if total > 0:
                return {
                    "olumlu": p/total, 
                    "olumsuz": n/total, 
                    "istek_gorus": neu/total,
                    "method": model_name.split('/')[-1]
                }
        except: pass
    return {"_error": "Yapay zeka yanıtı anlaşılamadı."}


def generate_dynamic_summary(analysis_results: List[Dict[str, Any]], model_name=None, provider=None):
    if provider is None:
        provider = st.session_state.get('current_ai_provider', 'Google Gemini')
    if model_name is None:
        model_name = st.session_state.get('current_ai_model', 'models/gemini-1.5-flash')
        
    if not analysis_results: return None
    valid_results = [r for r in analysis_results if r.get('Baskın Duygu') != "—"]
    if not valid_results: return "Yeterli veri analiz edilemedi."

    pos = [r['Yorum'] for r in valid_results if r['Baskın Duygu'] == "Olumlu"]
    neg = [r['Yorum'] for r in valid_results if r['Baskın Duygu'] == "Olumsuz"]
    neu = [r['Yorum'] for r in valid_results if r['Baskın Duygu'] == "İstek/Görüş"]
    
    def get_sample(texts, count=15):
        import random
        if len(texts) <= count: return "\n".join([f"- {t[:200]}" for t in texts])
        return "\n".join([f"- {t[:200]}" for t in random.sample(texts, count)])

    prompt = f"""Bir uygulama mağazası yorum analisti gibi davran. Aşağıdaki analiz sonuçlarını inceleyip derinlemesine bir rapor sun.
    {get_sample(neu)}

    RAPOR FORMATI:
    1. "Kullanıcı Deneyimi Özeti": (Dinamik ve profesyonel bir başlık ve 4-5 cümlelik derin analiz)
    2. "Öne Çıkan Güçlü Yönler": (Kullanıcıları en çok mutlu eden 2-3 madde)
    3. "Kritik Sorunlar ve Çözüm Önerileri": (En çok şikayet edilen konular ve geliştirici ekibe öneri)

    Dil: TÜRKÇE. Markdown formatında yaz. Link veya emoji kullanabilirsin.
    """
    try:
        response = GEMINI_CLIENT.models.generate_content(
            model=model_name,
            contents=prompt,
            config=genai_types.GenerateContentConfig(temperature=0.7)
        )
        return response.text
    except Exception as e:
        return f"Dinamik özet oluşturulurken bir hata oluştu: {str(e)[:100]}"


def heuristic_analysis(text, rating=None):
    """
    Heuristic Engine v3.0
    - 1750+ Instagram/App Store yorumuyla eğitildi
    - TR / EN / AR / RU / FR / DE / ES / NL / RO / BG destekli
    - Rating-aware: puan ile içerik çelişirse içeriği önce denetler
    - Sarkasm, pivot (ama/but), "öne çıksın" tuzağı tespiti
    """
    t = str(text).lower().strip()
    if not t or len(t) < 2:
        return {"olumlu": 0.33, "olumsuz": 0.34, "istek_gorus": 0.33, "method": "Heuristic+"}

    # ── 1. PUAN BAZLI HIZLI SINYALLER ────────────────────────────────────────
    # Rating parametresi varsa güçlü sinyal olarak kullan
    _rating = None
    if rating is not None:
        try:
            _rating = int(str(rating).strip().split('.')[0])
        except:
            pass

    # ── 2. PUAN MANİPÜLASYON TUZAĞI ("öne çıksın diye 5 yıldız") ────────────
    manipulation_patterns = [
        "öne çıksın diye", "üste çıksın diye", "en üste çıksın",
        "görülsün diye yüksek", "fark edilsin diye", "dikkat çeksin diye",
        "yüksek puan verdim ama", "5 yıldız verdim ama", "beş yıldız verdim ama",
        "puan verdim ama aslında", "5 puan ama",
    ]
    if any(p in t for p in manipulation_patterns):
        return {"olumlu": 0.05, "olumsuz": 0.88, "istek_gorus": 0.07, "method": "Heuristic+"}

    # ── 3. YILDIZ İFADESİ METİN İÇİNDE ──────────────────────────────────────
    if any(x in t for x in ["1 yıldız", "bir yıldız", "1 stern", "1 star", "1 étoile", "1/5", "one star"]):
        return {"olumlu": 0.03, "olumsuz": 0.94, "istek_gorus": 0.03, "method": "Heuristic+"}
    if any(x in t for x in ["5 yıldız", "beş yıldız", "5 stars", "5 étoiles", "5/5", "five stars", "5 stern"]):
        return {"olumlu": 0.94, "olumsuz": 0.03, "istek_gorus": 0.03, "method": "Heuristic+"}

    # ── 4. TAM EŞLEŞMELİ KISA METİNLER (exact match) ─────────────────────────
    EXACT_POS = {
        # TR
        "harika", "mükemmel", "süper", "güzel", "iyi", "başarılı", "şahane",
        "teşekkürler", "sağolun", "bayıldım", "muhteşem", "muq", "müq", "çok iyi",
        "çok güzel", "on numara", "bravo", "aferin", "efsane", "müthiş",
        # EN
        "best", "great", "amazing", "perfect", "love", "excellent", "wonderful",
        "fantastic", "awesome", "brilliant", "superb", "outstanding", "good",
        "nice", "top", "cool", "super",
        # AR
        "ممتاز", "احبه", "رائع", "جميل", "افضل", "الافضل", "تمام", "مبدع",
        # RU
        "отлично", "супер", "топ", "хорошо", "нравится", "класс",
        "великолепно", "замечательно", "прекрасно",
        # FR
        "génial", "magnifique", "parfait", "incroyable", "bravo",
        "top", "bien", "excellent",
        # DE
        "toll", "super", "ausgezeichnet", "wunderbar", "prima", "klasse", "perfekt",
        # ES/PT
        "excelente", "genial", "fantástico", "ótimo", "maravilhoso", "buenísimo",
        # PL/RO/Other
        "świetne", "niezawodny", "foarte bună", "super",
    }
    if t in EXACT_POS:
        return {"olumlu": 0.95, "olumsuz": 0.02, "istek_gorus": 0.03, "method": "Heuristic+"}

    EXACT_NEG = {
        # TR
        "çöp", "berbat", "rezalet", "rezil", "saçma", "iğrenç", "kötü",
        "berbatsin", "berbattın", "bk gibi", "çöp gibi",
        # EN
        "trash", "scam", "worst", "terrible", "horrible", "awful",
        "disgusting", "garbage", "pathetic", "useless", "rubbish",
        "bad", "broken",
        # AR
        "سيء", "أسوأ", "مروع", "فاشل",
        # RU
        "ужасно", "отстой", "мусор", "кошмар",
        # DE
        "schrecklich", "schlecht", "furchtbar", "mist",
        # FR
        "nul", "catastrophique", "horrible",
        # TR kısa
        "kötü", "çöp", "saçma",
    }
    if t in EXACT_NEG:
        return {"olumlu": 0.03, "olumsuz": 0.94, "istek_gorus": 0.03, "method": "Heuristic+"}

    # ── 5. SARKASM / İRONİ TESPİTİ ───────────────────────────────────────────
    SARKASM = [
        "ne indirin ne de indirin", "ne indirin nede",
        "aferin size", "aferin sizlere",
        "tebrikler size", "tebrikler size gerçekten",
        "çok faydalı olacaktır",  # "böyle yapmaya devam edin, meta'ya çok faydalı olacaktır"
        "böyle yapmaya devam edin",
        "sizi bildiği gibi yapsın", "allah belanızı versin", "başınıza taş yağsın",
        "bravo size", "helal olsun yine",  # "helal olsun yine çöktü" → negatif
        "indirdim ve bağımlı", "indirdim ve bir otist",
        "indirmeden önce çok normaldim",
    ]
    sarkasm_hit = any(p in t for p in SARKASM)

    # ── 6. KEYWORD LİSTELERİ ─────────────────────────────────────────────────

    NEG_WORDS = [
        # TR — Hesap sorunları (EN ÇOK şikayet)
        "askıya", "askıya alındı", "askıya alınmış", "askıya alınıyor",
        "hesabım kapatıldı", "hesabımı kapattılar", "hesaplarım kapandı",
        "hesabım kapandı", "kapatılmış", "kapatıldı",
        "itiraz", "itiraz ettim",
        "giriş yapamıyorum", "giremiyorum", "giriş yapamıyorum",
        "hesabıma giremiyorum", "hesabıma girilmiyor",
        "şifre yanlış", "şifremi doğru girdiğim halde",
        "şifre doğru ama yanlış", "şifre doğru olmasına rağmen",
        "durduruldu", "askıda",
        "ip ban", "cihaz ban", "cihaz banı", "ip banı",
        "yeni hesap açınca da kapanıyor", "açtığım her hesap",
        "her hesap kapatılıyor", "her hesabım kapanıyor",
        "20 hesap", "10 hesap", "5 hesap",  # "10 hesap açtım hepsi kapandı"

        # TR — Uygulama sorunları
        "donuyor", "kasıyor", "çöküyor", "çöktü", "kapanıyor",
        "çalışmıyor", "yavaş", "hata veriyor", "hata var",
        "bozuk", "bozuldu", "berbat", "kötü", "rezil", "rezalet",
        "sorun", "problem", "çöp", "saçma", "yaramaz", "iğrenç",
        "durduruldu", "sildim", "siliyorum", "kaldırdım", "kaldırıyorum",
        "yüklenmiyor", "açılmıyor", "gözükmüyor", "görünmüyor",
        "boş ekran", "lag", "atıyor", "uygulama atıyor",
        "uygulama çöküyor", "uygulama donuyor",

        # TR — Chat/mesaj sorunları
        "mesaj gitmiyor", "mesajlar gitmiyor", "mesaj gelmiyor",
        "mesajlara giremiyorum", "dm sorunu", "mesaj yüklenmiyor",
        "sohbet açılmıyor", "mesaj düşmüyor",

        # TR — Reklam
        "reklam çok", "aşırı reklam", "her yerden reklam",
        "full reklam", "çok reklam", "reklam dolu",
        "her reels reklam", "2 reels 1 reklam", "1 reklam 1 reels",
        "34 reels 19 reklam",  # spesifik sayım
        "reklam sayfasına atıyor", "reklam geçilmiyor",
        "reklam donuyor",

        # TR — İçerik/feed sorunları
        "eski gönderiler", "4 günlük", "günler önceki",
        "feed yenilenmiyor", "akış yenilenmiyor",
        "takip etmediğim", "alakasız videolar", "alakasız içerik",
        "yabancı dil videoları",

        # TR — Müzik
        "müzik yok", "müzik çalışmıyor", "müzik kaldırıldı",
        "ses yok", "ses gitmiyor", "ses çalışmıyor",
        "audio çalışmıyor",

        # TR — Tema/filtre
        "tema gitti", "temalar gitti", "temalar kaldırıldı", "tema yok",
        "filtreler gitti", "eski filtreler", "efektler kaldırıldı",

        # TR — Fotoğraf/galeri
        "fotoğraflar karışık", "galeri karışık", "fotoğraf seçemiyorum",
        "fotoğraf açılmıyor", "foto açılamadı", "resim yüklenmiyor",
        "fotoğraf yüklenmiyor", "profil fotoğrafı değişmiyor",
        "profil resmi yüklenmiyor",

        # TR — Arşiv/anı
        "anılar yok", "arşiv yok", "geçmiş hikayeler gözükmüyor",
        "eski hikayeler yok", "anılar çıkmıyor", "anım çıkmıyor",

        # TR — Kayıtlar
        "kaydedilenlerden silince başa dönüyor",
        "kaydedilenler başa atıyor", "kaydettiklerim karışık",

        # TR — Güncelleme
        "güncelleme kötü", "güncelleme bozdu", "yeni güncelleme kötü",
        "son güncelleme berbat", "güncelleme sonrası bozuldu",

        # TR — Reel/video
        "reels açılmıyor", "video açılmıyor", "reels izleyemiyorum",
        "video yüklenmiyor", "siyah ekran", "kara ekran",
        "videolar görünmüyor", "reels çalışmıyor",

        # TR — Genel kötü deneyim
        "mahvoldu", "batık", "mağdur", "mağdurum",
        "yeter artık", "bıktım artık", "artık bıktım",
        "gına geldi", "sinir bozucu", "can sıkıcı",
        "berbatlaştı", "kötüleşti", "giderek kötü",
        "eskiden iyiydi", "eskisi daha iyiydi",
        "eski haline getirin", "eski haline dönün",
        "eski instagram", "eski versiyonu",

        # TR — Safariden giriyor ama uygulamadan girmiyor
        "safariden giriyor ama uygulamadan",
        "google chrome giriyor ama uygulama",
        "telefonumdan giremiyorum",
        "kendi telefonumdan giremiyorum",
        "başka cihazdan giriyor ama",

        # TR — Özellik gelmiyor
        "bana gelmiyor", "hesabıma gelmiyor",
        "güncelleme gelmiyor", "özellik gelmiyor",
        "herkeste var bende yok",

        # TR — Moderasyon
        "haksız", "haksız yere", "sebepsiz", "sebepsiz yere",
        "hiçbir şey yapmadığım halde", "suçum yok ama",
        "topluluk kuralları ihlali yok ama",

        # EN — Account/ban
        "suspended", "suspension", "banned", "ban", "disabled",
        "account disabled", "account suspended", "account banned",
        "no reason", "false ban", "wrongly banned",
        "cant login", "can't login", "login loop", "login issue",
        "permanently banned", "permanently disabled", "permanently suspended",
        "lost my account", "lost access",
        "falsely banned for cse", "cse ban", "false cse",
        "wrongfully banned cse", "accused of cse",

        # EN — App issues
        "crashing", "crashes", "keeps crashing", "crash",
        "freezing", "freeze", "lag", "lagging", "laggy",
        "not working", "doesn't work", "won't work", "stopped working",
        "bug", "glitch", "glitching", "broken",
        "terrible", "horrible", "awful", "disgusting", "garbage",
        "worst app", "worst update", "hate this", "ruined",

        # EN — Ads
        "too many ads", "ads everywhere", "all ads", "ad every",
        "non stop ads", "constant ads", "flooded with ads",
        "ad breaks", "mid video ads",

        # EN — Photos out of order
        "photos out of order", "pictures out of order",
        "not in chronological order", "photos jumbled",
        "photos all mixed up", "gallery mixed up",

        # EN — Login via browser not app
        "can login on safari but not app",
        "works on browser not app",
        "can log in on computer but not app",

        # EN — No human support
        "no human support", "no human review", "no real person",
        "can't reach anyone", "no way to contact",
        "zero support", "no support contact",
        "appeal ignored", "appeal rejected instantly",

        # EN — Music
        "no music", "music not working", "music removed", "music banned",
        "audio unavailable", "no audio",

        # EN — Themes
        "themes gone", "themes removed", "themes disappeared",

        # EN — Messages
        "messages not loading", "cant send messages",
        "messages not working", "messages not sending",
        "messages failed", "dm not working",

        # EN — General
        "scam", "fraud",
        "fix this", "fix your app", "fix it",
        "something went wrong",
        "error", "not loading",

        # RU
        "заблокировали", "блокировка", "аккаунт заблокирован",
        "бан", "забанили", "не работает",
        "удалили", "пропало", "исчезло", "убрали",
        "не грузится", "не загружается", "глючит", "виснет",
        "зависает", "не открывается", "ошибка",
        "ужасно", "отвратительно", "верните",
        "перестало работать", "сломали", "испортили",
        "не приходят сообщения", "темы пропали",
        "музыка не работает", "музыку убрали",
        "фото не по порядку", "галерея перемешана",
        "не могу загрузить фото",

        # FR
        "suspendu", "banni", "compte supprimé", "compte suspendu",
        "ne fonctionne plus", "ne fonctionne pas",
        "plante", "bloqué", "erreur", "problème", "nul",
        "horrible", "catastrophique", "trop de pubs",
        "thèmes disparus", "messages ne chargent pas",
        "photos mélangées", "photos dans le désordre",

        # DE
        "gesperrt", "konto gesperrt", "account gesperrt",
        "funktioniert nicht", "abstürzt", "fehler",
        "schlecht", "schrecklich", "zu viel werbung",
        "themen weg", "nachrichten laden nicht",
        "fotos durcheinander", "bilder durcheinander",
        "grundlos gesperrt",

        # AR
        "حظر", "محظور", "تم حظر", "تعطيل", "معطل",
        "لا يعمل", "لا تعمل", "مشكلة", "خطأ",
        "سيء", "أسوأ", "مروع", "فشل",
        "الرسائل لا تصل", "لا يوجد موسيقى",

        # ES
        "suspendido", "baneado", "no funciona", "error",
        "terrible", "horrible", "demasiada publicidad",
        "fotos desordenadas", "fotos mezcladas",

        # RO/BG/Other
        "temele dispărut", "nu funcționează", "blocat", "suspendat",
        "темите изчезнаха", "не работи", "забранен",
    ]

    POS_WORDS = [
        # TR
        "teşekkür", "harika", "mükemmel", "güzel", "süper", "başarılı",
        "memnun", "seviyorum", "bayıldım", "efsane", "müthiş", "kusursuz",
        "pratik", "hızlı", "kaliteli", "faydalı", "yararlı", "şahane",
        "en iyi", "çok iyi", "beğendim", "beğeniyorum", "tavsiye ederim",
        "ideal", "keyifli", "harikasınız", "sağolun", "tebrikler",
        "iyiki", "çok güzel", "çok seviyorum", "seviyorum",
        "on numara", "muhteşem",

        # EN
        "love", "amazing", "great", "excellent", "perfect", "wonderful",
        "fantastic", "awesome", "best", "brilliant", "superb",
        "outstanding", "helpful", "useful", "recommend", "enjoy",
        "enjoying", "happy", "pleased", "satisfied", "good job",
        "well done", "keep it up",

        # AR
        "ممتاز", "احبه", "رائع", "جميل", "افضل", "الافضل",
        "احب", "رائعة", "مميز", "شكرا", "مبدع",

        # RU
        "отлично", "супер", "топ", "нравится", "класс",
        "великолепно", "замечательно", "лучшее", "люблю", "обожаю",
        "молодцы", "спасибо", "прекрасно",

        # FR
        "adore", "j'adore", "génial", "magnifique", "fantastique",
        "parfait", "excellent", "bravo", "merci",

        # DE
        "toll", "ausgezeichnet", "fantastisch", "wunderbar",
        "hervorragend", "prima", "danke", "perfekt",

        # ES/PT
        "ótimo", "excelente", "fantástico", "maravilhoso",
        "buenísimo", "genial",

        # Other
        "świetne", "niezawodny", "très bien", "très bonne",
    ]

    NEU_WORDS = [
        # TR — İstek
        "keşke", "gelse", "olsa", "olurdu", "ekleyin", "ekleseniz",
        "geri getirin", "geri getirilsin", "ne zaman", "neden gelmiyor",
        "yapın", "istiyoruz", "öneri", "eksik",
        "daha iyi olabilir", "bi baksanız", "eklense", "gelsin",
        "düzeltilsin", "düzeltin lütfen", "lütfen ekle",
        "fikrim", "önerim", "bekliyoruz", "özellik istiyorum",
        "ekleyebilirler", "ekleseler",

        # TR — Yeni özellik talepleri (bu veriden)
        "repost özelliği gelsin", "repost geri gelsin",
        "profil görüntüleme gelsin", "takipten çıkanları görelim",
        "hikaye yorumları gelsin", "canlı yayın herkese",
        "çoklu profil fotoğrafı gelsin", "eski filtreleri geri getir",
        "kronolojik sıra", "tarih sırasına göre sıralasın",

        # EN
        "please add", "please fix", "please bring back", "when will",
        "would be nice", "i wish", "suggestion", "request",
        "feature request", "bring back", "need this", "want this",
        "could you add", "consider adding", "hope you add",

        # AR
        "أتمنى", "أريد", "يرجى", "من فضلكم", "اقتراح", "متى",

        # RU
        "хотелось бы", "было бы хорошо", "добавьте", "верните",
        "просьба", "предлагаю", "когда добавят",

        # FR
        "j'aimerais", "serait bien", "s'il vous plaît", "suggestion",
        "quand est-ce",

        # DE
        "wäre schön", "bitte fügt", "wünsche mir", "vorschlag",
    ]

    # ── 7. KEYWORD SCORING ────────────────────────────────────────────────────
    neg_score = sum(1 for w in NEG_WORDS if w in t)
    pos_score = sum(1 for w in POS_WORDS if w in t)
    neu_score = sum(1 for w in NEU_WORDS if w in t)

    # Negatif kelimeler biraz daha ağır
    neg_score_w = neg_score * 1.25

    # Sarkasm bulunmuşsa → içeriği negatif say
    # (ne indirin ne de indirin → keyword yok ama sarkasm var)
    if sarkasm_hit:
        # Eğer açıkça pozitif keyword baskın değilse → olumsuz
        if pos_score <= neg_score or pos_score == 0:
            return {"olumlu": 0.05, "olumsuz": 0.90, "istek_gorus": 0.05, "method": "Heuristic+"}

    # ── 8. "AMA/BUT" PIVOT KURALI ─────────────────────────────────────────────
    PIVOT_TR = [" ama ", " fakat ", " lakin ", " ancak ", " ne var ki "]
    PIVOT_EN = [" but ", " however ", " although ", " though "]
    ALL_PIVOTS = PIVOT_TR + PIVOT_EN

    for pivot in ALL_PIVOTS:
        if pivot in t:
            parts = t.split(pivot, 1)
            after = parts[1] if len(parts) > 1 else ""

            after_neg = sum(1 for w in NEG_WORDS if w in after)
            after_pos = sum(1 for w in POS_WORDS if w in after)

            if after_neg > after_pos and after_neg > 0:
                conf = min(0.88, 0.70 + after_neg * 0.04)
                return {"olumlu": 0.06, "olumsuz": round(conf, 3),
                        "istek_gorus": round(1-conf-0.06, 3), "method": "Heuristic+"}
            if after_pos > after_neg and after_pos > 0:
                conf = min(0.88, 0.70 + after_pos * 0.04)
                return {"olumlu": round(conf, 3), "olumsuz": 0.06,
                        "istek_gorus": round(1-conf-0.06, 3), "method": "Heuristic+"}
            # pivot var ama net karar yok → genel keyword scoring devam etsin
            break

    # ── 9. RATING OVERRIDE (son adım) ────────────────────────────────────────
    # İçerik keyword'lerden karar veremediyse rating'e bak
    total_kw = pos_score + neg_score + neu_score

    if total_kw == 0:
        # Hiç keyword yok → rating'e bak
        if _rating == 1:
            return {"olumlu": 0.05, "olumsuz": 0.88, "istek_gorus": 0.07, "method": "Heuristic+"}
        if _rating == 2:
            return {"olumlu": 0.15, "olumsuz": 0.72, "istek_gorus": 0.13, "method": "Heuristic+"}
        if _rating == 4:
            return {"olumlu": 0.72, "olumsuz": 0.15, "istek_gorus": 0.13, "method": "Heuristic+"}
        if _rating == 5:
            return {"olumlu": 0.85, "olumsuz": 0.08, "istek_gorus": 0.07, "method": "Heuristic+"}
        # rating 3 veya bilinmiyor → nötr
        return {"olumlu": 0.35, "olumsuz": 0.33, "istek_gorus": 0.32, "method": "Heuristic+"}

    # Rating 1 + negatif keyword varsa → çok güçlü negatif sinyal
    if _rating == 1 and neg_score > 0:
        conf = min(0.96, 0.80 + neg_score * 0.04)
        return {"olumlu": round((1-conf)/2, 3), "olumsuz": round(conf, 3),
                "istek_gorus": round((1-conf)/2, 3), "method": "Heuristic+"}

    # Rating 5 ama içerikte net negatif var → content wins
    if _rating == 5 and neg_score > pos_score and neg_score >= 2:
        conf = min(0.88, 0.65 + neg_score * 0.05)
        return {"olumlu": 0.06, "olumsuz": round(conf, 3),
                "istek_gorus": round(1-conf-0.06, 3), "method": "Heuristic+"}

    # ── 10. NORMAL KARAR ──────────────────────────────────────────────────────
    if neg_score_w > pos_score and neg_score_w >= neu_score:
        conf = min(0.95, 0.68 + (neg_score_w / (pos_score + neg_score_w + neu_score)) * 0.27)
        return {"olumlu": round((1-conf)/2, 3), "olumsuz": round(conf, 3),
                "istek_gorus": round((1-conf)/2, 3), "method": "Heuristic+"}

    if pos_score > neg_score and pos_score >= neu_score:
        conf = min(0.95, 0.68 + (pos_score / (pos_score + neg_score_w + neu_score)) * 0.27)
        return {"olumlu": round(conf, 3), "olumsuz": round((1-conf)/2, 3),
                "istek_gorus": round((1-conf)/2, 3), "method": "Heuristic+"}

    if neu_score >= pos_score and neu_score >= neg_score:
        return {"olumlu": 0.08, "olumsuz": 0.07, "istek_gorus": 0.85, "method": "Heuristic+"}

    return {"olumlu": 0.35, "olumsuz": 0.33, "istek_gorus": 0.32, "method": "Heuristic+"}

def run_bulk_analysis(data_to_process, is_append=False):
    bulk_results = st.session_state.get("bulk_results", []) if is_append else []
    
    time_display = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    ticker_placeholder = st.empty() 
    quota_info = st.empty()
    st.warning("Analiz süresince bu sayfayı kapatmayın veya yenilemeyin. Verileriniz kaybolabilir.")
    st.session_state['_quota_hits'] = 0
            
    analysis_type = st.session_state.get("analysis_type", "Hızlı Analiz")
    mode_idx = st.session_state.get("analysis_mode", 0)
    
    if mode_idx == 0:
        ANALYSIS_MODEL = 'models/gemini-2.0-flash'
        RPM_LIMIT = 500
    else:
        ANALYSIS_MODEL = 'models/gemini-1.5-pro'
        RPM_LIMIT = 300

    start_time = time.time()
    
    
    if analysis_type == "Zengin Analiz":
        MAX_ITEMS = 500
        if len(data_to_process) > MAX_ITEMS:
            st.warning(f"⚠️ Zengin Analiz kotası: en fazla {MAX_ITEMS} yorum işleniyor.")
            data_to_process = data_to_process[:MAX_ITEMS]
    # Hızlı Analiz'de hiçbir üst sınır yok — tüm liste işlenir
    
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
        comment = str(entry.get("text", ""))[:1000] 
        is_valid = entry.get("is_valid", True)
        if not is_valid or not comment or len(comment.strip()) < 2:
            return idx, entry, {"olumlu": 0, "olumsuz": 0, "istek_gorus": 0}, "—", None
        
        current_rating = entry.get("rating")
        if analysis_type == "Hızlı Analiz":
            res_api = heuristic_analysis(comment, rating=current_rating)
            err = None
        else:
            res_api = get_ai_sentiment(comment, model_name=ANALYSIS_MODEL)
            err = None
            if res_api is None or "_error" in res_api:
                err = res_api["_error"] if res_api else "unknown"
                res_api = heuristic_analysis(comment, rating=current_rating)
        
        scores = {"Olumlu": res_api['olumlu'], "Olumsuz": res_api['olumsuz'], "İstek/Görüş": res_api['istek_gorus']}
        verdict = str(max(scores, key=lambda k: scores[k]))
        return idx, entry, res_api, verdict, err

    completed_count = 0
    workers = 10 if mode_idx == 0 else 6
    
    start_offset = len(bulk_results)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = [executor.submit(fetch_sentiment_worker, (i, e)) for i, e in enumerate(data_to_process)]
        
        for future in concurrent.futures.as_completed(tasks):
            i, entry, res, verdict, err = future.result()
            completed_count += 1
            
            progress_bar.progress(completed_count / total_items)
            status_text.text(f"Analiz ediliyor: {completed_count} / {total_items}")
            update_time(completed_count, total_items, start_time)
            
            if err == "quota":
                q = st.session_state.get('_quota_hits', 0) + 1
                st.session_state['_quota_hits'] = q
            elif err == "cost_limit":
                if not st.session_state.get('_cost_warned'):
                    st.error("🚨 Tahmini 50 TL faturaya yaklaşıldı. Analiz işlemi otomatik durduruldu!")
                    st.session_state['_cost_warned'] = True
                break
            elif err:
                st.warning(err)
            
            comment = entry["text"]
            date = entry.get("date")
            ticker_date = f"{date.strftime('%d-%m-%Y')}" if date else ""

            ticker_color = "#34D399" if verdict == "Olumlu" else ("#F87171" if verdict == "Olumsuz" else "#60A5FA")
            analysis_method = res.get("method", "Gemini")
            ticker_placeholder.markdown(f"""
            <div class="header-container" style="border-color: {ticker_color}; text-align: left; margin: 10px 0; width: 100%; box-sizing: border-box;">
                <div style="display: flex; justify-content: space-between; font-size: 0.85em; color: #64748b; margin-bottom: 8px;">
                    <span style="font-weight: 600;">ŞU AN EKLENEN (#{start_offset + i + 1}) — <span style="color: {ticker_color}">{analysis_method}</span></span>
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
    # Hızlı Analiz seçiliyse all_fetched_pool'dan tüm yorumları al
    current_analysis_type = st.session_state.get("analysis_type", "Hızlı Analiz")
    all_pool = st.session_state.get("all_fetched_pool", [])
    
    if current_analysis_type == "Hızlı Analiz" and all_pool:
        # Pool'dan tümünü al, limit yok
        data_for_run = all_pool
        st.session_state.comments_to_analyze = all_pool
    else:
        data_for_run = st.session_state.comments_to_analyze

    if not data_for_run:
        st.warning("Lütfen analiz edilecek bir metin girin veya dosya yükleyin.")
    else:
        run_bulk_analysis(data_for_run)


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
    st.markdown("## Analiz Sonuçları ve İstatistikler")
    
    analysis_df = df[df["Baskın Duygu"] != "—"].copy()
    counts = analysis_df["Baskın Duygu"].value_counts()
    
    
    m_olumlu = counts.get("Olumlu", 0)
    m_olumsuz = counts.get("Olumsuz", 0)
    m_istek = counts.get("İstek/Görüş", 0)
    m_skipped = len(df[df["Baskın Duygu"] == "—"])
    
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
        <div class="metric-card" style="border-style: dashed !important; opacity: 0.7;">
            <div class="metric-value" style="color: #64748b;">{m_skipped}</div>
            <div class="metric-label">Kısa / Atlandı</div>
        </div>
        <div class="metric-card">
            <div class="metric-value" style="color: #a78bfa;">{len(df)}</div>
            <div class="metric-label">Toplam Veri</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_pie, col_summary = st.columns([1, 1])
    
    with col_pie:
        pie_data = pd.DataFrame({"Duygu": counts.index, "Sayı": counts.values})
        
        order_map = {"Olumlu": 1, "Olumsuz": 2, "İstek/Görüş": 3}
        pie_data['order'] = pie_data['Duygu'].map(order_map).fillna(4)
        pie_data = pie_data.sort_values('order')
        
        color_map = {"Olumlu": "#34D399", "Olumsuz": "#FB7185", "İstek/Görüş": "#60A5FA"}
        pie_colors = [color_map.get(d, "#94a3b8") for d in pie_data["Duygu"]]
        
        t_val = m_olumlu + m_olumsuz + m_istek
        pos_pct = int((m_olumlu / t_val) * 100) if t_val > 0 else 0
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=pie_data["Duygu"], 
            values=pie_data["Sayı"],
            hole=0.82,
            marker=dict(
                colors=pie_colors,
                line=dict(color='#F0F9FF', width=6)
            ),
            textinfo='none',
            hoverinfo='label+percent+value',
            direction='clockwise',
            sort=False
        )])
        
        fig_pie.update_layout(
            annotations=[
                dict(
                    text=f"<span style='font-size: 3.5rem; font-weight: 800; color: #1E293B;'>{pos_pct}%</span><br><span style='font-size: 0.95rem; color: #64748B; font-weight: 700; letter-spacing: 1px;'>OLUMLU</span>",
                    x=0.5, y=0.5,
                    showarrow=False,
                    align="center"
                )
            ],
            height=360,
            showlegend=True,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#000000', family="Poppins, sans-serif"),
            legend=dict(
                orientation="h", xanchor="center", x=0.5, y=-0.1,
                font=dict(color="#475569", size=13)
            ),
            margin=dict(t=10, b=10, l=10, r=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True, config={'displayModeBar': False})
        
        total_valid = m_olumlu + m_olumsuz + m_istek
        if total_valid > 0:
            # 2. Genel Deneyim Skoru
            score = int(((m_olumlu * 100) + (m_istek * 50)) / total_valid)
            score_color = "#10b981" if score >= 70 else "#f59e0b" if score >= 40 else "#f43f5e"
            st.markdown(f"""
            <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 15px; margin-top: -10px; margin-bottom: 15px; text-align: center; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; margin-bottom: 5px; text-transform: uppercase; letter-spacing: 1px;">Genel Deneyim Skoru</div>
                <div style="font-size: 2.5rem; font-weight: 800; color: {score_color}; line-height: 1;">{score}<span style="font-size: 1.2rem; color: #94A3B8;">/100</span></div>
            </div>
            """, unsafe_allow_html=True)
            
            # 1. En Kritik Kelimeler
            import collections
            import re
            
            stop_words = set([
                "ve", "bir", "cok", "çok", "icin", "için", "bu", "da", "de", "ile", "ama", "fakat", "gibi", "kadar", "olan", "olarak", "daha", "en", "ki", "ise", "mi", "mu", "hem", "ne", "var", "yok", "sonra", "önce", "böyle", "şöyle", "her", "hic", "hiç", "sadece", "artık", "zaten", "çünkü", "nasıl", "neden", "niye", "bana", "beni", "benim", "sana", "seni", "senin", "ona", "onu", "onun", "bizi", "bize", "bizim", "sizi", "size", "sizin", "onlari", "onlara", "onlarin", "uygulama", "uygulaması", "uygulamada", "program", "iyi", "güzel", "kötü", "berbat", "harika", "mükemmel", "teşekkürler", "teşekkür", "ederim", "oldu", "olur", "olacak", "olmalı", "yapın", "yap", "yaptı", "yapıyor", "yapıldı", "ediyor", "edin", "ettim", "edildi", "geldi", "geliyor", "gitti", "gider", "gidin", "aldı", "alıyor", "alın", "aldım", "verdi", "veriyor", "verin", "verdim", "istiyorum", "istemiyorum", "istiyoruz", "lütfen", "merhaba", "slm", "selam", "gün", "saat", "dakika", "ay", "yıl", "hafta", "kere", "defa", "zaman", "şimdi", "hemen", "göre", "birlikte", "beraber", "ayrıca", "bazen", "bazı", "çoğu", "tüm", "bütün", "hiçbir", "başka", "diğer", "aynı", "kendi", "biri", "biraz", "birkaç", "fazla", "az", "hiçbiri", "öyle", "böylece", "şöylece", "tam", "sanki", "belki", "mutlaka", "kesinlikle", "tabii", "elbette", "aslında", "gerçekten", "sürekli", "tavsiye", "ederiz", "bunu", "şunu", "içinde"
            ])
            
            words = []
            kritik_df = analysis_df[analysis_df["Baskın Duygu"].isin(["Olumsuz", "İstek/Görüş"])]
            for text in kritik_df["Yorum"].astype(str):
                clean_text = re.sub(r'[^\w\s]', ' ', text.lower())
                clean_text = re.sub(r'\d+', ' ', clean_text)
                tokens = clean_text.split()
                filtered = [w for w in tokens if len(w) > 3 and w not in stop_words]
                words.extend(filtered)
            
            if len(words) >= 5:
                counter = collections.Counter(words)
                top_words = counter.most_common(12)
                
                tags_html = ""
                for word, count in top_words:
                    tags_html += f'<span style="display: inline-block; background-color: #F8FAFC; border: 1px solid #E2E8F0; color: #475569; padding: 4px 10px; margin: 3px; border-radius: 20px; font-size: 0.8rem; font-weight: 600;">#{word}</span>'
                
                st.markdown(f"""
                <div style="background-color: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 15px; margin-top: 5px; box-shadow: 0 4px 6px rgba(0,0,0,0.02);">
                    <div style="font-size: 0.85rem; color: #64748B; font-weight: 700; margin-bottom: 10px; text-transform: uppercase; letter-spacing: 1px;">En Kritik Kelimeler</div>
                    <div style="display: flex; flex-wrap: wrap; gap: 2px;">
                        {tags_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

    with col_summary:
        st.write("### Duygu Dağılımı")
        
        total_all = m_olumlu + m_olumsuz + m_istek
        diff_val = abs(m_olumlu - m_olumsuz)
        
        if st.session_state.get("analysis_type") == "Zengin Analiz":
            if "ai_summary_cache" not in st.session_state or st.session_state.get("last_results_len") != len(analysis_df):
                with st.spinner("🤖 Yapay zeka derinlemesine raporu hazırlıyor..."):
                    summary_text = generate_dynamic_summary(analysis_results=st.session_state.bulk_results)
                    st.session_state.ai_summary_cache = summary_text
                    st.session_state.last_results_len = len(analysis_df)
            
            st.markdown(f"""
            <div style="background: #FFFFFF; padding: 25px; border-radius: 12px; border: 2px solid #a78bfa; color: #1e293b; line-height: 1.6; box-shadow: 0 4px 15px rgba(167, 139, 250, 0.1);">
                <div style="font-weight: 800; font-size: 1.3rem; margin-bottom: 15px; color: #7c3aed; display: flex; align-items: center; gap: 10px;">
                    <span>✨ Yapay Zeka Derin Analiz Raporu</span>
                </div>
                <div style="font-size: 0.95rem;">
                    {st.session_state.ai_summary_cache}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            if total_all == 0:
                summary_body = "Analiz edilecek yorumlar geldikçe burası güncellenecektir."
                grad_bg, border_c, summary_title = "#F8FAFC", "#E2E8F0", "Henüz yeterli veri yok."
            elif total_all > 10 and (diff_val / total_all) < 0.15:
                summary_title = "Dengeli/Karmaşık bir kullanıcı deneyimi"
                summary_body = "Uygulama şu anda kullanıcı kitlesini neredeyse tam ortadan ikiye bölmüş durumda. Teknik aksaklıklar ile memnuniyetler başa baş gidiyor."
                grad_bg, border_c = "#fef9c3", "#eab308"
            elif counts.idxmax() == "Olumlu":
                summary_title = "Topluluk genel olarak Olumlu"
                summary_body = "Kullanıcılar uygulamadan genel olarak memnun. Arayüz ve hız beklentileri karşılıyor."
                grad_bg, border_c = "#dcfce7", "#10b981"
            else:
                summary_title = "Dikkat çeken Olumsuz bir eğilim"
                summary_body = "Kullanıcıların kronik teknik şikayetleri veya hizmet aksaklıkları olduğu görülüyor."
                grad_bg, border_c = "#fee2e2", "#f43f5e"

            st.markdown(f"""
            <div style="background: {grad_bg}; padding: 20px; border-radius: 12px; border: 2px solid {border_c}; color: #1e293b; line-height: 1.6;">
                <div style="font-weight: 700; font-size: 1.1rem; margin-bottom: 10px;">{summary_title}</div>
                <div style="font-size: 0.95rem; opacity: 0.9;">{summary_body}</div>
            </div>
            """, unsafe_allow_html=True)
        

    
    if "Puan" in df.columns and df["Puan"].notnull().any():
        st.markdown("---")
        
        
        st.write("### Puan Dağılımı")
        freq = st.radio("Zaman Ölçeği:", ["Günlük", "Haftalık", "Aylık"], index=2, horizontal=True, key="puan_freq_sel", label_visibility="collapsed")
        st.markdown("<div style='margin-bottom: 10px;'></div>", unsafe_allow_html=True)

        df_puan = df.dropna(subset=["Tarih", "Puan"]).copy()
        try:
            
            df_puan["Puan_val"] = pd.to_numeric(df_puan["Puan"], errors='coerce').fillna(0).astype(int)
            df_puan = df_puan[(df_puan["Puan_val"] >= 1) & (df_puan["Puan_val"] <= 5)]
            
            if not df_puan.empty:
                df_puan["Tarih_dt"] = pd.to_datetime(df_puan["Tarih"])
                
                
                min_d = df_puan["Tarih_dt"].min().strftime('%d-%m-%Y')
                max_d = df_puan["Tarih_dt"].max().strftime('%d-%m-%Y')
                st.caption(f"**Tespit Edilen Tarih Aralığı:** {min_d} ile {max_d}")

                
                tr_months = {1:"Ocak", 2:"Şubat", 3:"Mart", 4:"Nisan", 5:"Mayıs", 6:"Haziran", 
                             7:"Temmuz", 8:"Ağustos", 9:"Eylül", 10:"Ekim", 11:"Kasım", 12:"Aralık"}

                
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
                
                
                fig_dist.update_xaxes(type='category', tickangle=-45, tickfont={"color": "#000000"}, title_font={"color": "#000000"})
                fig_dist.update_yaxes(tickfont={"color": "#000000"}, title_font={"color": "#000000"})
                st.plotly_chart(fig_dist, use_container_width=True)
        except Exception as e:
            st.error(f"Grafik oluşturma hatası: {e}")

    
    
    all_pool = st.session_state.get("all_fetched_pool", [])
    if all_pool:
        
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
                analysis_type_now = st.session_state.get("analysis_type", "Hızlı Analiz")
                take_next = min(len(remaining_pool), 500) if analysis_type_now == "Zengin Analiz" else len(remaining_pool)

                label = (f"Sonraki {take_next} yorumu da analiz et" 
                         if analysis_type_now == "Zengin Analiz" 
                         else f"Kalan tüm {take_next} yorumu analiz et")

                if st.button(label, use_container_width=True):
                    next_batch = remaining_pool[:take_next]
                    run_bulk_analysis(next_batch, is_append=True)

    
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
            else: 
                df_dates['Grup'] = df_dates['Tarih'].dt.date
                xaxis_title = "Tarih (Günlük)"
                chart_title_prefix = "Günlük"

            trend_data = df_dates.groupby(['Grup', "Baskın Duygu"]).size().reset_index(name='Adet')
            
            
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
            
            
            selection = st.plotly_chart(fig_trend, use_container_width=True, on_select="rerun", key=f"chart_{key}")
            
            if selection and "selection" in selection and selection["selection"]["points"]:
                point = selection["selection"]["points"][0]
                
                sel_grup_str = point["customdata"][0]
                sel_grup = pd.to_datetime(sel_grup_str).tz_localize(None)
                sel_sentiment = str(point["customdata"][1]).strip()
                
                
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
            
            
            extra_info = ""
            if "Puan" in row and pd.notnull(row["Puan"]):
                extra_info += f" | Puan: {row['Puan']}"
            
            date_tag = ""
            if "Tarih" in row and pd.notnull(row["Tarih"]):
                try: 
                    d = pd.to_datetime(row["Tarih"])
                    date_tag = f"Tarih: {d.strftime('%d-%m-%Y')}"
                except: pass

            
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

    
    st.write("### Yorum Listesi")
    
    
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

    
    try:
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Analiz Sonuçları')
        
        
        st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
        st.subheader("Analiz Raporunu Paylaş")
        
        
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

        
        app_name = urllib.parse.unquote(st.session_state.get('detected_app_name', "Uygulama"))
        store_type = st.session_state.get('detected_store_type', "STORE")
        report_title = f"{app_name.upper()} {store_type.upper()} ANALİZ RAPORU"
        excel_filename = f"{app_name}_yorum_analizi.xlsx".replace(" ", "_").lower()

        
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

        
        def clean_html(h):
            return "\n".join([line.strip() for line in h.split('\n') if line.strip()])
        
        
        display_summary = st.session_state.get('ai_summary', 'Analiz özeti hazırlanıyor...')
        display_summary = display_summary.replace("`", "").replace("*", "").replace("#", "")
        
        
        import re
        
        display_summary = re.sub(r'\s+([.,;:!?])', r'\1', display_summary)
        
        display_summary = re.sub(r' {2,}', ' ', display_summary)
        
        display_summary = display_summary.replace('\n', '<br>')

        card_html = clean_html(f"""
            <div id="nlp-report-card" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 20px; padding: 35px; margin: 20px auto; box-shadow: 0 15px 35px rgba(0,0,0,0.08); font-family: 'Poppins', sans-serif; color: #1E293B; max-width: 600px; position: relative; overflow: hidden;">
                <style>
                    @media (max-width: 480px) {{
                        #nlp-report-card {{ padding: 20px !important; margin: 10px auto !important; }}
                        .metric-row {{ flex-wrap: wrap !important; gap: 8px !important; }}
                        .metric-box {{ flex: 1 1 40% !important; padding: 8px !important; }}
                        .chart-container {{ padding: 15px !important; flex-direction: column !important; gap: 20px !important; }}
                        .chart-svg-box {{ width: 100px !important; height: 100px !important; }}
                    }}
                </style>
                <div style="text-align: center; border-bottom: 2px solid #F1F5F9; padding-bottom: 15px; margin-bottom: 25px;">
                    <h2 style="margin: 0; color: #0F172A; font-size: 1.3rem; font-weight: 700;">{report_title}</h2>
                </div>
                
                <div class="metric-row" style="display: flex; justify-content: space-between; margin-bottom: 35px; gap: 10px;">
                    <div class="metric-box" style="text-align: center; flex: 1; background: #F8FAFC; padding: 12px; border-radius: 12px;">
                        <div style="font-size: 0.65rem; color: #64748B; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Analiz</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #334155;">{total_q}</div>
                    </div>
                    <div class="metric-box" style="text-align: center; flex: 1; background: #ECFDF5; padding: 12px; border-radius: 12px; border: 1px solid #D1FAE5;">
                        <div style="font-size: 0.65rem; color: #059669; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Olumlu</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #059669;">{t_pos}</div>
                    </div>
                    <div class="metric-box" style="text-align: center; flex: 1; background: #FEF2F2; padding: 12px; border-radius: 12px; border: 1px solid #FEE2E2;">
                        <div style="font-size: 0.65rem; color: #DC2626; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Olumsuz</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #DC2626;">{t_neg}</div>
                    </div>
                    <div class="metric-box" style="text-align: center; flex: 1; background: #EFF6FF; padding: 12px; border-radius: 12px; border: 1px solid #DBEAFE;">
                        <div style="font-size: 0.65rem; color: #2563EB; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Görüş</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #2563EB;">{t_neu}</div>
                    </div>
                </div>

                <div class="chart-container" style="display: flex; align-items: center; justify-content: space-around; background: #F8FAFC; border-radius: 20px; padding: 30px; margin-bottom: 25px;">
                    <div class="chart-svg-box" style="width: 140px; height: 140px; position: relative;">
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
                        Stratejik Özet
                    </div>
                    <div style="color: #475569; font-size: 0.9rem; line-height: 1.6; font-weight: 500;">
                        {display_summary}
                    </div>
                </div>
                <div style="margin-top: 30px; text-align: center; color: #94A3B8; font-size: 0.75rem; font-weight: 600; letter-spacing: 0.5px;">
                    AI SENTIMENT INTELLIGENCE
                </div>
            </div>
        """)
        st.markdown(card_html, unsafe_allow_html=True)
        st.info("Yukarıdaki kartı kopyalayabilir veya doğrudan paylaşabilirsiniz.")

        image_name = f"{app_name} ai sentiment report.png".replace(" ", "_").replace(":", "_")
        
        components.html(f"""
            <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
            <script>
                if (!window.parent.document.getElementById('uNotif')) {{
                    const div = window.parent.document.createElement('div');
                    div.id = 'uNotif';
                    div.innerHTML = '<span id="uMsg">Hazırlanıyor...</span>';
                    Object.assign(div.style, {{
                        position: 'fixed', top: '20px', left: '50%', transform: 'translateX(-50%) translateY(-20px) scale(0.9)',
                        background: '#10B981', color: 'white', padding: '14px 28px', borderRadius: '12px', fontWeight: '700',
                        opacity: '0', transition: 'all 0.4s cubic-bezier(0.19, 1, 0.22, 1)', zIndex: '9999999',
                        boxShadow: '0 15px 30px rgba(16, 185, 129, 0.4)', display: 'flex', alignItems: 'center', gap: '10px',
                        pointerEvents: 'none', fontFamily: '"Poppins", sans-serif', whiteSpace: 'nowrap'
                    }});
                    window.parent.document.body.appendChild(div);
                }}

                window.notifyBridge = function(msg, duration = 3000) {{
                    const n = window.parent.document.getElementById('uNotif');
                    const m = window.parent.document.getElementById('uMsg');
                    if(n && m) {{
                        m.innerText = msg; n.style.opacity = '1';
                        n.style.transform = 'translateX(-50%) translateY(0) scale(1)';
                        setTimeout(() => {{ 
                            n.style.opacity = '0';
                            n.style.transform = 'translateX(-50%) translateY(-20px) scale(0.9)';
                        }}, duration);
                    }}
                }};
                
                setInterval(function() {{
                    const btn = window.parent.document.getElementById('btn-png-download');
                    if (btn && !btn.hasAttribute('data-bound')) {{
                        btn.setAttribute('data-bound', 'true');
                        btn.addEventListener('click', function() {{
                            const target = window.parent.document.getElementById('nlp-report-card');
                            if(!target) return;
                            
                            window.notifyBridge("Görsel Hazırlanıyor... ⏳", 5000);
                            
                            window.html2canvas(target, {{ 
                                scale: 2, 
                                useCORS: true, 
                                backgroundColor: '#FFFFFF', 
                                logging: false,
                                allowTaint: true
                            }}).then(canvas => {{
                                const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
                                const dataUrl = canvas.toDataURL('image/png');

                                if (isIOS) {{
                                    const newTab = window.open();
                                    if (newTab) {{
                                        newTab.document.write('<img src="' + dataUrl + '" style="width:100%; height:auto;">');
                                        newTab.document.title = "Analiz Raporu - Kaydetmek icin Basılı Tut";
                                        window.notifyBridge("Görsel açıldı! Kaydetmek için üzerine basılı tutun. ⬇️");
                                    }} else {{
                                        const link = document.createElement('a');
                                        link.href = dataUrl;
                                        link.download = "{image_name}";
                                        link.click();
                                        window.notifyBridge("İndirme Başlatıldı! ⬇️");
                                    }}
                                }} else {{
                                    const link = document.createElement('a');
                                    link.href = dataUrl;
                                    link.download = "{image_name}";
                                    link.click();
                                    window.notifyBridge("İndirme Başlatıldı! ⬇️");
                                }}
                            }}).catch(e => {{
                                console.error(e);
                                window.notifyBridge("Hata oluştu! ❌");
                            }});
                        }});
                    }}
                }}, 1000);
            </script>
        """, height=0)

        
        share_ui = textwrap.dedent(f"""
            <style>
                @import url('https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css');
                .u-tray {{ display: flex; flex-wrap: wrap; gap: 12px; justify-content: center; margin-bottom: 10px; }}
                .u-btn {{
                    width: 48px; height: 48px; background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px;
                    display: flex; align-items: center; justify-content: center; font-size: 1.4rem; cursor: pointer;
                    transition: all 0.2s ease; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-decoration: none !important;
                }}
                .u-btn:hover {{ transform: translateY(-3px); box-shadow: 0 8px 15px rgba(0,0,0,0.1); border-color: #CBD5E1; }}
                .u-wa {{ color: #25D366 !important; }} 
                .u-li {{ color: #0077B5 !important; }} 
                .u-x {{ color: #000000 !important; }}
                .u-tg {{ color: #24A1DE !important; }} 
                .u-mail {{
                    background: conic-gradient(from 180deg at 50% 50%, #ea4335 0deg, #ea4335 90deg, #fbbc04 90deg, #fbbc04 180deg, #34a853 180deg, #34a853 270deg, #4285f4 270deg, #4285f4 360deg);
                    -webkit-background-clip: text;
                    -webkit-text-fill-color: transparent;
                }}
                
                .dl-main-btn {{
                    width: 100%; max-width: 600px; margin: 0 auto; min-width: 280px; min-height: 50px; background: #5a67d8; color: #FFFFFF; 
                    border: none; border-radius: 12px; cursor: pointer; font-size: 0.95rem; font-weight: 600; 
                    box-shadow: 0 4px 12px rgba(90, 103, 216, 0.3); transition: all 0.2s;
                    display: flex; align-items: center; justify-content: center; gap: 8px; font-family: 'Poppins', sans-serif;
                }}
                .dl-main-btn:hover {{ background: #4c51bf; transform: translateY(-1px); box-shadow: 0 6px 15px rgba(90, 103, 216, 0.4); }}
                
                div[data-testid="stDownloadButton"] button {{
                    background-color: #5CB85C !important;
                    color: white !important;
                    border: none !important;
                    border-radius: 12px !important;
                    height: 50px !important;
                    font-weight: 600 !important;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
                    transition: all 0.2s !important;
                }}
                div[data-testid="stDownloadButton"] button:hover {{
                    background-color: #4cae4c !important;
                    transform: translateY(-1px);
                    box-shadow: 0 6px 10px rgba(0,0,0,0.15) !important;
                }}
                div[data-testid="stDownloadButton"] button p {{
                    color: white !important;
                }}
            </style>

            <div class="u-tray">
                <a href="https://api.whatsapp.com/send?text={encoded_text}" target="_blank" class="u-btn u-wa"><i class="fa-brands fa-whatsapp"></i></a>
                <a href="https://www.linkedin.com/sharing/share-offsite/?url=https://cem-evecen.com&summary={encoded_text}" target="_blank" class="u-btn u-li"><i class="fa-brands fa-linkedin-in"></i></a>
                <a href="https://twitter.com/intent/tweet?text={encoded_text}" target="_blank" class="u-btn u-x"><i class="fa-brands fa-x-twitter"></i></a>
                <a href="https://t.me/share/url?url=https://cem-evecen.com&text={encoded_text}" target="_blank" class="u-btn u-tg"><i class="fa-brands fa-telegram"></i></a>
                <a href="mailto:?subject=NLP Analiz Raporu&body={encoded_text}" class="u-btn u-mail"><i class="fa-solid fa-envelope"></i></a>
            </div>
        """).strip()
        st.markdown(share_ui, unsafe_allow_html=True)

        btn_cols = st.columns(3)
        with btn_cols[0]:
            st.markdown("""
                <button id="btn-png-download" style="width: 100%; height: 50px; background: #5a67d8; color: white; border: none; border-radius: 12px; cursor: pointer; font-size: 1.1rem; font-weight: 600; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-family: 'Poppins', sans-serif; display: flex; align-items: center; justify-content: center; gap: 8px; transition: all 0.2s;">
                    📷 PNG
                </button>
            """, unsafe_allow_html=True)
        with btn_cols[1]:
            st.download_button("EXCEL", output.getvalue(), excel_filename, key="xl_dl", use_container_width=True)
        with btn_cols[2]:
            components.html(f"""
                <style>
                    body {{ margin: 0; padding: 0; overflow: hidden; font-family: sans-serif; }}
                    button:hover {{ filter: brightness(0.9); transform: translateY(-1px); }}
                </style>
                <button onclick='window.parent.print()' style='width: 100%; height: 50px; background: #F4A261; color: white; border: none; border-radius: 12px; cursor: pointer; font-size: 1.1rem; font-weight: 600; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-family: "Poppins", sans-serif; display: flex; align-items: center; justify-content: center; gap: 8px; transition: all 0.2s;'>
                    🖨️ PDF
                </button>
            """, height=48)
                    
    except Exception as e:
        st.error(f"Paylaşım sistemi hatası: {e}")



st.divider()
st.caption("Geliştiren: ivicin")

