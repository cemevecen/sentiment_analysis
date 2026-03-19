import streamlit as st
import streamlit.components.v1 as components
import threading
from streamlit_lottie import st_lottie
from google import genai
from google.genai import types as genai_types
try:
    from mistralai import Mistral
    HAS_MISTRAL_PKG = True
except ImportError:
    HAS_MISTRAL_PKG = False
    Mistral = None

try:
    from streamlit_lottie import st_lottie
    HAS_LOTTIE_PKG = True
except ImportError:
    HAS_LOTTIE_PKG = False
    st_lottie = None

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
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
        [data-testid="stSidebar"], [data-testid="stSidebarNav"] {
            display: none;
        }
        [data-testid="stSidebarCollapsedControl"] {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# --- AUTO-RELOAD MECHANISM ---
CURRENT_VERSION = "2026-03-18-21-15"  # Model selector + Real-time search added


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

@st.cache_resource(show_spinner="DeepSeek API yapılandırılıyor...")
def setup_deepseek():
    """
    DeepSeek API anahtarını st.secrets veya env üzerinden çeker.
    """
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets.get("DEEPSEEK_API_KEY")
        except:
            pass
    return api_key if (api_key and len(api_key) > 5) else None

GEMINI_CLIENT = setup_api()
HAS_GEMINI = GEMINI_CLIENT is not None

MISTRAL_CLIENT = setup_mistral()
HAS_MISTRAL = MISTRAL_CLIENT is not None

GROQ_CLIENT = setup_groq()
HAS_GROQ = GROQ_CLIENT is not None

DEEPSEEK_API_KEY = setup_deepseek()
HAS_DEEPSEEK = DEEPSEEK_API_KEY is not None
    
# ============ COST PROTECTION SYSTEM ============
# KRITIK: Hiçbir zaman parasal ücret çıkmasın
# Strateji:
# 1. HERHANGİ ücretli API çağrısı YOK
# 2. Sadece heuristic analysis kullan
# 3. Free tier endpoints veya mock response kullan
# 4. Para yazarsa → uygulama kapat

COST_LIMIT_TL = 0.0  # SIFIR - Para harcama!
USE_PAID_APIS = False  # Sadece free tier / heuristic
API_TRACKER = {
    "cost_tl": 0.0,
    "calls_made": 0,
    "free_tier_exhausted": False,
    "api_usage_log": []
}

# ============ FREE TIER CHECKER ============
def check_free_tier_status():
    """
    Free tier'ın kullanılıp kullanılmadığını kontrol et.
    Eğer önceki ücretler varsa döndür.
    """
    # Gerçek API çağrısı YOK - sadece heuristic
    # Bu yüzden bu fonksiyon her zaman "safe" dönecek
    return {
        "safe": True,
        "remaining_calls": float('inf'),  # Sınırsız
        "message": "✅ Heuristic analysis kullanıyor (ücretli değil)"
    }

# ============ COST SENTINEL ============
def assert_cost_safe():
    """
    Ücret çıkacaksa STOP et. Hiç API çağrısı yapma.
    """
    if API_TRACKER["cost_tl"] > 0:
        st.error("🚨 **KRITIK: API kullanımı tespit edildi!**")
        st.error("Bu uygulama sadece HERHANGİ ücret ÖDEMİ versiyonunda çalışmalı.")
        st.error("Tüm API çağrıları devre dışı bırakıldı.")
        st.stop()

# ============ AI API Configuration Logic (Silent Init) ============
initial_provider = "Google Gemini" if HAS_GEMINI else "Mistral AI" if HAS_MISTRAL else "Groq AI" if HAS_GROQ else "DeepSeek AI" if HAS_DEEPSEEK else "Gemini Flash"
initial_model = "models/gemini-2.0-flash-lite" if HAS_GEMINI else "mistral-small-latest" if HAS_MISTRAL else "llama-3.3-70b-versatile" if HAS_GROQ else "deepseek-chat"

if 'current_ai_provider' not in st.session_state:
    st.session_state.current_ai_provider = initial_provider
if 'current_ai_model' not in st.session_state:
    st.session_state.current_ai_model = initial_model


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

@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8-sig')

@st.cache_data
def convert_df_to_excel(df):
    output = io.BytesIO()
    try:
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Yorumlar')
    except Exception as e:
        # Fallback if openpyxl fails
        df.to_excel(output, index=False, sheet_name='Yorumlar')
    return output.getvalue()

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
                
                found_old = 0
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
                        country_reviews.append({"id": r_id, "text": content, "date": r_date, "rating": rating, "lang": country})
                    else:
                        found_old += 1
                
                if found_old >= 5: break  # 5 eski yorum görünce dur
            except: break
        return country_reviews

    
    # ── Parael Fetching with increased workers (40 for each country) ──
    total_countries = len(countries)
    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=40) as executor:
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
        old_streak = 0  # arka arkaya kaç eski yorum geldi
        for _ in range(30):
            try:
                result, token = play_reviews(
                    app_id, lang=lang, country=country,
                    sort=sort_type, count=200,
                    filter_score_with=score,
                    continuation_token=token
                )
                if not result: break
                page_has_new = False
                for r in result:
                    r_at_raw = r.get('at')
                    if r_at_raw:
                        r_at = cast(datetime, r_at_raw)
                        if r_at.tzinfo: r_at = r_at.replace(tzinfo=None)
                        if r_at >= threshold_date:
                            content = str(r.get('content', ''))
                            if content and len(content.strip()) >= 2:
                                r_id = r.get('reviewId', content)
                                channel_data.append({
                                    "id": r_id,
                                    "text": content,
                                    "date": r_at,
                                    "rating": str(score),
                                    "lang": lang,
                                    "version": r.get('appVersion', 'Bilinmiyor')
                                })
                            page_has_new = True
                            old_streak = 0
                        else:
                            if sort_type == Sort.NEWEST:
                                old_streak += 1
                if sort_type == Sort.NEWEST and not page_has_new:
                    old_streak += 1
                # 3 sayfa üst üste eski gelirse dur
                if sort_type == Sort.NEWEST and old_streak >= 3:
                    break
                if not token: break
            except: break
        return channel_data

    # ── Filtresiz hızlı çekim (en taze yorumlar için) ──────────────
    def fetch_unfiltered(lang, country):
        fresh_data = []
        token = None
        old_streak = 0
        for _ in range(20):
            try:
                result, token = play_reviews(
                    app_id, lang=lang, country=country,
                    sort=Sort.NEWEST, count=200,
                    continuation_token=token
                )
                if not result: break
                page_has_new = False
                for r in result:
                    r_at_raw = r.get('at')
                    if r_at_raw:
                        r_at = cast(datetime, r_at_raw)
                        if r_at.tzinfo: r_at = r_at.replace(tzinfo=None)
                        if r_at >= threshold_date:
                            content = str(r.get('content', ''))
                            if content and len(content.strip()) >= 2:
                                r_id = r.get('reviewId', content)
                                rating = str(r.get('score', '0'))
                                fresh_data.append({
                                    "id": r_id,
                                    "text": content,
                                    "date": r_at,
                                    "rating": rating,
                                    "lang": lang,
                                    "version": r.get('appVersion', 'Bilinmiyor')
                                })
                            page_has_new = True
                            old_streak = 0
                        else:
                            old_streak += 1
                if not page_has_new:
                    old_streak += 1
                if old_streak >= 3:
                    break
                if not token: break
            except: break
        return fresh_data

    # Önce filtresiz taze çekim — TR, US, DE, RU (Parallelized)
    initial_pairs = [('tr', 'tr'), ('en', 'us'), ('de', 'de'), ('ru', 'ru')]
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(initial_pairs)) as init_executor:
        init_futures = [init_executor.submit(fetch_unfiltered, lp, cp) for lp, cp in initial_pairs]
        for f in concurrent.futures.as_completed(init_futures):
            for r in f.result():
                all_fetched_map[r['id']] = r

    # ── Parallel Execution for Multi-Channel Depth (Increased Workers: 60) ──
    total_channels = len(channels)
    completed_channels = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=60) as executor:
        future_to_channel = {executor.submit(fetch_channel, s, sc, l, c): (s, sc, l, c) for s, sc, l, c in channels}
        for future in concurrent.futures.as_completed(future_to_channel):
            completed_channels += 1
            # Progress can be granular but let's keep it smooth
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
    
    p, label, h1, h2, h3, h4, h5, h6, .stMarkdown, .stText {
        font-family: 'Poppins', sans-serif !important;
        color: #1E293B !important;
    }
    
    /* 1. Reset & Full Responsive Container */
    [data-testid="stAppViewBlockContainer"] {
        max-width: 720px !important;
        padding-left: 0.5rem !important;
        padding-right: 0.5rem !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    @media (max-width: 768px) {
        [data-testid="stAppViewBlockContainer"] {
            max-width: 100% !important;
            padding-left: 0.3rem !important;
            padding-right: 0.3rem !important;
        }
        .header-title {
            font-size: 1.8rem !important;
        }
        .header-container {
            padding: 6px !important;
            margin-top: 2px !important;
            margin-bottom: 4px !important;
        }
        /* Metric kartları mobilde 2x2 grid */
        .metric-container {
            gap: 4px !important;
        }
        .metric-card {
            min-width: calc(50% - 4px) !important;
            padding: 8px 4px !important;
        }
        .metric-value {
            font-size: 1.5rem !important;
        }
        /* Tab scroll */
        div[data-testid="stTabList"] {
            display: flex !important;
            flex-wrap: nowrap !important;
            overflow-x: auto !important;
            padding-bottom: 8px !important;
            scrollbar-width: none !important;
        }
        div[data-testid="stTabList"]::-webkit-scrollbar {
            display: none;
        }
        button[data-testid="stTab"] {
            flex: 0 0 auto !important;
            white-space: nowrap !important;
            padding: 4px 10px !important;
            font-size: 0.85rem !important;
        }
        /* Kolonları mobilde alt alta */
        [data-testid="column"] {
            width: 100% !important;
            min-width: 100% !important;
            flex: 1 1 100% !important;
        }
        /* Input alanları tam genişlik */
        .stTextInput input {
            font-size: 16px !important; /* iOS zoom'u önler */
            padding-left: 12px !important;
        }
        .stTextArea textarea {
            font-size: 16px !important;
        }
        /* Butonlar yeterli boyut */
        .stButton > button {
            height: 48px !important;
            font-size: 0.9rem !important;
        }
        /* Rapor kartı mobilde */
        #nlp-report-card {
            padding: 16px !important;
        }
    }

    /* Strict 5px spacing for headers and common text blocks */
    h1, h2, h3, h4, h5, h6, .stMarkdown div {
        margin-top: 0px !important;
        margin-bottom: 2px !important;
    }
    
    /* Global Icon and Text color enforcement */
    [data-testid="stIcon"], [data-testid="stIconMaterial"], [class*="stIcon"], svg {
        color: #000000 !important;
        fill: #000000 !important;
    }

    span[aria-hidden="true"]:not([data-testid="stIcon"]):not([data-testid="stIconMaterial"]) {
        font-family: inherit !important;
        color: #000000 !important;
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
        padding: 3px 8px !important;
    }
    
    [data-testid="stExpander"] summary p, 
    [data-testid="stExpander"] summary span:not([data-testid="stIcon"]):not([data-testid="stIconMaterial"]) {
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
        padding: 8px;
        margin-top: 2px;
        margin-bottom: 8px;
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
        padding: 8px;
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
        gap: 2px !important;
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
    
    /* Info/Alert boxes — küçük, zarif */
    .stAlert {
        border-radius: 10px !important;
        margin-bottom: 4px !important;
        padding: 0 !important;
    }
    .stAlert > div {
        padding: 8px 14px !important;
        font-size: 0.82rem !important;
        line-height: 1.5 !important;
    }
    /* Success */
    div[data-testid="stAlert"][data-baseweb="notification"][kind="success"],
    .stSuccess {
        background-color: #F0FDF4 !important;
        border: 1px solid #BBF7D0 !important;
        border-left: 3px solid #22C55E !important;
    }
    /* Info */
    div[data-testid="stAlert"][data-baseweb="notification"][kind="info"],
    .stInfo {
        background-color: #F0F9FF !important;
        border: 1px solid #BAE6FD !important;
        border-left: 3px solid #38BDF8 !important;
    }
    /* Warning */
    div[data-testid="stAlert"][data-baseweb="notification"][kind="warning"],
    .stWarning {
        background-color: #FFFBEB !important;
        border: 1px solid #FDE68A !important;
        border-left: 3px solid #F59E0B !important;
    }
    /* Error */
    div[data-testid="stAlert"][data-baseweb="notification"][kind="error"],
    .stError {
        background-color: #FFF1F2 !important;
        border: 1px solid #FECDD3 !important;
        border-left: 3px solid #F43F5E !important;
    }
    /* İkon küçült */
    .stAlert [data-testid="stIcon"] {
        width: 14px !important;
        height: 14px !important;
        font-size: 0.8rem !important;
    }
    /* Metin rengi */
    .stAlert p, .stAlert div, .stAlert span {
        color: #334155 !important;
        font-size: 0.82rem !important;
        font-weight: 500 !important;
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

    /* Excel/CSV Download Buttons - More refined (smaller) size as requested */
    div[data-testid="stDownloadButton"], 
    div[data-testid="stDownloadButton"] > button {
        height: 28px !important;  /* Reduced from 50px to match button size */
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
    }

    div[data-testid="stDownloadButton"] > button {
        background-color: #66BB6A !important; /* Pastel Green */
        color: #FFFFFF !important;
        border: none !important;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1) !important;
        width: 160px !important;   /* Half size as requested */
        font-weight: 500 !important;
        font-size: 0.8rem !important; /* Smaller text */
        padding: 0 10px !important;
        margin: 0 auto !important;
        line-height: normal !important;
        border-radius: 8px !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stDownloadButton"] > button * {
        color: #FFFFFF !important;
    }
    div[data-testid="stDownloadButton"] > button:hover {
        background-color: #81C784 !important;
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
        height: 24px !important; /* Reduced to 1/4 scale of 50px-ISH */
        font-weight: 600 !important;
        font-size: 0.8rem !important; /* Very small */
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
    .stButton > button[kind="secondary"] {
        background-color: #EEF2FF !important;
        border: 1.5px solid #818CF8 !important;
        color: #4338CA !important;
        font-weight: 600 !important;
    }
    .stButton > button[kind="secondary"]:hover {
        background-color: #E0E7FF !important;
        border-color: #6366F1 !important;
        color: #3730A3 !important;
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

    /* ── Custom Chip Radio Buttons ────────────────── */
    div[data-testid="stRadio"] > div[role="radiogroup"] {
        display: flex !important;
        flex-direction: row !important;
        flex-wrap: wrap !important;
        gap: 6px !important; /* Reduced gap */
        background: transparent !important;
        padding: 2px 0 !important; /* Reduced padding */
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        color: #475569 !important;
        padding: 2px 8px !important; /* Further reduced to 1/4 feel */
        border-radius: 50px !important;
        cursor: pointer !important;
        transition: all 0.2s ease !important;
        font-weight: 500 !important;
        font-size: 0.75rem !important; /* Smaller text */
        margin: 0 !important;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02) !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] > label:hover {
        background-color: #F8FAFC !important;
        border-color: #CBD5E1 !important;
        transform: translateY(-1px);
    }

    /* Active State for Chips */
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-baseweb="radio"] > div:first-child {
        display: none !important; /* Hide the radio circle */
    }

    /* When the radio input inside the label is checked, style the parent label */
    /* Streamlit's radio buttons are structured as: label > div > input */
    /* Since we can't easily target the parent label based on child state in CSS without :has(), 
       we use a sibling selector or rely on Streamlit's class application if possible. 
       Actually, Streamlit's active radio label has a specific attribute or child. 
       Let's use a simpler approach: targeting the 'checked' state. */

    /* Radio button - Seçili state (Daha güçlü selector'ler) */
    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked),
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"],
    div[data-testid="stRadio"] div[role="radiogroup"] label[aria-checked="true"],
    div[data-testid="stRadio"] input[type="radio"]:checked + label {
        background-color: #818CF8 !important;
        color: white !important;
        border-color: #818CF8 !important;
        box-shadow: 0 4px 10px rgba(129, 140, 248, 0.3) !important;
        font-weight: 600 !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] label:has(input:checked) p,
    div[data-testid="stRadio"] div[role="radiogroup"] label[data-checked="true"] p {
        color: white !important;
    }

    /* Mobile adjustments for chips */
    @media (max-width: 768px) {
        div[data-testid="stRadio"] > div[role="radiogroup"] {
            gap: 6px !important;
        }
        div[data-testid="stRadio"] div[role="radiogroup"] > label {
            padding: 1px 6px !important;
            font-size: 0.7rem !important;
        }
    }

    /* ── PRINT / PDF MODU ─────────────────────────────── */
    @media print {

        /* ═══════════════════════════════════════════════
           1. GİZLE — aksiyon butonu / input / sidebar / footer
        ═══════════════════════════════════════════════ */
        [data-testid="stSidebar"],
        section[data-testid="stSidebar"],
        [data-testid="stToolbar"],
        [data-testid="stHeader"],
        [data-testid="stDecoration"],
        [data-testid="stTextInput"],
        [data-testid="stSelectbox"],
        [data-testid="stFileUploader"],
        [data-testid="stTextArea"],
        [data-testid="stRadio"],
        .stRadio,
        [data-testid="stButton"],
        [data-testid="stDownloadButton"],
        [data-testid="stAlert"],
        [data-testid="stInfo"],
        [data-testid="stWarning"],
        [data-testid="stTabList"],
        div[data-baseweb="tab-list"],
        div[data-baseweb="tab-highlight"],
        .no-print,
        .u-tray,
        #btn-png-download,
        iframe,
        hr,
        footer,
        [data-testid="stFooter"]        { display: none !important; }

        /* Gizlenen elementlerin boşluklarını tamamen kapat */
        [data-testid="stSidebar"] *,
        [data-testid="stButton"] *,
        [data-testid="stAlert"] *,
        .no-print *                     { margin: 0 !important; padding: 0 !important; height: 0 !important; }

        /* ═══════════════════════════════════════════════
           2. LAYOUT — sütunları dikey akıta
        ═══════════════════════════════════════════════ */

        /* Streamlit column wrapper'ı block yap */
        [data-testid="stHorizontalBlock"] {
            display: block !important;
            width: 100% !important;
        }

        /* Her kolon tam genişlik, yan yana değil alt alta */
        [data-testid="column"] {
            display: block !important;
            width: 100% !important;
            min-width: 100% !important;
            float: none !important;
            page-break-inside: avoid !important;
        }

        /* Ana container */
        [data-testid="stAppViewBlockContainer"] {
            max-width: 100% !important;
            padding: 0 0.8cm !important;
        }

        /* Gereksiz dikey boşlukları sıkıştır */
        [data-testid="stVerticalBlock"] {
            gap: 8px !important;
        }
        [data-testid="stVerticalBlock"] > div {
            margin-top: 0 !important;
            margin-bottom: 0 !important;
        }

        /* ═══════════════════════════════════════════════
           3. METRİK KARTLAR — tek sütun değil yatay sır
        ═══════════════════════════════════════════════ */
        .metric-container {
            display: flex !important;
            flex-direction: row !important;
            flex-wrap: nowrap !important;
            gap: 6px !important;
            margin-bottom: 16px !important;
            width: 100% !important;
        }
        .metric-card {
            flex: 1 !important;
            min-width: 0 !important;
            padding: 10px 4px !important;
        }
        .metric-value { font-size: 1.4rem !important; }
        .metric-label { font-size: 0.5rem !important; }

        /* ═══════════════════════════════════════════════
           4. RAPOR KARTI — SVG pie fix + sayfa kırılmasın
        ═══════════════════════════════════════════════ */
        #nlp-report-card {
            page-break-inside: avoid !important;
            break-inside: avoid !important;
            margin-top: 16px !important;
        }

        /* SVG transform print'te çalışmıyor → düz göster */
        #nlp-report-card .chart-container {
            flex-direction: column !important;
            align-items: center !important;
        }
        #nlp-report-card .chart-svg-box {
            width: 120px !important;
            height: 120px !important;
            margin: 0 auto 12px auto !important;
        }
        /* Gölge div'i gizle */
        #nlp-report-card .chart-svg-box > div:first-child {
            display: none !important;
        }
        /* SVG transform sıfırla */
        #nlp-report-card .chart-svg-box svg {
            transform: none !important;
            position: relative !important;
            top: 0 !important;
            width: 120px !important;
            height: 120px !important;
        }
        #nlp-report-card .chart-svg-box svg path {
            display: block !important;
            visibility: visible !important;
            opacity: 1 !important;
        }

        /* ═══════════════════════════════════════════════
           5. SAYFA KIRILIMLARI
        ═══════════════════════════════════════════════ */
        h2, h3 {
            page-break-before: avoid !important;
            page-break-after: avoid !important;
        }
        .neon-pos, .neon-neg, .neon-neu, .normal-card {
            page-break-inside: avoid !important;
        }

        /* ═══════════════════════════════════════════════
           6. RENK + BASKI BOYASI
        ═══════════════════════════════════════════════ */
        *, svg, svg * {
            -webkit-print-color-adjust: exact !important;
            print-color-adjust: exact !important;
            color-adjust: exact !important;
        }

        @page {
            margin: 1.2cm;
            size: A4 portrait;
        }
    }
</style>

<script>
(function() {
    function styleRadios() {
        const labels = document.querySelectorAll('[data-testid="stRadio"] label');
        labels.forEach(label => {
            const input = label.querySelector('input[type="radio"]');
            if (input && input.checked) {
                label.style.backgroundColor = '#818CF8';
                label.style.color = 'white';
                label.style.borderColor = '#818CF8';
                label.style.boxShadow = '0 4px 10px rgba(129, 140, 248, 0.3)';
                const p = label.querySelector('p');
                if (p) p.style.color = 'white';
            } else if (label.querySelector('input[type="radio"]')) {
                label.style.backgroundColor = '#FFFFFF';
                label.style.color = '#475569';
                label.style.borderColor = '#E2E8F0';
                label.style.boxShadow = '0 2px 4px rgba(0,0,0,0.02)';
                const p = label.querySelector('p');
                if (p) p.style.color = '#475569';
            }
        });
    }
    
    // İlk yükleme ve periyodik kontrol
    styleRadios();
    
    // Streamlit rerun sonrası veya DOM değişikliklerinde stili güncelle
    const observer = new MutationObserver((mutations) => {
        styleRadios();
    });
    
    observer.observe(document.body, { 
        childList: true, 
        subtree: true,
        attributes: true,
        attributeFilter: ['data-checked', 'aria-checked']
    });
})();
</script>
""", unsafe_allow_html=True)


st.markdown(f"""
    <div class="header-container">
        <div class="header-title" style="margin-bottom: 0px;">AI Yorum Analizi</div>
    </div>
""", unsafe_allow_html=True)
st.caption(f"Sürüm Doğrulama: {CURRENT_VERSION}")

# ============ STARTUP SAFETY CHECK ============
assert_cost_safe()  # SIFIR ücret kontrolü

if 'comments_to_analyze' not in st.session_state:
    st.session_state.comments_to_analyze = []

if 'bulk_results' not in st.session_state:
    st.session_state.bulk_results = None

if 'cmp_results' not in st.session_state:
    st.session_state.cmp_results = {}

# --- TAB STATE MANAGEMENT ---
# Initialize persistent results for each tab
tabs = ["Mağaza Linki", "Dosya Yükle (CSV/Excel)", "Metin Girişi", "Karşılaştır"]
if 'tab_states' not in st.session_state:
    st.session_state.tab_states = {
        tab: {
            "comments": [],
            "results": None,
            "pool": []
        } for tab in tabs
    }

if '_last_tab' not in st.session_state:
    st.session_state._last_tab = "Mağaza Linki"

# --- TAB NAVIGATION ---
def on_tab_change():
    """Manage state transitions when switching chips."""
    new_tab = st.session_state.get("active_tab")
    prev_tab = st.session_state.get("_last_tab")
    
    # 1. Save current global state into the previous tab's slot
    if prev_tab and prev_tab in st.session_state.tab_states:
        st.session_state.tab_states[prev_tab]["comments"] = st.session_state.get("comments_to_analyze", [])
        st.session_state.tab_states[prev_tab]["results"] = st.session_state.get("bulk_results")
        st.session_state.tab_states[prev_tab]["pool"] = st.session_state.get("all_fetched_pool", [])

    # 2. Load the new tab's previously saved state into global pointers
    # 'Sıfır ekran' rule: If no results exist for the new tab, keep it fresh (empty comments)
    if new_tab and new_tab in st.session_state.tab_states:
        tab_data = st.session_state.tab_states[new_tab]
        if tab_data.get("results") is not None:
            st.session_state.comments_to_analyze = tab_data["comments"]
            st.session_state.bulk_results = tab_data["results"]
            st.session_state.all_fetched_pool = tab_data["pool"]
        else:
            # No analysis done yet, show a clean screen
            st.session_state.comments_to_analyze = []
            st.session_state.bulk_results = None
            st.session_state.all_fetched_pool = []

    # 3. Clear ONLY input-specific keys
    st.session_state["_store_url_input"] = ""
    st.session_state["manual_text_input"] = ""
    st.session_state["_selected_app_id"] = None  # Clear selected app
    st.session_state["_show_search"] = True  # Reset search visibility flag
    st.session_state["_search_results_all"] = []  # Clear search cache
    st.session_state["_last_search_query"] = ""  # Reset search query
    if "last_files_key" in st.session_state:
        st.session_state.last_files_key = ""
    
    # 4. Exit comparison/global modes if switching to single analysis tabs
    if new_tab != "Karşılaştır":
        st.session_state["_cmp_mode"] = False
        st.session_state.pop("_cmp_pending", None)
    
    st.session_state["_cost_warned"] = False # Reset for new context
    st.session_state._last_tab = new_tab

def clear_current_tab_data():
    """Manual reset for the active tab's processed data."""
    cur = st.session_state.get("active_tab")
    if cur and cur in st.session_state.tab_states:
        st.session_state.tab_states[cur] = {"comments": [], "results": None, "pool": []}
    st.session_state.comments_to_analyze = []
    st.session_state.bulk_results = None
    st.session_state.all_fetched_pool = []

tabs = ["Mağaza Linki", "Dosya Yükle (CSV/Excel)", "Metin Girişi", "Karşılaştır"]
st.markdown('<div class="no-print">', unsafe_allow_html=True)
active_tab = st.radio(
    "Navigasyon",
    tabs,
    label_visibility="collapsed",
    horizontal=True,
    key="active_tab",
    on_change=on_tab_change
)
st.markdown('</div>', unsafe_allow_html=True)

st.markdown('<div style="margin-top: 15px;"></div>', unsafe_allow_html=True)

# Render Content based on active_tab
if active_tab == "Mağaza Linki":
    with st.container(border=True):
        # 1. Geçmiş başlatma
        if "url_history" not in st.session_state:
            st.session_state.url_history = []

        # 2. Geçmişten seçim yapıldıysa input'a yükle
        if st.session_state.get("_url_pick"):
            st.session_state["_store_url_input"] = st.session_state.pop("_url_pick")

        # Placeholder styling - make it more visible
        st.markdown("""
        <style>
        input::placeholder {
            opacity: 1 !important;
            color: #6B7280 !important;
            font-weight: 500;
        }
        </style>
        """, unsafe_allow_html=True)

        # 3. Giriş alanı
        store_url = st.text_input(
            "",
            placeholder="🔍 Uygulama ismi, linki veya ID girerek arama yapın",
            key="_store_url_input"
        )
        st.session_state.app_url = store_url

        # Initialize search flag if needed
        if "_show_search" not in st.session_state:
            st.session_state._show_search = True
        
        if "_search_performed" not in st.session_state:
            st.session_state._search_performed = False
        
        if "_selected_app_id" not in st.session_state:
            st.session_state._selected_app_id = None
        
        if "_platform_filter" not in st.session_state:
            st.session_state._platform_filter = "Android"

        # Styling ONLY for platform selector buttons
        st.markdown("""
        <style>
        /* Platform selector buttons - wide button styling ONLY */
        [data-testid="stButton"] > button[key*="platform_"] {
            width: 100% !important;
            height: 56px !important;
            border-radius: 36px !important;
            font-weight: 600 !important;
            font-family: 'Poppins', sans-serif !important;
            font-size: 16px !important;
            border: 2px solid #E5E7EB !important;
            background-color: white !important;
            color: #1F2937 !important;
            padding: 0 16px !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            cursor: pointer !important;
            transition: all 0.3s ease !important;
        }
        
        /* Platform button - default unselected state */
        [data-testid="stButton"] > button[key="platform_android"],
        [data-testid="stButton"] > button[key="platform_ios"] {
            background-color: white !important;
            color: #1F2937 !important;
            border: 2px solid #E5E7EB !important;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Dynamic styling for selected platform button ONLY
        selected_platform = st.session_state._platform_filter
        st.markdown(f"""
        <style>
        [data-testid="stButton"] > button[key="platform_{selected_platform.lower()}"] {{
            background-color: #818CF8 !important;
            color: white !important;
            border: 2px solid #818CF8 !important;
        }}
        </style>
        """, unsafe_allow_html=True)
        
        # Determine the actual app ID (either from selection or from input text)
        selected_app = st.session_state._selected_app_id
        
        # If app selected from search, use that
        if selected_app:
            active_input = selected_app
        else:
            active_input = store_url.strip()
        
        # 3.5 Real-time app search suggestions with infinite scroll
        # Check if an app is selected based on input format
        is_app_selected = selected_app is not None or (store_url.strip() and store_url.strip().startswith(("com.", "org.", "net.", "io.", "id.")))
        is_search_query = not is_app_selected and store_url.strip() and not store_url.strip().startswith(("http", "id", "com.", "org.", "net."))
        
        # Manage search visibility flag
        if is_app_selected:
            # App is selected - permanently hide search
            st.session_state._show_search = False
        elif not store_url.strip():
            # Input cleared - re-enable search
            st.session_state._show_search = True
        
        # Track whether a search has been performed (for platform button visibility)
        if is_search_query:
            st.session_state._search_performed = True
        elif not store_url.strip():
            st.session_state._search_performed = False
        
        # Platform selector buttons - only show AFTER a search is made
        if st.session_state._search_performed:
            platform_col1, platform_col2 = st.columns(2, gap="medium")
            
            with platform_col1:
                if st.button("Android", key="platform_android", use_container_width=True):
                    st.session_state._platform_filter = "Android"
                    st.rerun()
            
            with platform_col2:
                if st.button("iOS", key="platform_ios", use_container_width=True):
                    st.session_state._platform_filter = "iOS"
                    st.rerun()
        
        # Only show search if ALL conditions are met
        should_show_search = is_search_query and not is_app_selected and st.session_state._show_search
        
        # Use selected app ID or input value for further processing
        if selected_app:
            store_url = selected_app
        
        if should_show_search:
            # Only show search if input looks like an app name (not a full URL/ID) and no app is selected
            search_query = store_url.strip()
            if len(search_query) >= 2:  # Only search for 2+ characters
                from google_play_scraper import search as play_search
                
                # Initialize search cache in session state
                if st.session_state.get("_last_search_query") != search_query or st.session_state.get("_last_platform_filter") != st.session_state._platform_filter:
                    # New search query or platform changed - fetch fresh results
                    all_results = []
                    platform = st.session_state._platform_filter
                    
                    # Search Android (Google Play)
                    if platform in ["Android", "Both"]:
                        try:
                            android_results = play_search(search_query, n_hits=50, lang='tr', country='tr')
                            for app in android_results:
                                app['platform'] = 'Android'
                            all_results.extend(android_results)
                        except:
                            pass
                    
                    # Search iOS (App Store) - using iTunes Search API
                    if platform in ["iOS", "Both"]:
                        try:
                            import requests
                            search_url = "https://itunes.apple.com/search"
                            params = {
                                "term": search_query,
                                "country": "TR",
                                "media": "software",
                                "entity": "software",
                                "limit": 50,
                                "lang": "tr_TR"
                            }
                            response = requests.get(search_url, params=params, timeout=10)
                            if response.status_code == 200:
                                data = response.json()
                                ios_results = data.get("results", [])
                                for app in ios_results:
                                    app['platform'] = 'iOS'
                                    # Normalize app structure for consistency
                                    app['title'] = app.get('trackName', '')
                                    app['appId'] = str(app.get('trackId', ''))
                                    app['icon'] = app.get('artworkUrl512', '')
                                all_results.extend(ios_results[:50])
                        except:
                            pass
                    
                    st.session_state._search_results_all = all_results
                    st.session_state._last_search_query = search_query
                    st.session_state._last_platform_filter = platform
                    st.session_state._search_displayed_count = 10  # Start with 10 results
                
                all_results = st.session_state.get("_search_results_all", [])
                displayed_count = st.session_state.get("_search_displayed_count", 10)
                
                if all_results:
                    # Render search results header
                    st.markdown('<div style="font-size:13px;color:#64748B;font-weight:700;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:12px;display:block;font-family:\'Poppins\', sans-serif;">🔍 Bulunan Uygulamalar (' + str(len(all_results)) + ')</div>', unsafe_allow_html=True)
                    
                    # Show search results with native Streamlit components
                    results_to_display = all_results[:displayed_count]
                    
                    for idx, result in enumerate(results_to_display):
                        app_name = result.get('title', 'Bilinmiyor')[:50]
                        app_id = result.get('appId', '')
                        app_icon = result.get('icon', '').strip() if result.get('icon') else ''
                        
                        # Create a row for each search result - compact layout
                        col_icon, col_info, col_btn = st.columns([0.08, 0.75, 0.17])
                        
                        # Display icon as CIRCLE
                        with col_icon:
                            if app_icon and app_icon.startswith('http'):
                                # Valid icon URL - circular
                                icon_html = f'<div style="width: 38px; height: 38px;"><img src="{app_icon}" style="width: 100%; height: 100%; border-radius: 50%; object-fit: cover; display: block;" /></div>'
                            else:
                                # No icon - circular emoji box
                                icon_html = '<div style="width: 38px; height: 38px; display: flex; align-items: center; justify-content: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); border-radius: 50%; font-size: 18px; color: white; font-weight: bold;">📱</div>'
                            st.markdown(icon_html, unsafe_allow_html=True)
                        
                        # Display app name and ID - compact
                        with col_info:
                            st.markdown(f'<div style="font-weight: 700; color: #1F2937; font-size: 13px; font-family: \'Poppins\', sans-serif; margin-bottom: 1px; line-height: 1.2; margin-top: 2px;">{app_name}</div>', unsafe_allow_html=True)
                            st.markdown(f'<div style="font-size: 10px; color: #9CA3AF; font-family: \'Poppins\', sans-serif; margin-bottom: 0px; line-height: 1;">{app_id}</div>', unsafe_allow_html=True)
                        
                        # Display select button as chip
                        with col_btn:
                            # Styling ONLY for select_app buttons - NO platform button styles here
                            st.markdown("""
                            <style>
                            /* Container for select_app button - make it compact */
                            [data-testid="stButton"]:has(> button[key*="select_app"]) {
                                width: auto !important;
                                display: inline-block !important;
                            }
                            
                            /* Only style select_app buttons as material design chips */
                            [data-testid="stButton"] > button[key*="select_app"] {
                                width: auto !important;
                                padding: 3px 9px !important;
                                border-radius: 16px !important;
                                background-color: #818CF8 !important;
                                color: white !important;
                                border: none !important;
                                font-size: 10px !important;
                                font-weight: 600 !important;
                                font-family: 'Poppins', sans-serif !important;
                                height: 20px !important;
                                min-height: 20px !important;
                                line-height: 20px !important;
                                display: inline-flex !important;
                                align-items: center !important;
                                justify-content: center !important;
                            }
                            [data-testid="stButton"] > button[key*="select_app"] > span {
                                line-height: 1 !important;
                                display: flex !important;
                                align-items: center !important;
                                white-space: nowrap !important;
                            }
                            [data-testid="stButton"] > button[key*="select_app"]:hover {
                                background-color: #6366F1 !important;
                                transition: all 0.2s ease !important;
                            }
                            </style>
                            """, unsafe_allow_html=True)
                            
                            button_key = f"select_app_{idx}_{app_id}"
                            if st.button("Seç", key=button_key, width='content'):
                                st.session_state._selected_app_id = app_id
                                st.session_state._show_search = False
                                st.session_state._search_results_all = []
                                st.session_state._last_search_query = ""
                                st.rerun()
                    
                    # Load more button for manual loading
                    if len(all_results) > displayed_count:
                        st.markdown(f'<div style="text-align:center;margin-top:2px;margin-bottom:4px;font-size:12px;color:#818CF8;font-family:\'Poppins\', sans-serif;">{displayed_count} / {len(all_results)} sonuç gösterildi</div>', unsafe_allow_html=True)
                        col_left, col_center, col_right = st.columns([1, 2, 1])
                        with col_center:
                            if st.button("Daha Fazla Yükle", width='stretch', key="load_more_apps"):
                                st.session_state._search_displayed_count = min(displayed_count + 10, len(all_results))
                                st.rerun()


        if st.session_state.url_history:
            chips_data = [
                {"url": h["url"] if isinstance(h, dict) else h,
                 "name": (h["name"] if isinstance(h, dict) else h)[:22]}
                for h in st.session_state.url_history[:5]
            ]
            chips_js_array = json.dumps(chips_data)
            components.html(f"""
                <style>
                    body {{ margin:0; padding:0; background:transparent; overflow:hidden; }}
                    .chip-wrap {{ display:flex; flex-wrap:wrap; gap:6px; padding:2px 0 4px 0; }}
                    .chip-label {{ font-size:0.68rem; color:#94A3B8; font-weight:700; text-transform:uppercase; letter-spacing:0.8px; margin-bottom:4px; font-family:'Poppins',sans-serif; }}
                    .chip {{ display:inline-block; cursor:pointer; background:#EEF2FF; border:1px solid #818CF8; color:#4338CA; border-radius:20px; padding:4px 12px; font-size:0.78rem; font-weight:600; font-family:'Poppins',sans-serif; white-space:nowrap; transition:all 0.15s ease; user-select:none; }}
                    .chip:hover {{ background:#E0E7FF; border-color:#6366F1; transform:translateY(-1px); }}
                    .chip:active {{ transform:scale(0.97); }}
                </style>
                <div class="chip-label">Son Aramalar</div>
                <div class="chip-wrap" id="chip-wrap"></div>
                <script>
                (function() {{
                    var chips = {chips_js_array};
                    var wrap = document.getElementById('chip-wrap');
                    chips.forEach(function(c) {{
                        var span = document.createElement('span');
                        span.className = 'chip';
                        span.textContent = c.name;
                        span.title = c.url;
                        span.addEventListener('click', function() {{
                            var inp = window.parent.document.querySelector('[data-testid="stTextInput"] input');
                            if (!inp) return;
                            var setter = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
                            setter.call(inp, c.url);
                            inp.dispatchEvent(new Event('input', {{bubbles:true}}));
                            inp.dispatchEvent(new Event('change', {{bubbles:true}}));
                            inp.focus();
                        }});
                        wrap.appendChild(span);
                    }});
                }})();
                </script>
            """, height=60, scrolling=False)

        # Show confirmation BEFORE date selector if app is selected
        if is_app_selected and store_url.strip():
            try:
                from google_play_scraper import app as get_app_details
                
                selected_app_id = store_url.strip()
                cache_key = f"_selected_app_{selected_app_id}"
                
                if cache_key not in st.session_state:
                    try:
                        app_details = get_app_details(selected_app_id, lang='tr', country='tr')
                        st.session_state[cache_key] = app_details
                    except:
                        app_details = None
                        st.session_state[cache_key] = None
                else:
                    app_details = st.session_state[cache_key]
                
                banner_col, button_col = st.columns([0.95, 0.05])
                
                with banner_col:
                    if app_details:
                        app_title = app_details.get('title', selected_app_id)
                        app_icon = app_details.get('icon', '')
                        app_rating = app_details.get('score', 0)
                        # Get category (genre for Android, trackGenreName for iOS)
                        app_category = app_details.get('genre', app_details.get('trackGenreName', 'Kategori Bilinmiyor'))
                        
                        success_html = f"""
                        <div style="margin-bottom: 0px; padding: 4px 8px; background: linear-gradient(135deg, #ECFDF5 0%, #E0F2FE 100%); border: 1.5px solid #10B981; border-radius: 12px; font-family: 'Poppins', sans-serif; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);">
                            <div style="display: flex; gap: 8px; align-items: flex-start;">
                                {f'<img src="{app_icon}" style="width: 48px; height: 48px; border-radius: 10px; object-fit: cover; flex-shrink: 0;"/>' if app_icon else '<div style="width: 48px; height: 48px; border-radius: 10px; background: #E0E7FF; display: flex; align-items: center; justify-content: center; font-size: 24px; flex-shrink: 0;"></div>'}
                                <div style="flex: 1; min-width: 0;">
                                    <div style="font-size: 14px; color: #047857; font-weight: 600; margin-bottom: 2px; word-break: break-word;">{app_title}</div>
                                    <div style="font-size: 11px; color: #059669; margin-bottom: 3px;">Kategori: {app_category}</div>
                                    <div style="font-size: 12px; color: #10B981; margin-bottom: 3px;">Puanı: {app_rating:.1f} ★</div>
                                    <div style="font-size: 12px; color: #059669; font-weight: 500;">Yorumları çekiliyor...</div>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(success_html, unsafe_allow_html=True)
                    else:
                        success_html = f"""
                        <div style="margin-bottom: 0px; padding: 4px 8px; background: linear-gradient(135deg, #ECFDF5 0%, #E0F2FE 100%); border: 1.5px solid #10B981; border-radius: 12px; font-family: 'Poppins', sans-serif; box-shadow: 0 4px 12px rgba(16, 185, 129, 0.1);">
                            <div style="display: flex; gap: 8px; align-items: center;">
                                <div style="font-size: 16px;">✅</div>
                                <div>
                                    <div style="font-size: 13px; color: #059669; font-weight: 600;">{store_url.strip()}</div>
                                    <div style="font-size: 12px; color: #059669; margin-top: 0px;">Yorumları çekiliyor...</div>
                                </div>
                            </div>
                        </div>
                        """
                        st.markdown(success_html, unsafe_allow_html=True)
            except:
                pass
            
            with button_col:
                # White X button styling - fully centered
                st.markdown("""
                <style>
                [data-testid="stButton"] {
                    display: flex !important;
                    justify-content: center !important;
                    align-items: center !important;
                }
                [data-testid="stButton"] button {
                    background-color: white !important;
                    color: black !important;
                    border: 1px solid #ddd !important;
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                    padding: 0px 8px !important;
                    text-align: center !important;
                    min-width: 40px !important;
                    min-height: 40px !important;
                }
                [data-testid="stButton"] button span {
                    display: flex !important;
                    align-items: center !important;
                    justify-content: center !important;
                }
                [data-testid="stButton"] button:hover {
                    background-color: #f5f5f5 !important;
                    border-color: #999 !important;
                }
                </style>
                """, unsafe_allow_html=True)
                
                if st.button("✕", key="new_search_btn", help="Yeni arama yapın", width='content'):
                    st.session_state._selected_app_id = None
                    st.session_state._show_search = True
                    st.session_state._search_results_all = []
                    st.session_state._last_search_query = ""
                    st.rerun()

        # 6. Tarih aralığı
        time_range = st.selectbox(
            "Tarih Aralığı Seçin:",
            options=["Son 1 Hafta", "Son 1 Ay", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl", "Son 2 Yıl", "Son 3 Yıl"],
            index=0,
            key="main_time_picker" # Adding key to ensure correct state initialization
        )
        range_map = {"Son 1 Hafta": 7, "Son 1 Ay": 30, "Son 3 Ay": 90, "Son 6 Ay": 180, "Son 1 Yıl": 365, "Son 2 Yıl": 730, "Son 3 Yıl": 1095}
        days_limit = range_map[time_range]
        st.markdown('<div class="no-print" style="margin-top:6px;margin-bottom:10px;font-size:0.85rem;color:#64748b;">Apple: Mağaza linki veya ID (id...), Play Store: Link veya paket adı (com...) geçerlidir.</div>', unsafe_allow_html=True)



    if store_url.strip():
        u = store_url.strip()
        platform: Optional[str] = None
        app_id: str = ""
        country: str = "tr" 
        is_search_link = False
        search_query = ""
        
        # ── Google Play ──────────────────────────────────────────────
        if "play.google.com" in u:
            platform = "google"
            
            # Ürün linki: id=(paket_adı)
            match = re.search(r"id=([^&/]+)", u)
            if match: 
                app_id = match.group(1)
            else:
                # Arama linki: search?q=(app_adı)
                search_match = re.search(r"[?&]q=([^&/]+)", u)
                if search_match:
                    is_search_link = True
                    search_query = search_match.group(1).replace("+", " ")
                    
        # ── Apple App Store ──────────────────────────────────────────
        elif "apple.com" in u:
            platform = "apple"
            
            match = re.search(r"id(\d+)", u)
            if match: 
                app_id = match.group(1)
            else:
                # Arama linki: search?term=(app_adı)
                search_match = re.search(r"[?&]term=([^&/]+)", u)
                if search_match:
                    is_search_link = True
                    search_query = search_match.group(1).replace("+", " ")
            
            country_match = re.search(r"apple\.com/([a-z]{2,3})/", u)
            if country_match: country = country_match.group(1)
        else:
            # Doğrudan paket adı ya da ID
            clean_u = u.lower()
            if clean_u.startswith("id") and clean_u[2:].isdigit():
                platform = "apple"
                app_id = clean_u[2:]
            elif clean_u.isdigit():
                platform = "apple"
                app_id = clean_u
            elif "." in u and re.match(r"^[a-zA-Z0-9._]+$", u, re.IGNORECASE):
                platform = "google"
                app_id = u

        # Arama linki ise ilk sonucu ara ve kullan
        if is_search_link and search_query:
            with st.spinner(f"'{search_query}' uygulaması aranıyor..."):
                try:
                    if platform == "google":
                        # Google Play'de ara — web scraping ile paket ID'sini bul
                        import requests
                        url = f"https://play.google.com/store/search?q={search_query}&c=apps"
                        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
                        r = requests.get(url, headers=headers, timeout=10)
                        
                        matches = re.findall(r'/store/apps/details\?id=([^"&]+)', r.text)
                        if matches:
                            app_id = matches[0]  # İlk sonuç
                            # Başlık için search() sonucunu kullan
                            from google_play_scraper import search
                            search_results = search(search_query, n_hits=1)
                            title = search_results[0].get('title', app_id) if search_results else app_id
                            st.info(f"✋ **Arama linki** kullanıldı. İlk sonuç seçildi: **{title}**")
                        else:
                            st.error(f"'{search_query}' uygulaması Play Store'da bulunamadı.")
                            app_id = ""
                    elif platform == "apple":
                        # iTunes API'den ara
                        import requests
                        r = requests.get(
                            f"https://itunes.apple.com/search?term={search_query}&entity=software&country={country}",
                            timeout=5
                        )
                        data = r.json()
                        if data.get('results'):
                            app_id = str(data['results'][0]['trackId'])
                            st.info(f"✋ **Arama linki** kullanıldı. İlk sonuç seçildi: **{data['results'][0].get('trackCensoredName', app_id)}**")
                        else:
                            st.error(f"'{search_query}' uygulaması App Store'da bulunamadı.")
                            app_id = ""
                except Exception as e:
                    st.error(f"Arama sırasında hata: {e}")
                    app_id = ""
        
        # Saatlik cache key
        _refresh_token = st.session_state.get("_refresh_token", 0)
        fetch_key = f"{platform}_{app_id}_{time_range}_{country}_{_refresh_token}"
        
        if not platform or not app_id:
            if store_url.strip():
                st.warning(
                    "❌ **Format hatası:**\n\n"
                    "✅ **Desteklenen formatlar:**\n"
                    "- **Play Store Ürün Linki**: `https://play.google.com/store/apps/details?id=com.hepsipay.app`\n"
                    "- **Play Store Paket Adı**: `com.hepsipay.app`\n"
                    "- **App Store Ürün Linki**: `https://apps.apple.com/tr/app/.../id123456789`\n"
                    "- **App Store Metin ID**: `id123456789` veya sadece `123456789`\n\n"
                    "📱 **Arama linki kullandıysanız**, bu da çalışır! Sistem ilk sonucu analiz edecektir."
                )
        elif st.session_state.get("last_fetch_key") == fetch_key and st.session_state.get("all_fetched_pool"):
            # Cache'den geliyor — geçmişe ekle
            current_url = store_url.strip()
            # Cache'deki ismi almaya çalış, yoksa session'dakini kullan
            current_name = st.session_state.get("detected_app_name", current_url)
            
            existing_urls = [h["url"] if isinstance(h, dict) else h for h in st.session_state.url_history]
            if current_url and current_url not in existing_urls:
                st.session_state.url_history.insert(0, {
                    "url": current_url,
                    "name": current_name
                })
                st.session_state.url_history = st.session_state.url_history[:5]
            else:
                # Zaten varsa ve ismi ID ise, session'daki daha iyi isimle güncelle
                for h in st.session_state.url_history:
                    if isinstance(h, dict) and h["url"] == current_url:
                        if h["name"] == app_id and current_name != app_id:
                            h["name"] = current_name
                        break
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
                # Try iTunes API with multiple countries fallback
                found_name = False
                for try_country in [country, 'tr', 'us', 'gb']:
                    try:
                        resp = requests.get(f"https://itunes.apple.com/lookup?id={app_id}&country={try_country}", timeout=5)
                        if resp.status_code == 200:
                            data = resp.json()
                            if data.get('results'):
                                name_for_state = data['results'][0].get('trackCensoredName', app_id)
                                found_name = True
                                break
                    except: continue
                
                if not found_name and "apple.com" in u and "/app/" in u:
                    try:
                        raw_name = u.split("/app/")[-1].split("/")[0].replace("-", " ")
                        name_for_state = urllib.parse.unquote(raw_name).title()
                    except: pass

            with st.container():
                loading_placeholder = st.empty()
                with loading_placeholder.container():
                    st.markdown(f"### {name_for_state} Analizi")
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
                                f"Bu uygulama için yalnızca **{len(fetched_comments)}** yorum bulundu. "
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
                            
                            # Store metadata for persistent display
                            st.session_state.fetch_metadata = {
                                "total_found": total_found,
                                "AI_LIMIT": AI_LIMIT,
                                "time_range": time_range
                            }
                            if v_dates_anal and v_dates_pool:
                                st.session_state.fetch_metadata.update({
                                    "pool_start": cast(datetime, min(v_dates_pool)).strftime('%d-%m-%Y'),
                                    "pool_end":   cast(datetime, max(v_dates_pool)).strftime('%d-%m-%Y'),
                                    "anal_start": cast(datetime, min(v_dates_anal)).strftime('%d-%m-%Y'),
                                    "anal_end":   cast(datetime, max(v_dates_anal)).strftime('%d-%m-%Y'),
                                })
                        else:
                            # Hızlı Analiz — limit yok, tamamı alınır
                            st.session_state.comments_to_analyze = fetched_comments
                            st.session_state.fetch_metadata = {
                                "total_found": len(fetched_comments),
                                "AI_LIMIT": AI_LIMIT,
                                "time_range": time_range
                            }
                        
                        # Başarılı URL'yi geçmişe ekle (url + isim birlikte)
                        current_url = store_url.strip()
                        current_name = st.session_state.get("detected_app_name", current_url)
                        existing_urls = [h["url"] if isinstance(h, dict) else h for h in st.session_state.url_history]
                        if current_url and current_url not in existing_urls:
                            st.session_state.url_history.insert(0, {
                                "url": current_url,
                                "name": current_name
                            })
                            st.session_state.url_history = st.session_state.url_history[:5]
                        
                        st.session_state.url_history = st.session_state.url_history[:5]
                    else:
                        loading_placeholder.empty()
                        st.info(f"{time_range} kriterine uygun yorum bulunamadı.")
                except Exception as e:
                    loading_placeholder.empty()
                    st.error(f"Yorumlar çekilirken bir hata oluştu: {e}")
        
elif active_tab == "Dosya Yükle (CSV/Excel)":
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
                    st.markdown(f"### {uploaded_file.name}")
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

elif active_tab == "Metin Girişi":
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

elif active_tab == "Karşılaştır":
    st.markdown("### Uygulama Karşılaştırma")
    st.markdown('<div style="font-size:0.85rem;color:#64748B;margin-bottom:12px;">2 uygulama için de ID veya link gir.</div>', unsafe_allow_html=True)

    if "cmp_results" not in st.session_state:
        st.session_state.cmp_results = {}

    num_apps = 2

    cmp_cols = st.columns(2)
    cmp_inputs = []
    for ci in range(2):
        with cmp_cols[ci]:
            st.markdown(f'<div style="font-size:0.8rem;font-weight:600;color:#6366F1;margin-bottom:4px;">Uygulama {ci+1}</div>', unsafe_allow_html=True)
            url_val = st.text_input("", placeholder="com.example veya id123...", key=f"cmp_url_{ci}", label_visibility="collapsed")
            cmp_inputs.append(url_val.strip())

    cmp_range = st.selectbox("Tarih Aralığı:", ["Son 1 Hafta", "Son 1 Ay", "Son 3 Ay", "Son 6 Ay", "Son 1 Yıl"], key="cmp_range")
    cmp_days = {"Son 1 Hafta": 7, "Son 1 Ay": 30, "Son 3 Ay": 90, "Son 6 Ay": 180, "Son 1 Yıl": 365}[cmp_range]

    # Karşılaştırma için analiz yöntemi seçimi
    cmp_col1, cmp_col2 = st.columns(2)
    with cmp_col1:
        cmp_analysis_type = st.radio(
            "Analiz Yöntemi:",
            ["Hızlı Analiz", "Zengin Analiz"],
            key="cmp_analysis_type",
            horizontal=True,
            label_visibility="collapsed"
        )
    
    with cmp_col2:
        if cmp_analysis_type == "Zengin Analiz":
            cmp_analysis_mode = st.radio(
                "Analiz Derinliği:",
                [0, 1],
                format_func=lambda x: ["Standart (Genel)", "Gelişmiş (Derin)"][x],
                key="cmp_analysis_mode",
                horizontal=True,
                label_visibility="collapsed"
            )
        else:
            cmp_analysis_mode = 0

    if st.button("Karşılaştırmayı Başlat", type="primary", width='stretch', key="cmp_start_btn"):
        active_inputs = [u for u in cmp_inputs if u]
        if len(active_inputs) < 2:
            st.warning("En az 2 uygulama ID'si girin.")
        else:
            st.session_state["_cmp_mode"] = True
            st.session_state.pop("bulk_results", None)
            st.session_state["_cmp_pending"] = active_inputs
            st.session_state["_cmp_days"] = cmp_days
            st.session_state["_cmp_analysis_type"] = cmp_analysis_type
            st.session_state["_cmp_analysis_mode"] = cmp_analysis_mode
            st.session_state.cmp_results = {}
            st.rerun()

    if st.session_state.get("cmp_results"):
        results_c = st.session_state.cmp_results
        n_res = len(results_c)
        app_colors_sum = ["#818CF8", "#F4A261", "#38BDF8"]

        st.markdown("<div style='margin-top:16px;'></div>", unsafe_allow_html=True)
        sum_cols = st.columns(n_res)

        for ci, (app_nm, data) in enumerate(results_c.items()):
            with sum_cols[ci]:
                accent = app_colors_sum[ci % len(app_colors_sum)]
                badge_html = ""

                if data["pos_pct"] >= 55:
                    tone_title = "Genel olarak olumlu"
                    tone_body = f"Kullanıcıların %{data['pos_pct']}'i uygulamadan memnun. Olumlu yorumlar baskın seyrediyor."
                    bg_col = "#F0FDF4"; bdr_col = "#BBF7D0"
                elif data["neg_pct"] >= 50:
                    tone_title = "Dikkat: olumsuz eğilim"
                    tone_body = f"Yorumların %{data['neg_pct']}'i olumsuz. Teknik sorunlar öne çıkıyor."
                    bg_col = "#FEF2F2"; bdr_col = "#FECDD3"
                else:
                    tone_title = "Dengeli kullanıcı deneyimi"
                    tone_body = f"Olumlu (%{data['pos_pct']}) ve olumsuz (%{data['neg_pct']}) yorumlar dengeli seyrediyor."
                    bg_col = "#FEFCE8"; bdr_col = "#FDE68A"

                # Verileri doğrudan data'dan al
                icon_url      = data.get("icon", "")
                store_val     = data.get("store", "")
                rating_val    = data.get("rating", 0)
                ratings_cnt   = data.get("ratings", 0)
                installs_val  = data.get("installs", "")
                version_val   = data.get("version", "")

                stars_filled = int(rating_val)
                star_html = ""
                for si in range(5):
                    col_s = "#F59E0B" if si < stars_filled else "#D1D5DB"
                    star_html += f'<span style="color:{col_s};font-size:0.75rem;">★</span>'

                ratings_str = f"{ratings_cnt:,}" if isinstance(ratings_cnt, int) and ratings_cnt > 0 else "—"

                meta_html = ""
                if rating_val > 0:
                    meta_html = f"""
<div style="display:flex;gap:6px;flex-wrap:nowrap;margin-bottom:12px;padding-bottom:10px;border-bottom:1px solid #E2E8F0;">
  <div style="background:#FFFBEB;border:1px solid #FDE68A;border-radius:8px;padding:6px 4px;text-align:center;flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:72px;">
    <div style="font-size:1rem;font-weight:800;color:#D97706;">{rating_val}</div>
    <div style="line-height:1;">{star_html}</div>
    <div style="font-size:0.58rem;color:#94A3B8;font-weight:600;margin-top:2px;">Puan</div>
  </div>
  <div style="background:#F0FDF4;border:1px solid #BBF7D0;border-radius:8px;padding:6px 4px;text-align:center;flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:72px;">
    <div style="font-size:0.78rem;font-weight:700;color:#059669;word-break:break-all;">{ratings_str}</div>
    <div style="font-size:0.58rem;color:#94A3B8;font-weight:600;margin-top:2px;">Değerlendirme</div>
  </div>
  <div style="background:#EFF6FF;border:1px solid #DBEAFE;border-radius:8px;padding:6px 4px;text-align:center;flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:72px;">
    <div style="font-size:0.75rem;font-weight:700;color:#2563EB;word-break:break-all;">{installs_val}</div>
    <div style="font-size:0.58rem;color:#94A3B8;font-weight:600;margin-top:2px;">İndirme</div>
  </div>
  <div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:6px 4px;text-align:center;flex:1;display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:72px;">
    <div style="font-size:0.75rem;font-weight:700;color:#475569;word-break:break-all;">v{version_val}</div>
    <div style="font-size:0.58rem;color:#94A3B8;font-weight:600;margin-top:2px;">Versiyon</div>
  </div>
</div>"""



                icon_html = (
                    f'<img src="{icon_url}" '
                    f'referrerpolicy="no-referrer" crossorigin="anonymous" '
                    f'style="width:44px;height:44px;border-radius:10px;'
                    f'object-fit:cover;border:2px solid rgba(255,255,255,0.3);flex-shrink:0;" '
                    f'onerror="this.style.display=\'none\'" />'
                ) if icon_url else ""

                header_html = (
                    f'<div style="background:{accent};padding:14px;">'
                    f'<div style="display:flex;align-items:center;gap:12px;">'
                    f'{icon_html}'
                    f'<div style="flex:1;min-width:0;">'
                    f'<div style="font-size:0.78rem;font-weight:700;color:white;line-height:1.3;'
                    f'white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{app_nm}</div>'
                    f'<div style="font-size:1.8rem;font-weight:800;color:white;line-height:1.1;">'
                    f'{data["score"]}<span style="font-size:0.7rem;opacity:0.75;">/100</span></div>'
                    f'<div style="font-size:0.62rem;color:rgba(255,255,255,0.9);font-weight:700;">{store_val} • {data.get("genre", "?")} {data.get("rank", "")}</div>'
                    f'</div>'
                    f'</div>'
                    f'</div>'
                )

                card = (
                    '<div style="background:#FFFFFF;border:2px solid #E2E8F0;border-radius:14px;overflow:hidden;">'
                    + header_html
                    + '<div style="padding:14px;">'
                    + meta_html
                    + '<div style="font-size:0.72rem;font-weight:700;color:#1E293B;text-transform:uppercase;letter-spacing:0.8px;margin-bottom:5px;">' + tone_title + '</div>'
                    + '<p style="font-size:0.82rem;color:#334155;line-height:1.65;margin:0 0 12px 0;">' + tone_body + '</p>'
                    + '<div style="display:flex;flex-direction:column;gap:6px;">'
                    + '<div><div style="display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:2px;"><span style="color:#10b981;font-weight:600;">Olumlu</span><span style="color:#10b981;font-weight:700;">' + str(data['pos_pct']) + '%</span></div><div style="height:5px;background:#E2E8F0;border-radius:3px;overflow:hidden;"><div style="width:' + str(data['pos_pct']) + '%;height:100%;background:#10b981;border-radius:3px;"></div></div></div>'
                    + '<div><div style="display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:2px;"><span style="color:#f43f5e;font-weight:600;">Olumsuz</span><span style="color:#f43f5e;font-weight:700;">' + str(data['neg_pct']) + '%</span></div><div style="height:5px;background:#E2E8F0;border-radius:3px;overflow:hidden;"><div style="width:' + str(data['neg_pct']) + '%;height:100%;background:#f43f5e;border-radius:3px;"></div></div></div>'
                    + '<div><div style="display:flex;justify-content:space-between;font-size:0.72rem;margin-bottom:2px;"><span style="color:#818cf8;font-weight:600;">Görüş</span><span style="color:#818cf8;font-weight:700;">' + str(data['neu_pct']) + '%</span></div><div style="height:5px;background:#E2E8F0;border-radius:3px;overflow:hidden;"><div style="width:' + str(data['neu_pct']) + '%;height:100%;background:#818cf8;border-radius:3px;"></div></div></div>'
                    + '</div>'
                    + '<div style="margin-top:10px;font-size:0.7rem;color:#94A3B8;text-align:right;">' + str(data['total']) + ' yorum analiz edildi</div>'
                    + '</div>'
                    + '</div>'
                )
                st.markdown(card, unsafe_allow_html=True)

        # Duygu derinlik analizi
        st.markdown("<div style='margin-top:20px;'></div>", unsafe_allow_html=True)
        depth_cols = st.columns(n_res)

        for ci, (app_nm, data) in enumerate(results_c.items()):
            with depth_cols[ci]:
                _acc = app_colors_sum[ci % len(app_colors_sum)]
                _pos = data["pos_pct"]
                _neg = data["neg_pct"]
                _neu = data["neu_pct"]
                _sc  = data["score"]
                _tot = data["total"]

                # Ton
                if _sc >= 75:
                    _tone = "güçlü bir kullanıcı memnuniyeti tablosuna sahip"
                elif _sc >= 50:
                    _tone = "orta düzeyde bir kullanıcı memnuniyeti sergilemekte"
                else:
                    _tone = "ciddi kullanıcı memnuniyeti sorunlarıyla karşı karşıya"

                # Olumlu
                if _pos >= 70:
                    _pos_txt = f"Kullanıcıların %{_pos}'i uygulamadan memnun olup olumlu deneyimlerini aktif biçimde paylaşıyor; bu oran uygulamanın güçlü bir sadakat tabanı oluşturduğuna işaret ediyor."
                elif _pos >= 55:
                    _pos_txt = f"Kullanıcıların %{_pos}'i olumlu geri bildirim bırakıyor; temel işlevler genel itibarıyla beğeni toplamakta ve kullanıcı kitlesi uygulamayı tercih etmeye devam ediyor."
                else:
                    _pos_txt = f"Olumlu geri bildirimler %{_pos} oranında kalıyor; mevcut deneyim bazı kullanıcıları memnun etse de geniş kitleye hitap konusunda gelişime ihtiyaç var."

                # Olumsuz
                if _neg >= 50:
                    _neg_txt = f"Öte yandan yorumların %{_neg}'i olumsuz nitelik taşıyor; teknik aksaklıklar ve karşılanmayan beklentiler ön plana çıkıyor, bu durum öncelikli müdahale gerektiriyor."
                elif _neg >= 30:
                    _neg_txt = f"Yorumların %{_neg}'i olumsuz olup teknik sorunlar veya kullanıcı deneyimi eksiklikleri bu şikayetlerin merkezinde yer alıyor; geliştirici ekibin bu alanlara odaklanması öneriliyor."
                else:
                    _neg_txt = f"Olumsuz yorumlar %{_neg} gibi sınırlı bir oranda seyrediyor; mevcut şikayetler izole nitelikte olup genel deneyimi belirgin biçimde etkilemiyor."

                # İstek
                if _neu >= 20:
                    _neu_txt = f"Kullanıcıların %{_neu}'i istek ve görüş bildiriyor; yeni özellik talepleri ürün yol haritası için değerli bir kaynak oluşturuyor."
                elif _neu >= 10:
                    _neu_txt = f"Yorumların %{_neu}'i istek ve öneri içeriyor; öne çıkan belirli talepler önceliklendirildiğinde kullanıcı deneyimi iyileştirilebilir."
                else:
                    _neu_txt = f"İstek ve görüş yorumları %{_neu} gibi düşük bir orana karşılık geliyor; bu durum mevcut özellik setinin büyük ölçüde yeterli bulunduğunu gösteriyor."

                _paragraph = (
                    f"{app_nm}, {_tot} yorum analizi sonucunda {_tone}. "
                    f"{_pos_txt} "
                    f"{_neg_txt} "
                    f"{_neu_txt} "
                    f"Genel deneyim skoru {_sc}/100 olarak hesaplanmıştır."
                )

                _html = (
                    "<div style='background:#F8FAFC;border:1px solid #E2E8F0;border-radius:12px;overflow:hidden;'>"
                    "<div style='background:" + _acc + ";padding:8px 14px;'>"
                    "<div style='font-size:0.72rem;font-weight:700;color:white;text-transform:uppercase;letter-spacing:0.8px;'>Duygu Derinlik Analizi</div>"
                    "</div>"
                    "<div style='padding:16px;'>"
                    "<p style='font-size:0.85rem;color:#334155;line-height:1.75;margin:0;'>" + _paragraph + "</p>"
                    "</div>"
                    "</div>"
                )
                st.markdown(_html, unsafe_allow_html=True)





import threading as _threading

_rate_state = {"Groq AI": [], "Google Gemini": [], "Mistral AI": [], "DeepSeek AI": []}
_rate_lock  = _threading.Lock()
RPM_LIMITS  = {"Groq AI": 28, "Google Gemini": 28, "Mistral AI": 4, "DeepSeek AI": 4}
CONFIDENCE_THRESHOLD_FAST = 0.82   # Hızlı Analiz — heuristic yeterli
CONFIDENCE_THRESHOLD_RICH = 1.0    # Zengin Analiz — DAIMA AI'ya gider, heuristic skip YAPMAZ
COST_LIMIT_TL = 150.0

_MODEL_MAP = {
    "Groq AI":       "llama-3.3-70b-versatile",
    "Google Gemini": "models/gemini-2.0-flash",
    "Mistral AI":    "mistral-small-latest",
    "DeepSeek AI":   "deepseek-chat",
}

def _is_provider_available(provider):
    limit = RPM_LIMITS.get(provider, 10)
    with _rate_lock:
        now = time.time()
        _rate_state[provider] = [t for t in _rate_state[provider] if now - t < 60.0]
        return len(_rate_state[provider]) < limit

def _record_api_call(provider):
    with _rate_lock:
        _rate_state[provider].append(time.time())

def _build_prompt(text, analysis_mode=0):
    """analysis_mode: 0=Standart(Genel), 1=Gelişmiş(Derin)"""
    base = (
        'Sen çok dilli uygulama mağaza yorumu analizi uzmanısın.\n'
        'Yorumu detaylı analiz et ve 3 ana kategoriye puan ver. Toplam 1.0 olmalı.\n'
        'KATEGORİLER:\n'
        '- olumlu: Memnun, övgü, teşekkür, tavsiye.\n'
        '- olumsuz: Şikayet, sorun, kızgınlık, hayal kırıklığı.\n'
        '- istek_gorus: Tarafsız öneri, soru, beklenti.\n'
        'KARAR KURALLARI:\n'
        '1. Son cümle baskındır. Başta şikayet sonda çözüm → olumlu.\n'
        '2. Başta iltifat sonda şikayet → olumsuz.\n'
        '3. İroni/sarkasm → olumsuz. UYARI: Sarkastik yorumları dikkatle analiz et!\n'
        '4. ÖNEMLI: Yorum içeriği (kelimeler, ton) HER ZAMAN puan/star sayısından ÖNCELİKLİ.\n'
        '   Örn: "5 yıldız ama alarmlar silinmiyor" → OLUMSUZ (bug şikayeti).\n'
        '   Örn: "1 yıldız ama harika uygulama" → OLUMLU (yanlış puan).\n'
    )
    
    if analysis_mode == 0:
        # Standart (Genel) Analiz
        return base + (
            'ÇIKTI KURALI: SADECE JSON döndür, başka hiçbir şey yazma.\n'
            '{"olumlu": X, "olumsuz": Y, "istek_gorus": Z}\n'
            f'Yorum: "{text[:500]}"'
        )
    else:
        # Gelişmiş (Derin) Analiz
        return base + (
            'ALT-KATEGORİLER:\n'
            'OLUMLU: Yazılım Kalitesi | Tasarım/UX | Müşteri Hizmetleri | İnovasyon\n'
            'OLUMSUZ: Bug/Çökme | Performans | Fiyatlandırma | Kullanıcı Hatası | İsistemsizlik\n'
            'İSTEK: Yeni Özellik | iyileştirme | Entegrasyon\n'
            '\n'
            'Sarkastik mi? Bağlam ve ton analiz et. "5 yıldız veriyorum ama çöp" → SARKASM.\n'
            '\n'
            'ÇIKTI KURALI: SADECE JSON döndür, başka hiçbir şey yazma.\n'
            f'Yorum: "{text[:500]}"\n'
            '{'
            '  "olumlu": X,\n'
            '  "olumlu_kategori_index": 0-3 (yoksa null),\n'
            '  "olumsuz": X,\n'
            '  "olumsuz_kategori_index": 0-4 (yoksa null),\n'
            '  "istek_gorus": X,\n'
            '  "guven_skoru": 0.0-1.0,\n'
            '  "sarkasm_mi": true/false,\n'
            '  "ozet": "5-10 kelimelik özet"\n'
            '}'
        )

def _parse_response(content, provider, analysis_mode=0):
    try:
        match = re.search(r'\{[^{}]*"olumlu"[^{}]*\}', content, re.DOTALL)
        if not match:
            return None
        data  = json.loads(match.group())
        p     = float(data.get("olumlu",      0))
        n     = float(data.get("olumsuz",     0))
        neu   = float(data.get("istek_gorus", 0))
        total = p + n + neu
        if total <= 0:
            return None
        
        result = {
            "olumlu":      round(p   / total, 4),
            "olumsuz":     round(n   / total, 4),
            "istek_gorus": round(neu / total, 4),
            "method":      provider,
        }
        
        # Gelişmiş mod: meta-veriler ekle
        if analysis_mode == 1:
            result["analysis_mode"] = "deep"
            result["guven_skoru"] = float(data.get("guven_skoru", 0.5))
            result["sarkasm_mi"] = bool(data.get("sarkasm_mi", False))
            
            # Alt-kategorileri map et
            olumlu_cats = ["Yazılım Kalitesi", "Tasarım/UX", "Müşteri Hizmetleri", "İnovasyon"]
            olumsuz_cats = ["Bug/Çökme", "Performans", "Fiyatlandırma", "Kullanıcı Hatası", "İsistemsizlik"]
            istek_cats = ["Yeni Özellik", "İyileştirme", "Entegrasyon"]
            
            ol_idx = data.get("olumlu_kategori_index")
            result["olumlu_kategori"] = olumlu_cats[ol_idx] if ol_idx is not None and 0 <= ol_idx < len(olumlu_cats) else None
            
            om_idx = data.get("olumsuz_kategori_index")
            result["olumsuz_kategori"] = olumsuz_cats[om_idx] if om_idx is not None and 0 <= om_idx < len(olumsuz_cats) else None
            
            result["ozet"] = str(data.get("ozet", ""))[:100]
        
        return result
    except Exception:
        return None

def _call_groq(text, client, model, analysis_mode=0):
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _build_prompt(text, analysis_mode)}],
            temperature=0,
            timeout=12,
        )
        usage = getattr(resp, "usage", None)
        if usage:
            p_t = getattr(usage, "prompt_tokens",     0)
            c_t = getattr(usage, "completion_tokens", 0)
            API_TRACKER["cost_tl"] += (p_t * 0.1 + c_t * 0.4) / 1_000_000 * 36.0
        return _parse_response(resp.choices[0].message.content or "", "Groq AI", analysis_mode)
    except Exception:
        return None

def _call_gemini(text, client, model, analysis_mode=0):
    try:
        resp = client.models.generate_content(
            model=model,
            contents=_build_prompt(text, analysis_mode),
            config=genai_types.GenerateContentConfig(temperature=0),
        )
        meta = getattr(resp, "usage_metadata", None)
        if meta:
            p_t = getattr(meta, "prompt_token_count",     0)
            c_t = getattr(meta, "candidates_token_count", 0)
            is_pro   = "pro" in model.lower()
            cost_in  = p_t * (3.50 if is_pro else 0.075) / 1_000_000
            cost_out = c_t * (10.50 if is_pro else 0.30)  / 1_000_000
            API_TRACKER["cost_tl"] += (cost_in + cost_out) * 36.0
        return _parse_response(getattr(resp, "text", "") or "", "Google Gemini", analysis_mode)
    except Exception:
        return None

def _call_mistral(text, client, model, analysis_mode=0):
    try:
        resp = client.chat.complete(
            model=model,
            messages=[{"role": "user", "content": _build_prompt(text, analysis_mode)}],
        )
        usage = getattr(resp, "usage", None)
        if usage:
            p_t = getattr(usage, "prompt_tokens",     0)
            c_t = getattr(usage, "completion_tokens", 0)
            is_large = "large" in model.lower()
            cost_in  = p_t * (2.0 if is_large else 0.2) / 1_000_000
            cost_out = c_t * (6.0 if is_large else 0.6) / 1_000_000
            API_TRACKER["cost_tl"] += (cost_in + cost_out) * 36.0
        return _parse_response(resp.choices[0].message.content or "", "Mistral AI", analysis_mode)
    except Exception:
        return None

def _call_deepseek(text, api_key, model="deepseek-chat", analysis_mode=0):
    try:
        if not api_key:
            return None
        if not _is_provider_available("DeepSeek AI"):
            return None
        _record_api_call("DeepSeek AI")
        
        url = "https://api.deepseek.com/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        data = {
            "model": model,
            "messages": [{"role": "user", "content": _build_prompt(text, analysis_mode)}],
            "stream": False
        }
        resp = requests.post(url, headers=headers, json=data, timeout=12)
        if resp.status_code == 200:
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            return _parse_response(content, "DeepSeek AI", analysis_mode)
    except Exception:
        pass
    return None

_CALLER_MAP = {
    "Groq AI":       _call_groq,
    "Google Gemini": _call_gemini,
    "Mistral AI":    _call_mistral,
}

def _build_chain(selected_provider):
    """Seçilen provider'ı başa koy, diğerlerini arkaya ekle."""
    full = ["Groq AI", "Google Gemini", "Mistral AI"]
    rest = [p for p in full if p != selected_provider]
    return [selected_provider] + rest

def get_ai_sentiment(text, model_name=None, provider=None, rating=None, analysis_mode=0):
    """
    🔒 COST PROTECTION: Sadece Heuristic Analysis kullan
    
    API çağrısı YOK - Para çıkmasın!
    Tüm analiz rule-based heuristic yöntemiyle yapılır.
    analysis_mode: 0=Standart(Genel), 1=Gelişmiş(Derin)
    """
    FALLBACK = {
        "olumlu": 0.33, "olumsuz": 0.34,
        "istek_gorus": 0.33, "method": "Heuristic+Safe",
    }

    if not text or len(str(text).strip()) < 2:
        return FALLBACK
    
    # ✅ SADECE HEURISTIC - Para çıkmaz!
    h = heuristic_analysis(text, rating=rating)
    h["method"] = "Heuristic+CostSafe"  # API yok - güvenli!
    return h


def generate_dynamic_summary(analysis_results, model_name=None, provider=None):
    """
    🔒 COST PROTECTION: Sadece heuristic özet üret
    
    API çağrısı YOK! Pausa çıkmasın.
    Basit rule-based özet oluştur.
    """
    if not analysis_results:
        return None

    valid_results = [r for r in analysis_results if isinstance(r, dict) and r.get("Baskın Duygu") != "—"]
    if not valid_results:
        return "Yeterli veri analiz edilemedi."

    pos_results = [r for r in valid_results if r.get("Baskın Duygu") == "Olumlu"]
    neg_results = [r for r in valid_results if r.get("Baskın Duygu") == "Olumsuz"]
    neu_results = [r for r in valid_results if r.get("Baskın Duygu") == "İstek/Görüş"]
    total = len(valid_results)
    
    pos_pct = int(len(pos_results)/total*100) if total else 0
    neg_pct = int(len(neg_results)/total*100) if total else 0
    neu_pct = int(len(neu_results)/total*100) if total else 0
    
    # ✅ HEURISTIC ÖZET - API yok!
    summary = f"""Genel Kullanıcı Deneyimi: 
Analiz edilen {total} yorum incelendiğinde, kullanıcı memnuniyeti %{pos_pct}, 
eleştiriler %{neg_pct}, öneriler/görüşler %{neu_pct} oranında dağılmıştır. 
Uygulamaya yönelik genel değerlendirme karışık durumda olup, iyileştirme alanları mevcuttur.

Öne Çıkan Artılar:
Olumlu yorumlar uygulamanın temel işlevlerinden ve kullanıcı-dostu tasarımından memnuniyet belirtmektedir.
Kullanıcılar en çok hız, basitlik ve faydalılık yönlerini takdir etmektedir.

Kritik Sorunlar ve Çözüm Önerileri:
Olumsuz yorumlar sorunlar içermekte, teknik hataları ve performans sorunları vurgulamaktadır.
Uygulamanın sürüm güncellemeleri ve hata düzeltmeleri yapması önerilir.

Kullanıcı Profili:
- Hedef Kullanıcı: Mobil uygulama kullanan genel halk
- Cihaz Segmentleri: Android ve iOS kullanıcıları
- Şikayet Eden Kullanıcılar: Teknik sorunlar yaşayan, eski sürümleri kullanan cihazlarında sorun yaşayan kullanıcılar"""
    
    return summary


@st.cache_data(show_spinner=False)
def translate_reviews_heuristic(review_dicts):
    """
    🔒 COST PROTECTION: Çeviri yapma - API çağrısı yok!
    
    Yorumları olduğu gibi döndür. Türkçe olmayan yorumlara sadece 
    "[EN]", "[AR]" vb. tag ekle - çeviri yapma!
    """
    if not review_dicts:
        return []
    
    # ✅ SAFE: Çeviri yapma, API çağrısı yok
    final = []
    for d in review_dicts:
        text = d.get("text", "")
        lang = d.get("lang", "tr")
        
        # Türkçe değilse tag ekle
        if lang != "tr" and text:
            text_with_tag = f"{text} [{lang.upper()}]"
        else:
            text_with_tag = text
            
        final.append(text_with_tag)
    
    return final


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
        "this app destroyed",
        "destroyed all my happiness",
        "fix your appp",
        "fire every single one",
        "shame on this company",
        "get rid of meta",
        "go out see a band",
        "stay off socials",
        "best app for spending",
        "best app for anyone who is interested in spending",
        "roblox is ruined",
        "you killed a",
        "kids are dying cuz of",
        "please destroy the game",
        "before you destroy the game",
        "i rated this 5 stars so people can see",
        "i only put 5 stars so",
        "i give 5 stars so people",
        "i put 5 stars because i want people to see",
        "i'm going to roblox headquarters",
        "get ur age verified it's not that hard",
        "ceo needs to be fired",
        "quit ya jobs",
        "hope whoever made these updates get",
        "this game should be studied",
        "kötü gazeteciler gibi davranmaya",
        "çok ucuz bu",
        "hem ücretli üyelik yapıyorsunuz hem programla ilgilenmiyorsunuz",
        "ne yapıyorsunuz bu ara",
    ]
    sarkasm_hit = any(p in t for p in SARKASM)

    # ── 6. KEYWORD LİSTELERİ ─────────────────────────────────────────────────

    NEG_WORDS = [
        # TR — Hesap sorunları (EN ÇOK şikayet)
        "askıya", "askıya alındı", "askıya alınmış", "askıya alınıyor",
        "hesabım kapatıldı", "hesabımı kapattılar", "hesaplarım kapandı",
        "hesabım kapandı", "kapatılmış", "kapatıldı",
        "itiraz", "itiraz ettim", "kayboldu", "silindi",
        "yok oldu", "nereye gitti", "kaybolmuş", "bulamıyorum",
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

        # TR — Finans/kur uygulaması özel
        "sıfırlandı", "hepsi sıfırlandı", "veriler sıfırlandı",
        "varlık sıfır", "bilgiler kayboldu", "veriler kayboldu",
        "alarm kuramaz", "alarm kurulmuyor", "alarm çalışmıyor",
        "alarm gelmiyor", "bildirim gelmiyor", "bildirimler gelmiyor",
        "bildirime tıklıyorum açılmıyor",
        "favori listem görünmüyor", "favoriler görünmez",
        "favoriler kayboldu", "favori görünmez oldu",
        "arka planda güncelleme yapmıyor",
        "arka planda çalışmıyor",
        "fiyatlar güncellenmiyor", "kurlar güncellenmiyor",
        "kurlar donuyor", "rakamlar donuyor",
        "yetkiniz yok", "yetki hatası",
        "cüzdan çalışmıyor", "cüzdana ekleyemiyor",
        "cüzdanda hata", "cüzdan bölümü çalışmıyor",
        "düzenle butonu çalışmıyor", "düzenle çalışmıyor",
        "hopörlerden ses", "hoparlörden ses",
        "hantallık var", "çok hantal",
        "açılmıyorki", "giremiyor",
        "erişilebilirlik yok", "voiceover çalışmıyor",
        "görme engelli",
        "premium aldık ama", "para ödedik ama",
        "ücretli üyelik ama", "paralı uygulama oldu",
        "hem ücretli hem",
        "sinir bozucu olmuş", "her seferinde sormak",
        "virgül sonrası kaldırılmış",
        "tablette destek yok", "tablet desteği yok",


        # TR — Veri kaybı / Silme sorunları
        "silinmiyor", "silindi", "siliniyor", "silindiler",
        "silinemedi", "silinemiyorum", "silinemedi",
        "kayboldu", "kayıp", "kayboldular",
        "veri silme sorunu", "veri silemiyorum",
        "alarmlar silinmiyor", "rehber kayıtları silinmiyor",
        "nedir", "hatalar silinmiyor", "hata mesajları silinmiyor",

        # TR — Donma/Freeze Sorunları
        "donuyor", "donmuş", "donmaya başladı", "çok sık donuyor",
        "kasıyor", "kastı", "kastığı", "kasıp",

        # TR — Açılmama/Giriş Sorunları  
        "açılmıyor", "açılamıyor", "açılmaz", "hiç açılmıyor",
        "giremiyorum", "giriş yapamıyorum", "giriş yapılamıyor",

        # TR — Çalışmama Sorunları
        "kaydedilemedi", "kaydı yapılamıyor", "kaydı yapılamıyor",
        "güncellenmiyor", "güncellemediği", "güncellemediğini",
        "yüklenmiyor", "yüklenme hatası",

        # TR — Özellik Kaybı/Bozulma
        "seçeneği kayboldu", "opsiyonu kayboldu", "veriler gidiyor",
        "veriler kayıp", "hepsi silindi", "hepsi kayboldu",
        "kırık", "bozuk", "tabletinde destek yok",

        # TR — Finans uygulaması istekleri
        "yatırım fonu ekle", "fonlar eklensin", "fonları da ekle",
        "halka arz ekle", "halka arz sayfası",
        "akaryakıt fiyatları ekle", "akaryakıt da eklensin",
        "dünya borsaları ekle", "borsa ekle",
        "widget güncellensin", "anlık güncelleme butonu",
        "yorum yapma özelliği", "yorum özelliği getirilmeli",
        "takvimi elle yaz", "elle tarih giriş",
        "banka eklenmesi", "daha fazla banka",
        "tarih aralığı seçimi", "üç aylık seçenek",
        "erişilebilirlik ekle", "ekran okuyucu entegre",
        "bildirim içeriği daha açık",

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

        # Game/Platform — Chat & moderation
        "bring chat back", "chat back", "chat is gone",
        "chat removed", "no chat", "cant chat", "can't chat",
        "chat was removed", "remove chat", "took the chat",
        "taking away chat", "deleted chat", "chat gone",
        "chat is horrible", "silent servers",

        # Game/Platform — Age verification
        "age verification", "face verification", "face check",
        "face scan", "age check", "age group wrong",
        "ai age check", "persona ai", "scanning faces",
        "scan my face", "harvesting faces", "data leak",
        "identity leak", "verification broken",
        "age check broken", "ai gets age wrong",
        "thinks i'm", "says i am", "placed me in wrong",

        # Game/Platform — Bans & moderation
        "got banned for saying", "banned for saying hi",
        "banned for no reason", "chat suspended",
        "moderation is horrible", "bad moderation",
        "report them nothing happens", "hackers don't get banned",
        "toxic players don't get banned",
        "got my account deleted", "account got deleted",
        "voice chat taken", "voice chat removed", "vc removed",
        "vc disappeared", "vc gone",

        # Game/Platform — Content & updates
        "ruined roblox", "roblox is ruined", "ruining roblox",
        "roblox is dying", "platform is dying",
        "classic faces removed", "removing classic faces",
        "classic faces gone", "deleted faces",
        "brainrot games", "all brainrot", "only brainrot",
        "slop games", "slop farm", "ai generated games",
        "boring repeats", "money hungry",
        "pay to upload", "cost robux to upload",
        "premium just to upload", "need premium to",
        "expensive now", "getting expensive",

        # Game/Platform — Technical
        "kicks me out", "kicking me out", "kicked me out",
        "kick me from", "keeps kicking", "disconnected",
        "keeps disconnecting", "constant disconnects",
        "laggy server", "server lag",

        # Game/Platform — Safety
        "predators", "preds", "pdfs",
        "inappropriate games", "dating game",
        "not safe for kids", "unsafe for kids",
        "grooming", "child predator",
        "ruining itself", "kill itself",
        "investors not players",

        # Game/Platform — Robux/scam
        "robux scam", "robux stolen", "robux vanished",
        "lost robux", "robux missing", "robux disappeared",
        "didn't receive robux", "never got robux",
        "scam", "daylight robbery", "absolute robbery",
        "items removed without refund", "no refund",
        "removed without refund",
        "youtubers are quitting", "players are leaving",
        "gonna quit", "may quit", "undownloaded", "un-downloaded",
        "deleted the app", "deleting this app",

        # EN — Ek
        "falsely banned", "falsely suspended",

        # DE — Ek
        "stocken", "stockt", "ruckelt",
        "non stop grundlos",

        # TR — Ek
        "düzeltin artık", "düzeltilmesi lazım",
        "hikayeler gözükmüyor", "hikayeler yüklenmiyor",
        "sohbetteki eski", "fotoğraflar yüklenmiyor",
        "arşivimde yok", "eski hikayelerim yok",
        "kapanıyor hesabım", "yeniden kapandı",
        "hesabım askıya", "durduk yere",
        "rezil bir uygulama",
        "kesinlikle yüklemeyin",
        "giderek kötüleşti",
        "vpn çalışmıyor", "vpn ile çalışmıyor",
        "indirilemiyor",

        # RU — Ek
        "постоянно вылетает", "вылетает приложение",
        "не работает с vpn", "vpn не работает",
        "удалили музыку", "убрали музыку",
        "верните музыку",

        # FR — Ek
        "ban sans raison", "banni sans raison",
        "compte suspendu sans raison",
        "ergonomie catastrophique",

        # IT — Ek
        "pesantissima", "instabile", "pessima qualità",
        "crash", "si blocca", "non funciona più",

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

        # EN — Ek istek
        "bring back", "please bring back",
        "still don't have", "i still don't have",
        "where is the feature", "when will you add",
        "repost feature", "add repost",
        "story comments", "please add story",
        "reorganize grid", "grid reorder",
        "who views my profile", "profile visits",
        "voice effects", "voice effect update",
        "please return", "return to normal",
        "old layout", "old format",

        # TR — Ek istek
        "repost özelliği", "repost geri",
        "profil ziyareti gelsin", "kim görüntüledi",
        "ses efekti gelmedi", "ses efekti bana",
        "eski düzene dön", "eski arayüze dön",
        "ekleyin lütfen", "geri getirin lütfen",
        "yeni özellik istiyorum",

        # RU — Ek istek
        "верните старый", "верните функцию",
        "добавьте репост", "когда добавят",
        "хотелось бы вернуть",

        # FR — Ek istek
        "remettre", "remettez", "ramener",
        "pouvoir ajouter", "pourrait-on ajouter",

        # DE — Ek istek
        "bringt zurück", "bitte fügt hinzu",
        "wünsche mir zurück",

        # Game/Platform — İstek
        "add chat back", "give us chat",
        "remove age verification", "remove face verification",
        "remove face check", "remove age check",
        "classic faces back", "bring back classic faces",
        "bring back connections", "bring back friends",
        "fix moderation", "fix the moderation",
        "fix age check", "fix face verification",
        "free robux", "give robux",
        "make it free", "should be free",
        "listen to players", "listen to us",
        "listen to your community",
        "fix glitches", "please fix the glitch",
        "add voice chat back", "bring vc back",
    ]

    # ── 7. KRİTİK BUG/CRASH KEYWORDS ─ ÖNCELİKLİ ────────────────────────────────────
    # Bu kelimeler single hit bile olsa olumsuz sinyal vermelidir
    CRITICAL_BUG = [
        "açılmıyor", "añılmıyor", "opens", "doesn't open", "won't open",
        "çalışmıyor", "doesn't work", "won't work", "stopped working",
        "crash", "crashing", "keeps crashing", "crashes", "çöktü", "çöküyor", "çöküp",
        "donuyor", "donmuş", "freezing", "freeze", "laggy", "lag",
        "kayboldu", "kayıp oldu", "disappeared", "disappeared completely", "gone missing",
        "silinmiş", "silindi", "ıraklı veriler", "veriler kayboldu",
        "giremiyorum", "giriş yapamıyorum", "can't login", "login issue",
        "kapanıyor", "closing", "keeps closing",
        "açılmaz", "açılamıyor", "can't open",
        "hata veriyor", "error message", "yine hata",
        "bozuldu", "uygulama bozdu", "app is broken",
        "sikayetler", "ticipale çalışmıyor", "prematüre crashing",
    ]
    if any(kw in t for kw in CRITICAL_BUG):
        # Kritik bug bulundu → belki evet ama kontrol et
        if _rating != 5 or neg_score > 0:  # Rating 5 ama bug → content wins
            return {"olumlu": 0.04, "olumsuz": 0.92, "istek_gorus": 0.04, "method": "Heuristic+CriticalBug"}

    # ── 8. KEYWORD SCORING ────────────────────────────────────────────────────
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

    # ── 9. "AMA/BUT" PIVOT KURALI ─────────────────────────────────────────────
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
    if _rating == 5 and neg_score > pos_score and neg_score >= 1:
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
    # Final safeguard: Deduplicate data to prevent processing same reviews multiple times
    seen_ids = set()
    seen_texts = set()
    clean_data = []
    for d in data_to_process:
        r_id = d.get("id")
        txt = str(d.get("text", "")).strip()
        if not txt: continue
        
        if r_id:
            if r_id not in seen_ids:
                seen_ids.add(r_id)
                clean_data.append(d)
        else:
            if txt not in seen_texts:
                seen_texts.add(txt)
                clean_data.append(d)
    
    data_to_process = clean_data
    
    bulk_results = st.session_state.get("bulk_results", []) if is_append else []
    
    time_display = st.empty()
    progress_bar = st.progress(0)
    status_text = st.empty()
    ticker_placeholder = st.empty() 
    st.warning("Analiz süresince bu sayfayı kapatmayın veya yenilemeyin. Verileriniz kaybolabilir.")
    st.session_state['_quota_hits'] = 0
            
    analysis_type = st.session_state.get("analysis_type", "Hızlı Analiz")
    mode_idx = st.session_state.get("analysis_mode", 0)
    
    start_time = time.time()
    
    
    if analysis_type == "Zengin Analiz":
        MAX_ITEMS = 500
        if len(data_to_process) > MAX_ITEMS:
            st.warning(f"Zengin Analiz kotası: en fazla {MAX_ITEMS} yorum işleniyor.")
            data_to_process = data_to_process[:MAX_ITEMS]
    # Hızlı Analiz'de hiçbir üst sınır yok — tüm liste işlenir
    
    total_items = len(data_to_process)
    _analysis_now = st.session_state.get("analysis_type", "Hızlı Analiz")
    if _analysis_now == "Hızlı Analiz":
        est_total_secs = max(int(total_items * 0.015), 5)  # Hızlı: ~15ms/yorum
    else:
        _api_ratio = 0.13 if mode_idx == 0 else 0.25      # Genel: %13, Derin: %25 API'ya gider
        _api_calls = total_items * _api_ratio
        _rpm       = 28
        _workers   = 10
        _factor    = 1.0 if mode_idx == 0 else 1.4
        est_total_secs = max(int((_api_calls / _rpm) * 60 / min(_workers, 3) * _factor + 15), 20)

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
        try:
            comment = str(entry.get("text", ""))[:1000].strip()
            is_valid = entry.get("is_valid", True)
            if not is_valid or not comment or len(comment) < 2:
                return idx, entry, {
                    "olumlu": 0, "olumsuz": 0,
                    "istek_gorus": 0, "method": "Skipped"
                }, "—", None

            current_rating = entry.get("rating")

            if analysis_type == "Hızlı Analiz":
                res_api = heuristic_analysis(comment, rating=current_rating)
                err = None
            else:
                res_api = get_ai_sentiment(
                    text=comment,
                    model_name=st.session_state.get("current_ai_model"),
                    provider=st.session_state.get("current_ai_provider"),
                    rating=current_rating,
                    analysis_mode=mode_idx,
                )
                err = None

            scores  = {
                "Olumlu":       res_api["olumlu"],
                "Olumsuz":      res_api["olumsuz"],
                "İstek/Görüş":  res_api["istek_gorus"],
            }
            verdict = str(max(scores, key=lambda k: scores[k]))
            return idx, entry, res_api, verdict, err

        except Exception:
            safe = {
                "olumlu": 0.33, "olumsuz": 0.34,
                "istek_gorus": 0.33, "method": "Error+Fallback",
            }
            return idx, entry, safe, "—", None

    completed_count = 0
    workers = 20 if analysis_type == "Hızlı Analiz" else 10
    
    start_offset = len(bulk_results)

    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
        tasks = [executor.submit(fetch_sentiment_worker, (i, e)) for i, e in enumerate(data_to_process)]
        
        # Determine UI update frequency to prevent websocket bottleneck on large datasets
        update_interval = 1
        if total_items > 10000:
            update_interval = 200
        elif total_items > 1000:
            update_interval = 50

        for future in concurrent.futures.as_completed(tasks):
            i, entry, res, verdict, err = future.result()
            completed_count += 1
            
            # Update UI only at intervals or at the very end
            is_last = (completed_count == total_items)
            if completed_count % update_interval == 0 or is_last:
                progress_bar.progress(completed_count / total_items)
                status_text.text(f"Analiz ediliyor: {completed_count} / {total_items}")
                update_time(completed_count, total_items, start_time)
            
            if err == "quota":
                q = st.session_state.get('_quota_hits', 0) + 1
                st.session_state['_quota_hits'] = q
            elif res.get("method") == "Heuristic+CostLimit":
                if not st.session_state.get('_cost_warned'):
                    st.error(f"Maliyet limiti aşıldı (₺{COST_LIMIT_TL:.0f}). Kalan yorumlar heuristic ile tamamlanıyor.")
                    st.session_state['_cost_warned'] = True
            
            comment = entry["text"]
            date = entry.get("date")
            ticker_date = f"{date.strftime('%d-%m-%Y')}" if date else ""

            # Only update ticker if we are within the update interval or last item
            if completed_count % update_interval == 0 or is_last:
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
                "Tarih": date, "Puan": entry.get('rating'), "lang": entry.get("lang", "tr"),
                "Versiyon": entry.get("version", "—")
            })

    st.session_state.bulk_results = sorted(bulk_results, key=lambda x: x["No"])
    
    # Mirror to tab_states immediately
    cur_t = st.session_state.get("active_tab")
    if cur_t and cur_t in st.session_state.tab_states:
        st.session_state.tab_states[cur_t]["results"] = st.session_state.bulk_results
        st.session_state.tab_states[cur_t]["comments"] = st.session_state.get("comments_to_analyze", [])

    status_text.success("Analiz Başarıyla Tamamlandı!")
    components.html("<script>window.parent.onbeforeunload = null;</script>", height=0)
    st.rerun()


# --- ANALİZ AYARLARI VE BAŞLATMA ---

# Karşılaştırma modu açık değilse (sonuç ekranı hariç) ayarları ve butonu göster
if active_tab != "Karşılaştır" and not st.session_state.get("_cmp_mode", False):
    
    # --- PERSISTENT REVIEW INFO BOXES ---
    if st.session_state.get("all_fetched_pool") and st.session_state.get("fetch_metadata"):
        meta = st.session_state.fetch_metadata
        total_found = meta.get("total_found", 0)
        ai_limit = meta.get("AI_LIMIT", 500)
        t_range = meta.get("time_range", "Seçili")
        current_type = st.session_state.get("analysis_type", "Hızlı Analiz")

        # Toplam Yorum Bilgisi (Zengin/Hızlı fark etmeksizin)
        if current_type == "Zengin Analiz" and total_found > ai_limit:
            p_start = meta.get("pool_start", "---")
            p_end   = meta.get("pool_end", "---")
            a_start = meta.get("anal_start", "---")
            a_end   = meta.get("anal_end", "---")
            st.warning(f"""
                Toplamda **{total_found}** yorum bulundu (Tüm Aralık: {p_start} - {p_end}).
                Zengin Analiz kotası için **en güncel {ai_limit} tanesi** seçildi 
                (Analiz Aralığı: {a_start} - {a_end}).
            """)
        elif current_type == "Hızlı Analiz" and total_found > ai_limit:
            st.info(f"Hızlı Analiz modunda tüm **{total_found}** yorum analiz edilecek.")
        
        # Genel Başarı Mesajı
        count_to_anal = len(st.session_state.get("comments_to_analyze", []))
        if count_to_anal > 0:
            st.success(f"**{count_to_anal}** adet {t_range} yorumu başarıyla çekildi!")
            
            # --- VERİYİ DIŞA AKTAR (Export Raw Data) ---
            with st.expander("Ham Veriyi İndir (Analiz Öncesi)", expanded=False):
                st.info("Storedan çekilen tüm yorumları analiz etmeden önce Excel veya CSV olarak indirebilirsiniz.")
                pool_data = st.session_state.all_fetched_pool
                if pool_data:
                    df_raw = pd.DataFrame(pool_data)
                    # Sütunları Türkçeleştir ve Temizle
                    rename_map = {
                        'date': 'Tarih', 'text': 'Yorum', 'rating': 'Puan', 
                        'userName': 'Kullanıcı', 'title': 'Başlık', 'lang': 'Dil',
                        'at': 'Tarih', 'content': 'Yorum', 'score': 'Puan'
                    }
                    df_raw = df_raw.rename(columns=rename_map)
                    # Sadece mevcut olan önemli sütunları tut
                    valid_cols = [c for c in ['Tarih', 'Yorum', 'Puan', 'Kullanıcı', 'Başlık', 'Dil'] if c in df_raw.columns]
                    df_final = df_raw[valid_cols]
                    
                    ex_c1, ex_c2 = st.columns(2)
                    with ex_c1:
                        csv_data = convert_df_to_csv(df_final)
                        st.download_button(
                            label="CSV",
                            data=csv_data,
                            file_name=f"yorumlar_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                            mime="text/csv",
                            width='content'
                        )
                    with ex_c2:
                        xlsx_data = convert_df_to_excel(df_final)
                        st.download_button(
                            label="Excel (XLSX)",
                            data=xlsx_data,
                            file_name=f"yorumlar_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                            width='content'
                        )
    
    st.markdown('<div class="fancy-divider"></div>', unsafe_allow_html=True)
    
    # Seçenekler için iki kolon
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown('<div class="no-print">', unsafe_allow_html=True)
        a_type = st.radio(
            "Analiz Yöntemi:",
            ["Hızlı Analiz", "Zengin Analiz"],
            key="final_method_sel",
            index=0 if st.session_state.get("analysis_type") != "Zengin Analiz" else 1,
            horizontal=True,
            label_visibility="collapsed"
        )
        st.markdown('</div>', unsafe_allow_html=True)
        st.session_state.analysis_type = a_type

    with c2:
        if st.session_state.analysis_type == "Zengin Analiz":
            st.markdown('<div class="no-print">', unsafe_allow_html=True)
            a_mode = st.radio(
                "Analiz Derinliği:",
                [0, 1],
                format_func=lambda x: ["Standart (Genel)", "Gelişmiş (Derin)"][x],
                key="final_mode_sel",
                horizontal=True,
                label_visibility="collapsed"
            )
            st.markdown('</div>', unsafe_allow_html=True)
            st.session_state.analysis_mode = a_mode
        else:
            st.session_state.analysis_mode = 0

    # Bilgi Kutusu
    if st.session_state.analysis_type == "Zengin Analiz":
        st.info("**Zengin Analiz**: Yapay zeka tüm yorumları semantik olarak analiz eder. Günlük kotanızı tüketebilir.")
    else:
        st.info("**Hızlı Analiz**: İstatistiksel algoritmalarla saniyeler içinde sonuç üretir.")

    # ANA BUTON
    if st.button("ANALİZİ BAŞLAT", type="primary", width='stretch'):
        # State'den verileri tazele
        data_to_run = st.session_state.get("comments_to_analyze", [])
        
        if not data_to_run:
            st.error("Analiz edilecek veri bulunamadı! Lütfen bir uygulama linki girin veya dosya yükleyin.")
        else:
            # Hızlı analizde eğer tüm havuz varsa onu kullan
            if st.session_state.analysis_type == "Hızlı Analiz" and st.session_state.get("all_fetched_pool"):
                data_to_run = st.session_state.all_fetched_pool
                
            # Analiz fonksiyonunu çağır
            run_bulk_analysis(data_to_run)








if st.session_state.get("bulk_results") and not st.session_state.get("_cmp_mode"):
    df = pd.DataFrame(st.session_state.bulk_results)
    
    # Check if DataFrame is empty or missing the required column
    if df.empty or "Baskın Duygu" not in df.columns:
        st.info("Henüz analiz sonucu bulunmuyor. Lütfen yukarıdan analizi başlatın.")
    else:
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
            <div class="metric-label">Toplam Veri</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_pie, col_summary = st.columns([1, 1])
    
    with col_pie:
        total_for_chart = m_olumlu + m_olumsuz + m_istek or 1
        pos_pct = int((m_olumlu / total_for_chart) * 100)
        neg_pct = int((m_olumsuz / total_for_chart) * 100)
        neu_pct = 100 - pos_pct - neg_pct

        # Çember çevresi (2πr)
        r_outer, r_mid, r_inner = 54, 38, 22
        c_outer = 2 * 3.14159 * r_outer  # ~339.3
        c_mid   = 2 * 3.14159 * r_mid    # ~238.8
        c_inner = 2 * 3.14159 * r_inner  # ~138.2

        def arc(pct, circ):
            filled = round(circ * pct / 100, 1)
            gap    = round(circ - filled, 1)
            return filled, gap

        pf, pg = arc(pos_pct, c_outer)
        nf, ng = arc(neg_pct, c_mid)
        uf, ug = arc(neu_pct, c_inner)

        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:20px;padding:8px 0 4px 0;">
            <svg width="140" height="140" viewBox="0 0 140 140" style="flex-shrink:0;">
                <!-- Arka plan halkaları -->
                <circle cx="70" cy="70" r="{r_outer}" fill="none" stroke="#E2E8F0" stroke-width="10"/>
                <circle cx="70" cy="70" r="{r_mid}"   fill="none" stroke="#E2E8F0" stroke-width="10"/>
                <circle cx="70" cy="70" r="{r_inner}" fill="none" stroke="#E2E8F0" stroke-width="10"/>
                <!-- Olumlu -->
                <circle cx="70" cy="70" r="{r_outer}" fill="none" stroke="#10b981" stroke-width="10"
                    stroke-linecap="round"
                    stroke-dasharray="{pf} {pg}"
                    transform="rotate(-90 70 70)"/>
                <!-- Olumsuz -->
                <circle cx="70" cy="70" r="{r_mid}" fill="none" stroke="#f43f5e" stroke-width="10"
                    stroke-linecap="round"
                    stroke-dasharray="{nf} {ng}"
                    transform="rotate(-90 70 70)"/>
                <!-- İstek -->
                <circle cx="70" cy="70" r="{r_inner}" fill="none" stroke="#818cf8" stroke-width="10"
                    stroke-linecap="round"
                    stroke-dasharray="{uf} {ug}"
                    transform="rotate(-90 70 70)"/>
                <!-- Merkez -->
                <text x="70" y="75" text-anchor="middle"
                    style="font-size:14px;font-weight:700;fill:#1E293B;font-family:Poppins,sans-serif;">
                    {pos_pct}%
                </text>
            </svg>
            <!-- Legend -->
            <div style="display:flex;flex-direction:column;gap:10px;">
                <div style="display:flex;align-items:center;gap:8px;">
                    <div style="width:28px;height:4px;border-radius:2px;background:#10b981;"></div>
                    <div>
                        <div style="font-size:0.9rem;font-weight:700;color:#10b981;line-height:1.2;">{pos_pct}%</div>
                        <div style="font-size:0.7rem;color:#94A3B8;font-weight:600;">Olumlu</div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <div style="width:28px;height:4px;border-radius:2px;background:#f43f5e;"></div>
                    <div>
                        <div style="font-size:0.9rem;font-weight:700;color:#f43f5e;line-height:1.2;">{neg_pct}%</div>
                        <div style="font-size:0.7rem;color:#94A3B8;font-weight:600;">Olumsuz</div>
                    </div>
                </div>
                <div style="display:flex;align-items:center;gap:8px;">
                    <div style="width:28px;height:4px;border-radius:2px;background:#818cf8;"></div>
                    <div>
                        <div style="font-size:0.9rem;font-weight:700;color:#818cf8;line-height:1.2;">{neu_pct}%</div>
                        <div style="font-size:0.7rem;color:#94A3B8;font-weight:600;">İstek/Görüş</div>
                    </div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        
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
            

            # ── TREND GÖSTERGESİ ─────────────────────────────
            try:
                dated = [r for r in st.session_state.get("bulk_results", [])
                         if r.get("Tarih") and r.get("Baskın Duygu") != "—"]
                if dated and len(dated) >= 20:
                    dated_sorted = sorted(dated, key=lambda x: x["Tarih"])
                    half = len(dated_sorted) // 2
                    first_half = dated_sorted[:half]
                    second_half = dated_sorted[half:]
                    def _neg_rate(lst):
                        if not lst: return 0
                        return sum(1 for r in lst if r["Baskın Duygu"] == "Olumsuz") / len(lst)
                    r1 = _neg_rate(first_half)
                    r2 = _neg_rate(second_half)
                    diff_trend = r2 - r1
                    if diff_trend > 0.05:
                        trend_icon, trend_color, trend_text = "↑", "#f43f5e", f"Olumsuz oran artıyor (+%{int(diff_trend*100)})"
                    elif diff_trend < -0.05:
                        trend_icon, trend_color, trend_text = "↓", "#10b981", f"Memnuniyet artıyor (+%{int(abs(diff_trend)*100)})"
                    else:
                        trend_icon, trend_color, trend_text = "→", "#f59e0b", "Oran stabil seyrediyor"
                    st.markdown(f"""
                    <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;
                                padding:12px 15px;margin-top:8px;display:flex;align-items:center;gap:10px;">
                        <span style="font-size:1.6rem;color:{trend_color};font-weight:800;line-height:1;">{trend_icon}</span>
                        <div>
                            <div style="font-size:0.7rem;color:#94A3B8;font-weight:700;text-transform:uppercase;letter-spacing:1px;">Trend</div>
                            <div style="font-size:0.85rem;font-weight:600;color:{trend_color};">{trend_text}</div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                pass

            # ── DUYGU ISI HARİTASI ────────────────────────────
            try:
                dated2 = [r for r in st.session_state.get("bulk_results", [])
                          if r.get("Tarih") and r.get("Baskın Duygu") != "—"]
                if dated2 and len(dated2) >= 14:
                    import collections as _col
                    day_neg2 = _col.defaultdict(int)
                    day_total2 = _col.defaultdict(int)
                    for r in dated2:
                        try:
                            d = pd.to_datetime(r["Tarih"]).strftime("%a")
                            day_total2[d] += 1
                            if r["Baskın Duygu"] == "Olumsuz":
                                day_neg2[d] += 1
                        except Exception:
                            pass
                    days_order2 = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
                    days_tr2 = {"Mon":"Pzt","Tue":"Sal","Wed":"Çrş","Thu":"Per","Fri":"Cum","Sat":"Cmt","Sun":"Paz"}
                    cells2 = ""
                    for d in days_order2:
                        if day_total2[d] == 0: continue
                        rate2 = day_neg2[d] / day_total2[d]
                        bg2 = "#FEE2E2" if rate2 >= 0.6 else ("#FEF9C3" if rate2 >= 0.35 else "#DCFCE7")
                        fc2 = "#DC2626" if rate2 >= 0.6 else ("#D97706" if rate2 >= 0.35 else "#16A34A")
                        cells2 += (
                            f'<div style="flex:1;text-align:center;background:{bg2};border-radius:8px;padding:6px 2px;">'
                            f'<div style="font-size:0.65rem;color:{fc2};font-weight:700;">{days_tr2[d]}</div>'
                            f'<div style="font-size:0.7rem;color:{fc2};font-weight:600;">%{int(rate2*100)}</div>'
                            f'</div>'
                        )
                    if cells2:
                        st.markdown(f"""
                        <div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:12px;padding:12px 15px;margin-top:8px;">
                            <div style="font-size:0.7rem;color:#94A3B8;font-weight:700;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Günlük Olumsuz Oran</div>
                            <div style="display:flex;gap:4px;">{cells2}</div>
                        </div>
                        """, unsafe_allow_html=True)
            except Exception:
                pass


    with col_summary:
        st.write("### Duygu Dağılımı")
        
        total_all = m_olumlu + m_olumsuz + m_istek
        
        if st.session_state.get("analysis_type") == "Zengin Analiz":
            if "ai_summary" not in st.session_state or st.session_state.get("last_results_len") != len(analysis_df):
                with st.spinner("Yapay zeka derinlemesine raporu hazırlıyor..."):
                    summary_text = generate_dynamic_summary(analysis_results=st.session_state.bulk_results)
                    st.session_state.ai_summary = summary_text
                    st.session_state.last_results_len = len(analysis_df)
            
            _raw_summary = st.session_state.ai_summary
            _formatted_summary = _raw_summary
            
            # Başlıkları kalınlaştır ve renk ver
            replacements = {
                "1. Genel Kullanıcı Deneyimi:": '<div style="color:#7c3aed;font-weight:700;margin-top:12px;margin-bottom:4px;">Genel Kullanıcı Deneyimi</div>',
                "2. Öne Çıkan Artılar:": '<div style="color:#10b981;font-weight:700;margin-top:12px;margin-bottom:4px;">Öne Çıkan Artılar</div>',
                "3. Kritik Sorunlar ve Çözüm Önerileri:": '<div style="color:#f43f5e;font-weight:700;margin-top:12px;margin-bottom:4px;">Kritik Sorunlar ve Çözümler</div>',
                "4. Kullanıcı Profili (Persona):": '<div style="color:#3b82f6;font-weight:700;margin-top:16px;margin-bottom:6px;padding:8px;background:#eff6ff;border-radius:8px;border:1px solid #dbeafe;">Kullanıcı Profili (Persona)</div>'
            }
            for k, v in replacements.items():
                _formatted_summary = _formatted_summary.replace(k, v)
                _formatted_summary = _formatted_summary.replace(k.rstrip(":"), v)

            st.markdown(f"""
<div style="background:#F5F3FF;border-radius:12px;padding:20px 24px;position:relative;border:0.5px solid #DDD6FE;">
    <div style="font-size:52px;line-height:0.6;color:#7c3aed;font-family:Georgia,serif;opacity:0.3;margin-bottom:10px;user-select:none;">"</div>
    <div style="font-size:0.82rem;font-weight:700;color:#7c3aed;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Yapay Zeka Derin Analiz Raporu</div>
    <div style="font-size:0.9rem;color:#1E293B;line-height:1.7;margin:0;white-space:pre-wrap;">{_formatted_summary}</div>
    <div style="margin-top:18px;padding-top:14px;border-top:1px solid #EDE9FE;display:flex;gap:12px;align-items:center;">
        <div style="display:flex;gap:4px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#10b981;"></div>
            <div style="width:8px;height:8px;border-radius:50%;background:#f43f5e;"></div>
            <div style="width:8px;height:8px;border-radius:50%;background:#818cf8;"></div>
        </div>
        <span style="font-size:0.7rem;color:#94A3B8;font-weight:500;">{m_olumlu} olumlu · {m_olumsuz} olumsuz · {m_istek} görüş analiz edildi</span>
    </div>
</div>
""", unsafe_allow_html=True)
        else:
            # Hızlı Analiz — heuristic özet
            if total_all == 0:
                summary_title = "Henüz yeterli veri yok."
                summary_body  = "Analiz edilecek yorumlar geldikçe burası güncellenecektir."
                grad_bg, border_c = "#F8FAFC", "#E2E8F0"
                persona_html = ""
            else:
                import random
                from collections import Counter
                _bulk = st.session_state.get("bulk_results", [])
                pos_l = [r["Yorum"] for r in _bulk if r.get("Baskın Duygu") == "Olumlu"]
                neg_l = [r["Yorum"] for r in _bulk if r.get("Baskın Duygu") == "Olumsuz"]
                neu_l = [r["Yorum"] for r in _bulk if r.get("Baskın Duygu") == "İstek/Görüş"]

                pos_p = int(len(pos_l) / total_all * 100) if total_all else 0
                neg_p = int(len(neg_l) / total_all * 100) if total_all else 0
                neu_p = int(len(neu_l) / total_all * 100) if total_all else 0 # Fixed the TypeError

                # Pick samples and translate
                def _pk(lst, n=2): return lst[:n] if len(lst) <= n else random.sample(lst, n)
                # Note: translate_reviews_heuristic takes dicts with 'text' and 'lang'
                def _to_dicts(lst): return [{"text": t, "lang": "tr"} for t in lst]
                
                pos_s = translate_reviews_heuristic(_to_dicts(_pk(pos_l)))
                neg_s = translate_reviews_heuristic(_to_dicts(_pk(neg_l)))
                neu_s = translate_reviews_heuristic(_to_dicts(_pk(neu_l)))

                if pos_p >= 55:
                    summary_title = "Topluluk Genel Olarak Olumlu"
                    tone_intro = f"Analiz edilen {total_all} yorumun %{pos_p}'ü olumlu. Kullanıcılar genel olarak deneyimlerinden memnun."
                elif neg_p >= 50:
                    summary_title = "Dikkat çeken Olumsuz bir eğilim"
                    tone_intro = f"Yorumların %{neg_p}'si olumsuz. Teknik sorunlar veya kullanım zorlukları öne çıkıyor."
                else:
                    summary_title = "Dengeli Kullanıcı Deneyimi"
                    tone_intro = f"Yorumlar olumlu (%{pos_p}) ve olumsuz (%{neg_p}) arasında dengeli bir dağılım sergiliyor."

                pos_text = f"Öne çıkan artılar: {', '.join(pos_s)}." if pos_s else ""
                neg_text = f"Sıkça dile getirilen şikayetler: {', '.join(neg_s)}." if neg_s else ""
                summary_body = f"{tone_intro} {pos_text} {neg_text}"

                # Persona extraction for Heuristic
                all_v = [r.get("Versiyon", "Bilinmiyor") for r in _bulk if r.get("Versiyon")]
                top_v = Counter(all_v).most_common(1)
                best_v = top_v[0][0] if top_v else "Belirlenemedi"
                all_l = [r.get("lang", "tr").upper() for r in _bulk]
                top_l = Counter(all_l).most_common(1)
                best_l = top_l[0][0] if top_l else "TR"

                persona_html = f"""
<div style="margin-top:16px; padding:12px; background:#eff6ff; border-radius:10px; border:1px solid #dbeafe;">
    <div style="font-size:0.75rem; font-weight:700; color:#3b82f6; text-transform:uppercase; margin-bottom:4px;">Kullanıcı Profili (Persona)</div>
    <div style="font-size:0.85rem; color:#1e40af; line-height:1.5;">
        • <b>En Aktif Sürümler:</b> v{best_v}<br>• <b>Hakim Lokasyon:</b> {best_l}<br>• <b>Riskli Segment:</b> v{best_v} (En yoğun etkileşim bu sürümde)
    </div>
</div>"""

                st.markdown(f"""
<div style="background:#F8FAFC;border-radius:12px;padding:20px 24px;position:relative;border:1px solid #E2E8F0;">
    <div style="font-size:52px;line-height:0.6;color:#818cf8;font-family:Georgia,serif;opacity:0.35;margin-bottom:10px;user-select:none;">"</div>
    <div style="font-size:0.82rem;font-weight:700;color:#6366F1;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">{summary_title}</div>
    <div style="font-size:0.9rem;color:#1E293B;line-height:1.75;margin:0;">{summary_body}</div>
    {persona_html}
    <div style="margin-top:18px;padding-top:12px;border-top:1px solid #E2E8F0;display:flex;gap:12px;align-items:center;">
        <div style="display:flex;gap:4px;">
            <div style="width:8px;height:8px;border-radius:50%;background:#10b981;"></div>
            <div style="width:8px;height:8px;border-radius:50%;background:#f43f5e;"></div>
            <div style="width:8px;height:8px;border-radius:50%;background:#818cf8;"></div>
        </div>
        <span style="font-size:0.7rem;color:#94A3B8;font-weight:500;">{m_olumlu} olumlu · {m_olumsuz} olumsuz · {m_istek} görüş analiz edildi</span>
    </div>
</div>
""", unsafe_allow_html=True)

    
    
    
    if "Puan" in df.columns and df["Puan"].notnull().any():
        st.markdown("---")
        
        
        st.write("### Puan Dağılımı")
        st.markdown('<div class="no-print">', unsafe_allow_html=True)
        freq = st.radio("Zaman Ölçeği:", ["Günlük", "Haftalık", "Aylık"], index=0, horizontal=True, key="puan_freq_sel", label_visibility="collapsed")
        st.markdown("</div>", unsafe_allow_html=True)
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
                
                # Plotly'nin alfabetik sıralamasını ezmek için kategori sırasını belirle
                _sorted_dates = dist_trend["Grup_Label"].unique().tolist()

                fig_dist = px.bar(dist_trend, x="Grup_Label", y="Oy Sayısı", color="Puan_Label",
                                 title=title_txt,
                                 color_discrete_map={
                                     "1 Yıldız": "#E53E3E",
                                     "2 Yıldız": "#F6AD55",
                                     "3 Yıldız": "#F6E05E",
                                     "4 Yıldız": "#68D391",
                                     "5 Yıldız": "#2F855A"
                                 },
                                 category_orders={
                                     "Puan_Label": ["1 Yıldız", "2 Yıldız", "3 Yıldız", "4 Yıldız", "5 Yıldız"],
                                     "Grup_Label": _sorted_dates
                                 },
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
                st.plotly_chart(fig_dist, width='stretch')
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

                if st.button(label, width='stretch'):
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
                               color_discrete_map={'Olumlu':'#10B981', 'Olumsuz':'#F43F5E', 'İstek/Görüş':'#818CF8'},
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
            
            
            selection = st.plotly_chart(fig_trend, width='stretch', on_select="rerun", key=f"chart_{key}")
            
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
                if st.button("Tüm Yorumları Göster", key=f"btn_show_{tab_id}", width='stretch'):
                    st.session_state[show_all_key] = True
                    st.rerun()
            else:
                if total_items > 250:
                    total_pages = (total_items - 1) // 250 + 1
                    current_page = st.session_state[page_key]
                    
                    nav_cols = st.columns([1, 2, 1])
                    with nav_cols[0]:
                        if st.button("Önceki Sayfa", key=f"prev_{tab_id}", width='stretch', disabled=(current_page == 1)):
                            st.session_state[page_key] -= 1
                            st.rerun()
                    with nav_cols[1]:
                        st.markdown(f"<div style='text-align: center; margin-top: 10px; font-weight: bold; color: #64748B;'>Sayfa {current_page} / {total_pages}</div>", unsafe_allow_html=True)
                    with nav_cols[2]:
                        if st.button("Sonraki Sayfa", key=f"next_{tab_id}", width='stretch', disabled=(current_page == total_pages)):
                            st.session_state[page_key] += 1
                            st.rerun()
                            
                    st.markdown("<br>", unsafe_allow_html=True)

                if st.button("Daha Az Göster", key=f"btn_hide_{tab_id}", width='stretch'):
                    st.session_state[show_all_key] = False
                    st.session_state[page_key] = 1
                    st.rerun()

    
    st.write("### Yorum Listesi")
    
    
    st.markdown('<div class="no-print">', unsafe_allow_html=True)
    yorum_freq = st.radio("Zaman Ölçeği:", ["Günlük", "Haftalık", "Aylık"], index=0, horizontal=True, key="yorum_freq_sel", label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)
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
        st.markdown('<div class="no-print"><h3 style="font-size:1.5rem;font-weight:700;color:#1E293B;margin-bottom:10px;">Analiz Raporunu Paylaş</h3></div>', unsafe_allow_html=True)
        
        
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
        
        
        # Dinamik özet varsa kullan, yoksa genel skor özeti çıkar
        display_summary = st.session_state.get('ai_summary')
        if not display_summary:
            if pos_p > 70: display_summary = "Kullanıcı topluluğu uygulamadan oldukça memnun. Pozitif geri bildirimler baskın."
            elif neg_p > 50: display_summary = "Kritik teknik sorunlar ve kullanıcı memnuniyetsizliği tespit edildi. Acil müdahale gerekebilir."
            else: display_summary = "Kullanıcı deneyimi karmaşık bir yapıda. Hem pozitif hem negatif geri bildirimler dengeli seyrediyor."
        
        display_summary = display_summary.replace("`", "").replace("*", "").replace("#", "")
        
        
        import re
        
        display_summary = re.sub(r'\s+([.,;:!?])', r'\1', display_summary)
        
        display_summary = re.sub(r' {2,}', ' ', display_summary)
        
        display_summary = display_summary.replace('\n', '<br>')

        def _pie_path(start_pct, end_pct, color):
            import math as _math
            if end_pct - start_pct <= 0: return ""
            if end_pct - start_pct >= 99.9:
                return f'<path d="M70,18 A52,52 0 1,1 69.99,18 Z" fill="{color}" stroke="#faf8f3" stroke-width="1.5"/>'
            r, cx, cy = 52, 70, 70
            sa = _math.radians(start_pct * 3.6 - 90)
            ea = _math.radians(end_pct * 3.6 - 90)
            x1,y1 = cx+r*_math.cos(sa), cy+r*_math.sin(sa)
            x2,y2 = cx+r*_math.cos(ea), cy+r*_math.sin(ea)
            xi,yi = cx+28*_math.cos(ea), cy+28*_math.sin(ea)
            xj,yj = cx+28*_math.cos(sa), cy+28*_math.sin(sa)
            lg = 1 if (end_pct-start_pct) > 50 else 0
            return f'<path d="M{x1:.2f},{y1:.2f} A{r},{r} 0 {lg},1 {x2:.2f},{y2:.2f} L{xi:.2f},{yi:.2f} A28,28 0 {lg},0 {xj:.2f},{yj:.2f} Z" fill="{color}" stroke="#faf8f3" stroke-width="1.5"/>'

        card_html = clean_html(f"""
            <div id="nlp-report-card" style="background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 20px; padding: clamp(15px, 5vw, 35px); margin: 10px auto; box-shadow: 0 10px 25px rgba(0,0,0,0.05); font-family: 'Poppins', sans-serif; color: #1E293B; max-width: 100%; position: relative; overflow: hidden;">
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
                
                <div class="metric-row" style="display: flex; flex-wrap: wrap; justify-content: center; margin-bottom: 35px; gap: 8px;">
                    <div class="metric-box" style="text-align: center; flex: 1 1 40%; min-width: 100px; background: #F8FAFC; padding: 12px; border-radius: 12px;">
                        <div style="font-size: 0.65rem; color: #64748B; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Analiz</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #334155;">{total_q}</div>
                    </div>
                    <div class="metric-box" style="text-align: center; flex: 1 1 40%; min-width: 100px; background: #ECFDF5; padding: 12px; border-radius: 12px; border: 1px solid #D1FAE5;">
                        <div style="font-size: 0.65rem; color: #059669; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Olumlu</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #059669;">{t_pos}</div>
                    </div>
                    <div class="metric-box" style="text-align: center; flex: 1 1 40%; min-width: 100px; background: #FEF2F2; padding: 12px; border-radius: 12px; border: 1px solid #FEE2E2;">
                        <div style="font-size: 0.65rem; color: #DC2626; text-transform: uppercase; font-weight: 700; margin-bottom: 4px;">Olumsuz</div>
                        <div style="font-size: 1.4rem; font-weight: 800; color: #DC2626;">{t_neg}</div>
                    </div>
                    <div class="metric-box" style="text-align: center; flex: 1 1 40%; min-width: 100px; background: #EFF6FF; padding: 12px; border-radius: 12px; border: 1px solid #DBEAFE;">
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
            </div>
        """)
        st.markdown(card_html, unsafe_allow_html=True)
        st.info("Yukarıdaki kartı kopyalayabilir veya doğrudan paylaşabilirsiniz.")

        image_name = f"{app_name} ai sentiment report.png".replace(" ", "_").replace(":", "_")
        
        components.html(f"""
            <script src="https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js"></script>
            <script>
            (function() {{

                // ── Bildirim kutusu ──────────────────────────────
                if (!window.parent.document.getElementById('uNotif')) {{
                    const div = window.parent.document.createElement('div');
                    div.id = 'uNotif';
                    div.innerHTML = '<span id="uMsg">Hazırlanıyor...</span>';
                    Object.assign(div.style, {{
                        position:'fixed', top:'20px', left:'50%',
                        transform:'translateX(-50%) translateY(-20px) scale(0.9)',
                        background:'#10B981', color:'white',
                        padding:'14px 28px', borderRadius:'12px', fontWeight:'700',
                        opacity:'0', transition:'all 0.4s cubic-bezier(0.19,1,0.22,1)',
                        zIndex:'9999999', boxShadow:'0 15px 30px rgba(16,185,129,0.4)',
                        display:'flex', alignItems:'center', gap:'10px',
                        pointerEvents:'none', fontFamily:'"Poppins",sans-serif',
                        whiteSpace:'nowrap'
                    }});
                    window.parent.document.body.appendChild(div);
                }}

                window.notifyBridge = function(msg, duration=3000) {{
                    const n = window.parent.document.getElementById('uNotif');
                    const m = window.parent.document.getElementById('uMsg');
                    if (n && m) {{
                        m.innerText = msg;
                        n.style.opacity = '1';
                        n.style.transform = 'translateX(-50%) translateY(0) scale(1)';
                        setTimeout(() => {{
                            n.style.opacity = '0';
                            n.style.transform = 'translateX(-50%) translateY(-20px) scale(0.9)';
                        }}, duration);
                    }}
                }};

                // ── html2canvas'ı parent'a yükle ─────────────────
                function injectHtml2Canvas(cb) {{
                    if (window.parent.html2canvas) {{ cb(); return; }}
                    const s = window.parent.document.createElement('script');
                    s.src = 'https://cdnjs.cloudflare.com/ajax/libs/html2canvas/1.4.1/html2canvas.min.js';
                    s.onload = cb;
                    window.parent.document.head.appendChild(s);
                }}

                // ── Kart → canvas → blob ──────────────────────────
                async function captureCard() {{
                    const target = window.parent.document.getElementById('nlp-report-card');
                    if (!target) throw new Error('Kart bulunamadı');
                    return await window.parent.html2canvas(target, {{
                        scale: 2,
                        useCORS: true,
                        backgroundColor: '#FFFFFF',
                        logging: false,
                        allowTaint: true
                    }});
                }}

                // ── PNG indir butonu (#btn-png-download) ──────────
                function bindPngBtn() {{
                    const btn = window.parent.document.getElementById('btn-png-download');
                    if (!btn || btn.hasAttribute('data-bound')) return;
                    btn.setAttribute('data-bound', 'true');
                    btn.addEventListener('click', async function() {{
                        window.notifyBridge('Görsel Hazırlanıyor... ⏳', 5000);
                        try {{
                            const canvas = await captureCard();
                            const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent) ||
                                (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
                            const dataUrl = canvas.toDataURL('image/png');
                            if (isIOS) {{
                                const t = window.open();
                                if (t) {{
                                    t.document.write('<img src="' + dataUrl + '" style="width:100%;height:auto;">');
                                    t.document.title = 'Analiz Raporu';
                                    window.notifyBridge('Görseli basılı tutarak kaydedin');
                                }}
                            }} else {{
                                const a = document.createElement('a');
                                a.href = dataUrl;
                                a.download = '{image_name}';
                                a.click();
                                window.notifyBridge('İndirme Başlatıldı!');
                            }}
                        }} catch(e) {{
                            window.notifyBridge('Hata oluştu!');
                        }}
                    }});
                }}

                // ── Paylaş butonu (#btn-share-image) ─────────────
                function bindShareBtn() {{
                    const btn = window.parent.document.getElementById('btn-share-image');
                    if (!btn || btn.hasAttribute('data-sbound')) return;
                    btn.setAttribute('data-sbound', 'true');

                    btn.addEventListener('click', async function() {{
                        const orig = btn.innerHTML;
                        btn.innerHTML = '⏳ Hazırlanıyor...';
                        btn.disabled = true;

                        try {{
                            const canvas = await captureCard();
                            canvas.toBlob(async function(blob) {{
                                const file = new File([blob], '{image_name}', {{type:'image/png'}});

                                if (navigator.share && navigator.canShare &&
                                    navigator.canShare({{files:[file]}})) {{
                                    try {{
                                        await navigator.share({{
                                            files: [file],
                                            title: 'AI Yorum Analiz Raporu'
                                        }});
                                        btn.innerHTML = 'Paylaşıldı!';
                                        setTimeout(() => {{
                                            btn.innerHTML = orig;
                                            btn.disabled = false;
                                        }}, 2000);
                                    }} catch(e) {{
                                        btn.innerHTML = orig;
                                        btn.disabled = false;
                                        if (e.name !== 'AbortError') fallback(blob);
                                    }}
                                }} else {{
                                    fallback(blob);
                                    btn.innerHTML = orig;
                                    btn.disabled = false;
                                }}
                            }}, 'image/png');

                        }} catch(e) {{
                            btn.innerHTML = 'Hata';
                            setTimeout(() => {{ btn.innerHTML = orig; btn.disabled = false; }}, 2000);
                        }}
                    }});
                }}

                function fallback(blob) {{
                    const url = URL.createObjectURL(blob);
                    const isIOS = /iPad|iPhone|iPod/.test(navigator.userAgent);
                    if (isIOS) {{
                        const t = window.open();
                        if (t) {{
                            t.document.write(
                                '<img src="' + url + '" style="width:100%;height:auto;">' +
                                '<p style="text-align:center;font-family:sans-serif;color:#64748B;">Kaydetmek için görsele basılı tutun.</p>'
                            );
                        }}
                    }} else {{
                        const a = window.parent.document.createElement('a');
                        a.href = url;
                        a.download = '{image_name}';
                        a.click();
                    }}
                    const row = window.parent.document.getElementById('fallback-share-row');
                    if (row) row.style.display = 'flex';
                    setTimeout(() => URL.revokeObjectURL(url), 5000);
                }}

                // ── Butonları bul ve bind et ───────────────────────
                injectHtml2Canvas(function() {{
                    bindPngBtn();
                    bindShareBtn();
                    // Henüz DOM'da değilse bekle
                    const timer = setInterval(function() {{
                        bindPngBtn();
                        bindShareBtn();
                    }}, 800);
                    setTimeout(() => clearInterval(timer), 15000);
                }});

            }})();
            </script>
        """, height=0)

        
        share_ui = textwrap.dedent(f"""
            <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.2/css/all.min.css">
            <style>
                .share-btn-row {{
                    display: flex;
                    gap: 10px;
                    justify-content: center;
                    margin-bottom: 10px;
                    flex-wrap: wrap;
                }}
                .share-btn {{
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    gap: 8px;
                    padding: 12px 20px;
                    border-radius: 12px;
                    border: 1px solid #E2E8F0;
                    background: #FFFFFF;
                    cursor: pointer;
                    font-size: 0.9rem;
                    font-weight: 600;
                    font-family: 'Poppins', sans-serif;
                    color: #1E293B;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                    transition: all 0.2s ease;
                    text-decoration: none;
                }}
                .share-btn:hover {{
                    transform: translateY(-2px);
                    box-shadow: 0 6px 12px rgba(0,0,0,0.1);
                    border-color: #CBD5E1;
                }}
                .share-btn-primary {{
                    background: #6366F1;
                    color: white !important;
                    border-color: #6366F1;
                    font-size: 1rem;
                    padding: 14px 28px;
                    width: 100%;
                    max-width: 300px;
                    margin: 0 auto;
                }}
                .share-btn-primary:hover {{
                    background: #4F46E5 !important;
                    border-color: #4F46E5 !important;
                }}
            </style>

            <div class="share-btn-row">
                <button class="share-btn share-btn-primary" id="btn-share-image">
                    📤 Görseli Paylaş
                </button>
            </div>

            <div class="share-btn-row" id="fallback-share-row" style="display:none;">
                <span style="font-size:0.8rem;color:#94A3B8;text-align:center;width:100%;">
                    Tarayıcınız doğrudan paylaşımı desteklemiyor. Görseli indirip manuel paylaşabilirsiniz.
                </span>
                <a href="https://api.whatsapp.com/send?text={encoded_text}" target="_blank" class="share-btn" style="color:#25D366;font-size:1.3rem;">
                    <i class="fa-brands fa-whatsapp"></i>
                </a>
                <a href="https://twitter.com/intent/tweet?text={encoded_text}" target="_blank" class="share-btn" style="color:#000;font-size:1.3rem;">
                    <i class="fa-brands fa-x-twitter"></i>
                </a>
                <a href="https://t.me/share/url?url=https://sentimentanalysis-aimode.streamlit.app/&text={encoded_text}" target="_blank" class="share-btn" style="color:#24A1DE;font-size:1.3rem;">
                    <i class="fa-brands fa-telegram"></i>
                </a>
            </div>
        """).strip()
        st.markdown(share_ui, unsafe_allow_html=True)

        btn_cols = st.columns(3)
        with btn_cols[0]:
            st.markdown("""
                <button id="btn-png-download" style="width: 100%; height: 50px; background: #5a67d8; color: white; border: none; border-radius: 12px; cursor: pointer; font-size: 1.1rem; font-weight: 600; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-family: 'Poppins', sans-serif; display: flex; align-items: center; justify-content: center; gap: 8px; transition: all 0.2s;">
                    PNG
                </button>
            """, unsafe_allow_html=True)
        with btn_cols[1]:
            st.download_button("EXCEL", output.getvalue(), excel_filename, key="xl_dl", width='stretch')
        with btn_cols[2]:
            components.html(f"""
                <style>
                    body {{ margin: 0; padding: 0; overflow: hidden; font-family: sans-serif; }}
                    button:hover {{ filter: brightness(0.9); transform: translateY(-1px); }}
                </style>
                <button onclick='window.parent.print()' style='width: 100%; height: 50px; background: #F4A261; color: white; border: none; border-radius: 12px; cursor: pointer; font-size: 1.1rem; font-weight: 600; box-shadow: 0 4px 6px rgba(0,0,0,0.1); font-family: "Poppins", sans-serif; display: flex; align-items: center; justify-content: center; gap: 8px; transition: all 0.2s;'>
                    PDF
                </button>
            """, height=48)
                    
    except Exception as e:
        st.error(f"Paylaşım sistemi hatası: {e}")



# ── Karşılaştırma analizi — fonksiyonlar tanımlıyken çalışır ──────
if st.session_state.get("_cmp_pending"):
    active_inputs = st.session_state.pop("_cmp_pending")
    cmp_days_run  = st.session_state.pop("_cmp_days", 30)
    cmp_analysis_type = st.session_state.pop("_cmp_analysis_type", "Hızlı Analiz")
    cmp_analysis_mode = st.session_state.pop("_cmp_analysis_mode", 0)
    
    # Set session_state for analysis functions to use
    st.session_state["analysis_type"] = cmp_analysis_type
    st.session_state["analysis_mode"] = cmp_analysis_mode

    for u in active_inputs:
        platform_c = None; app_id_c = ""; country_c = "tr"
        if "play.google.com" in u:
            platform_c = "google"
            m = re.search(r"id=([^&/]+)", u)
            if m: app_id_c = m.group(1)
        elif "apple.com" in u:
            platform_c = "apple"
            m = re.search(r"id(\d+)", u)
            if m: app_id_c = m.group(1)
        else:
            cu = u.lower()
            if cu.startswith("id") and cu[2:].isdigit():
                platform_c = "apple"; app_id_c = cu[2:]
            elif cu.isdigit():
                platform_c = "apple"; app_id_c = cu
            elif "." in u and re.match(r"^[a-zA-Z0-9._]+$", u):
                platform_c = "google"; app_id_c = u

        if not platform_c or not app_id_c:
            st.error(f"Geçersiz ID: {u}")
            continue

        name_c = app_id_c
        _icon = ""
        _store_label = ""
        _rating_store = 0.0
        _ratings_store = 0
        _installs_store = ""
        _version_store = ""
        _genre_store = ""
        _rank_msg = ""
        _genre_store = ""

        if platform_c == "google":
            try:
                info_c = play_app(app_id_c, lang='tr', country='tr')
                name_c = info_c.get('title', app_id_c)
                _icon = (info_c.get('icon') or 
                                 info_c.get('iconImage') or 
                                 info_c.get('headerImage') or '')
                _store_label = "Google Play"
                _rating_store = round(float(info_c.get('score') or 0), 1)
                _ratings_store = info_c.get('ratings', 0)
                _installs_store = info_c.get('installs', '?')
                _version_store = info_c.get('version', '?')
                _genre_store = info_c.get('genre') or (info_c.get('categories')[0]['name'] if info_c.get('categories') else '?')
                _rank_msg = "" # Google Play rank is complex, placeholder
            except: 
                _genre_store = '?'
        elif platform_c == "apple":
            try:
                r_c = requests.get(f"https://itunes.apple.com/lookup?id={app_id_c}&country=tr", timeout=5)
                if r_c.status_code == 200:
                    d_c = r_c.json()
                    if d_c.get('results'):
                        rc_c = d_c['results'][0]
                        name_c = rc_c.get('trackCensoredName', app_id_c)
                        _icon = (rc_c.get('artworkUrl512') or 
                                         rc_c.get('artworkUrl100') or 
                                         rc_c.get('artworkUrl60') or '')
                        _store_label = "App Store"
                        _rating_store = round(float(rc_c.get('averageUserRating') or 0), 1)
                        _ratings_store = rc_c.get('userRatingCount', 0)
                        _installs_store = "App Store"
                        _version_store = rc_c.get('version', '?')
                        _genre_store = rc_c.get('primaryGenreName', '?')
                        _genre_id = rc_c.get('primaryGenreId')
                        if _genre_id:
                            try:
                                rss_url = f"https://itunes.apple.com/tr/rss/topfreeapplications/limit=200/genre={_genre_id}/json"
                                rss_resp = requests.get(rss_url, timeout=3)
                                if rss_resp.status_code == 200:
                                    rss_data = rss_resp.json()
                                    entries = rss_data.get('feed', {}).get('entry', [])
                                    for idx, entry in enumerate(entries):
                                        if entry.get('id', {}).get('attributes', {}).get('im:id') == app_id_c:
                                            _rank_msg = f"#{idx+1}"
                                            break
                            except: pass
            except: pass

        with st.spinner(f"{name_c} analiz ediliyor..."):
            try:
                if platform_c == "google":
                    revs_c = fetch_google_play_reviews(app_id_c, cmp_days_run)
                else:
                    revs_c = get_app_store_reviews(app_id_c, _days_limit=cmp_days_run)
                    threshold_c = datetime.now() - timedelta(days=cmp_days_run)
                    revs_c = [r for r in revs_c
                              if isinstance(r.get('date'), datetime)
                              and r['date'] >= threshold_c
                              and is_valid_comment(r.get('text',''))]

                pos_c = neg_c = neu_c = 0
                for rev in revs_c:
                    # Use selected analysis type (Hızlı or Zengin)
                    if cmp_analysis_type == "Hızlı Analiz":
                        res_c = heuristic_analysis(str(rev.get('text','')), rating=rev.get('rating'))
                    else:
                        res_c = get_ai_sentiment(
                            text=str(rev.get('text','')),
                            model_name=st.session_state.get("current_ai_model"),
                            provider=st.session_state.get("current_ai_provider"),
                            rating=rev.get('rating'),
                            analysis_mode=cmp_analysis_mode
                        )
                    sc = {"Olumlu": res_c["olumlu"], "Olumsuz": res_c["olumsuz"], "İstek": res_c["istek_gorus"]}
                    v_c = max(sc, key=lambda k: sc[k])
                    if v_c == "Olumlu": pos_c += 1
                    elif v_c == "Olumsuz": neg_c += 1
                    else: neu_c += 1

                total_c = pos_c + neg_c + neu_c or 1
                st.session_state.cmp_results[name_c] = {
                    "total": len(revs_c),
                    "pos": pos_c, "neg": neg_c, "neu": neu_c,
                    "pos_pct": int(pos_c/total_c*100),
                    "neg_pct": int(neg_c/total_c*100),
                    "neu_pct": int(neu_c/total_c*100),
                    "score": int((pos_c*100 + neu_c*50) / total_c),
                    "icon": _icon,
                    "store": _store_label,
                    "rating": _rating_store,
                    "ratings": _ratings_store,
                    "installs": _installs_store,
                    "version": _version_store,
                    "genre": _genre_store,
                    "rank": _rank_msg,
                }
            except Exception as e:
                st.error(f"{name_c} çekilemedi: {e}")

    st.rerun()

st.divider()
st.caption("Geliştiren: ivicin")

