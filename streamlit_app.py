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
Bu uygulama, girdiğiniz metnin duygu durumunu (Olumlu/Olumsuz/İstek-Görüş) yapay zeka kullanarak analiz eder.
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

                        # --- Date Column Detection (Avoiding numeric ID/millis columns) ---
                        date_keys = ["date", "time", "tarih", "saat", "submit"]
                        date_col = None
                        potential_date_cols = [c for c in df_upload.columns if any(dk in c.lower() for dk in date_keys) and "millis" not in c.lower() and "epoch" not in c.lower()]
                        
                        if potential_date_cols:
                            # Prefer 'date' or 'tarih' keywords
                            date_col = potential_date_cols[0]
                            for pc in potential_date_cols:
                                if "date" in pc.lower() or "tarih" in pc.lower():
                                    date_col = pc
                                    break
                        
                        col_name = st.selectbox(
                            f"Analiz edilecek sütun ({uploaded_file.name}):", 
                            df_upload.columns, 
                            index=default_index,
                            key=f"col_{uploaded_file.name}"
                        )
                        
                        if col_name:
                            def is_valid_comment(text):
                                text_str = str(text).strip()
                                text_lower = text_str.lower()
                                
                                if len(text_str) < 4: return False
                                if text_lower in ['nan', 'null', 'none', 'tr', 'en']: return False
                                
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
                                if any(rp in text_lower for rp in reply_patterns):
                                    return False
                                
                                # Filter formal addresses "Ad Soyad Bey/Hanım,"
                                if re.search(r'[a-zçğıöşü]+\s+(bey|hanım),', text_lower):
                                    return False
                                
                                # 4. Metadata/Numeric IDs
                                if re.match(r'^\d{4}-\d{2}-\d{2}.*', text_str): return False
                                if text_str.replace('.', '').replace('-', '').isdigit(): return False
                                if re.match(r'^\d{1,4}[./-]\d{1,2}[./-]\d{1,4}$', text_str): return False
                                
                                return True

                            # Store text and date together
                            for idx, row in df_upload.iterrows():
                                if col_name in row:
                                    val = row[col_name]
                                    if pd.notnull(val) and is_valid_comment(val):
                                        entry = {"text": str(val).strip()}
                                        if date_col:
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
else:
    text_input = st.text_input("Analiz edilecek metni girin:", placeholder="Örn: Bugün harika bir gün!")
    if text_input:
        comments_to_analyze = [{"text": text_input}]

def get_gemini_sentiment(text):
    if not HAS_GEMINI:
        return None
    try:
        model = genai.GenerativeModel('gemini-1.5-flash')
        prompt = f"""
        You are an expert Turkish app review sentiment analyzer.

        Read the ENTIRE review carefully and determine the DOMINANT, FINAL emotional tone.
        Do NOT focus on individual words — understand the full meaning and context.

        Categories:
        - OLUMLU: The overall tone is positive. Gratitude, praise, satisfaction. Even if the user mentions a past problem that was RESOLVED, the review is OLUMLU if they are happy now.
        - OLUMSUZ: The overall tone is negative. Active complaints, unresolved technical problems (opens with difficulty, crashes, freezes, data loss). If a problem is STILL ongoing ("problem devam ediyor", "hâlâ açılmıyor"), it is OLUMSUZ.
        - ISTEK/GORUS: Neutral suggestions or feature requests with no strong positive or negative emotion.

        EXAMPLES:
        - "Çok başarılı bir uygulama. Tüm ekibe çok teşekkür ederim." → OLUMLU (clear praise)
        - "Önce sorun yaşadım ama ekip çözdü, teşekkürler." → OLUMLU (problem was resolved, ends positively)
        - "Popup reklamlar var kafam şişti. Edit: Problem halen devam." → OLUMSUZ (unresolved problem)
        - "Giremiyorum artık bozuldu uygulama" → OLUMSUZ (active access failure)
        - "Keşke ekranda makas farkı da gösterilse çok iyi olurdu..." → ISTEK/GORUS (feature request)

        Return ONLY a JSON: {{"olumlu": score, "olumsuz": score, "istek_gorus": score}}
        Sum must equal 1.0.

        Review: "{text}"
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
            return data
    except Exception:
        return None
    return None

def heuristic_analysis(text):
    text_lower = text.lower()

    # Strong positive signals
    pos_words = [
        "teşekkür", "harika", "başarılı", "mükemmel", "süper", "güzel", "iyi",
        "memnun", "sev", "beğen", "hızlı", "kolay", "kaliteli", "çözdü", "giderildi"
    ]
    # Strong negative signals — words indicating ACTIVE, UNRESOLVED problems
    neg_words = [
        "açılmıyor", "giremiyorum", "girilmiyor", "donuyor", "kasıyor", "dondu",
        "berbat", "rezalet", "bozuk", "bozuldu", "çalışmıyor", "silinmiş",
        "kayboldu", "rezil", "mağdur", "kötü", "yavaş", "hata veriyor", "devam ediyor", "hâlâ", "hala"
    ]
    # Request / suggestion markers
    neutral_intent = ["keşke", "gelse", "olurdu", "gelebilir", "olsa", "eklense", "mı?", "mi?"]

    pos_score = sum(1 for w in pos_words if w in text_lower)
    neg_score = sum(1 for w in neg_words if w in text_lower)
    neu_score = sum(1 for w in neutral_intent if w in text_lower)

    # Balanced weighting: compare NET scores, not just presence of any negative word
    if neg_score > pos_score and neg_score > neu_score:
        p, n, neu = 0.05, 0.85, 0.1
    elif pos_score > neg_score and pos_score > neu_score:
        p, n, neu = 0.85, 0.05, 0.1
    elif neu_score > 0 and neg_score == 0:
        p, n, neu = 0.15, 0.15, 0.7
    elif pos_score > 0 and neg_score > 0:
        # Mixed review: let the positive score have slight edge (resolved situation)
        if pos_score >= neg_score:
            p, n, neu = 0.65, 0.25, 0.1
        else:
            p, n, neu = 0.1, 0.8, 0.1
    else:
        p, n, neu = 0.2, 0.2, 0.6

    return {"olumlu": p, "olumsuz": n, "istek_gorus": neu, "method": "Heuristic"}

# Analysis Trigger
if st.button("Duygu Durumunu Analiz Et", use_container_width=True):
    if not comments_to_analyze:
        st.warning("Lütfen analiz edilecek bir metin girin veya dosya yükleyin.")
    else:
        if is_bulk:
            bulk_results = []
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, entry in enumerate(comments_to_analyze):
                comment = entry["text"]
                date = entry.get("date")
                status_text.text(f"Analiz ediliyor ({i+1}/{len(comments_to_analyze)})...")
                res = get_gemini_sentiment(comment) or heuristic_analysis(comment)
                scores = {"Olumlu": res['olumlu'], "Olumsuz": res['olumsuz'], "İstek/Görüş": res['istek_gorus']}
                verdict = max(scores, key=scores.get)
                
                bulk_results.append({
                    "No": i + 1, "Yorum": comment, "Baskın Duygu": verdict,
                    "Olumlu %": f"{res['olumlu']:.2%}", "İstek/Görüş %": f"{res['istek_gorus']:.2%}", "Olumsuz %": f"{res['olumsuz']:.2%}",
                    "Tarih": date
                })
                progress_bar.progress((i + 1) / len(comments_to_analyze))
            
            st.session_state.bulk_results = bulk_results
            status_text.success("Analiz Başarıyla Tamamlandı!")
        else:
            # Single analysis logic
            with st.spinner("Analiz ediliyor..."):
                entry = comments_to_analyze[0]
                comment = entry["text"]
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

    col_pie, col_summary = st.columns([1, 1.5])
    
    with col_pie:
        pie_data = pd.DataFrame({"Duygu": counts.index, "Sayı": counts.values})
        fig_pie = px.pie(pie_data, values='Sayı', names='Duygu', hole=0.5,
                      title="Genel Dağılım",
                      color='Duygu', color_discrete_map={'Olumlu':'#2ecc71', 'Olumsuz':'#e74c3c', 'İstek/Görüş':'#3498db'})
        fig_pie.update_traces(pull=[0.05, 0.05, 0.05], textinfo='percent')
        fig_pie.update_layout(height=300, showlegend=True, 
                             legend={"orientation": "h", "yanchor": "bottom", "y": -0.2, "xanchor": "center", "x": 0.5},
                             margin={"t": 40, "b": 40, "l": 10, "r": 10})
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_summary:
        st.write("#### 📈 Hızlı Özet")
        st.write(f"Bugün toplam **{len(df)}** yorum incelendi. En baskın duygu: **{counts.idxmax()}**.")
        st.info("Aşağıdaki sekmeleri kullanarak hem yorumları hem de tarihsel gelişimlerini detaylıca inceleyebilirsiniz.")

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
st.caption("Geliştiren: Cem Evecen | Streamlit Cloud Deployment Ready")

