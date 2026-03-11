# AI App Review Analysis 🧠

Modern ve akıllı uygulama mağazası yorum analiz platformu. Bu uygulama, hem Google Play Store hem de Apple App Store yorumlarını otomatik olarak çeker ve en gelişmiş yapay zeka modelleriyle duygu durum analizi yapar.

## ✨ Temel Özellikler

- **Hibrit Analiz Sistemi:** Google Gemini AI (2.5 Pro/Flash) ile derinlemesine analiz veya kotasız yerel "Hızlı Tarama" seçenekleri.
- **Otomatik Veri Çekme:** Mağaza linki üzerinden tarih filtreli yorum çekme (Play Store & App Store).
- **Dosya Desteği:** CSV ve Excel dosyalarınızı yükleyerek toplu analiz yapabilme.
- **Duygu Filtreleme:** Olumlu, olumsuz ve istek/görüş kategorilerinde otomatik sınıflandırma.
- **Görsel Raporlama:** Etkileşimli grafikler ve indirilebilir analiz raporları.

## 🛠️ Kurulum

### 1. Dosyaları Klonlayın
```bash
git clone https://github.com/cemevecen/sentiment_analysis.git
cd sentiment_analysis
```

### 2. Gereksinimleri Yükleyin
```bash
pip install -r requirements.txt
```

### 3. Yapılandırma
Sistem anahtarını `.env` dosyasına veya Streamlit Secrets alanına ekleyin:
```env
GOOGLE_API_KEY=AIzaSy...
```

### 4. Çalıştırın
```bash
streamlit run streamlit_app.py
```

## 🚀 Güvenlik ve Performans

- **Maliyet Kontrolü:** API harcamalarını takip eden ve sınır koyan entegre takip sistemi.
- **Hız:** Eşzamanlı (Concurrent) analiz yeteneği ile saniyeler içinde yüzlerce yorum işleme.
- **Veri Güvenliği:** API anahtarları asla kod içerisinde yer almaz, çevre değişkenleri üzerinden yönetilir.

---
Geliştiren: **Cem Evecen**
