# AI App Review Analysis 🧠

Modern ve akıllı uygulama mağazası yorum analiz platformu. Bu uygulama, Google Play Store ve Apple App Store yorumlarını otomatik olarak çeker ve en gelişmiş yapay zeka modelleriyle duygu durum analizi yapar.

## ✨ Temel Özellikler

- **Hibrit Analiz Sistemi:** Google Gemini AI (Deep Analysis) ile derinlemesine analiz veya kotasız yerel "Hızlı Tarama" seçenekleri.
- **Otomatik Veri Çekme:** Mağaza linki üzerinden tarih filtreli yorum çekme.
- **Google İşletme Desteği:** Google Maps işletme linki veya işletme adı ile yorum çekme (API + Selenium fallback).
- **Dosya Desteği:** CSV ve Excel dosyalarınızı yükleyerek toplu analiz yapabilme.
- **Duygu Filtreleme:** Olumlu, Olumsuz ve İstek/Görüş kategorilerinde otomatik sınıflandırma.
- **Görsel Raporlama:** Etkileşimli grafikler ve paylaşılabilir/indirilebilir PNG rapor kartları.
- **Mobil Uyumlu:** Telefon ve tabletlerde tam responsive tasarım ve özel iOS indirme desteği.

## 🛠️ Kurulum

### 1. Dosyaları Hazırlayın
Projeyi indirin ve gerekli kütüphaneleri yükleyin:
```bash
pip install -r requirements.txt
```

### 2. Yapılandırma
Google Gemini API anahtarını `.env` dosyasına veya platformun "Secrets" alanına ekleyin:
```env
GOOGLE_API_KEY=YOUR_API_KEY_HERE
GOOGLE_PLACES_API_KEY=YOUR_GOOGLE_PLACES_API_KEY
```

Google İşletme için giriş örnekleri:
```text
https://www.google.com/maps/place/...
maps:Starbucks Kadıköy
```

### 3. Çalıştırın
```bash
streamlit run streamlit_app.py
```

## 🚀 Güvenlik ve Performans

- **Maliyet Kontrolü:** API harcamalarını takip eden ve sınır koyan entegre takip sistemi.
- **Paralel Çalışma:** Eşzamanlı (Concurrent) analiz yeteneği ile hızlı sonuç üretimi.
- **Veri Güvenliği:** API anahtarları çevre değişkenleri üzerinden güvenli bir şekilde yönetilir.

## 🌍 Canlıya Alma (Render)

Bu repoda `render.yaml` bulunduğu için Render'a tek tıkla deploy edebilirsiniz.

1. Render hesabınızla giriş yapın.
2. **New +** → **Blueprint** seçin.
3. Bu GitHub reposunu seçin.
4. Render, `render.yaml` dosyasını otomatik algılar ve servisi oluşturur.
5. Ortam değişkenlerini girin:
	- `GEMINI_API_KEY` veya `GOOGLE_API_KEY`
	- `GOOGLE_PLACES_API_KEY` (Google işletme yorumları için önerilir)
	- Opsiyonel: `GOOGLE_MAPS_API_KEY`, `MISTRAL_API_KEY`, `GROQ_API_KEY`, `DEEPSEEK_API_KEY`

Başlatma komutu otomatik olarak:
`streamlit run streamlit_app.py --server.port $PORT --server.address 0.0.0.0`

Not: Google Maps kaynaklı çekimlerde API sınırları veya anti-bot politikaları nedeniyle sonuçlar değişken olabilir. API anahtarı tanımlandığında Places API yolu daha stabil çalışır.

---
Geliştiren: **ivicin**
