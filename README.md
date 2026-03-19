# AI App Review Analysis 🧠

Modern ve akıllı uygulama mağazası yorum analiz platformu. Bu uygulama, Google Play Store ve Apple App Store yorumlarını otomatik olarak çeker ve en gelişmiş yapay zeka modelleriyle duygu durum analizi yapar.

## ✨ Temel Özellikler

- **Hibrit Analiz Sistemi:** Google Gemini AI (Deep Analysis) ile derinlemesine analiz veya kotasız yerel "Hızlı Tarama" seçenekleri.
- **Otomatik Veri Çekme:** Mağaza linki üzerinden tarih filtreli yorum çekme.
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
```

### 3. Çalıştırın
```bash
streamlit run streamlit_app.py
```

## 🚀 Güvenlik ve Performans

- **Maliyet Kontrolü:** API harcamalarını takip eden ve sınır koyan entegre takip sistemi.
- **Paralel Çalışma:** Eşzamanlı (Concurrent) analiz yeteneği ile hızlı sonuç üretimi.
- **Veri Güvenliği:** API anahtarları çevre değişkenleri üzerinden güvenli bir şekilde yönetilir.

---
Geliştiren: **ivicin**
