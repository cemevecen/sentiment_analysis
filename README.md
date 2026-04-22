# AI Sentiment Analysis (Duygu Analizi) 🧠

Bu proje, metinlerin duygu durumunu (Pozitif/Negatif) analiz eden modern bir web uygulamasıdır. Google Gemini AI desteği ile hibrit bir analiz yöntemi kullanır.

## Özellikler

-   **Gemini AI Analizi:** Google'ın en gelişmiş yapay zeka modellerini kullanarak derinlemesine duygu analizi.
-   **Heuristic (Kelime Bazlı) Analiz:** API'ye erişilemediği durumlarda otomatik devreye giren hızlı kelime eşleştirme sistemi.
-   **Modern Arayüz:** Streamlit ile geliştirilmiş, kullanıcı dostu ve şık tasarım.
-   **Hızlı Yanıt:** Düşük gecikme süreli analiz sonuçları.

## Kurulum ve Çalıştırma

### 1. Projeyi Klonlayın
```bash
git clone https://github.com/USERNAME/nlp-sentiment-project.git
cd nlp-sentiment-project
```

### 2. Bağımlılıkları Yükleyin
```bash
pip install -r requirements.txt
```

### 3. API Anahtarını Ayarlayın
Projenin kök dizininde bir `.env` dosyası oluşturun ve Gemini API anahtarınızı ekleyin:
```env
GEMINI_API_KEY=YOUR_API_KEY_HERE
```
> **Not:** API anahtarınızı [Google AI Studio](https://aistudio.google.com/) üzerinden ücretsiz alabilirsiniz.

### 4. Uygulamayı Başlatın
```bash
streamlit run streamlit_app.py
```

## Proje Yapısı

-   `streamlit_app.py`: Ana uygulama dosyası.
-   `requirements.txt`: Gerekli Python kütüphaneleri.
-   `.env`: Hassas veriler (Git'e eklenmez).
-   `README.md`: Proje bilgileri.

## Lisans
Bu proje MIT Lisansı ile lisanslanmıştır.

---
Geliştiren: **Cem Evecen**
