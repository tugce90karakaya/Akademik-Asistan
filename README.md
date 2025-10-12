# 🔬 Bilimsel Makale Özetleyici Chatbot

Bu proje, **Streamlit** ile oluşturulmuş, **Groq'un Llama 3.1** Büyük Dil Modelini (LLM) kullanarak **RAG (Retrieval Augmented Generation)** mimarisiyle bilimsel makalelerden bilgi çıkaran hızlı bir soru-cevap asistanıdır.

Uygulama, verilen demo makaleleri içinden ilgili bilgileri alıp (Retrieval) bu bağlamı kullanarak doğru ve kaynaklı yanıtlar üretir (Generation).

---

### 🚀 Özellikler

* **Groq Entegrasyonu:** Düşük gecikmeli, yüksek hızlı yanıtlar için Groq API'si kullanılır.
* **RAG Mimarisi:** Doğruluk ve kaynak gösterme yeteneği için LangChain ile RAG zinciri oluşturulmuştur.
* **Basit Embedding:** Harici model indirmeye gerek kalmadan, basit **Hash-tabanlı Embedding** sınıfı kullanılarak vektör veritabanı (ChromaDB) oluşturulur.
* **Streamlit Arayüzü:** Kullanıcı dostu, hızlı ve etkileşimli web arayüzü.

---

### ⚙️ Kurulum ve Çalıştırma

Bu uygulamayı yerel makinenizde veya bir bulut servisinde çalıştırmak için aşağıdaki adımları takip edin.

#### 1. Proje Dosyaları

* `app.py`: Uygulamanın tüm Python kodu bu dosyada yer alır.

#### 2. Gerekli Kütüphaneleri Yükleme

Terminalinizde aşağıdaki komutu çalıştırarak gerekli tüm bağımlılıkları yükleyin:

```bash
pip install streamlit langchain-chroma langchain-groq python-dotenv
