---
title: Türk Akademik Tez Araştırma Asistanı 🎓
emoji: 📚
colorFrom: blue
colorTo: indigo
sdk: streamlit
sdk_version: "1.33.0"
app_file: app.py
pinned: true
---

# 🎓 Türk Akademik Tez Araştırma Asistanı

Bu proje, **RAG (Retrieval Augmented Generation)** mimarisi kullanarak **Türk akademik tezlerinden bilgi çıkaran** bir yapay zekâ araştırma asistanıdır.  
Uygulama, **LangChain**, **ChromaDB**, ve **Groq API (Llama 3.1)** teknolojilerini bir araya getirir.  
Kullanıcı, Türkçe olarak tezlerle ilgili sorular sorar ve sistem en uygun tez özetlerinden **kaynaklı, doğru ve doğal** yanıtlar üretir.

---

## 🚀 Özellikler

| Özellik | Açıklama |
|----------|-----------|
| 🧠 **Groq Llama 3.1 Entegrasyonu** | Düşük gecikmeli, yüksek doğruluklu yanıtlar için Groq API’si kullanılır. |
| 🔍 **RAG (Retrieval-Augmented Generation)** | Bilgiyi doğrudan akademik tezlerden çekerek yanıt üretir. |
| 💾 **Hash-tabanlı Embedding** | Harici model indirmeden, SHA-256 tabanlı embedding ile hızlı vektör temsili sağlar. |
| 🗄️ **ChromaDB Vektör Veritabanı** | Tez parçacıkları embedding’lenir ve ChromaDB içinde benzerlik aramaları yapılır. |
| 🧩 **LangChain Pipeline** | Retriever + LLM + Prompt yönetimini otomatikleştirir. |
| 💬 **Streamlit Arayüzü** | Kullanıcı dostu web arayüzü ile etkileşimli soru-cevap deneyimi sağlar. |

---

## 📚 Kullanılan Teknolojiler

| Teknoloji | Amaç |
|------------|------|
| [Streamlit](https://streamlit.io) | Web arayüzü oluşturma |
| [LangChain](https://www.langchain.com) | RAG zinciri, prompt yönetimi ve pipeline oluşturma |
| [ChromaDB](https://www.trychroma.com) | Vektör veritabanı ve benzerlik araması |
| [Groq API](https://console.groq.com/) | Llama 3.1 LLM entegrasyonu |
| [Hugging Face Datasets](https://huggingface.co/datasets/umutertugrul/turkish-academic-theses-dataset) | Türk akademik tez verisi kaynağı |
| [Python hashlib](https://docs.python.org/3/library/hashlib.html) | Hash-tabanlı embedding oluşturma |

---

## 🧾 Veri Seti Hakkında

### 📘 Kaynak:
**Dataset Adı:** [`umutertugrul/turkish-academic-theses-dataset`](https://huggingface.co/datasets/umutertugrul/turkish-academic-theses-dataset)  
**Sağlayıcı:** [YÖK Tez Merkezi](https://tez.yok.gov.tr/UlusalTezMerkezi/)

### 🧩 İçerik:
Bu veri seti, YÖK Tez Merkezi’nden derlenmiş Türkçe yüksek lisans ve doktora tezlerinin **başlık**, **yazar**, **yıl**, **konu** ve **özet** bilgilerini içerir.  
Aşağıda örnek bir veri yapısı gösterilmektedir:

```json
{
  "title_tr": "Makine Öğrenmesi ile Hava Kirliliği Tahmini",
  "abstract_tr": "Bu tezde Türkiye'de hava kalitesinin tahmini için LSTM modelleri kullanılmıştır...",
  "author": "Zeynep Kaya",
  "year": "2022",
  "subject": "Çevre Mühendisliği"
}
---

## 💡 Projeyi Kendi Bilgisayarında Çalıştırmak İçin

Bu projeyi GitHub üzerinden klonlayarak kendi bilgisayarınızda birkaç adımda çalıştırabilirsiniz 👇  

### 1️⃣ Depoyu Klonlayın
GitHub sayfamdan projeyi kopyalayın:
```bash
git clone https://github.com/kullaniciadi/akademik-tez-asistani.git
cd akademik-tez-asistani
