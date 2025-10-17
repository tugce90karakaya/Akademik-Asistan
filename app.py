# =============================================================================
# TÜRK AKADEMİK TEZ ARAŞTIRMA ASİSTANI
# =============================================================================
# Bu uygulama, RAG (Retrieval Augmented Generation) mimarisi kullanarak
# Türk akademik tezlerinden bilgi çıkarır ve kullanıcı sorularına yanıt verir.
#
# Kullanılan Teknolojiler:
# - LangChain: RAG pipeline yönetimi
# - ChromaDB: Vektör veritabanı
# - Groq API (Llama 3.1): Dil modeli
# - Streamlit: Web arayüzü
# - Hugging Face Datasets: Veri kaynağı
# =============================================================================

import streamlit as st  # Web arayüzü için Streamlit kütüphanesi
import os  # İşletim sistemi işlemleri (environment variables, dizin oluşturma)
from langchain_chroma import Chroma  # ChromaDB vektör veritabanı entegrasyonu
from langchain_groq import ChatGroq  # Groq API ile Llama model entegrasyonu
from langchain.chains import create_retrieval_chain  # RAG zinciri oluşturma
from langchain.chains.combine_documents import create_stuff_documents_chain  # Belgeleri birleştirme
from langchain.prompts import ChatPromptTemplate  # Prompt şablonu oluşturma
from langchain.schema import Document  # LangChain Document formatı
from langchain.text_splitter import RecursiveCharacterTextSplitter  # Metin bölme
from langchain.embeddings.base import Embeddings  # Embedding base sınıfı
import hashlib  # Hash fonksiyonları (custom embedding için)
from datasets import load_dataset  # Hugging Face datasets kütüphanesi
from dotenv import load_dotenv  # ✅ .env dosyasını okumak için eklendi
load_dotenv()  # .env içindeki değişkenleri yükle

# =============================================================================
# HUGGING FACE CACHE AYARLARI
# =============================================================================
# Hugging Face Spaces'te varsayılan cache dizini (~/.cache) yazma korumalıdır.
# Bu yüzden tüm cache işlemlerini /tmp dizinine yönlendiriyoruz.
# /tmp dizini geçici ve yazılabilir bir dizindir.

os.environ["HF_HOME"] = "/tmp/huggingface"  # Ana Hugging Face dizini
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"  # Dataset cache
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"  # Model cache
os.environ["XDG_CACHE_HOME"] = "/tmp/huggingface"  # Genel cache dizini
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/hub"  # Hub cache

# Cache dizinlerini oluştur (eğer yoksa)
# exist_ok=True: Dizin zaten varsa hata verme
os.makedirs("/tmp/huggingface/datasets", exist_ok=True)
os.makedirs("/tmp/huggingface/transformers", exist_ok=True)
os.makedirs("/tmp/huggingface/hub", exist_ok=True)

# =============================================================================
# STREAMLIT SAYFA AYARLARI
# =============================================================================
# Uygulama başlığı, ikonu ve layout ayarları
st.set_page_config(
    page_title="Türk Akademik Tez Asistanı",  # Browser sekmesinde görünen başlık
    page_icon="🎓",  # Browser sekmesinde görünen ikon
    layout="wide"  # Geniş layout (varsayılan: centered)
)

# =============================================================================
# CUSTOM EMBEDDING SINIFI
# =============================================================================
# Model indirmeyi önlemek için hash tabanlı basit bir embedding sınıfı.
# Her metin SHA-256 hash'i ile vektöre dönüştürülür.
# Dezavantaj: Semantik benzerlik yakalamaz (sadece özdeşlik)
# Avantaj: Hızlı, model indirmez, offline çalışır

class SimpleHashEmbeddings(Embeddings):
    """
    Hash tabanlı embedding sınıfı.
    
    Metinleri SHA-256 hash algoritması ile sabit boyutlu vektörlere dönüştürür.
    Model indirme gerektirmez, bu yüzden deployment sorunları yaşanmaz.
    
    Args:
        dimension (int): Embedding vektörünün boyutu (varsayılan: 384)
    """
    
    def __init__(self, dimension=384):
        """
        Sınıfı başlat.
        
        Args:
            dimension (int): Çıktı vektörünün boyutu
        """
        self.dimension = dimension  # Vektör boyutunu sakla
    
    def embed_documents(self, texts):
        """
        Birden fazla metni embedding vektörlerine dönüştür.
        
        Args:
            texts (List[str]): Embedding'e çevrilecek metin listesi
            
        Returns:
            List[List[float]]: Her metin için embedding vektörü
        """
        embeddings = []  # Sonuç vektörlerini sakla
        
        for text in texts:
            # 1. Metni SHA-256 ile hash'le
            hash_obj = hashlib.sha256(text.encode())  # UTF-8 encode + hash
            
            # 2. Hash'i byte dizisine çevir
            hash_bytes = hash_obj.digest()  # 32 byte (256 bit)
            
            # 3. Byte'ları 0-1 arası float'a normalize et
            # İlk 'dimension' kadar byte'ı kullan
            vector = [float(b) / 255.0 for b in hash_bytes[:self.dimension]]
            
            # 4. Eğer dimension > 32 ise, geri kalanını 0 ile doldur
            vector += [0.0] * (self.dimension - len(vector))
            
            embeddings.append(vector)  # Listeye ekle
        
        return embeddings
    
    def embed_query(self, text):
        """
        Tek bir sorgu metnini embedding vektörüne dönüştür.
        
        Args:
            text (str): Embedding'e çevrilecek metin
            
        Returns:
            List[float]: Embedding vektörü
        """
        # embed_documents fonksiyonunu kullan (tek elemanlı liste)
        return self.embed_documents([text])[0]

# =============================================================================
# RAG SİSTEMİ KURULUM FONKSİYONU
# =============================================================================
# Bu fonksiyon tüm RAG pipeline'ını oluşturur:
# 1. Dataset yükleme
# 2. Metin bölme (chunking)
# 3. Embedding + Vektör DB oluşturma
# 4. Retriever + LLM + Prompt birleştirme

@st.cache_resource  # Sonucu cache'le, her reload'da yeniden çalıştırma
def setup_rag_system(api_key, hf_token=None):
    """
    RAG sistemini kur ve hazırla.
    
    Args:
        api_key (str): Groq API key
        hf_token (str, optional): Hugging Face token (private dataset için)
        
    Returns:
        Pipeline: Kullanıma hazır RAG chain veya None (hata durumunda)
    """
    
    # --- API Key Kontrolü ---
    if not api_key:
        st.error("❌ GROQ_API_KEY bulunamadı.")
        return None
    
    st.info("🚀 RAG sistemi başlatılıyor...")

    # --- 1. EMBEDDING MODELİ OLUŞTUR ---
    # Hash tabanlı custom embedding (256 boyutlu vektör)
    embedding_model = SimpleHashEmbeddings(dimension=256)
    
    # --- 2. DİL MODELİNİ OLUŞTUR ---
    # Groq API ile Llama 3.1 modeli (8 milyar parametre, instant versiyonu)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Model adı
        groq_api_key=api_key,  # API key
        temperature=0.1  # Düşük temperature = daha tutarlı yanıtlar
    )

    st.info("📚 Türk akademik tez veri seti yükleniyor...")
    rag_documents = []  # Yüklenecek belgeleri sakla

    # --- 3. DATASET YÜKLEME ---
    try:
        # Spinner ile kullanıcıya ilerleme göster
        with st.spinner("Dataset stream ediliyor... (İlk seferde 30 saniye sürebilir)"):
            
            # Hugging Face'ten dataset yükle
            # streaming=True: Tüm dataset'i indirmez, ihtiyaç duyulan kısmı alır
            dataset = load_dataset(
                "umutertugrul/turkish-academic-theses-dataset",  # Dataset adı
                split="train",  # Train split'ini kullan
                streaming=True,  # Stream mode (cache gerektirmez)
                token=hf_token,  # Gated dataset için token
                cache_dir="/tmp/huggingface"  # Geçici cache dizini
            )

            count = 0  # Yüklenen belge sayacı
            max_docs = 100  # Maksimum belge sayısı (demo için sınırlı)
            
            # Stream'den belgeleri tek tek al
            for item in dataset:
                # Maksimum sayıya ulaştıysak dur
                if count >= max_docs:
                    break
                
                # Türkçe başlık ve özet varsa işle
                if item.get('abstract_tr') and item.get('title_tr'):
                    # Metadata bilgilerini al
                    title = item['title_tr']  # Tez başlığı
                    abstract = item['abstract_tr']  # Tez özeti
                    author = item.get('author', 'Bilinmeyen')  # Yazar
                    year = item.get('year', 'Bilinmeyen')  # Yıl
                    subject = item.get('subject', 'Genel')  # Konu/alan

                    # Belge içeriğini formatla
                    # Format: Başlık + Metadata + Özet
                    content = f"Başlık: {title}\n\nYazar: {author}\nYıl: {year}\nKonu: {subject}\n\nÖzet:\n{abstract}"
                    
                    # LangChain Document objesi oluştur
                    rag_documents.append(Document(
                        page_content=content,  # Ana metin
                        metadata={  # Metadata (retrieval sonrası kullanılır)
                            "source": "YÖK Tez Merkezi",
                            "doc_id": count,
                            "title": title,
                            "author": author,
                            "year": str(year),
                            "subject": subject
                        }
                    ))
                    count += 1  # Sayacı artır

        # Başarı kontrolü
        if rag_documents:
            st.success(f"✅ {len(rag_documents)} Türk akademik tezi başarıyla stream edildi!")
        else:
            # Dataset boşsa exception fırlat
            raise Exception("Dataset boş döndü veya erişim sağlanamadı.")

    except Exception as e:
        # Hata durumunda yedek veri setini kullan
        error_msg = str(e)
        st.warning(f"⚠️ Dataset stream edilemedi: {error_msg[:200]}...")
        st.info("💡 Yedek veri seti kullanılıyor")

        # --- YEDEK VERİ SETİ ---
        # Dataset yüklenemezse kullanılacak örnek tezler
        rag_documents = [
            Document(
                page_content="""Başlık: Derin Öğrenme Yöntemleri ile Türkçe Doğal Dil İşleme

Yazar: Ahmet Yılmaz
Yıl: 2023
Konu: Bilgisayar Mühendisliği

Özet:
Bu tez çalışmasında, Türkçe dil işleme görevleri için derin öğrenme tabanlı yöntemler geliştirilmiştir. BERT ve GPT mimarileri Türkçe korpus üzerinde fine-tune edilmiş, metin sınıflandırma ve duygu analizi görevlerinde %92.5 F1-score elde edilmiştir. TürkçeBERT modeli geliştirilmiş ve açık kaynak olarak paylaşılmıştır.""",
                metadata={"source": "YÖK", "doc_id": 0, "title": "Derin Öğrenme ile Türkçe NLP", "author": "Ahmet Yılmaz", "year": "2023", "subject": "Bilgisayar Mühendisliği"}
            ),
            Document(
                page_content="""Başlık: Makine Öğrenmesi ile Türkiye'de Hava Kirliliği Tahmini

Yazar: Zeynep Kaya
Yıl: 2022
Konu: Çevre Mühendisliği

Özet:
Türkiye'nin büyük şehirlerinde hava kalitesi tahmini için LSTM ve Random Forest algoritmaları kullanılmıştır. İstanbul, Ankara ve İzmir'den toplanan 5 yıllık hava kalitesi verisi ile model eğitilmiş, PM2.5 ve PM10 değerleri %87 doğrulukla tahmin edilmiştir. Mevsimsel faktörlerin etkisi analiz edilmiştir.""",
                metadata={"source": "YÖK", "doc_id": 1, "title": "ML ile Hava Kirliliği Tahmini", "author": "Zeynep Kaya", "year": "2022", "subject": "Çevre Mühendisliği"}
            )
        ]
        st.success(f"✅ {len(rag_documents)} yedek Türk akademik tezi yüklendi")

    # --- 4. METİN BÖLME (CHUNKING) ---
    # Uzun belgeleri küçük parçalara böl (retrieval performansını artırır)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,  # Her chunk maksimum 700 karakter
        chunk_overlap=100,  # Chunk'lar arası 100 karakter örtüşme
        separators=["\n\n", "\n", ". ", " "]  # Bölme sırası: paragraf > satır > cümle > kelime
    )
    chunks = text_splitter.split_documents(rag_documents)  # Belgeleri böl
    st.info(f"📊 {len(chunks)} metin parçası oluşturuldu")

    # --- 5. VEKTÖR VERİTABANI OLUŞTUR ---
    # ChromaDB: Embedding'leri saklayan ve similarity search yapan veritabanı
    vectorstore = Chroma.from_documents(
        documents=chunks,  # Chunk'lanmış belgeler
        embedding=embedding_model,  # Embedding modeli (hash-based)
        persist_directory="/tmp/turkish_thesis_db"  # Kalıcı depolama dizini
    )

    # --- 6. RETRIEVER OLUŞTUR ---
    # Retriever: Sorguya en benzer belgeleri bulur
    retriever = vectorstore.as_retriever(
        search_type="similarity",  # Benzerlik tabanlı arama
        search_kwargs={"k": 4}  # En benzer 4 chunk'ı getir
    )

    # --- 7. PROMPT ŞABLONU OLUŞTUR ---
    # System prompt: Modele rolünü ve talimatları ver
    system_prompt = """Sen Türk akademik tezlerini analiz eden bir asistansın.

Görevlerin:
1. Verilen bağlam (tezler) kullanarak Türkçe cevap ver
2. Tez başlıklarını, yazarlarını ve yıllarını belirt
3. Bulguları ve yöntemleri açıkla
4. Emin olmadığın konularda "Bu bilgi verilen tezlerde yok" de

Bağlam (Tezler):
{context}"""  # {context} retriever'dan gelecek

    # ChatPromptTemplate: System + User mesajlarını birleştir
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),  # System mesajı (rol tanımı)
        ("human", "{input}")  # User mesajı (soru)
    ])

    # --- 8. RAG CHAIN'İ BİRLEŞTİR ---
    # Document chain: Belgeleri prompt'a ekle ve LLM'e gönder
    document_chain = create_stuff_documents_chain(llm, prompt)
    
    # Retrieval chain: Retriever + Document chain'i birleştir
    # Flow: Soru -> Retriever (belgeler) -> Prompt (belgeler + soru) -> LLM -> Cevap
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    st.success("✅ RAG Sistemi Hazır!")
    return rag_chain  # Kullanıma hazır RAG pipeline'ı döndür

# =============================================================================
# STREAMLIT KULLANICI ARAYÜZÜ
# =============================================================================

# --- Sayfa Başlığı ---
st.title("🎓 Akademik Asistan")
st.markdown("**RAG + Llama 3.1** ile Türk akademik tezlerinden bilgi çıkarır")

# --- Kullanım Kılavuzu (Genişletilebilir) ---
with st.expander("ℹ️ Nasıl Kullanılır?"):
    st.markdown("""
    **Örnek Sorular:**
    - "Türkçe NLP çalışmaları hakkında bilgi ver"
    - "Siber güvenlik alanında hangi tezler var?"
    - "Makine öğrenmesi ile ilgili tezleri listele"
    - "2023 yılındaki tezler neler?"
    - "Çevre mühendisliği alanında neler yapılmış?"
    """)

# =============================================================================
# API KEY ALMA FONKSİYONLARI
# =============================================================================

def get_api_key():
    """
    Groq API key'i environment variable veya Streamlit secrets'tan al.
    
    Returns:
        str or None: API key veya bulunamazsa None
    """
    # Önce environment variable'ı kontrol et
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key
    
    # Environment'ta yoksa Streamlit secrets'a bak
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return None

def get_hf_token():
    """
    Hugging Face token'ı environment variable veya Streamlit secrets'tan al.
    Gated/private dataset'ler için gerekli.
    
    Returns:
        str or None: HF token veya bulunamazsa None
    """
    # Farklı environment variable isimlerini kontrol et
    token = (os.environ.get("HF_TOKEN") or 
             os.environ.get("HUGGING_FACE_HUB_TOKEN") or 
             os.environ.get("HUGGINGFACE_TOKEN"))
    
    if token:
        return token
    
    # Environment'ta yoksa Streamlit secrets'a bak
    try:
        return (st.secrets.get("HF_TOKEN") or 
                st.secrets.get("HUGGING_FACE_HUB_TOKEN"))
    except:
        return None

# =============================================================================
# ANA UYGULAMA AKIŞI
# =============================================================================

# API key'leri al
groq_api_key = get_api_key()
hf_token = get_hf_token()

# API key kontrolü
if groq_api_key:
    # Key bulundu, başarı mesajı göster
    st.success("✅ Groq API Key bulundu!")
    
    # HF token kontrolü (opsiyonel, uyarı amaçlı)
    if not hf_token:
        st.warning("⚠️ Hugging Face Token (HF_TOKEN) bulunamadı. Kilitli veri setine erişim başarısız olabilir.")

    # RAG sistemini kur
    rag_chain = setup_rag_system(groq_api_key, hf_token)

    # RAG sistemi başarıyla kurulduysa chat arayüzünü göster
    if rag_chain:
        # --- SESSION STATE BAŞLATMA ---
        # Streamlit'te sayfa her yenilendiğinde değişkenler sıfırlanır.
        # session_state ile kalıcı veri saklayabiliriz (chat geçmişi gibi)
        if "messages" not in st.session_state:
            st.session_state.messages = []  # Boş mesaj listesi oluştur
        
        # --- CHAT GEÇMİŞİNİ GÖSTER ---
        # Daha önce yapılan konuşmaları ekrana yazdır
        for message in st.session_state.messages:
            # Chat mesajı bubble'ı oluştur (user veya assistant)
            with st.chat_message(message["role"]):
                st.markdown(message["content"])  # Mesaj içeriğini göster
        
        # --- KULLANICI GİRDİSİ AL ---
        # Chat input box (sayfa altında sabit kalır)
        if prompt := st.chat_input("Türk akademik tezleri hakkında soru sorun..."):
            # Kullanıcı bir şey yazdı ve enter'a bastı
            
            # 1. Kullanıcı mesajını session state'e ekle
            st.session_state.messages.append({"role": "user", "content": prompt})
            
            # 2. Kullanıcı mesajını ekranda göster
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # --- ASISTAN CEVABINI OLUŞTUR ---
            with st.chat_message("assistant"):
                # Spinner ile işlem devam ediyor mesajı göster
                with st.spinner("🔍 Tezler aranıyor..."):
                    try:
                        # RAG chain'i çalıştır
                        # Input: Kullanıcı sorusu
                        # Output: {'answer': ..., 'context': [...]}
                        response = rag_chain.invoke({"input": prompt})
                        
                        # Cevabı al
                        answer = response['answer']
                        
                        # Kaynak belgeleri al (retriever'dan gelen chunk'lar)
                        sources = response.get('context', [])
                        
                        # Eğer kaynak varsa, cevabın sonuna ekle
                        if sources:
                            titles = set()  # Tekrar eden başlıkları önlemek için set kullan
                            
                            # Her kaynak belgenin metadata'sını çıkar
                            for doc in sources:
                                title = doc.metadata.get('title', 'Bilinmeyen')
                                author = doc.metadata.get('author', '')
                                year = doc.metadata.get('year', '')
                                titles.add(f"{title} ({author}, {year})")  # Formatla ve ekle
                            
                            # Kaynakları cevaba ekle (maksimum 3 tane)
                            if titles:
                                answer += f"\n\n📚 **Kaynak Tezler:**\n" + "\n".join([f"- {t}" for t in list(titles)[:3]])
                    
                    except Exception as e:
                        # Hata oluştuysa kullanıcıya bildir
                        answer = f"❌ Hata: {str(e)}"
                
                # Cevabı ekrana yazdır
                st.markdown(answer)
            
            # Asistan cevabını session state'e ekle
            st.session_state.messages.append({"role": "assistant", "content": answer})
        
        # --- TEMİZLE BUTONU ---
        # Sohbet geçmişini sıfırla
        col1, col2 = st.columns([1, 5])  # 2 kolonlu layout
        with col1:
            if st.button("🗑️ Temizle"):
                st.session_state.messages = []  # Mesaj geçmişini boşalt
                st.rerun()  # Sayfayı yenile

else:
    # API key bulunamadı, hata mesajı göster
    st.error("""
    ❌ **GROQ_API_KEY bulunamadı!**
    
    **API Key almak için:**
    1. https://console.groq.com/keys
    2. Sign up → Create API Key
    3. Streamlit secrets veya environment variable olarak ekle
    """)

# =============================================================================
# SIDEBAR (YAN PANEL)
# =============================================================================

# --- Proje Bilgileri ---
st.sidebar.markdown("### 📊 Proje Bilgileri")
st.sidebar.info("""
**Veri Seti:**  
YÖK Tez Merkezi  
(turkish-academic-theses-dataset)

**Model:** Llama 3.1 (8B)  
**Embedding:** Hash-based  
**Vector DB:** ChromaDB  
**Framework:** LangChain
""")

# --- İstatistikler ---
st.sidebar.markdown("### 📌 İstatistikler")
# Eğer mesaj varsa, kullanıcı soru sayısını göster
if 'messages' in st.session_state:
    user_questions = len([m for m in st.session_state.messages if m["role"]=="user"])
    st.sidebar.metric("Toplam Soru", user_questions)

# =============================================================================
# KOD SONU
# =============================================================================
# Bu uygulamayı çalıştırmak için:
# 1. streamlit run app.py
# 2. GROQ_API_KEY environment variable'ını ayarla
# 3. (Opsiyonel) HF_TOKEN ayarla (gated dataset için)
# =============================================================================