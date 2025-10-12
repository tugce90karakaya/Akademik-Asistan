import streamlit as st
import os
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
import hashlib
from datasets import load_dataset # YENİ EKLENDİ

# --- ÖNEMLİ: Hugging Face cache yollarını düzelt ---
# Hugging Face Spaces'te önbellek sorunlarını önlemek için geçici dizin kullanır
os.environ["HF_HOME"] = "/tmp/huggingface"
os.environ["HF_DATASETS_CACHE"] = "/tmp/huggingface/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/tmp/huggingface/transformers"
os.environ["XDG_CACHE_HOME"] = "/tmp/huggingface"
os.environ["HF_HUB_CACHE"] = "/tmp/huggingface/hub"

os.makedirs("/tmp/huggingface/datasets", exist_ok=True)
os.makedirs("/tmp/huggingface/transformers", exist_ok=True)
os.makedirs("/tmp/huggingface/hub", exist_ok=True)

# --- Streamlit yapılandırması ---
st.set_page_config(
    page_title="Türk Akademik Tez Asistanı",
    page_icon="🎓",
    layout="wide"
)

# --- Hash tabanlı embedding sınıfı ---
class SimpleHashEmbeddings(Embeddings):
    """Hash tabanlı embedding - model indirmeye gerek yok"""
    def __init__(self, dimension=384):
        self.dimension = dimension
    
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            hash_obj = hashlib.sha256(text.encode())
            hash_bytes = hash_obj.digest()
            vector = [float(b) / 255.0 for b in hash_bytes[:self.dimension]]
            vector += [0.0] * (self.dimension - len(vector))
            embeddings.append(vector)
        return embeddings
    
    def embed_query(self, text):
        return self.embed_documents([text])[0]

# --- RAG sistemi kurulumu ---
@st.cache_resource
def setup_rag_system(api_key, hf_token=None):
    if not api_key:
        st.error("❌ GROQ_API_KEY bulunamadı.")
        return None
    
    st.info("🚀 RAG sistemi başlatılıyor...")

    embedding_model = SimpleHashEmbeddings(dimension=256)
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.1
    )

    st.info("📚 Türk akademik tez veri seti yükleniyor...")
    rag_documents = []

    # --- Dataset yükleme ---
    try:
        
        with st.spinner("Dataset stream ediliyor... (İlk seferde 30 saniye sürebilir)"):
            # load_dataset kullanımı, hf_token ile yetkilendirme sağlar
            dataset = load_dataset(
                "umutertugrul/turkish-academic-theses-dataset",
                split="train",
                streaming=True, 
                token=hf_token, # HF Token'ı burada kullanıyoruz
                cache_dir="/tmp/huggingface"
            )

            count = 0
            max_docs = 100 # Demo için 100 tezle sınırlıyoruz
            for item in dataset:
                if count >= max_docs:
                    break
                if item.get('abstract_tr') and item.get('title_tr'):
                    title = item['title_tr']
                    abstract = item['abstract_tr']
                    author = item.get('author', 'Bilinmeyen')
                    year = item.get('year', 'Bilinmeyen')
                    subject = item.get('subject', 'Genel')

                    content = f"Başlık: {title}\n\nYazar: {author}\nYıl: {year}\nKonu: {subject}\n\nÖzet:\n{abstract}"
                    rag_documents.append(Document(
                        page_content=content,
                        metadata={
                            "source": "YÖK Tez Merkezi",
                            "doc_id": count,
                            "title": title,
                            "author": author,
                            "year": str(year),
                            "subject": subject
                        }
                    ))
                    count += 1

        if rag_documents:
            st.success(f"✅ {len(rag_documents)} Türk akademik tezi başarıyla stream edildi!")
        else:
            raise Exception("Dataset boş döndü veya erişim sağlanamadı.")

    except Exception as e:
        error_msg = str(e)
        st.warning(f"⚠️ Dataset stream edilemedi: {error_msg[:200]}...")
        st.info("💡 Yedek veri seti kullanılıyor")

        # --- Yedek veri seti ---
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

    # --- Text splitting ---
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=700,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(rag_documents)
    st.info(f"📊 {len(chunks)} metin parçası oluşturuldu")

    # --- Vektör veritabanı ---
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="/tmp/turkish_thesis_db"
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 4}
    )

    # --- Türkçe prompt ---
    system_prompt = """Sen Türk akademik tezlerini analiz eden bir asistansın.

Görevlerin:
1. Verilen bağlam (tezler) kullanarak Türkçe cevap ver
2. Tez başlıklarını, yazarlarını ve yıllarını belirt
3. Bulguları ve yöntemleri açıkla
4. Emin olmadığın konularda "Bu bilgi verilen tezlerde yok" de

Bağlam (Tezler):
{context}"""

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])

    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    st.success("✅ RAG Sistemi Hazır!")
    return rag_chain

# --- UI ---
st.title("🎓 Türk Akademik Tez Araştırma Asistanı")
st.markdown("**RAG + Llama 3.1** ile Türk akademik tezlerinden bilgi çıkarır")

with st.expander("ℹ️ Nasıl Kullanılır?"):
    st.markdown("""
    **Örnek Sorular:**
    - "Türkçe NLP çalışmaları hakkında bilgi ver"
    - "Siber güvenlik alanında hangi tezler var?"
    - "Makine öğrenmesi ile ilgili tezleri listele"
    - "2023 yılındaki tezler neler?"
    - "Çevre mühendisliği alanında neler yapılmış?"
    """)

# --- API key fonksiyonları ---
def get_api_key():
    # Groq API Key
    api_key = os.environ.get("GROQ_API_KEY")
    if api_key:
        return api_key
    try:
        return st.secrets["GROQ_API_KEY"]
    except:
        return None

def get_hf_token():
    # Hugging Face Token (Gated Dataset Erişimi İçin)
    # st.secrets, HF_TOKEN veya HUGGING_FACE_HUB_TOKEN olarak ayarlanmış olabilir.
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if token:
        return token
    try:
        return st.secrets.get("HF_TOKEN") or st.secrets.get("HUGGING_FACE_HUB_TOKEN")
    except:
        return None

groq_api_key = get_api_key()
hf_token = get_hf_token()

# --- Ana RAG çalışma akışı ---
if groq_api_key:
    st.success("✅ Groq API Key bulundu!")
    
    # HF Token'ın varlığını kontrol ediyoruz
    if not hf_token:
        st.warning("⚠️ Hugging Face Token (HF_TOKEN) bulunamadı. Kilitli veri setine erişim başarısız olabilir.")

    rag_chain = setup_rag_system(groq_api_key, hf_token)

    if rag_chain:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Türk akademik tezleri hakkında soru sorun..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Tezler aranıyor..."):
                    try:
                        response = rag_chain.invoke({"input": prompt})
                        answer = response['answer']
                        sources = response.get('context', [])
                        if sources:
                            titles = set()
                            for doc in sources:
                                title = doc.metadata.get('title', 'Bilinmeyen')
                                author = doc.metadata.get('author', '')
                                year = doc.metadata.get('year', '')
                                titles.add(f"{title} ({author}, {year})")
                            if titles:
                                answer += f"\n\n📚 **Kaynak Tezler:**\n" + "\n".join([f"- {t}" for t in list(titles)[:3]])
                    except Exception as e:
                        answer = f"❌ Hata: {str(e)}"
                st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        
        col1, col2 = st.columns([1, 5])
        with col1:
            if st.button("🗑️ Temizle"):
                st.session_state.messages = []
                st.rerun()

else:
    st.error("""
    ❌ **GROQ_API_KEY bulunamadı!**
    
    **API Key almak için:**
    1. https://console.groq.com/keys
    2. Sign up → Create API Key
    3. Streamlit secrets veya environment variable olarak ekle
    """)

# --- Sidebar ---
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

st.sidebar.markdown("### 📌 İstatistikler")
if 'messages' in st.session_state:
    st.sidebar.metric("Toplam Soru", len([m for m in st.session_state.messages if m["role"]=="user"]))