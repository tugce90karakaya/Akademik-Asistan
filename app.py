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

st.set_page_config(
    page_title="Akademik Asistan",
    page_icon="🤖",
    layout="wide"
)

class SimpleHashEmbeddings(Embeddings):
    """Basit hash tabanlı embedding - model indirmeye gerek yok"""
    
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

@st.cache_resource
def setup_rag_system(api_key):
    if not api_key:
        st.error("❌ GROQ_API_KEY bulunamadı.")
        return None
    
    st.info("🚀 Sistem başlatılıyor...")
    
    embedding_model = SimpleHashEmbeddings(dimension=256)
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        groq_api_key=api_key,
        temperature=0.1
    )
    
    st.info("📚 Akademik makaleler yükleniyor...")
    
    # Veri setini kaldırıyoruz, sadece demo verisi kullanıyoruz
    rag_documents = [
        Document(
            page_content="""Yapay Zeka ve Makine Öğrenmesi
            
Yapay zeka (AI), bilgisayarların insan zekasını taklit etmesini sağlayan bir teknolojidir. Makine öğrenmesi, yapay zekanın bir alt dalıdır ve bilgisayarların deneyimlerden öğrenmesini sağlar.

Temel Kavramlar:
- Denetimli Öğrenme: Etiketli verilerle eğitim
- Denetimsiz Öğrenme: Etiketlenmemiş verilerde örüntü bulma
- Pekiştirmeli Öğrenme: Deneme yanılma ile öğrenme

Uygulamalar:
Görüntü tanıma, doğal dil işleme, otonom araçlar, tıbbi teşhis sistemleri.""",
            metadata={"source": "AI_Basics", "doc_id": 0, "title": "Yapay Zeka Temelleri"}
        ),
        Document(
            page_content="""Derin Öğrenme ve Yapay Sinir Ağları

Derin öğrenme, çok katmanlı yapay sinir ağları kullanarak karmaşık problemleri çözen bir makine öğrenmesi tekniğidir.

Önemli Mimariler:
- Konvolüsyonel Sinir Ağları (CNN): Görüntü işleme için
- Tekrarlayan Sinir Ağları (RNN): Zaman serisi ve dil modelleme için
- Transformer: Modern dil modelleri (GPT, BERT) için

Başarı Hikayeleri:
AlphaGo, GPT serisi, DALL-E, ChatGPT gibi sistemler derin öğrenme ile geliştirilmiştir.

Zorluklar:
Yüksek hesaplama maliyeti, büyük veri ihtiyacı, açıklanabilirlik sorunları.""",
            metadata={"source": "Deep_Learning", "doc_id": 1, "title": "Derin Öğrenme"}
        ),
        Document(
            page_content="""Doğal Dil İşleme (NLP)

Doğal dil işleme, bilgisayarların insan dilini anlaması ve işlemesi için kullanılan yapay zeka dalıdır.

Temel Görevler:
- Metin sınıflandırma
- Duygu analizi
- Makine çevirisi
- Soru-cevap sistemleri
- Metin üretimi

Modern Yaklaşımlar:
Transformer mimarisi ve transfer öğrenme, NLP'de devrim yaratmıştır. BERT, GPT gibi modeller, önceden eğitilmiş büyük dil modelleridir.

Türkçe NLP:
Türkçe morfolojik açıdan zengin bir dildir. Ek yapısı, NLP görevlerini zorlaştırır ancak son yıllarda Türkçe için özel modeller geliştirilmiştir.""",
            metadata={"source": "NLP", "doc_id": 2, "title": "Doğal Dil İşleme"}
        ),
        Document(
            page_content="""Bilgisayarlı Görü

Bilgisayarlı görü, makinelerin görsel dünyayı anlamasını sağlayan yapay zeka alanıdır.

Temel Uygulamalar:
- Nesne tanıma ve tespiti
- Yüz tanıma
- Görüntü segmentasyonu
- Otonom araç navigasyonu
- Tıbbi görüntü analizi

Teknolojiler:
CNN'ler, görüntü işlemede en başarılı modellerdir. ResNet, YOLO, U-Net gibi mimariler farklı görevlerde kullanılır.

Gelecek Trendler:
3D görü, video analizi, gerçek zamanlı işleme.""",
            metadata={"source": "Computer_Vision", "doc_id": 3, "title": "Bilgisayarlı Görü"}
        ),
        Document(
            page_content="""RAG (Retrieval Augmented Generation)

RAG, büyük dil modellerinin bilgi tabanından ilgili belgeleri alıp yanıt üretmesini sağlayan bir tekniktir.

Nasıl Çalışır:
1. Kullanıcı sorusu alınır
2. Embedding ile vektör haline getirilir
3. Vektör veritabanında benzer belgeler aranır
4. Bulunan belgeler LLM'e context olarak verilir
5. LLM, context kullanarak yanıt üretir

Avantajlar:
- Güncel bilgi kullanımı
- Hallüsinasyon azalması
- Domain-specific bilgi
- Kaynak gösterme

Bu chatbot da RAG mimarisi kullanmaktadır.""",
            metadata={"source": "RAG_Tech", "doc_id": 4, "title": "RAG Teknolojisi"}
        )
    ]
    
    st.success(f"✅ {len(rag_documents)} demo makalesi yüklendi!")
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " "]
    )
    chunks = text_splitter.split_documents(rag_documents)
    
    st.info(f"📊 {len(chunks)} metin parçası oluşturuldu...")
    
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory="/tmp/chroma_db"
    )
    
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    system_prompt = """Sen Türkçe konuşan bilimsel makale uzmanı bir assistantsın. 

Görevin:
- Sadece verilen bağlam kullanarak cevap ver
- Türkçe ve anlaşılır bir dil kullan
- Bilimsel terimleri açıkla
- Emin olmadığın konularda "Bu bilgi verilen metinde bulunmuyor" de

Bağlam: {context}"""
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}")
    ])
    
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)
    
    st.success("✅ RAG Sistemi Hazır! Artık soru sorabilirsiniz.")
    return rag_chain

st.title("🔬 Bilimsel Makale Özetleyici Chatbot")
st.markdown("**Gemini Flash ve RAG mimarisi** ile bilimsel makalelerden özetler çıkarır.")

with st.expander("ℹ️ Nasıl Kullanılır?"):
    st.markdown("""
    1. **Makale hakkında soru sorun:** "Yapay zeka nedir?"
    2. **Özet isteyin:** "Bu makaleleri özetle"
    3. **Karşılaştırma yapın:** "Hangi yöntemler kullanılmış?"
    
    **Not:** Chatbot sadece yüklenen makalelerden bilgi verir.
    """)

def get_huggingface_secret():
    try:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            return api_key
        try:
            api_key = st.secrets["GROQ_API_KEY"]
            return api_key
        except:
            pass
        return None
    except Exception as e:
        st.error(f"Secret alma hatası: {e}")
        return None

groq_api_key = get_huggingface_secret()

if groq_api_key:
    st.success("✅ API Key bulundu!")
    
    rag_chain = setup_rag_system(groq_api_key)
    
    if rag_chain:
        if "messages" not in st.session_state:
            st.session_state.messages = []
        
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        if prompt := st.chat_input("Makalelerle ilgili soru sorun (örn: 'Yapay zeka nedir?')"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.chat_message("assistant"):
                with st.spinner("🔍 Makaleler aranıyor ve cevap oluşturuluyor..."):
                    try:
                        response = rag_chain.invoke({"input": prompt})
                        answer = response['answer']
                        
                        sources = response.get('context', [])
                        if sources:
                            answer += f"\n\n📚 **Kaynak:** {len(sources)} makale parçasından bilgi kullanıldı."
                    except Exception as e:
                        answer = f"❌ Bir hata oluştu: {str(e)}\n\nLütfen soruyu yeniden deneyin."
                
                st.markdown(answer)
            
            st.session_state.messages.append({"role": "assistant", "content": answer})
        
        if st.session_state.messages:
            if st.button("🗑️ Sohbeti Temizle"):
                st.session_state.messages = []
                st.rerun()
else:
    st.error("""
    ❌ **GROQ_API_KEY bulunamadı!**
    
    **Ücretsiz Groq API Key almak için:**
    1. https://console.groq.com/keys adresine git
    2. Sign up ile üye ol (ücretsiz)
    3. Create API Key tıkla
    4. Key'i kopyala
    
    **Hugging Face Spaces'e ekle:**
    1. Space Settings → Repository secrets
    2. New secret: `GROQ_API_KEY` = `gsk_...`
    3. Sayfayı yenile
    
    **Neden Groq?**
    - Tamamen ücretsiz
    - Çok hızlı (Gemini'den 10x hızlı)
    - Llama 3 modeli kullanıyor
    """)
    
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔧 Sistem Bilgisi")
st.sidebar.info("""
**Model:** Llama 3.1 (8B Instant)  
**API:** Groq (Ücretsiz)  
**Embedding:** Hash-based  
**Veri:** 5 demo makalesi  
**RAG Framework:** LangChain
""")