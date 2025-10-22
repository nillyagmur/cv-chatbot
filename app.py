"""
RAG Tabanlı CV Chatbot - OpenAI Versiyonu
Nil Yağmur Muslu'nun CV bilgilerini kullanarak sorulara cevap verir.
"""
#Bölüm 1 Gerekli Kütüphanelerin Yüklenmesi
#Bu bölümde projenin çalışması için gereken araçları çağıracağız.

import os # işletim sistemi ile alakalı dosya var mı kontrolü vb.
import warnings 
warnings.filterwarnings('ignore')  #gereksiz uyarıları gizlemek için

import streamlit as st #web arayüzünü oluşturmak için kullanılan kütüphane
from langchain.text_splitter import RecursiveCharacterTextSplitter #metni küçük parçalara bölmek için
from langchain_community.vectorstores import FAISS #vektör veri tabanı kütüphanesi
try:
    from langchain_community.embeddings import HuggingFaceEmbeddings
except ImportError:
    from langchain.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI #gpt modellerine bağlanmak için
import pickle #python nesnelerini diske kaydetmek için

#BÖLÜM 2 Streamlift Sayfa Konfigürasyonu
#bu bölüm tarayıcıda nasıl gözükeceğini belirliyor.
# ===============================
# Sayfa Konfigürasyonu
# ===============================

st.set_page_config(
    page_title="Nil Yağmur Muslu - CV Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
#BÖLÜM 3 SAbit Değişkenler
# proje boyunca değişmeyecek ayarları tek bir yere toplama
# ===============================
# Konfigürasyon
# ===============================

DATA_FILE = "data.txt" #cv bilgisi içerir. RAG için kullanılır.
FAISS_INDEX_PATH = "./faiss_index.pkl" #oluşturulacak veri tabanının dosya adı
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# OpenAI API Key (sabit olarak ayarlandı)
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
#BÖLÜM 4
# ===============================
# Fonksiyonlar
# ===============================

@st.cache_resource  #bu fonksiyonun sonucunu aklında tut tekrar tekrar yazdırma demek
def load_vector_store():
    """FAISS vector store'u yükler veya oluşturur.Bu fonksiyon, FAISS vektör veritabanını yükler.Eğer 'faiss_index.pkl' dosyası diskte varsa, onu yükler.
    Eğer yoksa, 'data.txt' dosyasından sıfırdan bir index oluşturur."""
    if os.path.exists(FAISS_INDEX_PATH):
        with open(FAISS_INDEX_PATH, 'rb') as f:
            vectorstore = pickle.load(f)
        return vectorstore
    else:
        # Veriyi yükle
        with open(DATA_FILE, 'r', encoding='utf-8') as file:
            text = file.read()
        
        # Chunk'lara böl
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500, #her parça en fazla 500 karakter olsun
            chunk_overlap=100, #parçalar arası 100 karakter ortak olsun
            separators=["\n### ", "\n\n", "\n", " ", ""]
        )
        chunks = text_splitter.split_text(text)
        
        # Embeddings oluştur metin parçaları matematiksel vektörlere dönüşür.
        embeddings = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # FAISS index oluştur metin parçaları ve embedding modelini alıp yapar.
        vectorstore = FAISS.from_texts(texts=chunks, embedding=embeddings)
        
        # gelecekte kullanmak için kaydet
        with open(FAISS_INDEX_PATH, 'wb') as f:
            pickle.dump(vectorstore, f)
        
        return vectorstore

def get_response(question: str, vectorstore):
    """Kullanıcı sorusuna RAG kullanarak cevap üretir."""
    try:
        # OpenAI model'ini yapılandır
        llm = ChatOpenAI(
            api_key=OPENAI_API_KEY,
            model="gpt-3.5-turbo",
            temperature=0.3,
            max_tokens=500
        )
        
        # Alakalı chunk'ları bul
        retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
        relevant_docs = retriever.invoke(question)
        
        # Context oluştur
        context = "\n\n".join([doc.page_content for doc in relevant_docs])
        
        # Prompt
        prompt = f"""Sen Nil Yağmur Muslu'nun kişisel CV asistanısın. Verilen bağlam bilgilerini kullanarak soruları yanıtla.

BAĞLAM:
{context}

SORU: {question}

YANITLAMA KURALLARI:
1. Sadece verilen bağlam bilgilerini kullan
2. Türkçe olarak yanıtla
3. Doğal ve samimi bir üslup kullan
4. Eğer bilgi bağlamda yoksa, "Bu konuda bilgim yok" de
5. Kısa ve öz yanıtlar ver

YANIT:"""
        
        # Cevap üret
        response = llm.invoke(prompt)
        
        return response.content
    
    except Exception as e:
        return f"❌ Hata: {str(e)}"

         # BÖLÜM 5 web arayüzünü oluşturma
# ===============================
# Ana Uygulama
# ===============================

def main():
    # Başlık
    st.title("🤖 Nil Yağmur Muslu - CV Chatbot")
    st.markdown("### Kişisel Asistan")
    st.markdown("---")
    
    # Ana içerik
    st.info("💬 **Merhaba!** Ben Nil Yağmur Muslu'nun CV asistanıyım. Onun hakkında merak ettiklerinizi sorabilirsiniz.")
    
    # Vector store'u yükle
    try:
        with st.spinner("⏳ Vector store yükleniyor..."):
            vectorstore = load_vector_store()
        st.success("✅ Sistem hazır!")
    except Exception as e:
        st.error(f"❌ Hata: {e}")
        return
    
    # Chat geçmişi
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Chat geçmişini göster
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Kullanıcı girişi
    if prompt := st.chat_input("Sorunuzu buraya yazın..."):
        # Kullanıcı mesajını ekle
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Asistan cevabı
        with st.chat_message("assistant"):
            with st.spinner("💭 Düşünüyorum..."):
                #RAG fonksiyonunu şu an çağırıyoruz
                response = get_response(prompt, vectorstore)
            st.markdown(response)
        
        # Asistan mesajını ekle
        st.session_state.messages.append({"role": "assistant", "content": response})
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>💻 Nil Yağmur Muslu'nun kişisel CV asistanı</p>
        </div>
        """,
        unsafe_allow_html=True
    )

#eğer bu dosya doğrudan çalıştırılırsa main() fonksiyonunu başlat der
if __name__ == "__main__":
    main()
