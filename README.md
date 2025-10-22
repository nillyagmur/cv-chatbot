# cv-chatbot
Akbank GenAI Bootcamp için RAG tabanlı CV Chatbot Projesi

Bu proje, Akbank GenAI Bootcamp: Yeni Nesil Proje Kampı  kapsamında geliştirilmiştir. Proje, RAG (Retrieval-Augmented Generation)  mimarisini kullanarak benim CV'm hakkında sorulan sorulara yanıt veren interaktif bir chatbot uygulamasını içermektedir.


Proje Sahibi: Nil Yağmur Muslu

🚀 Projenin Canlı Demosu
Projenin çalışan web arayüzüne (web arayüzü)  aşağıdaki linkten erişebilirsiniz:

➡️ UYGULAMA LİNKİ: Network URL: http://10.108.49.56:8501

🎯 1. Projenin Amacı
Bu projenin temel amacı, belirli bir bağlam (Nil Yağmur Muslu'nun CV'si) üzerinde uzmanlaşmış, RAG tabanlı bir chatbot geliştirmektir. Chatbot, kullanıcıların CV hakkındaki sorularına, yalnızca sağlanan metin (data.txt) içerisindeki bilgilere dayanarak doğru ve tutarlı yanıtlar vermek üzere tasarlanmıştır.

📦 2. Veri Seti Hakkında Bilgi
Veri seti olarak, "Nil Yağmur Muslu" adına ait CV bilgileri kullanılmıştır. Bu bilgiler, data.txt adlı bir metin dosyasına ham metin olarak işlenmiştir. Veri seti, eğitim, iş deneyimleri, yetenekler ve projeler gibi standart CV bölümlerini içermektedir.


(Not: Bootcamp proje gereksinimleri  doğrultusunda, hassas veri içerebileceğinden data.txt dosyası bu depoya (repository) eklenmemiştir.)

🛠️ 3. Çözüm Mimarisi ve Kullanılan Teknolojiler
Proje, LangChain çatısı altında bir RAG  mimarisi kullanılarak geliştirilmiştir. Çözüm mimarisi aşağıdaki bileşenlerden oluşmaktadır:


Web Arayüzü: Streamlit

Kullanıcı ile interaktif bir sohbet arayüzü oluşturmak için kullanıldı.


LLM (Generation Model): OpenAI (gpt-3.5-turbo) 

Kullanıcı sorusu ve FAISS'ten alınan ilgili bağlam (context), OpenAI modeline bir prompt ile gönderilerek nihai cevap üretildi.


Embedding Modeli: Hugging Face (sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) 

Metin parçalarını (chunks) yüksek boyutlu vektörlere dönüştürmek için kullanıldı.


Vektör Veritabanı: FAISS (Facebook AI Similarity Search) 

Elde edilen vektörler, sunucu üzerinde hızlı ve verimli bir arama (similarity search) yapabilmek için FAISS index'i olarak saklandı.


Framework: LangChain 

Veri yükleme, parçalama (RecursiveCharacterTextSplitter), embedding modelini çağırma, FAISS veritabanını yönetme ve LLM'e prompt gönderme gibi tüm RAG akışını (pipeline) yönetmek için kullanıldı.

Dağıtım (Deployment): Hugging Face Spaces

Uygulamanın canlı bir web linki  olarak sunulması için kullanıldı.

📖 4. Çalışma Kılavuzu
Projenin yerel (local) bir bilgisayarda veya sunucuda çalıştırılması için aşağıdaki adımlar izlenmelidir.

1. Depoyu Klonlayın:

Bash

git clone https://github.com/[KullaniciAdiniz]/[RepoAdiniz].git
cd [RepoAdiniz]
2. Sanal Ortam (Virtual Env) Oluşturun ve Aktive Edin: 

Bash

python -m venv venv
venv\Scripts\activate  # Windows için
# source venv/bin/activate  # MacOS/Linux için
3. Gerekli Paketleri Yükleyin: Tüm bağımlılıklar requirements.txt dosyasında listelenmiştir.

Bash

pip install -r requirements.txt
4. API Anahtarını Ayarlayın: Bu proje OpenAI API anahtarı gerektirir. Anahtarınızı .streamlit/secrets.toml dosyası oluşturarak veya bir ortam değişkeni (environment variable) olarak ayarlamanız gerekmektedir.

5. Uygulamayı Çalıştırın: 

Bash

streamlit run app.py
📊 5. Elde Edilen Sonuçlar
Geliştirilen chatbot, sağlanan data.txt bağlamına sadık kalarak, CV ile ilgili sorulan sorulara başarılı bir şekilde yanıt verebilmektedir. Geliştirme sürecinde pydantic, faiss-cpu, openai ve langchain kütüphaneleri arasında yaşanan sürüm uyumsuzlukları (Dependency Hell), requirements.txt dosyasında yapılan spesifik versiyon ayarlamaları (pydantic<2, langchain==0.1.0 vb.) ile çözülmüştür.

Uygulama, pickle uyumsuzluklarını aşmak için faiss_index.pkl dosyasını sunucuya yüklemek yerine, ilk çalıştırmada sunucunun kendisinde sıfırdan oluşturacak (@st.cache_resource kullanarak) şekilde optimize edilmiştir.

(Bu satırın üstüne kadar olan kısmı kopyalayın)

Projemin Eksik Bir Kısmı Var mı?
Proje dosyanızı (PDF)  ve şu ana kadar yaptıklarımızı (çalışan Hugging Face linki, yorum satırlı app.py, çalışan requirements.txt) karşılaştırdığımda, projeniz PDF'te istenen tüm teknik ve fonksiyonel gereksinimleri karşılıyor görünmektedir.

RAG Tabanlı Chatbot: Tamamlandı. (PDF'te istenen RAG , LangChain , FAISS , OpenAI  kullandık).




Web Arayüzü: Tamamlandı. (PDF'te istenen "web arayüzü"  için Streamlit kullandık ve Hugging Face'e deploy ettik).

Geliştirme Ortamı: Şu an yapıyoruz. (GitHub'a yüklüyorsunuz ve app.py dosyamıza yorum satırları  ekledik).



Belgelendirme: Yukarıdaki README.md taslağı, PDF'te istenen tüm başlıkları (Amaç , Veri Seti , Mimari , Çalışma Kılavuzu , Web Linki ) eksiksiz olarak kapsamaktadır.
