
-------------------------------------------------------------------------------

# Proje Stratejisi ve Makine Öğrenmesi (ML) Yol Haritası 🧠🚀

Bu belge, "Neden bu adımları izliyoruz?" ve "Gelecekte modelleri nasıl eğiteceğiz?" sorularının detaylı mühendislik ve veri bilimi perspektifinden bir açıklamasıdır. Projenin mantığını ve gelecek planlarını anlatan bir mimari dokümantasyon görevi görür. Projemizin **veri ön işleme, görselleştirme araçları ve gelecek modelleme (Makine Öğrenmesi & Derin Öğrenme)** yol haritasını baştan sona özetler.

-------------------------------------------------------------------------------

## 1. Faz 1: Neden Önce "Boru Hattı (Pipeline)" Kurduk?

Makine öğrenmesinde altın bir kural vardır: **"Garbage in, Garbage out" (Çöp girerse, çöp çıkar).**
Yapay zeka modellerimiz ham siber güvenlik verilerini kendi başlarına anlayamazlar. Ağ trafiği inanılmaz derecede kirli veriler barındırır. Boru hattımızın (`src/preprocessing.py`) her bir parçası şu kritik görevleri yerine getirir:

* **Infinity ve NaN Temizliği:** Verilerdeki sonsuz bölme hataları veya kayıp (NaN) paketler, modelin matematik yaparken çökmesine neden olur. Bunları otomatik tespit edip eledik.

* **Ezber Önleyici "Feature Selection":** Kaynak IP adresi, Port Numarası veya Timestamp (Zaman damgası) gibi sütunlar DDoS'u belirleyen kalıplar değildir, sadece paketin kimliğidir. Eğer bunları modele verirsek, model **"DDoS saldırısının davranışını"** öğrenmek yerine, **"X numaralı IP kötüdür"** demeyi ezberler (Overfitting). Modeli paketin davranışsal istatistiklerine odaklanmaya zorladık.

* **MinMaxScaler (0 - 1 Ölçekleme):** Modeller, milyarlık devasa sayılar ile 0.5 gibi küçük sayıları yan yana gördüğünde aklı karışır. Tüm özellikleri aynı matematiksel evrene (0 ile 1 arasına) sıkıştırarak modelin daha hızlı ve kararlı eğitilmesini sağladık.

----------------------------------------------------------------------------------


### Neden Devasa Tek Bir CSV Oluşturmadık? (Batch Processing)
Veri setlerinin tamamı (2017 ve 2019) birleştirildiğinde devasa boyutlara ulaşır. Tüm dosyaları tek bir excel tablosunda kopyala/yapıştır yapıp RAM'e yüklemeye çalışmak doğrudan **Out of Memory (Bellek Çökmesi)** hatasına neden olur. Bunun yerine her bir dosyayı sırayla tek tek ele alan (döngülü) bir **Batch Processing** sistemi kurduk. Bu bize müthiş bir donanım optimizasyonu, esneklik ve hata ayıklama (debug) hızı kazandırdı. Ayrıca evrensel okunabilirlik adına çıktıları `.parquet` yerine standart `_cleaned.csv` formatında aldık.

----------------------------------------------------------------------------------


## 2. Görselleştirme Stratejimiz (3 Farklı Script Ne İşe Yarıyor?)

Projede veriyi körü körüne algoritmaya atmak yerine, süreci tamamen denetlenebilir kılan şeffaf bir görselleştirme katmanı (Visualization Layer) inşa ettik. Her scriptin farklı bir misyonu vardır:

1. **`preprocessing_summary_visual.py` (Boru Hattı Hasar Testi):** Bu script verilerin analizini değil, **sistemin kendi analizini** yapar. Temizleme sırasında ne kadar satır/kolon çöpe atıldı, sistem veri kaybına uğradı mı gibi soruları 4-panelli bir "Dashboard" ile raporlar.

2. **`master_visualization.py` (Kuş Bakışı Karşılaştırma):** Tüm veri setlerinin (yüzlerce milyon paketin) sadece durumunu ("Normal" veya "Attack") inanılmaz bir hızla okuyup, dev tek bir bar grafiğinde sınıf dağılımlarını kıyaslar.

3. **`visualization.py` (Detaylı EDA / Veri Bilimi Analizi):** Doğrudan yapay zeka modelini neyle besleyeceğimizi bulmak için tasarlanmıştır. Verilerin "Korelasyon Isı Haritalarını" çıkartıp, DDoS'u tespit eden matematiksel olarak **En Şüpheli 10 Özelliği** bulur ve bunların Kutu Grafiklerini (Boxplot) çizer.


----------------------------------------------------------------------------------------------


## 3. Gelecek Planı: Modelleri Nasıl Eğiteceğiz? (Veri Bölme Stratejisi)

Boru hattımız her şeyi temizledi ve AI için hazır hale getirdi. Biz bir sınıflandırma (Saldırı mı? Normal mi?) problemi çözüyoruz. Projemizin yaklaşan bölümünde verileri test ve eğitim için nasıl böleceğimiz planlanmıştır:

### 📚 Train (Eğitim) Seti: [CICIDS2017 Dosyalarının %80'i]
`data/processed/` klasöründeki temizlenmiş 2017 veri setleri eğitim amaçlı tek havuza yansıtılacak. Random Forest/XGBoost gibi algoritmalar bu verilere bakarak "Hangi özellik artarsa bu bir DDoS saldırısıdır?" sorusunun istatistiksel ağacını çıkartacak.

### 🤔 Validation (Doğrulama/Ara Sınav): [CICIDS2017 Dosyalarının %20'si] 
Model çalışırken arkada gizlenen %20 ile ona testler yaptıracağız. Model eğitimde %99 alıp bu Ara Sınav'da %60 alıyorsa, ezber yaptığını anlayıp (Overfitting) Hyperparameter Tuning ile ayarları dizginleyeceğiz.

### 📝 Test (Gerçek Dünya Yeteneği): [TÜM CICDDoS2019 DOSYALARI]
Model 2017 yapısını öğrendikten sonra, asla eğitimde kullanılmayan **2019 saldırı verilerini (.parquet)** vereceğiz. Model daha karmaşık, hiç görmediği Zero-day tarzı 2019 taktiklerini bu aşamada doğru bilebilirse müthiş bir **Generalization (Genelleştirme)** düzeyine geldik demektir.


-------------------------------------------------------------------------------------------------



## 4. Class Imbalance (Veri Dengesizliği) Nasıl Çözülecek?

Gelecekte model eğitim modülünü (`src/models/train.py` vb.) kurguladığımızda karşılaşacağımız son engel **Veri Dengesizliği**'dir: Diyelim ki veride **1 Milyon Normal**, **5 Bin Saldırı** trafiği var. Tembel AI, sürekli "Normal" yanıtı vererek yüksek başarı oranı gösterebilir ama bir siber güvenlik ürünü için bu intihardır. Çözümlerimiz:

1. **`class_weight='balanced'` Matrisi:** Az olan DDoS paketini yakalayamamanın katsayısını, Normal paketleri bulmaktan 100 kat daha "ağır cezalı" (Penalization) hale getireceğiz. Yapay Zeka ceza yememek için mecburen o azınlık DDoS'ları da bulmayı öğrenecek.

2. **SMOTE (Sentetik Azınlık Klonlama) Algoritması:** Eğer ceza yetmezse, Python kodlarımız sayesinde azınlıkta olan saldırı paketlerinin istatistiksel olarak tıpatıp "benzeri olan sahte ikizlerini" üreteceğiz. Veri havuzunu suni olarak 50-50'ye getirip modeli öyle besleyeceğiz.

*(Tüm bu süreçler projemizin sonraki aşaması olan "Adım 2: ML Models" klasöründe kodlanacaktır.)*



-------------------------------------------------------------------------------------------------

## 5. Sistem Mimarisi: Neden Tüm Ham Verileri Tek Bir Devasa .CSV'de Birleştirmedik? (Sektör Standardı)

Projemizde bilerek ve oldukça kararlı bir şekilde devasa bir "Tek CSV" dosyası yaratmaktan kaçındık. Bunun yerine yüzlerce dosyanın birbirine karışmadan ayrı ayrı işlenmesini (Loop) sağlayan **"Batch Processing (Toplu İşleme Boru Hattı)"** mimarisi kurduk. Şirketlerde ve siber güvenlik araştırmalarında kabul gören en mantıklı ve profesyonel yöntem budur. İşte o 3 hayati mühendislik sebebi:

1. **RAM Çökmesi (Out of Memory - OOM):**
CICIDS2017 ve 2019 ham verilerinin toplamı bilgisayarda onlarca GB yer tutar. Klasik (Junior) seviye yaklaşımlarda tüm dosyalar önceden tek bir Excel tablosunda `(All_Data_Merged.csv)` kopyala/yapıştır yapılarak birleştirilir. Dosya boyutu 30-50 GB seviyelerine ulaşır. Ekipteki herhangi biri projenin ilk kodunu çalıştırıp veri ön işlemeye (Preprocessing) kalkıştığında, sistem bu devasa veriyi doğrudan bilgisayarın RAM (Bellek) donanımına bindirmeye çalışır. Sistem kilitlenir veya direkt *MemoryError* fırlatarak çöker.
**Bizim Çözümümüz:** Döngülü (Loop) boru hattımız ise 500 GB dosya bile olsa hepsini sıraya dizer, sadece **1 dosyayı eline alır**, ufak bir RAM harcayarak saniyeler içinde temizler, kaydeder ve sonraki dosyaya geçer. Bilgisayar gücünden inanılmaz bir tasarruf edildi.

2. **Esneklik ve Analiz (Traceability):**
Araştırma sunumunuzda hocalarınız size *"Sadece Cuma (Friday) günü gelen PortScan saldırısındaki başarı oranınız nedir?"* veya *"Sadece Syn saldırısı üzerinde analiz çiz"* diye sorabilir. Bütün veriyi baştan birleştirirseniz Cuma gününü bulmak çileye döner. Bizim Data Pipeline (Boru Hattımız) her dosyayı tek çalıştırdığı için dosyalarımız `Friday-PortScan_cleaned.csv` veya `Syn_cleaned.csv` adıyla tıkır tıkır listelenir. Modelleme aşamasında ne isterseniz çeker eğitim havuzunuza koyarsınız.

3. **Hata Giderme ve Zaman Kazancı (Hız):**
Veriyi birleştirirseniz ve temizlemede bir hata yaptığınızı sonradan fark ederseniz (Örneğin gereksiz bir kimlik satırını silmeyi unuttunuz), o devasa büyüklükteki 50 GB'lık dosyayı baştan aşağı silip, bir daha temizleyip saatlerce beklemek zorunda kalırsınız. Bizim dinamik modüler sistemimizde kod çok seri akar, hatalı gün saniyeler içinde silinir ve tek başına düzeltilir!


*(Bu harika mimari öngörü ve tasarım sayesinde, hiçbir ekibin veya standart bilgisayar donanımının bilgisayarlarını iflas ettirmeden Big Data standartlarında devasa modeller inşa etmiş olduk.)*

-------------------------------------------------------------------------------------------------

## 6. Neden CICIDS2017 Eğitim (Train) İçin Seçildi ve Neden .CSV Çıktısı Alıyoruz?

Projeyi arkadaşlarınızla (veya hocalarınızla) incelerken akla gelebilecek en vizyoner sorulardan biri şudur: *"Neden 2019'da eğitip 2017'yi çözdürmüyoruz da tam tersini yapıyoruz? Ve veri okurken Parquet/CSV ayırt etmezken, veriyi temizledikten sonra neden Israrla .CSV olarak çıktı veriyoruz?"*

### A) Zaman Çizelgesi ve "Siber Evrim" Mantığı (Neden 2017?)
Makine öğrenmesi ve siber güvenlikte "Gelişim (Evrim)" ileriye doğru akar. 2017 yılındaki DDoS saldırıları (SYN Flood, HTTP GET vb.) modern saldırıların temel ağ karakteridir (Atasıdır). 2019 yılındaki saldırılar ise (Reflection, LDAP, MSSQL) bu temelleri kullanan daha "karmaşık" türevlerdir.
Siz yapay zekaya önce alfabeyi (2017 - Temel ağ karakterlerini) öğretirsiniz, sonra ondan okunması zor bir makaleyi (2019 - Karmaşık türevleri) çözmesini beklersiniz. Zaten "Sıfırıncı Gün (Zero-day)" felsefesi tam olarak budur: *Eskiyi bilerek, henüz hiç görmediğin modern bir taktiği sezebilmek.* Zamanın gerisine giderek eğitim yapmak gerçek dünya simülasyonuna (Ağ mantığına) terstir.

### B) Çıktı Olarak Neden Israrla .CSV Formatı?
Fark ettiyseniz kodumuz hem .CSV hem de .Parquet (.parquet) okuyabiliyor ama işini bitirince temizlenmiş (Processed) veriyi evrensel olarak `_cleaned.csv` formatında dışarı veriyor. Bunun profesyonel 3 sebebi var:
1. **Şeffaflık ve Debug (Hata Ayıklama):** Veri temizlediğiniz zaman projedeki bir mühendis "Acaba NaN'ler gerçekten silinmiş mi?" diye 1-2 satıra çıplak gözle bakmak isteyebilir. CSV dosyasına VS Code içinde çift tıkladığınız an normal bir metinmiş gibi açılır ve her şeyi okursunuz. Oysa Parquet (Binary) dosyasına tıklarsanız ekranda sadece bozuk karakterler (Makine kodu) görürsünüz.
2. **Evrensel Standart:** Projeyi GitHub'da yayınladığınızda o veriyi alıp Java, R, C++ veya JavaScript gibi başka dillerde analiz edecek bilim insanları çıkabilir. CSV veri biliminin evrensel konuşma dilidir. Parquet ise okumak için özel ağır kütüphaneler (Örn: pyarrow) gerektirir.
3. **Sorunsuz Modüler Bağlantı:** Yazdığımız Visualizasyon (Grafik Çizdirici) ve Makine Öğrenmesi kodlarımız standart Text (String) tabanlı analizler yapıyor. Sektörde bir yapay zeka taslağı (Prototip) %100 kusursuz hale gelene kadar daima CSV kullanılır, çünkü nerede verinin bozulduğunu anında bulmanızı sağlar. Sisteminiz kusursuz olduğunda 1 satır kodla çıktıları Parquet'e geçirmek çocuk oyuncağıdır, ama mimari CSV üzerine atılır.

-------------------------------------------------------------------------------------------------

### FAZ 2 -> Klasik Makine Öğrenmesi (ML) Modellerinin Eğitimi


Pipeline'dan çıkan tertemiz verilerle klasik Machine Learning aşamasına (Random Forest ve XGBoost vb.) geçeceğiz. Bu aşamadaki stratejilerimiz:

### 🚀 Zaman Çizelgesi ve "Siber Evrim" Yaklaşımı (Neden 2017'de Eğitiyoruz?)
Yapay zekaya öncelikle siber saldırıların "Alfabesini" yani temel ağ karakteristiğini (2017 CICIDS) öğreteceğiz. Model temel yapıyı öğrendikten sonra, hayatında hiç görmediği daha karmaşık ve evrimleşmiş "Zero-day" tarzı modern taktikleri barındıran **2019 CICDDoS test setini** vererek (Generalization) yeteneğini ölçeceğiz. Geçmişteki veriden öğrenip gelecekteki saldırıyı tahmin etmek, siber güvenlik vizyonunun ta kendisidir.

### ⚖️ Sınıf Dengesizliği (Class Imbalance) Çözümleri
Verilerimizde %90 Normal vs %10 Saldırı paketi gibi büyük dengesizlikler olacak. Modelin tembelleşmesini (Sürekli "Bu normaldir" diyerek %90 başarı puanı almasını) engellemek için:
- **`class_weight='balanced'` Matrisi:** DDoS paketini kaçırmanın veya yanlış bilmenin matematiksel "Ceza" katsayısını çok yüksek tutacağız.

- **SMOTE (Sentetik Veri Üretimi):** Gerekirse azınlık sınıfı olan saldırı paketlerini istatistiksel klonlama yöntemiyle sahte-ikizler yaratarak 50-50 dengesine kavuşturacağız.

-----------------------------------------------------------------------------------------------

### FAZ 3 -> Derin Öğrenme (CNN) Entegrasyonu


Klasik ağaç bazlı modellerden (XGBoost/RF) sonra performansın sınırlarını zorlamak için projeye **Deep Learning (Derin Öğrenme)** kurgusunu ekleyeceğiz:

* Tablolu (Tabular) istatistikleri doğrudan bir **1D-CNN (1-Boyutlu Evrişimli Sinir Ağı)** modeline besleyerek sadece değişkenlerin büyüklüğünü değil, bağlantının gizli soyut örüntülerini de ağlar katmanında süzeceğiz. 
* İleri safhada, ağ paket özelliklerini yapısal matrislere (mini-görsellere) dönüştürerek **2D-CNN** veya veri akışındaki "zaman sıralamasını" yakalamak için **LSTM/GRU (Tekrarlayan Sinir Ağları)** test edilecektir. Derin öğrenme, geleneksel modellerin gözden kaçırdığı mikro saldırı sekanslarını yakalamakta son vuruşumuz olacaktır.


-----------------------------------------------------------------------------------------------


### FAZ 4 -> Model Deployment ve Web Arayüzü (UI) Entegrasyonu

Modellerimiz (ML & DL) testleri geçip yüksek başarım oranlarına ulaştığında, projenin son adımı bu modelleri gerçek dünya senaryosuna uygun bir şekilde **canlıya almak (Deployment)** olacaktır:

* **Gerçek Zamanlı Web Uygulaması / Dashboard:** Eğitilen model bir API (Flask/FastAPI vb.) aracılığıyla dışarı aktarılacak ve modern bir Web arayüzüne (UI) bağlanacaktır.
* **Görsel Analiz ve İzleme:** Sisteme giren ağ trafiği arayüz üzerinden taratılarak şüpheli DDoS tespitleri ekranda görselleştirilecektir. Terminal çıktısının ötesine geçilerek, siber güvenlik uzmanlarının veya son kullanıcıların süreci kolayca gözlemleyebileceği interaktif ve estetik bir izleme paneli sunulacaktır.


*(Şu anda Faz 1 tamamen stabil çalışmaktadır. Gelecek adımlar, klasör mimarisine Faz 2, Faz 3 ve Faz 4 yapılarının kodlanmasıyla devam edecektir.)*