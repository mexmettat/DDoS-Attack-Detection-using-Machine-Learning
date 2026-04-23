# DDoS Attack Detection using Machine Learning 🛡️

Bu proje, Machine Learning (Makine Öğrenmesi) ve Deep Learning mimarileri kullanarak ağ trafiğindeki **Distributed Denial of Service (DDoS)** saldırılarını tespit etmek amacıyla geliştirilmektedir. Proje kapsamında **CICDDoS2019** ve **CICIDS2017** gibi sektör standardı veri setleri kullanılmaktadır.

---

## 🚀 Başlangıç ve Kurulum Rehberi

Projeyi kendi bilgisayarınızda sorunsuz bir şekilde çalıştırmak için aşağıdaki adımları sırasıyla uygulayınız.

### 1. Python Sanal Ortamı (Virtual Environment) Oluşturma
Makine öğrenmesi projeleri belirli kütüphane sürümlerine ihtiyaç duyar. Kendi bilgisayarınızdaki paketlerin çakışmasını önlemek için projeye özel izole bir "sanal kum havuzu" kurmalıyız.

Terminali (veya VS Code terminalini) projenin ana klasöründe (`DDos_attack`) açın ve şu komutu çalıştırın:
```bash
python -m venv venv
```

### 2. Sanal Ortamı Aktif Etme
Sanal ortamı oluşturduktan sonra terminale bu ortamı kullanmasını söylemeliyiz.

**Windows için:**
```bash
.\venv\Scripts\activate
```
**Mac/Linux için:**
```bash
source venv/bin/activate
```
*(Başarılı olursa terminal satırınızın başında `(venv)` yazısını göreceksiniz.)*

### 3. Gerekli Kütüphanelerin (Dependencies) Yüklenmesi
Projenin ihtiyaç duyduğu tüm Python kütüphanelerini (`pandas`, `scikit-learn`, `matplotlib` vb.) tek seferde yüklemek için paket yöneticisi `pip` kullanıyoruz:

```bash
pip install -r requirements.txt
```

---

## 🏃‍♂️ Projeyi Çalıştırma (Adım Adım Boru Hattı)

Veri işleme ve görselleştirme adımları bir "Pipeline (Boru Hattı)" mantığıyla sırasıyla çalıştırılmalıdır.

### Adım 1: Veri Setlerini Hazırlama
Analiz etmek istediğiniz ham veri setlerini (örneğin `.csv` formatındaki `Monday-Benign.csv` gibi dosyaları) projedeki `data/raw/` klasörünün içerisine atın.

### Adım 2: Veri Ön İşleme (Preprocessing) Robotu
Ham verilerdeki hatalı satırları (NaN, Infinity) temizlemek, gereksiz IP/Port gibi ezbere neden olan özellikleri silmek ve veriyi yapay zekaya uygun hale (0-1 arasına) getirmek için birinci scripti çalıştırın:

```bash
python src/preprocessing.py
```
*(Bu işlem sonucunda temizlenmiş, yapay zeka eğitimine hazır veriler `data/processed/` klasörüne `_cleaned.csv` uzantısıyla kaydedilecektir.)*

### Adım 3: Görselleştirme ve Hasar Tespit Dashboard'u (Opsiyonel ama Önemli)
Temizlenen verilerin durumunu görmek, veri kaybını analiz etmek ve DDoS saldırısının hangi "özellikleri" tetiklediğini anlamak için görselleştirme scriptlerini çalıştırın:

**Boru hattı durumu (Kayıpları vs. görmek için):**
```bash
python src/preprocessing_summary_visual.py
```

**Sınıf Dağılımları (Büyük Resim):**
```bash
python src/master_visualization.py
```

**Matematiksel İlişkiler ve Boxplot Analizi:**
```bash
python src/visualization.py
```
*(Oluşturulan tüm muazzam grafikler ve raporlar `output/visualizations/` klasörüne kaydedilecektir.)*

---

## 🏗️ Proje Mimarisi

* **`data/raw/`**: Sisteminizin işleyeceği ham veri setleri buraya atılır.
* **`data/processed/`**: Gürültüden arındırılmış, normalize edilmiş temiz yapay zeka verileri burada toplanır.
* **`src/`**: Projenin tüm teknik kodlarını (`preprocessing`, `visualization`, ve gelecekte `models` vb.) barındıran kaynak klasörüdür.
* **`output/visualizations/`**: Çizdirilen grafiklerin ve analizlerin çıktı noktasıdır.

>  **Not:** Stratejik detaylar, neden bazı adımları uyguladığımız, sistem mimarimiz, dengesiz veri (Class Imbalance) sorunu çözümleri ve Derin Öğrenme (CNN) yol haritamız için lütfen `ML_Strategy_Docs.md` dosyasını okuyunuz.
