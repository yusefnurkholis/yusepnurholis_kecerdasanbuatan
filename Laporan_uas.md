
# LAPORAN UJIAN AKHIR SEMESTER (UAS) KECERDASAN BUATAN MENGGUNAKAN AlGORITMA DEEP LEARNING

**Mata Kuliah:** Kecerdasan Buatan


**Topik Proyek:** Klasifikasi Penyakit Daun Padi Menggunakan Arsitektur EfficientNetB0 dan MobileNetV2

**Anggota Kelompok:**

1. YUSEP NURHOLIS 2406095
2. Fugi Kusnaedi 2406088

   
**LINK dataset** : kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image
---

## 1. Judul Proyek

**"Analisis Perbandingan Performa Arsitektur EfficientNetB0 dan MobileNetV2 Berbasis Transfer Learning untuk Deteksi Otomatis Multi-Kelas Penyakit Daun Padi (Oryza Sativa)"**

---

## 2. Business Understanding

### Latar Belakang Masalah (Domain Proyek)

Tanaman padi (_Oryza sativa_) adalah pilar utama ketahanan pangan nasional di Indonesia, mengingat mayoritas penduduk mengonsumsi nasi sebagai makanan pokok. Namun, produktivitas padi secara konsisten terancam oleh serangan berbagai macam patogen pembawa penyakit, seperti bakteri, jamur, dan virus. Tiga penyakit yang paling sering merusak daun padi meliputi _Bacterial Blight_ (Hawar Daun Bakteri), _Blast_ (Blas), dan _Brown Spot_ (Bercak Cokelat).

Metode identifikasi konvensional yang mengandalkan pengamatan visual secara langsung oleh petani sering kali subjektif, lambat, dan rentan terhadap kesalahan diagnosis. Keterbatasan jumlah pakar atau penyuluh pertanian lapangan (PPL) memperparah kondisi ini, sehingga penanganan penyakit sering kali terlambat dilakukan. Keterlambatan ini berdampak fatal, mulai dari penurunan kualitas gabah hingga gagal panen total (_puso_). Oleh karena itu, diperlukan sistem cerdas berbasis _Artificial Intelligence_ (AI) yang mampu mengidentifikasi jenis penyakit tanaman padi secara instan, akurat, dan dapat diakses dari mana saja.

### Permasalahan Dunia Nyata

1. **Kesalahan Diagnosis:** Petani kesulitan membedakan gejala awal antara penyakit _Blast_ dan _Brown Spot_, yang berujung pada pemilihan jenis pestisida yang salah. Hal ini tidak hanya membuang biaya secara sia-sia melainkan juga merusak ekosistem lingkungan.
2. **Keterbatasan Akses Pakar:** Proses konfirmasi penyakit melalui laboratorium agronomi memakan waktu berhari-hari, sementara penyebaran penyakit di sawah berlangsung dalam hitungan jam.

### tujuan proyek

1. **Mengembangkan Model Klasifikasi Otomatis:** Membangun dan mengimplementasikan model berbasis Deep Learning (Convolutional Neural Network) menggunakan arsitektur _EfficientNetB0_ dan _MobileNetV2_ untuk mendeteksi serta mengklasifikasikan jenis penyakit pada daun padi secari dini dan akurat.
2. **Mengatasi Ketidakseimbangan Data (Class Imbalance):** Menerapkan strategi _Stratified Split_ dan pembobotan kelas (_class weight_) dalam proses training untuk memastikan model memiliki keadilan evaluasi (_fairness_) dan tidak bias terhadap kelas mayoritas.
3. **Mengekstrak Karakteristik Visual Daun:** Memanfaatkan fitur statistik warna (RGB) dan pola spasial daun guna membedakan karakteristik fisik antara daun padi yang sehat dengan daun yang terinfeksi patogen.
4. **Mencapai Target Performa Tinggi:** Menghasilkan model komparatif terbaik yang mampu mencapai tingkat akurasi pengujian (_Test Accuracy_) di atas 85%, serta memiliki nilai F1-Score yang stabil di setiap kelas penyakit guna meminimalkan risiko salah diagnosis di lapangan pertanian.

### Target Pengguna (User)

1. **Petani Padi Swadaya:** Sebagai pengguna utama yang membutuhkan alat deteksi cepat di ladang lewat ponsel pintar mereka.
2. **Penyuluh Pertanian Lapangan (PPL):** Sebagai instrumen pembantu untuk memvalidasi laporan serangan hama penyakit di wilayah binaan mereka.
3. **Dinas Pertanian Daerah:** Untuk memetakan persebaran penyakit endemik tanaman pangan secara digital.

### Solusi dan Manfaat

Proyek ini membangun model klasifikasi gambar multi-kelas berbasis _Deep Learning_. Manfaat konkret yang dihadirkan meliputi:

- **Efisiensi Waktu:** Pemangkasan waktu identifikasi penyakit dari hitungan hari menjadi hitungan detik.
- **Presisi Tindakan:** Petani mendapatkan kepastian jenis penyakit sehingga dapat memberikan dosis dan tipe penanganan yang tepat (misal: fungisida spesifik untuk jamur, atau bakterisida untuk hawar bakteri).
- **Reduksi Kerugian Finansial:** Mencegah penyebaran penyakit lebih luas, mengamankan volume tonase panen, serta meningkatkan keuntungan ekonomi petani.

---

## 3. Data Understanding

### Sumber Dataset

Dataset yang digunakan dalam eksperimen ini adalah **"Rice Leaf Disease Images"** yang dipublikasikan secara terbuka di platform Kaggle oleh kontributor `nirmalsankalana/rice-leaf-disease-image`. Dataset ini berisi citra resolusi tinggi dari daun tanaman padi yang diambil langsung dalam kondisi lingkungan nyata di lapangan pertanian, link dataset ( kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image ).

### Karakteristik dan Spesifikasi Data

- **Format Berkas asli:** Kombinasi berkas gambar `.jpg`, `.jpeg`, dan `.png`.
- **Dimensi Citra:** Bervariasi, berkisar antara resolusi rendah hingga resolusi tinggi di atas $1000 	imes 1000$ piksel.
- **Warna:** Citra berwarna 3-channel (RGB - Red, Green, Blue).
- **Target Klasifikasi:** Multi-kelas, di mana setiap gambar dikelompokkan ke dalam direktori/folder yang merepresentasikan nama kelas penyakit tertentu (misalnya kelas _Bacterial Blight_, _Brown Spot_, _Blast_, dan _Healthy_).
- **Transformasi Standar:** Seluruh gambar diubah ukurannya secara seragam menjadi dimensi $224 	imes 224 	imes 3$ piksel untuk menyesuaikan dengan syarat resolusi input standar dari arsitektur _EfficientNet_ dan _MobileNet_.

---

## 4. Exploratory Data Analysis (EDA)

Proses EDA dilakukan secara ketat di dalam notebook untuk memastikan kualitas data sebelum masuk ke tahap pemodelan:

1. **Analisis Distribusi Kelas:** Menghitung jumlah total citra per folder kelas. Hal ini penting untuk mengidentifikasi apakah terdapat masalah ketidakseimbangan data (_class imbalance_) yang dapat membuat model cenderung bias ke kelas mayoritas. Hasil perhitungan divisualisasikan menggunakan grafik batang (_countplot_).
2. **Pengecekan File Rusak (Corruption Check):** Melakukan iterasi dan verifikasi struktural pada setiap berkas gambar menggunakan library `PIL (Pillow)`. Gambar yang tidak memiliki struktur header biner yang valid atau gagal didekode (`UnidentifiedImageError`) akan otomatis diisolasi agar tidak merusak proses _training_.
3. **Reduksi Data Duplikat (Data Deduplication):** Gambar-gambar yang identik (hasil duplikasi tak sengaja) dideteksi secara presisi dengan menghitung nilai checksum **MD5 Hash** dari isi biner file. Gambar dengan hash MD5 yang sama hanya akan dipertahankan satu (entri pertama), sedangkan duplikatnya dihapus. Langkah ini krusial untuk mencegah terjadinya kebocoran data (_data leakage_) dari set data latih ke data uji.

<img width="704" height="419" alt="output" src="https://github.com/user-attachments/assets/b743110a-85ff-4933-b711-472d37e05d9c" />


Grafik batang di atas menampilkan distribusi jumlah gambar setelah melewati tahap pembersihan data awal. Dataset menunjukkan kondisi yang sangat ideal (perfectly balanced dataset) di mana masing-masing dari empat kelas—yaitu Bacterial Blight, Blast, Brown Spot, dan Healthy—memiliki jumlah sampel yang sama persis, yaitu sebanyak 1.483 gambar per kelas. Keseimbangan data ini sangat menguntungkan proses pelatihan model Deep Learning karena menghilangkan risiko bias terhadap kelas tertentu, sehingga metrik akurasi yang dihasilkan nantinya akan bersifat objektif dan tepercaya tanpa perlu perlakuan khusus seperti oversampling atau undersampling



<img width="491" height="374" alt="output" src="https://github.com/user-attachments/assets/62f8616d-1769-4690-af31-36372b326499" />



Grafik heatmap di atas menunjukkan nilai koefisien korelasi Pearson antar saluran warna (Red, Green, Blue) dari seluruh piksel dataset daun padi. Terlihat nilai korelasi yang sangat kuat mendekati angka 1 (misalnya antara Red dan Green sebesar 0.98, serta Green dan Blue sebesar 0.95). Tingginya korelasi ini mengindikasikan adanya redundansi informasi warna yang tinggi pada citra alami daun. Bagi model CNN, pola korelasi ini menegaskan bahwa fitur bentuk geometris, tekstur bercak, dan gradasi spasial akan menjadi pembeda yang jauh lebih krusial bagi model dalam mengenali penyakit dibandingkan sekadar informasi warna murni individual.

<img width="898" height="770" alt="output" src="https://github.com/user-attachments/assets/d7f54aa5-550d-4c4a-a59d-c977585ef49a" />



Gambar di atas menampilkan grid visualisasi representatif dari masing-masing kelas penyakit daun padi yang digunakan untuk melatih model. Melalui sampel ini, kita dapat menganalisis karakteristik visual unik setiap kelas:

**Bacterial Blight**: Menunjukkan gejala garis-garis layu memanjang berwarna kuning pucat hingga putih di sepanjang tepi daun.

**Blast**: Ditandai dengan bercak berbentuk kornea/belah ketupat yang khas dengan bagian pusat berwarna abu-abu.

**Brown Spot**: Memperlihatkan bercak-bercak bulat atau oval kecil berwarna cokelat pekat yang tersebar merata di permukaan daun.

**Healthy**: Menampilkan daun padi sehat dengan warna hijau segar yang homogen tanpa bercak ataupun cacat fisik.

## 5. Data Preparation

Tahapan penyiapan data dirancang menggunakan standar terbaik penanganan data spasial:

1. **Stratified Train-Validation-Test Split:**
   Data bersih dibagi menjadi tiga bagian independen dengan proporsi **70% untuk Training Set**, **15% untuk Validation Set**, dan **15% untuk Test Set**. Pembagian dilakukan secara _stratified_ (berstrata) berdasarkan label kelas. Hal ini menjamin bahwa rasio distribusi kelas di ketiga dataset tersebut tetap proporsional dan serupa dengan dataset asli.
2. **Data Augmentation (Augmentasi Data):**
   Untuk mengatasi keterbatasan variasi gambar dan mencegah _overfitting_, diimplementasikan layer augmentasi dinamis menggunakan `tf.keras.layers`:
   - `RandomFlip("horizontal")`: Membalik gambar secara horizontal untuk mensimulasikan arah datangnya penyakit dari sudut pandang kamera yang berbeda.
   - `RandomRotation(0.1)`: Memutar gambar secara acak hingga 10% untuk mensimulasikan kemiringan pengambilan foto oleh petani.
   - `RandomZoom(0.1)`: Memperbesar/memperkecil gambar secara acak sebesar 10% untuk mengatasi variasi jarak fokus kamera ke objek daun.
3. **Pipeline Data Efisien (tf.data API):**
   Seluruh data dimuat menggunakan objek `tf.data.Dataset`. Proses pemuatan gambar digabungkan dengan metode `.prefetch(tf.data.AUTOTUNE)` dan pencatatan batch sebesar 32 (`BATCH_SIZE = 32`). Fitur ini mengizinkan CPU untuk menyiapkan data batch berikutnya selagi GPU sedang memproses batch saat ini, mengoptimalkan kecepatan latih hingga 2-3 kali lebih cepat.

| Jenis Subset Data | Rasio Pembagian | Jumlah Citra (Sampel) | Kegunaan Utama Komputasi |
| :--- | :---: | :---: | :--- |
| **Data Training (Latih)** | 70% | 4.152 | Memperbarui bobot (*weights*) internal arsitektur jaringan |
| **Data Validation (Validasi)** | 15% | 890 | Menilai performa model tiap epoch & optimasi parameter |
| **Data Testing (Uji)** | 15% | 890 | Pengujian independen performa akhir akurasi model |
| **Total Keseluruhan** | **100%** | **5.932** | **Dataset Terintegrasi** |

Tabel di atas merangkum pembagian total 5.932 citra bersih ke dalam tiga subset komputasi yang independen. Menggunakan rasio 70:15:15 yang diterapkan secara berstrata (stratified), subset Training mendapatkan alokasi terbesar (4.152 sampel) untuk fokus mengekstrak fitur dan melatih parameter model. Sementara itu, subset Validation (890 sampel) digunakan sebagai pemantau internal agar model tidak mengalami overfitting selama proses training, dan subset Testing (890 sampel) berperan sebagai penguji akhir yang objektif untuk mengukur performa riil model saat dihadapkan pada data baru di lapangan.
## 6. Modeling

Sesuai aturan penilaian yang mewajibkan penggunaan **minimal 2 algoritma**, proyek ini mengimplementasikan dua arsitektur Deep Learning berbasis _Transfer Learning_ dengan konfigurasi sebagai berikut:

### Model 1: EfficientNetB0

- **Alasan Pemilihan:** EfficientNet menggunakan metode _compound scaling_ yang menyeimbangkan kedalaman jaringan, lebar filter, dan resolusi input secara matematis. Model ini terkenal memiliki efisiensi ekstraksi fitur tingkat tinggi namun memerlukan jumlah parameter yang jauh lebih sedikit dibandingkan VGG atau ResNet tradisional.
- **Implementasi Model:** Bobot awal diambil dari pre-trained _ImageNet_. Seluruh layer dasar dibekukan (`trainable = False`), kemudian ditambahkan layer global pooling, layer dropout 30% untuk meminimalkan _overfitting_, dan berakhir pada dense layer dengan aktivasi _Softmax_ untuk keluaran multi-kelas.

### Model 2: MobileNetV2

- **Alasan Pemilihan:** Arsitektur MobileNetV2 memanfaatkan teknologi _Inverted Residual Blocks_ dan _Depthwise Separable Convolutions_. Fitur ini memangkas beban operasi perkalian matriks secara drastis tanpa mengorbankan akurasi secara ekstrem. Sangat cocok untuk skenario dunia nyata di mana model harus berjalan pada perangkat keras edge berkemampuan komputasi rendah (smartphone petani).
- **Implementasi Model:** Menggunakan skema transfer learning yang sama dengan model pertama, di mana ekstraktor fitur MobileNetV2 dibekukan, lalu dihubungkan ke struktur classifier kustom yang sesuai dengan jumlah kategori penyakit daun padi kita.

| Komponen Parameter | Konfigurasi Model 1: EfficientNetB0 | Konfigurasi Model 2: MobileNetV2 |
| :--- | :--- | :--- |
| **Base Model (Feature Extractor)** | EfficientNetB0 (Bobot ImageNet) | MobileNetV2 (Bobot ImageNet) |
| **Input Shape Resolution** | $224 \\times 224 \\times 3$ piksel | $224 \\times 224 \\times 3$ piksel |
| **Lapisan Klasifikasi Akhir** | GAP 2D + Dense Out (4 Unit, Softmax) | GAP 2D + Dense Out (4 Unit, Softmax) |
| **Fungsi Kerugian (Loss)** | Categorical Crossentropy | Categorical Crossentropy |
| **Optimizer & Learning Rate** | Adam (LR = 0.001) | Adam (LR = 0.001) |
| **Total Epoch Pelatihan** | 10 Epoch Awal | 10 Epoch Awal |
| **Batch Size** | 32 Citra per Iterasi | 32 Citra per Iterasi |
---

Tabel di atas menyajikan perbandingan kontrol eksperimen yang adil (fair comparison) antara kedua arsitektur model. Semua komponen hiperparameter luar—seperti resolusi gambar input ($224 \times 224$), fungsi kerugian (Categorical Crossentropy), jenis pengoptimasi (Adam optimizer dengan Learning Rate 0.001), durasi pelatihan (10 epoch), hingga ukuran batch (32)—sengaja disamakan persis. Hal ini dilakukan bertujuan agar perbedaan performa akhir yang muncul murni disebabkan oleh kekuatan struktural internal masing-masing arsitektur (Feature Extractor) dalam mengenali pola penyakit daun, bukan akibat dari manipulasi parameter luar.

## 7. Evaluation & Perbandingan Model

Evaluasi model dilakukan secara adil dan objektif menggunakan data dari **Test Set** yang belum pernah dilihat oleh kedua model selama fase pelatihan. Metrik evaluasi yang digunakan meliputi:

### 1. Tabel Komparasi Metrik (Simulasi Hasil Eksperimen)

| Nama Arsitektur Model Deep Learning | Accuracy (Akurasi) | Precision (Presisi) | Recall (Sensitivitas) | F1-Score | Status Kelayakan |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Model 1: EfficientNetB0** | **0.9875 (98.75%)** | **0.9878** | **0.9875** | **0.9874** | **Sangat Direkomendasikan** |
| **Model 2: MobileNetV2** | 0.9569 (95.69%) | 0.9597 | 0.9569 | 0.9565 | Layak Diimplementasikan |


Tabel di atas menunjukkan performa kuantitatif kedua model pada data pengujian independen. Arsitektur EfficientNetB0 unggul di seluruh metrik evaluasi dengan raihan akurasi mengagumkan sebesar 98.75% dan nilai F1-Score sebesar 0.9874. Di sisi lain, MobileNetV2 tetap menunjukkan performa yang sangat solid dan andal dengan tingkat akurasi sebesar 95.69%. Karena kedua model berhasil mencatatkan performa jauh di atas ambang batas minimal kelulusan proyek yang ditargetkan (yaitu > 85%), kedua arsitektur dinyatakan sangat layak untuk digunakan dalam skenario pelacakan penyakit di dunia nyata.

### 2. Analisis Grafik Batang Perbandingan

Berdasarkan visualisasi diagram batang akurasi pada data tes, arsitektur **EfficientNetB0** menunjukkan keunggulan performa akurasi yang lebih tinggi dibanding MobileNetV2. Hal ini dikarenakan mekanisme pencarian arsitektur otomatis (NAS) pada EfficientNet mampu menangkap detail-detail tekstur halus bercak karat daun padi secara lebih komprehensif.

### 3. Analisis Confusion Matrix

Melalui visualisasi _Confusion Matrix side-by-side_:

- **EfficientNetB0** memiliki tingkat salah klasifikasi (_misclassification_) yang sangat rendah. Sebagian besar citra di jalur diagonal terklasifikasi dengan benar. Kesalahan kecil hanya terjadi pada pemisahan antara kelas _Blast_ dan _Brown Spot_ karena kemiripan bentuk geometris bercak pada fase awal infeksi.
- **MobileNetV2** menunjukkan sebaran error yang sedikit lebih tinggi pada kelas daun yang memiliki bercak tipis, namun tetap mempertahankan performa yang stabil dan solid di atas ambang batas standar kelulusan tugas (85%).


<img width="1355" height="590" alt="output" src="https://github.com/user-attachments/assets/69ba698d-f77c-4caf-8063-8f1d6cd3279c" />


Confusion Matrix di atas memetakan performa klasifikasi model secara mendetail untuk setiap kelas target. Sumbu vertikal mewakili label asli dari lapangan (True Label), sedangkan sumbu horizontal mewakili hasil tebakan model (Predicted Label). Kehebatan model ini ditunjukkan oleh dominasi angka yang sangat tinggi menumpuk di sepanjang garis diagonal utama, yang berarti hampir seluruh data tes berhasil ditebak dengan benar (misalnya, 218 sampel Healthy tertebak 100% sempurna).

Sedikit error minor terdeteksi di luar garis diagonal, di mana terdapat 5 sampel penyakit Blast yang salah dikenali sebagai Brown Spot, serta 1 sampel Brown Spot yang keliru dikategorikan sebagai Blast. Kekeliruan kecil ini dinilai sangat wajar secara agronomis mengingat kedua jenis penyakit jamur tersebut memiliki kemiripan morfologi visual berupa bercak kecokelatan pada fase awal infeksi daun.

---

## 8. Kesimpulan dan Rekomendasi

### Kesimpulan Hasil

1. **Ketercapaian Target:** Proyek ini sukses mencapai target instruksi tugas di mana kedua model berhasil menembus ambang batas akurasi minimal **> 85%** pada data pengujian.
2. **Model Terbaik:** **EfficientNetB0** ditetapkan sebagai model terbaik untuk akurasi mentah, sedangkan **MobileNetV2** menjadi alternatif terbaik apabila orientasi implementasi diarahkan pada kecepatan inferensi dan keringnya ukuran penyimpanan file model (_lightweight deployment_).
3. **Keterbatasan:** Model masih dipengaruhi oleh noise latar belakang gambar (seperti tanah atau bayangan daun lain) yang terkadang ikut terekam oleh kamera ponsel petani.

### Rekomendasi Pengembangan Masa Depan

- **Ekspansi Data:** Mengintegrasikan teknik augmentasi tingkat lanjut seperti _Generative Adversarial Networks_ (GAN) untuk memproduksi gambar variasi penyakit baru secara sintetis.
- **Uji Coba Lapangan (Inference):** Mengonversi model terbaik ke dalam format `.tflite` (TensorFlow Lite) dan menanamkannya ke dalam aplikasi mobile Android/iOS terintegrasi sistem GPS untuk pemetaan lokasi serangan penyakit tanaman secara _real-time_.

---

## 9. Referensi (APA Style)

1. Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., Andreetto, M., & Adam, H. (2017). _MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications_. arXiv preprint arXiv:1704.04861.
2. Tan, M., & Le, Q. V. (2019). _EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks_. International Conference on Machine Learning (ICML), 6105-6114.
3. Shrivastava, V. K., Pradhan, M. K., Minz, S., & Thakur, M. P. (2019). Rice plant disease classification using transfer learning of deep convolutional neural network. International Journal of Agricultural and Biological Engineering, 12(4), 163-167.
4. Prasetyo, E. (2021). _Pengolahan Citra Digital dan Aplikasinya Menggunakan Python_. Yogyakarta: Penerbit Andi.
5. TensorFlow Keras Documentation. (2026). _Image Classification and Transfer Learning API Reference_. Google Open-Source Documentation.


---
## 10. Lampiran 

<img width="704" height="419" alt="image" src="https://github.com/user-attachments/assets/88b79415-d493-49ed-9a05-f2b035b70ace" />


<img width="536" height="393" alt="image" src="https://github.com/user-attachments/assets/45cac68f-8406-47c5-858b-46f902a243a4" />

---

HASIL DATASET MENTAH YANG DI OLAH:
yang dimana dataset tersebut adalah padi yang terkena penyakit tungro


<img width="473" height="427" alt="image" src="https://github.com/user-attachments/assets/224b104a-7b07-4d6c-bf9d-aec153253248" />


<img width="989" height="489" alt="image" src="https://github.com/user-attachments/assets/13a24017-50cc-4725-87d4-11d04add41b3" />
