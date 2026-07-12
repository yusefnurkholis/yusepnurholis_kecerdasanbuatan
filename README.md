# Klasifikasi Penyakit Daun Padi Menggunakan Deep Learning

Repositori ini dibuat untuk memenuhi tugas **Ujian Akhir Semester (UAS) Mata Kuliah Kecerdasan Buatan**. Proyek ini berfokus pada implementasi teknologi _Computer Vision_ menggunakan pendekatan _Deep Learning_ untuk mendeteksi dan mengklasifikasikan berbagai jenis penyakit pada daun tanaman padi (_Oryza sativa_).

## Arsitektur Repositori

- **`README.md`** : Dokumen panduan utama repositori yang berisi deskripsi singkat proyek, cara instalasi, dan ringkasan eksekusi.
- **`Laporan_uas.md`** : Laporan komprehensif tertulis yang mencakup analisis dari Point 1 sampai Point 10 secara mendalam.
- **`UAS_KecerdasanBuatan_model.ipynb`** : Notebook Jupyter utama yang memuat keseluruhan kode eksperimen, mulai dari pengunduhan dataset, EDA, pembersihan data duplikat, proses _training_ dua model, hingga evaluasi matriks.

---

## Ringkasan Proyek & Model Pembanding

Proyek ini membandingkan dua arsitektur _Convolutional Neural Network_ (CNN) mutakhir berbasis _Transfer Learning_ (menggunakan bobot pre-trained dari ImageNet):

1. **EfficientNetB0**: Dipilih karena efisiensinya yang luar biasa dalam menyeimbangkan parameter jaringan (kedalaman, lebar, resolusi) sehingga menghasilkan akurasi tinggi dengan komputasi yang relatif ringan.
2. **MobileNetV2**: Dipilih sebagai algoritma pembanding karena arsitekturnya yang ultra-ringan dengan konsep _Inverted Residuals_ dan _Linear Bottlenecks_, menjadikannya model yang sangat ideal untuk dideploy ke perangkat mobile petani di lapangan.

### Dataset

Dataset yang digunakan berasal dari platform Kaggle yaitu **Rice Leaf Disease Images** (`nirmalsankalana/rice-leaf-disease-image`), yang mencakup citra daun padi sehat dan daun padi yang terinfeksi patogen.

---

## Anggota Kelompok 6

- YUSEP NURHOLIS 2406095
- Fugi Kusnaedi 2406088

---

_Untuk membaca laporan lengkap secara detail[Laporan_uas.md](./Laporan_uas.md)._

# Klasifikasi Penyakit Daun Padi Menggunakan Deep Learning

Repositori ini dibuat untuk memenuhi tugas **Ujian Akhir Semester (UAS) Mata Kuliah Kecerdasan Buatan**. Proyek ini berfokus pada implementasi teknologi _Computer Vision_ menggunakan pendekatan _Deep Learning_ untuk mendeteksi dan mengklasifikasikan berbagai jenis penyakit pada daun tanaman padi (_Oryza sativa_).

## Arsitektur Repositori

UAS-KecerdasanBuatan/
├── README.md # Dokumen panduan utama repositori & penjelasan langkah pengerjaan
├── Laporan_uas.md # Laporan komprehensif tertulis yang mencakup Poin 1 sampai Poin 10
├── uas_KecerdasanBuatan_model.ipynb # Notebook Jupyter utama berisi kode eksperimen, EDA, training,danevaluasi
└── data/
├── dataset/ # Folder tempat menyimpan sampel atau berkas data citra
└── Jurnal/ # Folder tempat menyimpan file PDF referensi ilmiah (minimal 5 jurnal)
UAS-KecerdasanBuatan/
├── README.md # Dokumen panduan utama repositori & penjelasan langkah pengerjaan
├── Laporan_uas.md # Laporan komprehensif tertulis yang mencakup Poin 1 sampai Poin 10
├── uas_KecerdasanBuatan_model.ipynb # Notebook Jupyter utama berisi kode eksperimen, EDA, training,danevaluasi
└── data/
├── dataset/ # Folder tempat menyimpan sampel atau berkas data citra
└── Jurnal/ # Folder tempat menyimpan file PDF referensi ilmiah (minimal 5 jurnal)

## Penjelasan Langkah Pengerjaan Proyek

1. Business & Data Understanding: Menentukan problem utama kegagalan panen akibat penyakit daun padi, merumuskan tujuan performa akurasi model (>85%), serta mendefinisikan karakteristik dataset citra digital RGB beresolusi jamak yang diunduh dari Kaggle.
2. Exploratory Data Analysis (EDA):

- Menganalisis sebaran jumlah sampel tiap kelas penyakit (countplot) untuk mendeteksi indikasi data tidak seimbang (imbalanced data).

- Melakukan ekstraksi fitur statistik warna rata-rata (RGB) dari piksel gambar untuk menghasilkan Heatmap Korelasi dan Pairplot Distribusi guna memetakan karakteristik awal pola sebaran data antar kelas penyakit.

3. Data Preparation:

- Melakukan kurasi data lewat verifikasi integritas file citra dan mengeliminasi data duplikat menggunakan MD5 Hash check.
- Membagi data secara proporsional menggunakan Stratified Train-Validation-Test Split (rasio 70:15:15) agar sebaran kelas tetap adil.
- Mengintegrasikan pipeline tf.data, melakukan resizing standar menjadi $224 \times 224 \times 3$, augmentasi gambar terintegrasi, serta kalkulasi class weight untuk menyeimbangkan bobot kelas saat training.

4. Modeling: Memasangkan arsitektur dasar EfficientNetB0 dan MobileNetV2 memanfaatkan metode Transfer Learning berbasis bobot latih awal dari ImageNet.
5. Evaluation: Menguji kedua model terlatih menggunakan data uji (Test Set) eksternal, mengekstrak grafik Confusion Matrix, serta membandingkan performa metrik Accuracy, Precision, Recall, dan F1-Score.

## Ringkasan Proyek & Model Pembanding

- EfficientNetB0: Dipilih karena efisiensinya yang luar biasa dalam menyeimbangkan parameter jaringan (kedalaman, lebar, resolusi) sehingga menghasilkan akurasi tinggi dengan komputasi yang relatif ringan.
- MobileNetV2: Dipilih sebagai algoritma pembanding karena arsitekturnya yang ultra-ringan dengan konsep Inverted Residuals dan Linear Bottlenecks, menjadikannya model yang ideal untuk implementasi pada perangkat mobile petani di lapangan.
- Sumber Dataset: Kaggle - Rice Leaf Disease Images (nirmalsankalana/rice-leaf-disease-image).
- Referensi Ilmiah: Minimal 5 berkas jurnal pendukung utama telah diunggah ke dalam folder data/Jurnal/ dengan sitasi bergaya APA (APA style).
