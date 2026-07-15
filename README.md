# Klasifikasi Penyakit Daun Padi Menggunakan Deep Learning

Repositori ini dibuat untuk memenuhi tugas **Ujian Akhir Semester (UAS) Mata Kuliah Kecerdasan Buatan**. Proyek ini berfokus pada implementasi teknologi _Computer Vision_ menggunakan pendekatan _Deep Learning_ untuk mendeteksi dan mengklasifikasikan berbagai jenis penyakit pada daun tanaman padi (_Oryza sativa_).

---

## 📁 Arsitektur & Struktur Direktori Repositori

Penataan folder dan berkas di dalam repositori ini disusun secara sistematis sebagai berikut:

| Nama Berkas / Folder | Jenis | Keterangan |
| :--- | :---: | :--- |
| data/ | Folder | Direktori penyimpanan data dan referensi proyek |
| ├── dataset/ | Folder | Folder tempat menyimpan sampel atau berkas data citra daun padi |
| └── Jurnal/ | Folder | Folder tempat menyimpan file PDF referensi ilmiah (minimal 5 jurnal) |
| Laporan_uas.md | Berkas | Laporan komprehensif tertulis yang mencakup analisis Poin 1 sampai Poin 10 |
| README.md | Berkas | Dokumen panduan utama repositori & penjelasan langkah pengerjaan (berkas ini) |
| uas_KecerdasanBuatan_model.ipynb | Berkas | Notebook Jupyter berisi seluruh kode eksperimen, EDA, training, dan evaluasi |

---

## 📝 Alur dan Langkah Pengerjaan Proyek

Proses eksperimen dan pengembangan model dilakukan melalui tahapan sistematis berikut:

| Tahapan Pengerjaan | Deskripsi Aktivitas | Output / Hasil yang Diharapkan |
| :--- | :--- | :--- |
| Business & Data Understanding | Menentukan masalah utama kegagalan panen akibat patogen daun padi dan merumuskan batas minimal akurasi model sebesar 85%. | Identifikasi karakteristik awal dataset citra digital RGB dari Kaggle. |
| Exploratory Data Analysis (EDA) | Menganalisis sebaran jumlah sampel kelas menggunakan countplot serta mengekstraksi statistik warna piksel (RGB). | Deteksi dini isu imbalanced data serta peta sebaran visual karakteristik awal objek daun. |
| Data Preparation | Melakukan verifikasi integritas berkas citra, pembersihan gambar duplikat via MD5 Hash check, dan pembagian dataset dengan rasio 70:15:15. | Dataset siap latih yang bersih dilengkapi standardisasi ukuran gambar 224x224, augmentasi, dan bobot kelas. |
| Modeling | Mengintegrasikan arsitektur dasar EfficientNetB0 dan MobileNetV2 dengan menerapkan metode Transfer Learning berbasis bobot latih ImageNet. | Dua model klasifikasi terlatih yang siap diuji pada data eksternal secara objektif. |
| Evaluation | Menguji performa model pada Test Set murni, menggambar grafik Confusion Matrix, serta membandingkan metrik performa utama. | Tabel komparasi performa final berdasarkan nilai Accuracy, Precision, Recall, dan F1-Score. |

---

## 🤖 Ringkasan Proyek & Model Pembanding

Proyek ini membandingkan dua arsitektur _Convolutional Neural Network_ (CNN) mutakhir berbasis _Transfer Learning_ (menggunakan bobot pre-trained dari ImageNet):

| Model Pembanding | Alasan Pemilihan Arsitektur | Target Implementasi |
| :--- | :--- | :--- |
| EfficientNetB0 | Efisiensi tinggi dalam menyeimbangkan parameter kedalaman, lebar, dan resolusi jaringan secara bersamaan. | Akurasi maksimal pada lingkungan komputasi server awan (cloud). |
| MobileNetV2 | Struktur ultra-ringan berbasis konsep Inverted Residuals dan Linear Bottlenecks untuk meminimalkan beban komputasi. | Pemasangan luring (offline deployment) pada perangkat seluler milik petani di lapangan. |

### Informasi Tambahan
* **Sumber Dataset**: Kaggle - [Rice Leaf Disease Images](https://www.kaggle.com/datasets/nirmalsankalana/rice-leaf-disease-image) (`nirmalsankalana/rice-leaf-disease-image`).
* **Referensi Ilmiah**: Minimal 5 berkas jurnal pendukung utama telah diunggah ke dalam folder `data/Jurnal/` dengan sitasi bergaya APA (_APA style_).



_Untuk membaca laporan lengkap secara detail, silakan buka dokumen [Laporan_uas.md](./Laporan_uas.md)._



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

## 👥 Anggota Kelompok 6

* **YUSEP NURHOLIS** - 2406095
* **Fugi Kusnaedi** - 2406088

---

_Untuk membaca laporan lengkap secara detail, silakan buka dokumen [Laporan_uas.md](./Laporan_uas.md)._
