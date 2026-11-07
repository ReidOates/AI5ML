# Titanic Survival Prediction Using Decision Tree

## Deskripsi Proyek

Proyek ini merupakan implementasi machine learning untuk memprediksi kemungkinan keselamatan penumpang kapal Titanic menggunakan algoritma **Decision Tree Classifier**.

Tujuan proyek adalah membangun model klasifikasi yang dapat memprediksi apakah seorang penumpang selamat atau tidak berdasarkan beberapa informasi, seperti kelas tiket, jenis kelamin, usia, jumlah anggota keluarga yang ikut dalam perjalanan, tarif tiket, dan pelabuhan keberangkatan.

Proyek ini membandingkan dua konfigurasi Decision Tree, yaitu model tanpa pembatasan kedalaman pohon dan model dengan pembatasan kedalaman maksimum (`max_depth=4`). Hasil kedua model kemudian dievaluasi dan diimplementasikan ke dalam aplikasi web berbasis Flask.

## Dataset

Dataset yang digunakan berasal dari kompetisi Titanic: Machine Learning from Disaster di Kaggle.

**Sumber dataset:**  
https://www.kaggle.com/competitions/titanic/data

Dataset training terdiri dari 891 data penumpang dengan 12 kolom, termasuk kolom target `Survived`. Nilai target terdiri dari:
- `0` = Tidak selamat.
- `1` = Selamat.

File yang digunakan dalam proses pengembangan model adalah `train.csv`.

Pengguna perlu mengunduh dataset secara langsung dari Kaggle dan meletakkan file `train.csv` ke dalam folder `data/`.

## Fitur yang Digunakan

Fitur yang digunakan untuk membangun model:

| Fitur | Keterangan |
|---|---|
| Pclass | Kelas tiket penumpang |
| Sex | Jenis kelamin |
| Age | Usia penumpang |
| SibSp | Jumlah saudara atau pasangan yang ikut |
| Parch | Jumlah orang tua atau anak yang ikut |
| Fare | Tarif tiket |
| Embarked | Pelabuhan keberangkatan |

Kolom `PassengerId`, `Name`, `Ticket`, dan `Cabin` tidak digunakan sebagai fitur model.

## Metodologi

Tahapan pengerjaan proyek meliputi:

1. **Data Understanding**  
   Memahami struktur dataset, tipe data, jumlah data, distribusi target, dan kondisi missing values.

2. **Exploratory Data Analysis (EDA)**  
   Melakukan analisis awal untuk melihat hubungan karakteristik penumpang dengan tingkat keselamatan.

3. **Data Preprocessing**  
   - Menghapus kolom yang tidak digunakan.
   - Mengisi missing values pada `Age` menggunakan median data training.
   - Mengisi missing values pada `Embarked` menggunakan modus data training.
   - Mengubah fitur kategorikal menjadi numerik menggunakan One-Hot Encoding.

4. **Data Splitting**  
   Dataset dibagi menjadi data training sebesar 80% dan data testing sebesar 20%, menggunakan `random_state=42` dan `stratify`.

5. **Model Training**  
   Melatih dua model Decision Tree:
   - Model A: Decision Tree tanpa pembatasan `max_depth`.
   - Model B: Decision Tree dengan `max_depth=4`.

6. **Model Evaluation**  
   Mengevaluasi model menggunakan Accuracy, Precision, Recall, F1-Score, dan Confusion Matrix.

7. **Model Serialization**  
   Menyimpan model dan informasi preprocessing menggunakan Joblib agar dapat digunakan kembali tanpa melakukan training ulang.

8. **Web Application**  
   Mengimplementasikan model ke dalam aplikasi web menggunakan Flask dan HTML/CSS.

## Hasil Evaluasi

Pengujian dilakukan menggunakan 179 data testing.

| Metrik | Model A | Model B |
|---|---:|---:|
| Training Accuracy | 98.31% | 84.13% |
| Testing Accuracy | 82.68% | 79.33% |
| Precision (Selamat) | 80.65% | 86.36% |
| Recall (Selamat) | 72.46% | 55.07% |
| F1-Score (Selamat) | 76.34% | 67.26% |
| Kedalaman Pohon | 23 | 4 |
| Jumlah Node | 303 | 29 |
| Jumlah Leaf | 152 | 15 |

**Catatan:** Model B menggunakan pembatasan kedalaman maksimum sejak proses pembentukan pohon (`max_depth=4`), sehingga secara teknis termasuk pendekatan pre-pruning atau depth-limited tree.

### Analisis Hasil

Model A menghasilkan testing accuracy sebesar 82.68%, lebih tinggi dibandingkan Model B sebesar 79.33%. Model A juga memiliki recall kelas selamat yang lebih tinggi, yaitu 72.46%.

Namun, Model A memiliki perbedaan training accuracy dan testing accuracy yang cukup besar, yaitu sekitar 15.63 poin persentase. Hal ini menunjukkan adanya indikasi overfitting.

Model B memiliki testing accuracy sebesar 79.33% dengan selisih training dan testing accuracy sekitar 4.80 poin persentase. Struktur pohonnya juga lebih sederhana.

Dengan demikian, Model A unggul dalam hasil prediksi pada data testing, sedangkan Model B menawarkan struktur yang lebih sederhana dan selisih performa training-testing yang lebih kecil.

## Aplikasi Web

Model yang telah dilatih diintegrasikan ke dalam aplikasi berbasis Flask.

Fitur aplikasi:
- Memasukkan karakteristik penumpang.
- Memilih model Decision Tree A atau B.
- Menggunakan nilai median training apabila usia tidak diisi.
- Menggunakan nilai modus training apabila pelabuhan keberangkatan tidak diisi.
- Menampilkan hasil prediksi keselamatan penumpang.

**Demo aplikasi:**  
https://ai-5-ml.vercel.app/

## Teknologi yang Digunakan

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Joblib
- Flask
- HTML
- CSS
- Jupyter Notebook
- Git dan GitHub
- Vercel

## Struktur Folder

```text
TA-05-Titanic-Decision-Tree/
│
├── app/
│   ├── __init__.py
│   ├── app.py
│   └── templates/
│       └── index.html
│
├── public/
│   └── style.css
│
├── data/
│   └── train.csv
│
├── models/
│   └── titanic_artifacts.joblib
│
├── notebook/
│   └── TA05_Titanic.ipynb
│
├── .python-version
├── requirements.txt
├── README.md
└── .gitignore
```

## Cara Menjalankan Proyek

### 1. Clone Repository

```bash
git clone https://github.com/ReidOates/AI5ML.git
```

Masuk ke folder proyek:

```bash
cd TA-05-Titanic-Decision-Tree
```

Sesuaikan lokasi folder jika proyek berada di dalam direktori repository yang berbeda.

### 2. Buat Virtual Environment

```bash
python -m venv venv
```

Aktifkan virtual environment di Windows:

```powershell
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Siapkan Dataset

Unduh `train.csv` dari halaman Kaggle dan letakkan di dalam folder `data/`.

### 5. Jalankan Aplikasi

Dari direktori utama proyek:

```bash
python -m flask --app app.app run --debug
```

Buka browser dan akses:

```text
http://127.0.0.1:5000
```

Pastikan file model `models/titanic_artifacts.joblib` tersedia sebelum menjalankan aplikasi.

## Kesimpulan

Berdasarkan hasil pengujian, algoritma Decision Tree dapat digunakan untuk melakukan klasifikasi keselamatan penumpang Titanic berdasarkan karakteristik penumpang.

Model A menghasilkan testing accuracy yang lebih tinggi, sedangkan Model B memiliki struktur yang lebih sederhana dan indikasi overfitting yang lebih rendah. Perbandingan ini menunjukkan bahwa pemilihan model perlu mempertimbangkan lebih dari satu metrik evaluasi.

Proyek ini juga menunjukkan proses penerapan machine learning dari tahap eksplorasi data, preprocessing, pelatihan model, evaluasi, penyimpanan model, hingga implementasi ke dalam aplikasi web.

## Author

**Nama:** Muhammad Emil Mushthopa
**Program Studi:** Teknik Informatika  
**Universitas:** Universitas Bale Bandung (UNIBBA)

**Repository:**  
https://github.com/ReidOates/AI5ML

---

*Project developed for Machine Learning coursework.*
