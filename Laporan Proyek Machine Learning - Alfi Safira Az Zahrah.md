# Laporan Proyek _Machine Learning_ - Alfi Safira Az Zahrah

## Domain Proyek

Pada Proyek Pertama dengan studi kasus _Predictive Analytics_ ini, penulis memilih domain proyek Kesehatan.

### **Latar Belakang**

![Gambar](https://www.nhlbi.nih.gov/sites/default/files/styles/16x9_crop/public/2017-12/internal_structures_1_0.jpg?h=6c8b8c3d&itok=BNFPxPUV)

Menurut WHO, Penyakit jantung atau _Cardiovascular Disease_ (CVD) adalah kelompok gangguan yang terjadi pada jantung dan pembuluh darah, diantaranya seperti _coronary heart disease_, _cerebrovascular disease_, _rheumatic heart disease_, dan gangguan jantung lainnya. Pada tahun 2016, terdapat 17,5 juta jiwa (31%) dari 58 juta angka kematian di dunia disebabkan oleh penyakit jantung [[1]](http://eprints.ums.ac.id/73699/3/BAB%20I.pdf). Lebih dari 4 dari 5 kematian akibat CVD disebabkan oleh gagal jantung dan stroke [[2]](http://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2445/1498). Gagal jantung merupakan gangguan kegagalan jantung dalam melakukan fungsinya, yakni memompa darah sehingga kebutuhan metabolisme pada jaringan tidak dapat terpenuhi. Gangguan ini biasanya terjadi secara mendadak. Orang-orang penderita CVD atau yang berada pada risiko CVD tinggi (karena adanya satu atau lebih faktor risiko seperti hipertensi, diabetes, hiperlipidemia atau penyakit yang sudah ada) memerlukan deteksi dan manajemen dini di mana model _Machine Learning_ dapat sangat membantu. Oleh karena itu, dalam proyek ini penulis membuat model untuk memprediksi probabilitas penyakit gagal jantung menggunakan beberapa algoritma _Machine Learning_, diantaranya **K-Nearest Neighbor, Random Forest,** dan **Boosting Algorithm**. Dari ketiga algoritma tersebut nantinya akan dipilih salah satu model yang memiliki prediksi paling baik. Dengan adanya model ini diharapkan dapat memudahkan dalam memprediksi probabilitas seseorang mengalami gagal jantung.


## Business Understanding
---
### Problem Statements

Berdasarkan latar belakang yang telah diuraikan di atas, maka perumusan masalah yang akan diselesaikan pada proyek ini, diantaranya:
* Bagaimana cara membuat model untuk memprediksi probabilitas seseorang dalam mengalami gagal jantung menggunakan algoritma _machine learning_ berdasarkan fitur-fitur yang ada?
* Dari serangkaian fitur pada dataset, fitur manakah yang berpengaruh secara signifikan terhadap prediksi gagal jantung?
* Algoritma manakah yang memiliki prediksi terbaik?

### Goals

Berikut adalah tujuan dari pernyataan masalah:
- Mengetahui cara membuat model untuk memprediksi probabilitas seseorang dalam mengalami gagal jantung menggunakan algoritma _machine learning_ berdasarkan fitur-fitur yang ada 
- Menemukan fitur yang berpengaruh secara signifikan terhadap prediksi gagal jantung
- Mengetahui algoritma yang memiliki prediksi terbaik 

### Solution Statement
Berikut adalah solusi yang dilakukan untuk mencapai tujuan (_goals_) : 
* Pembuatan model prediksi probabilitas gagal jantung pada proyek ini akan dilakukan dengan menerapkan metodologi CRISP-DM (_Cross-Industry Standard Process for Data Mining_), berikut adalah tahapan-tahapan di dalamnya:

![gambar](https://miro.medium.com/max/900/1*jUGtCb1AS4GvutlX8ifflg.jpeg)

[Referensi Gambar](https://ruthsitorus.medium.com/mempelajari-modeling-cross-industry-standard-process-for-data-mining-atau-crisp-dm-166735c14159)

  **Metodologi CRISP-DM** merupakan salah satu metode standar untuk proses analitik yang paling umum digunakan. Metode ini akan membantu dalam mengelola proyek mulai dari tahap mendefinisikan masalah hingga mendapatkan _insight_ [[3]](https://www.dicoding.com/academies/319/tutorials/16994). Pada metodologi ini terdapat 6 fase proses analitik, namun pada proyek ini hanya akan sampai pada tahap _Evaluation_ (Evaluasi Model).

* Untuk menemukan fitur yang berpengaruh secara signifikan terhadap prediksi gagal jantung dilakukan pada tahap _Exploratory Data Analysis_ (EDA) menggunakan _Correlation Matrix_. Hubungan korelasi antar fitur ditentukan oleh nilai koefisien korelasi yang berkisar antara -1 dan +1. Semakin dekat nilainya ke angka 1 atau ke angka -1, maka korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke angka 0, maka korelasinya semakin lemah [[4]](https://www.dicoding.com/academies/319/tutorials/18570).

* Pada proyek ini, model yang dibangun akan menerapkan 3 algoritma _Machine Learning_, diantaranya **K-Nearest Neighbor, Random Forest,** dan **Boosting Algorithm**. Untuk metrik yang digunakan pada model prediksi ini adalah MSE (_Mean Square Error_) yang menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. Jikai nilai MSE kecil, maka model tersebut memiliki nilai prediksi yang baik. 


## Data Understanding
---
Dataset yang digunakan pada proyek ini yakni _[Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction)_. Dataset tersebut dibuat dengan menggabungkan dataset-dataset yang berbeda dan telah tersedia secara independen (tidak digabungkan) sebelumnya. Dataset-dataset yang digabungkan tersebut berasal dari observasi yang dilakukan diantaranya di Cleveland, Hungarian, Switzerland, Long Beach VA, dan Stalog. Berikut adalah informasi lebih lanjut mengenai dataset tersebut:

Tabel 1. Informasi Sumber Dataset
| Jenis | Keterangan |
| -------- | -------- |
| Sumber Dataset | Heart Failure Prediction Dataset : [Kaggle](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction) [5] |
| Pemilik Dataset | Fedesoriano |
| Kategori | Kesehatan, _Prediction Dataset_ |
| Lisensi | [Database : Open Database](https://opendatacommons.org/licenses/odbl/1-0/) |
| Jenis dan Ukuran Berkas | .csv (35.92 kb) |


Setelah melakukan proses unduh dataset, didapatkan informasi sebagai berikut:

Tabel 2. Informasi Dataset
| Jenis | Keterangan |
| -------- | -------- |
| Karakteristik Dataset | Multivariate |
| Karakteristik Atribut | Real |
| Jumlah Sampel | 918 |
| Jumlah Atribut | 12 |
| Missing Value | N/A |

### Variabel-variabel pada _Heart Failure Prediction Dataset_ adalah sebagai berikut:
1. Age : Usia pasien [years]
2. Sex : Jenis kelamin pasien [M: Male, F: Female]
3. ChestPainType : Tipe nyeri dada [TA: Typical Angina, ATA: Atypical Angina, NAP: Non-Anginal Pain, ASY: Asymptomatic]
4. RestingBP : Tekanan darah istirahat [mm Hg]
5. Cholesterol : Kolesterol serum [mm/dl]
6. FastingBS : Gula darah puasa [1: jika FastingBS > 120 mg/dl, 0: sebaliknya]
7. RestingECG : Hasil elektrokardiogram istirahat [Normal: Normal, ST: Memiliki kelainan gelombang ST-T (pembalikan gelombang T dan/atau elevasi atau depresi ST > 0.05 mV, LVH: Menunjukkan kemungkinan atau pasti hipertrofi ventrikel kiri menurut Estes' criteria]
8. MaxHR : Detak jantung maksimum tercapai [Nilai numerik antara 60 dan 202]
9. ExerciseAngina : Angina yang diinduksi oleh olahraga [Y: Ya, N: Tidak]
10. Oldpeak : oldpeak = ST [Nilai numerik diukur dalam depresi]
11. ST_Slope : Kemiringan puncak latihan segmen ST [Up: Upsloping, Flat: Flat, Down: Downsloping] 
12. HeartDisease : Keluaran [1: Penyakit jantung, 0: Normal]

## **Exploratory Data Analysis (EDA)**

**Menangani Missing Value**

Pendeteksian _missing value_ dilakukan menggunakan fungsi `isnull()`. Berikut hasil deteksi _missing value_ yang diperoleh:

Tabel 3. Hasil Deteksi Jumlah _Missing Value_
| Fitur | Jumlah _Missing Value_ |
| -------- | -------- |
| Age | 0 |
| Sex | 0 |
| ChestPainType | 0 |
| RestingBP | 0 |
| Cholesterol | 0 |
| FastingBS | 0 |
| MaxHR | 0 |
| ExerciseAngina | 0 |
| Oldpeak | 0 |
| ST_Slope | 0 |
| HeartDisease | 0 |

Pada Tabel 3 di atas dapat diketahui bahwa seluruh fitur pada dataset tidak memiliki _Missing Value_ (nilai NULL maupun NAN) sehingga dapat lanjut ke tahapan berikutnya.

**Mendeteksi dan Menangani Outliers**

Pada proyek ini, pendeteksian _outliers_ dilakukan dengan menerapkan teknik visualisasi data menggunakan _boxplot_. Berikut hasil pendeteksian _outliers_ untuk beberapa fitur pada dataset:

1. Fitur RestingBP

![Outliers RestingBP](https://user-images.githubusercontent.com/90541793/195747535-5713ddd3-9725-424e-8661-3ae7ac4625be.PNG)

2. Fitur Cholesterol

![Outliers Cholesterol](https://user-images.githubusercontent.com/90541793/195747783-9249b0fe-397f-4467-8aa3-fba2e199b98a.PNG)

3. Fitur MaxHR

![Outliers MaxHR](https://user-images.githubusercontent.com/90541793/195747834-ddd11487-2d31-48ad-bbf0-7130e9ab040e.PNG)

4. Fitur Oldpeak

![Outliers Oldpeak](https://user-images.githubusercontent.com/90541793/195747865-b804c7f9-d722-4ef0-94a1-6fa5cb559bc8.PNG)


Berdasarkan visualisasi boxplot di atas, dapat terlihat bahwa keempat fitur dataset, yakni `RestingBP`, `Cholesterol`, `MaxHR`, dan `Oldpeak` memiliki outliers. Untuk menangani outliers akan digunakan metode IQR (_Inter Quartile Range_).

Menurut Seltman dalam “Experimental Design and Analysis” [6], menyatakan bahwa outliers yang diidentifikasi oleh boxplot (disebut juga “boxplot outliers”) didefinisikan sebagai data yang nilainya 1.5 QR di atas Q3 atau 1.5 QR di bawah Q1.

Berikut persamaannya :
```
Batas bawah = Q1 - 1.5 * IQR
Batas atas = Q3 + 1.5 * IQR
```

Setelah menerapkan metode IQR untuk menangani _Outliers_, kemudian dilakukan pengecekan menggunakan fungsi `shape` untuk melihat ukuran dataset setelah _outliers_ dihilangkan. Berikut hasil outputnya :

```
(588,12)
```
Dapat dilihat bahwa ukuran dataset berkurang dari yang awalnya berjumlah 918 sampel menjadi 588 sampel dengan 12 kolom, yang artinya terdapat 330 baris sampel outliers yang telah dibersihkan.


## **Univariate Analysis**

Selanjutnya akan dilakukan proses analisa data dengan teknik Univariate Analysis untuk semua fitur, baik fitur numerikal maupun fitur kategorikal. 

**1. Analisis Categorical Features**

Berikut adalah visualisasi distribusi data untuk Categorical Features dalam bentuk bar.

![UA sex](https://user-images.githubusercontent.com/90541793/195747947-df9c9587-30ad-4e08-b526-c7466fd33c01.PNG)

Gambar 1. Distribusi Fitur Sex

Pada gambar 1 terlihat bahwa >50% pasien observasi berjenis kelamin Male (laki-laki).

![UA Resting ECG](https://user-images.githubusercontent.com/90541793/195747986-7238b857-43e9-44c0-b2e7-b0f9cb754d0e.PNG)

Gambar 2. Distribusi Fitur RestingECG

Pada gambar 2 terlihat label normal memiliki nilai tertinggi. Hal ini menunjukkan sebagian besar pasien observasi memiliki hasil ECG istirahat dalam kategori normal.

![UA chestpaintype](https://user-images.githubusercontent.com/90541793/195748027-d64c0898-3415-4bb5-93ba-626800ca189f.PNG)

Gambar 3. Distribusi Fitur Chest Pain Type

Pada gambar 3 terlihat label ASY tertinggi yang menunjukkan bahwa sebagian besar pasien observasi mengalami tipe nyeri dada ASY (Asymtomatic). Sedangkan untuk tipe nyeri TA (Typical Angina) hanya dialami oleh sebagian kecil pasien observasi.

![UA ExerciseAngina](https://user-images.githubusercontent.com/90541793/195748068-bc4fb92a-50c2-4d2b-bfed-4fb931421d33.PNG)

Gambar 4. Distribusi Fitur Exercise Angina

Pada gambar 4 di atas terlihat label N (Tidak) memiliki nilai tertinggi. Hal ini menunjukkan bahwa sebagian besar pasien tidak mengalami angina yang diinduksi oleh olahraga.

![UA ST Slope](https://user-images.githubusercontent.com/90541793/195748122-ce84a89b-fb9f-4409-86e8-92b9ee09ff7b.PNG)

Gambar 5. Distribusi Fitur ST Slope

Pada gambar 5 di atas terlihat label Up memiliki nilai tertinggi yang disusul dengan label Flat. Hal ini menunjukkan bahwa sebagian besar pasien memiliki kemiringan puncak latihan segmen ST yang Up (menaik) dan hanya sedikit yang memiliki kemiringan puncak latihan segmen ST yang Down (menurun).


**2. Analisis Numerikal Features**

Kemudian untuk melihat distribusi data pada tiap fitur numerik akan digunakan visualisasi dengan histogram sebagai berikut:

![Histogram Numerical Feature](https://user-images.githubusercontent.com/90541793/195748176-9d09724d-e2f3-4439-ac11-af750a9d0b13.png)

Gambar 6. Histogram Distribusi Fitur Numerik

Berdasarkan histogram di atas didapatkan informasi bahwa distribusi fitur `old peak` miring ke kanan (_right-skewed_). Hal ini dapat berimplikasi pada model


## **Multivariate Analysis**

Multivariate Analysis menunjukkan hubungan antara dua atau lebih variabel pada data. Selanjutnya akan dilakukan proses analisa data dengan teknik Multivariate Analysis baik untuk fitur numerikal maupun fitur kategorikal. 

**1. Analisis Categorical Features**

Pada tahap ini akan dilakukan pengecekan rata-rata probabilitas Heart Disease terhadap masing-masing fitur untuk mengetahui pengaruh fitur kategori terhadap Heart Disease.

![MA HD thd Sex](https://user-images.githubusercontent.com/90541793/195748227-109f4f50-58f0-45f1-bdfe-0d096635b666.PNG)

![MA HD thd ChestPainType](https://user-images.githubusercontent.com/90541793/195748265-81135e99-cfd8-4023-94a5-1cca17ce064c.PNG)

![MA HD thd RestingECG](https://user-images.githubusercontent.com/90541793/195748277-d4307be3-5689-4399-90cf-63fb257ce518.PNG)

![MA HD thd Exercise Angina](https://user-images.githubusercontent.com/90541793/195748348-a9262a68-d76c-418e-84b5-2b9a614fcdd0.PNG)

![MA HD thd ST Slope](https://user-images.githubusercontent.com/90541793/195748367-bac4b387-faf3-4c15-ada5-3e0e036b03ae.PNG)

Dengan mengamati rata-rata Heart Disease relatif terhadap fitur kategori di atas, maka diperoleh insight sebagai berikut:

- Pada fitur ‘Sex’, rata-rata Heart Disease cenderung lebih tinggi untuk Male dibandingkan dengan Female.

- Pada fitur ‘ChestPainType’, dapat terlihat bahwa tipe nyeri dada ASY memiliki nilai tinggi terhadap rata-rata relatif Heart Disease.

- Pada fitur ‘RestingECG’, rata-rata Heart Disease cenderung mirip dan tidak terlalu berbeda. Hal ini dapat berarti bahwa fitur 'RestingECG' memiliki pengaruh yang rendah terhadap Heart Disease. 

- Pada fitur 'ExerciseAngina', rata-rata Heart Disease cenderung lebih tinggi pada label 'Y'. 

- Pada fitur 'ST_Slope', rata-rata Heart Disease cenderung rendah untuk label 'Up' dan tinggi untuk label 'Down'. Hal ini dapat berarti bahwa fitur 'ST_Slope' memiliki pengaruh rendah terhadap Heart Disease.

**Kesimpulan**, beberapa fitur kategori memiliki pengaruh rendah dan beberapa tidak berpengaruh terhadap Heart Disease. 

**2. Analisis Numerikal Features**

Kemudian untuk mengamati hubungan antara fitur numerik, akan digunakan fungsi `pairplot()`, dengan output sebagai berikut:

![PAIRPLOT NUMERICAL FEATURE](https://user-images.githubusercontent.com/90541793/195748460-8c006871-a7aa-4952-8e8c-e0f3bbac789a.png)

Gambar 7. Pairplot Fitur Numerik

Berdasarkan pengamatan hasil visualisasi pairplot di atas dapat terlihat pola sebaran data pada fitur numerik. Pada pola sebaran tersebut terlihat bahwa fitur `MaxHR` memiliki korelasi negatif/berkebalikan dengan fitur `HeartDisease` (target). Sedangkan fitur `Age`, `RestingBP`, `Cholesterol`, `Oldpeak` memiliki korelasi positif dengan fitur `HeartDisease`.

**Korelasi antar Fitur Numerik**

Untuk mengevaluasi skor korelasi atau hubungan antar fitur numerik, akan digunakan fungsi `corr()` dengan output sebagai berikut.

![Correlation Matrix](https://user-images.githubusercontent.com/90541793/195748506-d9fa444f-629f-4212-8df7-59355bb3c83f.PNG)

Gambar 8. Correlation Matrix Fitur Numerik

Koefisien korelasi berkisar antara -1 dan +1. Semakin dekat nilainya angka 1 atau -1, maka korelasinya semakin kuat. Sedangkan, semakin dekat nilainya ke angka 0 maka korelasinya semakin lemah.

Berdasarkan gambar 8 di atas, dapat dilihat bahwa fitur `Oldpeak` (skor korelasi 0.52) dan `MaxHR` (skor korelasi -0.38) memiliki korelasi yang cukup kuat dengan fitur target HeartDisease dibandingkan dengan fitur-fitur lainnya.

## Data Preparation
Data preparation merupakan salah satu tahapan yang penting dalam proses pengembangan model machine learning. Pada tahapan ini akan dilakukan proses transformasi pada data sehingga menjadi bentuk yang cocok untuk proses pemodelan.

Pada proyek ini tahap Data Preparation yang dilakukan diantaranya sebagai berikut :
1. Encoding fitur kategori.

Proses encoding fitur kategori dilakukan dengan menggunakan teknik _one-hot-encoding_ dari library scikit-learn. Teknik ini berfungsi untuk mendapatkan fitur baru yang sesuai sehingga dapat mewakili setiap fitur kategori. Pada proyek ini terdapat 5 fitur kategori, yaitu 'Sex', 'ChestPainType', 'RestingECG', 'ExerciseAngina', dan 'ST_Slope'. Proses encoding dilakukan dengan fitur `get_dummies`. Berikut output nya :

![Encoding Ouput 1](https://user-images.githubusercontent.com/90541793/195748580-abf7c3cf-e479-47c9-a65f-671864bbd61e.PNG)
![Encoding Ouput 2](https://user-images.githubusercontent.com/90541793/195748588-892220a5-4c9e-467d-bba2-05c69c3e3f92.PNG)

2. Pembagian dataset menggunakan fungsi train_test_split dari library sklearn.

Proses membagi dataset menjadi data latih (train) dan data uji (test) merupakan hal yang harus dilakukan sebelum membuat model. Proses ini sebaiknya dilakukan di awal sebelum proses lainnya [[6]](https://www.oreilly.com/library/view/hands-on-predictive-analytics/9781789138719/), hal ini bertujuan agar kita tidak mengotori data uji dengan informasi yang kita dapat dari data latih. 

Pada proyek ini, pembagian dataset akan menggunakan proporsi pembagian 80:20 dengan fungsi train_test_split dari sklearn dengan output sebagai berikut.

Tabel 4. Hasil Pembagian Dataset

| Jumlah Data Latih (Train) | Jumlah Data Uji (Test) | Jumlah Total Data |
| -------- | -------- | -------- |
| 470 | 118 | 588 |

3. Standarisasi

Standarisasi merupakan teknik transformasi yang umum digunakan dalam tahap Data Preparation untuk pemodelan. Tujuan proses standarisasi adalah membantu untuk membuat fitur data menjadi bentuk yang lebih mudah diolah oleh algoritma. Pada proyek ini, fitur numerik akan distandarisasi menggunakan teknik `StandarScaler()` dari library scikit-learn.

StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0.

Berikut output yang dihasilkan dari proses standarisasi dengan teknik StandardScaler, output ditampilkan menggunakan fungsi `describe()` :

![Output Standarisasi](https://user-images.githubusercontent.com/90541793/195748676-9454ef76-71eb-4bdb-b070-483260c2f955.PNG)

Dapat dilihat bahwa setelah proses standarisasi sekarang nilai mean = 0 dan standar deviasi = 1.


## Modeling

Pada proyek ini, pemodelan dilakukan menggunakan tiga algoritma. Kemudian, selanjutnya akan dilakukan evaluasi performa masing-masing algoritma dan menentukan algoritma mana yang memberikan hasil prediksi terbaik. Berikut adalah algoritma yang akan digunakan, antara lain : 

**1. K-Nearest Neighbor (KNN)**

* Kelebihan : relatif sederhana, mudah dipahami dan digunakan.
* Kekurangan : jika dihadapkan pada jumlah fitur atau dimensi yang besar rawan terjadi bias (_curse of dimensionality_).

**2. Random Forest**

* Kelebihan : merupakan algoritma yang cukup sederhana tetapi memiliki stabilitas yang mumpuni, menggunakan teknik _bagging_ untuk mengatasi _overfitting_ dengan berjalan secara paralel, dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi.  
* Kekurangan : algoritma yang kompleks dan membutuhkan waktu relatif lebih lama dan daya komputasi yang lebih tinggi dibanding algoritma lain, seperti Decision Tree.

**3. Boosting Algorithm**

* Kelebihan : menggunakan teknik _boosting_ untuk mengurangi bias dengan berjalan secara sekuensial.
* Kekurangan : algoritma yang kompleks dan membutuhkan waktu relatif lebih lama dan daya komputasi yang lebih tinggi dibanding algoritma lain.


## **Model K-Nearest Neighbor (KNN)**

Algoritma KNN bekerja dengan membandingkan jarak satu sampel ke sampel pelatihan lain dengan memilih sejumlah k tetangga terdekat (dengan k adalah sebuah angka positif).

Pemilihan nilai k sangat penting dan berpengaruh terhadap performa model. Jika kita memilih k yang terlalu rendah, maka akan menghasilkan model yang overfit dan hasil prediksinya memiliki varians tinggi. Jika kita memilih k terlalu tinggi, maka model yang dihasilkan akan underfit dan prediksinya memiliki bias yang tinggi [[7]](https://www.oreilly.com/library/view/machine-learning-with/9781617296574/). 

Pada proyek ini kita akan mencoba beberapa nilai k yang berbeda (1 sampai 20), kemudian akan dibandingkan dan dipilih yang menghasilkan nilai metrik model terbaik. Metrik model yang digunakan yakni _mean squared error_ dan metrik ukuran jarak yang digunakan yakni Minkowski Distance pada library sklearn.

Berikut adalah tabel perbandingan nilai K terhadap nilai MSE.

Tabel 5. Perbandingan Nilai K terhadap MSE

| Nilai K | MSE |
| -------- | -------- |
| 1 | 0.3898305084745763 |
| 2 | 0.3792372881355932 |
| 3 | 0.36723163841807904 |
| 4 | 0.3315677966101695 |
| 5 | 0.32508474576271185 |
| 6 | 0.29755178907721286 |
| 7 | 0.2917675544794188 |
| 8 | 0.2770127118644068 |
| 9 | 0.2680477087256748 | 
| 10 | 0.2673728813559322 |
| 11 | 0.2712564784983891 |
| 12 | 0.26877354048964225 |
| 13 | 0.26727509778357234 |
| 14 | 0.25272397094431 |
| 15 | 0.24870056497175144 | 
| 16 | 0.2507944915254237 |
| 17 | 0.24558676910445132 |
| 18 | 0.23624189160912326 | 
| 19 | 0.23231137612094466 |
| 20 | 0.22970338983050845 |


Untuk mempermudah dalam menentukan nilai k terbaik akan dilakukan visualisasi menggunakan fungsi `plot()` untuk nilai MSE yang telah diperoleh sebelumnya.

![Plot Nilai K terhadap MSE](https://user-images.githubusercontent.com/90541793/195748751-f72ffc1b-9971-4f58-835e-ed42d8d730a2.png)

Berdasarkan output di atas dapat terlihat bahwa nilai MSE terbaik (terendah) dicapai ketika nilai `k = 20` yaitu sebesar 0.229. Oleh karena itu pada pemodelan data latih nilai k yang digunakan yakni `k = 20`.

## **Model Algoritma Random Forest**

Algoritma random forest merupakan salah satu algoritma supervised learning yang dapat digunakan untuk menyelesaikan masalah klasifikasi dan regresi. 

Random forest merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning. Pada model ensemble, setiap model harus membuat prediksi secara independen. Kemudian, prediksi dari setiap model ensemble ini digabungkan untuk membuat prediksi akhir. 

Pada proyek ini kita akan menggunakan `RandomForestRegressor` dari library scikit-learn. Berikut parameter-parameter yang digunakan :
* n_estimator: jumlah trees (pohon) di forest.
* max_depth: kedalaman atau panjang pohon. Ia merupakan ukuran seberapa banyak pohon dapat membelah (splitting) untuk membagi setiap node ke dalam jumlah pengamatan yang diinginkan.
* random_state: digunakan untuk mengontrol random number generator yang digunakan.
* n_jobs: jumlah job (pekerjaan) yang digunakan secara paralel. Ia merupakan komponen untuk mengontrol thread atau proses yang berjalan secara paralel. n_jobs=-1 artinya semua proses berjalan secara paralel.

Untuk menentukan nilai parameter (`n_estimators` dan `max_depth`) terbaik, kita akan melakukan tuning dengan `GridSearchCV`, berikut hasil outputnya. 

Tabel 6. Hasil Tuning Parameter dengan GridSearchCV
| Parameter | Daftar Nilai | Nilai Terbaik |
| -------- | -------- | -------- |
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 60 |
| max_depth | 4, 8, 16, 32 | 4 |
| RF GridSearch Score | | 0.5815682140332608 |

Berdasarkan output didapatkan nilai `max_depth` = 4 dan `n_estimators` = 60 dengan skor terbaik dari GridSearch = 0.581.

Oleh karena itu nilai-nilai tersebut yang akan kita terapkan pada pengaturan pemodelan.

Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

## **Model Boosting Algorithm**

Boosting Algorithm juga merupakan salah satu model machine learning yang termasuk ke dalam kategori ensemble (group) learning yang menggunakan teknik Boosting. Pada algoritma ini model dibangun dari data latih (train), kemudian membuat model kedua yang bertugas memperbaiki kesalahan dari model pertama. Model terus ditambahkan sampai data latih terprediksi dengan baik atau telah mencapai jumlah maksimum model untuk ditambahkan.

Pada proyek ini kita akan menggunakan Adaptive Boosting menggunakan `AdaBoostRegressor` dari library scikit-learn. Berikut parameter-parameter yang digunakan :
* n_estimator: jumlah estimator, ketika mencapai nilai jumlah tersebut algoritma Boosting akan dihentikan.
* learning_rate: bobot yang diterapkan pada setiap regressor di masing-masing iterasi Boosting.
* random_state: digunakan untuk mengontrol random number generator yang digunakan.

Untuk menentukan nilai parameter (`n_estimators` dan `learning_rate`) terbaik, kita akan melakukan tuning dengan `RandomizedSearchCV`, berikut hasil output nya. 

Tabel 7. Hasil Tuning Parameter dengan RandomizedSearchCV
| Parameter | Daftar Nilai | Nilai Terbaik |
| -------- | -------- | -------- |
| n_estimators | 10, 20, 30, 40, 50, 60, 70, 80, 90 | 90 |
| learning_rate | 0.001, 0.01, 0.1, 0.2 | 0.01 |
| Nilai MSE pada data latih | | 0.07942373927636698 |
| Nilai MSE pada data uji | | 0.16607175798162316 |

Berdasarkan output di atas didapatkan nilai `learning_rate` = 0.01 dan `n_estimators` = 90 yang menghasilkan nilai MSE terbaik pada data latih yakni 0.079 dan pada data uji 0.1504.

Oleh karena itu nilai-nilai tersebut yang akan kita terapkan pada pengaturan pemodelan.

## Evaluation

Pada tahap Modeling atau pemodelan kita telah membuat 3 model menggunakan algoritma yang berbeda. Selanjutnya pada tahap ini akan dilakukan evaluasi model-model yang telah dibuat menggunakan metrik MSE (_Mean Squared Error_). MSE menghitung jumlah selisih kuadrat rata-rata nilai sebenarnya dengan nilai prediksi. MSE didefinisikan dalam persamaan berikut

![Rumus MSE](https://user-images.githubusercontent.com/90541793/195748803-2d430422-d8d1-4074-b31f-4d6efde7f3a1.jpeg)

**Keterangan :**

MSE = Mean Squared Error
n = banyaknya data point (baris)
Y_i = nilai yang diobservasi (fitur target PE)
Y^_i = hasil prediksi

**Cara kerja :** 

Cara kerja Metrik MSE adalah dengan menghitung selisih hasil prediksi dengan nilai fitur target (PE). Nilai selisih tersebut, disebut juga sebagai nilai eror yang kemudian di kuadratkan untuk menangani nilai selisih negatif. Selanjutnya hasil pengkuadratan setiap nilai selisih dijumlahkan dan terakhir dibagi dengan banyak data point (n) untuk memperoleh nilai rata-ratanya. Rata-rata inilah yang disebut Mean Squared Error (MSE).

Berikut adalah tabel nilai MSE pada setiap model dengan data latih dan data uji :

Tabel 8. Nilai MSE Tiap Model

| Algoritma | Train | Test |
| -------- | -------- | -------- |
| KNN | 0.00009 | 0.000168 |
| RF | 0.000065 | 0.00018 |
| Boosting | 0.00008 | 0.00018 |

Untuk memudahkan dalam mengevaluasi model kita akan melakukan visualisasi hasil menggunakan bar chart sebagai berikut.

![Plot Metrik Evaluasi](https://user-images.githubusercontent.com/90541793/195748871-05fe5b68-812b-4dba-81e8-ee0ffa0359fe.png)

Gambar 9. Bar Chart Nilai MSE Tiap Model

Berdasarkan bar chart di atas dapat terlihat bahwa model dengan algoritma Random Forest (RF) memberikan nilai MSE yang paling kecil, sedangkan model dengan algoritma K-Nearest Neighbor (KNN) memberikan nilai MSE yang paling besar. 

Sebelum menarik kesimpulan terkait model terbaik untuk memprediksi probabilitas gagal jantung (_Heart Disease_) kita akan melakukan uji coba prediksi menggunakan beberapa sampel acak pada data uji sebagai berikut.

**Menguji dengan Beberapa Sampel Acak dari Dataset**

Berikut hasil uji prediksi menggunakan beberapa sampel acak dari dataset.

![Hasil uji prediksi](https://user-images.githubusercontent.com/90541793/195748926-bbe7a72c-267b-4b25-9d88-a4fdaf14c4d9.PNG)


## Kesimpulan

* Berdasarkan _Correlation Matrix_ dapat dilihat bahwa fitur yang memiliki korelasi cukup kuat dibandingkan fitur lainnya dengan fitur Heart Disease (target) adalah fitur `Oldpeak` (skor korelasi 0.52) dan `MaxHR` (skor korelasi -0.38).

* Berdasarkan hasil evaluasi dapat disimpulkan bahwa model terbaik untuk prediksi adalah model dengan algoritma Random Forest (RF). Dengan pengaturan parameter n_estimators = 60 dan max_depth = 4 didapatkan nilai metrik terkecil jika dibandingkan dengan model algoritma lain, yakni _Mean Squared Error_ (MSE) sebesar 0.000065 (data latih) dan 0.00018 (data uji). Selain itu pada uji prediksi menggunakan sampel acak dapat terlihat bahwa model Random Forest mampu secara signifikan memprediksi mendekati hasil yang benar.

## Daftar Referensi

[1] Murti, Tyan Adhi Kurnia and , Ns. Beti Kristinawati, M.Kep., Sp. Kep. M.B (2019) Gambaran Lama Hari Rawat Pasien Gagal Jantung di Rsud Dr.Moewardi Kota Surakarta. Skripsi thesis, Universitas Muhammadiyah Surakarta. [[Link]](http://eprints.ums.ac.id/73699/3/BAB%20I.pdf)

[2] S. P. Tamba and E. -, “PREDIKSI PENYAKIT GAGAL JANTUNG DENGAN MENGGUNAKAN RANDOM FOREST”, JUSIKOM PRIMA, vol. 5, no. 2, pp. 176 - 181, Mar. 2022. [[Link]](http://jurnal.unprimdn.ac.id/index.php/JUSIKOM/article/view/2445/1498) 

[5] fedesoriano. (September 2021). Heart Failure Prediction Dataset. Retrieved [Date Retrieved] from https://www.kaggle.com/fedesoriano/heart-failure-prediction.