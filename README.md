- Penjelasan Teori Yang Mendukung 
Filtering citra yaitu salah satu tahap dalam pengolahan citra digital yang bertujuan untuk memperbaiki kualitas citra. Filtering citra dapat dibagi menjadi dua kategori, yaitu filtering spasial dan filtering frekuensi.

    - Filtering Spasial

Filtering spasial adalah proses manipulasi kumpulan piksel dari sebuah citra untuk menghasilkan citra baru. Proses ini dilakukan dengan menggunakan kernel atau mask yang berupa kumpulan piksel berukuran 2x2, 3x3, 5x5, dan seterusnya. Kernel ini digunakan untuk mengkonvolusi citra asal dan menghasilkan citra baru.

Contoh dari filtering spasial adalah low-pass filtering dan high-pass filtering. Low-pass filtering adalah proses filter yang melewatkan komponen citra dengan nilai intensitas yang rendah dan meredam komponen citra dengan nilai intensitas yang tinggi. Low-pass filter akan menyebabkan citra menjadi lebih halus dan lebih blur.

High-pass filtering adalah proses filter yang melewatkan komponen citra dengan nilai intensitas yang tinggi dan meredam komponen citra dengan nilai intensitas yang rendah. High-pass filter akan menyebabkan tepi objek tampak lebih tajam dibandingkan sekitarnya.

    - Filtering Frekuensi

Filtering frekuensi yaitu proses yang dilakukan dengan menggunakan transformasi Fourier. Proses ini dilakukan dengan mengubah citra dari domain spasial ke domain frekuensi, melakukan proses filtering pada domain frekuensi, dan kemudian mengubah kembali citra ke domain spasial.

Contoh dari filtering frekuensi adalah Fast Fourier Transform (FFT). FFT adalah metode transformasi yang memindahkan informasi citra dari domain spasial ke dalam domain frekuensi. Setelah dilakukan proses filtering dalam domain frekuensi, informasi citra dikembalikan ke domain spasial.

- Operasi Titik

Operasi titik yaitu proses yang dilakukan dengan memodifikasi histogram citra masukan agar sesuai dengan karakteristik yang diharapkan. Teknik operasi titik antara lain adalah intensity adjustment dan histogram equalization.

Intensity adjustment bekerja dengan cara melakukan pemetaan linear terhadap nilai intensitas pada histogram awal menjadi nilai intensitas pada histogram yang baru.

Histogram equalization bertujuan untuk menghasilkan citra keluaran yang memiliki nilai histogram yang relatif sama.

- Median Filtering

Median filtering yaitu salah satu jenis low-pass filter yang bekerja dengan mengganti nilai suatu piksel pada citra asal dengan nilai median dari piksel tersebut dan lingkungan tetangganya. Filter ini lebih tidak sensitif terhadap perbedaan intensitas yang ekstrim.

Dalam pengolahan citra digital, filtering citra digunakan untuk memperbaiki kualitas citra, menonjolkan suatu ciri tertentu dalam citra, ataupun untuk memperbaiki aspek tampilan. Proses ini biasanya bersifat eksperimental, subjektif, dan bergantung pada tujuan yang hendak dicapai.



- PENJELASAN TAHAPAN MENYELESAIKAN PROYEK

 1.  INSTALASI PAKET
pip install opencv-python matplotlib numpy scikit-image
 
 2. IMPORT LIBRARY
import cv2
import matplotlib.pyplot as plt
import numpy as np
import skimage
from skimage.feature import greycomatrix, greycoprops

cv2: Library OpenCV untuk pengolahan gambar.
matplotlib.pyplot: Library untuk visualisasi data.
numpy: Library untuk operasi numerik.
skimage: Library untuk pengolahan gambar, termasuk fungsi-fungsi untuk ekstraksi fitur gambar.

 3. DEFINISI FUNGSI MEAN FILTER
def mean_filter(image, kernel_size=3):
    padded_image = np.pad(image, ((kernel_size // 2, kernel_size // 2), (kernel_size // 2, kernel_size // 2)), mode='constant')
    filtered_image = np.zeros_like(image)
    
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded_image[i:i + kernel_size, j:j + kernel_size]
            filtered_image[i, j] = np.mean(region)
    
    return filtered_image

mean_filter: Fungsi ini melakukan penyaringan rata-rata (mean filter) secara manual pada sebuah gambar.
np.pad: Menambahkan padding ke gambar untuk memastikan setiap piksel dapat diakses oleh kernel filter.
np.zeros_like: Membuat array dengan ukuran yang sama seperti gambar asli, diisi dengan nol.
Dua loop for: Mengiterasi setiap piksel dalam gambar untuk menghitung nilai rata-rata dari setiap region yang sesuai dengan ukuran kernel.

 4. LOAD DAN KONVERSI GAMBAR
image_path = 'Galih_Raka_Siwi.jpg'
img = io.imread(image_path)

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

io.imread: Membaca gambar dari file.
cv2.cvtColor: Mengkonversi gambar dari RGB ke grayscale.

 5. PENERAPAN FILTER MEDIAN DAN MEAN
# Apply median filter using OpenCV
median_filtered = cv2.medianBlur(img, 5)

# Apply mean filter
mean_filtered_gray = mean_filter(img_gray, kernel_size=3)
mean_filtered_rgb = cv2.cvtColor(mean_filtered_gray, cv2.COLOR_GRAY2RGB)

cv2.medianBlur: Menerapkan filter median dengan ukuran kernel 5.
mean_filter: Menerapkan filter rata-rata dengan ukuran kernel 3.
cv2.cvtColor: Mengkonversi gambar hasil filter rata-rata kembali ke RGB untuk keperluan visualisasi.


 6. PLOTTING GAMBAR
fig, axs = plt.subplots(2, 2, figsize=(14, 12))
ax = axs.ravel()

ax[0].imshow(img)
ax[0].set_title('Citra Asli')

ax[1].imshow(img_gray, cmap='gray')
ax[1].set_title('Original Image')

ax[2].imshow(median_filtered)
ax[2].set_title('After Filter Median')

ax[3].imshow(mean_filtered_rgb)
ax[3].set_title('Mean Filtered Image')

plt.tight_layout()
plt.show()

plt.subplots: Membuat layout 2x2 untuk plot gambar.
ax.imshow: Menampilkan gambar pada setiap subplot.
ax.set_title: Menambahkan judul pada setiap subplot.
plt.tight_layout: Menyesuaikan layout agar tidak ada tumpang tindih.
plt.show: Menampilkan plot.


 7. HITUNG MEAN DAN STANDARD DEVIATION
mean = np.mean(img_gray.ravel())
std = np.std(img_gray.ravel())

print(mean, std)

np.mean: Menghitung rata-rata piksel dari gambar grayscale.
np.std: Menghitung deviasi standar piksel dari gambar grayscale.
ravel: Mengubah array 2D menjadi 1D untuk keperluan perhitungan.

 8. EKSTRAKSI FITUR DENGAN GLCM (GRAY LEVEL Co-occurrence MATRIX)
glcm = greycomatrix(img_gray, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)

contrast = greycoprops(glcm, 'contrast')[0, 0]
dissimilarity = greycoprops(glcm, 'dissimilarity')[0, 0]
homogeneity = greycoprops(glcm, 'homogeneity')[0, 0]
energy = greycoprops(glcm, 'energy')[0, 0]
correlation = greycoprops(glcm, 'correlation')[0, 0]

print('contrast \t: %04f' % contrast)
print('dissimilarity \t: %04f' % dissimilarity)
print('homogeneity \t: %04f' % homogeneity)
print('energy \t: %04f' % energy)
print('correlation \t: %04f' % correlation)

- greycomatrix: Membuat matriks co-occurence tingkat abu-abu untuk menghitung tekstur gambar.
distances=[1]: Menggunakan jarak 1 piksel.
angles=[0]: Menggunakan sudut 0 derajat (horizontal).
levels=256: Menggunakan 256 level abu-abu.
symmetric=True: Membuat matriks simetris.
normed=True: Menormalkan matriks.
- greycoprops: Menghitung properti tekstur dari GLCM.
contrast, dissimilarity, homogeneity, energy, correlation: Menghitung berbagai fitur tekstur dari gambar.
- print: Menampilkan hasil perhitungan fitur tekstur.
