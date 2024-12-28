import numpy as np
import matplotlib.pyplot as plt
from imageio.v2 import imread
from scipy.ndimage import sobel

def load_image(image_path):
    image = imread(image_path, mode='F')
    return image

def apply_sobel(image):
    dx = sobel(image, axis=0)
    dy = sobel(image, axis=1)
    edge_magnitude = np.hypot(dx, dy)
    return edge_magnitude / np.max(edge_magnitude)

def basic_thresholding(image, threshold):
    binary_image = (image > threshold).astype(np.uint8)
    return binary_image

def main(image_path, threshold):

    original_image = load_image(image_path)

    edge_image = apply_sobel(original_image)

    segmented_image = basic_thresholding(edge_image, threshold)

    fig, ax = plt.subplots(1, 3, figsize=(15, 5))

    ax[0].imshow(original_image, cmap='gray')
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    ax[1].imshow(edge_image, cmap='gray')
    ax[1].set_title("Sobel Edge Detection")
    ax[1].axis("off")

    ax[2].imshow(segmented_image, cmap='gray')
    ax[2].set_title(f"Segmented Image (Threshold={threshold})")
    ax[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    image_path = "C:\\Users\\User\\Downloads\\mario-.png"  
    threshold = 0.3  
    main(image_path, threshold)
    

# 1. Gambar Asli (Original Image)
# Deskripsi:

# Menampilkan gambar input dalam skala abu-abu, yang menjadi dasar untuk proses lebih lanjut.
# Tujuannya adalah untuk memberikan pandangan awal terhadap objek yang ingin dideteksi atau disegmentasi.
# Analisis:

# Gambar yang memiliki kontras tinggi antara objek dan latar belakang akan memberikan hasil deteksi tepi dan segmentasi yang lebih baik.
# Jika gambar terlalu gelap, terlalu terang, atau memiliki banyak noise, ini bisa memengaruhi kualitas hasil deteksi tepi.

# 2. Gambar Deteksi Tepi (Sobel Edge Detection)
# Deskripsi:

# Hasil dari operator Sobel yang mendeteksi perubahan intensitas (gradien) pada gambar.
# Sobel menghitung gradien dalam arah x dan y, kemudian menggabungkannya menjadi magnitudo gradien.
# Analisis:

# Bagian tepi dari objek utama dalam gambar akan tampak lebih terang.
# Noise atau detail kecil pada gambar asli mungkin juga muncul sebagai tepi, tergantung pada kualitas gambar asli.
# Jika tepi objek tidak terdeteksi dengan baik, itu bisa disebabkan oleh kurangnya kontras atau adanya noise.

# 3. Gambar Segmentasi (Segmented Image)
# Deskripsi:

# Gambar hasil thresholding, di mana area dengan intensitas lebih besar dari nilai ambang (threshold) diubah menjadi putih (foreground), sedangkan area lain menjadi hitam (background).
# Analisis:

# Nilai ambang sangat memengaruhi kualitas segmentasi:
# Ambang terlalu rendah: Banyak area yang seharusnya latar belakang ikut tersegmentasi sebagai foreground (over-segmentation).
# Ambang terlalu tinggi: Area tepi objek mungkin hilang atau tidak terdeteksi (under-segmentation).
# Untuk gambar dengan distribusi intensitas yang bervariasi, thresholding sederhana bisa menghasilkan segmentasi yang kurang optimal, sehingga metode thresholding adaptif mungkin diperlukan.
# Jika deteksi tepi tidak berhasil menangkap objek utama, segmentasi ini juga akan gagal.