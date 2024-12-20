import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Fungsi untuk mendapatkan tanda tangan dan label dari semua subfolder
def get_signatures_and_labels(main_path="D:/MY PROG/PYTHON/ProjekUas/SignDataset"):
    signatures = []
    labels = []
    label_names = {}
    current_label = 0

    for folder_name in os.listdir(main_path):
        folder_path = os.path.join(main_path, folder_name)

        if os.path.isdir(folder_path):
            label_names[current_label] = folder_name
            print(f"Processing folder: {folder_name} with label {current_label}")

            for image_name in os.listdir(folder_path):
                image_path = os.path.join(folder_path, image_name)

                # Membaca gambar
                img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue  # Lewati jika gambar tidak valid

                # Preprocessing gambar tanda tangan
                _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
                resized = cv2.resize(binary, (150, 150))

                # Simpan tanda tangan dan labelnya
                signatures.append(resized)
                labels.append(current_label)

            current_label += 1

    return signatures, labels, label_names

# Preprocessing dan pelatihan model
def train_and_save_model(dataset_path="D:/MY PROG/PYTHON/ProjekUas/SignDataset"):
    signatures, labels, label_names = get_signatures_and_labels(dataset_path)

    if len(signatures) > 0:
        # Mengubah tanda tangan menjadi array datar (1D) untuk pelatihan SVM
        signatures_flattened = [signature.flatten() for signature in signatures]

        # Encode labels
        le = LabelEncoder()
        labels_encoded = le.fit_transform(labels)

        # Pisahkan dataset menjadi data latih dan uji
        X_train, X_test, y_train, y_test = train_test_split(signatures_flattened, labels_encoded, test_size=0.2, random_state=42)

        # Train SVM classifier
        clf = SVC(kernel='linear', probability=True)
        clf.fit(X_train, y_train)

        # Prediksi pada data uji
        y_pred = clf.predict(X_test)

        # Hitung akurasi
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Akurasi model: {accuracy * 100:.2f}%")

        # Simpan model pelatihan dan encoder
        joblib.dump(clf, 'svm_signature_model.pkl')
        joblib.dump(le, 'label_encoder_signature.pkl')
        joblib.dump(label_names, 'label_names.pkl')
        print("Model dan encoder berhasil disimpan!")
    else:
        print("Dataset kosong atau tidak valid. Pastikan dataset berisi gambar tanda tangan.")
        exit()

# Jalankan pelatihan
if __name__ == "__main__":
    train_and_save_model()
