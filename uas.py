# =====================================
# STEP 1: Install dan Import Library
# =====================================
!pip install -q tensorflow scikit-learn pandas

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np

# =====================================
# STEP 2: Dataset Minimal 20 Berita
# =====================================
texts = [
    "Pemerintah umumkan vaksin aman dan efektif",
    "Artis terkenal wafat karena kecelakaan",
    "Ilmuwan temukan obat baru untuk kanker",
    "Gempa bumi melanda Jakarta",
    "Menteri keuangan umumkan kebijakan pajak",
    "Awas! Buah ini bisa sebabkan kematian instan",
    "Minum kopi bisa sembuhkan Covid-19",
    "Makan telur mentah bikin IQ naik 2x lipat",
    "Cuci tangan bisa menyebabkan kanker kulit",
    "Vaksin mengandung chip pelacak",
    "Presiden resmikan jembatan terpanjang",
    "PLN umumkan pemadaman bergilir",
    "NASA tangkap sinyal alien dari Mars",
    "Minyak kayu putih bisa menyembuhkan kanker",
    "Bank Indonesia naikkan suku bunga",
    "Bulan akan jatuh ke bumi tahun depan",
    "Guru besar UI beri kuliah umum tentang AI",
    "Kecoa bisa jadi sumber energi masa depan",
    "Kemenkes siapkan vaksin baru untuk flu",
    "Kuku kotor bisa sebabkan kerusakan otak"
]
labels = [1,1,1,1,1,0,0,0,0,0,1,1,0,0,1,0,1,0,1,0]

# =====================================
# STEP 3: Preprocessing Teks
# =====================================
tokenizer = Tokenizer(num_words=1000, oov_token="<OOV>")
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)
padded = pad_sequences(sequences, padding='post', maxlen=20)

# =====================================
# STEP 4: Split Data (Train/Test)
# =====================================
X_train, X_test, y_train, y_test = train_test_split(
    padded, labels, test_size=0.2, random_state=42
)

# =====================================
# STEP 5: Bangun Model CNN
# =====================================
model = Sequential([
    Embedding(input_dim=1000, output_dim=64, input_length=20),
    Conv1D(64, 3, activation='relu'),
    GlobalMaxPooling1D(),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# =====================================
# STEP 6: Training Model
# =====================================
history = model.fit(
    np.array(X_train),
    np.array(y_train),
    epochs=10,
    validation_split=0.2,
    verbose=1
)

# =====================================
# STEP 7: Evaluasi Model
# =====================================
loss, accuracy = model.evaluate(np.array(X_test), np.array(y_test))
print(f"\nTest Loss: {loss:.2f}")
print(f"Test Accuracy: {accuracy:.2f}")

# =====================================
# STEP 8: Laporan Klasifikasi (Tabel)
# =====================================
y_pred = model.predict(np.array(X_test))
y_pred_classes = (y_pred > 0.5).astype("int32")

report_dict = classification_report(
    y_test,
    y_pred_classes,
    target_names=["Hoaks", "Asli"],
    output_dict=True
)

# Buat DataFrame untuk visualisasi
report_df = pd.DataFrame(report_dict).transpose()

# Tampilkan hasil dalam bentuk tabel
print("\n📊 Classification Report:")
print(report_df)

# =====================================
# STEP 9 (Optional): Simpan sebagai CSV
# =====================================
report_df.to_csv("laporan_klasifikasi_hoaks.csv")
print("\n📁 Laporan disimpan sebagai 'laporan_klasifikasi_hoaks.csv'")
