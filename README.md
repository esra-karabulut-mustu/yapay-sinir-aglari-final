# 🎓 YAPAY SİNİR AĞLARI - FİNAL ÖDEVİ

## 📚 Proje Bilgileri

**Öğrenci:** Esra Karabulut Muştu  
**Numara:** 244312029  
**Konu:** ISIC 2018 Deri Lezyonu Görüntülerinde İkili Sınıflandırma  
**Dataset:** [Kaggle - Skin Cancer ISIC](https://www.kaggle.com/datasets/nodoubttome/skin-cancer9-classesisic)

colab linki:
https://colab.research.google.com/drive/1hS-8Gn5CgMHhacOxI7q0a3-e78lpE72X?usp=sharing


github linki:
https://github.com/esra-karabulut-mustu/yapay-sinir-aglari-final

---

## 📁 Dizin Yapısı

```
teslim_edilecekler/
├── 1_notebook/              # Jupyter Notebook dosyaları
├── 2_modeller/              # Eğitilmiş model dosyaları (.keras)
├── 3_figürler/              # Eğitim grafikleri, confusion matrix, ROC
├── 4_gradcam/               # Grad-CAM görselleştirmeleri
├── 5_raporlar/              # JSON raporlar, metrikler
├── 6_outputs_zip/           # Tüm çıktıları içeren ZIP
└── README.md                # Bu dosya
```

---

## 🎯 Ödev Gereksinimleri ve Karşılanan Maddeler

### ✅ 1. Veri Seti Hazırlığı
- ✅ En yüksek örnek sayılı 2 sınıf seçildi
- ✅ İkili sınıflandırma için etiketleme yapıldı
- ✅ Train/Val/Test split (%70/%15/%15)

### ✅ 2. Veri İşleme
- ✅ Resize: 224×224
- ✅ Normalizasyon: [0-1] aralığına rescale
- ✅ Data augmentation (sadece train)

### ✅ 3. Model-1: Scratch CNN
- ✅ Önerilen mimari ile eğitim
- ✅ 100 epoch (EarlyStopping ile)
- ✅ Adam optimizer, lr=1e-3
- ✅ Callbacks: EarlyStopping, ReduceLROnPlateau

### ✅ 4. Model-2: MobileNetV2 Transfer Learning
- ✅ Freeze aşaması (100 epoch)
- ✅ Fine-tuning aşaması (son %25 katman, 100 epoch)
- ✅ Düşük learning rate (1e-5)

### ✅ 5. Model-3: EfficientNetB0 Transfer Learning
- ✅ Freeze aşaması (100 epoch)
- ✅ Fine-tuning aşaması (son %25 katman, 100 epoch)
- ✅ Düşük learning rate (1e-5)

### ✅ 6. Değerlendirme Metrikleri
- ✅ Accuracy, Precision, Recall, F1-score, ROC-AUC
- ✅ Confusion Matrix (3 model)
- ✅ ROC Curve karşılaştırması
- ✅ Karşılaştırma tablosu

### ✅ 7. Grad-CAM
- ✅ Scratch CNN: 6/6 görselleştirme
- ✅ MobileNetV2: 6/6 görselleştirme (GPU ile çözüldü)
- ✅ EfficientNetB0: 6/6 görselleştirme (GPU ile çözüldü)

---

## 📊 Model Performansları

### 10 Epoch Test Sonuçları:

| Model | Accuracy | Precision | Recall | F1 | AUC |
|-------|----------|-----------|--------|----|----|
| Scratch CNN | 0.514 | 0.264 | 0.514 | 0.349 | 0.573 |
| **MobileNetV2** | **0.643** | **0.643** | **0.643** | **0.643** | **0.658** |
| EfficientNetB0 | 0.550 | 0.554 | 0.550 | 0.547 | 0.563 |

**En İyi Model:** MobileNetV2 (%64.3 accuracy)

---

## 🔬 Grad-CAM Görselleştirmeleri

### ✅ Tüm Modeller İçin Başarıyla Tamamlandı

**Her model için:**
- 3 doğru sınıflandırma
- 3 yanlış sınıflandırma
- Toplam: 6 görselleştirme

**Modeller:**
- ✅ Scratch CNN
- ✅ MobileNetV2 (GPU ile çözüldü)
- ✅ EfficientNetB0 (GPU ile çözüldü)

Detaylı görselleştirmeler: `4_gradcam/`

---

## 📦 Dosya İçerikleri

### 1_notebook/
- `Ana_Notebook.ipynb`: Tam pipeline (eski 100 epoch eğitim)
- `Fix_Notebook_10epoch.ipynb`: 10 epoch test eğitimi

### 2_modeller/
- Scratch CNN modelleri (best + final)
- MobileNetV2 modelleri (freeze + finetune + 10epoch)
- EfficientNetB0 modelleri (freeze + finetune + 10epoch)

### 3_figürler/
- Training curves (accuracy + loss)
- Confusion matrices
- ROC curves
- Augmentation examples

### 4_gradcam/
- Scratch CNN: 6 görselleştirme
- MobileNetV2: 6 görselleştirme
- EfficientNetB0: 6 görselleştirme

### 5_raporlar/
- JSON formatında metrikler
- Model history logs
- Predictions (numpy arrays)

### 6_outputs_zip/
- Tüm çıktıların ZIP arşivi

---

## 🚀 Notebook Çalıştırma

### Google Colab:
```python
# Dataset download
import kagglehub
dataset_path = kagglehub.dataset_download('nodoubttome/skin-cancer9-classesisic')

# Notebook'u çalıştır
# Runtime > Run All
```

### Lokal:
```bash
# Setup
make setup

# GPU kontrolü
make gpu-check

# Notebook aç
make notebook
```

---

## 🎓 Sonuç

Bu proje, derin öğrenme pipeline'ını baştan sona uyguladı:
- ✅ Veri hazırlığı ve augmentation
- ✅ Scratch CNN eğitimi
- ✅ Transfer learning (freeze + finetune)
- ✅ Kapsamlı metrik analizi
- ✅ Grad-CAM (tüm modeller için başarıyla tamamlandı)

**Önerilen İyileştirmeler:**
- Transfer learning modellerinde preprocessing'i pipeline'da yapmak
- 100 epoch tam eğitim
- Daha fazla data augmentation
- Ensemble modeller

---

**Teşekkürler!** 🙏
