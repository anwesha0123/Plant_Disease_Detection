# Plant_Disease_Detection

This project uses a Convolutional Neural Network (CNN) to identify plant diseases from leaf images. The model was trained on a **manually created dataset** collected using the `bing-image-downloader`.

---

## 🎯 Objective

To develop a deep learning-based image classification model that can accurately detect and classify diseases in plant leaves, enabling early diagnosis and management.

---

## 📦 Dataset

- ✅ **Manually Created**
- 🖼️ **Categories:** Healthy & Diseased leaves
- 📁 Stored in train/test folders structured for image classification.

---

## 📊 Model Info

- ✅ Transfer Learning using VGG 16 architecture
- 📈 Achieved **86% validation accuracy**

---

## 🧪 How to Run the Model Notebook

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/plant-disease-detection.git
   cd plant-disease-detection

2. **Launch the notebook**
   ```bash
   jupyter notebook ML_Model_1_0.ipynb

3. **Ensure dependencies are installed**
   ```bash
   pip install streamlit tensorflow keras pillow

4. **Run the Streamlit app**
   ```bash  
   
   streamlit run disease_detection.py

