# Krishi Mitra 🌾

**Krishi Mitra** is an AI-powered agriculture assistant app that provides tools for crop price forecasting, demand analysis, pest & disease detection, and fertilizer recommendations — tailored for Indian farmers in multiple languages.

---

## 🌐 View Live Page

- 👉 [Krishi Mitra](https://sujay-kumar-13.github.io/krishi-mitra-page/)

---

## 🔽 Download APK(for android)

- 👉 [Krishi Mitra](https://github.com/sujay-kumar-13/krishi-mitra/raw/refs/heads/master/flutter/build/app/outputs/flutter-apk/app-release.apk)

---

## 🚀 Features

- 🔍 **Crop Price Trend** – Predicts upcoming crop prices using your trained ML models
- 📈 **Crop Demand Forecast** – Shows expected crop demand based on historical data
- 📸 **Pest & Disease Detection** – Gemini 1.5 Flash-powered image-based diagnosis
- 🧪 **Fertilizer Calculator** – AI-generated NPK recommendations via Gemini API
- 🌐 **Multilingual UI** – Language switching using custom Map-based translation logic (no ARB files)
- 💾 **User Persistence** – Remembers state, language, and preferences locally

---

## 🧑‍💻 Tech Stack

| Layer           | Tech Used                                              |
|-----------------|--------------------------------------------------------|
| Frontend        | Flutter (Dart)                                         |
| Localization    | Manual key-value Map (custom solution)                 |
| Backend         | Flask (Python), hosted on Google Cloud Run             |
| AI/ML APIs      | Gemini 1.5 Flash (Google Generative AI), Random Forest |
| Image Inference | Gemini prompt-based detection                          |
| Persistent Data | SharedPreferences for state + language                 |

---

## 📁 Folder Structure

```
main/
├── backend/              # Flask backend API for handling requests and predictions
├── flutter/              # Flutter mobile application source code
├── language_maps/        # Python scripts and CSV files to generate Dart localization maps
├── model_training/       # ML model training scripts and related files (joblib, datasets, etc.)
└── README.md             # Project overview and folder documentation
```

---

## ⚙️ Setup Instructions

### 1. 🧾 Clone the Repository

```bash
git clone https://github.com/sujay-kumar-13/krishi-mitra.git
cd krishi-mitra
```

---

### 2. 📱 Flutter Frontend

#### 📂 Change Directory

```bash
cd flutter
```

#### 📦 Install Dependencies

```bash
flutter pub get
```

#### ▶️ Run the App

```bash
flutter run
```

> For release build:

```bash
flutter build apk --release
```

---

### 3. 🔥 Flask Backend

#### 📂 Change Directory

```bash
cd backend
```

#### ▶️ Run the Flask Server

```bash
python app.py
```

---

## 🌍 Language Support

- English (default)
- Hindi, Tamil, Telugu, Kannada, Malayalam, Bengali, Gujarati, Marathi, Punjabi, Urdu, Odia

🔁 Translations are handled using a **custom Dart Map-based translation system** loaded from CSV-processed keys.

---

## ☁️ Google Cloud Hosting

- Flask backend is hosted on **Google Cloud Run**
- Endpoint: `https://url-provided-by-google-cloud/`
- CORS enabled for Flutter to access API

---

## 🧠 AI Integration

| Feature               | AI Model         | Input Type   | Output Format |
|-----------------------|------------------|--------------|---------------|
| Fertilizer Calculator | Gemini 1.5 Flash | JSON Prompt  | Clean JSON    |
| Pest & Disease        | Gemini 1.5 Flash | Image Input  | AI Diagnosis  |
| Price & Demand        | Random Forest    | Crop Details | JSON          |

---

## 🙌 Credits

- 🧠 [Gemini Generative AI](https://ai.google.dev/)
- 🐍 Flask on Google Cloud Run
- 📱 Flutter mobile app with responsive layout
- 🧠 Custom multi-language translation using Google Sheets + Dart Maps

---

## 📄 License

MIT License © 2025 Sujay — Building smart solutions for smart farming.