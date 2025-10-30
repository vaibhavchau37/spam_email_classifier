# Email Spam Classifier

<div align="center">

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![License](https://img.shields.io/badge/license-MIT-green)

</div>

A machine learning application that classifies emails as spam or legitimate (ham) using Natural Language Processing techniques.

## 📋 Table of Contents

- [Features](#features)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [How It Works](#how-it-works)
- [Screenshots](#screenshots)
- [Future Improvements](#future-improvements)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features

- **Email Classification**: Analyzes email text to determine if it's spam or legitimate
- **Modern UI**: Clean, intuitive interface with real-time feedback
- **Confidence Visualization**: Visual gauge showing classification confidence
- **Sample Emails**: Pre-loaded examples for quick testing
- **Detailed Analysis**: Shows key features that influenced the classification

## 🛠️ Technologies Used

- **Python 3.x**: Core programming language
- **Scikit-learn**: For machine learning algorithms and model training
- **NLTK**: For natural language processing and text preprocessing
- **Tkinter**: For building the graphical user interface
- **Matplotlib**: For data visualization and confidence gauge

## 🚀 Getting Started

### Prerequisites

Make sure you have Python 3.x installed on your system.

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/email-spam-classifier.git
   cd email-spam-classifier
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Application

To run the spam classifier:

```bash
python spam_classifier_ui.py
```

## 🔍 How It Works

1. **Data Processing**: The application uses a machine learning model (Naive Bayes) trained on email data
2. **Text Preprocessing**:
   - Lowercase conversion
   - Punctuation and number removal
   - Stopword removal
   - Stemming (reducing words to their root form)
3. **Classification**: The model analyzes the processed text and provides a classification with confidence score
4. **Visualization**: Results are displayed both textually and visually

## 📸 Screenshots

<img width="886" height="710" alt="image" src="https://github.com/user-attachments/assets/ad57b4f6-3aa4-4d9c-9a7b-53d0ec0de88e" />

## 🔮 Future Improvements

- Add ability to load emails from files
- Implement additional machine learning models
- Add support for training on custom datasets
- Create a web-based version

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.#
