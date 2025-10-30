# Email Spam Classifier

A machine learning application that classifies emails as spam or legitimate (ham) using Natural Language Processing techniques.

## Features

- **Email Classification**: Analyzes email text to determine if it's spam or legitimate
- **Modern UI**: Clean, intuitive interface with real-time feedback
- **Confidence Visualization**: Visual gauge showing classification confidence
- **Sample Emails**: Pre-loaded examples for quick testing
- **Detailed Analysis**: Shows key features that influenced the classification

## Technologies Used

- Python 3.x
- Scikit-learn for machine learning
- NLTK for natural language processing
- Tkinter for the user interface
- Matplotlib for data visualization

## Getting Started

### Prerequisites

Make sure you have Python 3.x installed on your system. You can install the required packages using:

```bash
pip install -r requirements.txt
```

### Running the Application

To run the spam classifier:

```bash
python spam_classifier_ui.py
```

## How It Works

1. The application uses a machine learning model (Naive Bayes) trained on email data
2. Text preprocessing includes:
   - Lowercase conversion
   - Punctuation and number removal
   - Stopword removal
   - Stemming (reducing words to their root form)
3. The model analyzes the processed text and provides a classification with confidence score
4. Results are displayed both textually and visually

## Screenshots

(Screenshots will be added here)

## Future Improvements

- Add ability to load emails from files
- Implement additional machine learning models
- Add support for training on custom datasets
- Create a web-based version

## License

This project is licensed under the MIT License - see the LICENSE file for details.