import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import string
import os

class SpamEmailClassifier:
    def __init__(self, classifier_type='naive_bayes'):
        """
        Initialize the spam email classifier
        
        Parameters:
        classifier_type (str): Type of classifier to use ('naive_bayes' or 'svm')
        """
        self.classifier_type = classifier_type
        self.vectorizer = TfidfVectorizer(max_features=5000)
        
        if classifier_type == 'naive_bayes':
            self.classifier = MultinomialNB()
        elif classifier_type == 'svm':
            self.classifier = SVC(kernel='linear', C=1.0, random_state=42)
        else:
            raise ValueError("Invalid classifier type. Choose 'naive_bayes' or 'svm'")
        
        # Download NLTK resources if not already downloaded
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
        
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        
    def load_sample_data(self):
        """
        Create a sample dataset for demonstration purposes
        """
        # Create a sample dataset with some spam and ham emails
        spam_emails = [
            "Congratulations! You've won a $1000 gift card. Click here to claim now!",
            "URGENT: Your account has been compromised. Verify your details immediately.",
            "FREE VIAGRA for you! Limited time offer, buy now!",
            "You have won the lottery! Send your details to claim the prize.",
            "Increase your income by 500% working from home! Click here.",
            "Dear customer, your bank account needs verification. Click the link below.",
            "Amazing investment opportunity! 100% returns guaranteed.",
            "Your PayPal account has been limited. Update your information now.",
            "Congratulations! You are our lucky winner. Claim your prize now!",
            "Get rich quick! Join our program and earn thousands daily."
        ]
        
        ham_emails = [
            "Hi John, can we schedule a meeting for tomorrow at 2 PM?",
            "Please find attached the report you requested yesterday.",
            "The project deadline has been extended to next Friday.",
            "Thank you for your email. I'll get back to you soon.",
            "Reminder: Team lunch tomorrow at 12:30 PM in the cafeteria.",
            "Here are the meeting notes from yesterday's discussion.",
            "Could you please review the document I shared with you?",
            "I'm out of office today. Will respond to your email tomorrow.",
            "The conference call is scheduled for 3 PM today.",
            "Happy birthday! Hope you have a great day."
        ]
        
        # Create labels (1 for spam, 0 for ham)
        emails = spam_emails + ham_emails
        labels = [1] * len(spam_emails) + [0] * len(ham_emails)
        
        # Create a DataFrame
        data = pd.DataFrame({
            'email': emails,
            'label': labels
        })
        
        # Shuffle the data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
        print(f"Sample data created with {len(data)} emails ({len(spam_emails)} spam, {len(ham_emails)} ham)")
        return data
    
    def load_data(self, file_path):
        """
        Load data from a CSV file
        
        Parameters:
        file_path (str): Path to the CSV file
        
        Returns:
        pandas.DataFrame: Loaded data
        """
        if not os.path.exists(file_path):
            print(f"File {file_path} not found. Using sample data instead.")
            return self.load_sample_data()
        
        try:
            data = pd.read_csv(file_path)
            print(f"Data loaded successfully from {file_path}. Shape: {data.shape}")
            return data
        except Exception as e:
            print(f"Error loading data: {e}. Using sample data instead.")
            return self.load_sample_data()
    
    def preprocess_text(self, text):
        """
        Preprocess text by removing punctuation, stopwords, and stemming
        
        Parameters:
        text (str): Text to preprocess
        
        Returns:
        str: Preprocessed text
        """
        # Convert to lowercase
        text = text.lower()
        
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        
        # Remove stopwords and apply stemming
        words = text.split()
        words = [self.stemmer.stem(word) for word in words if word not in self.stop_words]
        
        # Join words back into a string
        return ' '.join(words)
    
    def preprocess_data(self, data, text_column='email', label_column='label'):
        """
        Preprocess the data
        
        Parameters:
        data (pandas.DataFrame): Data to preprocess
        text_column (str): Name of the column containing the email text
        label_column (str): Name of the column containing the labels
        
        Returns:
        tuple: X_train, X_test, y_train, y_test
        """
        # Apply text preprocessing
        data['processed_text'] = data[text_column].apply(self.preprocess_text)
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            data['processed_text'], data[label_column], test_size=0.2, random_state=42)
        
        # Vectorize the text
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        print("Data preprocessing completed.")
        return X_train_vectorized, X_test_vectorized, y_train, y_test
    
    def train_model(self, X_train, y_train):
        """
        Train the classifier
        
        Parameters:
        X_train: Training features
        y_train: Training labels
        
        Returns:
        object: Trained classifier
        """
        print(f"Training {self.classifier_type} classifier...")
        self.classifier.fit(X_train, y_train)
        print("Model trained successfully.")
        return self.classifier
    
    def evaluate_model(self, X_test, y_test):
        """
        Evaluate the classifier
        
        Parameters:
        X_test: Testing features
        y_test: Testing labels
        
        Returns:
        tuple: accuracy, classification_report, confusion_matrix
        """
        # Make predictions
        y_pred = self.classifier.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=['Ham', 'Spam'])
        cm = confusion_matrix(y_test, y_pred)
        
        print(f"Accuracy: {accuracy:.4f}")
        print("\nClassification Report:")
        print(report)
        
        return accuracy, report, cm
    
    def visualize_results(self, cm, y_test, y_pred):
        """
        Visualize the results
        
        Parameters:
        cm: Confusion matrix
        y_test: True labels
        y_pred: Predicted labels
        """
        # Plot confusion matrix
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=['Ham', 'Spam'], yticklabels=['Ham', 'Spam'])
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.title('Confusion Matrix')
        plt.savefig('spam_confusion_matrix.png')
        plt.close()
        
        # Plot classification results
        plt.figure(figsize=(10, 6))
        df_results = pd.DataFrame({
            'True': y_test,
            'Predicted': y_pred
        })
        df_results['Correct'] = df_results['True'] == df_results['Predicted']
        
        sns.countplot(x='True', hue='Correct', data=df_results)
        plt.xticks([0, 1], ['Ham', 'Spam'])
        plt.title('Classification Results')
        plt.xlabel('Email Type')
        plt.ylabel('Count')
        plt.legend(title='Correctly Classified')
        plt.savefig('spam_classification_results.png')
        plt.close()
        
        print("Visualization completed. Check the saved images.")
    
    def predict(self, text):
        """
        Predict whether a new email is spam or ham
        
        Parameters:
        text (str): Email text
        
        Returns:
        int: 1 for spam, 0 for ham
        float: Probability of being spam (for Naive Bayes)
        """
        # Preprocess the text
        processed_text = self.preprocess_text(text)
        
        # Vectorize the text
        vectorized_text = self.vectorizer.transform([processed_text])
        
        # Make prediction
        prediction = self.classifier.predict(vectorized_text)[0]
        
        # Get probability for Naive Bayes
        if self.classifier_type == 'naive_bayes':
            probability = self.classifier.predict_proba(vectorized_text)[0][1]
            return prediction, probability
        else:
            return prediction, None

def main():
    # Create an instance of the classifier
    classifier = SpamEmailClassifier(classifier_type='naive_bayes')
    
    # Load data (will use sample data if file not found)
    data = classifier.load_sample_data()
    
    # Preprocess data
    X_train, X_test, y_train, y_test = classifier.preprocess_data(data)
    
    # Train the model
    model = classifier.train_model(X_train, y_train)
    
    # Evaluate the model
    accuracy, report, cm = classifier.evaluate_model(X_test, y_test)
    
    # Visualize results
    y_pred = model.predict(X_test)
    classifier.visualize_results(cm, y_train, y_pred)
    
    # Test with a new email
    test_email = "Congratulations! You've won a million dollars. Click here to claim your prize now!"
    prediction, probability = classifier.predict(test_email)
    
    if prediction == 1:
        print(f"\nTest email is classified as SPAM with {probability:.2f} probability.")
    else:
        print(f"\nTest email is classified as HAM with {1-probability:.2f} probability.")
    
    print("\nSpam Email Classification completed successfully!")

if __name__ == "__main__":
    main()