import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import datetime as dt
import sys
import os

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our model classes
from stock_price_predictor import StockPricePredictor
from spam_email_classifier import SpamEmailClassifier

class MLProjectsUI:
    def __init__(self, root):
        self.root = root
        self.root.title("ML Projects Demo")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Initialize models
        self.stock_predictor = StockPricePredictor()
        self.spam_classifier = SpamEmailClassifier()
        
        # Create notebook (tabs)
        self.notebook = ttk.Notebook(root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Create tabs
        self.stock_tab = ttk.Frame(self.notebook)
        self.spam_tab = ttk.Frame(self.notebook)
        
        self.notebook.add(self.stock_tab, text="Stock Price Predictor")
        self.notebook.add(self.spam_tab, text="Spam Email Classifier")
        
        # Setup each tab
        self.setup_stock_tab()
        self.setup_spam_tab()
    
    def setup_stock_tab(self):
        # Create frames
        input_frame = ttk.LabelFrame(self.stock_tab, text="Input Parameters")
        input_frame.pack(fill="x", expand=False, padx=10, pady=10)
        
        output_frame = ttk.LabelFrame(self.stock_tab, text="Results")
        output_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Input widgets
        ttk.Label(input_frame, text="Stock Ticker:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        self.ticker_var = tk.StringVar(value="AAPL")
        ttk.Entry(input_frame, textvariable=self.ticker_var, width=10).grid(row=0, column=1, padx=5, pady=5, sticky="w")
        
        ttk.Label(input_frame, text="Days of Historical Data:").grid(row=0, column=2, padx=5, pady=5, sticky="w")
        self.days_var = tk.IntVar(value=365)
        ttk.Entry(input_frame, textvariable=self.days_var, width=10).grid(row=0, column=3, padx=5, pady=5, sticky="w")
        
        ttk.Label(input_frame, text="Prediction Days:").grid(row=0, column=4, padx=5, pady=5, sticky="w")
        self.pred_days_var = tk.IntVar(value=30)
        ttk.Entry(input_frame, textvariable=self.pred_days_var, width=10).grid(row=0, column=5, padx=5, pady=5, sticky="w")
        
        # Buttons
        ttk.Button(input_frame, text="Run Prediction", command=self.run_stock_prediction).grid(row=1, column=0, columnspan=6, padx=5, pady=5)
        
        # Output widgets
        self.stock_output = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10)
        self.stock_output.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Frame for matplotlib figure
        self.stock_plot_frame = ttk.Frame(output_frame)
        self.stock_plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    def setup_spam_tab(self):
        # Create frames
        input_frame = ttk.LabelFrame(self.spam_tab, text="Email Input")
        input_frame.pack(fill="x", expand=False, padx=10, pady=10)
        
        output_frame = ttk.LabelFrame(self.spam_tab, text="Classification Result")
        output_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Input widgets
        ttk.Label(input_frame, text="Enter Email Text:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
        
        self.email_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=6)
        self.email_text.grid(row=1, column=0, columnspan=2, padx=5, pady=5, sticky="we")
        
        # Sample emails dropdown
        ttk.Label(input_frame, text="Or Select Sample:").grid(row=2, column=0, padx=5, pady=5, sticky="w")
        
        self.sample_emails = [
            "Hi John, can we schedule a meeting for tomorrow at 2 PM?",
            "Please find attached the report you requested yesterday.",
            "Congratulations! You've won a $1000 gift card. Click here to claim now!",
            "URGENT: Your account has been compromised. Verify your details immediately.",
            "FREE VIAGRA for you! Limited time offer, buy now!"
        ]
        
        self.sample_var = tk.StringVar()
        sample_dropdown = ttk.Combobox(input_frame, textvariable=self.sample_var, values=self.sample_emails, width=50)
        sample_dropdown.grid(row=2, column=1, padx=5, pady=5, sticky="we")
        sample_dropdown.bind("<<ComboboxSelected>>", self.load_sample_email)
        
        # Buttons
        ttk.Button(input_frame, text="Classify Email", command=self.classify_email).grid(row=3, column=0, columnspan=2, padx=5, pady=5)
        
        # Output widgets
        self.spam_output = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=10)
        self.spam_output.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Frame for matplotlib figure
        self.spam_plot_frame = ttk.Frame(output_frame)
        self.spam_plot_frame.pack(fill="both", expand=True, padx=5, pady=5)
    
    def load_sample_email(self, event):
        self.email_text.delete(1.0, tk.END)
        self.email_text.insert(tk.END, self.sample_var.get())
    
    def run_stock_prediction(self):
        # Clear previous output
        self.stock_output.delete(1.0, tk.END)
        self.stock_output.insert(tk.END, "Running stock price prediction...\n")
        self.root.update_idletasks()
        
        # Get input values
        ticker = self.ticker_var.get().strip().upper()
        days = self.days_var.get()
        pred_days = self.pred_days_var.get()
        
        # Validate inputs
        if not ticker:
            messagebox.showerror("Error", "Please enter a stock ticker symbol")
            return
        
        # Run prediction in a separate thread to avoid freezing the UI
        threading.Thread(target=self._run_stock_prediction_thread, 
                         args=(ticker, days, pred_days)).start()
    
    def _run_stock_prediction_thread(self, ticker, days, pred_days):
        try:
            # Set date range
            end_date = dt.datetime.now().strftime('%Y-%m-%d')
            start_date = (dt.datetime.now() - dt.timedelta(days=days)).strftime('%Y-%m-%d')
            
            # Update UI
            self._update_stock_output(f"Fetching data for {ticker} from {start_date} to {end_date}...\n")
            
            # Fetch data
            data = self.stock_predictor.fetch_data(ticker, start_date, end_date)
            
            # Preprocess data
            self._update_stock_output("Preprocessing data...\n")
            X_train, X_test, y_train, y_test = self.stock_predictor.preprocess_data()
            
            # Train model
            self._update_stock_output("Training model...\n")
            model = self.stock_predictor.train_model()
            
            # Evaluate model
            self._update_stock_output("Evaluating model...\n")
            mse, rmse, r2 = self.stock_predictor.evaluate_model()
            
            # Predict future price
            future_price = self.stock_predictor.predict_future(days=pred_days)
            
            # Display results
            self._update_stock_output(f"\nResults for {ticker}:\n")
            self._update_stock_output(f"Mean Squared Error: {mse:.4f}\n")
            self._update_stock_output(f"Root Mean Squared Error: {rmse:.4f}\n")
            self._update_stock_output(f"R² Score: {r2:.4f}\n")
            self._update_stock_output(f"\nPredicted price after {pred_days} days: ${future_price:.2f}\n")
            
            # Create and display plot
            self.root.after(0, self._create_stock_plot)
            
        except Exception as e:
            self._update_stock_output(f"Error: {str(e)}\n")
    
    def _update_stock_output(self, text):
        # Update the output text widget from the main thread
        self.root.after(0, lambda: self.stock_output.insert(tk.END, text))
        self.root.after(0, lambda: self.stock_output.see(tk.END))
    
    def _create_stock_plot(self):
        # Clear previous plot
        for widget in self.stock_plot_frame.winfo_children():
            widget.destroy()
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 4))
        
        # Get predictions
        y_pred = self.stock_predictor.model.predict(self.stock_predictor.X_test)
        
        # Plot actual vs predicted
        ax.plot(self.stock_predictor.y_test, label='Actual', color='blue')
        ax.plot(y_pred, label='Predicted', color='red')
        ax.set_title('Stock Price Prediction')
        ax.set_xlabel('Test Data Points')
        ax.set_ylabel('Stock Price')
        ax.legend()
        ax.grid(True)
        
        # Embed plot in tkinter
        canvas = FigureCanvasTkAgg(fig, master=self.stock_plot_frame)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def classify_email(self):
        # Clear previous output
        self.spam_output.delete(1.0, tk.END)
        self.spam_output.insert(tk.END, "Classifying email...\n")
        self.root.update_idletasks()
        
        # Get email text
        email_text = self.email_text.get(1.0, tk.END).strip()
        
        # Validate input
        if not email_text:
            messagebox.showerror("Error", "Please enter email text")
            return
        
        # Run classification in a separate thread
        threading.Thread(target=self._classify_email_thread, 
                         args=(email_text,)).start()
    
    def _classify_email_thread(self, email_text):
        try:
            # Load sample data and train model if not already trained
            if not hasattr(self.spam_classifier, 'classifier') or not hasattr(self.spam_classifier.classifier, 'classes_'):
                self._update_spam_output("Loading sample data and training model...\n")
                data = self.spam_classifier.load_sample_data()
                X_train, X_test, y_train, y_test = self.spam_classifier.preprocess_data(data)
                self.spam_classifier.train_model(X_train, y_train)
                self.spam_classifier.evaluate_model(X_test, y_test)
            
            # Classify email
            prediction, probability = self.spam_classifier.predict(email_text)
            
            # Display result
            if prediction == 1:
                result = f"SPAM (Probability: {probability:.2f})"
                details = "This email appears to be spam. It contains characteristics commonly found in unsolicited or fraudulent messages."
            else:
                result = f"HAM (Probability: {1-probability:.2f})"
                details = "This email appears to be legitimate. It doesn't contain typical spam characteristics."
            
            self._update_spam_output(f"\nClassification Result: {result}\n\n")
            self._update_spam_output(f"Details: {details}\n\n")
            
            # Show features that contributed to the classification
            self._update_spam_output("Key features in this email:\n")
            
            # Get the preprocessed text
            processed_text = self.spam_classifier.preprocess_text(email_text)
            words = processed_text.split()
            
            # Get the feature names from the vectorizer
            if hasattr(self.spam_classifier.vectorizer, 'get_feature_names_out'):
                feature_names = self.spam_classifier.vectorizer.get_feature_names_out()
            else:
                feature_names = self.spam_classifier.vectorizer.get_feature_names()
            
            # Find words that are in the vectorizer's vocabulary
            important_words = [word for word in words if word in feature_names]
            
            if important_words:
                self._update_spam_output("Important words detected: " + ", ".join(important_words[:10]) + "\n")
            else:
                self._update_spam_output("No significant features detected.\n")
            
        except Exception as e:
            self._update_spam_output(f"Error: {str(e)}\n")
    
    def _update_spam_output(self, text):
        # Update the output text widget from the main thread
        self.root.after(0, lambda: self.spam_output.insert(tk.END, text))
        self.root.after(0, lambda: self.spam_output.see(tk.END))

def main():
    root = tk.Tk()
    app = MLProjectsUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()