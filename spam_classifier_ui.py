import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import threading
import sys
import os
from PIL import Image, ImageTk

# Add the current directory to the path to import our modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import our model class
from spam_email_classifier import SpamEmailClassifier

class SpamClassifierUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Spam Email Classifier")
        self.root.geometry("900x700")
        self.root.resizable(True, True)
        
        # Set theme colors
        self.bg_color = "#f5f5f5"
        self.primary_color = "#4a6fa5"
        self.secondary_color = "#6b8cae"
        self.accent_color = "#e63946"
        self.text_color = "#333333"
        self.success_color = "#4caf50"
        self.warning_color = "#ff9800"
        
        # Configure style
        self.style = ttk.Style()
        self.style.theme_use('clam')
        
        # Configure styles for various widgets
        self.style.configure('TFrame', background=self.bg_color)
        self.style.configure('TLabelframe', background=self.bg_color)
        self.style.configure('TLabelframe.Label', background=self.bg_color, foreground=self.text_color, font=('Helvetica', 10, 'bold'))
        self.style.configure('TLabel', background=self.bg_color, foreground=self.text_color, font=('Helvetica', 10))
        self.style.configure('TButton', background=self.primary_color, foreground='white', font=('Helvetica', 10, 'bold'))
        self.style.map('TButton', background=[('active', self.secondary_color)])
        
        # Set root background
        self.root.configure(bg=self.bg_color)
        
        # Initialize model
        self.spam_classifier = SpamEmailClassifier()
        
        # Create main frame
        main_frame = ttk.Frame(root, style='TFrame')
        main_frame.pack(fill='both', expand=True, padx=20, pady=20)
        
        # Setup UI components
        self.setup_ui(main_frame)
    
    def setup_ui(self, parent):
        # Header with title
        header_frame = ttk.Frame(parent, style='TFrame')
        header_frame.pack(fill="x", pady=(0, 20))
        
        title_label = ttk.Label(header_frame, text="Email Spam Classifier", 
                               font=('Helvetica', 18, 'bold'), foreground=self.primary_color,
                               background=self.bg_color)
        title_label.pack(pady=10)
        
        subtitle_label = ttk.Label(header_frame, text="Analyze emails to detect spam using machine learning", 
                                  font=('Helvetica', 10), foreground=self.text_color,
                                  background=self.bg_color)
        subtitle_label.pack()
        
        # Create frames
        input_frame = ttk.LabelFrame(parent, text="Email Input")
        input_frame.pack(fill="x", expand=False, padx=10, pady=10)
        
        # Create a frame for the classification result and visualization
        results_container = ttk.Frame(parent, style='TFrame')
        results_container.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Split the results container into two columns
        output_frame = ttk.LabelFrame(results_container, text="Classification Result")
        output_frame.pack(side=tk.LEFT, fill="both", expand=True, padx=(0, 5), pady=10)
        
        visualization_frame = ttk.LabelFrame(results_container, text="Confidence Visualization")
        visualization_frame.pack(side=tk.RIGHT, fill="both", expand=True, padx=(5, 0), pady=10)
        
        # Input widgets with better layout
        input_label = ttk.Label(input_frame, text="Enter Email Text:", font=('Helvetica', 10, 'bold'))
        input_label.grid(row=0, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.email_text = scrolledtext.ScrolledText(input_frame, wrap=tk.WORD, height=6, 
                                                  font=('Helvetica', 10), bg='white',
                                                  fg=self.text_color)
        self.email_text.grid(row=1, column=0, columnspan=3, padx=10, pady=5, sticky="we")
        
        # Sample emails section
        sample_label = ttk.Label(input_frame, text="Or Select Sample:", font=('Helvetica', 10, 'bold'))
        sample_label.grid(row=2, column=0, padx=10, pady=(10, 5), sticky="w")
        
        self.sample_emails = [
            "Hi John, can we schedule a meeting for tomorrow at 2 PM? I need to discuss the project timeline.",
            "Please find attached the report you requested yesterday. Let me know if you need any clarification.",
            "Congratulations! You've won a $1000 gift card. Click here to claim now! Limited time offer!",
            "URGENT: Your account has been compromised. Verify your details immediately by clicking this link.",
            "FREE VIAGRA for you! Limited time offer, buy now! Increase your performance today!"
        ]
        
        self.sample_var = tk.StringVar()
        sample_dropdown = ttk.Combobox(input_frame, textvariable=self.sample_var, 
                                      values=self.sample_emails, width=60,
                                      font=('Helvetica', 10))
        sample_dropdown.grid(row=2, column=1, padx=10, pady=(10, 5), sticky="we")
        sample_dropdown.bind("<<ComboboxSelected>>", self.load_sample_email)
        
        # Button frame for better alignment
        button_frame = ttk.Frame(input_frame, style='TFrame')
        button_frame.grid(row=3, column=0, columnspan=3, padx=10, pady=15)
        
        # Clear button
        clear_button = ttk.Button(button_frame, text="Clear", command=self.clear_input, width=15)
        clear_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Classify button with custom style
        self.classify_button = ttk.Button(button_frame, text="Classify Email", 
                                        command=self.classify_email, width=20)
        self.classify_button.pack(side=tk.LEFT)
        
        # Progress bar (hidden initially)
        self.progress_var = tk.DoubleVar()
        self.progress = ttk.Progressbar(input_frame, orient="horizontal", 
                                      length=200, mode="indeterminate",
                                      variable=self.progress_var)
        self.progress.grid(row=4, column=0, columnspan=3, padx=10, pady=(0, 10), sticky="we")
        self.progress.grid_remove()  # Hide initially
        
        # Output widgets
        self.spam_output = scrolledtext.ScrolledText(output_frame, wrap=tk.WORD, height=15,
                                                   font=('Helvetica', 10), bg='white',
                                                   fg=self.text_color)
        self.spam_output.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Frame for matplotlib figure
        self.visualization_canvas = tk.Canvas(visualization_frame, bg='white', highlightthickness=0)
        self.visualization_canvas.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create initial visualization
        self.create_empty_visualization()
    
    def create_empty_visualization(self):
        # Clear previous visualization
        for widget in self.visualization_canvas.winfo_children():
            widget.destroy()
            
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.fig.patch.set_facecolor('white')
        
        # Create empty gauge chart
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.text(5, 5, "No classification yet", ha='center', va='center', 
                   fontsize=12, color='gray')
        self.ax.axis('off')
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def create_confidence_visualization(self, is_spam, probability):
        # Clear previous visualization
        for widget in self.visualization_canvas.winfo_children():
            widget.destroy()
            
        # Create figure and axis
        self.fig, self.ax = plt.subplots(figsize=(4, 4))
        self.fig.patch.set_facecolor('white')
        
        # Create gauge chart
        self.ax.set_xlim(0, 10)
        self.ax.set_ylim(0, 10)
        self.ax.axis('off')
        
        # Draw gauge background
        background = plt.Circle((5, 5), 4, fill=True, color='#f0f0f0')
        self.ax.add_patch(background)
        
        # Determine color based on classification
        if is_spam:
            color = '#e63946'  # Red for spam
            confidence_text = f"Spam: {probability:.1%}"
        else:
            color = '#4caf50'  # Green for ham
            confidence_text = f"Ham: {1-probability:.1%}"
        
        # Draw confidence arc
        confidence_value = probability if is_spam else 1-probability
        angle = 360 * confidence_value
        confidence_arc = plt.matplotlib.patches.Wedge(
            (5, 5), 4, 0, angle, width=1.5, fill=True, color=color)
        self.ax.add_patch(confidence_arc)
        
        # Add center circle
        center = plt.Circle((5, 5), 2.5, fill=True, color='white')
        self.ax.add_patch(center)
        
        # Add text
        self.ax.text(5, 5, confidence_text, ha='center', va='center', 
                   fontsize=14, fontweight='bold')
        
        classification = "SPAM" if is_spam else "HAM"
        self.ax.text(5, 3, classification, ha='center', va='center', 
                   fontsize=16, fontweight='bold', color=color)
        
        # Embed in tkinter
        canvas = FigureCanvasTkAgg(self.fig, master=self.visualization_canvas)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
    
    def load_sample_email(self, event):
        self.email_text.delete(1.0, tk.END)
        self.email_text.insert(tk.END, self.sample_var.get())
    
    def clear_input(self):
        self.email_text.delete(1.0, tk.END)
        self.sample_var.set("")
        self.spam_output.delete(1.0, tk.END)
        self.create_empty_visualization()
    
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
        
        # Show progress bar
        self.progress.grid()
        self.progress.start(10)
        self.classify_button.state(['disabled'])
        
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
            
            # Display result with styling
            is_spam = prediction == 1
            
            if is_spam:
                result = f"SPAM (Confidence: {probability:.1%})"
                details = "This email appears to be spam. It contains characteristics commonly found in unsolicited or fraudulent messages."
                self._update_spam_output_with_tag("\n⚠️ Classification Result: ", "normal")
                self._update_spam_output_with_tag(f"{result}\n\n", "spam_tag")
            else:
                result = f"HAM (Confidence: {1-probability:.1%})"
                details = "This email appears to be legitimate. It doesn't contain typical spam characteristics."
                self._update_spam_output_with_tag("\n✓ Classification Result: ", "normal")
                self._update_spam_output_with_tag(f"{result}\n\n", "ham_tag")
            
            self._update_spam_output(f"Details: {details}\n\n")
            
            # Show features that contributed to the classification
            self._update_spam_output_with_tag("Key features in this email:\n", "header_tag")
            
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
                self._update_spam_output("Important words detected: ")
                for i, word in enumerate(important_words[:10]):
                    tag = "spam_word" if is_spam else "ham_word"
                    self._update_spam_output_with_tag(word, tag)
                    if i < len(important_words[:10]) - 1:
                        self._update_spam_output(", ")
                self._update_spam_output("\n\n")
            else:
                self._update_spam_output("No significant features detected.\n\n")
            
            # Add tips section
            if is_spam:
                self._update_spam_output_with_tag("Tips to avoid spam:\n", "header_tag")
                self._update_spam_output("• Be cautious of emails with urgent calls to action\n")
                self._update_spam_output("• Avoid clicking on suspicious links\n")
                self._update_spam_output("• Check sender email addresses carefully\n")
                self._update_spam_output("• Be wary of emails with poor grammar or spelling\n")
            else:
                self._update_spam_output_with_tag("This email looks safe, but remember:\n", "header_tag")
                self._update_spam_output("• Always verify unexpected attachments\n")
                self._update_spam_output("• Confirm unusual requests through other channels\n")
                self._update_spam_output("• Keep your spam filter updated\n")
            
            # Update visualization
            self.root.after(0, lambda: self.create_confidence_visualization(is_spam, probability))
            
        except Exception as e:
            self._update_spam_output(f"Error: {str(e)}\n")
        finally:
            # Hide progress bar and enable button
            self.root.after(0, lambda: self.progress.stop())
            self.root.after(0, lambda: self.progress.grid_remove())
            self.root.after(0, lambda: self.classify_button.state(['!disabled']))
    
    def _update_spam_output(self, text):
        # Update the output text widget from the main thread
        self.root.after(0, lambda: self.spam_output.insert(tk.END, text))
        self.root.after(0, lambda: self.spam_output.see(tk.END))
    
    def _update_spam_output_with_tag(self, text, tag):
        # Configure tags if not already done
        if not hasattr(self, 'tags_configured'):
            self.spam_output.tag_configure("spam_tag", foreground="white", background=self.accent_color, 
                                         font=('Helvetica', 10, 'bold'))
            self.spam_output.tag_configure("ham_tag", foreground="white", background=self.success_color, 
                                         font=('Helvetica', 10, 'bold'))
            self.spam_output.tag_configure("header_tag", foreground=self.primary_color, 
                                         font=('Helvetica', 10, 'bold'))
            self.spam_output.tag_configure("spam_word", foreground=self.accent_color, 
                                         font=('Helvetica', 10, 'bold'))
            self.spam_output.tag_configure("ham_word", foreground=self.success_color, 
                                         font=('Helvetica', 10, 'bold'))
            self.tags_configured = True
        
        # Insert text with tag
        current_end = self.spam_output.index(tk.END)
        self.root.after(0, lambda: self.spam_output.insert(tk.END, text))
        new_end = self.spam_output.index(tk.END)
        self.root.after(0, lambda: self.spam_output.tag_add(tag, current_end, f"{new_end} - 1 chars"))
        self.root.after(0, lambda: self.spam_output.see(tk.END))

def main():
    root = tk.Tk()
    app = SpamClassifierUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()