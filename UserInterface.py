import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow import keras
import numpy as np
from keras.models import load_model
import cv2

class CIFAR10Classifier:
    def __init__(self, root):
        """Initialize the GUI window and load the pre-trained model."""
        self.root = root
        self.root.title("Image Classifier - Multiple Classifications")
        self.root.geometry("700x900")
        self.root.configure(bg='#f0f0f0')
        
        try:
            self.model = load_model('m_s1.h5')
            print("Model loaded successfully!")
        except Exception as e:
            messagebox.showerror("Error", f"Could not load model: {str(e)}")
            return
        
        self.class_labels = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                            'dog', 'frog', 'horse', 'ship', 'truck']
        
        self.classification_count = 0  
        self.setup_ui()  
        
    def setup_ui(self):
        """Create and arrange all GUI widgets."""

        title_label = tk.Label(self.root, text="CIFAR-10 Image Classifier", 
                              font=("Arial", 20, "bold"), 
                              bg='#f0f0f0', fg='#333')
        title_label.pack(pady=10)

        self.status_label = tk.Label(self.root, text="Ready to classify images", 
                                    font=("Arial", 12), 
                                    bg='#f0f0f0', fg='#666')
        self.status_label.pack()

        self.counter_label = tk.Label(self.root, text="Classifications performed: 0", 
                                     font=("Arial", 10, "italic"), 
                                     bg='#f0f0f0', fg='#888')
        self.counter_label.pack(pady=2)

        upload_frame = tk.Frame(self.root, bg='#f0f0f0')
        upload_frame.pack(pady=5)
        
        self.upload_btn = tk.Button(upload_frame, text="Upload New Image", 
                                   command=self.upload_image,
                                   font=("Arial", 12, "bold"),
                                   bg='#4CAF50', fg='white',
                                   padx=20, pady=10,
                                   cursor='hand2')
        self.upload_btn.pack()
        
        # Fixed: Create and pack the image frame
        self.image_frame = tk.Frame(self.root, bg='#f0f0f0')
        self.image_frame.pack(pady=5)

        self.image_label = tk.Label(self.image_frame, text="No image selected\nUpload an image to start classifying", 
                                   bg='#f0f0f0', fg='#666',
                                   font=("Arial", 12),
                                   justify='center')
        self.image_label.pack()
        
        self.predict_btn = tk.Button(self.root, text="Classify This Image", 
                                    command=self.predict_image,
                                    font=("Arial", 12, "bold"),
                                    bg='#2196F3', fg='white',
                                    padx=20, pady=10,
                                    cursor='hand2')

        self.action_frame = tk.Frame(self.root, bg='#f0f0f0')
        
        self.classify_another_btn = tk.Button(self.action_frame, text="Classify Another Image", 
                                             command=self.reset_for_new_classification,
                                             font=("Arial", 10, "bold"),
                                             bg='#FF9800', fg='white',
                                             padx=15, pady=5,
                                             cursor='hand2')
        
        self.clear_all_btn = tk.Button(self.action_frame, text="Clear All Results", 
                                      command=self.clear_all_results,
                                      font=("Arial", 10),
                                      bg='#f44336', fg='white',
                                      padx=15, pady=5,
                                      cursor='hand2')
        
        self.results_frame = tk.Frame(self.root, bg='#f0f0f0')
        
        self.result_label = tk.Label(self.results_frame, text="", 
                                    font=("Arial", 16, "bold"),
                                    bg='#f0f0f0', fg='#2196F3')
        
        self.confidence_label = tk.Label(self.results_frame, text="", 
                                        font=("Arial", 12),
                                        bg='#f0f0f0', fg='#666')
        
        self.predictions_frame = tk.Frame(self.results_frame, bg='#f0f0f0')
        
        predictions_title = tk.Label(self.predictions_frame, text="All Predictions:", 
                                    font=("Arial", 12, "bold"),
                                    bg='#f0f0f0', fg='#333')
        predictions_title.pack(anchor='w')
        
        listbox_frame = tk.Frame(self.predictions_frame, bg='#f0f0f0')
        listbox_frame.pack(fill='both', expand=True)
        
        self.predictions_listbox = tk.Listbox(listbox_frame, height=10, 
                                             font=("Arial", 9),
                                             selectmode=tk.SINGLE)
        self.predictions_listbox.pack(side='left', fill='both', expand=True)
        
        scrollbar = tk.Scrollbar(listbox_frame, orient='vertical')
        scrollbar.pack(side='right', fill='y')
        
        self.predictions_listbox.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.predictions_listbox.yview)
        
        self.current_image = None  

    def upload_image(self):
        """Handle image upload and display it in the GUI."""
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.gif *.tiff"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            try:
                self.display_image(file_path)
                self.current_image = file_path
                self.clear_results()
                self.predict_btn.pack(pady=5)
                self.predict_btn.configure(text="Classify This Image", state='normal')
                self.status_label.configure(text="Image loaded. Ready to classify!")
                self.upload_btn.configure(text="Upload Different Image")
                self.action_frame.pack_forget()
            except Exception as e:
                messagebox.showerror("Error", f"Could not load image: {str(e)}")
    
    def display_image(self, file_path):
        """Display the uploaded image in the GUI."""
        image = Image.open(file_path)
        display_size = (200, 200)
        image.thumbnail(display_size, Image.Resampling.LANCZOS)
        photo = ImageTk.PhotoImage(image)
        self.image_label.configure(image=photo, text="")
        self.image_label.image = photo
    
    def preprocess_image(self, file_path):
        """Preprocess the image for model input."""
        image = cv2.imread(file_path)
        if image is None:
            raise ValueError("Could not load image")
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (32, 32))
        image = image.astype('float32') / 255.0
        image = np.expand_dims(image, axis=0)
        return image
    
    def predict_image(self):
        """Classify the uploaded image and display results."""
        if not self.current_image:
            messagebox.showwarning("Warning", "Please upload an image first")
            return
        
        try:
            self.predict_btn.configure(text="Classifying...", state='disabled')
            self.status_label.configure(text="Analyzing image...")
            self.root.update()
            
            processed_image = self.preprocess_image(self.current_image)
            predictions = self.model.predict(processed_image)
            
            predicted_class_idx = np.argmax(predictions[0])
            predicted_class = self.class_labels[predicted_class_idx]
            confidence = predictions[0][predicted_class_idx] * 100
            
            self.classification_count += 1
            self.counter_label.configure(text=f"Classifications performed: {self.classification_count}")
            
            self.results_frame.pack(pady=5, fill='both', expand=True, padx=20)
            self.result_label.configure(text=f"Prediction: {predicted_class.upper()}")
            self.result_label.pack(pady=5)
            
            self.confidence_label.configure(text=f"Confidence: {confidence:.2f}%")
            self.confidence_label.pack(pady=2)
            
            self.show_all_predictions(predictions[0])
            
            self.action_frame.pack(pady=5)
            self.classify_another_btn.pack(side='left', padx=5)
            self.clear_all_btn.pack(side='left', padx=5)
            
            self.predict_btn.configure(text="Classify This Image Again", state='normal')
            self.status_label.configure(text="Classification complete! Upload another image or classify again.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
            self.predict_btn.configure(text="Classify This Image", state='normal')
            self.status_label.configure(text="Classification failed. Try again.")
    
    def show_all_predictions(self, predictions):
        """Display all prediction percentages in the listbox."""
        self.predictions_listbox.delete(0, tk.END)
        
        pred_with_idx = [(predictions[i], i, self.class_labels[i]) 
                        for i in range(len(predictions))]
        
        pred_with_idx.sort(reverse=True)
        
        for confidence, idx, class_name in pred_with_idx:
            percentage = confidence * 100
            self.predictions_listbox.insert(tk.END, 
                f"{class_name.capitalize()}: {percentage:.2f}%")
        
        self.predictions_frame.pack(pady=5, fill='both', expand=True)
    
    def reset_for_new_classification(self):
        """Reset the interface for a new classification."""
        self.current_image = None
        self.image_label.configure(image="", text="Upload a new image to classify")
        self.image_label.image = None
        self.predict_btn.pack_forget()
        self.upload_btn.configure(text="Upload New Image")
        self.status_label.configure(text="Ready for next classification")
        self.action_frame.pack_forget()
        self.clear_results()
    
    def clear_results(self):
        """Clear the current results without resetting the counter."""
        self.results_frame.pack_forget()
        self.action_frame.pack_forget()
    
    def clear_all_results(self):
        """Clear all results and reset the counter."""
        self.clear_results()
        self.classification_count = 0
        self.counter_label.configure(text="Classifications performed: 0")
        self.status_label.configure(text="All results cleared. Ready to start fresh.")

def main():
    """Run the application."""
    root = tk.Tk()
    app = CIFAR10Classifier(root)
    def on_closing():
        if messagebox.askokcancel("Quit", "Do you want to quit the classifier?"):
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()

if __name__ == "__main__":
    main()