import os
# Set OpenMP environment variable to avoid runtime conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import librosa
import tensorflow as tf
from datetime import datetime
import warnings
from scipy import signal
import time

# Filter warnings
warnings.filterwarnings('ignore')

# Constants - must match training parameters
SAMPLE_RATE = 16000
DURATION = 2.5
N_MELS = 64
HOP_LENGTH = 256
N_FFT = 1024
CONFIDENCE_THRESHOLD = 0.30  # Lower the threshold a bit more

# Enhanced pest information dictionary with frequency ranges and characteristics
PEST_INFO = {
    "Asian tiger mosquito": {
        "scientific_name": "Aedes albopictus",
        "description": "Invasive mosquito species with distinctive black and white stripes.",
        "impact": "Vector for diseases including dengue, chikungunya, and Zika viruses.",
        "frequency_range": "400-800 Hz",
        "sound_pattern": "High-pitched whining sound with frequency of 500-700 Hz"
    },
    "Caribbean fruit fly": {
        "scientific_name": "Anastrepha suspensa",
        "description": "Yellow-brown fruit fly with dark wing markings.",
        "impact": "Damages citrus and tropical fruits.",
        "frequency_range": "200-450 Hz",
        "sound_pattern": "Rapid wing-beat frequencies around 350 Hz"
    },
    "Fire ants": {
        "scientific_name": "Solenopsis invicta",
        "description": "Aggressive red ants that build large mound nests.",
        "impact": "Damage crops, electrical equipment, and deliver painful stings.",
        "frequency_range": "100-300 Hz",
        "sound_pattern": "Low frequency stridulation sounds"
    },
    "Termites": {
        "scientific_name": "Various species",
        "description": "Social insects that feed on wood and plant material.",
        "impact": "Cause structural damage to buildings and wooden structures.",
        "frequency_range": "50-250 Hz",
        "sound_pattern": "Head-banging sounds when alarmed, clicking and rustling"
    },
    "Asian longhorned beetle": {
        "scientific_name": "Anoplophora glabripennis",
        "description": "Large black beetle with white spots and long antennae.",
        "impact": "Invasive pest that kills hardwood trees by boring into the wood.",
        "frequency_range": "65-220 Hz",
        "sound_pattern": "Gnawing and chewing sounds when larvae feed on wood"
    },
    "Black vine weevil": {
        "scientific_name": "Otiorhynchus sulcatus",
        "description": "Black beetle with pear-shaped body and short snout.",
        "impact": "Damages ornamental plants and small fruits; larvae feed on roots.",
        "frequency_range": "80-250 Hz",
        "sound_pattern": "Quiet clicking and scraping sounds"
    },
    "Mediterranean fruit fly": {
        "scientific_name": "Ceratitis capitata",
        "description": "Fruit fly with yellowish body and dark markings on wings.",
        "impact": "Attacks over 250 types of fruits and vegetables.",
        "frequency_range": "200-600 Hz",
        "sound_pattern": "Wing-beat frequencies around 300-400 Hz"
    },
    "Butterfly": {
        "scientific_name": "Various species",
        "description": "Insects with large, often colorful wings and slender bodies.",
        "impact": "Mostly beneficial as pollinators, but some species can damage crops in larval stage.",
        "frequency_range": "30-100 Hz",
        "sound_pattern": "Mostly silent, occasional low frequency wing movements"
    },
    "June beetle": {
        "scientific_name": "Phyllophaga spp.",
        "description": "Medium to large beetles with reddish-brown coloration.",
        "impact": "Adults feed on leaves of trees and shrubs; larvae (white grubs) damage roots of grasses.",
        "frequency_range": "70-150 Hz",
        "sound_pattern": "Buzzing flight sound around 100 Hz"
    },
    "Queensland fruit fly": {
        "scientific_name": "Bactrocera tryoni",
        "description": "Reddish-brown fruit fly with yellow markings.",
        "impact": "Major pest of fruit crops in Australia and Pacific islands.",
        "frequency_range": "200-400 Hz",
        "sound_pattern": "Wing vibrations around 300 Hz"
    }
}

# Default information for pests not in the dictionary
DEFAULT_PEST_INFO = {
    "scientific_name": "Not available",
    "description": "A pest species detected by sound analysis.",
    "impact": "May cause damage to agricultural or structural systems.",
    "frequency_range": "Unknown",
    "sound_pattern": "Specific sound pattern not documented"
}

def denoise_audio(audio):
    """Apply simple noise reduction technique with error handling."""
    try:
        # Calculate noise threshold (assuming first 0.1s is background noise)
        noise_sample = audio[:int(SAMPLE_RATE * 0.1)]
        if len(noise_sample) > 0:  # Check if the slice has values
            noise_threshold = np.mean(np.abs(noise_sample)) * 2
            
            # Apply soft thresholding
            denoised = np.copy(audio)
            denoised[np.abs(denoised) < noise_threshold] *= 0.1
            return denoised
        return audio  # Return original if empty slice
    except:
        return audio  # Return original if any error occurs

def process_audio(file_path):
    """Process audio file to mel spectrogram with enhanced preprocessing."""
    try:
        # Use librosa directly for consistency with training
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Apply basic error checking
        if audio is None or len(audio) == 0:
            print(f"Warning: Empty audio data from {file_path}")
            return None, None, None
        
        # Apply noise reduction with error handling
        audio = denoise_audio(audio)
        
        # Pad or truncate to fixed duration
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
        # Check for NaN or Inf values
        if np.isnan(audio).any() or np.isinf(audio).any():
            print(f"Warning: Audio contains NaN or Inf values: {file_path}")
            # Replace NaN/Inf with zeros
            audio = np.nan_to_num(audio)
        
        # Compute mel spectrogram with consistent parameters
        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=SAMPLE_RATE,
            n_mels=N_MELS,
            hop_length=HOP_LENGTH,
            n_fft=N_FFT,
            fmin=50,
            fmax=8000
        )
        
        # Convert to log scale with error handling
        if np.all(mel_spec == 0):
            print(f"Warning: All zero mel spectrogram for {file_path}")
            # Create a small random spectrogram instead of all zeros
            mel_spec_norm = np.random.uniform(0, 0.01, (N_MELS, mel_spec.shape[1]))
            return mel_spec_norm, audio, 0  # Return 0 as dominant frequency
        
        # Convert to log scale
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        if mel_spec_db.max() != mel_spec_db.min():
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        else:
            mel_spec_norm = np.zeros_like(mel_spec_db)
        
        # Compute dominant frequency with error handling
        try:
            dominant_freq = extract_dominant_frequency(audio, SAMPLE_RATE)
        except:
            print(f"Warning: Could not extract dominant frequency for {file_path}")
            dominant_freq = 0
        
        return mel_spec_norm, audio, dominant_freq
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None, None, None

def extract_dominant_frequency(audio, sr):
    """Extract the dominant frequency from audio signal with error handling."""
    try:
        # Use Welch's method to estimate power spectral density
        freqs, psd = signal.welch(audio, sr, nperseg=1024)
        
        # Check for valid PSD
        if len(psd) == 0 or np.all(psd == 0) or np.isnan(psd).any() or np.isinf(psd).any():
            return 0
            
        # Find the frequency with maximum energy
        dominant_freq_idx = np.argmax(psd)
        dominant_freq = freqs[dominant_freq_idx]
        
        return dominant_freq
    except:
        return 0  # Return 0 if extraction fails

def generate_feature_variations(mel_spec):
    """Generate slight variations of the input feature to create an ensemble of predictions."""
    variations = [mel_spec]
    
    try:
        # Add slight noise
        noise_variation = mel_spec.copy()
        noise = np.random.normal(0, 0.01, mel_spec.shape)
        noise_variation = np.clip(noise_variation + noise, 0, 1)
        variations.append(noise_variation)
        
        # Small time shift using numpy roll (safer)
        shift_variation = mel_spec.copy()
        shift = 2
        shift_variation = np.roll(shift_variation, shift, axis=1)
        shift_variation[:, :shift] = 0
        variations.append(shift_variation)
    except:
        # If variations fail, just return the original
        return [mel_spec]
    
    return variations

class PestDetectionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Pest Detection System")
        self.root.geometry("1000x800")
        
        # Define color scheme
        self.colors = {
            'primary': '#2c3e50',
            'secondary': '#3498db',
            'accent': '#e74c3c',
            'success': '#2ecc71',
            'warning': '#f1c40f',
            'background': '#ecf0f1',
            'text': '#2c3e50',
            'button_hover': '#2980b9'
        }
        
        self.root.configure(bg=self.colors['background'])
        
        # Load model and class names
        self.load_model_and_classes()
        
        # Create GUI
        self.create_widgets()
        
        # Bind hover events for buttons
        self.bind_hover_events()
    
    def bind_hover_events(self):
        """Add hover effects to buttons"""
        def on_enter(e):
            e.widget['background'] = self.colors['button_hover']
        
        def on_leave(e):
            e.widget['background'] = self.colors['secondary']
        
        for widget in self.root.winfo_children():
            if isinstance(widget, tk.Button):
                widget.bind("<Enter>", on_enter)
                widget.bind("<Leave>", on_leave)
    
    def create_widgets(self):
        """Create GUI widgets with enhanced styling."""
        # Header with gradient effect
        header_frame = tk.Frame(self.root, bg=self.colors['primary'], height=100)
        header_frame.pack(fill=tk.X)
        
        title_label = tk.Label(
            header_frame, 
            text="Pest Detection System",
            font=('Helvetica', 28, 'bold'),
            fg='white',
            bg=self.colors['primary']
        )
        title_label.pack(pady=20)
        
        # Main content frame
        main_frame = tk.Frame(self.root, bg=self.colors['background'])
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        # File selection frame
        file_frame = tk.Frame(main_frame, bg=self.colors['background'])
        file_frame.pack(fill=tk.X, pady=10)
        
        self.file_label = tk.Label(
            file_frame, 
            text="No file selected",
            font=('Helvetica', 12),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        self.file_label.pack(side=tk.LEFT, padx=5)
        
        browse_button = tk.Button(
            file_frame, 
            text="Browse File",
            command=self.browse_file,
            font=('Helvetica', 12, 'bold'),
            bg=self.colors['secondary'],
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor='hand2'
        )
        browse_button.pack(side=tk.RIGHT, padx=5)
        
        # Control buttons frame
        control_frame = tk.Frame(main_frame, bg=self.colors['background'])
        control_frame.pack(fill=tk.X, pady=10)
        
        analyze_button = tk.Button(
            control_frame, 
            text="🔍 Analyze Audio",
            command=self.analyze_audio,
            font=('Helvetica', 12, 'bold'),
            bg=self.colors['success'],
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor='hand2'
        )
        analyze_button.pack(side=tk.LEFT, padx=5)
        
        play_button = tk.Button(
            control_frame, 
            text="▶ Play Audio",
            command=self.play_audio,
            font=('Helvetica', 12, 'bold'),
            bg=self.colors['secondary'],
            fg='white',
            relief=tk.FLAT,
            padx=20,
            pady=8,
            cursor='hand2'
        )
        play_button.pack(side=tk.LEFT, padx=5)
        
        # Visualization frame
        viz_frame = tk.Frame(main_frame, bg='white', bd=1, relief=tk.SOLID)
        viz_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        self.figure, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(12, 4))
        self.figure.patch.set_facecolor(self.colors['background'])
        
        for ax in [self.ax1, self.ax2]:
            ax.set_facecolor('white')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        self.canvas = FigureCanvasTkAgg(self.figure, viz_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Results frame
        results_frame = tk.Frame(main_frame, bg=self.colors['background'])
        results_frame.pack(fill=tk.BOTH, expand=True, pady=10)
        
        results_label = tk.Label(
            results_frame,
            text="Detection Results",
            font=('Helvetica', 14, 'bold'),
            bg=self.colors['background'],
            fg=self.colors['text']
        )
        results_label.pack(anchor='w')
        
        self.results_text = scrolledtext.ScrolledText(
            results_frame,
            font=('Helvetica', 11),
            bg='white',
            height=8,
            wrap=tk.WORD
        )
        self.results_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.results_text.insert(tk.END, "Detection results will appear here...")
        self.results_text.config(state=tk.DISABLED)
        
        # Status bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready")
        status_bar = tk.Label(
            self.root,
            textvariable=self.status_var,
            bg=self.colors['primary'],
            fg='white',
            font=('Helvetica', 10),
            pady=5
        )
        status_bar.pack(side=tk.BOTTOM, fill=tk.X)
    
    def load_model_and_classes(self):
        """Load model and class names."""
        try:
            # Try to load the best model if available
            best_model_path = 'pest_detection_model_best.h5'
            if os.path.exists(best_model_path):
                self.model = tf.keras.models.load_model(best_model_path)
                print("Loaded best model")
            else:
                self.model = tf.keras.models.load_model('pest_detection_model.h5')
                print("Loaded regular model")
                
            self.class_names = np.load('class_names.npy', allow_pickle=True)
            print(f"Loaded model and {len(self.class_names)} classes")
        except Exception as e:
            self.model = None
            self.class_names = None
            print(f"Error loading model: {str(e)}")
    
    def browse_file(self):
        """Open file dialog to select WAV file."""
        filetypes = [("WAV files", "*.wav")]
        file_path = filedialog.askopenfilename(
            title="Select Audio File",
            filetypes=filetypes
        )
        
        if file_path:
            self.audio_path = file_path
            self.file_label.config(text=f"Selected: {os.path.basename(file_path)}")
            self.status_var.set(f"Loaded: {os.path.basename(file_path)}")
            
            # Clear previous results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            self.results_text.config(state=tk.DISABLED)
            
            # Clear visualization
            self.ax1.clear()
            self.ax2.clear()
            self.canvas.draw()
    
    def analyze_audio(self):
        """Analyze the selected audio file with visual feedback."""
        if not hasattr(self, 'audio_path'):
            messagebox.showwarning(
                "Warning",
                "Please select an audio file first.",
                icon='warning'
            )
            return
            
        if self.model is None or self.class_names is None:
            messagebox.showerror("Error", "Model or class names not loaded. Please train the model first.")
            return
        
        try:
            self.status_var.set("⏳ Processing audio...")
            self.root.update()
            
            # Process audio
            start_time = time.time()
            mel_spec, audio, dominant_freq = process_audio(self.audio_path)
            
            if mel_spec is None:
                raise ValueError("Failed to process audio file")
            
            # Update visualizations
            self.ax1.clear()
            self.ax1.plot(audio)
            self.ax1.set_title('Audio Waveform')
            self.ax1.set_xlabel('Sample')
            self.ax1.set_ylabel('Amplitude')
            
            self.ax2.clear()
            self.ax2.imshow(mel_spec, aspect='auto', origin='lower')
            self.ax2.set_title('Mel-spectrogram')
            self.ax2.set_xlabel('Time')
            self.ax2.set_ylabel('Mel Band')
            
            self.figure.tight_layout()
            self.canvas.draw()
            
            # Generate variations for ensemble prediction
            variations = generate_feature_variations(mel_spec)
            
            # Prepare variations for prediction
            variation_inputs = [np.expand_dims(var, axis=[0, -1]) for var in variations]
            
            # Make ensemble predictions with error handling
            all_predictions = []
            for X in variation_inputs:
                try:
                    # Add error handling to prediction
                    if np.isnan(X).any() or np.isinf(X).any():
                        print("Warning: Input contains NaN or Inf values, replacing with zeros")
                        X = np.nan_to_num(X)
                    
                    pred = self.model.predict(X, verbose=0)
                    all_predictions.append(pred[0])
                except Exception as e:
                    print(f"Prediction error with variation: {str(e)}")
                    # Continue with other variations
                    continue
            
            # If all predictions failed, show error
            if len(all_predictions) == 0:
                raise ValueError("All prediction attempts failed")
            
            # Average predictions from all variations
            avg_predictions = np.mean(all_predictions, axis=0)
            
            # Verify if the top prediction meets confidence threshold
            max_conf = np.max(avg_predictions)
            if max_conf < CONFIDENCE_THRESHOLD:
                # If not confident, add warning
                low_confidence = True
            else:
                low_confidence = False
            
            # Get top predictions
            top_indices = np.argsort(avg_predictions)[-3:][::-1]
            top_pests = [(self.class_names[i], float(avg_predictions[i])) for i in top_indices]
            
            # Display results
            self.results_text.config(state=tk.NORMAL)
            self.results_text.delete(1.0, tk.END)
            
            primary_pest, primary_conf = top_pests[0]
            pest_info = PEST_INFO.get(primary_pest, DEFAULT_PEST_INFO)
            
            # Verify dominant frequency matches expected range
            freq_range_text = pest_info['frequency_range']
            if freq_range_text != "Unknown" and dominant_freq > 0:  # Only check if we have a valid frequency
                try:
                    # Parse the frequency range
                    freq_range = freq_range_text.split('-')
                    min_freq = int(freq_range[0])
                    max_freq = int(freq_range[1])
                    
                    # Check if dominant frequency is within expected range
                    freq_match = min_freq <= dominant_freq <= max_freq
                except:
                    freq_match = True  # if parsing fails, don't check
            else:
                freq_match = True
            
            result_text = f"DETECTED PEST: {primary_pest}\n"
            result_text += f"Confidence: {primary_conf:.2f}\n"
            
            # Add warnings if needed
            if low_confidence:
                result_text += "⚠️ WARNING: Low confidence detection\n"
                
                # Suggest alternative approach for very low confidence
                if primary_conf < 0.2:
                    result_text += "⚠️ The confidence is very low. Consider using a different audio sample.\n"
            
            if not freq_match and dominant_freq > 0:
                result_text += f"⚠️ WARNING: Dominant frequency ({dominant_freq:.0f} Hz) outside expected range\n"
                
            if dominant_freq > 0:
                result_text += f"\nDominant Frequency: {dominant_freq:.0f} Hz\n"
            else:
                result_text += "\nDominant Frequency: Not detected\n"
                
            result_text += f"Expected Frequency Range: {pest_info['frequency_range']}\n"
            result_text += f"Sound Pattern: {pest_info.get('sound_pattern', 'Unknown')}\n\n"
            
            result_text += f"Scientific Name: {pest_info['scientific_name']}\n"
            result_text += f"Description: {pest_info['description']}\n"
            result_text += f"Impact: {pest_info['impact']}\n\n"
            
            result_text += "Other possible detections:\n"
            for i, (pest, conf) in enumerate(top_pests[1:], 2):
                pest_info = PEST_INFO.get(pest, DEFAULT_PEST_INFO)
                result_text += f"{i}. {pest} (Confidence: {conf:.2f})\n"
                result_text += f"   Scientific Name: {pest_info['scientific_name']}\n"
            
            process_time = time.time() - start_time
            result_text += f"\nProcessing time: {process_time:.2f} seconds"
            
            self.results_text.insert(tk.END, result_text)
            self.results_text.config(state=tk.DISABLED)
            
            # Log detection
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            filename = os.path.basename(self.audio_path)
            log_message = f"{timestamp} - File: {filename} - Detected: {primary_pest} (Confidence: {primary_conf:.2f})"
            
            with open('detection_log.txt', 'a') as f:
                f.write(log_message + '\n')
            
            # Update status with success animation
            self.status_var.set("✅ Analysis complete!")
            
        except Exception as e:
            self.show_error(f"Error analyzing audio: {str(e)}")
    
    def play_audio(self):
        """Play the selected audio file."""
        if not hasattr(self, 'audio_path'):
            messagebox.showwarning("Warning", "Please select an audio file first.")
            return
        
        try:
            # Use system default audio player
            import platform
            import subprocess
            
            system = platform.system()
            
            if system == 'Windows':
                os.startfile(self.audio_path)
            elif system == 'Darwin':  # macOS
                subprocess.call(('open', self.audio_path))
            else:  # Linux
                subprocess.call(('xdg-open', self.audio_path))
                
            self.status_var.set("Playing audio")
            
        except Exception as e:
            self.show_error(f"Error playing audio: {str(e)}")
    
    def show_error(self, message):
        """Show error message with styled dialog."""
        messagebox.showerror(
            "Error",
            message,
            icon='error'
        )
        self.status_var.set("❌ Error occurred")

def main():
    root = tk.Tk()
    app = PestDetectionApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 