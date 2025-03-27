import os
# Set OpenMP environment variable to avoid runtime conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import tensorflow as tf
import sounddevice as sd
import librosa
import time
from datetime import datetime

# Constants
SAMPLE_RATE = 16000
DURATION = 2.5  # seconds
N_MELS = 128
HOP_LENGTH = 128
N_FFT = 256
CHANNELS = 1

def create_mel_spectrogram(audio):
    """Create mel-spectrogram from audio data."""
    # Create mel-spectrogram
    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH,
        n_fft=N_FFT
    )
    
    # Convert to log scale
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Normalize
    mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
    
    return mel_spec_norm

def audio_callback(indata, frames, time, status):
    """Callback function for audio recording."""
    if status:
        print(f"Status: {status}")
    
    # Convert audio data to the correct format
    audio_data = indata.flatten()
    
    # Create mel-spectrogram
    mel_spec = create_mel_spectrogram(audio_data)
    
    # Reshape for model input
    mel_spec = mel_spec.reshape(1, mel_spec.shape[0], mel_spec.shape[1], 1)
    
    # Make prediction
    prediction = model.predict(mel_spec, verbose=0)
    predicted_class = class_names[np.argmax(prediction)]
    confidence = np.max(prediction)
    
    # Log detection if confidence is above threshold
    if confidence > 0.7:  # Adjust threshold as needed
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_message = f"{timestamp} - Detected: {predicted_class} (Confidence: {confidence:.2f})"
        print(log_message)
        
        # Save to log file
        with open('detection_log.txt', 'a') as f:
            f.write(log_message + '\n')

def main():
    global model, class_names
    
    # Load the trained model and class names
    print("Loading model and class names...")
    model = tf.keras.models.load_model('pest_detection_model.h5')
    class_names = np.load('class_names.npy', allow_pickle=True)
    
    print("\nStarting real-time pest detection...")
    print("Press Ctrl+C to stop")
    
    # Create log file if it doesn't exist
    if not os.path.exists('detection_log.txt'):
        with open('detection_log.txt', 'w') as f:
            f.write("Pest Detection Log\n")
            f.write("=================\n\n")
    
    try:
        # Start audio stream
        with sd.InputStream(samplerate=SAMPLE_RATE,
                          channels=CHANNELS,
                          callback=audio_callback,
                          blocksize=int(SAMPLE_RATE * DURATION)):
            while True:
                time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nStopping pest detection...")
    finally:
        sd.stop()

if __name__ == "__main__":
    main() 