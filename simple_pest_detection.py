import os
import numpy as np
import tensorflow as tf
import librosa
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, models, regularizers
import glob
from tqdm import tqdm
import random
from collections import Counter
import warnings

# Filter warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)
random.seed(42)

# Constants
SAMPLE_RATE = 16000
DURATION = 2.5
N_MELS = 64
HOP_LENGTH = 256
N_FFT = 1024
BATCH_SIZE = 8
EPOCHS = 50

def load_and_preprocess_audio(file_path):
    """Load and preprocess audio file."""
    try:
        # Use only librosa for consistency
        audio, sr = librosa.load(file_path, sr=SAMPLE_RATE, duration=DURATION)
        
        # Apply basic noise reduction
        audio = denoise_audio(audio)
        
        # Pad or truncate to fixed duration
        target_length = int(SAMPLE_RATE * DURATION)
        if len(audio) < target_length:
            audio = np.pad(audio, (0, target_length - len(audio)))
        else:
            audio = audio[:target_length]
        
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
        
        # Convert to log scale (use a small offset to avoid log(0))
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        # Normalize
        if mel_spec_db.max() != mel_spec_db.min():
            mel_spec_norm = (mel_spec_db - mel_spec_db.min()) / (mel_spec_db.max() - mel_spec_db.min())
        else:
            mel_spec_norm = np.zeros_like(mel_spec_db)
        
        return mel_spec_norm
    except Exception as e:
        print(f"Error processing {file_path}: {str(e)}")
        return None

def denoise_audio(audio):
    """Apply simple noise reduction technique."""
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

def create_model(input_shape, num_classes):
    """Create a simplified but effective CNN model for small datasets."""
    # Input layer
    inputs = layers.Input(shape=input_shape)
    
    # First convolutional block
    x = layers.Conv2D(16, (3, 3), padding='same', activation='relu')(inputs)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    
    # Second convolutional block
    x = layers.Conv2D(32, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    # Third convolutional block - fewer filters than before
    x = layers.Conv2D(64, (3, 3), padding='same', activation='relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    # Flatten and dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    
    # Output layer
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Create model
    model = models.Model(inputs, outputs)
    
    # Compile with appropriate parameters
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

def augment_spectrogram(spec):
    """Apply safer augmentation to a spectrogram."""
    augmented = spec.copy()
    
    # Random noise 
    if random.random() > 0.3:
        noise_level = random.uniform(0.01, 0.05)
        noise = np.random.normal(0, noise_level, spec.shape)
        augmented = np.clip(augmented + noise, 0, 1)
    
    # Frequency masking - simpler version
    if random.random() > 0.3:
        freq_width = random.randint(1, 4)
        freq_start = random.randint(0, augmented.shape[0] - freq_width)
        augmented[freq_start:freq_start + freq_width, :] = 0
    
    # Time masking - simpler version
    if random.random() > 0.3:
        time_width = random.randint(1, 10)
        time_start = random.randint(0, augmented.shape[1] - time_width)
        augmented[:, time_start:time_start + time_width] = 0
    
    # Vertical shift (simulate pitch shift) - simpler version
    if random.random() > 0.5:
        shift = random.randint(-2, 2)
        if shift > 0:
            # Shift down (lose bottom rows)
            augmented = np.roll(augmented, shift, axis=0)
            augmented[:shift, :] = 0
        elif shift < 0:
            # Shift up (lose top rows)
            augmented = np.roll(augmented, shift, axis=0)
            augmented[shift:, :] = 0
    
    return augmented

def main():
    # Define dataset path
    dataset_path = 'dataset'
    
    # Get all class directories (exclude background noise)
    class_dirs = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d)) 
                   and d != "distinguish insect sounds from background noise"]
    
    print(f"Found {len(class_dirs)} pest classes")
    
    # Create maps for class names and indices
    class_index_map = {name: idx for idx, name in enumerate(sorted(class_dirs))}
    index_class_map = {idx: name for name, idx in class_index_map.items()}
    
    # Save class maps
    np.save('class_names.npy', np.array(list(class_index_map.keys())))
    
    # Process audio files
    X = []  # features
    y = []  # labels
    
    for class_name, class_idx in class_index_map.items():
        class_path = os.path.join(dataset_path, class_name)
        audio_files = glob.glob(os.path.join(class_path, "*.wav"))
        
        print(f"Processing {class_name}: {len(audio_files)} files")
        
        for file_path in tqdm(audio_files, desc=class_name):
            mel_spec = load_and_preprocess_audio(file_path)
            if mel_spec is not None:
                # Add the original spectrogram
                X.append(mel_spec)
                y.append(class_idx)
                
                # Create augmented versions - more for classes with fewer examples
                num_augmentations = max(0, 8 - len(audio_files))
                for _ in range(num_augmentations):
                    augmented = augment_spectrogram(mel_spec)
                    X.append(augmented)
                    y.append(class_idx)
    
    # Convert to numpy arrays
    X = np.array(X)
    y = np.array(y)
    
    # Add channel dimension
    X = np.expand_dims(X, axis=-1)
    
    # Report class distribution after augmentation
    class_counts = Counter(y)
    print("\nClass distribution after augmentation:")
    for class_idx, count in sorted(class_counts.items()):
        print(f"  {index_class_map[class_idx]}: {count} samples")
    
    print(f"\nDataset shape: {X.shape}, Labels shape: {y.shape}")
    
    # Shuffle the data
    indices = np.arange(len(X))
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]
    
    # Simple train-test split (no stratification due to small class sizes)
    split_idx = int(0.8 * len(indices))
    X_train, X_test = X_shuffled[:split_idx], X_shuffled[split_idx:]
    y_train, y_test = y_shuffled[:split_idx], y_shuffled[split_idx:]
    
    print(f"Training set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Create and train model
    input_shape = X_train.shape[1:]  # (height, width, channels)
    num_classes = len(class_index_map)
    
    model = create_model(input_shape, num_classes)
    model.summary()
    
    # Callbacks
    callbacks = [
        # Save best model during training
        tf.keras.callbacks.ModelCheckpoint('pest_detection_model_best.h5', 
                       monitor='val_accuracy', 
                       save_best_only=True, 
                       verbose=1),
        
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15, 
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce learning rate when plateauing
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5, 
            patience=5,
            min_lr=0.00001,
            verbose=1
        )
    ]
    
    # Train model with class weighting
    class_weights = {}
    max_count = max(class_counts.values())
    for class_idx, count in class_counts.items():
        # Assign higher weights to underrepresented classes
        class_weights[class_idx] = max_count / max(count, 1)  # Avoid division by zero
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,  # 20% of training data for validation
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=callbacks,
        class_weight=class_weights
    )
    
    # Load the best model for evaluation
    best_model = tf.keras.models.load_model('pest_detection_model_best.h5')
    
    # Evaluate best model
    test_loss, test_acc = best_model.evaluate(X_test, y_test)
    print(f"Test accuracy: {test_acc:.4f}")
    
    # Save model as the main model file
    best_model.save('pest_detection_model.h5')
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('training_history.png')
    
    # Generate confusion matrix on test data
    y_pred = best_model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    from sklearn.metrics import confusion_matrix, classification_report
    conf_matrix = confusion_matrix(y_test, y_pred_classes)
    
    # Get unique classes in the test data to fix the target_names issue
    unique_classes = sorted(np.unique(np.concatenate([y_test, y_pred_classes])))
    class_names_list = [index_class_map[i] for i in unique_classes]
    
    # Generate classification report with actual labels present in test data
    try:
        report = classification_report(
            y_test, 
            y_pred_classes, 
            labels=unique_classes,
            target_names=class_names_list,
            zero_division=0
        )
        print("\nClassification report:\n", report)
        
        # Save the classification report
        with open('classification_report.txt', 'w') as f:
            f.write(report)
    except Exception as e:
        print(f"Error generating classification report: {str(e)}")
        # Simplified report without target names
        report = classification_report(y_test, y_pred_classes, zero_division=0)
        print("\nSimplified classification report:\n", report)
        
        # Save the simplified report
        with open('classification_report.txt', 'w') as f:
            f.write(report)
    
    print("Training complete.")

if __name__ == "__main__":
    main() 