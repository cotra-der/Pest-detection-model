# Pest Detection System

A machine learning-based system for detecting and identifying agricultural pests using sound frequency analysis. The system uses deep learning to analyze audio recordings and identify various pest species based on their characteristic sound patterns.

## Features

- **Real-time Audio Analysis**: Process and analyze audio files in real-time
- **Interactive GUI**: Modern, colorful interface with visual feedback
- **Multi-species Detection**: Identify multiple pest species including:
  - Caribbean fruit fly
  - Fire ants
  - Termites
  - Asian longhorned beetle
  - Black vine weevil
  - Mediterranean fruit fly
  - And more...
- **Detailed Pest Information**: Comprehensive details for each detected pest including:
  - Scientific name
  - Description
  - Impact on agriculture
  - Frequency range
  - Sound pattern characteristics
- **Visual Analysis**: Display of audio waveform and mel-spectrogram
- **Confidence Scoring**: Multiple confidence levels for detections
- **Frequency Analysis**: Dominant frequency detection and range verification
- **Logging System**: Automatic logging of detection results

## Requirements

- Python 3.8+
- TensorFlow 2.x
- librosa
- numpy
- matplotlib
- scipy
- tkinter (usually comes with Python)

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd pest-detection-system
```

2. Install required packages:
```bash
pip install -r requirements.txt
```

## Usage

### Training the Model

1. Run the training script:
```bash
python simple_pest_detection.py
```

This will:
- Load and preprocess the audio dataset
- Train the CNN model
- Save the trained model and class names

### Using the GUI Application

1. Run the prediction interface:
```bash
python simple_predict.py
```

2. Using the interface:
   - Click "Browse File" to select a WAV audio file
   - View the audio waveform and mel-spectrogram visualizations
   - Click "🔍 Analyze Audio" to process the file
   - Click "▶ Play Audio" to listen to the selected file
   - View detailed detection results including:
     - Detected pest species
     - Confidence scores
     - Scientific information
     - Frequency analysis
     - Alternative detections

## GUI Features

- **Modern Design**: Clean, colorful interface with intuitive layout
- **Visual Feedback**: Real-time status updates and progress indicators
- **Interactive Elements**: Hover effects and responsive buttons
- **Results Display**: Scrollable text area with formatted pest information
- **Visualization**: Dual-panel display showing audio waveform and mel-spectrogram
- **Status Bar**: System status and operation feedback

## Technical Details

### Audio Processing
- Sample Rate: 16000 Hz
- Duration: 2.5 seconds
- Mel Bands: 64
- Hop Length: 256
- FFT Size: 1024

### Model Architecture
- CNN-based architecture optimized for audio classification
- Data augmentation for improved robustness
- Ensemble prediction system for better accuracy

### Detection Features
- Confidence threshold: 0.30
- Frequency range verification
- Multiple prediction variations for reliability
- Comprehensive error handling

## Logging

The system automatically logs detection results to `detection_log.txt` with the following information:
- Timestamp
- Audio file name
- Detected pest species
- Confidence score

## Error Handling

The system includes comprehensive error handling for:
- Audio file loading and processing
- Model prediction
- Frequency analysis
- GUI operations
- File system operations

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

