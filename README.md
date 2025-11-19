# Text-Aligned Speech Tokenization and Embedding (TASTE)

This project is an implementation of a simplified version of the TASTE (Text-Aligned Speech Tokenization and Embedding) model, as described in the paper by Tseng et al. (2025). The primary goal is to use a cross-attention mechanism to align speech features with text tokens and then use these aligned embeddings to reconstruct speech with a pretrained Text-to-Speech (TTS) model, CosyVoice.

This project is submitted as part of the coursework for MDS5122 / AIR5011 at The Chinese University of Hong Kong, Shenzhen.

## File Structure

```
.
├── data/
│   ├── raw_data/         # Raw LibriSpeech audio and text
│   ├── train_data/
│   │   ├── librispeech_train_paired.jsonl
│   │   ├── embeddings/
│   │   └── librispeech_speech_tokenizer/
│   └── test_data/
│       ├── librispeech_test_paired.jsonl
│       ├── embeddings/
│       └── librispeech_speech_tokenizer/
├── model/
│   └── CosyVoice-main/   # Pretrained CosyVoice model and tools
├── results/              # Training results (loss curves, reports)
├── src/
│   ├── Dataset Preparation.py
│   ├── utt2text_and_feature.py
│   ├── s3.sh
│   ├── train and test.py
│   └── example.ipynb
├── Assignment2.pdf       # Assignment description
├── requirements.txt      # Project dependencies
└── README.md             # This file
```

## 1. Environment Setup

To run this project, you need to set up a Python environment with the required dependencies.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/Enddonkey/TASTE-Text-Aligned-Speech-Tokenization-and-Embedding
    cd TASTE-Text-Aligned-Speech-Tokenization-and-Embedding
    ```

2.  **Install dependencies:**
    The required packages are listed in `requirements.txt`. You can install them using pip. It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```


## 2. Workflow and How to Run

This section describes the end-to-end pipeline to reproduce the results.

### Step 1: Data Preparation

1.  **Download LibriSpeech:**
    Download `train-clean-100.tar.gz` and `test-clean.tar.gz` from the official website: [https://www.openslr.org/12](https://www.openslr.org/12).

2.  **Extract the data:**
    Extract the downloaded archives into the `data/raw_data/` directory. The structure should look like this:
    ```
    data/raw_data/
    ├── train-clean-100/
    │   └── LibriSpeech/
    └── test-clean/
        └── LibriSpeech/
    ```

### Step 2: Preprocessing

Run the `Dataset Preparation.py` script to pair audio files with their transcriptions and save them in `.jsonl` format. This script also resamples the audio to 16kHz.

```bash
# For training data
python src/"Dataset Preparation.py" --root_dir "data/raw_data/train-clean-100/LibriSpeech/train-clean-100" --output_paired "data/train_data/librispeech_train_paired.jsonl" --max_samples 3000

# For test data
python src/"Dataset Preparation.py" --root_dir "data/raw_data/test-clean/LibriSpeech/test-clean" --output_paired "data/test_data/librispeech_test_paired.jsonl" --max_samples 100
```
This will generate `librispeech_train_paired.jsonl` and `librispeech_test_paired.jsonl` in their respective `train_data` and `test_data` directories. The `.jsonl` files will contain entries like:
`{"audio_path": "path/to/audio.flac", "text": "THE TRANSCRIPTION..."}`

### Step 3: Feature and Token Extraction

Next, extract the necessary features and tokens for the model.

1.  **Extract Text and Speech Features:**
    Run `src/utt2text_and_feature.py` to extract Whisper speech features and CosyVoice text embeddings.
    ```bash
    # For training data
    python src/utt2text_and_feature.py --jsonl "data/train_data/librispeech_train_paired.jsonl" --max_samples 3000 --model_dir "C:/Users/Administrator/.cache/modelscope/hub/models/iic/CosyVoice-300M" --output_text "data/train_data/embeddings/text_embeddings.pt" --output_whisper "data/train_data/embeddings/whisper_features.pt"

    # For test data
    python src/utt2text_and_feature.py --jsonl "data/test_data/librispeech_test_paired.jsonl" --max_samples 100 --model_dir "C:/Users/Administrator/.cache/modelscope/hub/models/iic/CosyVoice-300M" --output_text "data/test_data/embeddings/text_embeddings.pt" --output_whisper "data/test_data/embeddings/whisper_features.pt"
    ```
    *Note: Update the `--model_dir` path to your local CosyVoice model directory.*

2.  **Extract S3 Speech Tokens:**
    Run the `src/s3.sh` script to extract S3 tokens, which are the target for our model.
    ```bash
    # This script needs to be configured for train and test sets respectively.
    # Example for training set:
    # LIBRISPEECH_DIR="data/raw_data/train-clean-100/LibriSpeech"
    # OUT_DIR="data/train_data/librispeech_speech_tokenizer"
    bash src/s3.sh
    ```
    This will generate a `wav.scp` file and then run `extract_speech_token.py` to create `utt2speech_token.pt`.

### Step 4: Training and Evaluation

The main implementation, including the attention aggregator, model integration, and training loop, is in `src/example.ipynb` and `src/train and test.py`.

1.  **Open and run the notebook/script:**
    - For an interactive experience, launch Jupyter and open `src/example.ipynb`. Run the cells sequentially.
    - Alternatively, you can run the Python script directly from the command line:
      ```bash
      python src/"train and test.py"
      ```

2.  **Process:**
    The code will:
    - Load the preprocessed data and extracted features.
    - Define the `SimpleTextSpeechAggregator` and `CosyVoiceS3Model`.
    - Fine-tune the model by minimizing the cross-entropy loss.
    - Save training artifacts like the loss curve (`results/training_loss_curve.png`) and a report (`results/training_report.json`).
    - Evaluate the model on the test set and report the top-1 S3-token prediction accuracy.

## Scripts Description

-   **`src/Dataset Preparation.py`**: Pairs LibriSpeech audio files with transcriptions, resamples audio to 16kHz, and saves the data in `.jsonl` format.
-   **`src/utt2text_and_feature.py`**: Extracts text embeddings using CosyVoice and speech features (from intermediate and final layers) using Whisper's encoder.
-   **`src/s3.sh`**: A shell script that first generates a `wav.scp` file mapping utterance IDs to audio file paths, and then calls a CosyVoice tool to extract S3 speech tokens.
-   **`src/train and test.py` / `src/example.ipynb`**: The core of the project. It defines the model architecture, data loaders, and the complete training, validation, and prediction pipeline.
