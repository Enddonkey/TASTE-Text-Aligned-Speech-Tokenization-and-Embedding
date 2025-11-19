#!/usr/bin/env python3
"""
Dataset Preparation for LibriSpeech
This module handles LibriSpeech dataset pairing and preprocessing tasks.
Refactored from example.ipynb sections 1.1 and 1.2
"""

import argparse
import json
import os
import torch
import torchaudio
import soundfile as sf
import matplotlib.pyplot as plt
from tqdm import tqdm
from torchaudio.transforms import Resample, MelSpectrogram, AmplitudeToDB
from transformers import WhisperProcessor, WhisperForConditionalGeneration


def pair_librispeech_data(root_dir):
    """
    Pair LibriSpeech FLAC files with transcription text
    
    Args:
        root_dir (str): Root directory of LibriSpeech dataset
        
    Returns:
        list: List of dictionaries containing flac_path and transcript pairs
    """
    paired_data = []
    
    # Traverse dataset directory structure (speaker/chapter)
    for speaker_id in os.listdir(root_dir):
        speaker_path = os.path.join(root_dir, speaker_id)
        if not os.path.isdir(speaker_path):
            continue
            
        for chapter_id in os.listdir(speaker_path):
            chapter_path = os.path.join(speaker_path, chapter_id)
            if not os.path.isdir(chapter_path):
                continue
                
            # Read transcription file
            transcript_file = os.path.join(chapter_path, f"{speaker_id}-{chapter_id}.trans.txt")
            if not os.path.exists(transcript_file):
                continue
                
            # Parse transcription content
            with open(transcript_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    utt_id, transcript = line.split(" ", 1)
                    flac_path = os.path.join(chapter_path, f"{utt_id}.flac")
                    if os.path.exists(flac_path):
                        # Convert to standard path format (using / as the uniform separator)
                        normalized_path = flac_path.replace("\\", "/")
                        # Store as "audio_path" and "text" fields
                        paired_data.append({
                            "audio_path": normalized_path,
                            "text": transcript
                        })
                        print(f"Added: {utt_id}.flac -> text: {transcript[:20]}...")
    
    return paired_data


def resample_audio(flac_path, target_sr=16000):
    """
    Load FLAC with soundfile and resample with torchaudio
    
    Args:
        flac_path (str): Path to FLAC file
        target_sr (int): Target sampling rate
        
    Returns:
        numpy.ndarray: Resampled audio as 1D numpy array
    """
    # Load FLAC with soundfile (native support)
    audio, sr = sf.read(flac_path)  # Returns numpy array and sampling rate
    
    # Convert to torch tensor for resampling
    waveform = torch.tensor(audio).unsqueeze(0)  # Add channel dimension
    
    # Resample if necessary
    if sr != target_sr:
        resampler = Resample(sr, target_sr)
        waveform = resampler(waveform)
        
    return waveform.squeeze().numpy()  # Convert back to 1D numpy array


def save_to_jsonl(paired_data, output_path, max_samples=None):
    """
    Save paired data to JSONL format with preprocessing
    
    Args:
        paired_data (list): List of paired audio-text data
        output_path (str): Output JSONL file path
        max_samples (int, optional): Maximum number of samples to process
    """
    if max_samples:
        paired_data = paired_data[:max_samples]
        
    with open(output_path, "w", encoding="utf-8") as f:
        for item in tqdm(paired_data, desc="Saving to JSONL"):
            # Clearly use audio_path and text fields
            json.dump({
                "audio_path": item["audio_path"],
                "text": item["text"]
                # To keep audio array, add: "audio": resample_audio(item["audio_path"]).tolist()
            }, f, ensure_ascii=False)
            f.write("\n")


def plot_mel_spectrogram(audio, sr=16000, save_path=None):
    """
    Plot Mel spectrogram
    
    Args:
        audio (numpy.ndarray): Audio signal
        sr (int): Sampling rate
        save_path (str, optional): Path to save the plot
    """
    # Convert to Mel spectrogram
    mel_transform = MelSpectrogram(
        sample_rate=sr, n_mels=128, n_fft=400, hop_length=160
    )
    
    # Convert double to float
    mel_spec = mel_transform(torch.tensor(audio).float().unsqueeze(0))
    mel_spec_db = AmplitudeToDB()(mel_spec)  # Convert to decibel scale
    
    # Plot
    plt.figure(figsize=(12, 4))
    plt.imshow(
        mel_spec_db[0].numpy(),
        aspect="auto",
        origin="lower",
        cmap="magma"
    )
    plt.xlabel("Time Frames")
    plt.ylabel("Mel Bands")
    plt.title("Mel-Spectrogram (16kHz)")
    plt.colorbar(label="Intensity (dB)")
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Mel spectrogram saved to {save_path}")
    else:
        plt.show()

def main(args):
    """
    Main function to execute dataset preparation pipeline
    """
    print("Starting LibriSpeech dataset preparation...")
    
    # 1.1 Pair LibriSpeech data
    print(f"Pairing data from: {args.root_dir}")
    paired_data = pair_librispeech_data(args.root_dir)
    print(f"Paired data has {len(paired_data)} items.")

    # 1.2 Preprocessing
    print("Starting preprocessing...")
    # Save paired results
    os.makedirs(os.path.dirname(args.output_paired), exist_ok=True)
    save_to_jsonl(paired_data, args.output_paired, max_samples=args.max_samples)
    

    # Optional: Mel spectrogram visualization
    if args.visualize and len(paired_data) > 0:
        print("Generating Mel spectrogram visualization...")
        sample_audio = resample_audio(paired_data[0]["audio_path"])
        plot_mel_spectrogram(sample_audio, save_path=args.mel_plot_path)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LibriSpeech Dataset Preparation")
    parser.add_argument("--root_dir", type=str, required=True,
                        help="Root directory of LibriSpeech dataset")
    parser.add_argument("--output_paired", type=str, default="../data/train_data/librispeech_train_paired.jsonl",
                        help="Output file for paired data")
    parser.add_argument("--max_samples", type=int, default=100,
                        help="Maximum number of samples to preprocess")
    parser.add_argument("--visualize", action="store_true",
                        help="Generate Mel spectrogram visualization")
    parser.add_argument("--mel_plot_path", type=str, default="./results/mel_spectrogram.png",
                        help="Path to save Mel spectrogram plot")
    
    args = parser.parse_args()
    main(args)

# python "Dataset Preparation.py" --root_dir "../data/raw_data/train-clean-100/LibriSpeech/train-clean-100" --max_samples 3000 --output_paired "../data/train_data/librispeech_train_paired.jsonl" --visualize --mel_plot_path "../results/mel_spectrogram.png"
# python "Dataset Preparation.py" --root_dir "../data/raw_data/test-clean/LibriSpeech/test-clean" --max_samples 100 --output_paired "../data/test_data/librispeech_test_paired.jsonl"