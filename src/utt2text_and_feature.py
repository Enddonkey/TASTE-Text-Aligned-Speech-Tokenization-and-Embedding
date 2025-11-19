#!/usr/bin/env python3
import argparse
import json

import os
import torch
import torchaudio
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq
from cosyvoice.cli.cosyvoice import CosyVoice

# --- huggingface_hub compatibility patch (for CosyVoice) ---
try:
    import huggingface_hub as _hfh
    if not hasattr(_hfh, "cached_download"):
        from huggingface_hub import hf_hub_download as _hf_hub_download

        def cached_download(*args, **kwargs):
            return _hf_hub_download(*args, **kwargs)

        _hfh.cached_download = cached_download
except Exception:
    pass

def load_jsonl(path, max_count=None):
    items = []
    if max_count is not None and (not isinstance(max_count, int) or max_count <= 0):
        raise ValueError("max_count must be a positive integer or None")
    if max_count is not None:
        print(f"Loading up to {max_count} items from {path}")
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))
            if max_count is not None and len(items) >= max_count:
                break
    return items


def load_audio(path, target_sr=16000):
    audio, sr = torchaudio.load(path)
    if sr != target_sr:
        audio = torchaudio.transforms.Resample(orig_freq=sr, new_freq=target_sr)(audio)
    if audio.shape[0] > 1:
        audio = audio.mean(0, keepdim=True)
    return audio.squeeze(0), target_sr  # (T,), sr


def extract_whisper_encoder_feats(waveform, model, processor, device, max_duration=30.0):
    # waveform: 1D torch tensor at 16k
    num_seconds = waveform.numel() / 16000.0
    if num_seconds > max_duration:
        return None, None

    audio_np = waveform.numpy()
    inputs = processor(
        audio_np,
        sampling_rate=16000,
        return_tensors="pt",
    )
    input_features = inputs.input_features.to(device)

    with torch.no_grad():
        enc_out = model.model.encoder(
            input_features=input_features,
            output_hidden_states=True,
        )

    hidden_states = enc_out.hidden_states  # list: [layer0, layer1, ..., last]
    mid_idx = len(hidden_states) // 2
    mid_layer = hidden_states[mid_idx].cpu().squeeze(0)   # (T, D)
    final_layer = hidden_states[-1].cpu().squeeze(0)      # (T, D)
    return mid_layer, final_layer

def get_long_key(path_str):
    # ../raw_data/train-clean-100/LibriSpeech/train-clean-100/103/1240/103-1240-0000.flac
    # -> train-clean-100-103-1240-103-1240-0000
    parts = path_str.replace('\\', '/').split('/')
    # Expected structure: ..., 'train-clean-100', '103', '1240', '103-1240-0000.flac'
    try:
        filename = parts[-1]
        speaker_id = parts[-3]
        chapter_id = parts[-2]
        dataset_part = parts[-4]
        base_id = os.path.splitext(filename)[0]
        return f"{dataset_part}-{speaker_id}-{chapter_id}-{base_id}"
    except IndexError:
        return os.path.splitext(os.path.basename(path_str))[0]
    
def main(args):
    # load CosyVoice for text embedding
    cosy = CosyVoice(args.model_dir)
    emb_layer = cosy.model.llm.text_embedding

    # load Whisper encoder
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = AutoProcessor.from_pretrained("C:\\Users\\Administrator\\.cache\\modelscope\\hub\\models\\openai-mirror\\whisper-large-v3")
    whisper_model = AutoModelForSpeechSeq2Seq.from_pretrained(
        "C:\\Users\\Administrator\\.cache\\modelscope\\hub\\models\\openai-mirror\\whisper-large-v3"
    ).to(device)
    whisper_model.eval()

    data = load_jsonl(args.jsonl, max_count=args.max_samples)

    utt2text_emb = {}
    utt2whisper_mid = {}
    utt2whisper_final = {}

    print(f"Loaded {len(data)} items from {args.jsonl}")


    for item in tqdm(data):
        audio_path = item["audio_path"]
        text = item["text"]
        
        utt_id = get_long_key(audio_path)
        # print(f"Processing {utt_id}")
        # ----- CosyVoice text embedding -----
        text_token, text_token_len = cosy.frontend._extract_text_token(text)  # [1, L], [1]
        with torch.no_grad():
            text_token = text_token.to(emb_layer.weight.device).long()
            text_emb = emb_layer(text_token)  # [1, L, D]
        utt2text_emb[utt_id] = text_emb.squeeze(0).cpu()  # [L, D]

        # ----- Whisper encoder features -----
        waveform, _ = load_audio(audio_path, target_sr=16000)
        mid_feat, final_feat = extract_whisper_encoder_feats(
            waveform, whisper_model, processor, device, max_duration=args.max_duration
        )
        utt2whisper_mid[utt_id] = mid_feat
        utt2whisper_final[utt_id] = final_feat

    # save outputs
    os.makedirs(os.path.dirname(args.output_text), exist_ok=True)
    torch.save(utt2text_emb, args.output_text)
    print(f"Saved CosyVoice text embeddings for {len(utt2text_emb)} items to {args.output_text}")

    whisper_output = {
        "mid": utt2whisper_mid,
        "final": utt2whisper_final,
    }
    os.makedirs(os.path.dirname(args.output_whisper), exist_ok=True)
    torch.save(whisper_output, args.output_whisper)
    print(f"Saved Whisper features for {len(utt2whisper_mid)} items to {args.output_whisper}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", type=str, required=True, default="../data/train_data/librispeech_train_paired.jsonl", help="Input jsonl with audio_path and text")
    parser.add_argument("--max_samples", type=int, default=100,help="Maximum number of samples to preprocess")
    parser.add_argument("--model_dir", type=str, required=True, default="C:\\Users\\Administrator\\.cache\\modelscope\\hub\\models\\iic\\CosyVoice-300M", help="CosyVoice1 model dir")
    parser.add_argument("--output_text", type=str, required=True, default="../data/train_data/embeddings/text_embeddings.pt",help="Output .pt for CosyVoice text embeddings")
    parser.add_argument("--output_whisper", type=str, required=True, default="../data/train_data/embeddings/whisper_features.pt", help="Output .pt for Whisper features")
    parser.add_argument("--max_duration", type=float, default=30.0, help="Max audio length (seconds) to process")
    main(parser.parse_args())

# python utt2text_and_feature.py --jsonl "../data/train_data/librispeech_train_paired.jsonl" --max_samples 3000 --model_dir "C:\\Users\\Administrator\\.cache\\modelscope\\hub\\models\\iic\\CosyVoice-300M" --output_text "../data/train_data/embeddings/text_embeddings.pt" --output_whisper "../data/train_data/embeddings/whisper_features.pt" --max_duration 30.0
# python utt2text_and_feature.py --jsonl "../data/test_data/librispeech_test_paired.jsonl" --max_samples 100 --model_dir "C:\\Users\\Administrator\\.cache\\modelscope\\hub\\models\\iic\\CosyVoice-300M" --output_text "../data/test_data/embeddings/text_embeddings.pt" --output_whisper "../data/test_data/embeddings/whisper_features.pt" --max_duration 30.0