#!/usr/bin/env bash
LIBRISPEECH_DIR="../data/raw_data/train-clean-100/LibriSpeech"
OUT_DIR="../data/train_data/librispeech_speech_tokenizer"
ONNX_PATH="C:/Users/Administrator/.cache/modelscope/hub/models/iic/CosyVoice-300M/speech_tokenizer_v1.onnx" # speech_tokenizer_v1.onnx under CosyVoice-300M model dir

mkdir -p "$OUT_DIR"
find "$LIBRISPEECH_DIR" -type f \( -iname "*.flac" -o -iname "*.wav" \) | sort | while read -r f; do
  rel="${f#"$LIBRISPEECH_DIR"/}"
  id="${rel%.*}"; id="${id//\//-}"
  echo "$id $f"
done > "$OUT_DIR/wav.scp"

python3 "../model/CosyVoice-main/tools/extract_speech_token.py" \
  --dir "$OUT_DIR" \
  --onnx_path "$ONNX_PATH"
# python "../model/CosyVoice-main/tools/extract_speech_token.py" --dir "../data/train_data/librispeech_speech_tokenizer" --max_samples 3000 --onnx_path "C:/Users/Administrator/.cache/modelscope/hub/models/iic/CosyVoice-300M/speech_tokenizer_v1.onnx"
# python "../model/CosyVoice-main/tools/extract_speech_token.py" --dir "../data/test_data/librispeech_speech_tokenizer" --max_samples 100 --onnx_path "C:/Users/Administrator/.cache/modelscope/hub/models/iic/CosyVoice-300M/speech_tokenizer_v1.onnx"
