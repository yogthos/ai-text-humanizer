# Manual Download Instructions for all-MiniLM-L6-v2 Model

## Option 1: Using huggingface-cli (Recommended)

If you have `huggingface_hub` installed:

```bash
# Install huggingface-cli if needed
pip install huggingface_hub

# Download the model
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2

# Or with custom cache directory
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2 --cache-dir ~/.cache/huggingface/hub/
```

## Option 2: Using Python Script with Extended Timeout

Run the provided script:

```bash
python3 scripts/download_model_manual.py --timeout 120
```

## Option 3: Manual Download from Hugging Face Website

1. Visit: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
2. Click "Files and versions" tab
3. Download all files to:
   ```
   ~/.cache/huggingface/hub/models--sentence-transformers--all--MiniLM--L6--v2/
   ```

Required files:
- `config.json`
- `sentence_bert_config.json`
- `config_sentence_transformers.json`
- `pytorch_model.bin` (or `model.safetensors`)
- `tokenizer_config.json`
- `vocab.txt`
- `modules.json`
- `1_Pooling/config.json`
- `2_Dense/config.json` (if exists)
- `2_Dense/pytorch_model.bin` (if exists)

## Option 4: Direct Download with wget/curl

```bash
# Create cache directory
mkdir -p ~/.cache/huggingface/hub/models--sentence-transformers--all--MiniLM--L6--v2/snapshots/main/

# Download files (replace with actual URLs from Hugging Face)
cd ~/.cache/huggingface/hub/models--sentence-transformers--all--MiniLM--L6--v2/snapshots/main/

# Download config files
wget --timeout=60 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config.json
wget --timeout=60 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/sentence_bert_config.json
wget --timeout=60 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/config_sentence_transformers.json
wget --timeout=60 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/pytorch_model.bin
wget --timeout=60 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/tokenizer_config.json
wget --timeout=60 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/vocab.txt
wget --timeout=60 https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/modules.json

# Download subdirectory files
mkdir -p 1_Pooling
wget --timeout=60 -O 1_Pooling/config.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/1_Pooling/config.json

mkdir -p 2_Dense
wget --timeout=60 -O 2_Dense/config.json https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/2_Dense/config.json
wget --timeout=60 -O 2_Dense/pytorch_model.bin https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2/resolve/main/2_Dense/pytorch_model.bin
```

## Verify Installation

After downloading, verify the model loads:

```bash
python3 -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('all-MiniLM-L6-v2'); print('✓ Model loaded successfully!')"
```

## Troubleshooting

If you still get timeout errors:

1. **Increase timeout in environment:**
   ```bash
   export HF_HUB_DOWNLOAD_TIMEOUT=300
   ```

2. **Use a VPN or proxy** if Hugging Face is blocked in your region

3. **Download from a mirror** if available

4. **Check your internet connection** - the model files are ~90MB total

