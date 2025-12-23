#!/usr/bin/env python3
"""测试 HuggingFace 镜像配置是否正确"""
import os
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent
    env_file = project_root / ".env"
    if env_file.exists():
        load_dotenv(env_file, override=False)
        print(f"✓ Loaded .env from: {env_file}")
except ImportError:
    print("⚠ python-dotenv not installed")

# Set HF_ENDPOINT before importing
if os.getenv("HF_ENDPOINT"):
    hf_endpoint = os.getenv("HF_ENDPOINT")
    os.environ["HF_ENDPOINT"] = hf_endpoint
    os.environ["HUGGINGFACE_HUB_ENDPOINT"] = hf_endpoint
    print(f"✓ HF_ENDPOINT = {hf_endpoint}")
else:
    print("✗ HF_ENDPOINT not set!")

# Now import HuggingFace libraries
print("\n" + "="*70)
print("Testing HuggingFace Hub configuration")
print("="*70)

# Check huggingface_hub configuration
try:
    from huggingface_hub import constants
    print(f"\nHuggingFace Hub constants:")
    print(f"  ENDPOINT = {constants.ENDPOINT}")
    if hasattr(constants, 'HF_HUB_OFFLINE'):
        print(f"  HF_HUB_OFFLINE = {constants.HF_HUB_OFFLINE}")
except Exception as e:
    print(f"Error checking constants: {e}")

# Test sentence-transformers download
print("\n" + "="*70)
print("Testing Sentence-Transformers model download")
print("="*70)

try:
    from sentence_transformers import SentenceTransformer

    model_name = "paraphrase-multilingual-MiniLM-L12-v2"
    print(f"\nAttempting to load: {model_name}")
    print(f"Expected source: {os.getenv('HF_ENDPOINT', 'https://huggingface.co')}")

    # Try loading model
    model = SentenceTransformer(model_name, device='cpu')
    print(f"✓ Model loaded successfully!")
    print(f"  Embedding dimension: {model.get_sentence_embedding_dimension()}")

    # Test encoding
    test_texts = ["测试文本", "Test text"]
    embeddings = model.encode(test_texts)
    print(f"✓ Encoding test passed: shape={embeddings.shape}")

except Exception as e:
    print(f"✗ Error: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*70)
print("Test complete")
print("="*70)
