#!/usr/bin/env python3
"""
MR603 Dining Analytics — Standalone Inference Script
=====================================================
Models:
  1. ResNet-50  → Food-101 image classification (101 classes)
     Weights: ./models/resnet50_food101.pth
  2. BERT       → Yelp review sentiment (3-class: Negative / Neutral / Positive)
     Checkpoint: ./models/bert_yelp/checkpoint-1350/
  3. Multimodal Fusion → Cross-attention over BERT + ResNet embeddings

Usage:
  python inference.py --image path/to/food.jpg
  python inference.py --text "Amazing sushi, best I've ever had!"
  python inference.py --image path/to/food.jpg --text "Amazing sushi!"
"""

import argparse
import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from transformers import (
    BertForSequenceClassification,
    BertModel,
    BertTokenizer,
)

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RESNET_WEIGHTS_PATH = "./models/resnet50_food101.pth"
BERT_CHECKPOINT_DIR = "./models/bert_yelp/checkpoint-1350"

NUM_FOOD_CLASSES = 101
SENTIMENT_LABELS = {0: "Negative", 1: "Neutral", 2: "Positive"}

# Food-101 class names (alphabetical order as in the HuggingFace dataset)
FOOD101_CLASSES = [
    "apple_pie", "baby_back_ribs", "baklava", "beef_carpaccio", "beef_tartare",
    "beet_salad", "beignets", "bibimbap", "bread_pudding", "breakfast_burrito",
    "bruschetta", "caesar_salad", "cannoli", "caprese_salad", "carrot_cake",
    "ceviche", "cheese_plate", "cheesecake", "chicken_curry", "chicken_quesadilla",
    "chicken_wings", "chocolate_cake", "chocolate_mousse", "churros", "clam_chowder",
    "club_sandwich", "crab_cakes", "creme_brulee", "croque_madame", "cup_cakes",
    "deviled_eggs", "donuts", "dumplings", "edamame", "eggs_benedict",
    "escargots", "falafel", "filet_mignon", "fish_and_chips", "foie_gras",
    "french_fries", "french_onion_soup", "french_toast", "fried_calamari", "fried_rice",
    "frozen_yogurt", "garlic_bread", "gnocchi", "greek_salad", "grilled_cheese_sandwich",
    "grilled_salmon", "guacamole", "gyoza", "hamburger", "hot_and_sour_soup",
    "hot_dog", "huevos_rancheros", "hummus", "ice_cream", "lasagna",
    "lobster_bisque", "lobster_roll_sandwich", "macaroni_and_cheese", "macarons",
    "miso_soup", "mussels", "nachos", "omelette", "onion_rings",
    "oysters", "pad_thai", "paella", "pancakes", "panna_cotta",
    "peking_duck", "pho", "pizza", "pork_chop", "poutine",
    "prime_rib", "pulled_pork_sandwich", "ramen", "ravioli", "red_velvet_cake",
    "risotto", "samosa", "sashimi", "scallops", "seaweed_salad",
    "shrimp_and_grits", "spaghetti_bolognese", "spaghetti_carbonara", "spring_rolls",
    "steak", "strawberry_shortcake", "sushi", "tacos", "takoyaki",
    "tiramisu", "tuna_tartare", "waffles",
]

# ---------------------------------------------------------------------------
# Image transforms (must match training)
# ---------------------------------------------------------------------------
IMAGE_SIZE = 224

image_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

# ---------------------------------------------------------------------------
# Model definitions
# ---------------------------------------------------------------------------

class MultiModalFusionModel(nn.Module):
    """
    Cross-attention fusion of BERT text embeddings (768-d) and
    ResNet image embeddings (2048-d) → 3-class sentiment.
    """
    def __init__(self, text_dim=768, image_dim=2048,
                 hidden_dim=512, num_classes=3, dropout=0.3):
        super().__init__()
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.image_proj = nn.Sequential(
            nn.Linear(image_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=8,
            dropout=0.1, batch_first=True,
        )
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, text_emb, image_emb):
        t = self.text_proj(text_emb).unsqueeze(1)
        v = self.image_proj(image_emb).unsqueeze(1)
        fused, _ = self.cross_attention(t, v, v)
        fused = fused.squeeze(1)
        combined = torch.cat([t.squeeze(1), fused], dim=-1)
        return self.classifier(combined)


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------

def load_resnet(weights_path: str = RESNET_WEIGHTS_PATH) -> nn.Module:
    """Load the fine-tuned ResNet-50 for Food-101 classification."""
    if not os.path.isfile(weights_path):
        sys.exit(f"[ERROR] ResNet weights not found: {weights_path}")

    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, NUM_FOOD_CLASSES)
    state_dict = torch.load(weights_path, map_location=DEVICE, weights_only=True)
    model.load_state_dict(state_dict)
    model.to(DEVICE).eval()
    print(f"[OK] ResNet-50 loaded from {weights_path}")
    return model


def load_bert_sentiment(checkpoint_dir: str = BERT_CHECKPOINT_DIR):
    """Load the fine-tuned BERT sentiment model + tokenizer."""
    if not os.path.isdir(checkpoint_dir):
        sys.exit(f"[ERROR] BERT checkpoint not found: {checkpoint_dir}")

    tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
    model = BertForSequenceClassification.from_pretrained(checkpoint_dir)
    model.to(DEVICE).eval()
    print(f"[OK] BERT sentiment model loaded from {checkpoint_dir}")
    return tokenizer, model


def load_bert_encoder(checkpoint_dir: str = BERT_CHECKPOINT_DIR):
    """Load the underlying BERT encoder for embedding extraction (fusion)."""
    tokenizer = BertTokenizer.from_pretrained(checkpoint_dir)
    encoder = BertModel.from_pretrained("bert-base-uncased")
    encoder.to(DEVICE).eval()
    return tokenizer, encoder


# ---------------------------------------------------------------------------
# Inference functions
# ---------------------------------------------------------------------------

def classify_food_image(image_path: str, model: nn.Module, top_k: int = 5) -> list:
    """
    Classify a food image.

    Returns:
        list of (class_name, probability) tuples, length = top_k.
    """
    img = Image.open(image_path).convert("RGB")
    tensor = image_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_probs, top_idxs = probs.topk(top_k)
    results = []
    for prob, idx in zip(top_probs, top_idxs):
        idx = idx.item()
        name = FOOD101_CLASSES[idx] if idx < len(FOOD101_CLASSES) else f"class_{idx}"
        results.append((name.replace("_", " ").title(), prob.item()))
    return results


def predict_sentiment(text: str, tokenizer, model) -> dict:
    """
    Predict sentiment of a text review / tweet.

    Returns:
        dict with keys: label (str), confidence (float), logits (list).
    """
    enc = tokenizer(
        text,
        max_length=128,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**enc).logits
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    return {
        "label": SENTIMENT_LABELS[pred_idx],
        "confidence": probs[pred_idx].item(),
        "probabilities": {
            SENTIMENT_LABELS[i]: round(probs[i].item(), 4) for i in range(3)
        },
    }


def extract_text_embedding(text: str, tokenizer, encoder) -> torch.Tensor:
    """Return the BERT [CLS] embedding (1, 768) for a single text."""
    enc = tokenizer(
        text, max_length=128, padding="max_length",
        truncation=True, return_tensors="pt",
    ).to(DEVICE)
    with torch.no_grad():
        out = encoder(**enc)
    return out.last_hidden_state[:, 0, :]  # (1, 768)


def extract_image_embedding(image_path: str, resnet: nn.Module) -> torch.Tensor:
    """Return the ResNet-50 pre-FC embedding (1, 2048) for a single image."""
    feat_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
    feat_extractor.eval()

    img = Image.open(image_path).convert("RGB")
    tensor = image_transform(img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        feats = feat_extractor(tensor)
    return feats.squeeze(-1).squeeze(-1)  # (1, 2048)


def multimodal_inference(
    text: str,
    image_path: str,
    bert_tokenizer,
    bert_encoder,
    resnet: nn.Module,
    fusion_model: MultiModalFusionModel,
) -> dict:
    """
    Run the full multimodal pipeline: BERT embedding + ResNet embedding
    → cross-attention fusion → 3-class sentiment prediction.
    """
    text_emb = extract_text_embedding(text, bert_tokenizer, bert_encoder)
    image_emb = extract_image_embedding(image_path, resnet)

    with torch.no_grad():
        logits = fusion_model(text_emb, image_emb)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = probs.argmax().item()
    return {
        "label": SENTIMENT_LABELS[pred_idx],
        "confidence": probs[pred_idx].item(),
        "probabilities": {
            SENTIMENT_LABELS[i]: round(probs[i].item(), 4) for i in range(3)
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="MR603 Dining Analytics — Inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Classify a food image (top-5 predictions)
  python inference.py --image photo.jpg

  # Predict sentiment of a review
  python inference.py --text "The ramen was absolutely incredible!"

  # Multimodal: image + text → fused sentiment
  python inference.py --image photo.jpg --text "The ramen was incredible!"

  # Adjust top-k for image classification
  python inference.py --image photo.jpg --top_k 10
        """,
    )
    parser.add_argument("--image", type=str, help="Path to a food image (jpg/png)")
    parser.add_argument("--text", type=str, help="Review or tweet text")
    parser.add_argument("--top_k", type=int, default=5, help="Top-K for image classification (default: 5)")
    parser.add_argument("--fusion_weights", type=str, default=None,
                        help="Path to trained fusion model weights (optional)")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device: 'cpu' or 'cuda' (default: auto-detect)")
    args = parser.parse_args()

    if args.device:
        global DEVICE
        DEVICE = torch.device(args.device)

    if args.top_k < 1 or args.top_k > NUM_FOOD_CLASSES:
        sys.exit(f"[ERROR] --top_k must be between 1 and {NUM_FOOD_CLASSES}.")

    print(f"[INFO] Using device: {DEVICE}")

    if not args.image and not args.text:
        parser.print_help()
        sys.exit("\n[ERROR] Provide at least --image or --text.")

    # ── Image-only mode ──────────────────────────────────────────────────
    if args.image and not args.text:
        resnet = load_resnet()
        results = classify_food_image(args.image, resnet, top_k=args.top_k)
        print(f"\n{'─' * 50}")
        print(f"  Food Image Classification — {os.path.basename(args.image)}")
        print(f"{'─' * 50}")
        for rank, (name, prob) in enumerate(results, 1):
            bar = "█" * int(prob * 40)
            print(f"  {rank}. {name:<30} {prob:.4f}  {bar}")
        return

    # ── Text-only mode ───────────────────────────────────────────────────
    if args.text and not args.image:
        tokenizer, bert = load_bert_sentiment()
        result = predict_sentiment(args.text, tokenizer, bert)
        print(f"\n{'─' * 50}")
        print(f"  Sentiment Prediction")
        print(f"{'─' * 50}")
        print(f"  Text : {args.text[:80]}{'...' if len(args.text) > 80 else ''}")
        print(f"  Label: {result['label']}  (confidence: {result['confidence']:.4f})")
        print(f"  Probs: {result['probabilities']}")
        return

    # ── Multimodal mode (image + text) ───────────────────────────────────
    if args.image and args.text:
        resnet = load_resnet()

        # Also run standalone image classification
        img_results = classify_food_image(args.image, resnet, top_k=args.top_k)

        # Standalone text sentiment
        sent_tokenizer, sent_bert = load_bert_sentiment()
        text_result = predict_sentiment(args.text, sent_tokenizer, sent_bert)

        # Fusion
        fusion_tokenizer, fusion_encoder = load_bert_encoder()
        fusion_model = MultiModalFusionModel().to(DEVICE)

        if args.fusion_weights and os.path.isfile(args.fusion_weights):
            fusion_model.load_state_dict(
                torch.load(args.fusion_weights, map_location=DEVICE, weights_only=True)
            )
            print(f"[OK] Fusion weights loaded from {args.fusion_weights}")
        else:
            print("[WARN] No trained fusion weights — using random initialization (demo only)")

        fusion_model.eval()
        fusion_result = multimodal_inference(
            args.text, args.image,
            fusion_tokenizer, fusion_encoder, resnet, fusion_model,
        )

        # ── Display results ──────────────────────────────────────────────
        print(f"\n{'═' * 60}")
        print(f"  MULTIMODAL INFERENCE RESULTS")
        print(f"{'═' * 60}")

        print(f"\n  Image: {os.path.basename(args.image)}")
        print(f"  Text : {args.text[:80]}{'...' if len(args.text) > 80 else ''}")

        print(f"\n  ┌── Food Classification (ResNet-50) ──")
        for rank, (name, prob) in enumerate(img_results, 1):
            print(f"  │  {rank}. {name:<28} {prob:.4f}")

        print(f"  ├── Text Sentiment (BERT) ──")
        print(f"  │  {text_result['label']}  ({text_result['confidence']:.4f})")

        print(f"  ├── Fused Sentiment (Cross-Attention) ──")
        print(f"  │  {fusion_result['label']}  ({fusion_result['confidence']:.4f})")
        print(f"  │  Probs: {fusion_result['probabilities']}")
        print(f"  └{'─' * 55}")
        return


if __name__ == "__main__":
    main()