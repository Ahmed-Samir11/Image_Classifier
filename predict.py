# predict.py

import argparse
import torch
from utils import load_checkpoint, predict
import json

def main():
    parser = argparse.ArgumentParser(description="Predict the class of an image using a trained model.")
    parser.add_argument('image_path', type=str, help='Path to the input image.')
    parser.add_argument('checkpoint', type=str, help='Path to the model checkpoint.')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return.')
    parser.add_argument('--category_names', type=str, default='cat_to_name.json', help='Path to JSON file mapping categories to names.')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference if available.')
    
    args = parser.parse_args()
    
    device = 'cuda' if args.gpu and torch.cuda.is_available() else 'cpu'
    if args.gpu and device == 'cpu':
        print("GPU requested but not available. Using CPU instead.")

    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)
    else:
        cat_to_name = {}

    model = load_checkpoint(args.checkpoint)
    probs, classes = predict(args.image_path, model, topk=args.top_k, device=device)
    
    if cat_to_name:
        class_names = [cat_to_name.get(cls, cls) for cls in classes]
    else:
        class_names = classes

    print("Top K Predictions:")
    for i in range(len(probs)):
        print(f"{i+1}: {class_names[i]} with probability {probs[i]*100:.2f}%")

if __name__ == "__main__":
    main()