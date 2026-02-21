"""
Simple tester for exported TF-IDF pipeline.

Usage examples:
  # Evaluate on a CSV (expects columns 'Hasil OCR' and 'Kategori')
  python scripts/test_exported_model.py --evaluate --data datasets/my_test.csv

  # Predict a single text (CLI aliases: --predict or --text)
  python scripts/test_exported_model.py --predict "contoh teks hasil OCR"
  # or
  python scripts/test_exported_model.py --text "contoh teks hasil OCR"

  # Predict multiple texts from a file (one line per sample)
  python scripts/test_exported_model.py --predict-file examples/samples.txt

Requirements:
  pip install scikit-learn pandas joblib

Saves optional predictions to `predictions.csv` when evaluating.
"""
import argparse
import json
import os
import sys
from typing import List

import joblib
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

MODEL_PATH = "../models/tfidf_pipeline.joblib"
META_PATH = "../models/tfidf_pipeline_meta.json"


def load_model(model_path: str = MODEL_PATH):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    pipeline = joblib.load(model_path)
    return pipeline


def load_meta(meta_path: str = META_PATH):
    if not os.path.exists(meta_path):
        return {}
    with open(meta_path, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate(pipeline, csv_path: str):
    df = pd.read_csv(csv_path)
    if "Hasil OCR" not in df.columns:
        raise ValueError("CSV harus memiliki kolom 'Hasil OCR'")

    X = df[["Hasil OCR"]]
    y_true = df["Kategori"] if "Kategori" in df.columns else None

    y_pred = pipeline.predict(X)

    out = pd.DataFrame({"Hasil OCR": df["Hasil OCR"], "predicted": y_pred})
    out.to_csv("predictions.csv", index=False)
    print("Saved predictions -> predictions.csv")

    if y_true is not None:
        print("\nClassification report:")
        print(classification_report(y_true, y_pred))
        print("Accuracy:", accuracy_score(y_true, y_pred))
    else:
        print("No ground truth labels ('Kategori') found in CSV. Predictions saved to predictions.csv")


def predict_texts(pipeline, texts: List[str]):
    df = pd.DataFrame({"Hasil OCR": texts})
    preds = pipeline.predict(df)
    for t, p in zip(texts, preds):
        print(f"=> {p}\t| {t}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="Test exported TF-IDF pipeline")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--evaluate", action="store_true", help="Evaluate on CSV file")
    group.add_argument("--predict", type=str, help="Predict single text")
    group.add_argument("--text", type=str, dest="predict", help="Predict single text (alias for --predict)")
    group.add_argument("--predict-file", type=str, help="Predict multiple texts from file (one per line)")

    parser.add_argument("--data", type=str, help="Path to CSV file for evaluation")

    args = parser.parse_args(argv)

    pipeline = load_model()
    meta = load_meta()
    if meta:
        print("Loaded metadata:", json.dumps(meta, ensure_ascii=False))

    if args.evaluate:
        if not args.data:
            parser.error("--evaluate requires --data PATH to a CSV file")
        evaluate(pipeline, args.data)
    elif args.predict:
        predict_texts(pipeline, [args.predict])
    elif args.predict_file:
        if not os.path.exists(args.predict_file):
            raise FileNotFoundError(args.predict_file)
        with open(args.predict_file, "r", encoding="utf-8") as f:
            lines = [l.strip() for l in f if l.strip()]
        predict_texts(pipeline, lines)


if __name__ == "__main__":
    main()
