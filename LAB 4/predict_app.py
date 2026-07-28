"""
predict_app.py — Application layer for the Wine MLP classifier
------------------------------------------------------------------
This is the "application" built on top of mlp_multiclass.py.
It loads the already-trained model (mlp_wine_model.keras), the
StandardScaler used during training (scaler.pkl), and metadata
(meta.pkl), then lets a user type in real wine measurements and get
a live prediction with class probabilities.

IMPORTANT: run mlp_multiclass.py first (once) to generate
mlp_wine_model.keras / scaler.pkl / meta.pkl. This script does not
retrain anything — it simulates how a trained model would be used
in a real product (e.g. a lab technician entering test results).

Run: python predict_app.py
"""

import sys
import numpy as np
import joblib
from tensorflow import keras

MODEL_PATH = "mlp_wine_model.keras"
SCALER_PATH = "scaler.pkl"
META_PATH = "meta.pkl"

# Typical real-world ranges (from the training data) shown as hints to the user.
# min, mean, max — helps someone unfamiliar with wine chemistry enter sane values.
FEATURE_HINTS = {
    "alcohol":                       (11.03, 13.00, 14.83),
    "malic_acid":                    (0.74, 2.34, 5.80),
    "ash":                           (1.36, 2.37, 3.23),
    "alcalinity_of_ash":             (10.60, 19.49, 30.00),
    "magnesium":                     (70.00, 99.74, 162.00),
    "total_phenols":                 (0.98, 2.30, 3.88),
    "flavanoids":                    (0.34, 2.03, 5.08),
    "nonflavanoid_phenols":          (0.13, 0.36, 0.66),
    "proanthocyanins":               (0.41, 1.59, 3.58),
    "color_intensity":               (1.28, 5.06, 13.00),
    "hue":                           (0.48, 0.96, 1.71),
    "od280/od315_of_diluted_wines":  (1.27, 2.61, 4.00),
    "proline":                       (278.00, 746.89, 1680.00),
}


def load_artifacts():
    try:
        model = keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        meta = joblib.load(META_PATH)
        return model, scaler, meta
    except FileNotFoundError:
        print("ERROR: trained artifacts not found.")
        print("Run 'python mlp_multiclass.py' first to train and save the model.")
        sys.exit(1)


def get_sample_from_dataset(index, meta):
    """Fallback / demo mode: predict on a known sample from the dataset by index."""
    from sklearn.datasets import load_wine
    data = load_wine()
    if not (0 <= index < len(data.data)):
        raise ValueError(f"Index must be between 0 and {len(data.data) - 1}")
    return data.data[index], data.target[index]


def prompt_manual_input(feature_names):
    print("\nEnter the 13 wine measurements below.")
    print("(Press Enter with no input to auto-fill the average value shown in brackets.)\n")
    values = []
    for name in feature_names:
        lo, mean, hi = FEATURE_HINTS.get(name, (None, None, None))
        hint = f"  [typical range {lo}-{hi}, avg {mean}]" if mean is not None else ""
        while True:
            raw = input(f"{name}{hint}: ").strip()
            if raw == "":
                values.append(mean)
                break
            try:
                values.append(float(raw))
                break
            except ValueError:
                print("  Please enter a numeric value.")
    return np.array(values, dtype=float)


def predict(model, scaler, meta, raw_features):
    class_names = meta["class_names"]
    X = np.array(raw_features, dtype=float).reshape(1, -1)
    X_scaled = scaler.transform(X)
    probs = model.predict(X_scaled, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    return class_names[pred_idx], probs, class_names


def print_prediction(pred_class, probs, class_names, true_label=None):
    print("\n--- Prediction ---")
    print(f"Predicted cultivar: {pred_class}")
    print("Class probabilities:")
    for name, p in zip(class_names, probs):
        bar = "#" * int(p * 40)
        marker = "  <-- predicted" if name == pred_class else ""
        print(f"  {name:10s} {p*100:6.2f}%  {bar}{marker}")
    if true_label is not None:
        print(f"(Actual label for this sample: {class_names[true_label]})")
    print()


def main():
    model, scaler, meta = load_artifacts()
    feature_names = meta["feature_names"]
    class_names = meta["class_names"]

    print("=" * 60)
    print(" Wine Cultivar Classifier — Application Demo")
    print(" Model: Keras MLP  |  Classes:", ", ".join(class_names))
    print("=" * 60)

    while True:
        print("\nChoose input mode:")
        print("  1) Enter measurements manually")
        print("  2) Predict a sample from the dataset by index (0-177) [demo/testing]")
        print("  3) Quit")
        choice = input("Choice: ").strip()

        if choice == "1":
            raw = prompt_manual_input(feature_names)
            pred_class, probs, class_names = predict(model, scaler, meta, raw)
            print_prediction(pred_class, probs, class_names)

        elif choice == "2":
            try:
                idx = int(input("Enter dataset index (0-177): ").strip())
                raw, true_label = get_sample_from_dataset(idx, meta)
                pred_class, probs, class_names = predict(model, scaler, meta, raw)
                print_prediction(pred_class, probs, class_names, true_label=true_label)
            except ValueError as e:
                print(f"  {e}")

        elif choice == "3":
            print("Goodbye.")
            break

        else:
            print("Please choose 1, 2, or 3.")


if __name__ == "__main__":
    main()
