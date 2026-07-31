"""
Generates a synthetic two-class image dataset (cat-like vs dog-like faces)
because this sandboxed environment has no internet access to Kaggle /
tensorflow-datasets storage. Images are simple procedurally-drawn faces with
class-distinguishing features (ear shape, eye shape, snout), plus random
color/position/rotation/noise so the classification task is non-trivial and
augmentation actually matters.

Class 0 = "cat"  -> triangular pointed ears, slit eyes, small nose
Class 1 = "dog"  -> floppy rounded ears, round eyes, protruding snout+tongue

Folder layout produced (matches ImageDataGenerator.flow_from_directory):
  data/train/cats/*.png   data/train/dogs/*.png
  data/val/cats/*.png     data/val/dogs/*.png
"""
import os
import random
import numpy as np
from PIL import Image, ImageDraw

random.seed(42)
np.random.seed(42)

IMG_SIZE = 64
N_TRAIN_PER_CLASS = 300
N_VAL_PER_CLASS = 60


def rand_color(base, jitter=40):
    return tuple(int(np.clip(c + random.randint(-jitter, jitter), 0, 255)) for c in base)


def draw_face(is_dog: bool) -> Image.Image:
    img = Image.new("RGB", (IMG_SIZE, IMG_SIZE), rand_color((235, 235, 225), 15))
    draw = ImageDraw.Draw(img)

    cx, cy = IMG_SIZE // 2 + random.randint(-4, 4), IMG_SIZE // 2 + random.randint(-4, 4)
    r = random.randint(18, 24)
    fur = rand_color((150, 100, 60) if not is_dog else (200, 160, 90), 30)

    # ears
    ear_h = random.randint(10, 16)
    if is_dog:
        # floppy rounded ears (ellipses hanging at the sides)
        draw.ellipse([cx - r - 8, cy - 4, cx - r + 6, cy - 4 + ear_h + 10], fill=fur)
        draw.ellipse([cx + r - 6, cy - 4, cx + r + 8, cy - 4 + ear_h + 10], fill=fur)
    else:
        # triangular pointed ears
        draw.polygon([(cx - r, cy - r // 2), (cx - r + 10, cy - r - ear_h), (cx - r + 18, cy - r // 2)], fill=fur)
        draw.polygon([(cx + r - 18, cy - r // 2), (cx + r - 10, cy - r - ear_h), (cx + r, cy - r // 2)], fill=fur)

    # face
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=fur)

    # eyes
    eye_color = (20, 20, 20)
    if is_dog:
        draw.ellipse([cx - 10, cy - 4, cx - 4, cy + 2], fill=eye_color)
        draw.ellipse([cx + 4, cy - 4, cx + 10, cy + 2], fill=eye_color)
    else:
        draw.line([(cx - 10, cy - 1), (cx - 4, cy - 1)], fill=eye_color, width=2)
        draw.line([(cx + 4, cy - 1), (cx + 10, cy - 1)], fill=eye_color, width=2)

    # nose / snout
    if is_dog:
        draw.ellipse([cx - 6, cy + 4, cx + 6, cy + 14], fill=rand_color((230, 200, 160)))
        draw.ellipse([cx - 3, cy + 10, cx + 3, cy + 14], fill=(30, 30, 30))
        draw.line([(cx, cy + 14), (cx, cy + 20)], fill=(200, 60, 90), width=3)  # tongue
    else:
        draw.polygon([(cx - 2, cy + 3), (cx + 2, cy + 3), (cx, cy + 6)], fill=(200, 120, 140))
        draw.line([(cx - 14, cy + 6), (cx - 2, cy + 4)], fill=(80, 80, 80), width=1)  # whisker
        draw.line([(cx + 14, cy + 6), (cx + 2, cy + 4)], fill=(80, 80, 80), width=1)

    # random rotation + noise for realism / augmentation relevance
    img = img.rotate(random.randint(-12, 12), fillcolor=rand_color((235, 235, 225), 15))
    arr = np.array(img).astype(np.int16)
    noise = np.random.randint(-10, 10, arr.shape)
    arr = np.clip(arr + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


def build_split(split: str, n_per_class: int, out_root: str):
    for label, cls in [(0, "cats"), (1, "dogs")]:
        d = os.path.join(out_root, split, cls)
        os.makedirs(d, exist_ok=True)
        for i in range(n_per_class):
            face = draw_face(is_dog=bool(label))
            face.save(os.path.join(d, f"{cls}_{i:04d}.png"))


if __name__ == "__main__":
    root = os.path.join(os.path.dirname(__file__), "data")
    build_split("train", N_TRAIN_PER_CLASS, root)
    build_split("val", N_VAL_PER_CLASS, root)
    print("Dataset generated at", root)
