"""
KolamAI Backend - Flask API
Loads trained KolamNetV2 (EfficientNetB3) for real ML predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import base64
import io
import math
import random
import os
import traceback
from PIL import Image, ImageDraw, ImageFilter

app = Flask(__name__)
CORS(app)

# ─────────────────────────────────────────────
# LOAD TRAINED MODEL (once at startup)
# ─────────────────────────────────────────────
MODEL_PATH = "kolam_model.keras"
IMG_SIZE   = 300

CLASSES = [
    "Dot_Grid",
    "Flower_Motif",
    "Geometric",
    "Sikku_Pattern",
    "Spiral_Design",
    "Star_Pattern",
]

CLASS_DESCRIPTIONS = {
    "Dot_Grid":      "Lattice dot-based kolam with structured grid symmetry",
    "Flower_Motif":  "Floral motif kolam with radial petal symmetry",
    "Geometric":     "Geometric kolam with angular precision and sharp lines",
    "Sikku_Pattern": "Sikku kolam with interlocking loops and high complexity",
    "Spiral_Design": "Spiral / curved continuous-line kolam",
    "Star_Pattern":  "Star-shaped kolam with radiating symmetry points",
}

# Try to load the trained model
ml_model = None
try:
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input as efficientnet_preprocess
    ml_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print(f"✅ Trained model loaded from {MODEL_PATH}")
except Exception as e:
    print(f"⚠️  Model not found or incompatible ({type(e).__name__}).")
    print("   Falling back to OpenCV rule-based classification.")

# ─────────────────────────────────────────────
# UTILITIES
# ─────────────────────────────────────────────
def decode_image(b64_string):
    if "," in b64_string:
        b64_string = b64_string.split(",")[1]
    img_bytes = base64.b64decode(b64_string)
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return img

def encode_image(img_array):
    _, buffer = cv2.imencode(".png", img_array)
    return "data:image/png;base64," + base64.b64encode(buffer).decode("utf-8")

def pil_to_b64(pil_img):
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("utf-8")

# ─────────────────────────────────────────────
# ML PREDICTION (uses trained model if available)
# ─────────────────────────────────────────────
def predict_with_model(img_bgr):
    """
    Run the trained EfficientNetB3 model on a BGR image.
    Returns (class_name, confidence, all_probs_dict)
    """
    # Convert BGR → RGB, resize, normalise
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img_resized.astype("float32") / 255.0, axis=0)

    probs = ml_model.predict(img_array, verbose=0)[0]
    idx = int(np.argmax(probs))
    confidence = float(probs[idx]) * 100

    all_probs = {CLASSES[i]: round(float(probs[i]) * 100, 2) for i in range(len(CLASSES))}
    return CLASSES[idx], confidence, all_probs

# ─────────────────────────────────────────────
# OPENCV ANALYSIS HELPERS (always run, model-independent)
# ─────────────────────────────────────────────
def analyze_symmetry(img_gray):
    h, w = img_gray.shape
    top    = img_gray[: h // 2, :]
    bottom = cv2.flip(img_gray[h // 2:, :], 0)
    min_h  = min(top.shape[0], bottom.shape[0])
    h_sym  = 1 - np.mean(np.abs(top[:min_h].astype(float) - bottom[:min_h].astype(float))) / 255

    left  = img_gray[:, : w // 2]
    right = cv2.flip(img_gray[:, w // 2:], 1)
    min_w = min(left.shape[1], right.shape[1])
    v_sym = 1 - np.mean(np.abs(left[:, :min_w].astype(float) - right[:, :min_w].astype(float))) / 255

    rotated = cv2.rotate(img_gray, cv2.ROTATE_180)
    rot_sym = 1 - np.mean(np.abs(img_gray.astype(float) - rotated.astype(float))) / 255

    return {
        "horizontal": round(float(h_sym) * 100, 1),
        "vertical":   round(float(v_sym) * 100, 1),
        "rotational": round(float(rot_sym) * 100, 1),
        "overall":    round(float((h_sym + v_sym + rot_sym) / 3) * 100, 1),
    }

def detect_dots(img_gray):
    blurred = cv2.GaussianBlur(img_gray, (5, 5), 0)
    circles = cv2.HoughCircles(
        blurred, cv2.HOUGH_GRADIENT, dp=1, minDist=10,
        param1=50, param2=25, minRadius=2, maxRadius=15,
    )
    if circles is not None:
        return len(circles[0])
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return len([c for c in contours if 4 < cv2.contourArea(c) < 200])

def detect_lines(img_gray):
    edges = cv2.Canny(img_gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30,
                             minLineLength=20, maxLineGap=5)
    return len(lines) if lines is not None else 0

def compute_complexity(img_gray):
    lap = cv2.Laplacian(img_gray, cv2.CV_64F)
    return round(float(np.var(lap)) / 100, 2)

def infer_grid(img_gray):
    fft       = np.fft.fft2(img_gray)
    magnitude = np.abs(np.fft.fftshift(fft))
    rows      = np.sum(magnitude, axis=1)
    peaks     = np.where((rows[1:-1] > rows[:-2]) & (rows[1:-1] > rows[2:]))[0]
    grid_size = max(3, min(13, len(peaks) // 2 + 3))
    return grid_size if grid_size % 2 == 1 else grid_size + 1

# Fallback rule-based classifier (used when model is not loaded)
def classify_pattern_rules(symmetry, dot_count, line_count):
    score = symmetry["overall"]
    if score > 85 and dot_count > 20:
        return "Dot_Grid"
    elif score > 75 and line_count > 30:
        return "Sikku_Pattern"
    elif score > 70 and dot_count > 10:
        return "Flower_Motif"
    elif score > 65:
        return "Geometric"
    elif line_count > 25:
        return "Spiral_Design"
    else:
        return "Star_Pattern"

# ─────────────────────────────────────────────
# RECREATION ENGINE
# ─────────────────────────────────────────────
def draw_pulli_kolam(grid_n, size=512, style="classic"):
    img  = Image.new("RGB", (size, size), (20, 10, 35))
    draw = ImageDraw.Draw(img)
    step   = size // (grid_n + 2)
    offset = step
    color_line   = (255, 200, 100) if style == "classic" else (100, 220, 255)
    color_dot    = (255, 255, 220)
    color_accent = (255, 120, 60)  if style == "classic" else (255, 80, 180)

    dots = []
    for i in range(grid_n):
        for j in range(grid_n):
            x, y = offset + j * step, offset + i * step
            dots.append((x, y))
            draw.ellipse([x - 4, y - 4, x + 4, y + 4], fill=color_dot)

    for idx, (x, y) in enumerate(dots):
        if idx % 3 == 0 and idx + grid_n < len(dots):
            nx, ny = dots[idx + grid_n]
            mx = (x + nx) // 2 + random.randint(-step // 3, step // 3)
            my = (y + ny) // 2 + random.randint(-step // 3, step // 3)
            draw.line([(x, y), (mx, my), (nx, ny)], fill=color_line, width=2)
        if idx % 5 == 0 and idx + 1 < len(dots):
            draw.line([(x, y), dots[idx + 1]], fill=color_accent, width=1)

    cx, cy = size // 2, size // 2
    for r in range(step // 2, step * 2, step // 3):
        draw.arc([cx - r, cy - r, cx + r, cy + r], 0, 360, fill=color_accent, width=2)

    return img.filter(ImageFilter.GaussianBlur(radius=0.5))

def draw_sikku_kolam(grid_n, size=512):
    img  = Image.new("RGB", (size, size), (15, 25, 40))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    step   = size // (grid_n + 2)
    colors = [(255, 180, 50), (255, 100, 150), (100, 220, 200), (200, 150, 255)]

    for layer in range(1, grid_n // 2 + 2):
        r     = layer * step
        color = colors[layer % len(colors)]
        draw.line([(cx, cy - r), (cx + r, cy), (cx, cy + r), (cx - r, cy), (cx, cy - r)],
                  fill=color, width=2)
        for angle in [0, 90, 180, 270]:
            rad = math.radians(angle)
            px, py = int(cx + r * math.cos(rad)), int(cy + r * math.sin(rad))
            ar = step // 2
            draw.arc([px - ar, py - ar, px + ar, py + ar], angle, angle + 90, fill=color, width=2)

    for i in range(grid_n):
        for j in range(grid_n):
            x = (size // (grid_n + 1)) * (j + 1)
            y = (size // (grid_n + 1)) * (i + 1)
            draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=(255, 255, 220))

    return img.filter(ImageFilter.GaussianBlur(radius=0.4))

def draw_kambi_kolam(grid_n, size=512):
    img  = Image.new("RGB", (size, size), (10, 20, 15))
    draw = ImageDraw.Draw(img)
    cx, cy = size // 2, size // 2
    color_main   = (80, 220, 140)
    color_accent = (255, 200, 80)
    petals = 8
    for layer in range(1, min(grid_n, 6)):
        r_outer = layer * size // (grid_n * 2)
        r_inner = r_outer * 0.6
        for p in range(petals * layer):
            angle = (2 * math.pi / (petals * layer)) * p
            ox = cx + r_outer * math.cos(angle)
            oy = cy + r_outer * math.sin(angle)
            ix = cx + r_inner * math.cos(angle + math.pi / (petals * layer))
            iy = cy + r_inner * math.sin(angle + math.pi / (petals * layer))
            draw.line([(cx, cy), (int(ox), int(oy))], fill=color_main, width=1)
            draw.line([(int(ox), int(oy)), (int(ix), int(iy))], fill=color_accent, width=2)
    for r in [size // 3, size // 2 - 20]:
        draw.arc([cx - r, cy - r, cx + r, cy + r], 0, 360, fill=color_accent, width=2)
    return img.filter(ImageFilter.GaussianBlur(radius=0.5))

# ─────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────
@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "model_loaded": ml_model is not None,
        "version": "2.0.0",
    })


@app.route("/api/analyze", methods=["POST"])
def analyze():
    try:
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        img      = decode_image(data["image"])
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # OpenCV structural analysis (always runs)
        symmetry   = analyze_symmetry(img_gray)
        dot_count  = detect_dots(img_gray)
        line_count = detect_lines(img_gray)
        complexity = compute_complexity(img_gray)
        grid_size  = infer_grid(img_gray)

        # Classification — ML model preferred, rules as fallback
        if ml_model is not None:
            pattern_type, confidence, all_probs = predict_with_model(img)
            classification_source = "ml_model"
        else:
            pattern_type = classify_pattern_rules(symmetry, dot_count, line_count)
            confidence   = None
            all_probs    = {}
            classification_source = "rule_based"

        description = CLASS_DESCRIPTIONS.get(pattern_type, "")

        # Edge visualisation
        edges         = cv2.Canny(img_gray, 50, 150)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        edge_b64      = encode_image(edges_colored)

        response = {
            "success": True,
            "pattern_type": pattern_type,
            "description": description,
            "classification_source": classification_source,
            "symmetry": symmetry,
            "dot_count": dot_count,
            "line_count": line_count,
            "complexity_score": complexity,
            "inferred_grid": grid_size,
            "edge_visualization": edge_b64,
            "design_principles": [
                f"Grid-based dot layout ({grid_size}×{grid_size})",
                f"Symmetry type: {'4-fold' if symmetry['rotational'] > 70 else '2-fold'}",
                "Line-loop threading pattern" if line_count > 30 else "Curved stroke motifs",
                f"Pattern complexity: {'High' if complexity > 50 else 'Medium' if complexity > 20 else 'Low'}",
            ],
        }

        if confidence is not None:
            response["confidence"] = round(confidence, 2)
            response["all_probabilities"] = all_probs

        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


@app.route("/api/recreate", methods=["POST"])
def recreate():
    try:
        data       = request.get_json()
        style      = data.get("style", "pulli")
        grid_n     = max(3, min(15, int(data.get("grid", 7))))
        color_style= data.get("color_style", "classic")

        if style == "pulli":
            img = draw_pulli_kolam(grid_n, style=color_style)
        elif style == "sikku":
            img = draw_sikku_kolam(grid_n)
        else:
            img = draw_kambi_kolam(grid_n)

        return jsonify({"success": True, "image": pil_to_b64(img), "style": style, "grid": grid_n})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/generate", methods=["POST"])
def generate():
    try:
        data         = request.get_json()
        pattern_type = data.get("pattern_type", "Dot_Grid")
        grid         = max(3, min(13, int(data.get("grid", 7))))
        variations   = int(data.get("variations", 4))

        style_map = {
            "Dot_Grid":      "pulli",
            "Flower_Motif":  "kambi",
            "Geometric":     "sikku",
            "Sikku_Pattern": "sikku",
            "Spiral_Design": "kambi",
            "Star_Pattern":  "pulli",
        }
        style = style_map.get(pattern_type, "pulli")

        results = []
        for i in range(variations):
            g = max(3, min(13, grid + (i - variations // 2)))
            color_styles = ["classic", "colored", "classic", "colored"]
            if style == "pulli":
                img = draw_pulli_kolam(g, style=color_styles[i % len(color_styles)])
            elif style == "sikku":
                img = draw_sikku_kolam(g)
            else:
                img = draw_kambi_kolam(g)
            results.append({"image": pil_to_b64(img), "grid": g, "variant": i + 1})

        return jsonify({"success": True, "variations": results, "pattern_type": pattern_type})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/dataset/stats", methods=["GET"])
def dataset_stats():
    return jsonify({
        "total_images": 5631,
        "categories": {
            "Dot_Grid":      1000,
            "Flower_Motif":  1021,
            "Geometric":     1010,
            "Sikku_Pattern": 600,
            "Spiral_Design": 1000,
            "Star_Pattern":  1000,
        },
        "model": "KolamNetV2 (EfficientNetB3 + Custom Head)",
        "model_loaded": ml_model is not None,
        "augmentation": ["rotation±40°", "flip H/V", "zoom±25%",
                         "brightness", "shear", "channel shift"],
        "split": {"train": 0.8, "val": 0.2},
    })


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)