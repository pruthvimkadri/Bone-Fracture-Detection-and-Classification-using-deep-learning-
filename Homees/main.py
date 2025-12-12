#!/usr/bin/env python3
"""
main.py — Frac-AI (updated: use PyTorch xray_validation_model.pt for X-ray validation)
Only X-ray validation / gradcam parts converted from Keras -> PyTorch.
Other app routes and DB code kept intact with minimal changes.
"""
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
torch.set_num_threads(1)
torch.set_num_interop_threads(1)

import os
import io
import sqlite3
import random
import json
import traceback
import time
from datetime import datetime, timedelta
from functools import wraps
from email.message import EmailMessage

from flask import (
    Flask, render_template, request, redirect, url_for, session,
    flash, jsonify, send_file, abort, current_app
)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from dotenv import load_dotenv
from PIL import Image, ImageOps
import numpy as np
import cv2

# ---- PyTorch (for xray validation + other models) ----
try:
    import torch
    import torch.nn.functional as F
    import torchvision
    from torchvision import transforms
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False
    transforms = None

# PDF (reportlab)
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

# ---- Authentication decorator ----
from functools import wraps as _wraps
from flask import session as _session, redirect as _redirect, url_for as _url_for

def login_required(f):
    @_wraps(f)
    def decorated(*args, **kwargs):
        if 'user_id' not in session:
            return redirect(url_for('login'))
        return f(*args, **kwargs)
    return decorated

# ---- Config ----
load_dotenv()
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
STATIC_DIR = os.path.join(BASE_DIR, 'static')
MODEL_DIR = os.path.join(STATIC_DIR, 'models')
UPLOAD_FOLDER = os.path.join(STATIC_DIR, 'uploads')
RESULT_FOLDER = os.path.join(STATIC_DIR, 'results')
TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
DB_PATH = os.path.join(BASE_DIR, 'app.db')
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp'}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULT_FOLDER, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

app = Flask(__name__, template_folder=TEMPLATE_FOLDER, static_folder=STATIC_DIR)
app.secret_key = os.getenv('SECRET_KEY', 'sai_abhyankkar_04_november_2004')

# Email config (optional)
MAIL_USERNAME = os.getenv('MAIL_USERNAME')
MAIL_PASSWORD = os.getenv('MAIL_PASSWORD')
MAIL_SERVER = os.getenv('MAIL_SERVER', 'smtp.gmail.com')
MAIL_PORT = int(os.getenv('MAIL_PORT', 587) or 587)
MAIL_USE_TLS = os.getenv('MAIL_USE_TLS', 'True').lower() in ('true', '1', 'yes')
_mail_user = os.getenv("MAIL_USERNAME")
_mail_pass = os.getenv("MAIL_PASSWORD")
print(f"[DEBUG] MAIL_USERNAME: {_mail_user}")
print(f"[DEBUG] MAIL_PASSWORD present: {bool(_mail_pass)}")

# Paths to model files (expected)
XRAY_PTH_PATH = os.path.join(MODEL_DIR, 'xray_validation_model.pt')   # <-- new PyTorch model for validation
BINARY_PTH_PATH = os.path.join(MODEL_DIR, 'binary_classifier.pt')
MULTI_PTH_PATH = os.path.join(MODEL_DIR, 'multi_classifier.pth')

# ---- DB helpers ----
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    conn = get_db()
    cur = conn.cursor()
    cur.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            first_name TEXT,
            middle_name TEXT,
            last_name TEXT,
            email TEXT UNIQUE,
            phone TEXT,
            gender TEXT,
            age INTEGER,
            password_hash TEXT,
            created_at TEXT
        );
    ''')
    cur.execute('''
    CREATE TABLE IF NOT EXISTS results (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        filename TEXT,
        stage TEXT,
        predicted TEXT,
        confidence REAL,
        gradcam TEXT,
        canny TEXT,
        hybrid TEXT,
        report_path TEXT,
        fracture_type TEXT,
        created_at TEXT
    );
    ''')
    cur.execute('''
        CREATE TABLE IF NOT EXISTS otps (
            email TEXT PRIMARY KEY,
            otp TEXT,
            expires_at TEXT
        );
    ''')
    conn.commit()
    conn.close()

init_db()

# ---- Model loading (PyTorch) ----
xray_model = None
binary_model = None
multi_model = None
torch_transform = None

def safe_log(msg, level='INFO'):
    print(f"{datetime.now().isoformat()} [{level}] {msg}")

def build_mobilenet_xray(num_classes=3, device='cpu'):
    """
    Create a MobileNetV2 skeleton like your training script:
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, 3)
    """
    if not TORCH_AVAILABLE:
        return None
    try:
        model = torchvision.models.mobilenet_v2(weights=torchvision.models.MobileNet_V2_Weights.IMAGENET1K_V1)
    except Exception:
        # fallback to weights=None if torchvision version doesn't provide enum
        model = torchvision.models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features if hasattr(model.classifier[1], 'in_features') else getattr(model, 'last_channel', None)
    model.classifier[1] = torch.nn.Linear(model.last_channel, num_classes)
    return model

def ensure_models_loaded():
    global xray_model, binary_model, multi_model, torch_transform
    # Prepare Torch transforms (consistent with training)
    if TORCH_AVAILABLE:
        torch_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])
        ])
    else:
        torch_transform = None

    # Load X-ray validation model (PyTorch)
    safe_log(f"🔍 Looking for xray validation model at: {XRAY_PTH_PATH}")
    if TORCH_AVAILABLE and os.path.exists(XRAY_PTH_PATH):
        try:
            # Try loading file — could be state_dict or whole model
            loaded = torch.load(XRAY_PTH_PATH, map_location=torch.device('cpu'))
            if isinstance(loaded, dict) and not any(isinstance(v, torch.nn.Module) for v in loaded.values()):
                # Likely a state_dict
                model_skel = build_mobilenet_xray(num_classes=3)
                try:
                    model_skel.load_state_dict(loaded, strict=False)
                except Exception as e:
                    safe_log(f"Warning: strict load failed for xray state_dict: {e}", 'WARN')
                    model_skel.load_state_dict(loaded, strict=False)
                model_skel.eval()
                xray_model = model_skel
                safe_log("✅ Loaded xray_validation_model.pt as state_dict into MobileNetV2 skeleton")
            else:
                # loaded is either a Module or contains keys like 'model_state_dict'
                if isinstance(loaded, dict) and 'model_state_dict' in loaded:
                    state = loaded['model_state_dict']
                    model_skel = build_mobilenet_xray(num_classes=3)
                    try:
                        model_skel.load_state_dict(state, strict=False)
                        model_skel.eval()
                        xray_model = model_skel
                        safe_log("✅ Loaded xray_validation_model.pt from model_state_dict")
                    except Exception as e:
                        safe_log(f"Failed to load model_state_dict into skeleton: {e}", 'ERROR')
                        xray_model = None
                elif isinstance(loaded, torch.nn.Module):
                    loaded.eval()
                    xray_model = loaded
                    safe_log("✅ Loaded xray_validation_model.pt as full Module")
                else:
                    # last resort — attempt to treat as state_dict
                    try:
                        model_skel = build_mobilenet_xray(num_classes=3)
                        model_skel.load_state_dict(loaded, strict=False)
                        model_skel.eval()
                        xray_model = model_skel
                        safe_log("✅ Loaded xray_validation_model.pt (fallback)")
                    except Exception as e:
                        safe_log(f"Could not interpret xray file: {e}", 'ERROR')
                        xray_model = None
        except Exception as e:
            safe_log(f"Error loading xray model: {e}", level='ERROR')
            xray_model = None
    else:
        safe_log("Torch not available or xray model missing.", level='WARN')
        xray_model = None

    # The rest: binary & multi loaders (keep prior behavior)
    safe_log(f"🔍 Looking for binary model at: {BINARY_PTH_PATH}")
    if TORCH_AVAILABLE and os.path.exists(BINARY_PTH_PATH):
        try:
            state = torch.load(BINARY_PTH_PATH, map_location=torch.device('cpu'))
            inferred_num = None
            if isinstance(state, dict):
                for k, v in state.items():
                    if k.endswith('fc.weight') and hasattr(v, 'shape'):
                        inferred_num = int(v.shape[0])
                        break
                if 'model_state_dict' in state and (inferred_num is None):
                    sd = state['model_state_dict']
                    for k, v in sd.items():
                        if k.endswith('fc.weight') and hasattr(v, 'shape'):
                            inferred_num = int(v.shape[0])
                            break
                    state = sd
            res = torchvision.models.resnet50(weights=None)
            in_features = res.fc.in_features
            num_classes = int(inferred_num or 2)
            res.fc = torch.nn.Linear(in_features, num_classes)
            try:
                res.load_state_dict(state, strict=False)
                res.eval()
                binary_model = res
                safe_log(f"✅ PyTorch binary model loaded (num_classes={num_classes})")
            except Exception as e:
                safe_log(f"PyTorch binary load error: {e}", level='ERROR')
                binary_model = None
        except Exception as e:
            safe_log(f"PyTorch binary load error: {e}", level='ERROR')
            binary_model = None
    else:
        safe_log("Torch not available or binary model missing.", level='WARN')
        binary_model = None

    # Load multi classifier (PyTorch)
    safe_log(f"🔍 Looking for multi model at: {MULTI_PTH_PATH}")
    if TORCH_AVAILABLE and os.path.exists(MULTI_PTH_PATH):
        try:
            state = torch.load(MULTI_PTH_PATH, map_location=torch.device('cpu'))
            inferred_num = None
            if isinstance(state, dict):
                for k, v in state.items():
                    if k.endswith('fc.weight') and hasattr(v, 'shape'):
                        inferred_num = int(v.shape[0])
                        break
                if 'model_state_dict' in state and (inferred_num is None):
                    sd = state['model_state_dict']
                    for k, v in sd.items():
                        if k.endswith('fc.weight') and hasattr(v, 'shape'):
                            inferred_num = int(v.shape[0])
                            break
                    state = sd
            res = torchvision.models.resnet50(weights=None)
            in_features = res.fc.in_features
            num_classes = int(inferred_num or 11)
            res.fc = torch.nn.Linear(in_features, num_classes)
            try:
                res.load_state_dict(state, strict=False)
                res.eval()
                multi_model = res
                safe_log(f"✅ PyTorch multi model loaded (num_classes={num_classes})")
            except Exception as e:
                safe_log(f"PyTorch multi load error: {e}", level='ERROR')
                multi_model = None
        except Exception as e:
            safe_log(f"PyTorch multi load error: {e}", level='ERROR')
            multi_model = None
    else:
        safe_log("Torch not available or multi model missing.", level='WARN')
        multi_model = None

# initial load
try_load_models()

# ---- Utilities ----
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def send_otp_email(to_email, otp_code):
    if not MAIL_USERNAME or not MAIL_PASSWORD:
        safe_log("Mail not configured; skipping send", 'WARN')
        return False, "Mail not configured"
    try:
        msg = EmailMessage()
        msg['Subject'] = 'Your OTP for Frac-AI'
        msg['From'] = MAIL_USERNAME
        msg['To'] = to_email
        msg.set_content(f'Your OTP is: {otp_code}. It expires in 5 minutes.')
        import smtplib
        with smtplib.SMTP(MAIL_SERVER, MAIL_PORT) as smtp:
            if MAIL_USE_TLS:
                smtp.starttls()
            smtp.login(MAIL_USERNAME, MAIL_PASSWORD)
            smtp.send_message(msg)
        return True, "OTP sent"
    except Exception as e:
        safe_log(f"Error sending email: {e}", 'ERROR')
        return False, str(e)

def save_result_record(user_id, filename, stage, predicted, confidence, gradcam=None, canny=None, hybrid=None, report_path=None, fracture_type=None):
    conn = get_db()
    cur = conn.cursor()
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cur.execute('''
        INSERT INTO results (user_id, filename, stage, predicted, confidence, gradcam, canny, hybrid, report_path, fracture_type, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (user_id, filename, stage, predicted, confidence, gradcam, canny, hybrid, report_path, fracture_type, created_at))
    conn.commit()
    rid = cur.lastrowid
    conn.close()
    return rid

def get_user_by_email(email):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT * FROM users WHERE email = ?', (email,))
    row = cur.fetchone()
    conn.close()
    return row

# ---- Grad-CAM for PyTorch (lightweight) ----
def generate_gradcam_torch(img_path, out_path, model=None, target_layer_name=None, input_size=(224,224)):
    """
    Lightweight grad-cam for PyTorch model.
    - model: torch.nn.Module (eval)
    - target_layer_name: name of last conv layer (if None, find first 4D output layer by forward hook)
    This function writes overlayed heatmap to out_path (BGR).
    """
    if not TORCH_AVAILABLE or model is None:
        raise RuntimeError("PyTorch not available or model not provided for gradcam")

    model.eval()
    device = torch.device('cpu')

    # Prepare image tensor
    img = Image.open(img_path).convert('RGB')
    orig_w, orig_h = img.size
    preprocess = transforms.Compose([
        transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # find a candidate conv layer
    target_layer = None
    if target_layer_name:
        try:
            target_layer = dict(model.named_modules()).get(target_layer_name, None)
        except Exception:
            target_layer = None

    if target_layer is None:
        # find last module that outputs 4D feature map by hooking
        activations = {}
        def hook_fn(module, input, output):
            activations['feat'] = output.detach()

        # try to attach to common conv layers by reverse order
        for name, module in reversed(list(model.named_modules())):
            # choose module with 4D output if possible
            try:
                # attach hook, run one forward, then remove
                h = module.register_forward_hook(hook_fn)
                with torch.no_grad():
                    _ = model(input_tensor)
                h.remove()
                if 'feat' in activations:
                    target_layer = module
                    break
            except Exception:
                activations.clear()
                continue

    if target_layer is None and 'feat' not in activations:
        # can't find conv activation, fallback to simple color map of grayscale
        gray = cv2.imread(img_path, 0)
        heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET) if gray is not None else None
        if heat is not None:
            cv2.imwrite(out_path, heat)
            return out_path
        else:
            raise RuntimeError("GradCAM: cannot produce heatmap")

    # compute gradcam using last forward/backward pass with registered hooks
    features = {}
    gradients = {}

    def forward_hook(module, input, output):
        features['value'] = output.detach()

    def backward_hook(module, grad_in, grad_out):
        gradients['value'] = grad_out[0].detach()

    handle_fwd = target_layer.register_forward_hook(forward_hook)
    handle_bwd = target_layer.register_backward_hook(backward_hook)

    try:
        input_tensor.requires_grad = True
        out = model(input_tensor)
        # ensure 2D or 1D output -> pick highest logit
        if isinstance(out, (tuple, list)):
            out_tensor = out[-1]
        else:
            out_tensor = out
        # if batch dim
        if out_tensor.dim() == 2:
            score = out_tensor[0, out_tensor[0].argmax()]
        elif out_tensor.dim() == 1:
            score = out_tensor[out_tensor.argmax()]
        else:
            score = out_tensor.flatten().max()

        model.zero_grad()
        score.backward(retain_graph=False)

        if 'value' not in features or 'value' not in gradients:
            raise RuntimeError("GradCAM: failed to capture activations/gradients")

        fmap = features['value'][0].cpu().numpy()  # CxHxW
        grads = gradients['value'][0].cpu().numpy()  # CxHxW
        weights = np.mean(grads, axis=(1,2))  # C
        cam = np.zeros(fmap.shape[1:], dtype=np.float32)  # HxW
        for i, w in enumerate(weights):
            cam += w * fmap[i, :, :]
        cam = np.maximum(cam, 0)
        cam = cam - np.min(cam)
        cam = cam / (np.max(cam) + 1e-8)
        cam = cv2.resize(cam, (orig_w, orig_h))
        heatmap = (cam * 255).astype(np.uint8)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        orig = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        overlay = cv2.addWeighted(orig, 0.6, heatmap, 0.4, 0)
        cv2.imwrite(out_path, overlay)
        return out_path
    finally:
        try:
            handle_fwd.remove()
        except Exception:
            pass
        try:
            handle_bwd.remove()
        except Exception:
            pass

# Canny and hybrid generation (unchanged)
def generate_canny(img_path, out_path):
    img = cv2.imread(img_path, 0)
    if img is None:
        raise RuntimeError("can't read image for canny")
    v = np.median(img)
    lower = int(max(0, 0.66 * v))
    upper = int(min(255, 1.33 * v))
    edges = cv2.Canny(img, lower, upper)
    cv2.imwrite(out_path, edges)
    return out_path

def generate_hybrid(gradcam_path, canny_path, out_path):
    grad = cv2.imread(gradcam_path)
    canny = cv2.imread(canny_path, cv2.IMREAD_GRAYSCALE)

    if grad is None or canny is None:
        raise RuntimeError("Cannot read gradcam or canny image for hybrid generation.")

    if grad.shape[:2] != canny.shape[:2]:
        canny = cv2.resize(canny, (grad.shape[1], grad.shape[0]))
    canny_bgr = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)
    hybrid = cv2.addWeighted(grad, 0.7, canny_bgr, 0.3, 0)
    cv2.imwrite(out_path, hybrid)
    return out_path

# ---- Prediction wrappers ----

def validate_xray(image_path):
    """
    Validate if the uploaded image is:
    - bone_xray
    - non_bone_xray
    - non_xray

    Final mapping (default):
    0 -> bone_xray (VALID)
    1 -> non_bone_xray
    2 -> non_xray

    This function uses the PyTorch xray_model if available; otherwise a heuristic fallback.
    """
    try:
        if TORCH_AVAILABLE and xray_model is not None and torch_transform is not None:
            img = Image.open(image_path).convert('RGB')
            t = torch_transform(img).unsqueeze(0)
            with torch.no_grad():
                out = xray_model(t)
            # handle tuple/list outputs
            if isinstance(out, (tuple, list)):
                out_tensor = out[-1]
            else:
                out_tensor = out
            # if batched: take first
            if out_tensor.dim() == 2 and out_tensor.shape[0] == 1:
                out_np = out_tensor[0].cpu().numpy()
            elif out_tensor.dim() == 1:
                out_np = out_tensor.cpu().numpy()
            else:
                out_np = np.array(out_tensor.cpu().numpy()).flatten()

            # Convert logits to probabilities if multiple classes
            if out_np.size > 1:
                exps = np.exp(out_np - np.max(out_np))
                probs = exps / (np.sum(exps) + 1e-12)
            else:
                # single-output: use sigmoid
                probs = np.array([1.0 / (1.0 + np.exp(-out_np[0]))])

            idx = int(np.argmax(probs))
            confidence = float(np.max(probs))
            # Default mapping (keep your previous mapping).
            if idx == 0:
                return "bone_xray", True, confidence
            elif idx == 1:
                return "non_bone_xray", False, confidence
            else:
                return "non_xray", False, confidence
        else:
            # heuristic fallback: Canny edge ratio
            img = cv2.imread(image_path, 0)
            if img is None:
                return "non_xray", False, 0.0
            h, w = img.shape[:2]
            edges = cv2.Canny(img, 50, 150)
            edge_count = np.sum(edges > 0)
            ratio = edge_count / (h * w + 1e-8)
            if ratio > 0.0035:
                return "non_bone_xray", False, float(min(0.99, ratio * 100.0))
            else:
                return "non_xray", False, float(1.0 - min(0.99, ratio * 100.0))
    except Exception as e:
        safe_log(f"validate_xray error: {e}", "ERROR")
        return "error", False, 0.0

# alias used in some routes
predict_xray = validate_xray

def predict_binary(image_path):
    """
    Returns Normal OR Fractured
    Uses PyTorch binary_model if available; otherwise fallback to default.
    """
    if TORCH_AVAILABLE and binary_model and torch_transform:
        try:
            img = Image.open(image_path).convert('RGB')
            t = torch_transform(img).unsqueeze(0)
            with torch.no_grad():
                out = binary_model(t)
                probs = F.softmax(out, dim=1).cpu().numpy()[0]
            if len(probs) >= 2:
                fracture_prob = float(probs[0])
                normal_prob = float(probs[1])
                if fracture_prob > normal_prob:
                    return "Fractured", fracture_prob
                else:
                    return "Normal", normal_prob
            else:
                val = float(probs[0])
                return ("Fractured" if val > 0.5 else "Normal"), val
        except Exception as e:
            safe_log(f"binary predict error: {e}", "ERROR")
    return "Normal", 0.5

def predict_multi(image_path):
    """
    Predict specific fracture class (11 classes).
    Returns (label, confidence).
    """
    fracture_types = [
        "Avulsion",
        "Comminuted",
        "Fracture Dislocation",
        "Greenstick",
        "Hairline",
        "Impacted",
        "Longitudinal",
        "Oblique",
        "Pathological",
        "Spiral",
        "Unknown"
    ]

    if not (TORCH_AVAILABLE and multi_model and torch_transform):
        return "Unknown", 0.0

    try:
        img = Image.open(image_path).convert('RGB')
        t = torch_transform(img).unsqueeze(0)
        with torch.no_grad():
            out = multi_model(t)
        if out.dim() == 1:
            out = out.unsqueeze(0)
        probs = F.softmax(out, dim=1).cpu().numpy()[0]
        if len(probs) != len(fracture_types):
            safe_log(f"MISMATCH! model outputs {len(probs)}, expected {len(fracture_types)}", "ERROR")
            return "Unknown", float(np.max(probs))
        idx = int(np.argmax(probs))
        confidence = float(probs[idx])
        if confidence < 0.60:
            return "Unknown", confidence
        return fracture_types[idx], confidence
    except Exception as e:
        safe_log(f"multi predict error: {e}", "ERROR")
        return "Unknown", 0.0

# PDF generation (unchanged)
def generate_pdf_report(result_row, out_pdf_path):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("ReportLab not available")

    c = canvas.Canvas(out_pdf_path, pagesize=letter)
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, 770, "AI-Based Bone Fracture Diagnosis Report")
    c.setFont("Helvetica", 12)
    c.drawString(50, 750, "Generated by: Frac-AI Bone Analysis System")
    c.line(50, 745, 560, 745)

    # Patient info
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 720, "Patient Information")
    c.line(50, 717, 180, 717)
    c.setFont("Helvetica", 12)
    c.drawString(50, 700, f"Patient ID: {result_row.get('user_id', 'N/A')}")
    c.drawString(50, 685, f"Uploaded File: {result_row.get('filename', 'N/A')}")
    c.drawString(50, 670, f"Uploaded On: {result_row.get('created_at', 'N/A')}")

    # Diagnosis
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 645, "AI Diagnosis Summary")
    c.line(50, 642, 195, 642)
    c.setFont("Helvetica", 12)
    c.drawString(50, 625, f"X-Ray Validation: {result_row.get('stage', 'N/A')}")
    c.drawString(50, 610, f"Prediction: {result_row.get('predicted', 'N/A')}")
    c.drawString(50, 595, f"Confidence: {round(result_row.get('confidence', 0.0), 4)}")

    multiclass = result_row.get("fracture_type")
    if multiclass:
        c.drawString(50, 580, f"Fracture Type: {multiclass}")

    # Images
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, 550, "AI Explainability Visualizations")
    c.line(50, 547, 320, 547)
    y = 520
    img_size = (190, 150)
    images = {
        "Grad-CAM": result_row.get("gradcam"),
        "Canny Edges": result_row.get("canny"),
        "Hybrid Visualization": result_row.get("hybrid")
    }
    for label, img_path in images.items():
        if img_path and os.path.exists(img_path):
            try:
                c.drawImage(img_path, 50, y - img_size[1], width=img_size[0], height=img_size[1])
                c.drawString(260, y - 20, label)
                y -= (img_size[1] + 40)
            except Exception:
                pass

    # Explanation
    c.setFont("Helvetica-Bold", 14)
    c.drawString(50, y - 30, "AI Explainability Notes")
    c.line(50, y - 33, 240, y - 33)
    c.setFont("Helvetica", 11)
    explanation = (
        "• Grad-CAM highlights bone regions influencing the AI's decision.\n"
        "• Canny detects fracture edges and bone outlines.\n"
        "• Hybrid overlay combines heatmap and edges for precise localization.\n"
        "• This helps doctors verify the AI prediction and interpret the fracture region."
    )
    text_obj = c.beginText(50, y - 50)
    text_obj.setFont("Helvetica", 11)
    for line in explanation.split("\n"):
        text_obj.textLine(line)
    c.drawText(text_obj)

    c.showPage()
    c.save()
    return out_pdf_path

# ---- Routes (unchanged except uses validate_xray etc.) ----
@app.route('/')
@app.route('/home')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

# Signup/Login/Logout/OTP/Reset ... (unchanged - re-use earlier code)
# For brevity in this file, keep implementations the same as your provided code.
# I'll paste them back verbatim (with no changes) to ensure routes work exactly as before.

# Signup
@app.route('/signup', methods=['GET','POST'])
def signup():
    if request.method == 'POST':
        data = request.form
        first_name = data.get('first_name','').strip()
        middle_name = data.get('middle_name','').strip()
        last_name = data.get('last_name','').strip()
        email = data.get('email','').strip().lower()
        phone = data.get('phone','').strip()
        gender = data.get('gender','').strip()
        try:
            age = int(data.get('age',0))
        except:
            age = 0
        password = request.form.get('password','')
        if not (first_name and email and password):
            flash("Please provide required fields", "danger")
            return render_template('signup.html')
        if len(password) < 8 or password.lower()==password or password.upper()==password or not any(c.isdigit() for c in password):
            flash("Use stronger password (mix of upper/lower/digits, min 8 chars)", "danger")
            return render_template('signup.html')
        conn = get_db()
        cur = conn.cursor()
        try:
            cur.execute('INSERT INTO users (first_name, middle_name, last_name, email, phone, gender, age, password_hash, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)',
                        (first_name, middle_name, last_name, email, phone, gender, age, generate_password_hash(password), datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
            conn.commit()
            flash("Signup successful. Please login.", "success")
            return redirect(url_for('login'))
        except sqlite3.IntegrityError:
            flash("Email already exists. Please login or use another email.", "danger")
        finally:
            conn.close()
    return render_template('signup.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email','').strip().lower()
        password = request.form.get('password','')
        user = get_user_by_email(email)
        if user and check_password_hash(user['password_hash'], password):
            session['user_id'] = user['id']
            session['first_name'] = user['first_name']
            return redirect(url_for('dashboard'))
        else:
            flash("Invalid email or password", "danger")
            return render_template('login.html')
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.clear()
    flash("Logged out", "info")
    return redirect(url_for('home'))

@app.route('/forgot_password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        user = get_user_by_email(email)
        if not user:
            flash("Email not found. Please sign up first.", "danger")
            return render_template('forgot_password.html')
        otp_code = "%06d" % random.randint(0, 999999)
        expires_at = (datetime.now() + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db()
        cur = conn.cursor()
        cur.execute('REPLACE INTO otps (email, otp, expires_at) VALUES (?, ?, ?)', 
                    (email, otp_code, expires_at))
        conn.commit()
        conn.close()
        ok, msg = send_otp_email(email, otp_code)
        if ok:
            flash("✅ OTP sent successfully! Check your email (or spam).", "success")
            return redirect(url_for('otp_verify', email=email))
        else:
            flash(f"❌ Failed to send OTP: {msg}", "danger")
    return render_template('forgot_password.html')

@app.route('/otp_verify', methods=['GET', 'POST'])
def otp_verify():
    email = request.args.get('email') or request.form.get('email')
    if not email:
        flash("Missing email context. Please restart the process.", "danger")
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        otp = request.form.get('otp', '').strip()
        conn = get_db()
        cur = conn.cursor()
        cur.execute("SELECT otp, expires_at FROM otps WHERE email = ?", (email,))
        row = cur.fetchone()
        conn.close()
        if not row:
            flash("OTP not found. Request again.", "danger")
            return redirect(url_for('forgot_password'))
        if row['otp'] != otp:
            flash("Invalid OTP. Please try again.", "danger")
            return render_template('otp_verify.html', email=email)
        if datetime.now() > datetime.strptime(row['expires_at'], "%Y-%m-%d %H:%M:%S"):
            flash("OTP expired. Please request a new one.", "warning")
            return redirect(url_for('forgot_password'))
        session['reset_email'] = email
        flash("✅ OTP verified! Please set a new password.", "success")
        return redirect(url_for('reset_password'))
    return render_template('otp_verify.html', email=email)

@app.route('/resend_otp/<email>')
def resend_otp(email):
    try:
        otp_code = "%06d" % random.randint(0, 999999)
        expires_at = (datetime.now() + timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M:%S")
        conn = get_db()
        cur = conn.cursor()
        cur.execute('REPLACE INTO otps (email, otp, expires_at) VALUES (?, ?, ?)', (email, otp_code, expires_at))
        conn.commit()
        conn.close()
        ok, msg = send_otp_email(email, otp_code)
        if ok:
            flash("✅ OTP re-sent successfully! Check your inbox or spam.", "success")
        else:
            flash(f"❌ Failed to resend OTP: {msg}", "danger")
    except Exception as e:
        flash(f"Error resending OTP: {e}", "danger")
    return redirect(url_for('otp_verify', email=email))

@app.route('/reset_password', methods=['GET', 'POST'])
def reset_password():
    email = session.get('reset_email')
    if not email:
        flash("Session expired. Please restart the reset process.", "danger")
        return redirect(url_for('forgot_password'))
    if request.method == 'POST':
        new_pass = request.form.get('new_password', '').strip()
        confirm = request.form.get('confirm_password', '').strip()
        if new_pass != confirm:
            flash("Passwords do not match!", "danger")
            return render_template('reset_password.html')
        if len(new_pass) < 8:
            flash("Password too short. Minimum 8 characters.", "warning")
            return render_template('reset_password.html')
        conn = get_db()
        cur = conn.cursor()
        cur.execute("UPDATE users SET password_hash = ? WHERE email = ?", 
                    (generate_password_hash(new_pass), email))
        conn.commit()
        conn.close()
        session.pop('reset_email', None)
        flash("🎉 Password reset successful! Please log in.", "success")
        return redirect(url_for('login'))
    return render_template('reset_password.html')

@app.route('/testmail')
def testmail():
    from email.message import EmailMessage
    import smtplib, os
    load_dotenv()
    user = os.getenv("MAIL_USERNAME")
    pwd = os.getenv("MAIL_PASSWORD")
    msg = EmailMessage()
    msg["Subject"] = "✅ Flask Test Email"
    msg["From"] = user
    msg["To"] = user
    msg.set_content("This is a test email from Flask — success means SMTP works!")
    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as smtp:
            smtp.starttls()
            smtp.login(user, pwd)
            smtp.send_message(msg)
        return "✅ Test email sent successfully! Check your inbox (or spam)."
    except Exception as e:
        return f"❌ Failed to send email: {e}"

@app.route('/dashboard', methods=['GET'])
@login_required
def dashboard():
    image_name = request.args.get('image_name')
    return render_template('dashboard.html', user=session.get('first_name'), image_name=image_name)

# Serve uploaded files safely
@app.route('/uploads/<path:filename>')
@login_required
def serve_upload(filename):
    path = os.path.join(UPLOAD_FOLDER, os.path.basename(filename))
    if not os.path.exists(path):
        abort(404)
    return send_file(path)

# Serve results
@app.route('/serve_result/<path:filename>')
@login_required
def serve_result(filename):
    path = os.path.join(RESULT_FOLDER, os.path.basename(filename))
    if not os.path.exists(path):
        abort(404)
    return send_file(path, as_attachment=False)

# Generate PDF report
@app.route('/generate_report/<int:result_id>')
@login_required
def generate_report(result_id):
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT * FROM results WHERE id = ?', (result_id,))
    row = cur.fetchone()
    conn.close()
    if not row:
        flash("Result not found", "danger")
        return redirect(url_for('myresults'))
    result_row = dict(row)
    pdf_name = f"report_{result_id}.pdf"
    pdf_path = os.path.join(RESULT_FOLDER, pdf_name)
    try:
        generate_pdf_report(result_row, pdf_path)
    except Exception as e:
        safe_log(f"PDF generation error: {e}", 'ERROR')
        flash("Failed to generate report", "danger")
        return redirect(url_for('myresults'))
    conn = get_db()
    cur = conn.cursor()
    cur.execute('UPDATE results SET report_path = ? WHERE id = ?', (pdf_path, result_id))
    conn.commit()
    conn.close()
    return send_file(pdf_path, as_attachment=True)

@app.route('/myresults')
@login_required
def myresults():
    uid = session.get('user_id')
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT * FROM results WHERE user_id = ? ORDER BY id DESC', (uid,))
    rows = cur.fetchall()
    conn.close()
    return render_template('myresults.html', rows=rows)

# The main processing endpoint
@app.route('/xrayvalidation', methods=['POST'])
@login_required
def xrayvalidation():

    # ---------------- FILE HANDLING ----------------
    if 'file' not in request.files:
        file = request.files.get('xray_image') or request.files.get('xray')
    else:
        file = request.files['file']

    if not file or file.filename == '':
        return jsonify({'error': 'No file uploaded'}), 400

    if not allowed_file(file.filename):
        return jsonify({'error': 'File type not allowed'}), 400

    filename = secure_filename(
        f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{file.filename}"
    )
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)
    user_id = session.get('user_id')

    safe_log(f"🟦 Received image upload: {filename}")

    # =====================================================================
    # ---------------- STEP 1: X-RAY VALIDATION (USING PyTorch model) ------
    # =====================================================================
    try:
        validation_label, valid_flag, val_conf = validate_xray(filepath)
        validation = validation_label
        confidence = float(val_conf)
        safe_log(f"[Validation] label={validation}, valid_flag={valid_flag}, confidence={confidence}")
    except Exception as e:
        safe_log(f"Validation error: {e}", "ERROR")
        validation = "error"
        confidence = 0.0
        valid_flag = False

    # ---------------- RETURN EARLY IF INVALID / NON-XRAY ----------------
    if validation != "bone_xray":
        rid = save_result_record(
            user_id,
            filename,
            'xray_validation',
            validation,
            confidence,
            gradcam=None,
            canny=None,
            hybrid=None,
            report_path=None,
            fracture_type=None
        )
        return jsonify({
            "validation": validation,
            "binary": None,
            "binary_confidence": None,
            "multiclass": None,
            "multiclass_confidence": None,
            "image_name": filename,
            "gradcam": None,
            "canny": None,
            "hybrid": None,
            "result_id": rid
        })

    # =====================================================================
    # ---------------- STEP 2: BINARY CLASSIFICATION ----------------------
    # =====================================================================
    try:
        binary_label, binary_conf = predict_binary(filepath)
    except Exception as e:
        safe_log(f"Binary prediction error: {e}", "ERROR")
        binary_label, binary_conf = "Unknown", 0.0

    # =====================================================================
    # ---------------- STEP 3: MULTICLASS IF FRACTURED --------------------
    # =====================================================================
    multi_label = None
    multi_conf = None
    if binary_label == "Fractured":
        try:
            multi_label, multi_conf = predict_multi(filepath)
        except Exception as e:
            safe_log(f"Multiclass prediction error: {e}", "ERROR")
            multi_label, multi_conf = None, None

    # =====================================================================
    # ----------- STEP 4: VISUALIZATION (GRADCAM, CANNY, HYBRID) ----------
    # =====================================================================
    grad_name = f"gradcam_{filename}"
    canny_name = f"canny_{filename}"
    hybrid_name = f"hybrid_{filename}"

    grad_path = os.path.join(RESULT_FOLDER, grad_name)
    canny_path = os.path.join(RESULT_FOLDER, canny_name)
    hybrid_path = os.path.join(RESULT_FOLDER, hybrid_name)

    # GRADCAM (PyTorch implementation) or fallback
    try:
        if TORCH_AVAILABLE and xray_model:
            try:
                generate_gradcam_torch(filepath, grad_path, model=xray_model, input_size=(224,224))
            except Exception as e:
                safe_log(f"PyTorch Grad-CAM failed: {e}", "WARN")
                # fallback: write original
                img = cv2.imread(filepath)
                if img is not None:
                    cv2.imwrite(grad_path, img)
                else:
                    grad_path = None
        else:
            gray = cv2.imread(filepath, 0)
            if gray is None:
                orig = cv2.imread(filepath)
                if orig is not None:
                    cv2.imwrite(grad_path, orig)
                else:
                    grad_path = None
            else:
                heat = cv2.applyColorMap(gray, cv2.COLORMAP_JET)
                cv2.imwrite(grad_path, heat)
    except Exception as e:
        safe_log(f"Grad-CAM error: {e}", "ERROR")
        grad_path = None

    # CANNY + HYBRID
    try:
        if grad_path:
            generate_canny(filepath, canny_path)
            generate_hybrid(grad_path, canny_path, hybrid_path)
        else:
            generate_canny(filepath, canny_path)
            hybrid_path = None
    except Exception as e:
        safe_log(f"Visualization error: {e}", "ERROR")
        canny_path = None
        hybrid_path = None

    # =====================================================================
    # ---------------- STEP 5: SAVE FULL RESULT TO DATABASE ---------------
    # =====================================================================
    rid = save_result_record(
        user_id,
        filename,
        "completed",
        binary_label,
        float(binary_conf),
        gradcam=grad_path,
        canny=canny_path,
        hybrid=hybrid_path,
        report_path=None,
        fracture_type=(multi_label if multi_label else None)
    )

    # =====================================================================
    # ---------------- STEP 6: FINAL JSON RESPONSE ------------------------
    # =====================================================================
    response = {
        "validation": validation,
        "binary": binary_label,
        "binary_confidence": float(binary_conf),
        "multiclass": multi_label,
        "multiclass_confidence": float(multi_conf) if multi_conf else None,
        "image_name": filename,
        "gradcam": os.path.basename(grad_path) if grad_path else None,
        "canny": os.path.basename(canny_path) if canny_path else None,
        "hybrid": os.path.basename(hybrid_path) if hybrid_path else None,
        "result_id": rid
    }

    safe_log(f"✅ Sending AI JSON response (result_id={rid})")
    return jsonify(response)

# Explain route (keeps behavior; uses our PyTorch validate_xray/predict functions)
@app.route('/explain', methods=['GET','POST'])
@app.route('/explain/<image_name>', methods=['GET','POST'])
@login_required
def explain(image_name=None):
    if not image_name:
        image_name = request.args.get('image_name') or request.form.get('image_name')
    if request.method == 'POST' and 'file' in request.files:
        f = request.files['file']
        if f and allowed_file(f.filename):
            filename = secure_filename(f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{f.filename}")
            fp = os.path.join(UPLOAD_FOLDER, filename)
            f.save(fp)
            validation_label, valid_flag, conf = validate_xray(fp)
            if validation_label != "bone_xray":
                flash("Uploaded file is not recognized as valid bone X-ray", "danger")
                return render_template('explain.html', image_name=filename, predicted="Invalid X-Ray", confidence=0)
            binary_label, binary_conf = predict_binary(fp)
            multi_label, multi_conf = (None, None)
            if binary_label == 'Fractured':
                multi_label, multi_conf = predict_multi(fp)
            grad_path = os.path.join(RESULT_FOLDER, f"gradcam_{filename}")
            canny_path = os.path.join(RESULT_FOLDER, f"canny_{filename}")
            hybrid_path = os.path.join(RESULT_FOLDER, f"hybrid_{filename}")
            try:
                if TORCH_AVAILABLE and xray_model:
                    try:
                        generate_gradcam_torch(fp, grad_path, model=xray_model, input_size=(224,224))
                    except Exception:
                        img = cv2.imread(fp)
                        if img is not None:
                            cv2.imwrite(grad_path, img)
                else:
                    img = cv2.imread(fp)
                    if img is not None:
                        cv2.imwrite(grad_path, img)
                generate_canny(fp, canny_path)
                generate_hybrid(grad_path, canny_path, hybrid_path)
            except Exception as e:
                safe_log(f"viz err: {e}", 'ERROR')
            rid = save_result_record(session.get('user_id'), filename, 'explain', binary_label, float(binary_conf),
                                     gradcam=grad_path, canny=canny_path, hybrid=hybrid_path,
                                     report_path=None, fracture_type=(multi_label if multi_label else None))
            return render_template('explain.html', image_name=filename, gradcam_image=os.path.basename(grad_path) if grad_path else None,
                                   canny_image=os.path.basename(canny_path) if canny_path else None,
                                   hybrid_image=os.path.basename(hybrid_path) if hybrid_path else None,
                                   fracture_type=(multi_label or binary_label), confidence=round(binary_conf*100, 2),
                                   ai_summary=f"Binary: {binary_label}. Multi: {multi_label or 'N/A'}", model_name="AI")
        else:
            flash("No valid file in request", "danger")
            return redirect(url_for('dashboard'))
    if not image_name:
        flash("No image specified to explain", "danger")
        return redirect(url_for('dashboard'))
    conn = get_db()
    cur = conn.cursor()
    cur.execute('SELECT * FROM results WHERE filename = ? ORDER BY id DESC LIMIT 1', (image_name,))
    row = cur.fetchone()
    conn.close()
    grad = canny = hybrid = predicted = confidence = ai_summary = fracture = None
    if row:
        grad = os.path.basename(row['gradcam']) if row['gradcam'] else None
        canny = os.path.basename(row['canny']) if row['canny'] else None
        hybrid = os.path.basename(row['hybrid']) if row['hybrid'] else None
        predicted = row['predicted']
        confidence = row['confidence']
        ai_summary = f"Stage: {row['stage']}| Confidence: {row['confidence']:.3f}"
        fracture = row['fracture_type'] if 'fracture_type' in row.keys() else row['predicted']
    tmpl = 'explain.html' if os.path.exists(os.path.join(TEMPLATE_FOLDER, 'explain.html')) else 'dashbord.html'
    return render_template(tmpl,
                           image_name=image_name,
                           gradcam_image=grad,
                           canny_image=canny,
                           hybrid_image=hybrid,
                           fracture_type=fracture,
                           confidence=round(confidence * 100, 2) if confidence else None,
                           ai_summary=ai_summary,
                           model_name="AI")

@app.route('/health')
def health():
    return jsonify({'status': 'ok', 'user': session.get('first_name')})

# Route listing for debugging
@app.route('/_routes')
def list_routes():
    routes = []
    for rule in app.url_map.iter_rules():
        routes.append((rule.endpoint, str(rule)))
    return jsonify(routes)

# Debug helper to inspect xray model outputs quickly
@app.route('/inspect_xray', methods=['GET'])
def inspect_xray():
    """
    Query: /inspect_xray?img=uploads/filename.jpg
    Returns raw logits/probs from xray_model to help mapping.
    """
    img = request.args.get('img')
    if not img:
        return jsonify({'error': 'pass ?img=uploads/yourfile.jpg'}), 400
    if img.startswith('uploads/'):
        img_path = os.path.join(UPLOAD_FOLDER, os.path.basename(img.replace('uploads/','')))
    else:
        img_path = os.path.join(UPLOAD_FOLDER, os.path.basename(img))
    if not os.path.exists(img_path):
        return jsonify({'error': 'image not found', 'path': img_path}), 404
    if not TORCH_AVAILABLE or xray_model is None:
        return jsonify({'error': 'xray model not loaded'}), 500
    try:
        img_pil = Image.open(img_path).convert('RGB')
        t = torch_transform(img_pil).unsqueeze(0)
        with torch.no_grad():
            out = xray_model(t)
        if isinstance(out, (tuple, list)):
            out_t = out[-1]
        else:
            out_t = out
        out_np = out_t.cpu().numpy()
        if out_np.ndim == 2 and out_np.shape[0] == 1:
            out_np = out_np[0]
        # softmax for multi
        if out_np.size > 1:
            exps = np.exp(out_np - np.max(out_np))
            probs = (exps / (exps.sum()+1e-12)).tolist()
        else:
            probs = [1.0/(1.0+np.exp(-float(out_np[0])))]
        return jsonify({'raw': out_np.tolist(), 'probs': probs})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

def debug_one_image(path):
    print("---- DEBUG:", path)
    if TORCH_AVAILABLE and xray_model:
        img = Image.open(path).convert('RGB')
        t = torch_transform(img).unsqueeze(0)
        with torch.no_grad():
            p = xray_model(t)
        print("xray raw output:", p)
    if TORCH_AVAILABLE and binary_model and torch_transform:
        t = torch_transform(Image.open(path).convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            out = binary_model(t)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
            print("torch binary probs:", probs, "argmax:", np.argmax(probs))
    if TORCH_AVAILABLE and multi_model:
        t = torch_transform(Image.open(path).convert('RGB')).unsqueeze(0)
        with torch.no_grad():
            out = multi_model(t)
            probs = F.softmax(out, dim=1).cpu().numpy()[0]
            print("torch multi probs:", probs[:5], "...", "argmax:", np.argmax(probs))

# ---- startup ----
if __name__ == '__main__':
    try_load_models()
    safe_log("Starting Frac-AI server (PyTorch X-ray validation)")
    safe_log("Routes: " + ", ".join([r.rule for r in app.url_map.iter_rules()]))
    app.run(debug=True)
