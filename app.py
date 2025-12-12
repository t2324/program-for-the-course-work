import io
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import cv2
import joblib
from skimage.feature import hog, graycomatrix, graycoprops
from skimage.util import img_as_ubyte
from tensorflow import keras
import torch
from torchvision import transforms
import torch.nn.functional as F
import timm

st.set_page_config(
    page_title="Brain MRI Classifier (multi-model)",
    page_icon="🧠",
    layout="centered"
)

st.title("MRI Classifier — ResNet50, ViT, KNN, RF, SVM")



CLASS_NAMES = ["glioma", "meningioma", "notumor", "pituitary"]

RESNET_PATH = "resnet50_best.keras"
VIT_PATH = "vit_brain_mri_best.pt"
CNN_PATH = "baseline_cnn_final.keras"

KNN_MODEL_PATH = "knn_model.pkl"
KNN_SCALER_PATH = "scaler.pkl"
KNN_CLASSES_PATH = "class_names.pkl"

SVM_MODEL_PATH = "best_svm_model.joblib"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def pil_from_upload(uploaded_file):
    image = Image.open(uploaded_file).convert("RGB")
    return image

def preprocess_tf_image(img_pil, target_size):
    img = img_pil.resize(target_size)
    arr = np.array(img).astype("float32")
    arr = np.expand_dims(arr, axis=0)
    return arr



IMG_SIZE_KNN = 64

def extract_hog_knn(img_pil):
    img = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_SIZE_KNN, IMG_SIZE_KNN))
    feat = hog(
        img,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys"
    )
    return feat.astype("float32")

@st.cache_resource
def load_knn_bundle():
    knn = joblib.load(KNN_MODEL_PATH)
    scaler = joblib.load(KNN_SCALER_PATH)
    class_names_knn = joblib.load(KNN_CLASSES_PATH)
    return knn, scaler, class_names_knn

def predict_knn(img_pil):
    knn, scaler, _ = load_knn_bundle()
    feat = extract_hog_knn(img_pil)
    X = scaler.transform([feat])
    probs = knn.predict_proba(X)[0]
    labels = knn.classes_
    return labels, probs



def extract_features_svm(img_pil):
    bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
    bgr = cv2.resize(bgr, (224, 224))
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0

    f_hog = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        transform_sqrt=True,
        block_norm="L2-Hys",
        feature_vector=True
    )

    g8 = img_as_ubyte(np.clip(gray, 0, 1))
    glcm = graycomatrix(
        g8,
        distances=(1, 2, 4),
        angles=(0, np.pi/4, np.pi/2, 3*np.pi/4),
        symmetric=True,
        normed=True
    )
    props = ["contrast", "energy", "homogeneity", "correlation"]
    vals = [graycoprops(glcm, p).ravel() for p in props]
    f_glcm = np.concatenate(vals).astype("float32")

    feat = np.concatenate([f_hog, f_glcm], axis=0)
    return feat

@st.cache_resource
def load_svm_model():
    model = joblib.load(SVM_MODEL_PATH)   # Pipeline(scaler, PCA, SVC)
    return model

def softmax(x):
    x = np.array(x, dtype=np.float64)
    if x.ndim == 1:
        x = x.reshape(1, -1)
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return (e / e.sum(axis=1, keepdims=True))[0]

def predict_svm(img_pil):
    model = load_svm_model()
    feat = extract_features_svm(img_pil)
    X = feat.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
    else:
        scores = model.decision_function(X)
        probs = softmax(scores)

    classes_model = getattr(model, "classes_", np.arange(len(probs)))

    labels = []
    probs_out = []
    for c, p in zip(classes_model, probs):
        if isinstance(c, str):
            name = c
        else:
            if 0 <= int(c) < len(CLASS_NAMES):
                name = CLASS_NAMES[int(c)]
            else:
                name = str(c)
        labels.append(name)
        probs_out.append(p)

    return np.array(labels), np.array(probs_out)


@st.cache_resource
def load_resnet_model():
    model = keras.models.load_model(RESNET_PATH)
    return model

def predict_resnet(img_pil):
    model = load_resnet_model()
    x = preprocess_tf_image(img_pil, (224, 224))
    probs = model.predict(x, verbose=0)[0]
    probs = probs / probs.sum()
    return np.array(CLASS_NAMES), probs


@st.cache_resource
def load_cnn_model():
    model = keras.models.load_model(CNN_PATH)
    return model

def predict_cnn(img_pil):
    model = load_cnn_model()
    # твій CNN тренувався на 192x192
    x = preprocess_tf_image(img_pil, (192, 192))
    probs = model.predict(x, verbose=0)[0]
    probs = probs / probs.sum()  # нормалізація, щоб на 100%
    return np.array(CLASS_NAMES), probs



IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)

@st.cache_resource
def load_vit_model_and_transform():
    model = timm.create_model(
        "vit_base_patch16_224",
        pretrained=False,
        num_classes=len(CLASS_NAMES)
    )
    ckpt = torch.load(VIT_PATH, map_location=DEVICE)
    state_dict = ckpt.get("model", ckpt)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

    return model, transform

def predict_vit(img_pil):
    model, transform = load_vit_model_and_transform()
    x = transform(img_pil).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    return np.array(CLASS_NAMES), probs



MODEL_FUNCS = {
    "ResNet-50 (Keras)": predict_resnet,
    "Baseline CNN (Keras)": predict_cnn,
    "Vision Transformer (ViT, PyTorch)": predict_vit,
    "K-NN (HOG)": predict_knn,
    "SVM (HOG + GLCM)": predict_svm,
}



uploaded_file = st.file_uploader(
    "📤 Завантажте зображення (jpg / png / bmp)",
    type=["jpg", "jpeg", "png", "bmp"]
)

if uploaded_file is not None:
    img_pil = pil_from_upload(uploaded_file)
    st.image(img_pil, caption="Вхідне зображення", use_container_width=True)

    st.markdown("### ⚙️ Режим передбачення")
    mode = st.radio(
        "Оберіть режим:",
        ["Одна модель", "Всі моделі одразу"]
    )

    model_names = list(MODEL_FUNCS.keys())
    selected_model_name = st.selectbox(
        "Оберіть модель для одиночного передбачення:",
        model_names
    )

    col1, col2 = st.columns(2)
    run_single = col1.button("🔮 Передбачити вибраною моделлю")
    run_all = col2.button("🚀 Передбачити всіма моделями")


    if run_single or (mode == "Одна модель" and run_all):
        st.markdown(f"## Результат — {selected_model_name}")
        try:
            labels, probs = MODEL_FUNCS[selected_model_name](img_pil)
            probs = np.array(probs)
            order = np.argsort(probs)[::-1]
            labels_sorted = labels[order]
            probs_sorted = probs[order]

            top_label = labels_sorted[0]
            top_prob = probs_sorted[0] * 100.0

            st.write(f"**Передбачений клас:** `{top_label}` з ймовірністю **{top_prob:.1f}%**")

            df = pd.DataFrame({
                "Клас": labels_sorted,
                "Ймовірність, %": (probs_sorted * 100.0).round(2)
            })
            st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"Помилка при інференсі: {e}")


    if run_all:
        st.markdown("## Порівняння всіх моделей")
        rows = []
        for name, func in MODEL_FUNCS.items():
            try:
                labels, probs = func(img_pil)
                probs = np.array(probs)
                order = np.argsort(probs)[::-1]
                top_label = labels[order][0]
                top_prob = probs[order][0] * 100.0
                rows.append({
                    "Модель": name,
                    "Топ-клас": top_label,
                    "Ймовірність, %": round(top_prob, 2)
                })
            except Exception as e:
                rows.append({
                    "Модель": name,
                    "Топ-клас": f"Помилка: {type(e).__name__}",
                    "Ймовірність, %": np.nan
                })

        df_all = pd.DataFrame(rows)
        st.dataframe(df_all, use_container_width=True)

        st.markdown("### Детальні ймовірності по моделях")
        for name, func in MODEL_FUNCS.items():
            st.markdown(f"#### {name}")
            try:
                labels, probs = func(img_pil)
                probs = np.array(probs)
                order = np.argsort(probs)[::-1]
                labels_sorted = labels[order]
                probs_sorted = probs[order]
                df = pd.DataFrame({
                    "Клас": labels_sorted,
                    "Ймовірність, %": (probs_sorted * 100.0).round(2)
                })
                st.dataframe(df, use_container_width=True)
            except Exception as e:
                st.error(f"Помилка при інференсі {name}: {e}")

else:
    st.info("Спочатку завантажте зображення.")
