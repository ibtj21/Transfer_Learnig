"""
PSA Translation Demo — English/Kiswahili -> Ekegusii
Streamlit app serving the fine-tuned mT5-small and NLLB-200-distilled-600M models.

Expects the two fine-tuned model folders (produced by mt5_training.ipynb and
nllb_training.ipynb) to be available at the paths below -- update MT5_MODEL_DIR /
NLLB_MODEL_DIR if you place them elsewhere, or push them to the Hugging Face Hub
and swap these for repo ids (e.g. "your-username/mt5-en-guz").

Run locally with:
    streamlit run app.py
"""

import streamlit as st
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

MT5_MODEL_DIR = "checkpoints/mt5_combined_guz/final"
NLLB_MODEL_DIR = "checkpoints/nllb_combined_guz/final"
NLLB_TGT_PLACEHOLDER = "swh_Latn"  # placeholder tag for Ekegusii (unsupported by NLLB-200)
MAX_LEN = 128

st.set_page_config(page_title="PSA Translator — Ekegusii", page_icon="🌍")


@st.cache_resource(show_spinner="Loading mT5 model...")
def load_mt5():
    tok = AutoTokenizer.from_pretrained(MT5_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(MT5_MODEL_DIR)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model


@st.cache_resource(show_spinner="Loading NLLB model...")
def load_nllb():
    tok = AutoTokenizer.from_pretrained(NLLB_MODEL_DIR)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL_DIR)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    return tok, model


def mt5_prefix(source_lang):
    return "translate English to Ekegusii: " if source_lang == "en" else "translate Kiswahili to Ekegusii: "


def nllb_src_code(source_lang):
    return "eng_Latn" if source_lang == "en" else "swh_Latn"


def translate_mt5(text, source_lang):
    tok, model = load_mt5()
    inputs = tok(mt5_prefix(source_lang) + text, return_tensors="pt",
                 truncation=True, max_length=MAX_LEN).to(model.device)
    with torch.no_grad():
        out = model.generate(**inputs, max_length=MAX_LEN)
    return tok.decode(out[0], skip_special_tokens=True)


def translate_nllb(text, source_lang):
    tok, model = load_nllb()
    tok.src_lang = nllb_src_code(source_lang)
    inputs = tok(text, return_tensors="pt", truncation=True, max_length=MAX_LEN).to(model.device)
    forced_bos = tok.convert_tokens_to_ids(NLLB_TGT_PLACEHOLDER)
    with torch.no_grad():
        out = model.generate(**inputs, forced_bos_token_id=forced_bos, max_length=MAX_LEN)
    return tok.decode(out[0], skip_special_tokens=True)


st.title("🌍 PSA Translator: English/Kiswahili → Ekegusii")
st.caption(
    "Fine-tuned mT5-small and NLLB-200-distilled-600M models trained on a curated "
    "Kenyan Public Service Announcement corpus, targeting Ekegusii — a Bantu language "
    "with no coverage in either model's original pretraining data."
)

col1, col2 = st.columns(2)
with col1:
    source_lang_label = st.selectbox("Source language", ["English", "Kiswahili"])
    source_lang = "en" if source_lang_label == "English" else "sw"
with col2:
    model_choice = st.selectbox("Model", ["mT5-small", "NLLB-200-distilled-600M", "Both (compare)"])

text = st.text_area(
    "Sentence to translate",
    placeholder="e.g. Farmers are urged to prioritize safe agrochemical usage this season.",
    height=100,
)

if st.button("Translate", type="primary", disabled=not text.strip()):
    with st.spinner("Translating..."):
        if model_choice in ("mT5-small", "Both (compare)"):
            st.subheader("mT5-small")
            st.success(translate_mt5(text, source_lang))
        if model_choice in ("NLLB-200-distilled-600M", "Both (compare)"):
            st.subheader("NLLB-200-distilled-600M")
            st.success(translate_nllb(text, source_lang))

st.divider()
st.caption(
    "Note: Ekegusii has no native language code in NLLB-200, so a placeholder target-language "
    "tag is used to steer generation. Both models were fine-tuned with layer freezing on a "
    "low-resource English/Kiswahili → Ekegusii dataset; see the accompanying training notebooks "
    "and results tables for hyperparameters and evaluation metrics."
)
