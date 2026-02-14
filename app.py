import streamlit as st
import spacy
import pandas as pd
import plotly.express as px
from pdfminer.high_level import extract_text
import re

# --- CONFIGURATION ---
st.set_page_config(page_title="Advanced AI Resume Analyzer", layout="wide")

# --- 1. LOAD THE ADVANCED MODEL ---
@st.cache_resource
def load_nlp_model():
    # Load the folder you unzipped
    try:
        nlp = spacy.load("./nlp_model")
        return nlp
    except OSError:
        st.error("Error: Model not found! Make sure 'nlp_model' folder is in the same directory.")
        return None

nlp = load_nlp_model()

# --- 2. HELPER FUNCTIONS ---
def extract_text_from_pdf(pdf_file):
    return extract_text(pdf_file)

def clean_text(text):
    # Remove extra spaces and newlines for cleaner analysis
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

# --- 3. UI LAYOUT ---
st.title("🚀 Advanced AI Resume Analyzer")
st.markdown("Use Deep Learning to extract entities, score resumes, and find skill gaps.")

# Sidebar for controls
with st.sidebar:
    st.header("Upload Section")
    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    
    st.divider()
    st.markdown("**Job Description (JD)**")
    jd_text = st.text_area("Paste the JD here to check for match:", height=150)
    
    analyze_button = st.button("Analyze Resume")

# --- 4. MAIN LOGIC ---
if uploaded_file and analyze_button:
    with st.spinner("Processing PDF and running Neural Network..."):
        
        # A. Extraction
        raw_text = extract_text_from_pdf(uploaded_file)
        cleaned_text = clean_text(raw_text)
        
        # B. AI Inference (The Magic)
        doc = nlp(cleaned_text)
        
        # C. Categorize Entities
        entities = {"SKILL": [], "ORG": [], "PERSON": [], "GPE": [], "EDU": []}
        
        # Map your custom labels to these categories
        # Note: Your trained labels might differ (e.g., COMPANY vs ORG)
        for ent in doc.ents:
            label = ent.label_
            if label in ["SKILL", "WORK_OF_ART", "PRODUCT"]: # Adjust based on your training
                entities["SKILL"].append(ent.text)
            elif label in ["ORG", "COMPANY"]:
                entities["ORG"].append(ent.text)
            elif label in ["PERSON"]:
                entities["PERSON"].append(ent.text)
            elif label in ["GPE", "LOC"]:
                entities["GPE"].append(ent.text)
        
        # Remove duplicates
        for key in entities:
            entities[key] = list(set(entities[key]))

    # --- 5. RESULTS DASHBOARD ---
    
    # Row 1: Profile Overview
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.success("Analysis Complete ✅")
        st.metric("Skills Detected", len(entities["SKILL"]))
        if entities["PERSON"]:
            st.write(f"**Candidate:** {entities['PERSON'][0]}")
    
    with col2:
        # Visualizing Skill Distribution
        if entities["SKILL"]:
            st.write("### 🧠 Detected Skills Cloud")
            st.write(", ".join([f"`{skill}`" for skill in entities["SKILL"]]))

    st.divider()

    # Row 2: Advanced Charts
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("🏢 Experience & Organizations")
        if entities["ORG"]:
            df_org = pd.DataFrame(entities["ORG"], columns=["Organization"])
            st.table(df_org)
        else:
            st.info("No specific organizations detected.")

    with col4:
        st.subheader("📍 Locations Found")
        if entities["GPE"]:
             st.write(entities["GPE"])
        else:
            st.info("No location data found.")

    # --- 6. JD MATCHING (BONUS ADVANCED FEATURE) ---
    if jd_text and entities["SKILL"]:
        st.divider()
        st.subheader("📊 Job Match Score")
        
        # Simple Set Intersection for now (Upgrade to SBERT later)
        jd_skills = set(jd_text.lower().split()) # Basic tokenization
        resume_skills_lower = set([s.lower() for s in entities["SKILL"]])
        
        # Find matches (Approximate)
        matched = []
        for r_skill in resume_skills_lower:
            if r_skill in jd_text.lower():
                matched.append(r_skill)
        
        match_score = len(matched) * 10 # Arbitrary scoring logic
        if match_score > 100: match_score = 100
        
        st.progress(match_score)
        st.write(f"**Match Score: {match_score}%** based on keyword overlap with AI extracted skills.")
        
        st.write("Matched Keywords:", matched)