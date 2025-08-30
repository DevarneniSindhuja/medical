# main_app.py

from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import pipeline
from typing import List, Dict, Optional
import uvicorn
import streamlit as st
import requests

# ------------------- BACKEND LOGIC -------------------

app = FastAPI()

# Dummy drug interaction DB
DRUG_DATABASE = {
    "ibuprofen": {"interacts_with": ["aspirin"], "min_age": 12, "dosage": "200mg"},
    "aspirin": {"interacts_with": ["ibuprofen"], "min_age": 16, "dosage": "100mg"},
    "paracetamol": {"interacts_with": [], "min_age": 1, "dosage": "500mg"},
}

# Alternative suggestions
ALTERNATIVES = {
    "ibuprofen": "paracetamol",
    "aspirin": "paracetamol"
}

# NLP Pipeline using HuggingFace model
nlp = pipeline("ner", model="dslim/bert-base-NER", grouped_entities=True)

class PrescriptionInput(BaseModel):
    text: str
    age: int

@app.post("/analyze/")
async def analyze_prescription(data: PrescriptionInput):
    extracted_drugs = extract_drugs(data.text)
    interactions = check_interactions(extracted_drugs)
    dosage_info = recommend_dosage(extracted_drugs, data.age)
    alternatives = suggest_alternatives(extracted_drugs)

    return {
        "extracted_drugs": extracted_drugs,
        "interactions": interactions,
        "dosage_info": dosage_info,
        "alternatives": alternatives
    }

def extract_drugs(text: str) -> List[str]:
    entities = nlp(text)
    drugs = [ent['word'].lower() for ent in entities if ent['entity_group'] == 'MISC']
    return list(set(drugs))

def check_interactions(drugs: List[str]) -> List[str]:
    interactions = []
    for i, drug in enumerate(drugs):
        for other in drugs[i+1:]:
            if other in DRUG_DATABASE.get(drug, {}).get("interacts_with", []):
                interactions.append(f"{drug} interacts with {other}")
    return interactions

def recommend_dosage(drugs: List[str], age: int) -> Dict[str, str]:
    dosage = {}
    for drug in drugs:
        info = DRUG_DATABASE.get(drug)
        if info:
            if age >= info["min_age"]:
                dosage[drug] = info["dosage"]
            else:
                dosage[drug] = f"Not recommended under age {info['min_age']}"
    return dosage

def suggest_alternatives(drugs: List[str]) -> Dict[str, str]:
    return {drug: ALTERNATIVES.get(drug, "No alternative found") for drug in drugs if drug in ALTERNATIVES}

# ------------------- STREAMLIT FRONTEND -------------------

def run_streamlit():
    st.title("💊 AI Prescription Verifier")

    st.markdown("*Enter medical prescription text:*")
    user_input = st.text_area("Prescription Text", height=150)

    age = st.slider("Select Patient Age", 0, 100, 25)

    if st.button("Analyze Prescription"):
        with st.spinner("Analyzing..."):
            response = requests.post("http://localhost:8000/analyze/", json={"text": user_input, "age": age})
            if response.status_code == 200:
                data = response.json()
                st.subheader("📄 Extracted Drug Names")
                st.write(data["extracted_drugs"])

                st.subheader("⚠ Drug Interactions")
                st.write(data["interactions"] or "No harmful interactions found.")

                st.subheader("📋 Dosage Recommendations")
                st.write(data["dosage_info"])

                st.subheader("✅ Alternative Suggestions")
                st.write(data["alternatives"])
            else:
                st.error("Backend Error")

# ------------------- ENTRY POINT -------------------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("medical:app", host="127.0.0.1", port=8000,reload=True)

    # Run FastAPI in a separate thread
    threading.Thread(target=run_api daemon=True).start()

    # Run Streamlit
    run_streamlit()