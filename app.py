# --- Smart Ambulance Clinical Dashboard ---
#
# IMPORTANT: Before running, ensure all required packages are installed.
# Create a file named 'requirements.txt' with the content below and run 'pip install -r requirements.txt'
#
# requirements.txt:
# streamlit
# pandas
# numpy
# scikit-learn
# joblib
# firebase-admin
# Pyrebase4
# glob2
# cryptography
# toml # NEW: toml library for secrets parsing (if needed, though Streamlit handles it)
# -----------------------------------------------------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os
import glob
import random
import pyrebase
import firebase_admin
from firebase_admin import credentials
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier

from cryptography.fernet import Fernet
import base64


# --- Page Configuration ---
st.set_page_config(page_title="Smart Ambulance Clinical Dashboard", page_icon="🚑", layout="wide")

# --- [UI ENHANCEMENT] Custom CSS for a professional look ---
def load_css():
    """Injects custom CSS for styling the application."""
    css = """
    <style>
        /* --- General Theme --- */
        body {
            color: #FAFAFA;
        }
        .stApp {
            background-color: #0E1117;
        }
        h1, h2, h3, h4, h5, h6 {
            color: #FAFAFA;
        }
        .st-emotion-cache-18ni7ap {
            background-color: #161A25;
        }
        
        /* --- Login Page --- */
        .login-container {
            display: flex;
            justify-content: center;
            align-items: center;
            height: 80vh;
        }
        .login-form {
            background-color: #1E222B;
            padding: 40px;
            border-radius: 10px;
            border: 1px solid #303642;
            width: 400px;
        }

        /* --- Patient Cards --- */
        .patient-card {
            background-color: #1E222B;
            border-radius: 10px;
            padding: 20px;
            margin-bottom: 15px;
            border: 1px solid #303642;
            box-shadow: 0 4px 8px 0 rgba(0,0,0,0.2);
            transition: 0.3s;
        }
        .patient-card:hover {
            box-shadow: 0 8px 16px 0 rgba(0,0,0,0.3);
            border: 1px solid #4A5469;
        }
        .card-header {
            padding: 12px;
            border-radius: 8px 8px 0 0;
            margin: -20px -20px 15px -20px;
            color: white;
            font-weight: bold;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .card-header-critical { background-color: #D9534F; }
        .card-header-warning { background-color: #F0AD4E; }
        .card-header-normal { background-color: #5CB85C; }
        .card-header-error { background-color: #4A5469; }

        /* --- Metrics Display --- */
        .metric-container {
            text-align: center;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #A0A0A0;
        }
        .metric-value {
            font-size: 1.75rem;
            font-weight: bold;
            color: #FAFAFA;
        }
        .metric-delta {
            font-size: 0.9rem;
            font-weight: bold;
        }
        .delta-up { color: #5CB85C; }
        .delta-down { color: #D9534F; }

        /* --- Custom Dividers & Details --- */
        .custom-divider {
            margin-top: 15px;
            margin-bottom: 15px;
            border-top: 1px solid #303642;
        }
        .detail-label {
            font-weight: bold;
            color: #A0A0A0;
        }
        .detail-value-conscious { color: #5CB85C; font-weight: bold; }
        .detail-value-unconscious { color: #D9534F; font-weight: bold; }
        
        /* --- Alert List Styling --- */
        .alert-list ul {
            padding-left: 20px;
            margin: 0;
        }
        .alert-list li {
            margin-bottom: 5px;
        }

        /* --- Buttons --- */
        .stButton>button {
            border-radius: 8px;
            border: 1px solid #007BFF;
            background-color: transparent;
            color: #007BFF;
            width: 100%;
            transition: 0.3s;
        }
        .stButton>button:hover {
            background-color: #007BFF;
            color: white;
            border: 1px solid #007BFF;
        }
        .stButton>button:focus {
            box-shadow: none !important;
        }
    </style>
    """
    st.markdown(css, unsafe_allow_html=True)

# --- Firebase Authentication ---
@st.cache_resource
def initialize_firebase():
    """Initializes Firebase Admin SDK and Pyrebase."""
    try:
        if "firebase_credentials" not in st.secrets or "firebase_config" not in st.secrets:
             st.error("🚨 Firebase configuration is missing from Streamlit secrets.")
             st.warning("Please ensure you have a `.streamlit/secrets.toml` file with your credentials.")
             # --- DEBUG: Raw secrets output for parsing issue ---
             secrets_file_path = os.path.join(".streamlit", "secrets.toml")
             if os.path.exists(secrets_file_path):
                 st.error(f"DEBUG: Raw content of secrets.toml being read by Streamlit:")
                 with open(secrets_file_path, "r", encoding='utf-8') as f:
                     st.code(f.read(), language='toml')
             return None
        
        firebase_creds = dict(st.secrets["firebase_credentials"])
        firebase_config = dict(st.secrets["firebase_config"])

        if not firebase_admin._apps:
            if "type" not in firebase_creds or "private_key" not in firebase_creds:
                 st.error("Firebase credentials in secrets.toml are malformed. Check 'type' and 'private_key'.")
                 return None
            cred = credentials.Certificate(firebase_creds)
            firebase_admin.initialize_app(cred)
        
        firebase = pyrebase.initialize_app(firebase_config)
        return firebase.auth()
    except Exception as e:
        st.error(f"🚨 Firebase initialization failed: {e}")
        # --- DEBUG: Print loaded secrets if an error occurs here ---
        st.info(f"DEBUG (initialize_firebase): st.secrets content at failure: {st.secrets.to_dict()}")
        return None

def login_page():
    """Renders the improved login page and handles authentication."""
    st.title("🚑 Smart Ambulance Clinical Decision Support")
    
    _, col, _ = st.columns([1, 1.2, 1])
    
    with col:
        st.markdown('<div class="login-form">', unsafe_allow_html=True)
        with st.form("login_form"):
            st.markdown("<h3 style='text-align: center;'>Login to Dashboard</h3>", unsafe_allow_html=True)
            email = st.text_input("Email", placeholder="user@ambulance.in")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            submit_button = st.form_submit_button("Login", use_container_width=True)

            if submit_button:
                if not email or not password:
                    st.error("Email and Password are required.")
                else:
                    auth = initialize_firebase()
                    if auth:
                        try:
                            with st.spinner("Authenticating..."):
                                user = auth.sign_in_with_email_and_password(email, password)
                                st.session_state.logged_in = True
                                st.session_state.user_email = user['email']
                                st.rerun()
                        except Exception:
                            st.error("Authentication failed. Please check your credentials.")
        st.markdown('</div>', unsafe_allow_html=True)

# --- [UI DESIGN] Protocol Library (COMPLETE) ---
PROTOCOLS = {
"Primary Survey": """
🟩 **Initial Assessment (Primary Survey – ABCDE)**
- **A (Airway):** Check if blocked. If so, clear manually, suction, or insert an airway adjunct.
- **B (Breathing):** Check if breathing. If not, give 2 rescue breaths, provide O₂, or use a bag-valve-mask.
- **C (Circulation):** Check for a pulse. If absent, start CPR and attach an AED. If major bleeding, apply direct pressure.
- **D (Disability):** Check consciousness using the AVPU scale. If unresponsive, protect the airway.
- **E (Exposure):** Fully expose the patient to check for injuries, burns, bleeding, or fractures.
""",
"Direct Pressure": """
🟥 **For Major Bleeding**
- **Apply Direct Pressure:** Use sterile gauze to apply firm, direct pressure to the wound.
- **If Bleeding Continues:** For limb injuries, apply a tourniquet proximal to the wound.
- **Internal Bleeding Signs:** If suspected, keep the patient flat and slightly elevate their legs.
- **Monitor Vitals:** Re-check vital signs every 2-3 minutes.

*💡 Note: Administer oxygen and ensure rapid transport to a trauma center.*
""",
"Start CPR": """
🟦 **For Cardiac Arrest (No Breathing / No Pulse)**
- **Start CPR:** Begin high-quality chest compressions immediately.
- **Attach Defibrillator:** As soon as available, attach an AED or manual defibrillator.
- **Follow Prompts:** Follow the voice commands from the AED. Minimize interruptions.
- **Continue Until ROSC:** Continue CPR until there is a return of spontaneous circulation (ROSC) or care is transferred.
""",
"Administer Oxygen": """
🟨 **For Respiratory Distress (SpO₂ < 94%)**
- **Give High-Flow Oxygen:** Administer 10–15 L/min via a non-rebreather mask.
- **Check for Wheezing:** If present, administer a bronchodilator (e.g., salbutamol).
- **Suspect Cardiac Cause:** If cyanosis or chest pain is present, move to the cardiac protocol.
""",
"Administer Medication": """
🟧 **For Suspected Heart Attack / Chest Pain**
- **Administer Oxygen:** Ensure the patient is receiving oxygen.
- **Administer Aspirin:** Give chewable aspirin if the patient is conscious and not allergic.
- **Monitor ECG:** If available, establish ECG monitoring to check for changes.
- **Prepare for Defibrillation:** Have the defibrillator ready and attached.

*💡 Note: Ensure rapid transport to a cardiac center.*
""",
"Recovery Position": """
🟪 **For Unconscious but Breathing Patient**
- **Place in Recovery Position:** Carefully roll the patient onto their left side.
- **Maintain Airway:** Ensure the head is tilted back to keep the airway open.
- **Monitor Continuously:** Monitor breathing and pulse every minute.
""",
"Immobilize Limb": """
🟫 **For Suspected Fracture / Major Trauma**
- **Immobilize Limb:** Secure the injured limb using a splint to prevent further movement.
- **Control Bleeding:** Address any associated bleeding with direct pressure.
- **Avoid Movement:** Do not attempt to straighten a deformed limb. Splint it as it lies.
""",
"Cool Burns": """
⚫ **For Burns**
- **Cool the Burn:** Cool the area with clean water for 10-20 minutes. Do not use ice.
- **Cover the Burn:** Apply a sterile, non-stick dressing. Do not use any ointments.
- **Monitor for Shock:** Be vigilant for signs of shock (low BP, rapid pulse).
""",
"Manage Shock": """
⚪ **For Shock (Low BP, Rapid Pulse, Pale, Sweaty)**
- **Position the Patient:** Lay the patient flat on their back.
- **Elevate Legs:** Raise the patient's legs (unless head/spine trauma is suspected).
- **Administer Oxygen:** Provide high-flow oxygen.
- **Control Bleeding:** Ensure any external bleeding is controlled.
- **Keep Warm:** Use a blanket to keep the patient warm.
""",
"Seizure Care": """
🔵 **For an Active Seizure**
- **Ensure Safety:** Protect the patient from injury. Do not restrain them.
- **Post-Seizure Care:** Once the seizure stops, place the patient in the recovery position.
- **Check ABCs:** Check airway and breathing.
- **Prolonged Seizure:** If seizure > 5 mins, administer emergency medication per protocol.
""",
"Default": """
🔶 **Ongoing Care & Transport**
- Continue any required interventions (oxygen, IV fluids).
"""
}

# --- MASSIVELY EXPANDED CLINICAL KNOWLEDGE BASE (COMPLETE) ---
knowledge_base_source = [
    (["unconscious", "unknown cause"], ["no response", "limp"], "Unconscious - unknown cause", "Check airway, breathing, circulation (ABC), oxygen, rapid transport", "Hypoxia, brain damage", "Unconscious - Unknown Cause"),
    (["unconscious", "head injury"], ["vomiting", "unequal pupils"], "Unconscious - head injury", "Airway, cervical spine immobilization, oxygen, transport", "Brain hemorrhage", "Head Injury / Brain Trauma"),
    (["unconscious", "hypoglycemia"], ["sweating", "confusion"], "Unconscious - hypoglycemia", "IV glucose if available, monitor airway, transport", "Seizure, brain injury", "Hypoglycemia"),
    (["unconscious", "cardiac arrest"], ["no pulse", "no breathing"], "Unconscious - cardiac arrest", "CPR, defibrillation if available, rapid transport", "Death", "Cardiac Arrest"),
    (["chest pain", "mi"], ["crushing chest pain", "sweating"], "Chest Pain - MI", "Oxygen, ECG, aspirin, rapid transport", "Cardiac arrest", "Myocardial Infarction (MI)"),
    (["chest pain", "angina"], ["pressure", "mild sob"], "Chest Pain - Angina", "Oxygen, monitor vitals, transport", "Risk of MI", "Angina"),
    (["chest pain", "pulmonary embolism"], ["sudden chest pain", "dyspnea"], "Chest Pain - Pulmonary embolism", "Oxygen, monitor vitals, transport", "Cardiac arrest", "Pulmonary Embolism"),
    (["chest pain", "pneumothorax"], ["sharp pain", "dyspnea"], "Chest Pain - Pneumothorax", "Oxygen, monitor, rapid transport", "Respiratory failure", "Pneumothorax"),
    (["accident", "trauma", "fracture"], ["pain", "swelling", "deformity", "bleeding"], "Accident/Trauma - fracture", "Immobilize, control bleeding, monitor vitals, transport", "Blood loss, infection, shock", "Fracture"),
    (["bleeding", "hemorrhage"], ["major bleeding", "trauma", "laceration"], "Major Bleeding / Hemorrhage", "Apply Direct Pressure, elevate limb, consider tourniquet for uncontrollable limb bleeding", "Hypovolemic Shock", "Major Bleeding"),
    (["accident", "trauma", "head injury"], ["confusion", "vomiting"], "Accident/Trauma - head injury", "Airway, cervical immobilization, oxygen, transport", "Brain bleed, coma", "Head Injury / Brain Trauma"),
    (["accident", "trauma", "spinal injury"], ["paralysis", "numbness"], "Accident/Trauma - spinal injury", "Spinal immobilization, airway, transport", "Permanent Disability", "Spinal Injury"),
    (["fever", "sepsis"], ["low bp", "confusion", "high temp"], "Fever - sepsis", "Oxygen, IV fluids, rapid transport", "Organ failure", "Sepsis"),
    (["seizure", "tonic-clonic"], ["convulsions", "loss of consciousness"], "Seizure - tonic-clonic", "Protect patient, monitor airway, oxygen, transport", "Hypoxia, injury", "Tonic-Clonic Seizure"),
    (["poisoning", "ingestion"], ["vomiting", "abdominal pain"], "Poisoning - Ingestion", "Identify poison, monitor airway, transport", "Organ failure, shock", "Poisoning - ingestion"),
    (["poisoning", "overdose"], ["confusion", "slow breathing"], "Poisoning - Drug Overdose", "Airway, oxygen, antidote (if available), transport", "Coma, respiratory arrest", "Poisoning - opioid overdose"),
    (["poisoning", "inhalation"], ["dizziness", "breathlessness"], "Poisoning - Inhalation (Toxic Gas)", "Remove exposure, give oxygen, transport", "Respiratory failure", "Poisoning - toxic inhalation"),
    (["stroke", "ischemic"], ["facial droop", "arm weakness", "slurred speech"], "Stroke - Ischemic", "Oxygen, monitor vitals, rapid transport to stroke center", "Paralysis, brain damage", "Stroke - ischemic"),
    (["stroke", "hemorrhagic"], ["sudden severe headache", "vomiting"], "Stroke - Hemorrhagic", "Airway, oxygen, elevate head, rapid transport", "Brain bleed, death", "Stroke - hemorrhagic"),
    (["stroke", "tia"], ["temporary weakness", "slurred speech"], "Stroke - TIA (Mini Stroke)", "Monitor, maintain airway, urgent hospital evaluation", "Recurrent stroke risk", "Stroke - transient ischemic attack"),
    (["chest pain", "severe"], ["pain radiating to arm/jaw", "sweating"], "Acute Coronary Syndrome", "O₂, ECG monitoring, aspirin, cardiac support", "Myocardial infarction", "Acute Coronary Syndrome"),
    (["unconsciousness"], ["no response", "low pulse"], "Coma / Cardiac Arrest", "ABC check, O₂, recovery position", "Brain injury, cardiac arrest", "Coma / Cardiac Arrest"),
    (["bleeding", "severe"], ["continuous bleeding", "pallor"], "Hemorrhagic Shock", "Apply pressure, IV fluids, urgent transfer", "Shock, organ failure", "Hemorrhagic Shock"),
    (["stroke"], ["slurred speech", "facial droop"], "Acute Stroke", "FAST test, O₂, rapid transport", "Brain ischemia, paralysis", "Acute Stroke"),
    (["breathing difficulty", "severe"], ["cyanosis", "low spo2"], "Respiratory Distress", "O₂, airway management, monitor SpO₂", "Respiratory failure", "Respiratory Distress"),
    (["accident trauma", "major"], ["fractures", "bleeding"], "Multi-System Trauma", "Immobilize spine, airway, IV fluids", "Internal bleeding", "Multi-System Trauma"),
    (["seizure", "ongoing"], ["convulsions >5 min"], "Neurological Emergency", "Protect from injury, O₂, IV access", "Status epilepticus", "Neurological Emergency"),
    (["poisoning", "severe"], ["vomiting", "confusion", "seizure"], "Toxicological Crisis", "Identify poison, airway, O₂, rapid transport", "Respiratory arrest, coma", "Toxicological Crisis"),
    (["pregnancy", "bleeding"], ["vaginal bleeding", "dizziness"], "Obstetric Hemorrhage", "Left lateral position, O₂, transport fast", "Miscarriage, hemorrhage", "Obstetric Hemorrhage"),
    (["abdominal pain", "severe"], ["guarding", "rigidity"], "GI / Internal Injury", "IV fluids, NPO, hospital transfer", "Peritonitis, internal bleeding", "GI / Internal Injury"),
    (["chest pain", "mild"], ["localized pain", "tender ribs"], "Musculoskeletal Pain", "Reassure, rest", "Muscular pain", "Musculoskeletal Pain"),
    (["fever", "high grade"], [">102°f", "chills"], "Infectious Fever", "Tepid sponge, fluids, antibiotics", "Sepsis, meningitis", "Infectious Fever"),
    (["fever", "low grade"], ["99–101°f", "mild fatigue"], "Viral Infection", "Rest, fluids, paracetamol", "Viral infection", "Viral Infection"),
    (["bleeding", "minor"], ["small wound"], "Minor Injury", "Clean wound, antiseptic dressing", "Local infection", "Minor Injury"),
    (["seizure", "single"], ["short convulsion", "recovery"], "Isolated Seizure", "Protect from injury, monitor", "Epileptic event", "Isolated Seizure"),
    (["pregnancy", "labor"], ["regular contractions", "back pain"], "Active Labor", "Prepare delivery kit, monitor fetal movement", "Normal / preterm labor", "Active Labor"),
    (["accident", "minor"], ["bruises", "abrasions"], "Minor Trauma", "Clean wound, apply dressing", "Local infection", "Minor Trauma"),
    (["stroke", "tia"], ["temporary weakness"], "Mini Stroke", "FAST test, O₂, referral", "Major stroke risk", "Mini Stroke"),
    (["unconsciousness", "fainting"], ["short loss of consciousness"], "Syncope", "Lay flat, check glucose", "Hypoglycemia, dehydration", "Syncope"),
    (["breathing difficulty", "asthma"], ["wheezing", "tight chest"], "Asthma Attack", "Sit upright, O₂, inhaler", "Asthma exacerbation", "Asthma Attack"),
    (["poisoning", "mild"], ["nausea", "vomiting"], "Mild Toxicity", "Oral fluids, monitor", "GI irritation", "Mild Toxicity"),
    (["abdominal pain", "mild"], ["cramps", "bloating"], "GI Discomfort", "Fluids, rest", "Gastritis", "GI Discomfort"),
    (["fever", "dengue suspect"], ["high fever", "rash", "joint pain"], "Dengue / Viral Hemorrhagic Fever", "IV fluids, O₂ if low BP", "Shock, dehydration", "Dengue / Viral Hemorrhagic Fever"),
    (["accident trauma", "head injury"], ["bleeding from scalp", "vomiting"], "Head Trauma", "Immobilize head, O₂, rapid transfer", "Brain injury, internal bleed", "Head Trauma"),
    (["seizure", "postpartum"], ["convulsions after delivery"], "Postpartum Eclampsia", "Protect airway, MgSO₄, urgent transfer", "Eclampsia", "Postpartum Eclampsia"),
    (["pregnancy", "high bp"], ["swelling", "headache", "blurred vision"], "Pregnancy Hypertension", "Monitor BP, left lateral, hospital transfer", "Pre-eclampsia", "Pregnancy Hypertension"),
    (["chest pain", "anxiety"], ["fast breathing", "panic"], "Anxiety Episode", "Reassure, deep breathing", "Hyperventilation", "Anxiety Episode"),
    (["breathing difficulty", "copd"], ["chronic cough", "fatigue"], "COPD Exacerbation", "O₂ support, nebulization", "Hypoxia", "COPD Exacerbation"),
    (["abdominal pain", "pregnancy"], ["cramping", "back pain"], "Labor Onset", "Monitor contractions, prepare for delivery", "Preterm labor", "Labor Onset"),
    (["bleeding", "nosebleed"], ["nasal bleeding"], "Epistaxis", "Pinch nose, tilt forward, ice", "Hypertension, local injury", "Epistaxis"),
    (["stroke", "severe"], ["unconscious", "unequal pupils"], "Hemorrhagic Stroke", "O₂, rapid neuro referral", "Cerebral hemorrhage", "Hemorrhagic Stroke"),
    (["fever", "child"], ["crying", "hot skin"], "Pediatric Fever", "Tepid sponge, paracetamol", "Febrile seizure", "Pediatric Fever"),
    (["unconsciousness", "after seizure"], ["postictal confusion"], "Post-Seizure State", "O₂, airway check", "Brain hypoxia", "Post-Seizure State"),
    (["pregnancy", "normal"], ["mild back pain", "nausea"], "Normal Pregnancy", "Hydration, observation", "Stable", "Normal Pregnancy"),
    (["accident trauma", "chest"], ["pain", "difficulty breathing"], "Chest Trauma", "Immobilize, O₂, urgent care", "Rib fracture, pneumothorax", "Chest Trauma"),
    (["poisoning", "inhalation"], ["cough", "breathlessness"], "Inhalation Poisoning", "Remove from area, O₂", "Chemical pneumonitis", "Inhalation Poisoning"),
    (["abdominal pain", "child"], ["crying", "vomiting"], "Pediatric GI Emergency", "NPO, transport fast", "Appendicitis", "Pediatric GI Emergency"),
    (["breathing difficulty", "allergy"], ["swelling", "rash", "low bp"], "Anaphylactic Shock", "Adrenaline, O₂, IV fluids", "Anaphylaxis", "Anaphylactic Shock"),
    (["seizure", "fever induced"], ["high fever", "jerking"], "Febrile Convulsion", "Tepid sponge, monitor", "Febrile seizure", "Febrile Convulsion"),
    (["pregnancy", "postpartum bleeding"], ["heavy bleeding after delivery"], "Postpartum Hemorrhage", "Fundal massage, IV fluids, urgent transfer", "Shock, death", "Postpartum Hemorrhage"),
]

clinical_knowledge_base = [{"primary_complaints": item[0], "secondary_signs": item[1], "Possible Medical Cause": item[2], "Immediate Action / Ambulance / Doctor Steps": item[3], "Possible Complications / What Might Happen": item[4]} for item in knowledge_base_source]

# --- COMPLAINT_TO_PROTOCOL dictionary ---
COMPLAINT_TO_PROTOCOL = {
"bleeding": "Direct Pressure", "breathing": "Administer Oxygen", "unconscious": "Recovery Position",
"cardiac": "Start CPR", "arrest": "Start CPR", "chest pain": "Administer Medication",
"heart attack": "Administer Medication", "fracture": "Immobilize Limb", "trauma": "Immobilize Limb",
"burn": "Cool Burns", "shock": "Manage Shock", "seizure": "Seizure Care",
"stroke": "Recovery Position", "poison": "Default", "accident": "Immobilize Limb", "fever": "Default",
"abdominal": "Default", "pregnancy": "Default", "allergy": "Administer Oxygen"
}

# --- Encryption/Decryption Helpers ---
@st.cache_resource
def get_fernet_cipher():
    """Initializes and caches the Fernet cipher for AES encryption."""
    try:
        # --- DEBUG: Print loaded secrets before access ---
        # st.info(f"DEBUG (get_fernet_cipher): st.secrets content at entry: {st.secrets.to_dict()}")
        aes_key_str = st.secrets["encryption"]["aes_encryption_key"]
        return Fernet(aes_key_str.encode())
    except KeyError:
        st.error("🚨 AES encryption key is missing from Streamlit secrets.toml under [encryption].aes_encryption_key")
        st.stop()
    except Exception as e:
        st.error(f"🚨 Failed to initialize AES cipher: {e}. Check your AES key format.")
        # --- DEBUG: Print loaded secrets if an error occurs here ---
        st.info(f"DEBUG (get_fernet_cipher): st.secrets content at failure: {st.secrets.to_dict()}")
        st.stop()

def encrypt_data(text, cipher):
    """Encrypts a string using Fernet cipher."""
    if pd.isna(text) or text is None:
        return None
    try:
        return cipher.encrypt(str(text).encode()).decode()
    except Exception as e:
        st.warning(f"Encryption failed for data: '{text}'. Error: {e}")
        return None 

def decrypt_data(encrypted_text, cipher):
    """Decrypts a string using Fernet cipher."""
    if pd.isna(encrypted_text) or encrypted_text is None:
        return "N/A (Encrypted data missing)"
    try:
        return cipher.decrypt(encrypted_text.encode()).decode()
    except Exception as e:
        st.warning(f"Decryption failed for data: '{encrypted_text}'. Error: {e}")
        return "Decryption Error"

# --- Part 1: DATA GENERATION & MODEL TRAINING ---
def generate_synthetic_dataset_if_needed(target_csv_path):
    """Generates a synthetic dataset for model training if no other CSV is found."""
    st.warning(f"No existing `ambulance_dataset_300.csv` found. Generating synthetic data to `{target_csv_path}` for model training and app data.")
    records = []
    p_id_counter = 1000
    for item in knowledge_base_source:
        primary_complaints, secondary_signs, _, _, _, ml_prediction = item
        for i in range(random.randint(15, 25)): # Generate a variable number of records per condition for more realism
            p_id_counter += 1
            base_record = {
                'age': random.randint(20, 80), 'heart_rate_bpm': random.randint(70, 95),
                'systolic_bp_mmHg': random.randint(110, 130), 'diastolic_bp_mmHg': random.randint(70, 85), # Ensure _mmHg here
                'respiratory_rate_bpm': random.randint(14, 20), 'spo2_percent': random.randint(96, 99),
                'temperature_c': round(random.uniform(36.6, 37.4), 1), 'consciousness': 'Alert', 'gcs': 15,
            }
            complaint_parts = [random.choice(primary_complaints)]
            num_secondary = random.randint(0, len(secondary_signs))
            if num_secondary > 0:
                complaint_parts.extend(random.sample(secondary_signs, num_secondary))
            random.shuffle(complaint_parts)
            base_record['chief_complaint'] = ", ".join(complaint_parts)
            base_record['treatment_given'] = ml_prediction
            if "cardiac arrest" in primary_complaints:
                base_record.update({'heart_rate_bpm': 0, 'respiratory_rate_bpm': 0, 'consciousness': 'Unresponsive', 'gcs': 3, 'spo2_percent': random.randint(0, 40)})
            elif "sepsis" in primary_complaints:
                base_record.update({'systolic_bp_mmHg': random.randint(70, 95), 'heart_rate_bpm': random.randint(110, 150), 'temperature_c': round(random.uniform(38.5, 40.5), 1), 'consciousness': 'Confused', 'gcs': random.randint(10, 13)})
            elif any(p in primary_complaints for p in ["shock", "severe bleeding", "anaphylaxis"]):
                 base_record.update({'systolic_bp_mmHg': random.randint(60, 90), 'heart_rate_bpm': random.randint(120, 160)})
            elif "hypoglycemia" in primary_complaints:
                base_record.update({'consciousness': 'Confused', 'gcs': random.randint(11, 14), 'heart_rate_bpm': random.randint(90, 115)})
            elif "mi" in primary_complaints or "severe chest pain" in primary_complaints:
                base_record.update({'systolic_bp_mmHg': random.randint(85, 110), 'heart_rate_bpm': random.randint(95, 120)})
            elif "fracture" in primary_complaints:
                base_record.update({'heart_rate_bpm': random.randint(90, 125)})
            records.append({**{'p_id': p_id_counter}, **base_record})
    df = pd.DataFrame(records)
    df.to_csv(target_csv_path, index=False)
    st.success(f"Generated and saved synthetic dataset to `{target_csv_path}`.")


def train_models_if_needed():
    """Trains and saves ML models if they don't exist, using provided data or generating synthetic."""
    models_dir = './ml_models'
    model_file_exists = os.path.exists(os.path.join(models_dir, 'best_model.pkl'))
    clean_data_file_path = "clean_ambulance_dataset.csv"
    user_data_source_path = "ambulance_dataset_300.csv" # Explicitly refer to the user's file

    # --- Part 1: Ensure clean_ambulance_dataset.csv exists and is up-to-date ---
    
    # Always attempt to process the user's file if it exists, overwriting clean_ambulance_dataset.csv
    if os.path.exists(user_data_source_path):
        st.info(f"Processing provided dataset `{os.path.basename(user_data_source_path)}` for app data.")
        try:
            df_raw = pd.read_csv(user_data_source_path)
            # Make columns lowercase AND strip whitespace
            df_raw.columns = df_raw.columns.str.strip().str.lower() 
            
            # --- SUPER ROBUST RENAMING (Revised) ---
            target_cols_standard_casing = {
                'heart_rate_bpm', 'spo2_percent', 'systolic_bp_mmHg', 'diastolic_bp_mmHg',
                'respiratory_rate_bpm', 'temperature_c', 'gcs' # Include GCS here as it's often numeric
            }
            variations_map = {
                'heart_rate': 'heart_rate_bpm',
                'spo2': 'spo2_percent',
                'systolic_bp_reading': 'systolic_bp_mmHg',
                'diastolic_bp_reading': 'diastolic_bp_mmHg',
                'respiratory_rate': 'respiratory_rate_bpm',
                'temp': 'temperature_c',
                'systolic_bp_mmhg': 'systolic_bp_mmHg', # Correct lowercase mmhg
                'diastolic_bp_mmhg': 'diastolic_bp_mmHg' # Correct lowercase mmhg
            }
            
            new_columns_mapping = {}
            for col in df_raw.columns:
                lower_stripped_col = col.strip().lower()
                if lower_stripped_col in variations_map:
                    new_columns_mapping[col] = variations_map[lower_stripped_col]
                elif lower_stripped_col in target_cols_standard_casing:
                    new_columns_mapping[col] = lower_stripped_col # Keep if already in standard casing (e.g., 'age', 'gcs')
                else:
                    new_columns_mapping[col] = col # Keep other columns as is (e.g., p_id, city, chief_complaint, etc.)
            
            df_raw.rename(columns=new_columns_mapping, inplace=True)

            # --- CRITICAL VALIDATION AFTER RENAMING ---
            expected_numeric_cols = ['age', 'heart_rate_bpm', 'systolic_bp_mmHg', 'diastolic_bp_mmHg', 'respiratory_rate_bpm', 'spo2_percent', 'temperature_c']
            missing_after_rename = [col for col in expected_numeric_cols if col not in df_raw.columns]
            if missing_after_rename:
                st.error(f"🚨 After processing `{user_data_source_path}`, essential numeric columns are still missing: {missing_after_rename}. Please check your original CSV headers against expected formats.")
                raise ValueError("Missing critical columns after data processing.")


            df_raw.to_csv(clean_data_file_path, index=False)
            st.success(f"Processed user data saved to `{clean_data_file_path}`.")
        except Exception as e:
            st.error(f"🚨 Error processing `{user_data_source_path}`: {e}")
            st.error("Attempting to generate synthetic data as a fallback. Please check console for full error.")
            if os.path.exists(clean_data_file_path):
                os.remove(clean_data_file_path)
            generate_synthetic_dataset_if_needed(clean_data_file_path)
    elif not os.path.exists(clean_data_file_path): # Only generate synthetic if no user file AND no clean file
        generate_synthetic_dataset_if_needed(clean_data_file_path)
    
    if not os.path.exists(clean_data_file_path):
        st.error("🚨 CRITICAL: No data source could be found or generated for the application."); return False

    if model_file_exists: 
        st.info("✅ ML models already trained. Loading existing models.")
        return True

    with st.spinner(f"First time setup: Training AI models on `{os.path.basename(clean_data_file_path)}`... This may take a moment."):
        try:
            df = pd.read_csv(clean_data_file_path)
            
            numeric_features = ['age', 'heart_rate_bpm', 'systolic_bp_mmHg', 'diastolic_bp_mmHg', 'respiratory_rate_bpm', 'spo2_percent', 'temperature_c']
            
            missing_for_training = [col for col in numeric_features if col not in df.columns]
            if missing_for_training:
                st.error(f"🚨 Missing critical numeric features for training from `{clean_data_file_path}`: {missing_for_training}. This indicates a data preparation issue.")
                raise ValueError("Missing critical columns for model training.")

            if 'critical_status' not in df.columns:
                df['critical_status'] = np.select(
                    [(df['spo2_percent'] < 90) | (df['heart_rate_bpm'] > 130) | (df['systolic_bp_mmHg'] < 90) | (df['gcs'] <= 8),
                     (df['spo2_percent'] < 94) | (df['heart_rate_bpm'] > 110)],
                    [2, 1], default=0)

            df.dropna(subset=numeric_features + ['chief_complaint', 'treatment_given', 'critical_status'], inplace=True)
            if df.empty:
                raise ValueError("DataFrame is empty after dropping NaN values in critical columns. Cannot train models.")

            X_status = df[numeric_features]; y_status = df['critical_status']
            X_train_s, X_test_s, y_train_s, y_test_s = train_test_split(X_status, y_status, test_size=0.2, random_state=42, stratify=y_status)
            scaler = StandardScaler().fit(X_train_s)
            status_model = RandomForestClassifier(n_estimators=100, random_state=42).fit(scaler.transform(X_train_s), y_train_s)

            X_treat = df[numeric_features + ['chief_complaint']]; y_treat = df['treatment_given']
            preprocessor = ColumnTransformer(transformers=[('num', StandardScaler(), numeric_features), ('text', TfidfVectorizer(stop_words='english', min_df=2), 'chief_complaint')])
            treatment_model = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', RandomForestClassifier(n_estimators=150, random_state=42, class_weight='balanced'))])
            treatment_model.fit(X_treat, y_treat)

            os.makedirs(models_dir, exist_ok=True)
            joblib.dump(status_model, os.path.join(models_dir, 'best_model.pkl'))
            joblib.dump(treatment_model, os.path.join(models_dir, 'treatment_prediction_model.pkl'))
            joblib.dump(scaler, os.path.join(models_dir, 'scaler.pkl'))
            joblib.dump(numeric_features, os.path.join(models_dir, 'feature_cols.pkl'))
            st.success("✅ AI models trained successfully! The app will now reload.")
            time.sleep(3); st.rerun()
            return True
        except Exception as e:
            st.error(f"🚨 FAILED TO TRAIN MODELS: {e}"); return False

# --- Part 2: Main Application Logic ---
@st.cache_resource
def load_models_and_scaler():
    """Loads the pre-trained models and scaler from disk."""
    try: return {'status': joblib.load('./ml_models/best_model.pkl'), 'treatment': joblib.load('./ml_models/treatment_prediction_model.pkl'), 'scaler': joblib.load('./ml_models/scaler.pkl'), 'numeric_cols': joblib.load('./ml_models/feature_cols.pkl')}
    except FileNotFoundError: return None

@st.cache_data
def load_data_and_references():
    """Loads, cleans, and caches the main patient dataset from CSV files, then encrypts sensitive data."""
    try:
        data_file_to_load = "clean_ambulance_dataset.csv" 
        if not os.path.exists(data_file_to_load):
            st.error(f"Source file '{data_file_to_load}' not found. This indicates models were not trained correctly. Please ensure your `ambulance_dataset_300.csv` exists and restart the app after clearing old models.")
            return None, None, None

        df = pd.read_csv(data_file_to_load)
        df.columns = df.columns.str.strip().str.lower() # Safeguard, should be done in train_models_if_needed

        # --- ROBUST RENAMING FOR LOADED CLEAN DATA (SAFEGUARD) ---
        renaming_map = {
            'heart_rate': 'heart_rate_bpm',
            'spo2': 'spo2_percent',
            'systolic_bp_reading': 'systolic_bp_mmHg',
            'diastolic_bp_reading': 'diastolic_bp_mmHg',
            'respiratory_rate': 'respiratory_rate_bpm',
            'temp': 'temperature_c',
            'systolic_bp_mmhg': 'systolic_bp_mmHg',
            'diastolic_bp_mmhg': 'diastolic_bp_mmHg'
        }
        df.rename(columns={k: v for k, v in renaming_map.items() if k in df.columns}, inplace=True)
        
        # Initialize Fernet cipher for encryption
        cipher = get_fernet_cipher()

        # Encrypt a sensitive column (e.g., 'chief_complaint') for demonstration
        df['encrypted_chief_complaint'] = df['chief_complaint'].apply(lambda x: encrypt_data(x, cipher))

        # Now, use the standardized names for calculations
        avg_hr = pd.to_numeric(df.get('heart_rate_bpm'), errors='coerce').mean()
        avg_spo2 = pd.to_numeric(df.get('spo2_percent'), errors='coerce').mean()
        
        return df, avg_hr, avg_spo2
    except Exception as e:
        st.error(f"An error occurred while loading data for app references: {e}"); return None, None, None

def get_semantic_info(row_for_model, models):
    """Processes a patient data row to generate AI-driven alerts and status."""
    numeric_cols = models['numeric_cols']
    default_return = ("Error", "Data Error", "grey", "N/A", 2, "Unknown", "grey", 0, 0, 0, 0, 0, 0, [])
    
    # Ensure all required numeric columns are present in the incoming row
    missing_cols_in_row = [col for col in numeric_cols if col not in row_for_model.columns]
    if missing_cols_in_row:
        st.error(f"Model prediction failed. Missing columns in current row for semantic info: {missing_cols_in_row}.")
        return default_return

    # --- ROBUST SCALAR VALUE EXTRACTION ---
    # Extract scalar values from the 1-row DataFrame.
    # Use .iloc[0] to get the value from the single-element Series that results from df['col_name'].
    # Then pd.to_numeric for conversion and finally handle potential NaNs with 0.
    hr = pd.to_numeric(row_for_model['heart_rate_bpm'].iloc[0], errors='coerce')
    spo2 = pd.to_numeric(row_for_model['spo2_percent'].iloc[0], errors='coerce')
    sbp = pd.to_numeric(row_for_model['systolic_bp_mmHg'].iloc[0], errors='coerce')
    dbp = pd.to_numeric(row_for_model['diastolic_bp_mmHg'].iloc[0], errors='coerce')
    rr = pd.to_numeric(row_for_model['respiratory_rate_bpm'].iloc[0], errors='coerce')
    temp = pd.to_numeric(row_for_model['temperature_c'].iloc[0], errors='coerce')
    # Use .get() with a default Series for 'gcs' in case it's entirely missing, then .iloc[0]
    gcs_val = pd.to_numeric(row_for_model.get('gcs', pd.Series([np.nan])).iloc[0], errors='coerce') 
    
    # Fill any remaining NaNs with a default for safe comparisons (e.g., 0 for vitals is safer for critical status logic)
    hr, spo2, sbp, dbp, rr, temp, gcs_val_filled = [val if pd.notna(val) else 0 for val in [hr, spo2, sbp, dbp, rr, temp, gcs_val]]

    # Prepare data for model prediction - ensure correct order and types
    model_input_data = pd.DataFrame([[
        pd.to_numeric(row_for_model['age'].iloc[0], errors='coerce') if 'age' in row_for_model.columns else 0, # Safeguard for age
        hr, sbp, dbp, rr, spo2, temp, 
        row_for_model['chief_complaint'].iloc[0] if 'chief_complaint' in row_for_model.columns else '' # Safeguard for chief_complaint
    ]], columns=['age', 'heart_rate_bpm', 'systolic_bp_mmHg', 'diastolic_bp_mmHg', 'respiratory_rate_bpm', 'spo2_percent', 'temperature_c', 'chief_complaint'])
    
    # Re-order to match numeric_cols for scaler and then for pipeline's numeric part
    model_input_data = model_input_data[numeric_cols + ['chief_complaint']]

    ai_treatment = models['treatment'].predict(model_input_data)[0]

    audit_findings = []
    if hr < 40: audit_findings.append({'severity': 2, 'message': 'Critical: Severe Bradycardia (HR < 40)', 'short': 'Critically Low HR'})
    elif hr < 50: audit_findings.append({'severity': 1, 'message': 'Warning: Bradycardia (HR < 50)', 'short': 'Low HR'})
    if hr > 150: audit_findings.append({'severity': 2, 'message': 'Critical: Extreme Tachycardia (HR > 150)', 'short': 'Critically High HR'})
    elif hr > 110: audit_findings.append({'severity': 1, 'message': 'Warning: Tachycardia (HR > 110)', 'short': 'High HR'}) 
    if spo2 < 85: audit_findings.append({'severity': 2, 'message': 'Critical: Severe Hypoxia (SpO₂ < 85%)', 'short': 'Critically Low SpO₂'})
    elif spo2 < 92: audit_findings.append({'severity': 1, 'message': 'Warning: Hypoxia (SpO₂ < 92%)', 'short': 'Low SpO₂'})
    if sbp > 180 or dbp > 120: audit_findings.append({'severity': 2, 'message': 'Critical: Hypertensive Crisis (BP > 180/120)', 'short': 'Hypertensive Crisis'})
    elif sbp > 160 or dbp > 100: audit_findings.append({'severity': 1, 'message': 'Warning: Severe Hypertension', 'short': 'High BP'})
    if sbp < 90: audit_findings.append({'severity': 2, 'message': 'Critical: Severe Hypotension (SBP < 90)', 'short': 'Critically Low BP'})
    elif sbp < 100: audit_findings.append({'severity': 1, 'message': 'Warning: Hypotension (SBP < 100)', 'short': 'Low BP'})
    if rr < 8: audit_findings.append({'severity': 2, 'message': 'Critical: Severe Bradypnea (RR < 8)', 'short': 'Low Resp. Rate'})
    elif rr < 12: audit_findings.append({'severity': 1, 'message': 'Warning: Bradypnea (RR < 12)', 'short': 'Low Resp. Rate'})
    if rr > 30: audit_findings.append({'severity': 2, 'message': 'Critical: Severe Tachypnea (RR > 30)', 'short': 'High Resp. Rate'})
    elif rr > 22: audit_findings.append({'severity': 1, 'message': 'Warning: Tachypnea (RR > 22)', 'short': 'High Resp. Rate'})
    if temp > 40.0: audit_findings.append({'severity': 2, 'message': 'Critical: Hyperpyrexia (Temp > 40°C)', 'short': 'Critically High Temp'})
    elif temp > 38.5: audit_findings.append({'severity': 1, 'message': 'Warning: High Fever (Temp > 38.5°C)', 'short': 'High Temp'})
    if temp < 35.0: audit_findings.append({'severity': 2, 'message': 'Critical: Hypothermia (Temp < 35°C)', 'short': 'Low Temp'})

    if not audit_findings:
        max_severity, alert, short_alerts = 0, "All vitals stable.", []
    else:
        max_severity = max(f['severity'] for f in audit_findings)
        sorted_findings = sorted(audit_findings, key=lambda x: x['severity'], reverse=True)
        alert = " | ".join(f['message'] for f in sorted_findings)
        short_alerts = [f['short'] for f in sorted_findings]

    status_map = {0: "Normal", 1: "Warning", 2: "Critical"}
    status, priority = status_map[max_severity], max_severity
    color = {"Critical": "#D9534F", "Warning": "#F0AD4E", "Normal": "#5CB85C"}[status]

    consciousness_val = row_for_model.get('consciousness', pd.Series([""])).iloc[0].lower()
    consciousness_state, consciousness_color = ("Unconscious", "#D9534F") if 'unresponsive' in consciousness_val or (pd.notna(gcs_val_filled) and gcs_val_filled <= 8) else ("Conscious", "#5CB85C")

    NORMAL_HR, NORMAL_SPO2, NORMAL_TEMP = 80, 97, 37.0
    hr_delta_val = hr - NORMAL_HR
    spo2_delta_val = spo2 - NORMAL_SPO2
    temp_delta_val = temp - NORMAL_TEMP

    return status, alert, color, ai_treatment, priority, consciousness_state, consciousness_color, hr, spo2, temp, hr_delta_val, spo2_delta_val, temp_delta_val, short_alerts

def get_clinical_insights(patient_row, models_for_insights):
    """Analyzes patient data against a clinical knowledge base to find possible causes."""
    complaint = patient_row.get('chief_complaint', pd.Series([""])).iloc[0].lower()
    consciousness = patient_row.get('consciousness', pd.Series([""])).iloc[0].lower()
    
    # --- ROBUST SCALAR VALUE EXTRACTION ---
    hr = pd.to_numeric(patient_row['heart_rate_bpm'].iloc[0], errors='coerce')
    sys_bp = pd.to_numeric(patient_row['systolic_bp_mmHg'].iloc[0], errors='coerce')
    gcs_score = pd.to_numeric(patient_row.get('gcs', pd.Series([np.nan])).iloc[0], errors='coerce')
    temp_c = pd.to_numeric(patient_row['temperature_c'].iloc[0], errors='coerce')
    
    # Fill any remaining NaNs with a default for safe comparisons
    hr, sys_bp, gcs_score, temp_c_filled = [val if pd.notna(val) else 0 for val in [hr, sys_bp, gcs_score, temp_c]]

    is_low_gcs = gcs_score <= 8 if pd.notna(gcs_score) else False
    is_unconscious = 'unresponsive' in consciousness or 'unconscious' in complaint or is_low_gcs
    is_low_bp = sys_bp < 90 if pd.notna(sys_bp) else False
    is_high_hr = hr > 100 if pd.notna(hr) else False
    is_high_temp = temp_c > 38.0 if pd.notna(temp_c) else False

    ROOT_CAUSE_KEYWORDS = ["bleeding", "hemorrhage", "trauma", "fracture", "accident", "chest pain", "mi", "heart attack", "cardiac arrest", "burn", "seizure", "stroke", "poisoning", "overdose", "pregnancy", "allergy"]
    scored_rules = []
    patient_has_root_cause_keyword = any(keyword in complaint for keyword in ROOT_CAUSE_KEYWORDS)

    for rule in clinical_knowledge_base:
        primary_text_score, secondary_text_score, vitals_score, root_cause_bonus = 0, 0, 0, 0
        matched_symptoms = []

        for p_complaint in rule["primary_complaints"]:
            if p_complaint in complaint: primary_text_score += 10; matched_symptoms.append(p_complaint.title())
        for s_sign in rule["secondary_signs"]:
            if s_sign in complaint: secondary_text_score += 2; matched_symptoms.append(s_sign.title())

        if is_unconscious and any(s in rule["secondary_signs"] for s in ["unconscious", "confusion", "no response"]): vitals_score += 3; matched_symptoms.append("Unconscious/Confused")
        if is_low_bp and "low bp" in rule["secondary_signs"]: vitals_score += 3; matched_symptoms.append("Low BP")
        if is_high_hr and any(s in rule["secondary_signs"] for s in ["fast hr", "rapid pulse"]): vitals_score += 3; matched_symptoms.append("High HR")
        if is_high_temp and any(s in rule["secondary_signs"] for s in ["fever", "high temp"]): vitals_score += 3; matched_symptoms.append("Fever")

        if patient_has_root_cause_keyword and primary_text_score > 0 and any(p in ROOT_CAUSE_KEYWORDS for p in rule["primary_complaints"]):
            root_cause_bonus = 50

        total_score = primary_text_score + secondary_text_score + vitals_score + root_cause_bonus

        if total_score > 0:
            insight_data = rule.copy()
            insight_data["Patient Signs / Symptoms Matched"] = ", ".join(list(set(matched_symptoms))) if matched_symptoms else "Based on Chief Complaint"
            scored_rules.append({"score": total_score, "insight": insight_data})

    sorted_insights = sorted(scored_rules, key=lambda x: x['score'], reverse=True)
    return [item['insight'] for item in sorted_insights[:3]]

def generate_patient_report(df_full, models, cipher):
    """Generates a comprehensive report for all patients in the dataset."""
    st.subheader("Comprehensive Patient Report")
    
    search_query = st.text_input("Search Patient Report (ID, Complaint, Status, Insight, Treatment, Alerts)", "").lower()

    report_summary_data = []
    unique_pids = df_full['p_id'].unique()

    for pid in unique_pids:
        patient_df_all_records = df_full[df_full['p_id'] == pid].reset_index(drop=True)
        # Get the latest record for summary and current state assessment
        latest_row = patient_df_all_records.iloc[[len(patient_df_all_records) - 1]] 

        status, alert, color, ai_treatment, priority, cons_state, cons_color, hr, spo2, temp, hr_delta, spo2_delta, temp_delta, short_alerts = get_semantic_info(latest_row, models)
        insights = get_clinical_insights(latest_row, models)

        decrypted_cc = decrypt_data(latest_row['encrypted_chief_complaint'].iloc[0], cipher)

        report_summary_data.append({
            'Patient ID': pid,
            'Age': latest_row['age'].iloc[0],
            'Gender': latest_row['gender'].iloc[0],
            'Chief Complaint': decrypted_cc,
            'Heart Rate (bpm)': int(hr),
            'SpO₂ (%)': int(spo2),
            'BP (Systolic/Diastolic)': f"{int(latest_row['systolic_bp_mmHg'].iloc[0])}/{int(latest_row['diastolic_bp_mmHg'].iloc[0])}",
            'Resp. Rate (bpm)': int(latest_row['respiratory_rate_bpm'].iloc[0]),
            'Temp (°C)': temp,
            'Consciousness': cons_state,
            'AI Status': status,
            'AI Treatment': ai_treatment,
            'Top Insight': insights[0]['Possible Medical Cause'] if insights else 'N/A',
            'Alerts': ", ".join(short_alerts) if short_alerts else "None"
        })
    
    report_summary_df = pd.DataFrame(report_summary_data)

    # Filter report data based on search query
    if search_query:
        report_summary_df = report_summary_df[
            report_summary_df.apply(lambda row: 
                search_query in str(row['Patient ID']).lower() or
                search_query in str(row['Chief Complaint']).lower() or
                search_query in str(row['AI Status']).lower() or
                search_query in str(row['Top Insight']).lower() or
                search_query in str(row['AI Treatment']).lower() or
                search_query in str(row['Alerts']).lower(), axis=1
            )
        ]
        st.info(f"Displaying **{len(report_summary_df)}** patients matching '{search_query}'.")
    else:
        st.info(f"Displaying **{len(report_summary_df)}** unique patients in the report.")


    if not report_summary_df.empty:
        st.markdown("#### Summary of All Patients:")
        st.dataframe(report_summary_df.style.background_gradient(
            subset=['Heart Rate (bpm)', 'SpO₂ (%)', 'Temp (°C)'], cmap='RdYlGn_r'
        ).applymap(
            lambda x: f"background-color: {'#D9534F' if 'Critical' in str(x) else '#F0AD4E' if 'Warning' in str(x) else ''}",
            subset=['AI Status']
        ), use_container_width=True)

        st.markdown("#### Detailed Patient Records:")
        for i, data in report_summary_df.iterrows():
            pid = data['Patient ID']
            with st.expander(f"Patient {pid}: {data['Chief Complaint']} ({data['AI Status']})"):
                patient_full_data = df_full[df_full['p_id'] == pid].reset_index(drop=True)
                
                display_df = patient_full_data.copy()
                if 'encrypted_chief_complaint' in display_df.columns:
                    display_df['Chief Complaint (Decrypted)'] = patient_full_data['encrypted_chief_complaint'].apply(lambda x: decrypt_data(x, cipher))
                    display_df.drop(columns=['chief_complaint', 'encrypted_chief_complaint'], inplace=True, errors='ignore')
                else:
                    display_df.rename(columns={'chief_complaint': 'Chief Complaint'}, inplace=True)
                
                st.dataframe(display_df, use_container_width=True)
                
                latest_row_model = patient_full_data.iloc[[len(patient_full_data) - 1]].copy()
                _, _, _, _, _, _, _, _, _, _, _, _, _, latest_short_alerts = get_semantic_info(latest_row_model, models)

                st.markdown(f"**AI Suggested Action (Latest):** `{data['AI Treatment']}`")
                st.markdown(f"**Current Alerts (Latest):** {', '.join(latest_short_alerts) if latest_short_alerts else 'None'}")

                st.markdown("**Clinical Insights (Latest):**")
                insights_for_pid = get_clinical_insights(latest_row_model, models)
                if not insights_for_pid:
                    st.info("No specific clinical insights available for this patient's latest state.")
                else:
                    for idx, insight in enumerate(insights_for_pid):
                        st.markdown(f"**{idx+1}. Possible Cause:** {insight['Possible Medical Cause']}")
                        st.markdown(f"**Matched Symptoms:** {insight['Patient Signs / Symptoms Matched']}")
                        st.markdown(f"**Immediate Action:** {insight['Immediate Action / Ambulance / Doctor Steps']}")
                        st.markdown(f"**Possible Complications:** {insight['Possible Complications / What Might Happen']}")
                        if idx < len(insights_for_pid) - 1: st.markdown("---")
    else:
        st.info("No patients match your search query in the report.")


def display_single_patient_full_details(df_full, models, cipher, pid_to_show):
    """Displays full historical details for a single selected patient."""
    st.subheader(f"Full Details for Patient {pid_to_show}")
    patient_full_data = df_full[df_full['p_id'] == pid_to_show].reset_index(drop=True)

    if patient_full_data.empty:
        st.warning(f"No data found for Patient {pid_to_show}.")
        if st.button("⬅️ Back to Live Feed"):
            st.session_state.show_details_for_pid = None
            st.rerun()
        return

    latest_row_model = patient_full_data.iloc[[len(patient_full_data) - 1]].copy()
    status, alert, color, ai_treatment, priority, cons_state, cons_color, hr, spo2, temp, hr_delta, spo2_delta, temp_delta, short_alerts = get_semantic_info(latest_row_model, models)
    insights = get_clinical_insights(latest_row_model, models)

    st.markdown(f'<div style="border: 1px solid {color}; border-left: 5px solid {color}; padding: 10px; border-radius: 5px; background-color: #262730; margin-bottom: 10px;"><b>🚨 Clinical Audit Details (Latest):</b><br>{alert}</div>', unsafe_allow_html=True)

    left, right = st.columns([1, 1])
    with left:
        st.subheader("📋 Full Patient Record History")
        with st.container(border=True):
            display_df = patient_full_data.copy()
            if 'encrypted_chief_complaint' in display_df.columns:
                display_df['Chief Complaint (Decrypted)'] = patient_full_data['encrypted_chief_complaint'].apply(lambda x: decrypt_data(x, cipher))
                display_df.drop(columns=['chief_complaint', 'encrypted_chief_complaint'], inplace=True, errors='ignore')
            else:
                 display_df.rename(columns={'chief_complaint': 'Chief Complaint'}, inplace=True)

            st.dataframe(display_df, use_container_width=True)
        st.subheader("🔬 Prioritized Clinical Insights (Latest)")
        with st.container(border=True):
            if not insights: st.info("No clinical insights match the current condition.")
            else:
                for i, insight in enumerate(insights):
                    st.markdown(f"**Possible Cause:** {insight['Possible Medical Cause']}<br>**Matched Symptoms:** {insight['Patient Signs / Symptoms Matched']}<br>**Immediate Action:** {insight['Immediate Action / Ambulance / Doctor Steps']}<br>**Possible Complications:** {insight['Possible Complications / What Might Happen']}", unsafe_allow_html=True)
                    if i < len(insights) - 1: st.divider()
    with right:
        st.subheader("🚑 Emergency Protocols (Latest)")
        with st.container(border=True):
            st.success(f"**AI Suggested Action:** `{ai_treatment}`")
            st.divider()
            display_chief_complaint = decrypt_data(latest_row_model['encrypted_chief_complaint'].iloc[0], cipher).lower()
            relevant_keys = {p for k, p in COMPLAINT_TO_PROTOCOL.items() if k in display_chief_complaint}
            if ai_treatment in PROTOCOLS: relevant_keys.add(ai_treatment)

            if not relevant_keys:
                st.info("No specific protocols match. Displaying default care protocol.")
                st.markdown(PROTOCOLS["Default"])
            else:
                st.markdown("**Relevant Protocols for Reference:**")
                for key in sorted(list(relevant_keys)):
                    expander_title = PROTOCOLS[key].splitlines()[0].strip() if PROTOCOLS[key].splitlines() else key
                    with st.expander(f"View Protocol: {expander_title}", expanded=(key==ai_treatment)):
                        st.markdown(PROTOCOLS[key])
    
    if st.button("⬅️ Back to Live Feed"):
        st.session_state.show_details_for_pid = None
        st.session_state.run_simulation = False
        st.rerun()


def main_dashboard():
    """ The main application dashboard, shown after successful login. """
    load_css()

    if 'run_simulation' not in st.session_state: st.session_state.run_simulation = False
    if 'time_step' not in st.session_state: st.session_state.time_step = 0
    if 'patient_data' not in st.session_state: st.session_state.patient_data = []
    if 'show_details_for_pid' not in st.session_state: st.session_state.show_details_for_pid = None
    if 'show_report' not in st.session_state: st.session_state.show_report = False

    def set_details_and_stop_simulation(pid):
        st.session_state.show_details_for_pid = pid
        st.session_state.run_simulation = False
        st.session_state.show_report = False
        st.rerun()

    models_are_ready = train_models_if_needed()
    if not models_are_ready:
        st.error("Models could not be loaded or trained. The application cannot proceed.")
        st.stop()

    models = load_models_and_scaler()
    df_full, avg_hr_dummy, avg_spo2_dummy = load_data_and_references()

    if df_full is None:
        st.error("🚨 CRITICAL ERROR: Failed to load data. Please ensure 'ambulance_dataset_300.csv' exists and is correctly processed."); st.stop()

    st.title("🚑 Smart Ambulance - Clinical Decision Support")
    
    with st.sidebar:
        st.title("Dashboard Controls")
        
        with st.popover("👤 Profile & Actions", use_container_width=True):
            st.markdown(f"**Logged in as:**\n`{st.session_state.get('user_email', 'Not logged in')}`")
            st.divider()
            if st.button("Logout", use_container_width=True, type="primary"):
                for key in list(st.session_state.keys()):
                    del st.session_state[key]
                st.rerun()
        
        st.divider()

        df_full.rename(columns={df_full.columns[0]: 'p_id'}, inplace=True, errors='ignore')
        df_full['p_id'] = df_full['p_id'].astype(str)
        patient_options = sorted(df_full['p_id'].unique())
        default_selection = patient_options[:4] if len(patient_options) >= 4 else patient_options
        selected_patients = st.multiselect("Select Patients", options=patient_options, default=default_selection)

        st.subheader("Simulation Control")
        if st.button("Start / Restart Live Feed", type="secondary", use_container_width=True):
            st.session_state.run_simulation = True
            st.session_state.time_step = 0
            st.session_state.show_report = False
            st.session_state.show_details_for_pid = None
            st.rerun()
        if st.button("Stop Live Feed", use_container_width=True):
            st.session_state.run_simulation = False
            st.rerun()

        st.divider()
        if st.button("Generate Patient Report", type="primary", use_container_width=True):
            st.session_state.show_report = True
            st.session_state.run_simulation = False
            st.session_state.show_details_for_pid = None
            st.rerun()

    if st.session_state.show_details_for_pid:
        display_single_patient_full_details(df_full, models, get_fernet_cipher(), st.session_state.show_details_for_pid)
    elif st.session_state.show_report:
        generate_patient_report(df_full, models, get_fernet_cipher())
    elif not selected_patients:
        st.warning("Please select at least one patient from the sidebar to begin.")
    else:
        if not st.session_state.run_simulation:
            st.info("Dashboard is ready. Press 'Start / Restart Live Feed' to begin the simulation.")
        else:
            patient_data_at_timestep = []
            max_len = 0
            for pid in selected_patients:
                patient_df_selected = df_full[df_full['p_id'] == pid]
                if len(patient_df_selected) > max_len:
                    max_len = len(patient_df_selected)

            if st.session_state.time_step < max_len:
                cipher = get_fernet_cipher()
                for pid in selected_patients:
                    patient_df = df_full[df_full['p_id'] == pid].reset_index(drop=True)
                    current_index = min(st.session_state.time_step, len(patient_df) - 1)
                    current_row_original = patient_df.iloc[[current_index]]

                    model_row = current_row_original.copy()
                    
                    status, alert, color, ai_treatment, priority, cons_state, cons_color, hr, spo2, temp, hr_delta, spo2_delta, temp_delta, short_alerts = get_semantic_info(model_row, models)
                    
                    decrypted_chief_complaint = decrypt_data(current_row_original['encrypted_chief_complaint'].iloc[0], cipher) if 'encrypted_chief_complaint' in current_row_original.columns else current_row_original['chief_complaint'].iloc[0]

                    patient_data_at_timestep.append({
                        'pid': pid, 'original_row': current_row_original, 'status': status, 'alert': alert, 'short_alerts': short_alerts,
                        'color': color, 'insights': get_clinical_insights(model_row, models), 'ai_treatment': ai_treatment, 'priority': priority,
                        'consciousness_state': cons_state, 'consciousness_color': cons_color,
                        'hr': hr, 'spo2': spo2, 'temp_c': temp,
                        'hr_delta_val': hr_delta, 'spo2_delta_val': spo2_delta, 'temp_delta_val': temp_delta,
                        'decrypted_chief_complaint': decrypted_chief_complaint
                    })
                st.session_state.patient_data = sorted(patient_data_at_timestep, key=lambda x: x['priority'], reverse=True)
            else:
                if st.session_state.run_simulation:
                    st.success("All selected patient data streams have concluded.")
                    st.session_state.run_simulation = False

            st.subheader(f"Live Feed (Time Step: {st.session_state.time_step})")
            if st.session_state.patient_data:
                cols = st.columns(min(len(st.session_state.patient_data), 4))
                for i, data in enumerate(st.session_state.patient_data):
                    with cols[i % 4]:
                        with st.container():
                            st.markdown(f"""
                            <div class="patient-card">
                                <div class="card-header card-header-{data['status'].lower()}">
                                    <span>Patient: {data['pid']}</span>
                                    <span>{data['status']}</span>
                                </div>
                            """, unsafe_allow_html=True)
                            
                            if data['status'] == 'Error':
                                st.error(f"{data['alert']}")
                            else:
                                c1, c2, c3 = st.columns(3)
                                with c1:
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <div class="metric-label">❤️ Heart Rate</div>
                                        <div class="metric-value">{int(data['hr'])} <span style="font-size: 1rem;">bpm</span></div>
                                        <div class="metric-delta {'delta-up' if data['hr_delta_val'] > 0 else 'delta-down'}">
                                            {'↑' if data['hr_delta_val'] > 0 else '↓'} {abs(data['hr_delta_val']):.0f}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with c2:
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <div class="metric-label">💨 SpO₂</div>
                                        <div class="metric-value">{int(data['spo2'])} <span style="font-size: 1rem;">%</span></div>
                                        <div class="metric-delta {'delta-up' if data['spo2_delta_val'] > 0 else 'delta-down'}">
                                            {'↑' if data['spo2_delta_val'] > 0 else '↓'} {abs(data['spo2_delta_val']):.0f}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)
                                with c3:
                                    st.markdown(f"""
                                    <div class="metric-container">
                                        <div class="metric-label">🌡️ Temp</div>
                                        <div class="metric-value">{data['temp_c']:.1f} <span style="font-size: 1rem;">°C</span></div>
                                        <div class="metric-delta {'delta-up' if data['temp_delta_val'] > 0.5 else 'delta-down'}">
                                            {'↑' if data['temp_delta_val'] > 0 else '↓'} {abs(data['temp_delta_val']):.1f}
                                        </div>
                                    </div>
                                    """, unsafe_allow_html=True)

                                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                                st.markdown(f"<span class='detail-label'>🧠 Consciousness:</span> <span class='detail-value-{data['consciousness_state'].lower()}'>{data['consciousness_state']}</span>", unsafe_allow_html=True)
                                
                                alert_list_html = "".join([f"<li>{a}</li>" for a in data['short_alerts']]) if data['short_alerts'] else "<li>All Vitals Stable</li>"
                                st.markdown(f"<div class='detail-label'>🚨 Alerts:</div><div class='alert-list'><ul>{alert_list_html}</ul></div>", unsafe_allow_html=True)
                                
                                st.markdown('<div class="custom-divider"></div>', unsafe_allow_html=True)
                                top_cause = data['insights'][0]['Possible Medical Cause'] if data['insights'] else "Awaiting Data..."
                                st.markdown(f"<span class='detail-label'>🔬 Possible Cause:</span> {top_cause}", unsafe_allow_html=True)
                                st.markdown(f"<span class='detail-label'>💡 Suggested Action:</span> `{data['ai_treatment']}`", unsafe_allow_html=True)
                                
                                st.markdown("</div>", unsafe_allow_html=True)
                                st.button("View Full Details", key=f"details_live_{data['pid']}", on_click=set_details_and_stop_simulation, args=[data['pid']])
            
            if st.session_state.run_simulation:
                st.session_state.time_step += 1
                time.sleep(1000000)
                st.rerun()

# --- Main App Router ---
if __name__ == "__main__":
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False

    load_css()
    if not st.session_state.logged_in:
        login_page()
    else:
        main_dashboard()