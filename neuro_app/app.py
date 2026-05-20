"""
Neuroimmunology Unified Diagnostic System - v8.3
Streamlit app: ML Risk Score · SHAP · Condition Predictor · Antibody Recommender

Enhancements in v8.3:
- Searchable symptom selector with up to 20 selections.
- Searchable co-morbidity selector with up to 15 selections.
- Expanded neuroimmunology symptom and co-morbidity knowledge base.
- Condition scoring now incorporates expanded symptoms, co-morbidities, tests,
  demographics, antibody results, titre burden, and sample type.
- Cleaner dark clinical UI optimized for doctors entering cases quickly.
"""

import datetime
import html
import io
import json
import math
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


APP_DIR = Path(__file__).resolve().parent
LEARNING_DB_PATH = APP_DIR / "neuro_learning.sqlite"
ADAPTIVE_MODEL_PATH = APP_DIR / "adaptive_model.json"


# ─────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Neuroimmunology Unified Diagnostic System",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ─────────────────────────────────────────────
#  CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Manrope:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
    --bg-0: #030712;
    --bg-1: #07111f;
    --bg-2: #0b1628;
    --panel: rgba(12, 24, 42, 0.72);
    --panel-strong: rgba(15, 29, 51, 0.88);
    --line: rgba(125, 211, 252, 0.20);
    --line-soft: rgba(148, 163, 184, 0.16);
    --text: #e5f2ff;
    --muted: #8ea8c3;
    --subtle: #5f7894;
    --cyan: #22d3ee;
    --cyan-2: #67e8f9;
    --blue: #38bdf8;
    --green: #34d399;
    --amber: #fbbf24;
    --rose: #fb7185;
    --shadow: 0 24px 70px rgba(0, 0, 0, 0.40);
}

html, body, [class*="css"] {
    font-family: 'Manrope', sans-serif !important;
}

#MainMenu, footer, header { visibility: hidden; }
.block-container {
    padding-top: 0 !important;
    max-width: 1420px !important;
    color: var(--text);
}

.stApp,
[data-testid="stAppViewContainer"],
[data-testid="stMain"] {
    background:
      linear-gradient(rgba(34, 211, 238, 0.035) 1px, transparent 1px),
      linear-gradient(90deg, rgba(34, 211, 238, 0.030) 1px, transparent 1px),
      radial-gradient(circle at 18% 0%, rgba(34, 211, 238, 0.20), transparent 34rem),
      radial-gradient(circle at 86% 12%, rgba(56, 189, 248, 0.14), transparent 28rem),
      linear-gradient(180deg, var(--bg-0) 0%, var(--bg-1) 38%, var(--bg-2) 100%) !important;
    background-size: 34px 34px, 34px 34px, auto, auto, auto !important;
}

.top-banner {
    position: relative;
    overflow: hidden;
    background:
      linear-gradient(135deg, rgba(7, 17, 31, 0.82), rgba(13, 38, 64, 0.70)),
      radial-gradient(circle at 8% 12%, rgba(34, 211, 238, 0.24), transparent 24rem);
    color: var(--text);
    padding: 28px 34px 26px;
    border-radius: 0 0 22px 22px;
    margin-bottom: 28px;
    box-shadow: var(--shadow);
    border: 1px solid rgba(125, 211, 252, 0.22);
    backdrop-filter: blur(18px);
}
.top-banner::after {
    content: "";
    position: absolute;
    inset: 0;
    pointer-events: none;
    background:
      linear-gradient(90deg, transparent 0%, rgba(103, 232, 249, 0.22) 50%, transparent 100%);
    height: 1px;
    top: auto;
}
.top-banner h1 {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: clamp(26px, 3vw, 40px);
    font-weight: 700;
    letter-spacing: -0.3px;
    margin: 0 0 6px 0;
    color: var(--text) !important;
}
.top-banner p  { font-size: 13px; color: #a8dff0 !important; margin: 0; }
.top-banner span { color: inherit !important; }
.badge-row { display: flex; gap: 9px; flex-wrap: wrap; margin-top: 15px; }
.badge {
    background: rgba(8, 20, 36, 0.76);
    border: 1px solid rgba(103, 232, 249, 0.26);
    color: #dffaff !important;
    font-size: 11px;
    font-weight: 800;
    padding: 5px 12px;
    border-radius: 999px;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.06);
}

div[data-testid="stMarkdownContainer"] h1,
div[data-testid="stMarkdownContainer"] h2,
div[data-testid="stMarkdownContainer"] h3,
div[data-testid="stMarkdownContainer"] h4,
div[data-testid="stMarkdownContainer"] h5 {
    font-family: 'Space Grotesk', sans-serif !important;
    color: var(--text) !important;
    letter-spacing: -0.2px;
}
div[data-testid="stMarkdownContainer"] p,
div[data-testid="stMarkdownContainer"] strong,
div[data-testid="stMarkdownContainer"] em,
div[data-testid="stMarkdownContainer"] li,
div[data-testid="stMarkdownContainer"] small {
    color: var(--muted) !important;
}
div[data-testid="stMarkdownContainer"] strong { color: var(--text) !important; }

div[data-testid="stWidgetLabel"] p,
div[data-testid="stCheckbox"] p,
div[data-testid="stSlider"] p,
div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] p,
div[data-testid="stCheckbox"] label,
div[data-testid="stCheckbox"] span,
div[data-testid="stNumberInput"] label,
div[data-testid="stSelectbox"] label,
div[data-testid="stMultiSelect"] label {
    color: var(--text) !important;
    font-weight: 700;
}
div[data-testid="stCaptionContainer"],
div[data-testid="stCaptionContainer"] p {
    color: var(--muted) !important;
    font-weight: 500;
}

.sec-label {
    display: flex;
    align-items: center;
    gap: 13px;
    margin: 32px 0 16px;
    padding: 14px 16px;
    background: linear-gradient(135deg, rgba(15, 29, 51, 0.72), rgba(8, 20, 36, 0.58));
    border: 1px solid var(--line-soft);
    border-left: 3px solid var(--cyan);
    border-radius: 16px;
    box-shadow: 0 14px 34px rgba(0,0,0,0.22);
}
.sec-num {
    width: 34px;
    height: 34px;
    border-radius: 10px;
    background: linear-gradient(135deg, var(--cyan), #2563eb);
    color: #00111d !important;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 14px;
    font-weight: 900;
    flex-shrink: 0;
    box-shadow: 0 0 28px rgba(34, 211, 238, 0.35);
}
.sec-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 18px;
    font-weight: 700;
    color: var(--text);
    letter-spacing: -0.2px;
}
.sec-sub { font-size: 12px; color: var(--muted); margin-top: 2px; }

.metric-strip {
    display: grid;
    grid-template-columns: repeat(4, minmax(0, 1fr));
    gap: 12px;
    margin: 16px 0 10px;
}
.mini-metric {
    background: linear-gradient(180deg, rgba(15, 29, 51, 0.88), rgba(7, 17, 31, 0.74));
    border: 1px solid rgba(125, 211, 252, 0.20);
    border-radius: 14px;
    padding: 13px 15px;
    box-shadow: 0 14px 36px rgba(0, 0, 0, 0.28);
}
.mini-metric .label {
    font-size: 10px;
    color: var(--muted);
    text-transform: uppercase;
    letter-spacing: 0.8px;
    font-weight: 900;
}
.mini-metric .value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px;
    color: var(--cyan-2);
    font-weight: 700;
    margin-top: 4px;
}

.hint-box {
    background: linear-gradient(180deg, rgba(15, 29, 51, 0.92), rgba(6, 18, 32, 0.80));
    border: 1px solid rgba(103, 232, 249, 0.22);
    color: var(--muted);
    border-radius: 14px;
    padding: 14px 16px;
    font-size: 12px;
    line-height: 1.65;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04);
}
.hint-box strong { color: var(--cyan-2) !important; }

.chip-wrap {
    display: flex;
    flex-wrap: wrap;
    gap: 8px;
    max-height: 128px;
    overflow-y: auto;
    padding: 4px 3px 8px;
}
.chip {
    background: rgba(34, 211, 238, 0.12);
    color: #b8f7ff;
    border: 1px solid rgba(103, 232, 249, 0.28);
    font-size: 11px;
    font-weight: 800;
    padding: 5px 10px;
    border-radius: 999px;
}
.chip.comorb {
    background: rgba(52, 211, 153, 0.12);
    color: #c6f7df;
    border-color: rgba(52, 211, 153, 0.30);
}
.chip.hot {
    background: rgba(251, 113, 133, 0.13);
    color: #ffd6dd;
    border-color: rgba(251, 113, 133, 0.30);
}

div[data-testid="stVerticalBlockBorderWrapper"] {
    background: rgba(9, 20, 36, 0.66);
    border-color: rgba(125, 211, 252, 0.20) !important;
    border-radius: 16px !important;
    box-shadow: 0 14px 34px rgba(0,0,0,0.22);
}

div[data-testid="stFileUploader"] {
    background: rgba(9, 20, 36, 0.72);
    border: 1px solid rgba(125, 211, 252, 0.18);
    border-radius: 16px;
    padding: 12px;
    box-shadow: 0 16px 38px rgba(0,0,0,0.22);
}
div[data-testid="stFileUploader"] section {
    background: rgba(3, 7, 18, 0.58) !important;
    border: 1px dashed rgba(103, 232, 249, 0.30) !important;
    border-radius: 12px !important;
}

div[data-testid="stMultiSelect"] [data-baseweb="select"],
div[data-testid="stSelectbox"] [data-baseweb="select"],
div[data-testid="stNumberInput"] input {
    background: rgba(5, 12, 24, 0.92) !important;
    color: var(--text) !important;
    border: 1px solid rgba(125, 211, 252, 0.22) !important;
    border-radius: 12px !important;
    box-shadow: inset 0 1px 0 rgba(255,255,255,0.04), 0 12px 26px rgba(0,0,0,0.18);
}
div[data-testid="stMultiSelect"] [data-baseweb="select"]:focus-within,
div[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within,
div[data-testid="stNumberInput"] input:focus {
    border-color: rgba(34, 211, 238, 0.72) !important;
    box-shadow: 0 0 0 1px rgba(34, 211, 238, 0.35), 0 0 26px rgba(34, 211, 238, 0.12);
}
div[data-testid="stMultiSelect"] [data-baseweb="tag"] {
    max-width: 100%;
    background: rgba(34, 211, 238, 0.14) !important;
    color: #dffaff !important;
    border: 1px solid rgba(103, 232, 249, 0.26) !important;
}
div[data-baseweb="popover"],
div[data-baseweb="menu"],
ul[role="listbox"] {
    background: rgba(5, 12, 24, 0.98) !important;
    border: 1px solid rgba(103, 232, 249, 0.25) !important;
    border-radius: 14px !important;
    box-shadow: 0 22px 54px rgba(0,0,0,0.48) !important;
    color: var(--text) !important;
}
li[role="option"],
div[role="option"] {
    color: var(--text) !important;
    background: transparent !important;
}
li[role="option"]:hover,
div[role="option"]:hover {
    background: rgba(34, 211, 238, 0.13) !important;
}
div[data-baseweb="popover"] * {
    color: var(--text) !important;
}

div[data-testid="stButton"] > button {
    background: linear-gradient(135deg, #22d3ee, #2563eb) !important;
    color: #00111d !important;
    font-weight: 900 !important;
    font-size: 15px !important;
    padding: 12px 30px !important;
    border: none !important;
    border-radius: 12px !important;
    box-shadow: 0 16px 34px rgba(34,211,238,0.20) !important;
    width: 100%;
}
div[data-testid="stButton"] > button:hover {
    filter: brightness(1.08);
    transform: translateY(-1px);
}
div[data-testid="stButton"] > button p {
    color: #00111d !important;
    font-weight: 900 !important;
}

div[data-testid="stSlider"] [data-baseweb="slider"] div {
    color: var(--text) !important;
}
div[data-testid="stCheckbox"] [data-testid="stWidgetLabel"] p {
    color: var(--text) !important;
}
hr { margin: 1.5rem 0; border-color: rgba(125,211,252,0.18); }

@media (max-width: 900px) {
    .metric-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    .top-banner { padding: 22px 20px; }
}
</style>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  KNOWLEDGE BASE
# ─────────────────────────────────────────────
SYMPTOM_GROUPS = {
    "Motor / Strength / Motor Neuron": [
        "Symmetrical weakness", "Asymmetrical weakness", "Proximal weakness", "Distal weakness",
        "Progressive weakness", "Acute onset weakness", "Subacute onset weakness", "Fluctuating weakness",
        "Episodic weakness", "Muscle weakness", "Fatigable weakness", "Exercise-induced weakness",
        "Weakness improves with rest", "Weakness improves with exercise", "Rapidly progressive weakness",
        "Ascending weakness", "Descending weakness", "Quadriparesis", "Hemiparesis", "Monoparesis",
        "Paraparesis", "Progressive quadriparesis", "Acute flaccid weakness", "Relapsing weakness",
        "Generalized weakness", "Unilateral weakness", "Bilateral weakness", "Limb-girdle weakness",
        "Scapular winging", "Pelvic girdle weakness", "Neck flexor weakness", "Neck extensor weakness",
        "Neck weakness", "Head drop", "Respiratory muscle weakness", "Loss of grip strength",
        "Grip weakness", "Dropping objects", "Dropping objects frequently", "Difficulty opening bottles",
        "Difficulty buttoning shirts", "Difficulty writing", "Hand clumsiness", "Poor finger dexterity",
        "Difficulty using keys", "Weak wrist extension", "Wrist drop", "Finger weakness", "Thumb weakness",
        "Difficulty lifting objects overhead", "Shoulder girdle weakness", "Arm fatigue while combing hair",
        "Difficulty holding utensils", "Difficulty reaching overhead", "Difficulty lifting arms",
        "Difficulty carrying bags", "Loss of hand muscle bulk", "Thenar wasting", "Interosseous wasting",
        "Foot drop", "Toe dragging", "Difficulty climbing stairs", "Difficulty rising from chair",
        "Difficulty standing from squatting", "Frequent tripping", "Knee buckling", "Difficulty running",
        "Leg heaviness", "Difficulty walking long distances", "Difficulty standing for long periods",
        "Difficulty getting out of bed", "Difficulty turning in bed", "Difficulty rising from floor",
        "Difficulty stepping over obstacles", "Calf weakness", "Hip flexor weakness", "Ankle dorsiflexion weakness",
        "Toe walking due to weakness", "Muscle wasting", "Fasciculations", "Tongue fasciculations",
        "Tongue atrophy", "Limb atrophy", "Split hand syndrome", "Spasticity", "Hyperreflexia",
        "Brisk reflexes", "Babinski sign", "Jaw jerk exaggerated", "Emotional lability",
        "Pseudobulbar affect", "Mixed upper and lower motor neuron signs", "Motor neuron involvement concern",
        "Upper motor neuron signs", "Lower motor neuron signs", "Diffuse fasciculations",
        "Cramps with fasciculations", "Reduced reflexes", "Absent reflexes", "Areflexia", "Hyporeflexia",
        "Asymmetric reflexes", "Delayed reflex relaxation", "Rigidity", "Stiffness", "Stiffness and rigidity",
        "Painful spasms", "Cramping", "Muscle tightness", "Muscle twitching", "Myokymia", "Neuromyotonia",
        "Dystonia", "Chorea", "Tremor", "Rest tremor", "Intention tremor", "Action tremor", "Postural tremor",
        "Bradykinesia", "Freezing of gait", "Abnormal posturing", "Myoclonus", "Stimulus-triggered spasms",
        "Startle-induced spasms", "Axial rigidity", "Limb stiffness", "Back spasms", "Abdominal wall spasms",
        "Trismus", "Exercise-induced cramps",
    ],
    "Sensory / Peripheral Nerve": [
        "Tingling", "Numbness", "Pins and needles sensation", "Burning sensation", "Electric shock sensation",
        "Electric shock sensations", "Crawling sensation", "Tight band sensation", "Cold sensation in limbs",
        "Hot sensation in limbs", "Shooting pain", "Radicular pain", "Allodynia", "Hyperesthesia",
        "Hyperalgesia", "Neuropathic pain", "Painful paresthesias", "Loss of sensation",
        "Reduced touch sensation", "Reduced vibration sense", "Loss of vibration sense",
        "Reduced joint position sense", "Loss of proprioception", "Distal sensory loss",
        "Glove and stocking sensory loss", "Saddle anesthesia", "Large-fiber sensory loss",
        "Small-fiber symptoms", "Length-dependent sensory loss", "Patchy sensory loss",
        "Sensory level", "Band-like sensory level", "Loss of balance", "Coordination issues",
        "Cerebellar ataxia", "Sensory ataxia", "Broad-based gait", "Veering while walking",
        "Frequent falls", "Difficulty tandem walking", "Difficulty walking in dark",
        "Swaying while standing", "Positive Romberg", "Positive Romberg sign", "Dizziness",
        "Vertigo", "Oscillopsia", "Carpal tunnel symptoms", "Ulnar neuropathy symptoms",
        "Peroneal neuropathy symptoms", "Facial numbness", "Mononeuritis multiplex pattern",
        "Multifocal neuropathy pattern", "Stocking distribution numbness", "Hand numbness",
        "Feet numbness", "Burning feet", "Painful numbness",
    ],
    "Cranial Nerve / Bulbar / NMJ": [
        "Ptosis", "Diplopia", "Blurred vision", "Ophthalmoplegia", "Gaze palsy", "Difficulty focusing",
        "Visual fatigue", "Eye movement restriction", "Nystagmus", "Dysphagia", "Choking while eating",
        "Nasal speech", "Slurred speech", "Dysarthria", "Bulbar weakness", "Bulbar involvement",
        "Difficulty chewing", "Jaw fatigue", "Pooling of saliva", "Excessive drooling", "Tongue weakness",
        "Difficulty swallowing liquids", "Difficulty swallowing solids", "Weak cough", "Facial weakness",
        "Facial droop", "Facial twitching", "Hearing loss", "Tinnitus", "Loss of taste",
        "Reduced facial sensation", "Diurnal fluctuation", "Diurnal fluctuation of weakness",
        "Symptoms worse in evening", "Symptoms worsen in evening", "Proximal weakness improves with exercise",
        "Diplopia worsens with fatigue", "Ptosis worsens through the day", "Neck weakness after prolonged activity",
        "Weakness worsens with exertion", "Myasthenic crisis symptoms", "Jaw claudication", "Hoarse voice",
        "Hypophonia", "Dysphonia", "Nasal regurgitation", "Aspiration while swallowing",
    ],
    "Respiratory / Chest Muscle": [
        "Shortness of breath", "Orthopnea", "Breathlessness while lying down", "Reduced cough strength",
        "Difficulty taking deep breaths", "Morning headaches", "Daytime sleepiness", "Nocturnal hypoventilation",
        "Breathlessness on exertion", "Accessory muscle use", "Reduced spirometry values", "Respiratory fatigue",
        "Recurrent aspiration", "Recurrent chest infections", "Respiratory involvement", "Reduced forced vital capacity",
        "Sleep-related hypoventilation", "Difficulty clearing secretions", "Breath counting reduced",
        "Respiratory failure symptoms", "Non-invasive ventilation requirement", "Weak sneeze",
    ],
    "CNS / Autoimmune Encephalitis": [
        "Memory loss", "Short-term memory impairment", "Confusion", "Behavioral change", "Personality change",
        "Irritability", "Anxiety", "Depression", "Hallucinations", "Delusions", "Agitation",
        "Reduced attention span", "Executive dysfunction", "Disorientation", "Catatonia", "Psychiatric symptoms",
        "Delirium", "Cognitive decline", "Rapidly progressive dementia", "Encephalopathy", "Seizures",
        "Faciobrachial dystonic seizures", "Focal seizures", "Generalized seizures", "Status epilepticus",
        "New onset seizures", "Decreased consciousness", "Altered sensorium", "Language dysfunction",
        "Aphasia", "Cognitive slowing", "Sleep disorder", "REM sleep behavior disorder",
        "Excessive daytime sleepiness", "Insomnia", "Severe insomnia", "Hypersomnia", "Orofacial dyskinesias",
        "Movement disorder (orofacial dyskinesias)", "Orofacial dyskinesia", "Parkinsonism", "Ataxia",
        "Headache", "Fever with encephalopathy",
    ],
    "Visual / Spinal Cord / Demyelination": [
        "Visual loss", "Visual loss (optic neuritis)", "Optic neuritis", "Bilateral optic neuritis",
        "Eye pain", "Pain on eye movement", "Painful eye movements", "Reduced color vision",
        "Central scotoma", "Double vision", "Transverse myelitis", "Spinal cord symptoms",
        "Longitudinally extensive myelitis", "Sensory level on trunk", "Lhermitte sign",
        "Band-like trunk sensation", "Weakness below waist", "Bilateral leg weakness", "Spastic gait",
        "Stiff legs", "Bowel dysfunction", "Bladder dysfunction", "Urinary urgency", "Urinary retention",
        "Urinary incontinence", "Constipation due to myelopathy", "Sexual dysfunction",
        "Conus symptoms", "Brown-Sequard pattern", "Myelopathic hand", "Spinal shock symptoms",
        "Area postrema symptoms", "Intractable vomiting", "Hiccups and nausea (area postrema)",
    ],
    "Autonomic / Systemic": [
        "Autonomic instability", "Orthostatic hypotension", "Postural dizziness", "Orthostatic dizziness",
        "Syncope", "Abnormal sweating", "Reduced sweating", "Excessive sweating", "Anhidrosis",
        "Palpitations", "Resting tachycardia", "Tachycardia episodes", "Blood pressure fluctuations",
        "Gastroparesis", "Early satiety", "Constipation", "Diarrhea", "Erectile dysfunction",
        "Dry mouth", "Dry eyes", "Fatigue", "Weight loss", "Unintentional weight loss", "Fever",
        "Night sweats", "Loss of appetite", "Hyponatraemia", "Chronic pain", "Exercise intolerance",
        "Heat intolerance", "Cold intolerance", "Postural intolerance", "Urinary hesitancy",
    ],
    "Pain Features": [
        "Burning pain", "Electric shock pain", "Stabbing pain", "Tingling pain", "Neck pain",
        "Back pain", "Shoulder pain", "Muscle pain", "Joint pain", "Calf pain", "Painful cramps",
        "Diffuse body pain", "Root pain", "Plexus pain", "Mechanical back pain", "Inflammatory back pain",
        "Radicular neck pain", "Radicular back pain", "Painful spasms at night", "Pain-limited walking",
    ],
    "Gait / Mobility / Function": [
        "Difficulty walking", "Slow gait", "Wide-based gait", "Shuffling gait", "Steppage gait",
        "Ataxic gait", "Festinating gait", "Toe walking", "Difficulty turning", "Falls", "Near falls",
        "Requires support to walk", "Wheelchair dependence", "Cane/walker dependence", "Unable to climb stairs",
        "Gait instability", "Waddling gait", "Reduced walking speed", "Difficulty initiating gait",
        "Difficulty stopping while walking", "Difficulty on uneven ground", "Unable to stand unaided",
        "Unable to walk unaided", "Bedbound due to weakness",
    ],
    "Pediatric / Hereditary / Exposure Clues": [
        "Delayed milestones", "Family history of neurological disease", "Consanguinity",
        "Recurrent episodic weakness", "Childhood onset symptoms", "Progressive hereditary neuropathy",
        "Foot deformities", "Pes cavus", "Scoliosis", "High arched feet", "Hammer toes",
        "Longstanding clumsiness", "Exercise intolerance since childhood", "Developmental regression",
        "Heavy metal exposure symptoms", "Occupational chemical exposure symptoms", "Toxin exposure symptoms",
        "Medication-associated weakness", "Chemotherapy-associated neuropathy symptoms",
    ],
    "Red Flag / Tertiary Referral Features": [
        "Rapid progression", "Severe autonomic instability", "Cancer history", "Immunosuppression",
        "Exposure to toxins", "Heavy metal exposure", "Occupational chemical exposure",
        "Secondary cause workup needed", "Genetic testing indication", "SOD1 mutation concern",
        "Exome sequencing indication", "Autoimmune neuropathy panel indication", "Spirometry abnormality",
        "Breathing assessment abnormality", "Progressive disability", "Acute bulbar decline",
        "Acute respiratory decline", "Severe weight loss with neuropathy", "Multisystem neurological involvement",
    ],
}

SYMPTOM_GROUP_LIMITS = {
    "Motor / Strength / Motor Neuron": 85,
    "Sensory / Peripheral Nerve": 45,
    "Cranial Nerve / Bulbar / NMJ": 40,
    "Respiratory / Chest Muscle": 18,
    "CNS / Autoimmune Encephalitis": 38,
    "Visual / Spinal Cord / Demyelination": 30,
    "Autonomic / Systemic": 25,
    "Pain Features": 12,
    "Gait / Mobility / Function": 12,
    "Pediatric / Hereditary / Exposure Clues": 4,
    "Red Flag / Tertiary Referral Features": 3,
}

HIGH_VALUE_SYMPTOMS = [
    "Loss of grip strength", "Dropping objects", "Foot drop", "Difficulty climbing stairs",
    "Difficulty rising from chair", "Fasciculations", "Tongue fasciculations", "Muscle wasting",
    "Head drop", "Weak cough", "Orthopnea", "Frequent falls", "Sensory ataxia",
    "Positive Romberg", "Diurnal fluctuation", "Symptoms worsen in evening", "Choking while eating",
    "Pooling of saliva", "Urinary retention", "Orthostatic dizziness", "Lhermitte sign",
    "Electric shock sensations", "Glove and stocking sensory loss", "Band-like sensory level",
    "Pain on eye movement", "Faciobrachial dystonic seizures", "REM sleep behavior disorder",
    "Emotional lability", "Pseudobulbar affect", "Split hand syndrome",
]


def build_limited_options(groups, limits, required, total):
    selected = []
    seen = set()
    for group_name, items in groups.items():
        for item in items[: limits.get(group_name, len(items))]:
            if item not in seen:
                selected.append(item)
                seen.add(item)

    all_items = {item for items in groups.values() for item in items}
    for item in required:
        if item in all_items and item not in seen:
            selected.append(item)
            seen.add(item)

    required_set = set(required)
    while len(selected) > total:
        for idx in range(len(selected) - 1, -1, -1):
            if selected[idx] not in required_set:
                seen.remove(selected.pop(idx))
                break
        else:
            selected.pop()

    if len(selected) < total:
        for items in groups.values():
            for item in items:
                if item not in seen:
                    selected.append(item)
                    seen.add(item)
                    if len(selected) == total:
                        break
            if len(selected) == total:
                break

    return sorted(selected)


ALL_SYMPTOMS = build_limited_options(SYMPTOM_GROUPS, SYMPTOM_GROUP_LIMITS, HIGH_VALUE_SYMPTOMS, 312)

COMORBIDITY_GROUPS = {
    "Metabolic / Endocrine": [
        "Diabetes mellitus", "Diabetes mellitus type 2", "Type 1 diabetes mellitus", "Prediabetes",
        "Thyroid disease", "Autoimmune thyroid disease", "Autoimmune thyroiditis", "Hashimoto thyroiditis",
        "Graves disease", "Hypothyroidism", "Hyperthyroidism", "Obesity", "Metabolic syndrome",
        "Vitamin B12 deficiency", "Vitamin D deficiency", "Folate deficiency", "Electrolyte disorders",
        "Hyponatraemia history", "Chronic kidney disease", "Liver disease", "Chronic liver disease",
        "Hyperlipidemia", "Pernicious anemia",
    ],
    "Autoimmune / Inflammatory": [
        "Rheumatoid arthritis", "Systemic lupus erythematosus", "Sjogren syndrome", "Sjögren syndrome",
        "Vasculitis", "Sarcoidosis", "Celiac disease", "Psoriasis", "Inflammatory bowel disease",
        "Ulcerative colitis", "Crohn disease", "Behcet disease", "Mixed connective tissue disease",
        "Antiphospholipid syndrome", "Myasthenia gravis", "Myasthenia gravis history",
        "Neuromyelitis optica", "Multiple sclerosis", "CIDP", "Previous CIDP",
        "Autoimmune encephalitis history", "Prior autoimmune encephalitis", "Autoimmune hepatitis",
        "Primary biliary cholangitis", "Vitiligo",
    ],
    "Infectious": [
        "HIV", "HIV infection", "Hepatitis B", "Hepatitis C", "Tuberculosis", "Tuberculosis history",
        "Lyme disease", "Syphilis", "COVID-19 history", "Post-viral syndrome", "EBV infection",
        "CMV infection", "Recent gastrointestinal infection", "Recent respiratory infection",
        "Recent viral infection", "Recent vaccination", "Postpartum state",
    ],
    "Malignancy / Paraneoplastic": [
        "Lung cancer", "Small cell lung cancer", "Breast cancer", "Ovarian cancer", "Ovarian teratoma",
        "Thymoma", "Lymphoma", "Hodgkin lymphoma", "Non-Hodgkin lymphoma", "Leukemia",
        "Multiple myeloma", "Prior chemotherapy", "Paraneoplastic syndrome", "Melanoma",
        "Testicular tumor", "Colon cancer", "Renal cancer", "Thyroid cancer",
        "Unexplained lymphadenopathy", "Monoclonal gammopathy", "MGUS", "Waldenström macroglobulinemia",
    ],
    "Genetic / Family History": [
        "Family history of ALS", "Family history of neuropathy", "Family history of muscular dystrophy",
        "Family history of ataxia", "Family history of epilepsy", "Known genetic mutation",
        "SOD1 mutation", "Consanguinity", "Family history of neurological disease",
    ],
    "Toxic / Environmental": [
        "Alcohol use disorder", "Smoking", "Heavy metal exposure", "Lead exposure", "Mercury exposure",
        "Arsenic exposure", "Industrial chemical exposure", "Pesticide exposure", "Drug-induced neuropathy",
        "Chemotherapy exposure", "Recreational drug use", "Occupational toxin exposure",
    ],
    "Cardiovascular / Stroke Risk": [
        "Hypertension", "Atrial fibrillation", "Coronary artery disease", "Heart failure",
        "Stroke history", "TIA history", "Hypercoagulable state",
    ],
    "Respiratory / Sleep": [
        "Asthma", "COPD", "Obstructive sleep apnea", "Chronic respiratory failure",
        "Prior ventilator requirement",
    ],
    "Neurological History": [
        "Prior neuropathy", "Prior myelitis", "Prior optic neuritis", "Migraine", "Epilepsy",
        "Epilepsy history", "Parkinson disease", "Dementia", "Previous Guillain-Barre syndrome",
        "Degenerative spine disease",
    ],
    "Surgical / Transplant / Immunosuppression": [
        "Organ transplant", "Long-term steroid use", "Immunosuppressive therapy", "Recent surgery",
        "Bariatric surgery", "Immunosuppression",
    ],
    "Medication Associations": [
        "Statin use", "Steroid use", "Immunotherapy drugs", "Checkpoint inhibitor therapy",
        "Antipsychotic use", "Antiepileptic drugs", "Anti-TB medications", "Chemotherapy agents",
    ],
}

COMORBIDITY_GROUP_LIMITS = {
    "Metabolic / Endocrine": 20,
    "Autoimmune / Inflammatory": 20,
    "Infectious": 12,
    "Malignancy / Paraneoplastic": 15,
    "Genetic / Family History": 7,
    "Toxic / Environmental": 10,
    "Cardiovascular / Stroke Risk": 6,
    "Respiratory / Sleep": 4,
    "Neurological History": 6,
    "Surgical / Transplant / Immunosuppression": 3,
    "Medication Associations": 2,
}

HIGH_VALUE_COMORBIDITIES = [
    "Diabetes mellitus", "Vitamin B12 deficiency", "Thyroid disease", "Systemic lupus erythematosus",
    "Sjogren syndrome", "Sjögren syndrome", "Vasculitis", "Sarcoidosis", "HIV", "Hepatitis B",
    "Hepatitis C", "Tuberculosis", "Lung cancer", "Breast cancer", "Ovarian cancer", "Thymoma",
    "Lymphoma", "Multiple myeloma", "Family history of ALS", "Known genetic mutation",
    "Heavy metal exposure", "Lead exposure", "Mercury exposure", "Arsenic exposure",
    "Chemotherapy exposure", "Hypertension", "Stroke history", "Obstructive sleep apnea",
    "Previous Guillain-Barre syndrome", "Prior optic neuritis", "Checkpoint inhibitor therapy",
]

ALL_COMORBIDITIES = build_limited_options(
    COMORBIDITY_GROUPS,
    COMORBIDITY_GROUP_LIMITS,
    HIGH_VALUE_COMORBIDITIES,
    105,
)

ANTIBODIES = [
    ("NF140", "Neurofascin-140", "Most common India 49.5% | Rituximab first-line for nodopathy"),
    ("NF155", "Neurofascin-155", "Less common India 12.2% | IVIg resistant"),
    ("NF186", "Neurofascin-186", "2nd most common India 32.7% | Rituximab first-line"),
    ("SULFATIDE", "Sulfatide IgM", "Sensory neuropathy | Often IgM / DADS phenotype"),
    ("GAD65", "Glutamic Acid Decarboxylase", "SPS / Ataxia / Epilepsy spectrum"),
    ("CASPR1", "Contactin-Associated Prot-1", "Paranodal CIDP | Severe, often IVIg refractory"),
    ("CASPR2", "CASPR-2", "Encephalitis / Morvan | CT chest needed"),
    ("CONTACTIN1", "Contactin-1", "Paranodal CIDP | CASPR1 partner"),
    ("VGKC", "Voltage-Gated K+ Channel", "Subtype to LGI1 / CASPR2"),
    ("MAG", "Myelin-Assoc Glycoprotein", "DADS | Distal sensory; elderly males"),
    ("LGI1", "LGI-1 Antibody", "Limbic encephalitis | FBDS / hyponatraemia"),
]

INITIAL_TESTS = [
    "NCS (Nerve Conduction Study)",
    "EMG (Electromyography)",
    "CSF Analysis",
    "MRI Brain",
    "MRI Spine",
    "MRI Orbit / Optic Nerve",
    "EEG",
    "CT Chest",
    "CT Abdomen/Pelvis",
    "Pelvic Ultrasound / MRI",
    "PET-CT",
    "Nerve Biopsy",
    "Skin Biopsy (IENFD)",
    "Blood Work",
    "Autonomic Tests",
]

NCS_OUTCOMES = ["Normal", "Demyelinating", "Axonal", "Mixed", "Conduction block", "Temporal dispersion", "Sensory > motor involvement"]
EMG_OUTCOMES = ["Normal", "Myopathic", "Neurogenic", "NMJ defect", "Neuromyotonia / myokymia", "Fibrillation potentials"]
CSF_OUTCOMES = ["Normal", "Elevated protein", "Pleocytosis", "Oligoclonal bands", "High IgG index", "Xanthochromia"]
MRI_B_OUTCOMES = [
    "Normal",
    "Hippocampal T2/FLAIR signal",
    "Temporal lobe FLAIR changes",
    "Basal ganglia T1 hyperintensity",
    "Cortical / subcortical FLAIR (ADEM pattern)",
    "Cerebellar atrophy",
    "White matter lesions (MS-like)",
    "Area postrema lesion",
]
MRI_S_OUTCOMES = ["Normal", "Longitudinally extensive lesion (LETM)", "Short segment lesion", "Conus lesion", "Gadolinium enhancement"]
GENERIC_OUTCOMES = ["Normal", "Abnormal", "Borderline", "Pending"]


def get_test_outcomes(test_name):
    t = test_name.upper()
    if "NCS" in t:
        return NCS_OUTCOMES
    if "EMG" in t:
        return EMG_OUTCOMES
    if "CSF" in t:
        return CSF_OUTCOMES
    if "MRI BRAIN" in t:
        return MRI_B_OUTCOMES
    if "MRI SPINE" in t:
        return MRI_S_OUTCOMES
    return GENERIC_OUTCOMES


# ─────────────────────────────────────────────
#  CONDITION KNOWLEDGE BASE
# ─────────────────────────────────────────────
CONDITIONS = [
    {
        "name": "Autoimmune Nodopathy (NF140/NF186/NF155/CASPR1/CNTN1-IgG4)",
        "icd": "G61.89",
        "urgency": "URGENT",
        "ivg": "Often IVIg RESISTANT",
        "desc": "IgG4 antibodies against nodal/paranodal proteins. Can resemble CIDP but often has tremor, ataxia, severe weakness, high CSF protein, and IVIg refractoriness.",
        "rx": "Rituximab is favored for IgG4 nodopathy; consider PLEX for acute severe disease.",
        "ab_keys": ["NF140", "NF186", "NF155", "CASPR1", "CONTACTIN1"],
        "sym_weights": {
            "Reduced reflexes": 10,
            "Areflexia": 10,
            "Symmetrical weakness": 7,
            "Proximal weakness": 7,
            "Distal weakness": 7,
            "Difficulty walking": 6,
            "Gait instability": 5,
            "Tremor": 5,
            "Sensory ataxia": 7,
            "Numbness": 5,
            "Neuropathic pain": 4,
            "Progressive symptoms": 5,
            "Foot drop": 5,
        },
        "comorbidity_weights": {"Monoclonal gammopathy": 3, "MGUS": 3, "Autoimmune thyroid disease": 2},
        "test_weights": {"Demyelinating": 10, "Conduction block": 8, "Temporal dispersion": 8, "Elevated protein": 6},
        "base_prob": 6,
        "treatments": [("Rituximab", "High efficacy", "IgG4 nodopathy"), ("Plasma Exchange", "Moderate", "Acute severe phase"), ("Physiotherapy", "Essential", "Disability prevention")],
        "investigations": ["NF140/NF155/NF186/CASPR1/CNTN1 IgG subclass", "NCS demyelinating criteria", "CSF protein and cell count", "MRI roots/plexus if available"],
        "prognosis": "Earlier antibody-directed treatment improves recovery and may prevent axonal loss.",
        "red_flags": [("🚨", "red", "Nodal/paranodal antibody positivity with CIDP phenotype suggests IVIg resistance; discuss rituximab early.")],
        "mimics": ["Classic CIDP", "GBS", "POEMS", "Hereditary neuropathy"],
        "risk_untreated": [("amber", "Progressive", "Worsening gait and sensory ataxia"), ("red", "Advanced", "Severe disability and axonal loss")],
    },
    {
        "name": "Chronic Inflammatory Demyelinating Polyradiculoneuropathy",
        "icd": "G61.81",
        "urgency": "URGENT",
        "ivg": "IVIg responsive",
        "desc": "Immune demyelinating neuropathy with proximal and distal weakness, sensory loss, reduced reflexes, and supportive NCS/CSF findings.",
        "rx": "IVIg, corticosteroids, or plasma exchange depending on phenotype and contraindications.",
        "ab_keys": ["NF140"],
        "sym_weights": {
            "Symmetrical weakness": 8,
            "Proximal weakness": 7,
            "Distal weakness": 7,
            "Reduced reflexes": 10,
            "Areflexia": 10,
            "Numbness": 5,
            "Tingling": 4,
            "Difficulty walking": 6,
            "Progressive symptoms": 6,
            "Foot drop": 4,
            "Grip weakness": 4,
        },
        "comorbidity_weights": {"Diabetes mellitus type 2": -2, "Vitamin B12 deficiency": -2, "MGUS": 2},
        "test_weights": {"Demyelinating": 12, "Conduction block": 8, "Temporal dispersion": 8, "Elevated protein": 6},
        "base_prob": 10,
        "treatments": [("IVIg", "High", "First-line"), ("Corticosteroids", "Moderate-High", "Maintenance/relapsing disease"), ("Plasma Exchange", "Moderate", "Refractory or severe")],
        "investigations": ["NCS demyelinating criteria", "CSF protein", "Exclude mimics: diabetes, B12, paraprotein", "Nodal/paranodal antibody panel if IVIg-refractory"],
        "prognosis": "Many patients improve with first-line immunotherapy; refractory cases need mimic and antibody review.",
        "red_flags": [],
        "mimics": ["Diabetic neuropathy", "B12 deficiency", "POEMS", "Amyloidosis"],
        "risk_untreated": [("amber", "Moderate", "Gait impairment"), ("red", "Chronic", "Fixed disability")],
    },
    {
        "name": "Anti-CASPR2 Autoimmune Encephalitis / Morvan Syndrome",
        "icd": "G04.81",
        "urgency": "URGENT",
        "ivg": "IVIg responsive",
        "desc": "CASPR2 disease can present with limbic encephalitis, neuromyotonia, dysautonomia, neuropathic pain, severe insomnia, and thymoma association.",
        "rx": "Steroids + IVIg or PLEX; screen for thymoma; rituximab for refractory disease.",
        "ab_keys": ["CASPR2", "VGKC"],
        "sym_weights": {
            "Neuromyotonia": 12,
            "Myokymia": 10,
            "Fasciculations": 6,
            "Severe insomnia": 10,
            "Sleep disorder": 7,
            "Autonomic instability": 9,
            "Excessive sweating": 6,
            "Neuropathic pain": 6,
            "Memory loss": 5,
            "Confusion": 5,
            "Behavioral change": 5,
            "Seizures": 5,
            "Weight loss": 4,
        },
        "comorbidity_weights": {"Thymoma": 12, "Lung cancer": 4, "Recent viral infection": 1},
        "test_weights": {"Neuromyotonia / myokymia": 12, "Hippocampal T2/FLAIR signal": 8, "Temporal lobe FLAIR changes": 8, "Abnormal": 3},
        "base_prob": 4,
        "treatments": [("Corticosteroids", "High", "Acute encephalitis"), ("IVIg", "Moderate-High", "Induction"), ("Rituximab", "High", "Refractory/Morvan")],
        "investigations": ["CASPR2 serum and CSF", "CT chest for thymoma", "EMG for neuromyotonia", "MRI brain and EEG"],
        "prognosis": "Often immunotherapy responsive, but Morvan syndrome and tumor-associated disease need urgent recognition.",
        "red_flags": [("🚨", "red", "Neuromyotonia + insomnia + dysautonomia strongly supports CASPR2/Morvan workup and CT chest.")],
        "mimics": ["Viral encephalitis", "Prion disease", "Toxic-metabolic encephalopathy"],
        "risk_untreated": [("red", "Early", "Seizures and autonomic instability"), ("red", "Advanced", "Severe encephalopathy")],
    },
    {
        "name": "LGI1 Limbic Encephalitis",
        "icd": "G04.81",
        "urgency": "URGENT",
        "ivg": "IVIg responsive",
        "desc": "Limbic encephalitis with faciobrachial dystonic seizures, memory loss, behavioral change, and hyponatraemia.",
        "rx": "High-dose corticosteroids early; add IVIg/PLEX when severe; seizure drugs alone are insufficient.",
        "ab_keys": ["LGI1", "VGKC"],
        "sym_weights": {
            "Faciobrachial dystonic seizures": 16,
            "Focal seizures": 8,
            "Seizures": 7,
            "Memory loss": 9,
            "Short-term memory impairment": 9,
            "Behavioral change": 6,
            "Confusion": 6,
            "Hyponatraemia": 12,
            "Sleep disorder": 3,
            "Rapidly progressive dementia": 4,
        },
        "comorbidity_weights": {"Autoimmune thyroid disease": 2, "Thymoma": 1},
        "test_weights": {"Hippocampal T2/FLAIR signal": 10, "Temporal lobe FLAIR changes": 10, "Pleocytosis": 3, "High IgG index": 2},
        "base_prob": 4,
        "treatments": [("Corticosteroids", "High", "First-line"), ("IVIg", "Moderate-High", "Adjunct"), ("Anti-seizure medication", "Partial", "Symptom control")],
        "investigations": ["LGI1 antibody serum/CSF", "Serum sodium", "MRI brain temporal lobes", "EEG"],
        "prognosis": "Early immunotherapy reduces risk of persistent amnesia and seizures.",
        "red_flags": [("🚨", "red", "FBDS should trigger urgent LGI1 testing and early immunotherapy discussion.")],
        "mimics": ["HSV encephalitis", "Psychiatric disorder", "Prion disease"],
        "risk_untreated": [("red", "Acute", "Frequent seizures"), ("red", "Chronic", "Persistent memory impairment")],
    },
    {
        "name": "NMDAR Autoimmune Encephalitis",
        "icd": "G04.81",
        "urgency": "URGENT",
        "ivg": "IVIg responsive",
        "desc": "Psychiatric onset, seizures, orofacial dyskinesia, autonomic instability, catatonia, and reduced consciousness; ovarian teratoma screen in females.",
        "rx": "Steroids + IVIg/PLEX first-line; remove tumor when present; rituximab/cyclophosphamide for refractory disease.",
        "ab_keys": [],
        "sym_weights": {
            "Psychiatric symptoms": 10,
            "Behavioral change": 8,
            "Hallucinations": 7,
            "Seizures": 8,
            "Status epilepticus": 9,
            "Orofacial dyskinesia": 12,
            "Movement disorder (orofacial dyskinesias)": 12,
            "Catatonia": 10,
            "Autonomic instability": 8,
            "Decreased consciousness": 9,
            "Language dysfunction": 5,
            "Fever with encephalopathy": 4,
        },
        "comorbidity_weights": {"Ovarian teratoma": 14, "Recent viral infection": 2, "Postpartum state": 2},
        "test_weights": {"Pleocytosis": 6, "High IgG index": 4, "Abnormal": 3, "Normal": 0},
        "base_prob": 4,
        "treatments": [("Steroids + IVIg", "High", "First-line"), ("Plasma Exchange", "High", "Severe disease"), ("Rituximab", "High", "Second-line")],
        "investigations": ["NMDAR antibody in CSF", "MRI brain and EEG", "Pelvic imaging in females", "Tumor screening by age/sex"],
        "prognosis": "Most patients improve with prompt treatment and tumor removal when relevant.",
        "red_flags": [("🚨", "red", "Psychosis plus seizures/movement disorder is autoimmune encephalitis until proven otherwise.")],
        "mimics": ["Primary psychiatric disorder", "HSV encephalitis", "Neuroleptic malignant syndrome"],
        "risk_untreated": [("red", "Acute", "ICU-level autonomic instability"), ("red", "Advanced", "Coma or death")],
    },
    {
        "name": "NMOSD / MOG Antibody Disease",
        "icd": "G36.0",
        "urgency": "URGENT",
        "ivg": "Subtype dependent",
        "desc": "Opticospinal inflammatory disease with optic neuritis, LETM, area postrema syndrome, brainstem symptoms, and relapsing attacks.",
        "rx": "Acute IV methylprednisolone; PLEX if severe; relapse prevention differs for AQP4+ NMOSD and MOGAD.",
        "ab_keys": [],
        "sym_weights": {
            "Visual loss (optic neuritis)": 11,
            "Bilateral optic neuritis": 12,
            "Eye pain": 5,
            "Painful eye movements": 6,
            "Transverse myelitis": 11,
            "Longitudinally extensive myelitis": 14,
            "Spinal cord symptoms": 8,
            "Paraparesis": 6,
            "Sensory level": 6,
            "Urinary retention": 6,
            "Hiccups and nausea (area postrema)": 11,
            "Intractable vomiting": 9,
            "Area postrema lesion": 10,
        },
        "comorbidity_weights": {"Sjögren syndrome": 6, "Systemic lupus erythematosus": 5, "Autoimmune thyroid disease": 2},
        "test_weights": {"Longitudinally extensive lesion (LETM)": 14, "Area postrema lesion": 12, "Gadolinium enhancement": 5, "Oligoclonal bands": -2},
        "base_prob": 5,
        "treatments": [("IV methylprednisolone", "High", "Acute attack"), ("PLEX", "High", "Severe attack"), ("Rituximab/approved biologic", "High", "AQP4 relapse prevention")],
        "investigations": ["AQP4-IgG cell-based assay", "MOG-IgG serum", "MRI brain/spine/orbits", "CSF to distinguish MS/NMOSD"],
        "prognosis": "Relapse prevention is crucial because disability accumulates with attacks.",
        "red_flags": [("⚠️", "amber", "Optic neuritis plus myelitis/area postrema symptoms needs urgent AQP4 and MOG testing.")],
        "mimics": ["MS", "Sarcoidosis", "SLE/Sjögren myelitis", "B12 deficiency"],
        "risk_untreated": [("red", "Relapse", "Cumulative visual/spinal disability"), ("red", "Chronic", "Blindness or paraplegia")],
    },
    {
        "name": "Anti-GAD65 Stiff-Person Spectrum / Cerebellar Ataxia",
        "icd": "G25.82",
        "urgency": "Soon",
        "ivg": "IVIg responsive",
        "desc": "GAD65 spectrum includes stiff-person syndrome, cerebellar ataxia, epilepsy, and limbic encephalitis; often coexists with systemic autoimmunity.",
        "rx": "Diazepam/baclofen for spasms; IVIg for disabling SPS; consider rituximab in selected refractory cases.",
        "ab_keys": ["GAD65"],
        "sym_weights": {
            "Stiffness and rigidity": 12,
            "Axial rigidity": 12,
            "Limb stiffness": 8,
            "Painful spasms": 10,
            "Stimulus-triggered spasms": 10,
            "Back spasms": 8,
            "Abdominal wall spasms": 8,
            "Startle-induced spasms": 9,
            "Loss of balance": 7,
            "Coordination issues": 8,
            "Gait instability": 5,
            "Seizures": 4,
        },
        "comorbidity_weights": {
            "Type 1 diabetes mellitus": 10,
            "Autoimmune thyroid disease": 6,
            "Hashimoto thyroiditis": 5,
            "Pernicious anemia": 5,
            "Vitiligo": 4,
            "Celiac disease": 3,
            "Breast cancer": 3,
        },
        "test_weights": {"Cerebellar atrophy": 8, "Pleocytosis": 2, "High IgG index": 2},
        "base_prob": 4,
        "treatments": [("Baclofen + diazepam", "Symptomatic", "Spasm control"), ("IVIg", "Moderate", "Disabling SPS"), ("Rituximab", "Selected", "Refractory disease")],
        "investigations": ["GAD65 high-titre testing", "Screen T1DM and thyroid disease", "MRI brain for cerebellar atrophy", "Paraneoplastic screen when atypical"],
        "prognosis": "Stiffness can respond symptomatically; ataxia may be less reversible if longstanding.",
        "red_flags": [],
        "mimics": ["Functional neurological disorder", "Progressive MS", "Hereditary ataxia"],
        "risk_untreated": [("amber", "Progressive", "Falls and spasm-related disability"), ("red", "Chronic", "Fixed gait impairment")],
    },
    {
        "name": "Sulfatide / MAG IgM Sensory Neuropathy",
        "icd": "G62.81",
        "urgency": "Monitor",
        "ivg": "Often IVIg RESISTANT",
        "desc": "IgM-mediated sensory-predominant neuropathy, often with DADS phenotype, sensory ataxia, pain, and paraprotein association.",
        "rx": "Rituximab for selected IgM/paraprotein cases; symptomatic neuropathic pain treatment.",
        "ab_keys": ["SULFATIDE", "MAG"],
        "sym_weights": {
            "Numbness": 7,
            "Tingling": 6,
            "Burning sensation": 8,
            "Neuropathic pain": 8,
            "Sensory ataxia": 9,
            "Loss of vibration sense": 8,
            "Loss of proprioception": 8,
            "Positive Romberg sign": 7,
            "Large-fiber sensory loss": 6,
            "Loss of balance": 5,
        },
        "comorbidity_weights": {"MGUS": 10, "Monoclonal gammopathy": 10, "Waldenström macroglobulinemia": 12, "Multiple myeloma": 6},
        "test_weights": {"Sensory > motor involvement": 10, "Axonal": 4, "Mixed": 5},
        "base_prob": 4,
        "treatments": [("Rituximab", "Moderate", "IgM B-cell mediated"), ("Pregabalin/gabapentin", "Symptomatic", "Pain control"), ("Hematology review", "Essential", "M-band or malignancy")],
        "investigations": ["Sulfatide IgM and MAG IgM", "NCS sensory pattern", "SPEP/IFE/free light chains", "Hematology workup if M-band"],
        "prognosis": "Usually slow progression; sensory ataxia may persist even after IgM reduction.",
        "red_flags": [],
        "mimics": ["Diabetic neuropathy", "B12 deficiency", "Amyloid neuropathy"],
        "risk_untreated": [("amber", "Moderate", "Falls and chronic pain"), ("orange", "Advanced", "Disabling sensory ataxia")],
    },
    {
        "name": "Myasthenia Gravis (AChR / MuSK / LRP4)",
        "icd": "G70.01",
        "urgency": "Soon",
        "ivg": "IVIg responsive",
        "desc": "Neuromuscular junction autoimmune disease with fatigable ocular, bulbar, limb, or respiratory weakness.",
        "rx": "Pyridostigmine, corticosteroid or steroid-sparing therapy; IVIg/PLEX for crisis; thymectomy for selected AChR+ patients.",
        "ab_keys": [],
        "sym_weights": {
            "Ptosis": 10,
            "Diplopia": 8,
            "Fatigable weakness": 12,
            "Diurnal fluctuation": 10,
            "Dysphagia": 8,
            "Dysarthria": 7,
            "Bulbar weakness": 9,
            "Facial weakness": 6,
            "Nasal speech": 5,
            "Jaw fatigue": 7,
            "Neck flexor weakness": 6,
            "Respiratory weakness": 11,
            "Weakness worsens with exertion": 9,
            "Proximal weakness improves with exercise": -4,
        },
        "comorbidity_weights": {"Thymoma": 12, "Autoimmune thyroid disease": 4, "Graves disease": 4, "Rheumatoid arthritis": 1},
        "test_weights": {"NMJ defect": 12, "Abnormal": 3},
        "base_prob": 4,
        "treatments": [("Pyridostigmine", "Symptomatic", "Mild/moderate MG"), ("IVIg/PLEX", "High", "Crisis"), ("Thymectomy", "Selected", "AChR+ or thymoma")],
        "investigations": ["AChR/MuSK/LRP4 antibodies", "Repetitive nerve stimulation", "Single-fibre EMG", "CT chest for thymoma"],
        "prognosis": "Most patients achieve good control; respiratory or bulbar symptoms need urgent escalation.",
        "red_flags": [("🚨", "red", "Bulbar or respiratory weakness can indicate myasthenic crisis risk.")],
        "mimics": ["Lambert-Eaton syndrome", "Botulism", "Brainstem lesion"],
        "risk_untreated": [("amber", "Moderate", "Bulbar aspiration risk"), ("red", "Crisis", "Respiratory failure")],
    },
    {
        "name": "Multi-antibody Overlap Neuroimmunological Syndrome",
        "icd": "G61.89",
        "urgency": "URGENT",
        "ivg": "Variable",
        "desc": "Mixed CNS/PNS phenotype with overlapping antibody positivity or systemic autoimmunity.",
        "rx": "Treat dominant syndrome; often requires combined induction and maintenance immunotherapy.",
        "ab_keys": ["NF140", "NF186", "CASPR2", "GAD65", "LGI1", "VGKC"],
        "sym_weights": {
            "Symmetrical weakness": 5,
            "Numbness": 4,
            "Behavioral change": 5,
            "Autonomic instability": 5,
            "Seizures": 4,
            "Stiffness and rigidity": 4,
            "Visual loss (optic neuritis)": 4,
        },
        "comorbidity_weights": {"Systemic lupus erythematosus": 4, "Sjögren syndrome": 4, "Recent viral infection": 2, "Postpartum state": 2},
        "test_weights": {"Abnormal": 3, "Pleocytosis": 3, "Demyelinating": 3},
        "base_prob": 5,
        "treatments": [("Rituximab", "Selected", "B-cell mediated overlap"), ("IVIg", "Variable", "Acute management"), ("Azathioprine/MMF", "Moderate", "Maintenance")],
        "investigations": ["Comprehensive antibody panel", "NCS/EMG", "MRI brain/spine", "CSF and malignancy screen as indicated"],
        "prognosis": "Depends on dominant syndrome and speed of treatment.",
        "red_flags": [],
        "mimics": ["Paraneoplastic syndrome", "Vasculitis", "Infection"],
        "risk_untreated": [("amber", "Early", "Progressive deficits"), ("red", "Advanced", "Severe disability")],
    },
]

ANTIBODY_RECOMMENDER = [
    ("Anti-CASPR2 (Contactin-Associated Protein-like 2)", "CASPR2 encephalitis, Morvan syndrome", "STAT", 41, "Steroids + IVIg/PLEX; CT chest; rituximab if refractory."),
    ("VGKC Complex Panel (LGI1 + CASPR2 subtyping)", "LGI1/CASPR2 encephalitis", "STAT", 40, "Subtype-guided treatment; avoid relying on VGKC alone."),
    ("Anti-Neurofascin-140 (Nodal)", "CIDP, autoimmune nodopathy", "Urgent", 39, "If nodopathy confirmed, rituximab often favored."),
    ("Anti-Neurofascin-186 (Nodal)", "Nodal neuropathy", "Urgent", 36, "Rituximab-first discussion if positive."),
    ("Anti-Neurofascin-155 (Paranodal)", "Autoimmune nodopathy", "Urgent", 34, "Often IVIg-refractory."),
    ("Anti-CASPR1 / Anti-Contactin-1", "Paranodal nodopathy", "Urgent", 32, "Rituximab; PLEX in severe disease."),
    ("Anti-GAD65", "SPS, ataxia, epilepsy", "Urgent", 28, "Baclofen/diazepam; IVIg for SPS; screen autoimmune disease."),
    ("Anti-LGI1", "Limbic encephalitis, FBDS", "Urgent", 27, "High-dose steroids early; add IVIg/PLEX."),
    ("Anti-Sulfatide IgM", "IgM sensory neuropathy", "Routine", 25, "Evaluate paraprotein; rituximab in selected cases."),
    ("Anti-MAG IgM", "DADS / MAG-IgM neuropathy", "Routine", 24, "Hematology link; rituximab in selected cases."),
]


# ─────────────────────────────────────────────
#  SCORING ENGINE
# ─────────────────────────────────────────────
def flatten_test_outcomes(tests):
    return [outcome for _, outcome in tests]


SYMPTOM_SCORING_ALIASES = {
    "Progressive weakness": ["Progressive symptoms"],
    "Progressive quadriparesis": ["Quadriparesis", "Progressive symptoms", "Rapidly progressive weakness"],
    "Rapid progression": ["Rapidly progressive weakness", "Progressive symptoms"],
    "Absent reflexes": ["Areflexia", "Reduced reflexes"],
    "Hyporeflexia": ["Reduced reflexes"],
    "Brisk reflexes": ["Hyperreflexia"],
    "Loss of grip strength": ["Grip weakness", "Hand weakness"],
    "Dropping objects": ["Grip weakness", "Hand weakness"],
    "Difficulty buttoning shirts": ["Hand weakness", "Poor finger dexterity"],
    "Difficulty standing from squatting": ["Difficulty rising from chair", "Proximal weakness"],
    "Tongue fasciculations": ["Fasciculations", "Bulbar weakness"],
    "Tongue atrophy": ["Tongue weakness", "Bulbar weakness"],
    "Split hand syndrome": ["Hand weakness", "Muscle wasting"],
    "Head drop": ["Neck weakness", "Neck extensor weakness"],
    "Respiratory muscle weakness": ["Respiratory weakness"],
    "Electric shock sensations": ["Electric shock sensation"],
    "Glove and stocking sensory loss": ["Distal sensory loss", "Numbness"],
    "Positive Romberg": ["Positive Romberg sign", "Sensory ataxia", "Loss of balance"],
    "Band-like sensory level": ["Sensory level", "Band-like trunk sensation"],
    "Pain on eye movement": ["Painful eye movements", "Eye pain"],
    "Optic neuritis": ["Visual loss (optic neuritis)", "Visual loss"],
    "Double vision": ["Diplopia"],
    "Symptoms worsen in evening": ["Diurnal fluctuation", "Fatigable weakness"],
    "Symptoms worse in evening": ["Diurnal fluctuation", "Fatigable weakness"],
    "Diurnal fluctuation of weakness": ["Diurnal fluctuation", "Fatigable weakness"],
    "Choking while eating": ["Dysphagia", "Bulbar weakness"],
    "Pooling of saliva": ["Bulbar weakness"],
    "Orthostatic dizziness": ["Orthostatic hypotension", "Autonomic instability"],
    "Postural dizziness": ["Orthostatic hypotension", "Autonomic instability"],
    "Weak cough": ["Respiratory weakness", "Bulbar weakness"],
    "Orthopnea": ["Respiratory weakness"],
    "Reduced spirometry values": ["Respiratory weakness"],
    "REM sleep behavior disorder": ["Sleep disorder"],
    "Orofacial dyskinesias": ["Orofacial dyskinesia", "Movement disorder (orofacial dyskinesias)"],
    "New onset seizures": ["Seizures"],
    "Encephalopathy": ["Decreased consciousness", "Confusion"],
    "Area postrema symptoms": ["Hiccups and nausea (area postrema)"],
    "Urinary retention": ["Bladder dysfunction"],
}

COMORBIDITY_SCORING_ALIASES = {
    "Sjogren syndrome": ["Sjögren syndrome"],
    "Diabetes mellitus": ["Diabetes mellitus type 2"],
    "Thyroid disease": ["Autoimmune thyroid disease"],
    "Autoimmune thyroiditis": ["Autoimmune thyroid disease", "Hashimoto thyroiditis"],
    "Tuberculosis": ["Tuberculosis history"],
    "HIV": ["HIV infection"],
    "CIDP": ["Previous CIDP"],
    "Autoimmune encephalitis history": ["Prior autoimmune encephalitis"],
    "Heavy metal exposure": ["Lead exposure", "Mercury exposure", "Arsenic exposure"],
    "Prior chemotherapy": ["Chemotherapy exposure"],
}


def expand_with_aliases(values, aliases):
    expanded = set(values)
    for value in list(values):
        expanded.update(aliases.get(value, []))
    return expanded


# ─────────────────────────────────────────────
#  ADAPTIVE LEARNING LAYER
# ─────────────────────────────────────────────
def init_learning_db():
    with sqlite3.connect(LEARNING_DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS confirmed_cases (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                final_diagnosis TEXT NOT NULL,
                top_prediction TEXT,
                prediction_correct INTEGER,
                confidence INTEGER,
                confirmed_antibodies TEXT,
                treatment_response TEXT,
                notes TEXT,
                case_json TEXT NOT NULL,
                model_version_used TEXT
            )
            """
        )
        conn.commit()


def get_learning_stats():
    init_learning_db()
    with sqlite3.connect(LEARNING_DB_PATH) as conn:
        total = conn.execute("SELECT COUNT(*) FROM confirmed_cases").fetchone()[0]
        labels = conn.execute(
            "SELECT final_diagnosis, COUNT(*) FROM confirmed_cases GROUP BY final_diagnosis ORDER BY COUNT(*) DESC, final_diagnosis"
        ).fetchall()
    model = load_adaptive_model()
    return {"total": total, "labels": labels, "model": model}


def case_feature_tokens(symptoms, comorbidities, antibody_results, tests, age, gender, csf_sample):
    tokens = set()
    for sym in expand_with_aliases(symptoms, SYMPTOM_SCORING_ALIASES):
        tokens.add(f"sym::{sym}")
    for comorb in expand_with_aliases(comorbidities, COMORBIDITY_SCORING_ALIASES):
        tokens.add(f"comorb::{comorb}")
    for ab_key, result in antibody_results.items():
        tokens.add(f"ab_tested::{ab_key}")
        if result.get("positive"):
            tokens.add(f"ab_positive::{ab_key}")
            titre = float(result.get("titre", 0) or 0)
            if titre >= 500:
                tokens.add(f"ab_titre_high::{ab_key}")
            elif titre >= 100:
                tokens.add(f"ab_titre_moderate::{ab_key}")
            elif titre > 0:
                tokens.add(f"ab_titre_low::{ab_key}")
    for test_name, outcome in tests:
        tokens.add(f"test::{test_name}")
        tokens.add(f"test_outcome::{test_name}::{outcome}")
        if outcome not in ["Normal", "Pending"]:
            tokens.add(f"abnormal_test::{test_name}")

    age_bucket = "child" if age < 18 else ("young_adult" if age < 40 else ("middle_age" if age < 60 else "older_adult"))
    tokens.add(f"age::{age_bucket}")
    tokens.add(f"gender::{gender}")
    tokens.add(f"sample::{'CSF' if csf_sample else 'Serum'}")
    return sorted(tokens)


def load_confirmed_cases():
    init_learning_db()
    with sqlite3.connect(LEARNING_DB_PATH) as conn:
        rows = conn.execute(
            """
            SELECT id, created_at, final_diagnosis, top_prediction, prediction_correct,
                   confidence, confirmed_antibodies, treatment_response, notes, case_json, model_version_used
            FROM confirmed_cases
            ORDER BY id DESC
            """
        ).fetchall()
    cases = []
    for row in rows:
        cases.append(
            {
                "id": row[0],
                "created_at": row[1],
                "final_diagnosis": row[2],
                "top_prediction": row[3],
                "prediction_correct": row[4],
                "confidence": row[5],
                "confirmed_antibodies": row[6],
                "treatment_response": row[7],
                "notes": row[8],
                "case": json.loads(row[9]),
                "model_version_used": row[10],
            }
        )
    return cases


def save_confirmed_case(case_payload, final_diagnosis, prediction_correct, confidence, confirmed_antibodies, treatment_response, notes):
    init_learning_db()
    created_at = datetime.datetime.now().isoformat(timespec="seconds")
    model = load_adaptive_model()
    with sqlite3.connect(LEARNING_DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO confirmed_cases (
                created_at, final_diagnosis, top_prediction, prediction_correct, confidence,
                confirmed_antibodies, treatment_response, notes, case_json, model_version_used
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                created_at,
                final_diagnosis,
                case_payload.get("top_prediction"),
                int(prediction_correct),
                int(confidence),
                confirmed_antibodies,
                treatment_response,
                notes,
                json.dumps(case_payload, ensure_ascii=False),
                model.get("version") if model else None,
            ),
        )
        conn.commit()


def train_adaptive_model(min_cases=5):
    cases = list(reversed(load_confirmed_cases()))
    if len(cases) < min_cases:
        return None, f"Need at least {min_cases} confirmed cases; currently have {len(cases)}."

    label_counts = Counter()
    token_counts = defaultdict(Counter)
    token_totals = Counter()
    vocab = set()

    for item in cases:
        label = item["final_diagnosis"]
        tokens = item["case"].get("feature_tokens", [])
        label_counts[label] += 1
        token_counts[label].update(tokens)
        token_totals[label] += len(tokens)
        vocab.update(tokens)

    model = {
        "version": datetime.datetime.now().strftime("adaptive-%Y%m%d-%H%M%S"),
        "trained_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "n_cases": len(cases),
        "labels": sorted(label_counts),
        "label_counts": dict(label_counts),
        "token_counts": {label: dict(counts) for label, counts in token_counts.items()},
        "token_totals": dict(token_totals),
        "vocab": sorted(vocab),
        "model_type": "supervised_multinomial_naive_bayes",
    }
    metrics = evaluate_adaptive_model(model, cases)
    model["training_metrics"] = metrics
    ADAPTIVE_MODEL_PATH.write_text(json.dumps(model, indent=2, ensure_ascii=False), encoding="utf-8")
    return model, f"Trained {model['version']} on {len(cases)} confirmed cases."


def load_adaptive_model():
    if not ADAPTIVE_MODEL_PATH.exists():
        return None
    try:
        return json.loads(ADAPTIVE_MODEL_PATH.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def predict_adaptive_label_probs(model, feature_tokens):
    if not model or not model.get("labels"):
        return {}
    tokens = set(feature_tokens)
    labels = model["labels"]
    label_counts = model["label_counts"]
    token_counts = {label: Counter(counts) for label, counts in model["token_counts"].items()}
    token_totals = Counter(model["token_totals"])
    vocab_size = max(len(model.get("vocab", [])), 1)
    total_cases = sum(label_counts.values())

    log_scores = {}
    for label in labels:
        prior = (label_counts.get(label, 0) + 1) / (total_cases + len(labels))
        score = math.log(prior)
        denom = token_totals.get(label, 0) + vocab_size
        for token in tokens:
            score += math.log((token_counts[label].get(token, 0) + 1) / denom)
        log_scores[label] = score

    max_log = max(log_scores.values())
    exp_scores = {label: math.exp(score - max_log) for label, score in log_scores.items()}
    total = sum(exp_scores.values()) or 1
    return {label: value / total for label, value in exp_scores.items()}


def adaptive_condition_boosts(model, feature_tokens, condition_names):
    probs = predict_adaptive_label_probs(model, feature_tokens)
    return {name: probs.get(name, 0.0) for name in condition_names}


def evaluate_adaptive_model(model, cases):
    if not cases:
        return {"accuracy": 0.0, "correct": 0, "total": 0}
    correct = 0
    for item in cases:
        probs = predict_adaptive_label_probs(model, item["case"].get("feature_tokens", []))
        pred = max(probs, key=probs.get) if probs else None
        correct += int(pred == item["final_diagnosis"])
    return {"accuracy": round(correct / len(cases), 3), "correct": correct, "total": len(cases)}


def compute_risk_score(symptoms, comorbidities, antibody_results, n_tests, age, gender, csf_sample):
    pos_abs = [a for a, v in antibody_results.items() if v["positive"]]
    n_pos = len(pos_abs)
    n_tested = sum(1 for v in antibody_results.values() if v["tested"])
    max_titre = max((v["titre"] for v in antibody_results.values() if v["positive"]), default=0)
    sym_score = len(symptoms)
    comorb_score = len(comorbidities)

    high_signal_symptoms = {
        "Faciobrachial dystonic seizures",
        "Neuromyotonia",
        "Tongue fasciculations",
        "Split hand syndrome",
        "Progressive quadriparesis",
        "Head drop",
        "Weak cough",
        "Orthopnea",
        "Orofacial dyskinesia",
        "Orofacial dyskinesias",
        "Movement disorder (orofacial dyskinesias)",
        "Longitudinally extensive myelitis",
        "Bilateral optic neuritis",
        "Respiratory weakness",
        "Respiratory muscle weakness",
        "Autonomic instability",
        "Orthostatic dizziness",
        "Hyponatraemia",
        "Status epilepticus",
        "REM sleep behavior disorder",
        "Rapid progression",
        "Bulbar involvement",
    }
    high_signal_comorbidities = {
        "Thymoma",
        "Ovarian teratoma",
        "Type 1 diabetes mellitus",
        "Sjögren syndrome",
        "Systemic lupus erythematosus",
        "MGUS",
        "Monoclonal gammopathy",
        "Waldenström macroglobulinemia",
    }

    signal_sym_count = len(expand_with_aliases(symptoms, SYMPTOM_SCORING_ALIASES) & high_signal_symptoms)
    signal_comorb_count = len(expand_with_aliases(comorbidities, COMORBIDITY_SCORING_ALIASES) & high_signal_comorbidities)

    ml_score = 0.0
    ml_score += min(n_tested, 11) * 0.0180
    ml_score += min(n_pos, 11) * 0.0760
    ml_score += (0.0561 if "NF140" in pos_abs else 0)
    ml_score += (0.0540 if "NF186" in pos_abs else 0)
    ml_score += (0.0475 if "GAD65" in pos_abs else 0)
    ml_score += (0.0500 if any(a in pos_abs for a in ["CASPR2", "LGI1"]) else 0)
    ml_score += min(n_pos * n_tested, 50) * 0.0009
    ml_score += min(max_titre / 300, 1) * 0.0383
    ml_score += min(sym_score, 20) * 0.0095
    ml_score += min(comorb_score, 15) * 0.0065
    ml_score += signal_sym_count * 0.018
    ml_score += signal_comorb_count * 0.014
    ml_score += (0.02 if n_tests > 0 else 0)
    ml_score += (0.015 if csf_sample else 0)
    ml_score = min(ml_score, 1.0)

    clin_score = 0.0
    clin_score += min(sym_score / 20, 1.0) * 0.38
    clin_score += min(comorb_score / 15, 1.0) * 0.13
    clin_score += min(n_pos / 5, 1.0) * 0.29
    clin_score += min(max_titre / 500, 1.0) * 0.20
    clin_score += min(signal_sym_count * 0.04 + signal_comorb_count * 0.025, 0.18)
    clin_score = min(clin_score, 1.0)

    final = (ml_score * 0.78 + clin_score * 0.22) * 100
    return round(min(final, 99.5), 1), round(ml_score * 100, 1), round(clin_score * 100, 1), sym_score, comorb_score


def compute_conditions(symptoms, comorbidities, antibody_results, age, gender, tests, adaptive_scores=None):
    pos_abs = [key for key, v in antibody_results.items() if v["positive"]]
    sym_set = expand_with_aliases(symptoms, SYMPTOM_SCORING_ALIASES)
    comorb_set = expand_with_aliases(comorbidities, COMORBIDITY_SCORING_ALIASES)
    test_outcomes = flatten_test_outcomes(tests)
    adaptive_scores = adaptive_scores or {}

    ranked = []
    for cond in CONDITIONS:
        score = float(cond["base_prob"])

        for ab in cond["ab_keys"]:
            if ab in pos_abs:
                score += {
                    "NF140": 12,
                    "NF186": 11,
                    "NF155": 10,
                    "CASPR2": 12,
                    "LGI1": 12,
                    "GAD65": 10,
                    "CASPR1": 10,
                    "CONTACTIN1": 9,
                    "SULFATIDE": 8,
                    "MAG": 8,
                    "VGKC": 7,
                }.get(ab, 6)

        for sym, weight in cond.get("sym_weights", {}).items():
            if sym in sym_set:
                score += weight

        for comorb, weight in cond.get("comorbidity_weights", {}).items():
            if comorb in comorb_set:
                score += weight

        for outcome in test_outcomes:
            score += cond.get("test_weights", {}).get(outcome, 0)

        if gender == "Female" and cond["name"] in ["NMDAR Autoimmune Encephalitis", "NMOSD / MOG Antibody Disease"]:
            score += 4
        if gender == "Male" and cond["name"] in ["LGI1 Limbic Encephalitis", "Anti-CASPR2 Autoimmune Encephalitis / Morvan Syndrome"]:
            score += 3
        if age >= 50 and cond["name"] in ["LGI1 Limbic Encephalitis", "Anti-CASPR2 Autoimmune Encephalitis / Morvan Syndrome", "Sulfatide / MAG IgM Sensory Neuropathy"]:
            score += 3
        if age <= 40 and cond["name"] == "NMDAR Autoimmune Encephalitis":
            score += 3

        # Multi-antibody overlap should rise when antibody burden is genuinely mixed.
        if cond["name"].startswith("Multi-antibody") and len(pos_abs) >= 2:
            score += len(pos_abs) * 4

        adaptive_prob = adaptive_scores.get(cond["name"], 0.0)
        score += adaptive_prob * 14

        ranked.append((max(score, 0.5), cond))

    ranked.sort(key=lambda x: -x[0])
    total = sum(s for s, _ in ranked[:10]) or 1
    result = []
    for i, (score, cond) in enumerate(ranked[:5]):
        result.append({
            **cond,
            "raw_score": round(score, 1),
            "prob": round(score / total * 100, 1),
            "adaptive_prob": round(adaptive_scores.get(cond["name"], 0.0) * 100, 1),
            "rank": i + 1,
        })
    return result


def compute_shap(antibody_results, symptoms, comorbidities, tests):
    pos_abs = [k for k, v in antibody_results.items() if v["positive"]]
    n_tested = sum(1 for v in antibody_results.values() if v["tested"])
    n_pos = len(pos_abs)
    max_titre = max((v["titre"] for v in antibody_results.values() if v["positive"]), default=0)

    features = [
        ("No. Antibodies Tested", True, round(n_tested * 0.018, 4)),
        ("No. Positive Antibodies", True, round(n_pos * 0.076, 4)),
        ("Symptom Burden", True, round(min(len(symptoms), 20) * 0.0095, 4)),
        ("Co-morbidity Burden", True, round(min(len(comorbidities), 15) * 0.0065, 4)),
        ("No. of Tests Ordered", False, round(-0.0083 - len(tests) * 0.002, 4)),
        ("Peak Serum Titre", True, round(min(max_titre / 300, 1) * 0.0383, 4)),
        ("Total Antibody Burden", True, round(min(n_pos * n_tested, 50) * 0.0009, 4)),
    ]

    if "NF140" in pos_abs:
        features.append(("NF140 Positive", True, 0.0561))
    if "NF186" in pos_abs:
        features.append(("NF186 Positive", True, 0.0540))
    if "GAD65" in pos_abs:
        features.append(("GAD65 Positive", True, 0.0475))
    if any(a in pos_abs for a in ["CASPR2", "LGI1"]):
        features.append(("Limbic Encephalitis Antibody Signal", True, 0.0500))

    high_signal_symptoms = {
        "Faciobrachial dystonic seizures": 0.052,
        "Neuromyotonia": 0.046,
        "Orofacial dyskinesia": 0.044,
        "Longitudinally extensive myelitis": 0.048,
        "Bilateral optic neuritis": 0.044,
        "Respiratory weakness": 0.042,
        "Hyponatraemia": 0.040,
    }
    high_signal_comorbs = {
        "Thymoma": 0.042,
        "Ovarian teratoma": 0.044,
        "Type 1 diabetes mellitus": 0.034,
        "Sjögren syndrome": 0.030,
        "MGUS": 0.032,
        "Waldenström macroglobulinemia": 0.036,
    }

    for sym in symptoms:
        if sym in high_signal_symptoms:
            features.append((sym, True, high_signal_symptoms[sym]))
    for comorb in comorbidities:
        if comorb in high_signal_comorbs:
            features.append((comorb, True, high_signal_comorbs[comorb]))

    for _, outcome in tests:
        if outcome not in ["Normal", "Pending"]:
            features.append((f"Test finding: {outcome}", True, 0.018))

    features.sort(key=lambda x: -abs(x[2]))
    return features[:12]


# ─────────────────────────────────────────────
#  HTML HELPERS
# ─────────────────────────────────────────────
def esc(value):
    return html.escape(str(value), quote=True)


def priority_badge_html(p):
    colors = {"STAT": ("#fee2e2", "#dc2626"), "Urgent": ("#fef3c7", "#d97706"), "Routine": ("#f0fdf4", "#16a34a"), "Monitor": ("#f1f5f9", "#475569")}
    bg, fc = colors.get(p, ("#f1f5f9", "#475569"))
    return f"<span style='background:{bg};color:{fc};font-size:10px;font-weight:700;padding:2px 9px;border-radius:20px'>{esc(p)}</span>"


def gauge_svg(pct):
    arc_len = 251.2
    offset = arc_len - (pct / 100) * arc_len
    needle_angle = -180 + (pct / 100) * 180
    rad = math.radians(needle_angle)
    nx = 100 + 72 * math.cos(rad)
    ny = 100 - 72 * math.sin(rad)
    color = "#dc2626" if pct >= 70 else ("#d97706" if pct >= 40 else "#16a34a")
    return f"""
<svg width="220" height="128" viewBox="0 0 200 115">
  <defs>
    <linearGradient id="gtrack" x1="0%" y1="0%" x2="100%" y2="0%">
      <stop offset="0%" stop-color="#16a34a"/>
      <stop offset="50%" stop-color="#d97706"/>
      <stop offset="100%" stop-color="#dc2626"/>
    </linearGradient>
  </defs>
  <path d="M20,100 A80,80 0 0,1 180,100" fill="none" stroke="#e5e7eb" stroke-width="16" stroke-linecap="round"/>
  <path d="M20,100 A80,80 0 0,1 180,100" fill="none" stroke="url(#gtrack)" stroke-width="16"
        stroke-linecap="round" stroke-dasharray="{arc_len}" stroke-dashoffset="{offset:.1f}"/>
  <line x1="100" y1="100" x2="{nx:.1f}" y2="{ny:.1f}" stroke="{color}" stroke-width="3" stroke-linecap="round"/>
  <circle cx="100" cy="100" r="6" fill="{color}"/>
  <text x="16" y="112" font-size="9" fill="#64748b">0%</text>
  <text x="91" y="15" font-size="9" fill="#64748b">50%</text>
  <text x="174" y="112" font-size="9" fill="#64748b">100%</text>
</svg>"""


def chips_html(items, cls="chip"):
    if not items:
        return "<span style='color:#94a3b8;font-size:12px'>None selected</span>"
    return "".join(f"<span class='{cls}'>{esc(item)}</span>" for item in items)


def generate_html_report(age, gender, csf_sample, symptoms, comorbidities, antibody_results, initial_tests, risk_score, ml_prob, clin_prob, sym_score, comorb_score, conditions, shap_features, n_pos_abs, max_titre):
    now = datetime.datetime.now().strftime("%d %b %Y, %H:%M")
    sample = "CSF" if csf_sample else "Serum"
    risk_label = "HIGH RISK" if risk_score >= 70 else ("MODERATE RISK" if risk_score >= 40 else "LOW RISK")
    risk_color = "#dc2626" if risk_score >= 70 else ("#d97706" if risk_score >= 40 else "#16a34a")
    risk_bg = "#fee2e2" if risk_score >= 70 else ("#fef3c7" if risk_score >= 40 else "#f0fdf4")

    sym_chips = "".join(f"<span class='chip'>{esc(s)}</span>" for s in symptoms) or "<span class='muted'>None selected</span>"
    comorb_chips = "".join(f"<span class='chip green'>{esc(c)}</span>" for c in comorbidities) or "<span class='muted'>None selected</span>"
    pos_ab_chips = "".join(
        f"<span class='chip red'>{esc(k)} + {esc(v['titre'])} U/mL</span>"
        for k, v in antibody_results.items()
        if v["positive"]
    ) or "<span class='muted'>None positive</span>"
    tests_rows = "".join(f"<tr><td>{esc(t)}</td><td>{esc(o)}</td></tr>" for t, o in initial_tests) or "<tr><td colspan='2' class='muted'>None entered</td></tr>"

    max_shap = max(abs(f[2]) for f in shap_features) if shap_features else 1
    shap_rows = ""
    for name, raises, val in shap_features:
        arrow = "▲" if raises else "▼"
        fc = "#dc2626" if raises else "#2563eb"
        bar_w = int(abs(val) / max_shap * 100)
        sign = "+" if val >= 0 else ""
        shap_rows += f"""
        <tr>
          <td>{arrow} {esc(name)}</td>
          <td style='text-align:right;color:{fc};font-weight:700'>{sign}{val:.4f}</td>
          <td style='width:190px'><div class='bar'><div style='background:{fc};width:{bar_w}%'></div></div></td>
        </tr>"""

    cond_cards = ""
    for cond in conditions:
        is_top = cond["rank"] == 1
        prob_color = "#dc2626" if cond["prob"] >= 25 else ("#d97706" if cond["prob"] >= 12 else "#16a34a")
        cond_cards += f"""
        <div class='cond-card {"top" if is_top else ""}'>
          {'<div class="top-match">TOP MATCH</div>' if is_top else ''}
          <div class='cond-main'>
            <div>
              <div class='cond-title'>{esc(cond["name"])}</div>
              <div class='muted'>ICD {esc(cond["icd"])} · raw score {esc(cond.get("raw_score", "—"))}</div>
            </div>
            <div class='prob' style='color:{prob_color}'>{cond["prob"]}%<span>probability</span></div>
          </div>
          <div class='badge-line'><span>{esc(cond["urgency"])}</span><span>{esc(cond["ivg"])}</span></div>
          {'<p>' + esc(cond["desc"]) + '</p><p><strong>Rx:</strong> ' + esc(cond["rx"]) + '</p>' if is_top else ''}
        </div>"""

    ab_rows = ""
    for i, (name, diag, priority, score, rx) in enumerate(ANTIBODY_RECOMMENDER):
        ab_rows += f"""
        <tr class='{"hi" if i == 0 else ""}'>
          <td>{'★ ' if i == 0 else ''}{esc(name)}</td>
          <td>{esc(diag)}</td>
          <td>{priority_badge_html(priority)}</td>
          <td>{score}</td>
          <td>{esc(rx)}</td>
        </tr>"""

    top_cond = conditions[0] if conditions else None
    red_flags_html = ""
    if top_cond:
        for emoji, level, msg in top_cond.get("red_flags", []):
            red_flags_html += f"<div class='alert {level}'><span>{esc(emoji)}</span><div>{esc(msg)}</div></div>"
    if "Neuromyotonia" in symptoms:
        red_flags_html += "<div class='alert red'><span>🚨</span><div>Neuromyotonia should trigger EMG and CASPR2/Morvan evaluation, including CT chest.</div></div>"
    if "Visual loss (optic neuritis)" in symptoms or "Bilateral optic neuritis" in symptoms:
        red_flags_html += "<div class='alert amber'><span>⚠️</span><div>Optic neuritis pattern should trigger AQP4-IgG and MOG-IgG testing urgently.</div></div>"
    red_flags_html = red_flags_html or "<div class='muted'>No critical red flags for current inputs.</div>"

    top_name = top_cond["name"] if top_cond else "—"
    top_rx = top_cond["rx"] if top_cond else "Further antibody testing required"
    top_prog = top_cond["prognosis"] if top_cond else "N/A"
    mimics_html = "".join(f"<span class='chip yellow'>{esc(m)}</span>" for m in top_cond.get("mimics", [])) if top_cond else ""
    treatments_html = "".join(f"<div class='tx'><strong>{esc(t)}</strong><span>{esc(e)}</span><em>{esc(f)}</em></div>" for t, e, f in top_cond.get("treatments", [])) if top_cond else ""
    investigations_html = "".join(f"<div class='inv'>📌 {esc(inv)}</div>" for inv in top_cond.get("investigations", [])) if top_cond else ""
    risk_prog = "".join(f"<div class='timeline'><b>{esc(stage)}:</b> {esc(desc)}</div>" for _, stage, desc in top_cond.get("risk_untreated", [])) if top_cond else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Neuroimmunology Diagnostic Report</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>
  :root {{ --navy:#0c2340; --blue:#185FA5; --bg:#f1f5f9; --card:#fff; --text:#1f2937; --muted:#64748b; --border:#e2e8f0; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; font-family:'DM Sans', sans-serif; background:var(--bg); color:var(--text); }}
  .topbar {{ background:linear-gradient(135deg,#0c2340,#185FA5); color:white; padding:16px 32px; display:flex; justify-content:space-between; align-items:center; position:sticky; top:0; z-index:20; box-shadow:0 8px 26px rgba(2,8,23,0.22); }}
  .brand {{ font-weight:800; }} .brand span {{ display:block; font-size:11px; color:#bfdbfe; margin-top:2px; font-weight:600; }}
  button {{ background:#fff; color:#0c2340; border:0; border-radius:8px; padding:8px 14px; font-weight:800; cursor:pointer; }}
  .main {{ max-width:1260px; margin:0 auto; padding:28px 22px; }}
  .hero {{ background:linear-gradient(135deg,#0c2340,#1a3d6e); color:white; border-radius:16px; padding:26px 30px; display:flex; justify-content:space-between; gap:20px; box-shadow:0 10px 24px rgba(12,35,64,0.18); }}
  h1 {{ font-family:'DM Serif Display', serif; font-size:30px; font-weight:400; margin:0 0 8px; }}
  .meta {{ display:flex; flex-wrap:wrap; gap:14px 22px; font-size:13px; color:#bfdbfe; }} .meta strong {{ color:white; }}
  .section {{ display:flex; gap:10px; align-items:center; margin:28px 0 12px; }} .num {{ width:28px; height:28px; border-radius:999px; background:#60A5FA; color:white; display:flex; align-items:center; justify-content:center; font-weight:800; }}
  .section h2 {{ margin:0; color:var(--navy); font-size:17px; }} .section p {{ margin:1px 0 0; color:var(--muted); font-size:12px; }}
  .card {{ background:var(--card); border:1px solid var(--border); border-radius:14px; padding:20px 24px; box-shadow:0 2px 10px rgba(15,23,42,0.06); }}
  .gauge {{ display:flex; gap:36px; align-items:center; flex-wrap:wrap; }} .pct {{ font-family:'DM Serif Display', serif; font-size:58px; line-height:1; color:{risk_color}; }}
  .risk {{ display:inline-block; background:{risk_bg}; color:{risk_color}; border-radius:999px; padding:4px 13px; font-weight:800; margin:6px 0 12px; }}
  .stats {{ display:flex; flex-wrap:wrap; gap:10px; }} .stat {{ background:#f8fafc; border:1px solid var(--border); border-radius:10px; padding:10px 13px; min-width:125px; }}
  .stat b {{ display:block; color:var(--navy); font-size:22px; }} .stat span,.muted {{ color:var(--muted); font-size:12px; }}
  .grid3 {{ display:grid; grid-template-columns:repeat(3,1fr); gap:18px; }} .grid2 {{ display:grid; grid-template-columns:repeat(2,1fr); gap:22px; }}
  .chip {{ display:inline-block; background:#dbeafe; color:#1e40af; border:1px solid #bfdbfe; font-size:11px; font-weight:700; padding:4px 10px; border-radius:999px; margin:3px; }}
  .chip.green {{ background:#ecfdf5; color:#047857; border-color:#bbf7d0; }} .chip.red {{ background:#fee2e2; color:#991b1b; border-color:#fecaca; }} .chip.yellow {{ background:#fef9c3; color:#713f12; border-color:#fde68a; }}
  table {{ width:100%; border-collapse:collapse; }} th {{ background:#0c2340; color:white; text-align:left; padding:9px 12px; font-size:12px; }} td {{ border-bottom:1px solid var(--border); padding:8px 12px; font-size:12px; vertical-align:top; }}
  tr.hi {{ background:#eff6ff; font-weight:700; }} .bar {{ background:#e5e7eb; height:10px; border-radius:4px; overflow:hidden; }} .bar div {{ height:10px; }}
  .cond-card {{ background:white; border:1px solid var(--border); border-radius:14px; padding:16px; margin-bottom:12px; position:relative; overflow:hidden; }} .cond-card.top {{ border:2px solid #185FA5; background:#f0f6ff; box-shadow:0 8px 22px rgba(24,95,165,0.12); }}
  .top-match {{ position:absolute; top:0; left:0; background:#185FA5; color:white; font-size:10px; font-weight:800; padding:4px 10px; border-radius:0 0 8px 0; }}
  .cond-main {{ display:flex; justify-content:space-between; gap:16px; margin-top:10px; }} .cond-title {{ font-weight:800; color:#0c2340; }}
  .prob {{ font-size:24px; font-weight:900; text-align:right; }} .prob span {{ display:block; color:var(--muted); font-size:10px; font-weight:600; }}
  .badge-line {{ display:flex; flex-wrap:wrap; gap:6px; margin-top:10px; }} .badge-line span {{ background:#eef2ff; color:#1e3a8a; border-radius:999px; padding:3px 9px; font-size:10px; font-weight:800; }}
  .alert {{ display:flex; gap:10px; padding:11px 13px; border-radius:0 10px 10px 0; margin-bottom:9px; font-size:13px; line-height:1.5; }} .alert.red {{ background:#fee2e2; border-left:4px solid #dc2626; }} .alert.amber {{ background:#fef3c7; border-left:4px solid #d97706; }}
  .tx,.inv {{ background:#f8fafc; border:1px solid var(--border); border-radius:10px; padding:12px; margin-bottom:8px; }} .tx span,.tx em {{ display:block; color:var(--muted); font-size:12px; margin-top:3px; }}
  .timeline {{ font-size:12px; margin:6px 0; }} .disclaimer {{ color:#64748b; font-size:12px; line-height:1.6; margin:28px 0; border-top:3px solid #60A5FA; background:#fff; padding:16px; border-radius:0 0 12px 12px; }}
  @media(max-width:900px) {{ .grid3,.grid2 {{ grid-template-columns:1fr; }} .topbar,.hero {{ flex-direction:column; align-items:flex-start; }} }}
</style>
</head>
<body>
<nav class="topbar">
  <div class="brand">🧠 Neuroimmunology Unified Diagnostic System <span>ML Risk Score · SHAP · Condition Predictor · Antibody Recommender · v8.3</span></div>
  <button onclick="window.print()">Print / Save PDF</button>
</nav>
<main class="main">
  <section class="hero">
    <div><h1>Clinical Diagnostic Report</h1><div class="meta">
      <div><strong>{age} yrs</strong> Age</div><div><strong>{esc(gender)}</strong> Gender</div><div><strong>{sample}</strong> Sample</div>
      <div><strong>{n_pos_abs}</strong> Positive Antibodies</div><div><strong>{len(symptoms)}</strong> Symptoms</div><div><strong>{len(comorbidities)}</strong> Co-morbidities</div>
    </div></div>
    <div class="meta"><div><strong>Generated</strong><br>{now}</div></div>
  </section>

  <div class="section"><div class="num">1</div><div><h2>ML Antibody Positivity Risk Score</h2><p>Weighted ensemble simulation with clinical modifiers</p></div></div>
  <div class="card gauge">{gauge_svg(risk_score)}<div><div class="pct">{risk_score}%</div><div class="risk">{risk_label}</div>
    <div class="muted">ML probability: {ml_prob}% · Clinical probability: {clin_prob}%</div>
    <div class="stats"><div class="stat"><span>Symptoms</span><b>{sym_score}/20</b></div><div class="stat"><span>Co-morbidities</span><b>{comorb_score}/15</b></div><div class="stat"><span>Positive Ab</span><b>{n_pos_abs}/11</b></div><div class="stat"><span>Peak Titre</span><b>{int(max_titre)}</b></div></div>
  </div></div>

  <div class="section"><div class="num">i</div><div><h2>Patient Input Summary</h2><p>Symptoms, co-morbidities, antibodies, and initial tests</p></div></div>
  <div class="card grid3"><div><b>Symptoms</b><br>{sym_chips}</div><div><b>Co-morbidities</b><br>{comorb_chips}</div><div><b>Positive antibodies</b><br>{pos_ab_chips}</div></div>
  <div class="card" style="margin-top:12px"><b>Initial tests</b><table style="margin-top:8px"><tbody>{tests_rows}</tbody></table></div>

  <div class="section"><div class="num">2</div><div><h2>SHAP Explainability</h2><p>Feature attribution · ▲ raises risk · ▼ lowers risk</p></div></div>
  <div class="card" style="padding:0;overflow:hidden"><table><thead><tr><th>Feature</th><th style="text-align:right">Impact</th><th>Contribution</th></tr></thead><tbody>{shap_rows}</tbody></table></div>

  <div class="section"><div class="num">3</div><div><h2>Condition Predictor</h2><p>Ranked differential diagnosis</p></div></div>
  {cond_cards}

  <div class="section"><div class="num">4</div><div><h2>Antibody Testing Recommendations</h2><p>Clinical syndrome and India-calibrated nodopathy sequence</p></div></div>
  <div class="card" style="padding:0;overflow:hidden"><table><thead><tr><th>Antibody Test</th><th>Conditions Diagnosed</th><th>Priority</th><th>Score</th><th>If Positive → Treatment</th></tr></thead><tbody>{ab_rows}</tbody></table></div>

  <div class="section"><div class="num">5</div><div><h2>Clinical Insights & Red Flags</h2><p>High-signal pattern detection</p></div></div>
  <div class="card">{red_flags_html}</div>

  <div class="section"><div class="num">6</div><div><h2>Treatment · Investigations · Prognosis</h2><p>Based on top-ranked condition: {esc(top_name)}</p></div></div>
  <div class="card grid2"><div><b>First-line Treatment</b><p>{esc(top_rx)}</p><b>Mimics to Exclude</b><div>{mimics_html}</div><b>Risk if Untreated</b>{risk_prog}</div><div><b>Prognosis</b><p>{esc(top_prog)}</p><b>Treatment Options</b>{treatments_html}<b>Key Investigations</b>{investigations_html}</div></div>

  <div class="disclaimer"><strong>Clinical Decision Support Only.</strong> Results must be confirmed by a qualified neurologist with full examination, NCS/EMG, MRI, CSF when indicated, and validated antibody assays. Co-morbidities are diagnostic modifiers, not standalone diagnostic proof.</div>
</main>
</body>
</html>"""


# ─────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────
if "test_entries" not in st.session_state:
    st.session_state.test_entries = []
if "last_case" not in st.session_state:
    st.session_state.last_case = None

learning_stats = get_learning_stats()
active_adaptive_model = learning_stats["model"]


# ─────────────────────────────────────────────
#  TOP BANNER
# ─────────────────────────────────────────────
st.markdown(
    f"""
<div class="top-banner">
  <h1>🧠 Neuroimmunology Unified Diagnostic System <span style="font-size:14px;font-weight:500;opacity:0.75">v8.3</span></h1>
  <p>ML Risk Score · SHAP Explainability · Condition Predictor · Antibody Recommender · Searchable clinical input</p>
  <div class="badge-row">
    <span class="badge">Searchable symptoms: {len(ALL_SYMPTOMS)}</span>
    <span class="badge">Searchable co-morbidities: {len(ALL_COMORBIDITIES)}</span>
    <span class="badge">Max symptoms 20</span>
    <span class="badge">Max co-morbidities 15</span>
    <span class="badge">Condition-weighted scoring</span>
    <span class="badge">Confirmed cases: {learning_stats["total"]}</span>
    <span class="badge">Adaptive model: {active_adaptive_model["version"] if active_adaptive_model else "not trained"}</span>
    <span class="badge">India-calibrated NF140 → NF186 → NF155</span>
  </div>
</div>
""",
    unsafe_allow_html=True,
)


# ─────────────────────────────────────────────
#  UPLOAD SECTION
# ─────────────────────────────────────────────
st.markdown("### 📂 Upload Dataset (Optional)")
uploaded_file = st.file_uploader("Upload patient PDF (for reference / batch mode)", type=["pdf"], label_visibility="collapsed")
if uploaded_file:
    st.success(f"Uploaded: {uploaded_file.name} ({uploaded_file.size:,} bytes)")


# ─────────────────────────────────────────────
#  PATIENT DETAILS
# ─────────────────────────────────────────────
st.markdown("### 👤 Patient Details")
col_age, col_gender, col_csf = st.columns([2, 2, 1])
with col_age:
    age = st.slider("Age", min_value=1, max_value=100, value=40, format="%d yrs")
with col_gender:
    gender = st.selectbox("Gender", ["Select", "Male", "Female"], index=0)
with col_csf:
    csf_sample = st.checkbox("CSF sample\n(not serum)")


# ─────────────────────────────────────────────
#  SEARCHABLE SYMPTOMS + COMORBIDITIES
# ─────────────────────────────────────────────
st.markdown(
    f"""
<div class="sec-label">
  <div class="sec-num">1</div>
  <div>
    <div class="sec-title">Clinical Symptoms & Co-morbidities</div>
    <div class="sec-sub">Type any part of the word, e.g. "weight", "optic", "thymoma", "diabetes". Streamlit filters matching options instantly.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

sel_col, guide_col = st.columns([2.2, 1])
with sel_col:
    selected_symptoms = st.multiselect(
        "Search and select symptoms",
        options=ALL_SYMPTOMS,
        max_selections=20,
        placeholder="Start typing symptom name...",
        help="Select up to 20 observed symptoms. These directly modify risk score and condition probabilities.",
    )
    selected_comorbidities = st.multiselect(
        "Search and select co-morbidities / diagnostic modifiers",
        options=ALL_COMORBIDITIES,
        max_selections=15,
        placeholder="Start typing co-morbidity, cancer, autoimmune disease, infection...",
        help="Select up to 15 co-morbidities. These act as diagnostic modifiers for condition ranking.",
    )
    st.markdown(
        f"""
<div class="metric-strip">
  <div class="mini-metric"><div class="label">Symptoms selected</div><div class="value">{len(selected_symptoms)}/20</div></div>
  <div class="mini-metric"><div class="label">Co-morbidities selected</div><div class="value">{len(selected_comorbidities)}/15</div></div>
  <div class="mini-metric"><div class="label">Symptom options</div><div class="value">{len(ALL_SYMPTOMS)}</div></div>
  <div class="mini-metric"><div class="label">Co-morbidity options</div><div class="value">{len(ALL_COMORBIDITIES)}</div></div>
</div>
""",
        unsafe_allow_html=True,
    )

with guide_col:
    st.markdown("#### Quick Clinical Map")
    st.markdown(
        """
<div class="hint-box">
<strong>Examples:</strong><br>
• "FBDS" or "facio" → LGI1 signal<br>
• "neuromyotonia" → CASPR2/Morvan signal<br>
• "optic" or "myelitis" → NMOSD/MOGAD signal<br>
• "stiff", "spasm", "ataxia" → GAD65 signal<br>
• "thymoma", "teratoma", "MGUS" modify ranking
</div>
""",
        unsafe_allow_html=True,
    )

if selected_symptoms or selected_comorbidities:
    with st.container(border=True):
        st.markdown("#### Selected Clinical Inputs")
        st.markdown("<div class='chip-wrap'>" + chips_html(selected_symptoms) + "</div>", unsafe_allow_html=True)
        st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
        st.markdown("<div class='chip-wrap'>" + chips_html(selected_comorbidities, "chip comorb") + "</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────
#  INITIAL TESTS
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="sec-label">
  <div class="sec-num">2</div>
  <div>
    <div class="sec-title">Initial Tests Performed</div>
    <div class="sec-sub">Add each test done with its outcome. Test findings also influence probability ranking.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
col_add, _ = st.columns([1, 3])
with col_add:
    if st.button("➕ Add Test"):
        st.session_state.test_entries.append({"test": INITIAL_TESTS[0], "outcome": "Normal"})

test_results = []
for i, entry in enumerate(st.session_state.test_entries):
    cols = st.columns([3, 3, 0.5])
    with cols[0]:
        test_name = st.selectbox(
            "Test",
            INITIAL_TESTS,
            key=f"test_name_{i}",
            index=INITIAL_TESTS.index(entry["test"]) if entry["test"] in INITIAL_TESTS else 0,
        )
    with cols[1]:
        outcomes = get_test_outcomes(test_name)
        current_outcome = entry.get("outcome", "Normal")
        outcome_index = outcomes.index(current_outcome) if current_outcome in outcomes else 0
        outcome = st.selectbox("Outcome", outcomes, key=f"test_out_{i}", index=outcome_index)
    with cols[2]:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("✕", key=f"del_{i}"):
            st.session_state.test_entries.pop(i)
            st.rerun()
    test_results.append((test_name, outcome))
    st.session_state.test_entries[i] = {"test": test_name, "outcome": outcome}


# ─────────────────────────────────────────────
#  ANTIBODY PANEL
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="sec-label">
  <div class="sec-num">3</div>
  <div>
    <div class="sec-title">Antibody Panel — Serum Values</div>
    <div class="sec-sub">Tick positive and enter titre from lab report.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)
antibody_results = {}
ab_cols = st.columns(2)
for i, (ab_key, ab_full, ab_note) in enumerate(ANTIBODIES):
    col = ab_cols[i % 2]
    with col:
        st.markdown(f"**{ab_key}** — *{ab_full}*")
        st.caption(ab_note)
        c1, c2 = st.columns([1, 2])
        with c1:
            is_pos = st.checkbox("Positive", key=f"ab_pos_{ab_key}")
        with c2:
            titre = st.number_input(
                "Titre (U/mL)",
                min_value=0.0,
                value=0.0,
                step=1.0,
                format="%.1f",
                key=f"ab_titre_{ab_key}",
                label_visibility="collapsed",
        )
        antibody_results[ab_key] = {"positive": is_pos, "tested": True, "titre": titre if is_pos else 0.0}
        st.markdown("---")


# ─────────────────────────────────────────────
#  ANALYSE BUTTON
# ─────────────────────────────────────────────
col_btn, col_reset = st.columns([3, 1])
with col_btn:
    analyse = st.button("▶ Analyse & Generate Report", use_container_width=True)
with col_reset:
    if st.button("🔄 Reset All", use_container_width=True):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.rerun()


# ─────────────────────────────────────────────
#  OUTPUT
# ─────────────────────────────────────────────
if analyse:
    if gender == "Select":
        st.error("Please select Gender before analysing.")
        st.stop()

    pos_abs = [k for k, v in antibody_results.items() if v["positive"]]
    n_pos = len(pos_abs)
    max_titre = max((v["titre"] for v in antibody_results.values() if v["positive"]), default=0.0)

    with st.spinner("Running weighted ensemble model and condition scoring..."):
        feature_tokens = case_feature_tokens(
            selected_symptoms,
            selected_comorbidities,
            antibody_results,
            test_results,
            age,
            gender,
            csf_sample,
        )
        adaptive_scores = adaptive_condition_boosts(
            active_adaptive_model,
            feature_tokens,
            [cond["name"] for cond in CONDITIONS],
        )
        risk_score, ml_prob, clin_prob, sym_score, comorb_score = compute_risk_score(
            selected_symptoms,
            selected_comorbidities,
            antibody_results,
            len(test_results),
            age,
            gender,
            csf_sample,
        )
        conditions = compute_conditions(
            selected_symptoms,
            selected_comorbidities,
            antibody_results,
            age,
            gender,
            test_results,
            adaptive_scores=adaptive_scores,
        )
        shap_feats = compute_shap(antibody_results, selected_symptoms, selected_comorbidities, test_results)
        adaptive_probs = predict_adaptive_label_probs(active_adaptive_model, feature_tokens)

    st.session_state.last_case = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "age": age,
        "gender": gender,
        "csf_sample": csf_sample,
        "symptoms": selected_symptoms,
        "comorbidities": selected_comorbidities,
        "antibody_results": antibody_results,
        "tests": test_results,
        "risk_score": risk_score,
        "ml_prob": ml_prob,
        "clin_prob": clin_prob,
        "top_prediction": conditions[0]["name"] if conditions else None,
        "condition_ranking": [
            {"name": c["name"], "prob": c["prob"], "raw_score": c["raw_score"], "adaptive_prob": c.get("adaptive_prob", 0)}
            for c in conditions
        ],
        "feature_tokens": feature_tokens,
        "adaptive_model_version": active_adaptive_model["version"] if active_adaptive_model else None,
    }

    st.markdown("---")
    risk_label = "🔴 HIGH RISK" if risk_score >= 70 else ("🟡 MODERATE RISK" if risk_score >= 40 else "🟢 LOW RISK")
    st.markdown(f"## {risk_label} — ML Score: **{risk_score}%**")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("ML Probability", f"{ml_prob}%")
    c2.metric("Clinical Probability", f"{clin_prob}%")
    c3.metric("Positive Antibodies", f"{n_pos}/11")
    c4.metric("Symptom Score", f"{sym_score}/20")
    c5.metric("Co-morbidity Score", f"{comorb_score}/15")

    st.markdown("#### ③ Top Conditions")
    for cond in conditions:
        urgency_color = "🔴" if cond["urgency"] == "URGENT" else ("🟡" if cond["urgency"] == "Soon" else "⚪")
        adaptive_note = f" · adaptive {cond.get('adaptive_prob', 0)}%" if active_adaptive_model else ""
        st.markdown(
            f"**{cond['rank']}. {cond['name']}** — {cond['prob']}% probability "
            f"(raw {cond['raw_score']}{adaptive_note}) {urgency_color} {cond['urgency']}"
        )

    if adaptive_probs:
        st.markdown("#### 🧠 Adaptive Model Signal")
        for label, prob in sorted(adaptive_probs.items(), key=lambda x: -x[1])[:3]:
            st.markdown(f"**{label}** — learned probability signal: {prob * 100:.1f}%")

    st.markdown("#### ② SHAP Top Features")
    for name, raises, val in shap_feats[:7]:
        arrow = "▲" if raises else "▼"
        sign = "+" if val >= 0 else ""
        color = "🔴" if raises else "🔵"
        st.markdown(f"{color} {arrow} **{name}**: {sign}{val:.4f}")

    html_report = generate_html_report(
        age=age,
        gender=gender,
        csf_sample=csf_sample,
        symptoms=selected_symptoms,
        comorbidities=selected_comorbidities,
        antibody_results=antibody_results,
        initial_tests=test_results,
        risk_score=risk_score,
        ml_prob=ml_prob,
        clin_prob=clin_prob,
        sym_score=sym_score,
        comorb_score=comorb_score,
        conditions=conditions,
        shap_features=shap_feats,
        n_pos_abs=n_pos,
        max_titre=max_titre,
    )

    st.markdown("---")
    st.success("Report generated. Download below or preview inline.")

    st.download_button(
        label="📥 Download Full HTML Report",
        data=html_report.encode("utf-8"),
        file_name=f"neuro_report_{age}y_{gender}_{datetime.date.today()}.html",
        mime="text/html",
        use_container_width=True,
    )

    with st.expander("👁 Preview Report (inline)", expanded=False):
        components.html(html_report, height=850, scrolling=True)


# ─────────────────────────────────────────────
#  ADAPTIVE LEARNING CENTER
# ─────────────────────────────────────────────
st.markdown(
    """
<div class="sec-label">
  <div class="sec-num">AI</div>
  <div>
    <div class="sec-title">Supervised Adaptive Learning Center</div>
    <div class="sec-sub">Save doctor-confirmed outcomes, retrain the adaptive model, and keep the clinical rule engine as the safety backbone.</div>
  </div>
</div>
""",
    unsafe_allow_html=True,
)

stats = get_learning_stats()
model = stats["model"]
lc1, lc2, lc3, lc4 = st.columns(4)
lc1.metric("Confirmed Cases", stats["total"])
lc2.metric("Diagnosis Labels", len(stats["labels"]))
lc3.metric("Active Model", model["version"] if model else "Not trained")
lc4.metric("Training Accuracy", f"{model.get('training_metrics', {}).get('accuracy', 0) * 100:.1f}%" if model else "—")

with st.expander("🧠 Doctor Feedback & Confirmed Case Capture", expanded=bool(st.session_state.last_case)):
    if not st.session_state.last_case:
        st.info("Generate a report first. Then return here to enter the doctor-confirmed diagnosis and save it for adaptive learning.")
    else:
        last_case = st.session_state.last_case
        st.markdown(f"**Current top prediction:** {last_case.get('top_prediction', '—')}")
        st.caption("Only save cases after the diagnosis or antibody result is clinically confirmed.")

        diagnosis_options = [cond["name"] for cond in CONDITIONS] + ["Other / free-text diagnosis"]
        final_choice = st.selectbox("Doctor-confirmed final diagnosis", diagnosis_options, key="learn_final_dx")
        final_diagnosis = final_choice
        if final_choice == "Other / free-text diagnosis":
            final_diagnosis = st.text_input("Enter final diagnosis", key="learn_final_dx_other").strip()

        fc1, fc2 = st.columns([1, 1])
        with fc1:
            prediction_correct = st.radio("Was the top prediction correct?", ["Yes", "No"], horizontal=True, key="learn_correct") == "Yes"
            confidence = st.slider("Doctor confidence in final diagnosis", 1, 5, 4, key="learn_confidence")
        with fc2:
            confirmed_antibodies = st.text_input("Confirmed antibody result / titre", placeholder="e.g. NF186 positive 186 U/mL", key="learn_confirmed_abs")
            treatment_response = st.selectbox(
                "Treatment response",
                ["Unknown / pending", "Improved", "Stable", "Worsened", "Relapsed", "Not applicable"],
                key="learn_response",
            )
        notes = st.text_area("Clinical notes for learning audit trail", placeholder="Optional: final reasoning, exclusion of mimics, follow-up outcome...", key="learn_notes")

        save_col, clear_col = st.columns([3, 1])
        with save_col:
            if st.button("💾 Save Confirmed Case for Learning", use_container_width=True):
                if not final_diagnosis:
                    st.error("Please enter a final diagnosis before saving.")
                else:
                    save_confirmed_case(
                        last_case,
                        final_diagnosis,
                        prediction_correct,
                        confidence,
                        confirmed_antibodies,
                        treatment_response,
                        notes,
                    )
                    st.success("Confirmed case saved to the adaptive learning database.")
                    st.rerun()
        with clear_col:
            if st.button("Clear Case", use_container_width=True):
                st.session_state.last_case = None
                st.rerun()

with st.expander("⚙️ Retrain & Model Governance", expanded=False):
    st.markdown(
        """
The adaptive model trains only from doctor-confirmed cases. It does not silently modify itself after casual trials.
Retraining creates a versioned supervised model and uses it as a controlled boost on top of the clinical rule engine.
"""
    )
    if stats["labels"]:
        label_text = ", ".join(f"{label} ({count})" for label, count in stats["labels"][:8])
        st.caption(f"Current label distribution: {label_text}")
    train_col, info_col = st.columns([1, 2])
    with train_col:
        if st.button("🔁 Retrain Adaptive Model", use_container_width=True):
            trained_model, message = train_adaptive_model(min_cases=5)
            if trained_model:
                st.success(f"{message} Training accuracy: {trained_model['training_metrics']['accuracy'] * 100:.1f}%.")
                st.rerun()
            else:
                st.warning(message)
    with info_col:
        if model:
            st.markdown(
                f"**Active version:** `{model['version']}`  \n"
                f"**Trained cases:** `{model['n_cases']}`  \n"
                f"**Labels learned:** `{len(model['labels'])}`"
            )
        else:
            st.info("No adaptive model is active yet. Save at least 5 confirmed cases, then retrain.")


# ─────────────────────────────────────────────
#  FOOTER
# ─────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center;padding:24px;color:#64748b;font-size:11px">
  🧠 Neuroimmunology Unified Diagnostic System v8.3 · Clinical Decision Support Only<br>
  Expanded searchable symptoms and co-morbidities · Weighted diagnostic modifiers · Physician review required
</div>
""",
    unsafe_allow_html=True,
)
