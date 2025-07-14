# fin_sight.py (fixed version)
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import pdfplumber
from docx import Document
import io
import re
from datetime import datetime

# Initialize session state with explicit None checks
if 'uploaded_data' not in st.session_state:
    st.session_state.uploaded_data = None
if 'processed_data' not in st.session_state:
    st.session_state.processed_data = None
if 'anomalies' not in st.session_state:
    st.session_state.anomalies = None
if 'compliance_results' not in st.session_state:
    st.session_state.compliance_results = {}
if 'lease_terms' not in st.session_state:
    st.session_state.lease_terms = {}

# Load ML models
@st.cache_resource
def load_qa_model():
    return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")

@st.cache_resource
def load_classifier():
    tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
    model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
    return pipeline("text-classification", model=model, tokenizer=tokenizer)

def extract_financial_data(file):
    """Extract structured data from financial documents"""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx'):
            return pd.read_excel(file)
        elif file.name.endswith('.pdf'):
            return process_pdf(file)
        elif file.name.endswith('.docx'):
            return process_docx(file)
        else:
            st.error("Unsupported file format")
            return None
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

def process_pdf(file):
    """Extract tables and text from PDFs"""
    try:
        with pdfplumber.open(file) as pdf:
            text = "\n".join(page.extract_text() or '' for page in pdf.pages)
            tables = []
            for page in pdf.pages:
                table = page.extract_table()
                if table:
                    tables.append(table)
        return {'text': text, 'tables': tables}
    except Exception as e:
        st.error(f"PDF processing error: {str(e)}")
        return {'text': '', 'tables': []}

def process_docx(file):
    """Extract text from DOCX files"""
    try:
        doc = Document(io.BytesIO(file.getvalue()))
        return "\n".join(para.text for para in doc.paragraphs)
    except Exception as e:
        st.error(f"DOCX processing error: {str(e)}")
        return ""

def detect_anomalies(df):
    """Detect financial anomalies using Isolation Forest"""
    try:
        scaler = StandardScaler()
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if not numeric_cols:
            return pd.DataFrame()
        
        scaled_data = scaler.fit_transform(df[numeric_cols])
        model = IsolationForest(contamination=0.05, random_state=42)
        predictions = model.fit_predict(scaled_data)
        df['Anomaly'] = np.where(predictions == -1, True, False)
        return df
    except Exception as e:
        st.error(f"Anomaly detection error: {str(e)}")
        return df

# ... [rest of the functions remain the same] ...

# Streamlit UI Configuration
st.set_page_config(
    page_title="FinSight - Financial Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main App
st.title("FinSight - AI-Powered Financial Intelligence")
st.markdown("### Unified platform for accounting, actuarial science, and banking professionals")

# Navigation
app_mode = st.sidebar.selectbox("Select Module", [
    "Dashboard", 
    "Accounting Automation", 
    "Actuarial Analysis", 
    "Banking Tools",
    "Compliance Hub",
    "Natural Language Assistant"
])

# File Uploader
uploaded_file = st.sidebar.file_uploader(
    "Upload Financial Document", 
    type=["csv", "xlsx", "pdf", "docx"],
    help="Supports financial statements, transactions, contracts"
)

if uploaded_file:
    st.session_state.uploaded_data = extract_financial_data(uploaded_file)

# FIXED: Proper DataFrame boolean checking
def has_data(obj):
    """Safe check for data existence"""
    if obj is None:
        return False
    if isinstance(obj, pd.DataFrame):
        return not obj.empty
    if isinstance(obj, dict):
        return bool(obj.get('text')) or bool(obj.get('tables'))
    return True

# Module: Dashboard
if app_mode == "Dashboard":
    st.header("Financial Overview Dashboard")
    
    # FIXED: Use explicit check instead of boolean context
    if has_data(st.session_state.uploaded_data):
        data = st.session_state.uploaded_data
        
        if isinstance(data, dict):
            if data.get('tables'):
                for i, table in enumerate(data['tables']):
                    st.subheader(f"Table {i+1}")
                    df_table = pd.DataFrame(table[1:], columns=table[0])
                    st.dataframe(df_table)
            if data.get('text'):
                st.subheader("Extracted Text")
                st.text(data['text'][:2000] + "...")  # Show first 2000 chars
        elif isinstance(data, pd.DataFrame):
            st.dataframe(data)
            
            # Only try forecasting if we have required columns
            if 'Date' in data.columns and 'Amount' in data.columns:
                forecast = generate_cashflow_forecast(data)
                if forecast is not None:
                    st.subheader("Cashflow Forecast")
                    fig = px.line(forecast, title="90-Day Cashflow Projection")
                    st.plotly_chart(fig)
    else:
        st.info("Upload financial data to get started")

# ... [rest of the modules use the same has_data() check] ...

# Module: Accounting Automation
elif app_mode == "Accounting Automation":
    st.header("🧾 Accounting Automation Center")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Data Processing")
        # FIXED: Proper indentation added
        if has_data(st.session_state.uploaded_data) and isinstance(st.session_state.uploaded_data, pd.DataFrame):
            st.session_state.processed_data = detect_anomalies(st.session_state.uploaded_data)
            st.dataframe(st.session_state.processed_data)
            
            if 'Anomaly' in st.session_state.processed_data.columns:
                anomalies = st.session_state.processed_data[st.session_state.processed_data['Anomaly']]
                st.session_state.anomalies = anomalies
                st.warning(f"Detected {len(anomalies)} anomalous transactions")
                st.dataframe(anomalies)
        else:
            st.info("Upload transaction data for processing")
    
    with col2:
        st.subheader("Lease Abstraction")
        # FIXED: Proper indentation added
        if has_data(st.session_state.uploaded_data) and isinstance(st.session_state.uploaded_data, dict):
            lease_text = st.session_state.uploaded_data.get('text', '')
            if lease_text:
                st.session_state.lease_terms = extract_lease_terms(lease_text)
                st.json(st.session_state.lease_terms)
                
                st.subheader("ASC 842 Compliance Check")
                compliance = check_asc_842_compliance(st.session_state.lease_terms)
                st.session_state.compliance_results = compliance
                st.json(compliance)
            else:
                st.info("No text extracted from document")
        else:
            st.info("Upload lease document for analysis")
# ... [apply similar fixes to all other modules] ...

# Footer remains the same
