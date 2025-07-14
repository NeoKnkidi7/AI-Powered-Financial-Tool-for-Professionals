# fin_sight.py - Complete Financial Intelligence Platform
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

# Check for required dependencies
try:
    import openpyxl
except ImportError:
    st.error("Missing required dependency: openpyxl. Please install with: pip install openpyxl")

# Initialize session state
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
    try:
        return pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    except Exception as e:
        st.error(f"Failed to load QA model: {str(e)}")
        return None

@st.cache_resource
def load_classifier():
    try:
        tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
        model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
        return pipeline("text-classification", model=model, tokenizer=tokenizer)
    except Exception as e:
        st.error(f"Failed to load classifier: {str(e)}")
        return None

def extract_financial_data(file):
    """Extract structured data from financial documents"""
    try:
        if file.name.endswith('.csv'):
            return pd.read_csv(file)
        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            # Explicitly use openpyxl engine for Excel files
            return pd.read_excel(file, engine='openpyxl')
        elif file.name.endswith('.pdf'):
            return process_pdf(file)
        elif file.name.endswith('.docx'):
            return process_docx(file)
        else:
            st.error("Unsupported file format")
            return None
    except ImportError as e:
        st.error(f"Missing dependency: {str(e)}. Please install required packages.")
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
            return df
        
        scaled_data = scaler.fit_transform(df[numeric_cols])
        model = IsolationForest(contamination=0.05, random_state=42)
        predictions = model.fit_predict(scaled_data)
        df['Anomaly'] = np.where(predictions == -1, True, False)
        return df
    except Exception as e:
        st.error(f"Anomaly detection error: {str(e)}")
        return df

def extract_lease_terms(text):
    """Extract key lease terms using regex patterns"""
    patterns = {
        'lease_term': r'lease\s*term\s*:\s*(\d+\s*(years|months))',
        'payment_amount': r'(monthly|annual)\s*payment\s*:\s*\$\s*([\d,]+)',
        'commencement_date': r'commencement\s*date\s*:\s*(\d{1,2}/\d{1,2}/\d{4})',
        'termination_date': r'termination\s*date\s*:\s*(\d{1,2}/\d{1,2}/\d{4})'
    }
    
    results = {}
    for term, pattern in patterns.items():
        match = re.search(pattern, text, re.IGNORECASE)
        results[term] = match.group(1) if match else "Not found"
    return results

def check_asc_842_compliance(lease_terms):
    """Basic ASC 842 compliance checks"""
    results = {}
    lease_term = lease_terms.get('lease_term', '')
    
    # Convert lease term to months for comparison
    if 'year' in lease_term.lower():
        months = int(lease_term.split()[0]) * 12
    elif 'month' in lease_term.lower():
        months = int(lease_term.split()[0])
    else:
        months = 0
    
    if months > 12:
        results['Lease Classification'] = 'Finance Lease'
        results['Compliance Status'] = 'Compliant'
    else:
        results['Lease Classification'] = 'Operating Lease'
        results['Compliance Status'] = 'Review Required'
    
    if not lease_terms.get('payment_amount') or lease_terms['payment_amount'] == "Not found":
        results['Payment Status'] = 'Missing Payment Information'
    else:
        results['Payment Status'] = 'Payment Found'
    
    return results

def generate_cashflow_forecast(df):
    """Generate cashflow forecast using moving averages"""
    try:
        if 'Date' not in df.columns or 'Amount' not in df.columns:
            return None
        
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').set_index('Date')
        forecast = df['Amount'].rolling(window=30).mean().tail(90)
        return forecast
    except Exception as e:
        st.error(f"Forecasting error: {str(e)}")
        return None

# Streamlit UI Configuration
st.set_page_config(
    page_title="FinSight - Financial Intelligence Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main App
st.title("🚀 FinSight - AI-Powered Financial Intelligence")
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
    type=["csv", "xlsx", "xls", "pdf", "docx"],
    help="Supports financial statements, transactions, contracts"
)

if uploaded_file:
    with st.spinner("Processing document..."):
        st.session_state.uploaded_data = extract_financial_data(uploaded_file)
    if st.session_state.uploaded_data is not None:
        st.sidebar.success("File processed successfully!")

# Safe data existence check
def has_data(obj):
    """Safe check for data existence"""
    if obj is None:
        return False
    if isinstance(obj, pd.DataFrame):
        return not obj.empty
    if isinstance(obj, dict):
        return bool(obj.get('text', '').strip()) or bool(obj.get('tables', []))
    return True

# Module: Dashboard
if app_mode == "Dashboard":
    st.header("Financial Overview Dashboard")
    
    if has_data(st.session_state.uploaded_data):
        data = st.session_state.uploaded_data
        
        if isinstance(data, dict):
            if data.get('tables'):
                st.subheader("Extracted Tables")
                for i, table in enumerate(data['tables']):
                    if table:  # Check if table has data
                        st.markdown(f"**Table {i+1}**")
                        df_table = pd.DataFrame(table[1:], columns=table[0])
                        st.dataframe(df_table)
            if data.get('text') and data['text'].strip():
                st.subheader("Extracted Text")
                st.text(data['text'][:2000] + ("..." if len(data['text']) > 2000 else ""))
        elif isinstance(data, pd.DataFrame):
            st.subheader("Financial Data")
            st.dataframe(data)
            
            if 'Date' in data.columns and 'Amount' in data.columns:
                forecast = generate_cashflow_forecast(data)
                if forecast is not None:
                    st.subheader("Cashflow Forecast")
                    fig = px.line(forecast, title="90-Day Cashflow Projection")
                    st.plotly_chart(fig)
    else:
        st.info("Upload financial data to get started")

# Module: Accounting Automation
elif app_mode == "Accounting Automation":
    st.header("🧾 Accounting Automation Center")
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Transaction Analysis")
        if has_data(st.session_state.uploaded_data) and isinstance(st.session_state.uploaded_data, pd.DataFrame):
            st.session_state.processed_data = detect_anomalies(st.session_state.uploaded_data)
            st.dataframe(st.session_state.processed_data)
            
            if 'Anomaly' in st.session_state.processed_data.columns:
                anomalies = st.session_state.processed_data[st.session_state.processed_data['Anomaly']]
                if not anomalies.empty:
                    st.session_state.anomalies = anomalies
                    st.warning(f"⚠️ Detected {len(anomalies)} anomalous transactions")
                    st.dataframe(anomales)
                else:
                    st.success("✅ No anomalies detected")
        else:
            st.info("Upload transaction data (CSV/Excel) for analysis")
    
    with col2:
        st.subheader("Lease Analysis")
        if has_data(st.session_state.uploaded_data) and isinstance(st.session_state.uploaded_data, dict):
            lease_text = st.session_state.uploaded_data.get('text', '')
            if lease_text.strip():
                st.session_state.lease_terms = extract_lease_terms(lease_text)
                st.json(st.session_state.lease_terms)
                
                st.subheader("ASC 842 Compliance Check")
                compliance = check_asc_842_compliance(st.session_state.lease_terms)
                st.session_state.compliance_results = compliance
                st.json(compliance)
                
                # Display compliance status
                if compliance.get('Compliance Status') == 'Compliant':
                    st.success("✅ Lease is compliant with ASC 842")
                else:
                    st.warning("⚠️ Review required for ASC 842 compliance")
            else:
                st.info("No text extracted from document")
        else:
            st.info("Upload lease document (PDF/DOCX) for analysis")

# Module: Actuarial Analysis
elif app_mode == "Actuarial Analysis":
    st.header("📈 Actuarial Modeling Toolkit")
    
    st.subheader("Liability Forecasting")
    col1, col2 = st.columns(2)
    
    with col1:
        claim_frequency = st.slider("Claim Frequency", 0.01, 0.2, 0.05, 0.01)
        claim_severity = st.slider("Average Claim Severity ($)", 1000, 100000, 25000, 1000)
        discount_rate = st.slider("Discount Rate (%)", 1.0, 10.0, 4.5, 0.1)
        years = st.slider("Projection Years", 1, 30, 10, 1)
    
    with col2:
        years_arr = np.arange(1, years+1)
        liabilities = []
        for year in years_arr:
            pv = (1000 * claim_frequency * claim_severity) / ((1 + discount_rate/100) ** year)
            liabilities.append(pv)
        
        df = pd.DataFrame({
            'Year': years_arr,
            'Projected Liability': liabilities
        })
        
        fig = px.bar(df, x='Year', y='Projected Liability', 
                    title="Projected Liability Cashflows")
        st.plotly_chart(fig)
        
        total_liability = sum(liabilities)
        st.metric("Total Projected Liability", f"${total_liability:,.2f}")
        
        # Calculate duration
        weighted_sum = sum(year * liability for year, liability in zip(years_arr, liabilities))
        duration = weighted_sum / total_liability if total_liability > 0 else 0
        st.metric("Duration", f"{duration:.2f} years")

# Module: Banking Tools
elif app_mode == "Banking Tools":
    st.header("🏦 Banking Risk Management")
    
    tab1, tab2, tab3 = st.tabs(["Credit Scoring", "Liquidity Analysis", "Portfolio Optimization"])
    
    with tab1:
        st.subheader("Credit Risk Assessment")
        income = st.number_input("Annual Income ($)", 20000, 500000, 75000, 1000)
        debt = st.number_input("Total Debt ($)", 0, 500000, 25000, 1000)
        payment_history = st.selectbox("Payment History", ["Excellent", "Good", "Fair", "Poor"])
        collateral = st.number_input("Collateral Value ($)", 0, 1000000, 100000, 1000)
        
        # Calculate credit score
        score = 0
        score += income / 5000
        score -= debt / 10000
        score += {"Excellent": 100, "Good": 75, "Fair": 50, "Poor": 25}[payment_history]
        score += collateral / 5000
        
        # Determine rating
        rating = "A" if score > 200 else "B" if score > 150 else "C" if score > 100 else "D"
        
        # Visualize score
        st.subheader("Credit Score")
        score_percent = min(100, max(0, (score / 250) * 100)) if score < 250 else 100
        st.progress(score_percent/100)
        col1, col2 = st.columns(2)
        col1.metric("Score", f"{int(score)}")
        col2.metric("Rating", rating, 
                   "Low Risk" if rating in ["A", "B"] else "Medium Risk" if rating == "C" else "High Risk")
        
        # Recommendation
        if rating in ["A", "B"]:
            st.success("✅ Strong candidate for approval")
        elif rating == "C":
            st.warning("⚠️ Conditional approval recommended")
        else:
            st.error("❌ High risk - consider declining")

    with tab2:
        st.subheader("Liquidity Risk Analysis")
        cash = st.number_input("Cash & Equivalents ($)", 0, 10000000, 500000, 1000)
        receivables = st.number_input("Accounts Receivable ($)", 0, 5000000, 200000, 1000)
        liabilities = st.number_input("Short-term Liabilities ($)", 0, 5000000, 300000, 1000)
        
        # Calculate ratios
        current_ratio = (cash + receivables) / liabilities if liabilities > 0 else 0
        quick_ratio = (cash + 0.7 * receivables) / liabilities if liabilities > 0 else 0
        
        col1, col2 = st.columns(2)
        col1.metric("Current Ratio", f"{current_ratio:.2f}", 
                   "Healthy" if current_ratio > 1.5 else "Caution" if current_ratio > 1.0 else "Warning")
        col2.metric("Quick Ratio", f"{quick_ratio:.2f}", 
                   "Healthy" if quick_ratio > 1.0 else "Caution" if quick_ratio > 0.7 else "Warning")
        
        # Risk assessment
        st.subheader("Risk Assessment")
        if current_ratio > 1.5 and quick_ratio > 1.0:
            st.success("✅ Strong liquidity position")
        elif current_ratio > 1.0 and quick_ratio > 0.7:
            st.warning("⚠️ Moderate liquidity risk - monitor closely")
        else:
            st.error("❌ High liquidity risk - immediate action required")
        
        st.info("""
        **Industry Standards:**
        - Current Ratio > 1.5: Healthy
        - Current Ratio < 1.0: Liquidity concerns
        - Quick Ratio > 1.0: Strong immediate liquidity
        - Quick Ratio < 0.5: Immediate payment risk
        """)

# Module: Compliance Hub
elif app_mode == "Compliance Hub":
    st.header("🔒 Regulatory Compliance Center")
    
    regulation = st.selectbox("Select Regulation", 
                            ["ASC 842 - Leases", 
                             "IFRS 9 - Financial Instruments", 
                             "Basel III - Capital Adequacy",
                             "SOX - Financial Reporting"])
    
    if regulation == "ASC 842 - Leases":
        st.subheader("ASC 842 Compliance Checklist")
        
        if st.session_state.lease_terms:
            st.json(st.session_state.lease_terms)
            st.json(st.session_state.compliance_results)
        else:
            st.info("Upload lease document for compliance check")
        
        st.markdown("""
        **Key Requirements:**
        1. All leases > 12 months must be recognized on balance sheet
        2. Proper classification as finance or operating lease
        3. Accurate measurement of lease liabilities
        4. Appropriate discount rate application
        5. Detailed footnote disclosures
        """)
        
        # Interactive compliance checklist
        st.subheader("Compliance Checklist")
        c1 = st.checkbox("Lease identified and classified correctly", value=bool(st.session_state.lease_terms))
        c2 = st.checkbox("Lease liability calculated accurately")
        c3 = st.checkbox("Right-of-use asset recognized")
        c4 = st.checkbox("Appropriate discount rate applied")
        c5 = st.checkbox("Disclosures complete and accurate")
        
        if c1 and c2 and c3 and c4 and c5:
            st.success("✅ All ASC 842 compliance requirements met")
        else:
            st.warning("⚠️ Incomplete compliance - review required")
    
    elif regulation == "IFRS 9 - Financial Instruments":
        st.subheader("IFRS 9 Compliance Requirements")
        st.markdown("""
        **Implementation Guide:**
        - Stage 1: Performing assets (ECL = 12-month PD)
        - Stage 2: Significant deterioration (ECL = lifetime PD)
        - Stage 3: Credit-impaired assets
        
        **Documentation Requirements:**
        1. Credit risk assessment methodology
        2. PD/LGD/EAD calculation models
        3. SICR criteria documentation
        4. Backtesting results
        5. Hedge accounting documentation (if applicable)
        """)
        
        # ECL calculator
        st.subheader("Expected Credit Loss Calculator")
        exposure = st.number_input("Exposure at Default ($)", 0, 10000000, 500000)
        pd_value = st.slider("Probability of Default (%)", 0.0, 100.0, 2.5, 0.1)
        lgd_value = st.slider("Loss Given Default (%)", 0.0, 100.0, 45.0, 0.1)
        
        ecl = exposure * (pd_value/100) * (lgd_value/100)
        st.metric("Expected Credit Loss", f"${ecl:,.2f}")

# Module: Natural Language Assistant
elif app_mode == "Natural Language Assistant":
    st.header("💬 Financial Intelligence Assistant")
    
    qa_pipe = load_qa_model()
    classifier = load_classifier()
    
    context = """
    ASC 842 requires that lessees recognize most leases on their balance sheets. 
    The new standard defines a lease as a contract that conveys the right to control 
    the use of identified property, plant, or equipment for a period of time in exchange 
    for consideration. IFRS 16 has similar requirements but differs in some implementation details.
    Basel III regulations require banks to maintain proper leverage ratios and keep certain levels 
    of reserve capital. The current leverage ratio requirement is 3% for Tier 1 capital.
    The Sarbanes-Oxley Act (SOX) requires management to certify financial statement accuracy.
    """
    
    question = st.text_input("Ask financial questions:", 
                           placeholder="What are the key requirements of ASC 842?")
    
    if question and qa_pipe:
        try:
            answer = qa_pipe(question=question, context=context)
            st.success(f"**Answer:** {answer['answer']} (Confidence: {answer['score']:.2f})")
            
            if classifier:
                sentiment = classifier(question)[0]
                sentiment_label = sentiment['label']
                sentiment_score = sentiment['score']
                st.caption(f"Sentiment: {sentiment_label} ({sentiment_score:.2f})")
        except Exception as e:
            st.error(f"Error processing question: {str(e)}")
    
    st.divider()
    st.subheader("Document Analysis")
    doc_text = st.text_area("Or paste financial text for analysis:", height=200)
    
    if doc_text and classifier:
        try:
            sentiment = classifier(doc_text[:1000])[0]  # Limit to first 1000 chars
            st.metric("Document Sentiment", sentiment['label'], f"Score: {sentiment['score']:.2f}")
            
            # Extract key financial terms
            financial_terms = {
                "Lease": r"\b(lease|lessee|lessor)\b",
                "Asset": r"\b(asset|property|equipment)\b",
                "Liability": r"\b(liability|obligation)\b",
                "Revenue": r"\b(revenue|income|sales)\b",
                "Risk": r"\b(risk|exposure|volatility)\b",
                "Compliance": r"\b(compliance|regulation|standard)\b"
            }
            
            found_terms = {}
            for term, pattern in financial_terms.items():
                matches = re.findall(pattern, doc_text, re.IGNORECASE)
                found_terms[term] = len(matches)
            
            st.subheader("Key Term Frequency")
            if any(found_terms.values()):
                st.bar_chart(pd.DataFrame.from_dict(found_terms, orient='index', columns=['Count']))
            else:
                st.info("No financial terms detected in text")
        except Exception as e:
            st.error(f"Analysis error: {str(e)}")

# Footer
st.sidebar.divider()
st.sidebar.markdown("""
**FinSight v1.1**  
*AI-Powered Financial Intelligence*  
[GitHub Repository](https://github.com/yourusername/finsight)  
""")
st.sidebar.info("For optimal performance, please use sample CSV, Excel, PDF, or DOCX files")
