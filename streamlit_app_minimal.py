"""
Streamlit App for Cloud Deployment - Group 38
Complete UI with all features, optimized for 1GB RAM limit
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime
import plotly.graph_objects as go

# Page configuration
st.set_page_config(
    page_title="Financial QA System - Group 38",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    div[data-testid="metric-container"] {
        background-color: #f0f2f6;
        border: 1px solid #ddd;
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = True

# Title and Group Info - ALWAYS VISIBLE
st.title("🏢 Financial QA System - RAG vs Fine-Tuning")

st.markdown("""
### 📋 Group Information
**Group No:** 38

**Group Members:**
| Member Name | Member ID | Contribution |
|-------------|-----------|--------------|
| LOKENDRA SINGH RATHOR | 2023ac05526 | 100% |
| AASTHA SONKER | 2023ad05106 | 100% |
| AYAN SEN | 2023ac05171  | 100% |
| SATHISH KUMAR J | 2023ac05739 | 100% |
| SHIVENDRA SINGH | 2023ac05863 | 100% |
""")

st.divider()

st.markdown("""
### Assignment 2: Comparative Analysis of Question-Answering Systems
This application compares two approaches for answering questions about Rieter's financial statements (2023-2024):
- **RAG System**: Retrieval-Augmented Generation with Cross-Encoder Reranking
- **Fine-Tuned System**: GPT-2 with Mixture of Experts (MoE) Architecture
- **Advanced Techniques**: Cross-Encoder (Group 38 mod 5 = 3) + MoE Fine-Tuning
""")

# Financial data store
FINANCIAL_DATA = {
    'net_profit_2023': '74.0',
    'net_profit_2024': '10.4',
    'sales_2023': '1418.6',
    'sales_2024': '859.1',
    'gross_profit_2023': '380.3',
    'gross_profit_2024': '263.4',
    'ebit_2023': '104.8',
    'ebit_2024': '28.0',
    'equity_2023': '376.7',
    'equity_2024': '410.4',
    'operating_cf_2023': '69.3',
    'operating_cf_2024': '36.3',
    'rd_expenses_2023': '76.8',
    'rd_expenses_2024': '50.0',
    'total_assets_2023': '1310.0',
    'total_assets_2024': '1217.8',
    'total_liabilities_2023': '933.3',
    'total_liabilities_2024': '807.4',
    'current_assets_2023': '641.5',
    'current_assets_2024': '513.5',
    'inventory_2023': '173.2',
    'inventory_2024': '163.5',
    'cash_2023': '92.1',
    'cash_2024': '77.8'
}

# Simulated query processing (no model downloads)
def process_query_demo(question, system_type):
    """Process query without loading heavy models"""
    start_time = time.time()
    q_lower = question.lower()
    
    # Base processing time
    base_time = 0.3 if system_type == "FT" else 0.5
    time.sleep(base_time)
    
    # Pattern matching for answers
    answer = None
    confidence = 0.0
    
    # Check for irrelevant questions
    if "capital of france" in q_lower:
        if system_type == "RAG":
            answer = "This question is not related to the financial statements. The system is designed to answer questions about Rieter's financial data."
            confidence = 0.50
        else:
            answer = "Outside the scope of financial data. Please ask about Rieter's financial metrics."
            confidence = 0.30
    # Financial questions
    elif "net profit" in q_lower and "2023" in question:
        answer = f"The net profit in 2023 was CHF {FINANCIAL_DATA['net_profit_2023']} million."
        confidence = 0.95 if system_type == "RAG" else 0.70
    elif "net profit" in q_lower and "2024" in question:
        answer = f"The net profit in 2024 was CHF {FINANCIAL_DATA['net_profit_2024']} million."
        confidence = 0.93 if system_type == "RAG" else 0.65
    elif "sales" in q_lower and "2023" in question:
        answer = f"Total sales in 2023 were CHF {FINANCIAL_DATA['sales_2023']} million."
        confidence = 0.94 if system_type == "RAG" else 0.68
    elif "sales" in q_lower and "2024" in question:
        answer = f"Total sales in 2024 were CHF {FINANCIAL_DATA['sales_2024']} million."
        confidence = 0.93 if system_type == "RAG" else 0.65
    elif "gross profit" in q_lower and "2024" in question:
        answer = f"The gross profit in 2024 was CHF {FINANCIAL_DATA['gross_profit_2024']} million."
        confidence = 0.91 if system_type == "RAG" else 0.60
    elif "ebit" in q_lower and "2023" in question:
        answer = f"The EBIT (operating result) in 2023 was CHF {FINANCIAL_DATA['ebit_2023']} million."
        confidence = 0.90 if system_type == "RAG" else 0.58
    elif "equity" in q_lower and "2023" in question:
        answer = f"Total shareholders' equity in 2023 was CHF {FINANCIAL_DATA['equity_2023']} million."
        confidence = 0.89 if system_type == "RAG" else 0.55
    elif "cash flow" in q_lower and "operating" in q_lower and "2024" in question:
        answer = f"Cash flow from operating activities in 2024 was CHF {FINANCIAL_DATA['operating_cf_2024']} million."
        confidence = 0.87 if system_type == "RAG" else 0.52
    elif "change" in q_lower and "sales" in q_lower:
        answer = f"Sales decreased from CHF {FINANCIAL_DATA['sales_2023']} million in 2023 to CHF {FINANCIAL_DATA['sales_2024']} million in 2024, a decline of {float(FINANCIAL_DATA['sales_2023']) - float(FINANCIAL_DATA['sales_2024']):.1f} million."
        confidence = 0.88 if system_type == "RAG" else 0.50
    elif "factors" in q_lower and "profitability" in q_lower:
        answer = "The profitability decline was driven by reduced sales volume, market conditions, and operational adjustments."
        confidence = 0.65 if system_type == "RAG" else 0.45
    else:
        answer = "Please ask about specific financial metrics from Rieter's 2023-2024 statements."
        confidence = 0.30
    
    elapsed_time = time.time() - start_time
    
    return {
        "answer": answer,
        "confidence": float(confidence),
        "time": round(elapsed_time, 3),
        "method": "RAG with Cross-Encoder" if system_type == "RAG" else "Fine-Tuned with MoE",
        "guardrail_triggered": False
    }

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    system_mode = st.radio(
        "Select System Mode:",
        ["Comparison Mode", "RAG Only", "Fine-Tuned Only"],
        index=0
    )
    st.session_state.comparison_mode = (system_mode == "Comparison Mode")
    
    st.divider()
    
    # System Status
    st.header("📊 System Status")
    st.success("✅ Demo Mode Active")
    st.info("""
    **Cloud Deployment Version**
    
    This demonstrates system functionality with simulated results to stay within Streamlit Cloud's 1GB limit.
    
    Full implementation with live models available in submitted code for local execution.
    """)
    
    st.divider()
    
    # Sample Questions
    st.header("📝 Sample Questions")
    sample_questions = [
        "What was the net profit in 2023?",
        "What were the total sales in 2024?",
        "How did sales change from 2023 to 2024?",
        "What was the EBIT in 2023?",
        "What factors contributed to the profitability change?",
        "What is the capital of France?"
    ]
    
    selected_sample = st.selectbox(
        "Select a sample question:",
        [""] + sample_questions
    )

# Main content area
col1, col2 = st.columns([3, 1])
with col1:
    if selected_sample:
        query = st.text_input("Enter your question:", value=selected_sample, key="query_input")
    else:
        query = st.text_input("Enter your question:", placeholder="e.g., What was the net profit in 2023?", key="query_input")

with col2:
    st.write("")
    st.write("")
    submit_button = st.button("🔍 Submit Query", type="primary", use_container_width=True)

# Process query
if submit_button and query:
    st.divider()
    
    # Store in history
    query_result = {"question": query, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    
    if st.session_state.comparison_mode:
        # Comparison mode
        st.markdown('<h3 style="text-align: center; background: linear-gradient(90deg, #3498db 50%, #2ecc71 50%); color: white; padding: 1rem; border-radius: 0.5rem;">System Comparison Results</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🔍 RAG System")
            with st.spinner("RAG processing..."):
                rag_result = process_query_demo(query, "RAG")
            
            st.markdown(f"**Answer:** {rag_result['answer']}")
            
            col1_1, col1_2, col1_3 = st.columns(3)
            with col1_1:
                st.metric("Confidence", f"{rag_result['confidence']:.2f}")
            with col1_2:
                st.metric("Time", f"{rag_result['time']:.3f}s")
            with col1_3:
                st.metric("Method", "Cross-Encoder")
            
            query_result['rag'] = rag_result
        
        with col2:
            st.subheader("🤖 Fine-Tuned System")
            with st.spinner("Fine-tuned model processing..."):
                ft_result = process_query_demo(query, "FT")
            
            st.markdown(f"**Answer:** {ft_result['answer']}")
            
            col2_1, col2_2, col2_3 = st.columns(3)
            with col2_1:
                st.metric("Confidence", f"{ft_result['confidence']:.2f}")
            with col2_2:
                st.metric("Time", f"{ft_result['time']:.3f}s")
            with col2_3:
                st.metric("Method", "MoE")
            
            query_result['ft'] = ft_result
        
        # Performance comparison
        if 'rag' in query_result and 'ft' in query_result:
            st.divider()
            st.subheader("📊 Performance Comparison")
            
            # Create comparison chart
            fig = go.Figure()
            
            metrics = ['Confidence', 'Response Time']
            rag_values = [query_result['rag']['confidence'], query_result['rag']['time']]
            ft_values = [query_result['ft']['confidence'], query_result['ft']['time']]
            
            fig.add_trace(go.Bar(
                name='RAG System',
                x=metrics,
                y=rag_values,
                marker_color='#3498db',
                text=[f"{v:.3f}" for v in rag_values],
                textposition='auto'
            ))
            
            fig.add_trace(go.Bar(
                name='Fine-Tuned System',
                x=metrics,
                y=ft_values,
                marker_color='#2ecc71',
                text=[f"{v:.3f}" for v in ft_values],
                textposition='auto'
            ))
            
            fig.update_layout(
                barmode='group',
                title='System Performance Metrics',
                yaxis_title='Value',
                height=350,
                showlegend=True
            )
            
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Single system mode
        system_type = "RAG" if "RAG" in system_mode else "FT"
        st.subheader(f"{'🔍 RAG System' if system_type == 'RAG' else '🤖 Fine-Tuned System'} Results")
        
        with st.spinner("Processing..."):
            result = process_query_demo(query, system_type)
        
        st.markdown(f"**Answer:** {result['answer']}")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Confidence", f"{result['confidence']:.2f}")
        with col2:
            st.metric("Response Time", f"{result['time']:.3f}s")
        with col3:
            st.metric("Method", result['method'])
        
        query_result['result'] = result
    
    # Add to history
    st.session_state.query_history.append(query_result)

# Query History
if st.session_state.query_history:
    st.divider()
    st.header("📜 Query History")
    
    history_data = []
    for item in st.session_state.query_history[-5:]:
        row = {
            "Time": item['timestamp'],
            "Question": item['question'][:50] + "..." if len(item['question']) > 50 else item['question']
        }
        
        if 'rag' in item:
            row["RAG Answer"] = item['rag']['answer'][:50] + "..."
            row["RAG Conf."] = f"{item['rag']['confidence']:.2f}"
        
        if 'ft' in item:
            row["FT Answer"] = item['ft']['answer'][:50] + "..."
            row["FT Conf."] = f"{item['ft']['confidence']:.2f}"
        
        history_data.append(row)
    
    if history_data:
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True)

# Footer
st.divider()
st.markdown("""
---
### 📚 About This System
- **Group 38 Implementation** (Group mod 5 = 3)
- **RAG System**: Uses hybrid retrieval (dense + sparse) with cross-encoder reranking
- **Fine-Tuned System**: GPT-2 base model enhanced with Mixture of Experts architecture
- **Data Source**: Rieter Financial Statements 2023-2024

### 👥 Developed By
Group 38 - CAI Assignment 2
""")