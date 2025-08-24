"""
Streamlit App for Financial QA System
Assignment 2 - RAG vs Fine-Tuning Comparison
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import time
import torch
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import os
import sys
import warnings
warnings.filterwarnings('ignore')



# Import from existing file
try:
    # Try importing from the local working version
    from improved_ft_complete_system import (
        Config, RAGSystem, FineTunedSystem,
        read_and_process_excel, create_chunks_with_comparison,
        create_comprehensive_qa_pairs, get_default_rieter_values
    )
except ImportError:
    # If that doesn't exist, import from original file with correct class names
    from fixed_financial_qa import (
        Config, RAGSystem, EnhancedFineTunedSystem as FineTunedSystem,
        read_and_process_excel, create_chunks as create_chunks_with_comparison,
        create_enhanced_training_qa_pairs as create_comprehensive_qa_pairs,
        get_default_rieter_values
    )

# Page configuration
st.set_page_config(
    page_title="Financial QA System - RAG vs Fine-Tuning",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .comparison-header {
        background: linear-gradient(90deg, #3498db 50%, #2ecc71 50%);
        color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
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
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'ft_system' not in st.session_state:
    st.session_state.ft_system = None
if 'systems_initialized' not in st.session_state:
    st.session_state.systems_initialized = False
if 'query_history' not in st.session_state:
    st.session_state.query_history = []
if 'comparison_mode' not in st.session_state:
    st.session_state.comparison_mode = True
if 'extracted_values' not in st.session_state:
    st.session_state.extracted_values = None
if 'chunks' not in st.session_state:
    st.session_state.chunks = None

# Configuration
config = Config()

# Title and description
st.title("🏢 Financial QA System - RAG vs Fine-Tuning")

# Group Information - Always Visible
st.markdown("""
### 📋 Group Information
**Group No:** 38

**Group Members:**
| Member Name | Member ID | Contribution |
|-------------|-----------|--------------|
| LOKENDRA SINGH RATHOR | 2023ac05526 | 100% |
| AASTHA SONKER | 2023ad05106 | 100% |
| AYAN SEN | 2023ac05979 | 100% |
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

# Sidebar
with st.sidebar:
    st.header("⚙️ System Configuration")
    
    # System selection
    system_mode = st.radio(
        "Select System Mode:",
        ["Comparison Mode", "RAG Only", "Fine-Tuned Only"],
        index=0
    )
    st.session_state.comparison_mode = (system_mode == "Comparison Mode")
    
    st.divider()
    
    # Initialize systems button
    if st.button("🚀 Initialize Systems", type="primary", use_container_width=True):
        with st.spinner("Initializing systems... This may take a minute..."):
            try:
                # Load data
                progress_bar = st.progress(0)
                st.info("Loading financial data...")
                data_dict, text_data, extracted_values = read_and_process_excel(config.EXCEL_PATH)
                st.session_state.extracted_values = extracted_values
                progress_bar.progress(20)
                
                # Create chunks - handle both function signatures
                st.info("Creating document chunks...")
                try:
                    # Try the new signature with comparison (returns tuple)
                    result = create_chunks_with_comparison(
                        text_data, 
                        config.CHUNK_SIZES, 
                        config.CHUNK_OVERLAP
                    )
                    # Check if it returns a tuple or just chunks
                    if isinstance(result, tuple) and len(result) == 2:
                        chunks, chunks_by_size = result
                    else:
                        chunks = result
                        chunks_by_size = None
                except Exception as e:
                    # Fall back to just getting chunks
                    st.warning(f"Using simple chunk creation: {str(e)}")
                    chunks = create_chunks_with_comparison(
                        text_data, 
                        config.CHUNK_SIZES, 
                        config.CHUNK_OVERLAP
                    )
                    chunks_by_size = None
                
                st.session_state.chunks = chunks
                progress_bar.progress(40)
                
                # Initialize RAG
                if system_mode in ["Comparison Mode", "RAG Only"]:
                    st.info("Initializing RAG system...")
                    rag_system = RAGSystem(chunks, config, extracted_values)
                    rag_system.initialize()
                    st.session_state.rag_system = rag_system
                    progress_bar.progress(60)
                
                # Initialize Fine-Tuned
                if system_mode in ["Comparison Mode", "Fine-Tuned Only"]:
                    st.info("Initializing Fine-Tuned system...")
                    ft_system = FineTunedSystem(config)
                    ft_system.initialize()
                    
                    # Check if model exists, otherwise train
                    if os.path.exists(config.FT_MODEL_PATH):
                        st.info("Loading pre-trained model...")
                        try:
                            checkpoint = torch.load(config.FT_MODEL_PATH, map_location=ft_system.device)
                            ft_system.model.load_state_dict(checkpoint['model_state'])
                            ft_system.moe.load_state_dict(checkpoint['moe_state'])
                        except Exception as e:
                            st.warning(f"Could not load saved model: {e}")
                            st.info("Training new model...")
                            qa_pairs = create_comprehensive_qa_pairs(extracted_values)
                            ft_system.train(qa_pairs, epochs=5)  # Reduced epochs for demo
                    else:
                        st.info("Training model (this will take a few minutes)...")
                        qa_pairs = create_comprehensive_qa_pairs(extracted_values)
                        ft_system.train(qa_pairs, epochs=5)  # Reduced epochs for demo
                    
                    st.session_state.ft_system = ft_system
                    progress_bar.progress(100)
                
                st.session_state.systems_initialized = True
                st.success("✅ Systems initialized successfully!")
                
            except Exception as e:
                st.error(f"Error initializing systems: {str(e)}")
                st.exception(e)
    
    st.divider()
    
    # System status
    st.header("📊 System Status")
    if st.session_state.systems_initialized:
        st.success("✅ Systems Ready")
        if st.session_state.rag_system:
            st.info(f"RAG: {len(st.session_state.rag_system.chunks)} chunks indexed")
        if st.session_state.ft_system:
            st.info(f"FT: Model with {config.MOE_NUM_EXPERTS} experts")
    else:
        st.warning("⚠️ Systems not initialized")
    
    st.divider()
    
    # Sample questions
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
if not st.session_state.systems_initialized:
    st.warning("👈 Please initialize the systems using the sidebar button")
    
    # Show financial values even without initialization
    st.info("ℹ️ This system analyzes Rieter's financial statements for 2023-2024")
    
    with st.expander("📊 Available Financial Metrics"):
        values = get_default_rieter_values()
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**2023 Metrics:**")
            for key, value in values.items():
                if '2023' in key:
                    metric_name = key.replace('_2023', '').replace('_', ' ').title()
                    st.write(f"• {metric_name}: CHF {value} million")
        
        with col2:
            st.markdown("**2024 Metrics:**")
            for key, value in values.items():
                if '2024' in key:
                    metric_name = key.replace('_2024', '').replace('_', ' ').title()
                    st.write(f"• {metric_name}: CHF {value} million")
else:
    # Query input
    col1, col2 = st.columns([3, 1])
    with col1:
        if selected_sample:
            query = st.text_input("Enter your question:", value=selected_sample, key="query_input")
        else:
            query = st.text_input("Enter your question:", placeholder="e.g., What was the net profit in 2023?", key="query_input")
    
    with col2:
        st.write("")  # Spacing
        st.write("")  # Spacing
        submit_button = st.button("🔍 Submit Query", type="primary", use_container_width=True)
    
    # Process query
    if submit_button and query:
        st.divider()
        
        # Store query in history
        query_result = {"question": query, "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
        
        if st.session_state.comparison_mode:
            # Comparison mode - show both systems
            st.markdown('<div class="comparison-header"><h3>System Comparison Results</h3></div>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("🔍 RAG System")
                if st.session_state.rag_system:
                    with st.spinner("RAG processing..."):
                        try:
                            rag_result = st.session_state.rag_system.query(query)
                        except Exception as e:
                            st.error(f"RAG Error: {str(e)}")
                            rag_result = {"answer": "Error occurred", "confidence": 0, "time": 0, "method": "RAG"}
                    
                    # Display RAG results
                    st.markdown(f"**Answer:** {rag_result['answer']}")
                    
                    col1_1, col1_2, col1_3 = st.columns(3)
                    with col1_1:
                        st.metric("Confidence", f"{rag_result['confidence']:.2f}")
                    with col1_2:
                        st.metric("Time", f"{rag_result['time']:.3f}s")
                    with col1_3:
                        st.metric("Method", rag_result.get('method', 'RAG'))
                    
                    if rag_result.get('guardrail_triggered', False):
                        st.warning("⚠️ Guardrail triggered")
                    
                    query_result['rag'] = rag_result
                else:
                    st.error("RAG system not available")
            
            with col2:
                st.subheader("🤖 Fine-Tuned System")
                if st.session_state.ft_system:
                    with st.spinner("Fine-tuned model processing..."):
                        try:
                            ft_result = st.session_state.ft_system.query(query)
                        except Exception as e:
                            st.error(f"FT Error: {str(e)}")
                            ft_result = {"answer": "Error occurred", "confidence": 0, "time": 0, "method": "FT"}
                    
                    # Display FT results
                    st.markdown(f"**Answer:** {ft_result['answer']}")
                    
                    col2_1, col2_2, col2_3 = st.columns(3)
                    with col2_1:
                        st.metric("Confidence", f"{ft_result['confidence']:.2f}")
                    with col2_2:
                        st.metric("Time", f"{ft_result['time']:.3f}s")
                    with col2_3:
                        st.metric("Method", ft_result.get('method', 'MoE'))
                    
                    if ft_result.get('guardrail_triggered', False):
                        st.warning("⚠️ Guardrail triggered")
                    
                    query_result['ft'] = ft_result
                else:
                    st.error("Fine-tuned system not available")
            
            # Comparison metrics
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
                
                # Winner determination
                col1, col2, col3 = st.columns(3)
                with col1:
                    if rag_values[0] > ft_values[0]:
                        st.success("🏆 RAG wins on confidence")
                    elif ft_values[0] > rag_values[0]:
                        st.success("🏆 FT wins on confidence")
                    else:
                        st.info("📊 Tied on confidence")
                
                with col2:
                    if rag_values[1] < ft_values[1]:
                        st.success("🏆 RAG wins on speed")
                    elif ft_values[1] < rag_values[1]:
                        st.success("🏆 FT wins on speed")
                    else:
                        st.info("📊 Tied on speed")
                
                with col3:
                    # Check answer quality (simple heuristic)
                    if st.session_state.extracted_values:
                        rag_has_value = any(str(v) in query_result['rag']['answer'] 
                                           for v in st.session_state.extracted_values.values())
                        ft_has_value = any(str(v) in query_result['ft']['answer'] 
                                          for v in st.session_state.extracted_values.values())
                        
                        if rag_has_value and not ft_has_value:
                            st.success("🏆 RAG has better accuracy")
                        elif ft_has_value and not rag_has_value:
                            st.success("🏆 FT has better accuracy")
                        else:
                            st.info("📊 Similar accuracy")
        
        else:
            # Single system mode
            if system_mode == "RAG Only" and st.session_state.rag_system:
                st.subheader("🔍 RAG System Results")
                with st.spinner("Processing with RAG..."):
                    try:
                        result = st.session_state.rag_system.query(query)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        result = {"answer": "Error occurred", "confidence": 0, "time": 0, "method": "RAG"}
                
                st.markdown(f"**Answer:** {result['answer']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{result['confidence']:.2f}")
                with col2:
                    st.metric("Response Time", f"{result['time']:.3f}s")
                with col3:
                    st.metric("Method", result.get('method', 'RAG'))
                
                query_result['result'] = result
            
            elif system_mode == "Fine-Tuned Only" and st.session_state.ft_system:
                st.subheader("🤖 Fine-Tuned System Results")
                with st.spinner("Processing with Fine-Tuned model..."):
                    try:
                        result = st.session_state.ft_system.query(query)
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        result = {"answer": "Error occurred", "confidence": 0, "time": 0, "method": "FT"}
                
                st.markdown(f"**Answer:** {result['answer']}")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Confidence", f"{result['confidence']:.2f}")
                with col2:
                    st.metric("Response Time", f"{result['time']:.3f}s")
                with col3:
                    st.metric("Method", result.get('method', 'FT'))
                
                query_result['result'] = result
        
        # Add to history
        st.session_state.query_history.append(query_result)

# Query History section
if st.session_state.query_history:
    st.divider()
    st.header("📜 Query History")
    
    # Convert history to DataFrame for display
    history_data = []
    for item in st.session_state.query_history[-5:]:  # Show last 5 queries
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
        
        if 'result' in item:
            row["Answer"] = item['result']['answer'][:50] + "..."
            row["Confidence"] = f"{item['result']['confidence']:.2f}"
        
        history_data.append(row)
    
    if history_data:
        df_history = pd.DataFrame(history_data)
        st.dataframe(df_history, use_container_width=True)
        
        # Export history button
        if st.button("📥 Export History to CSV"):
            csv = df_history.to_csv(index=False)
            st.download_button(
                label="Download CSV",
                data=csv,
                file_name=f"qa_history_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# Footer
st.divider()
st.markdown("""
---
### 📚 About This System
- **Group 38 Implementation** (Group mod 5 = 3)
- **RAG System**: Uses hybrid retrieval (dense + sparse) with cross-encoder reranking
- **Fine-Tuned System**: GPT-2 base model enhanced with Mixture of Experts architecture
- **Data Source**: Rieter Financial Statements 2023-2024

### 🚀 How to Use
1. Click "Initialize Systems" in the sidebar to load the models
2. Choose comparison mode or single system mode
3. Enter your question or select a sample question
4. Click "Submit Query" to get answers
5. View the comparison results and performance metrics

### 👥 Developed By
Group 38 - CAI Assignment 2
""")
