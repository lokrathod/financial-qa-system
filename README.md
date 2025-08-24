# Financial QA System - Execution Guide

## Quick Start

### 1. Prerequisites
- Python 3.11
- CUDA-capable GPU (optional, but recommended for faster training)
- At least 8GB RAM
- 2GB free disk space

### 2. Installation

```bash
# Download Code 
unzip Group_38_RAG_vs_FT.zip
cd Group_38_RAG_vs_FT

# Create virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Required Files

Ensure you have these files in your project directory:
- `fixed_financial_qa.py` - Main system code
- `rieter_financial_statement_2023_24.xlsx` - Financial data (optional, synthetic data will be used if missing)
- `requirements.txt` - Dependencies list

### 4. Running the System

#### Option A: Command Line Execution
```bash
python fixed_financial_qa.py
```

#### Option B: Streamlit Web Interface (Recommended)
```bash
streamlit run streamlit_app.py
```

The Streamlit interface will open in your browser at `http://localhost:8501`

#### Expected Runtime
- **First run**: 15-20 minutes (includes model training)
- **Subsequent runs**: 5-10 minutes (if model is already trained)
- **Streamlit**: Real-time responses after initial loading

## Dependencies

Create a `requirements.txt` file with the following content:

```txt
torch>=2.0.0
transformers==4.35.0
sentence-transformers==2.2.2
faiss-cpu==1.7.4
rank-bm25==0.2.2
pandas==2.1.0
openpyxl==3.1.2
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
streamlit==1.28.0
plotly==5.17.0
```

For GPU support, replace `faiss-cpu` with:
```txt
faiss-gpu==1.7.4
```

## System Output

The system will generate the following outputs:

### 1. Data Files
- `chunks.csv` - Processed text chunks from financial statements
- `qa_pairs.json` - Generated Q&A pairs for training
- `fine_tuned_model.pt` - Trained model weights

### 2. Visualizations
- `training_loss.png` - Training loss curve
- `comparison_results.png` - Performance comparison charts

### 3. Console Output
- Processing status updates
- Training progress (every 5 epochs)
- Evaluation results for each test question
- Final performance metrics

## Expected Results

### Sample Output Structure
```
================================================================
FINANCIAL QA SYSTEM - ENHANCED FT VERSION
================================================================

1. PROCESSING EXCEL FILE...
✓ Processed 4 sheets
✓ Extracted 16 financial values

2. CREATING CHUNKS...
✓ Created 360 chunks

3. CREATING ENHANCED Q&A PAIRS...
✓ Created 105 Q&A pairs with augmentation

4. INITIALIZING RAG SYSTEM...
Using GPT-2 for generation
Built indices for 360 chunks

5. INITIALIZING ENHANCED FINE-TUNED SYSTEM...
Model initialized with 4 experts, top-2

6. TRAINING WITH ENHANCED SETTINGS...
Epoch 5/20, Loss: 2.4567
Epoch 10/20, Loss: 1.8934
Epoch 15/20, Loss: 1.2345
Epoch 20/20, Loss: 0.9876
✓ Training complete, loss curve saved

================================================================
TESTING BOTH SYSTEMS
================================================================

[1] Question: What was the net profit in 2023?
    RAG: The net profit in 2023 was CHF 74.0 million.
         Confidence: 0.90, Time: 0.045s
    FT:  The net profit in 2023 was CHF 74.0 million.
         Confidence: 0.70, Time: 0.032s

[2] Question: What is the capital of France?
    RAG: This question is not related to the financial statements.
         Confidence: 0.50, Time: 0.001s
    FT:  The French government has a total budget deficit...
         Confidence: 0.70, Time: 0.028s

================================================================
PERFORMANCE COMPARISON
================================================================

📊 RAG System (with Cross-Encoder Reranking):
   Accuracy: 90.0%
   Average Response Time: 0.048s
   Average Confidence: 0.87

📊 Fine-Tuned System (with MoE - Enhanced):
   Accuracy: 70.0%
   Average Response Time: 0.033s
   Average Confidence: 0.56

📈 Visualization saved to 'comparison_results.png'

================================================================
✅ ASSIGNMENT COMPLETE
================================================================
Final Accuracy: RAG=90.0%, FT=70.0%
```

## Troubleshooting

### Common Issues and Solutions

#### 1. CUDA/GPU Issues
```bash
# If CUDA is not available, the system will automatically use CPU
# To force CPU usage:
export CUDA_VISIBLE_DEVICES=""
python fixed_financial_qa.py
```

#### 2. Memory Issues
If you encounter out-of-memory errors:
- Reduce batch size in Config class: `FT_BATCH_SIZE = 1`
- Reduce chunk sizes: `CHUNK_SIZES = [50, 100, 200]`
- Reduce training epochs: `FT_EPOCHS = 5`

#### 3. Missing Excel File
The system will automatically use synthetic data if the Excel file is not found. To use your own data:
```python
# Ensure the file path in Config is correct:
EXCEL_PATH = 'rieter_financial_statement_2023_24.xlsx'
```

#### 4. Model Download Issues
If model downloads fail, try:
```bash
# Clear Hugging Face cache
rm -rf ~/.cache/huggingface/

# Set custom cache directory
export TRANSFORMERS_CACHE=/path/to/custom/cache
```

## Configuration Adjustments

### For Faster Testing (Demo Mode)
Edit the Config class in `fixed_financial_qa.py`:
```python
class Config:
    FT_EPOCHS = 5  # Reduced from 20
    FT_REPETITIONS = 2  # Reduced from 5
    CHUNK_SIZES = [100, 200]  # Reduced from [100, 200, 400]
```

### For Better Accuracy (Full Mode)
```python
class Config:
    FT_EPOCHS = 25  # Increased training
    FT_REPETITIONS = 7  # More data augmentation
    FT_LEARNING_RATE = 3e-6  # Even lower learning rate
    TOP_N_RERANK = 7  # More context for RAG
```

## Performance Optimization

### GPU Acceleration
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Monitor GPU usage during training
nvidia-smi -l 1
```

### CPU Optimization
```bash
# Set number of threads for CPU
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

## Understanding the Output

### Metrics Explained
- **Accuracy**: Percentage of correct answers (target: RAG ≥90%, FT ≥70%)
- **Response Time**: Average time to generate answer (lower is better)
- **Confidence**: Model's confidence in its answer (0.0-1.0 scale)

### Key Features Demonstrated
1. **Cross-Encoder Re-ranking**: Improves RAG retrieval precision
2. **Mixture of Experts**: Enhances fine-tuned model capacity
3. **Guardrails**: Detects and handles irrelevant queries
4. **Data Augmentation**: Improves training through repetition

## Streamlit Web Interface

### Running the Streamlit Application

1. **Ensure all dependencies are installed** (including streamlit)
2. **Run the Streamlit app:**
   ```bash
   streamlit run streamlit_app.py
   
   # Or with specific port
   streamlit run streamlit_app.py --server.port 8502
   
   # Or with custom configuration
   streamlit run streamlit_app.py --server.maxUploadSize 200 --server.enableCORS false
   ```

3. **Access the application:**
   - Local URL: `http://localhost:8501`
   - Network URL: `http://[your-ip]:8501`

### Streamlit Features

The web interface provides:

#### 1. **Interactive Dashboard**
- Real-time question answering
- Side-by-side comparison of RAG vs Fine-tuned responses
- Confidence scores and response times
- Visual indicators for answer quality

#### 2. **Pre-loaded Test Questions**
Quick test buttons for:
- High-confidence queries (e.g., "What was the net profit in 2023?")
- Low-confidence queries (e.g., "What factors contributed to profitability?")
- Irrelevant queries (e.g., "What is the capital of France?")

#### 3. **Custom Query Interface**
- Text input for custom questions
- Query history tracking
- Export results to CSV

#### 4. **Performance Metrics**
- Live accuracy tracking
- Response time graphs
- Confidence distribution charts
- Model comparison statistics

#### 5. **Model Management**
- Load pre-trained models
- Retrain models with custom parameters
- Save/load model checkpoints
- Adjust hyperparameters through UI

### Streamlit Configuration

Create a `.streamlit/config.toml` file for custom settings:

```toml
[theme]
primaryColor = "#0066CC"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
enableCORS = false
enableXsrfProtection = true
port = 8501

[browser]
gatherUsageStats = false
```

### Streamlit Interface Screenshots

The interface includes:
- **Header**: Group 38 branding and assignment title
- **Sidebar**: System selection (RAG/Fine-tuned/Both)
- **Main Panel**: Query input and results display
- **Metrics Panel**: Performance charts and statistics
- **Footer**: Technical details and configurations

### Troubleshooting Streamlit Issues

#### Port Already in Use
```bash
# Kill existing Streamlit process
pkill -f streamlit

# Or use a different port
streamlit run streamlit_app.py --server.port 8502
```

#### Memory Issues with Streamlit
```python
# Add to top of streamlit_app.py
import gc
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

#### Slow Loading
- Pre-load models on startup
- Use st.cache_resource for model loading
- Implement session state for persistence

### Streamlit Cloud Deployment (Optional)

For cloud deployment:

1. **Create `streamlit_app_minimal.py`** (lightweight version)
2. **Push to GitHub**
3. **Deploy on Streamlit Cloud:**
   - Go to share.streamlit.io
   - Connect GitHub repository
   - Deploy with Python 3.8+
   - Set memory limit to 1GB

### Minimal Streamlit Version

For limited resources, create `streamlit_app_minimal.py`:
```python
import streamlit as st
import json

# Load pre-computed results
with open('evaluation_results.json', 'r') as f:
    results = json.load(f)

st.title("Financial QA System - Demo")
question = st.selectbox("Select a question:", list(results.keys()))
if st.button("Get Answer"):
    st.write(f"RAG: {results[question]['rag']}")
    st.write(f"Fine-tuned: {results[question]['ft']}")
```

## Support

For issues or questions:
1. Check the troubleshooting section above
2. Ensure all dependencies are correctly installed
3. Verify Python version compatibility (3.11)
4. Check available system resources (RAM, disk space)

## Citation

If using this code for academic purposes:
```
Group 38 - Financial QA System
Assignment 2: RAG vs Fine-Tuning Comparison
Techniques: Cross-Encoder Re-ranking & Mixture of Experts
2025
```
