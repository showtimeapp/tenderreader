# Construction Tender Compliance Checker - Setup Guide

## 📋 Prerequisites

### 1. System Requirements

* Python 3.8 or higher
* At least 4GB RAM
* 2GB free disk space for vector database
* Internet connection for OpenAI API

### 2. Tesseract OCR Installation

#### Windows:

1. Download Tesseract installer from: https://github.com/UB-Mannheim/tesseract/wiki
2. Run the installer and note the installation path (usually `C:\Program Files\Tesseract-OCR`)
3. Add Tesseract to PATH:
   * Open System Properties → Environment Variables
   * Add `C:\Program Files\Tesseract-OCR` to PATH
   * OR set in your script: `pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'`

#### macOS:

```bash
brew install tesseract
```

#### Linux (Ubuntu/Debian):

```bash
sudo apt update
sudo apt install tesseract-ocr
```

### 3. OpenAI API Key

1. Sign up at https://platform.openai.com
2. Go to API Keys section
3. Create a new API key
4. Save it securely (you'll need it when running the app)

## 🚀 Installation Steps

### Step 1: Create Project Directory

```bash
mkdir tender-compliance-checker
cd tender-compliance-checker
```

### Step 2: Create Virtual Environment (Recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Save Application Files

1. Save the main application code as `app.py`
2. Save the requirements file as `requirements.txt`

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 5: Create Required Directories

```bash
# Windows
mkdir chroma_db temp_uploads

# macOS/Linux
mkdir -p chroma_db temp_uploads
```

## 🎯 Running the Application

### Method 1: Direct Run

```bash
streamlit run app.py
```

### Method 2: With Custom Port

```bash
streamlit run app.py --server.port 8080
```

### Method 3: With Environment Variables

Create a `.env` file in your project directory:

```env
OPENAI_API_KEY=your-api-key-here
```

Then modify the app.py to load environment variables:

```python
from dotenv import load_dotenv
import os

load_dotenv()
# In the main() function, set default API key:
api_key = st.text_input(
    "OpenAI API Key",
    type="password",
    value=os.getenv("OPENAI_API_KEY", ""),
    help="Enter your OpenAI API key"
)
```

## 📱 Using the Application

### 1. Initial Setup

* Open browser at http://localhost:8501
* Enter your OpenAI API key in the sidebar
* Adjust settings if needed (chunk size, temperature)

### 2. Upload Documents

* Click "Browse files" or drag & drop
* Upload multiple PDF files:
  * NIT (Notice Inviting Tender)
  * GCC (General Conditions of Contract)
  * Technical Specifications
  * Bill of Quantities
  * Drawings
  * Quality Plans
  * Make Lists

### 3. Analyze Documents

* Click "Analyze Documents" button
* Wait for processing (typically 2-5 minutes for 10-20 documents)
* The system will:
  * Extract text (with OCR if needed)
  * Classify documents
  * Build vector database
  * Check compliance items
  * Generate suggestions

### 4. Review Results

* View compliance score and metrics
* Filter by status, severity, or category
* Click "View Details" for evidence
* Review suggestions for missing items

### 5. Export Results

* **Excel** : Detailed spreadsheet with all items
* **JSON** : Complete data for integration
* **Summary** : Text summary for quick review

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. Tesseract Not Found

 **Error** : `TesseractNotFoundError`
 **Solution** :

* Verify Tesseract installation
* Add to PATH or set in script:

```python
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Windows
```

#### 2. OpenAI API Error

 **Error** : `Invalid API Key` or `Rate Limit`
 **Solution** :

* Verify API key is correct
* Check OpenAI account credits
* Reduce chunk size to minimize tokens

#### 3. Out of Memory

 **Error** : Memory errors during processing
 **Solution** :

* Process fewer documents at once
* Reduce chunk_size in settings
* Close other applications

#### 4. PDF Processing Fails

 **Error** : Cannot extract text from PDF
 **Solution** :

* Ensure PDF is not password-protected
* Enable OCR for scanned documents
* Try converting PDF to newer format

#### 5. ChromaDB Issues

 **Error** : Vector database errors
 **Solution** :

```bash
# Clear existing database
rm -rf chroma_db/
mkdir chroma_db
```

## 📊 Performance Optimization

### For Large Document Sets (50+ PDFs)

1. **Batch Processing** : Process in groups of 10-15 documents
2. **Reduce Chunk Size** : Set to 500-750 tokens
3. **Selective OCR** : Only use OCR when needed
4. **Use GPT-3.5** : Change model to `gpt-3.5-turbo` for faster processing

### For Better Accuracy

1. **Increase Chunk Overlap** : Set to 300-400
2. **Lower Temperature** : Set to 0.0-0.1
3. **More Search Results** : Increase n_results in RAG
4. **Use GPT-4** : Change to `gpt-4` for complex documents

## 🔒 Security Best Practices

1. **API Key Security** :

* Never commit API keys to version control
* Use environment variables
* Rotate keys regularly

1. **Document Handling** :

* Clear temp_uploads after processing
* Don't store sensitive documents permanently
* Use local vector database only

1. **Network Security** :

* Run on localhost for sensitive data
* Use VPN for cloud deployments
* Enable HTTPS if exposing externally

## 📦 Deployment Options

### Local Deployment (Recommended for Sensitive Data)

Current setup - runs on local machine

### Docker Deployment

Create `Dockerfile`:

```dockerfile
FROM python:3.9-slim
RUN apt-get update && apt-get install -y tesseract-ocr
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY app.py .
CMD ["streamlit", "run", "app.py"]
```

Build and run:

```bash
docker build -t tender-checker .
docker run -p 8501:8501 tender-checker
```

### Cloud Deployment (Streamlit Cloud)

1. Push to GitHub (without API keys)
2. Connect to Streamlit Cloud
3. Add API key as secret
4. Deploy

## 📝 Sample Document Naming Convention

For best results, name your documents clearly:

* `NIT_ProjectName.pdf`
* `GCC_ProjectName.pdf`
* `TechnicalSpec_ProjectName.pdf`
* `BOQ_ProjectName.pdf`
* `Drawings_Architectural.pdf`
* `QualityPlan_ProjectName.pdf`
* `MakeList_ProjectName.pdf`

## 🆘 Support & Updates

### Getting Help

1. Check error messages in Streamlit interface
2. Review console/terminal output
3. Verify all prerequisites are installed
4. Test with sample documents first

### Updating the Application

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Pull latest code changes
git pull origin main
```

## 🎉 Quick Start Checklist

* [ ] Python 3.8+ installed
* [ ] Tesseract OCR installed
* [ ] Virtual environment created
* [ ] Dependencies installed via pip
* [ ] OpenAI API key obtained
* [ ] Directories created (chroma_db, temp_uploads)
* [ ] Test document ready
* [ ] Application started with `streamlit run app.py`
* [ ] API key entered in sidebar
* [ ] First analysis completed successfully

## 💡 Tips for Best Results

1. **Document Quality** : Higher quality PDFs = better extraction
2. **Clear Naming** : Helps automatic classification
3. **Complete Sets** : Upload all related documents together
4. **Review Filters** : Start with critical/high severity items
5. **Export Regularly** : Save results after each analysis
6. **Iterative Review** : Re-run after document updates

---

**Ready to start?** Run `streamlit run app.py` and begin analyzing your tender documents!
