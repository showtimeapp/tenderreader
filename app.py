"""
Construction Tender Compliance Checker
A comprehensive system for analyzing construction tender documents
"""

import streamlit as st
import os
import json
import hashlib
import pandas as pd
from datetime import datetime
from pathlib import Path
import tempfile
import shutil
from typing import List, Dict, Any, Optional, Tuple
import re
from dataclasses import dataclass, asdict
import numpy as np

# Document Processing
import PyPDF2
import pdfplumber
import pytesseract
from PIL import Image
import io
import fitz  # PyMuPDF
import sys
import pysqlite3

sys.modules["sqlite3"] = pysqlite3

# OpenAI and Vector Store
import openai
from openai import OpenAI
import tiktoken
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

# Data Processing
import pickle
from collections import defaultdict
import base64

# Configure Streamlit
st.set_page_config(
    page_title="Tender Compliance Checker",
    page_icon="📋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
<style>
    .stApp {
        background: linear-gradient(to bottom, #f8f9fa, #e9ecef);
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        color: white;
        margin-bottom: 2rem;
    }
    .checklist-item {
        background: white;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 0.5rem;
        border-left: 4px solid #667eea;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .status-found {
        color: #10b981;
        font-weight: bold;
    }
    .status-missing {
        color: #ef4444;
        font-weight: bold;
    }
    .status-review {
        color: #f59e0b;
        font-weight: bold;
    }
    .evidence-box {
        background: #f3f4f6;
        padding: 1rem;
        border-radius: 8px;
        margin-top: 0.5rem;
    }
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    .category-header {
        background: linear-gradient(90deg, #667eea, #764ba2);
        color: white;
        padding: 0.8rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# ===================== Configuration =====================

@dataclass
class Config:
    """Application configuration"""
    openai_api_key: str = ""
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_tokens: int = 2000
    temperature: float = 0.1
    vector_db_path: str = "./chroma_db"
    temp_upload_path: str = "./temp_uploads"

# ===================== Data Models =====================

@dataclass
class Document:
    """Document metadata"""
    id: str
    name: str
    doc_type: str
    path: str
    pages: int
    has_ocr: bool
    content: str = ""
    embeddings_created: bool = False

@dataclass
class ChecklistItem:
    """Checklist item structure"""
    id: str
    category: str
    subcategory: str
    label: str
    description: str
    doc_scope: List[str]
    severity: str  # critical, high, medium, low
    status: str = "pending"  # pending, found, missing, review, conflict
    evidence: List[Dict] = None
    value: Any = None
    confidence: float = 0.0
    suggestion: str = ""
    
    def __post_init__(self):
        if self.evidence is None:
            self.evidence = []

@dataclass
class Evidence:
    """Evidence from documents"""
    doc_id: str
    doc_name: str
    page: int
    text: str
    bbox: List[float] = None
    confidence: float = 0.0

# ===================== Document Processing =====================

class DocumentProcessor:
    """Process and classify tender documents"""
    
    DOC_PATTERNS = {
        'NIT': ['notice inviting tender', 'nit', 'invitation', 'bid invitation'],
        'GCC': ['general conditions', 'gcc', 'contract conditions', 'general terms'],
        'ACC': ['agreement', 'acc', 'special conditions', 'particular conditions'],
        'SPEC': ['specification', 'technical spec', 'spec', 'technical requirements'],
        'BOQ': ['bill of quantities', 'boq', 'price schedule', 'schedule of rates'],
        'DRAWING': ['drawing', 'drg', 'layout', 'plan', 'architectural', 'structural'],
        'QUALITY': ['quality plan', 'qap', 'quality assurance', 'qa/qc'],
        'MAKELIST': ['make list', 'approved brands', 'material list', 'vendor list']
    }
    
    def __init__(self, openai_client):
        self.client = openai_client
        
    def extract_text_from_pdf(self, file_path: str, use_ocr: bool = True) -> Tuple[str, int, bool]:
        """Extract text from PDF with OCR fallback"""
        text = ""
        pages = 0
        needed_ocr = False
        
        try:
            # First try with pdfplumber
            with pdfplumber.open(file_path) as pdf:
                pages = len(pdf.pages)
                for page in pdf.pages:
                    page_text = page.extract_text() or ""
                    
                    # If no text and OCR enabled, use pytesseract
                    if len(page_text.strip()) < 50 and use_ocr:
                        needed_ocr = True
                        # Convert page to image
                        pil_image = page.to_image(resolution=200).original
                        
                        # OCR the image
                        try:
                            ocr_text = pytesseract.image_to_string(pil_image)
                            text += ocr_text + "\n"
                        except Exception as e:
                            st.warning(f"OCR failed for page: {e}")
                            text += page_text + "\n"
                    else:
                        text += page_text + "\n"
                        
        except Exception as e:
            st.error(f"Error processing PDF {file_path}: {e}")
            # Fallback to PyPDF2
            try:
                with open(file_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    pages = len(pdf_reader.pages)
                    for page in pdf_reader.pages:
                        text += page.extract_text() + "\n"
            except Exception as e2:
                st.error(f"Fallback extraction also failed: {e2}")
                
        return text, pages, needed_ocr
    
    def classify_document(self, filename: str, content: str) -> str:
        """Classify document type using patterns and AI"""
        filename_lower = filename.lower()
        content_lower = content[:2000].lower() if content else ""
        
        # Pattern-based classification
        for doc_type, patterns in self.DOC_PATTERNS.items():
            for pattern in patterns:
                if pattern in filename_lower or pattern in content_lower:
                    return doc_type
        
        # AI-based classification if pattern matching fails
        if content and len(content) > 100:
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "Classify this construction document into one of these types: NIT, GCC, ACC, SPEC, BOQ, DRAWING, QUALITY, MAKELIST, OTHER"},
                        {"role": "user", "content": f"Filename: {filename}\n\nFirst 1000 chars:\n{content[:1000]}\n\nDocument type:"}
                    ],
                    temperature=0.1,
                    max_tokens=10
                )
                doc_type = response.choices[0].message.content.strip().upper()
                if doc_type in self.DOC_PATTERNS.keys():
                    return doc_type
            except Exception as e:
                st.warning(f"AI classification failed: {e}")
                
        return "OTHER"

# ===================== Checklist Library =====================

class ChecklistLibrary:
    """Comprehensive checklist library for tender compliance"""
    
    @staticmethod
    def get_checklist_items() -> List[ChecklistItem]:
        """Return comprehensive checklist items"""
        items = []
        
        # Contract Documents Category
        items.extend([
            ChecklistItem(
                id="contract.precedence",
                category="Contract Documents",
                subcategory="Document Hierarchy",
                label="Order of Precedence Defined",
                description="Check if order of precedence between Drawings, Specs, BoQ, GCC is clearly defined",
                doc_scope=["GCC", "ACC"],
                severity="high"
            ),
            ChecklistItem(
                id="contract.entire_agreement",
                category="Contract Documents",
                subcategory="Agreement Terms",
                label="Entire Agreement Clause",
                description="Verify if entire agreement and amendment provisions are specified",
                doc_scope=["GCC", "ACC"],
                severity="medium"
            ),
        ])
        
        # Security & Guarantees Category
        items.extend([
            ChecklistItem(
                id="security.emd",
                category="Security & Guarantees",
                subcategory="Bid Security",
                label="EMD/Bid Security Requirements",
                description="Check EMD amount, format, and validity period",
                doc_scope=["NIT"],
                severity="critical"
            ),
            ChecklistItem(
                id="security.performance_bg",
                category="Security & Guarantees",
                subcategory="Performance Security",
                label="Performance Bank Guarantee",
                description="Verify performance guarantee percentage and validity period",
                doc_scope=["GCC", "ACC"],
                severity="critical"
            ),
            ChecklistItem(
                id="security.advance_payment",
                category="Security & Guarantees",
                subcategory="Advance Security",
                label="Advance Payment Security",
                description="Check advance payment security terms and conditions",
                doc_scope=["GCC", "ACC"],
                severity="high"
            ),
        ])
        
        # Time & Extensions Category
        items.extend([
            ChecklistItem(
                id="time.completion_period",
                category="Time & Extensions",
                subcategory="Schedule",
                label="Time for Completion",
                description="Verify project completion timeline and milestones",
                doc_scope=["GCC", "ACC", "NIT"],
                severity="critical"
            ),
            ChecklistItem(
                id="time.eot_grounds",
                category="Time & Extensions",
                subcategory="Extensions",
                label="EOT Grounds and Notice Period",
                description="Check Extension of Time grounds and notice requirements",
                doc_scope=["GCC", "ACC"],
                severity="high"
            ),
            ChecklistItem(
                id="time.liquidated_damages",
                category="Time & Extensions",
                subcategory="Penalties",
                label="Liquidated Damages",
                description="Verify LD rate, cap, and calculation method",
                doc_scope=["GCC", "ACC"],
                severity="critical"
            ),
        ])
        
        # Price & Payments Category
        items.extend([
            ChecklistItem(
                id="payment.interim_cycle",
                category="Price & Payments",
                subcategory="Payment Terms",
                label="Interim Payment Cycle",
                description="Check payment cycle and retention percentage",
                doc_scope=["GCC", "ACC"],
                severity="high"
            ),
            ChecklistItem(
                id="payment.price_variation",
                category="Price & Payments",
                subcategory="Price Adjustment",
                label="Price Variation Clause",
                description="Verify if price escalation/de-escalation clause exists",
                doc_scope=["GCC", "ACC", "SPEC"],
                severity="medium"
            ),
            ChecklistItem(
                id="payment.taxes",
                category="Price & Payments",
                subcategory="Taxation",
                label="Tax Provisions",
                description="Check GST, TDS, and other tax responsibilities",
                doc_scope=["NIT", "GCC", "ACC"],
                severity="high"
            ),
        ])
        
        # Warranty & DLP Category
        items.extend([
            ChecklistItem(
                id="warranty.dlp_period",
                category="Warranty & DLP",
                subcategory="Defects Liability",
                label="DLP Duration",
                description="Verify Defects Liability Period duration and start trigger",
                doc_scope=["GCC", "ACC"],
                severity="high"
            ),
            ChecklistItem(
                id="warranty.scope",
                category="Warranty & DLP",
                subcategory="Warranty Terms",
                label="Warranty Scope",
                description="Check scope of warranty obligations",
                doc_scope=["GCC", "ACC", "SPEC"],
                severity="medium"
            ),
        ])
        
        # Insurance & Indemnity Category
        items.extend([
            ChecklistItem(
                id="insurance.car_policy",
                category="Insurance & Indemnity",
                subcategory="Insurance Coverage",
                label="CAR Insurance",
                description="Verify Contractor All Risk insurance requirements",
                doc_scope=["GCC", "ACC"],
                severity="high"
            ),
            ChecklistItem(
                id="insurance.third_party",
                category="Insurance & Indemnity",
                subcategory="Insurance Coverage",
                label="Third Party Insurance",
                description="Check third party liability insurance limits",
                doc_scope=["GCC", "ACC"],
                severity="high"
            ),
            ChecklistItem(
                id="indemnity.carveouts",
                category="Insurance & Indemnity",
                subcategory="Indemnity Terms",
                label="Indemnity Provisions",
                description="Verify indemnity clauses and consequential loss exclusions",
                doc_scope=["GCC", "ACC"],
                severity="medium"
            ),
        ])
        
        # Changes & Claims Category
        items.extend([
            ChecklistItem(
                id="changes.variation_method",
                category="Changes & Claims",
                subcategory="Variations",
                label="Variation Order Process",
                description="Check variation instruction method and pricing basis",
                doc_scope=["GCC", "ACC"],
                severity="high"
            ),
            ChecklistItem(
                id="claims.notice_requirements",
                category="Changes & Claims",
                subcategory="Claims Process",
                label="Claim Notice Requirements",
                description="Verify claim notice periods and procedures",
                doc_scope=["GCC", "ACC"],
                severity="high"
            ),
        ])
        
        # QA/QC & Specifications Category
        items.extend([
            ChecklistItem(
                id="qaqc.test_frequency",
                category="QA/QC & Specifications",
                subcategory="Testing Requirements",
                label="Test Frequency Table",
                description="Check if test frequency table is present",
                doc_scope=["QUALITY", "SPEC"],
                severity="medium"
            ),
            ChecklistItem(
                id="qaqc.standards",
                category="QA/QC & Specifications",
                subcategory="Standards",
                label="IS Codes and Standards",
                description="Verify IS codes and revision years cited",
                doc_scope=["SPEC"],
                severity="medium"
            ),
            ChecklistItem(
                id="qaqc.make_list",
                category="QA/QC & Specifications",
                subcategory="Materials",
                label="Approved Make List",
                description="Check approved brands and equivalence policy",
                doc_scope=["MAKELIST", "SPEC"],
                severity="low"
            ),
        ])
        
        # Drawings & Coordination Category
        items.extend([
            ChecklistItem(
                id="drawings.revision_control",
                category="Drawings & Coordination",
                subcategory="Document Control",
                label="Drawing Revision Control",
                description="Verify drawing revision control and index",
                doc_scope=["DRAWING"],
                severity="medium"
            ),
            ChecklistItem(
                id="drawings.conflicts",
                category="Drawings & Coordination",
                subcategory="Coordination",
                label="Drawing Conflicts",
                description="Check for dimensional conflicts between drawings and specs",
                doc_scope=["DRAWING", "SPEC", "BOQ"],
                severity="high"
            ),
        ])
        
        # BoQ & Measurement Category
        items.extend([
            ChecklistItem(
                id="boq.measurement_method",
                category="BoQ & Measurement",
                subcategory="Measurement",
                label="Method of Measurement",
                description="Verify method of measurement referenced",
                doc_scope=["BOQ", "SPEC"],
                severity="medium"
            ),
            ChecklistItem(
                id="boq.provisional_sums",
                category="BoQ & Measurement",
                subcategory="Provisional Items",
                label="Provisional Sums",
                description="Identify provisional sums and NMR items",
                doc_scope=["BOQ"],
                severity="medium"
            ),
            ChecklistItem(
                id="boq.unit_consistency",
                category="BoQ & Measurement",
                subcategory="Units",
                label="Unit Consistency",
                description="Check unit consistency across BoQ, Spec, and Drawings",
                doc_scope=["BOQ", "SPEC", "DRAWING"],
                severity="high"
            ),
        ])
        
        # NIT Essentials Category
        items.extend([
            ChecklistItem(
                id="nit.bid_dates",
                category="NIT Essentials",
                subcategory="Key Dates",
                label="Bid Submission Dates",
                description="Verify bid submission and opening dates",
                doc_scope=["NIT"],
                severity="critical"
            ),
            ChecklistItem(
                id="nit.eligibility",
                category="NIT Essentials",
                subcategory="Eligibility",
                label="Eligibility Criteria",
                description="Check turnover, experience, and JV requirements",
                doc_scope=["NIT"],
                severity="critical"
            ),
        ])
        
        # Dispute Resolution Category
        items.extend([
            ChecklistItem(
                id="dispute.mechanism",
                category="Dispute Resolution",
                subcategory="Resolution Process",
                label="Dispute Resolution Mechanism",
                description="Check arbitration/adjudication clauses and governing law",
                doc_scope=["GCC", "ACC"],
                severity="high"
            ),
        ])
        
        # HSE & Environment Category
        items.extend([
            ChecklistItem(
                id="hse.safety_plan",
                category="HSE & Environment",
                subcategory="Safety",
                label="Safety Plan Requirements",
                description="Verify safety plan and permit requirements",
                doc_scope=["GCC", "SPEC", "QUALITY"],
                severity="high"
            ),
            ChecklistItem(
                id="hse.environmental",
                category="HSE & Environment",
                subcategory="Environment",
                label="Environmental Compliance",
                description="Check environmental protection and waste management clauses",
                doc_scope=["GCC", "SPEC"],
                severity="medium"
            ),
        ])
        
        return items

# ===================== RAG System =====================

class AdvancedRAG:
    """Advanced RAG system for document analysis"""
    
    def __init__(self, openai_client, vector_db_path: str):
        self.client = openai_client
        self.vector_db_path = vector_db_path
        self.embedding_function = embedding_functions.OpenAIEmbeddingFunction(
            api_key=st.session_state.config.openai_api_key,
            model_name="text-embedding-3-small"
        )
        self.chroma_client = None
        self.collection = None
        
    def initialize_vector_store(self, collection_name: str = "tender_docs"):
        """Initialize ChromaDB vector store"""
        self.chroma_client = chromadb.PersistentClient(
            path=self.vector_db_path,
            settings=Settings(anonymized_telemetry=False)
        )
        
        # Create or get collection
        try:
            self.collection = self.chroma_client.create_collection(
                name=collection_name,
                embedding_function=self.embedding_function,
                metadata={"hnsw:space": "cosine"}
            )
        except:
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_function
            )
    
    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[Dict]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = ' '.join(chunk_words)
            chunks.append({
                'text': chunk_text,
                'start_idx': i,
                'end_idx': min(i + chunk_size, len(words))
            })
            
        return chunks
    
    def add_documents(self, documents: List[Document]):
        """Add documents to vector store"""
        for doc in documents:
            if not doc.content:
                continue
                
            chunks = self.chunk_text(
                doc.content,
                chunk_size=st.session_state.config.chunk_size,
                overlap=st.session_state.config.chunk_overlap
            )
            
            # Prepare data for ChromaDB
            ids = [f"{doc.id}_{i}" for i in range(len(chunks))]
            texts = [chunk['text'] for chunk in chunks]
            metadatas = [
                {
                    'doc_id': doc.id,
                    'doc_name': doc.name,
                    'doc_type': doc.doc_type,
                    'chunk_idx': i,
                    'start_idx': chunk['start_idx'],
                    'end_idx': chunk['end_idx']
                }
                for i, chunk in enumerate(chunks)
            ]
            
            # Add to collection
            if texts:
                self.collection.add(
                    ids=ids,
                    documents=texts,
                    metadatas=metadatas
                )
    
    def search_similar(self, query: str, filter_dict: Dict = None, n_results: int = 5) -> List[Dict]:
        """Search for similar documents"""
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results,
            where=filter_dict
        )
        
        if not results['documents'][0]:
            return []
        
        formatted_results = []
        for i in range(len(results['documents'][0])):
            formatted_results.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i] if 'distances' in results else 0
            })
            
        return formatted_results
    
    def extract_checklist_evidence(self, item: ChecklistItem, documents: List[Document]) -> Tuple[str, List[Evidence], Any, float]:
        """Extract evidence for a checklist item using RAG"""
        
        # Build search queries based on item
        search_queries = [
            item.label,
            item.description,
            ' '.join(item.doc_scope) + ' ' + item.label
        ]
        
        all_results = []
        for query in search_queries:
            # Filter by document type if specified
            filter_dict = None
            if item.doc_scope:
                filter_dict = {"doc_type": {"$in": item.doc_scope}}
            
            results = self.search_similar(query, filter_dict, n_results=3)
            all_results.extend(results)
        
        # Deduplicate results
        seen = set()
        unique_results = []
        for result in all_results:
            key = (result['metadata']['doc_id'], result['metadata']['chunk_idx'])
            if key not in seen:
                seen.add(key)
                unique_results.append(result)
        
        if not unique_results:
            return "missing", [], None, 0.0
        
        # Prepare context for LLM
        context = "\n\n".join([
            f"[Document: {r['metadata']['doc_name']}]\n{r['text']}"
            for r in unique_results[:5]
        ])
        
        # Create targeted extraction prompt
        extraction_prompt = f"""
        Analyze the following tender documents to find information about: {item.label}
        
        Description: {item.description}
        
        Context from documents:
        {context}
        
        Please extract:
        1. Status: Is this requirement clearly present? (found/missing/needs_review)
        2. Specific value or details found (if any)
        3. Exact quote from the document supporting your finding
        4. Confidence level (0-1)
        
        Respond in JSON format:
        {{
            "status": "found/missing/needs_review",
            "value": "extracted value or null",
            "quote": "exact quote from document or null",
            "confidence": 0.0-1.0,
            "explanation": "brief explanation"
        }}
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are an expert construction contract analyst. Extract specific information accurately."},
                    {"role": "user", "content": extraction_prompt}
                ],
                temperature=0.1,
                max_tokens=500,
                response_format={"type": "json_object"}
            )
            
            result = json.loads(response.choices[0].message.content)
            
            # Create evidence objects
            evidence_list = []
            for r in unique_results[:3]:
                evidence_list.append(Evidence(
                    doc_id=r['metadata']['doc_id'],
                    doc_name=r['metadata']['doc_name'],
                    page=0,  # Would need page extraction
                    text=r['text'][:500],
                    confidence=1 - r['distance']
                ))
            
            return result['status'], evidence_list, result.get('value'), result.get('confidence', 0.5)
            
        except Exception as e:
            st.error(f"Error extracting evidence: {e}")
            return "needs_review", [], None, 0.0

# ===================== Compliance Analyzer =====================

class ComplianceAnalyzer:
    """Main compliance analysis engine"""
    
    def __init__(self, openai_client):
        self.client = openai_client
        self.processor = DocumentProcessor(openai_client)
        self.rag = AdvancedRAG(openai_client, st.session_state.config.vector_db_path)
        self.checklist_library = ChecklistLibrary()
        
    def analyze_tender_package(self, uploaded_files) -> Tuple[List[Document], List[ChecklistItem], Dict]:
        """Main analysis pipeline"""
        
        # Step 1: Process documents
        st.info("📄 Processing documents...")
        documents = []
        
        progress_bar = st.progress(0)
        for idx, file in enumerate(uploaded_files):
            # Save uploaded file temporarily
            temp_path = Path(st.session_state.config.temp_upload_path) / file.name
            temp_path.parent.mkdir(exist_ok=True)
            
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            
            # Extract text
            text, pages, needed_ocr = self.processor.extract_text_from_pdf(str(temp_path))
            
            # Classify document
            doc_type = self.processor.classify_document(file.name, text)
            
            # Create document object
            doc = Document(
                id=hashlib.md5(file.name.encode()).hexdigest()[:8],
                name=file.name,
                doc_type=doc_type,
                path=str(temp_path),
                pages=pages,
                has_ocr=needed_ocr,
                content=text
            )
            documents.append(doc)
            
            progress_bar.progress((idx + 1) / len(uploaded_files))
        
        # Step 2: Initialize RAG system
        st.info("🔍 Building knowledge base...")
        self.rag.initialize_vector_store(f"tender_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.rag.add_documents(documents)
        
        # Step 3: Get checklist items
        checklist_items = self.checklist_library.get_checklist_items()
        
        # Step 4: Analyze each checklist item
        st.info("✅ Analyzing compliance...")
        progress_bar = st.progress(0)
        
        for idx, item in enumerate(checklist_items):
            # Skip if no relevant documents
            if item.doc_scope:
                relevant_docs = [d for d in documents if d.doc_type in item.doc_scope]
                if not relevant_docs:
                    item.status = "missing"
                    item.confidence = 0.0
                    progress_bar.progress((idx + 1) / len(checklist_items))
                    continue
            
            # Extract evidence
            status, evidence, value, confidence = self.rag.extract_checklist_evidence(item, documents)
            
            item.status = status
            item.evidence = [asdict(e) for e in evidence]
            item.value = value
            item.confidence = confidence
            
            # Generate suggestion if missing or needs review
            if status in ["missing", "needs_review"]:
                item.suggestion = self.generate_suggestion(item)
            
            progress_bar.progress((idx + 1) / len(checklist_items))
        
        # Step 5: Calculate metrics
        metrics = self.calculate_metrics(checklist_items)
        
        return documents, checklist_items, metrics
    
    def generate_suggestion(self, item: ChecklistItem) -> str:
        """Generate suggestion for missing or problematic items"""
        prompt = f"""
        The following checklist item is missing or needs review in a construction tender:
        
        Item: {item.label}
        Description: {item.description}
        Category: {item.category}
        
        Provide a brief, professional suggestion for standard contract language that addresses this requirement.
        Keep it concise (2-3 sentences) and practical.
        """
        
        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a construction contract expert. Provide practical contract language suggestions."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=150
            )
            return response.choices[0].message.content.strip()
        except:
            return "Consider adding standard clause for this requirement."
    
    def calculate_metrics(self, checklist_items: List[ChecklistItem]) -> Dict:
        """Calculate compliance metrics"""
        total = len(checklist_items)
        found = sum(1 for item in checklist_items if item.status == "found")
        missing = sum(1 for item in checklist_items if item.status == "missing")
        review = sum(1 for item in checklist_items if item.status in ["needs_review", "conflict"])
        
        critical_items = [item for item in checklist_items if item.severity == "critical"]
        critical_missing = sum(1 for item in critical_items if item.status == "missing")
        
        return {
            'total_items': total,
            'found': found,
            'missing': missing,
            'needs_review': review,
            'compliance_score': round((found / total) * 100, 1) if total > 0 else 0,
            'critical_missing': critical_missing,
            'categories': self.get_category_metrics(checklist_items)
        }
    
    def get_category_metrics(self, checklist_items: List[ChecklistItem]) -> Dict[str, Dict]:
        """Get metrics by category"""
        categories = defaultdict(lambda: {'total': 0, 'found': 0, 'missing': 0, 'review': 0})
        
        for item in checklist_items:
            cat = item.category
            categories[cat]['total'] += 1
            if item.status == "found":
                categories[cat]['found'] += 1
            elif item.status == "missing":
                categories[cat]['missing'] += 1
            else:
                categories[cat]['review'] += 1
        
        return dict(categories)

# ===================== UI Components =====================

def render_header():
    """Render application header"""
    st.markdown("""
    <div class="main-header">
        <h1 style="margin: 0; font-size: 2.5rem;">📋 Construction Tender Compliance Checker</h1>
        <p style="margin: 0.5rem 0 0 0; opacity: 0.9;">Automated compliance analysis for construction tender documents</p>
    </div>
    """, unsafe_allow_html=True)

def render_metrics(metrics: Dict):
    """Render compliance metrics"""
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #667eea; margin: 0;">Compliance Score</h3>
            <h1 style="margin: 0.5rem 0;">{metrics['compliance_score']}%</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #10b981; margin: 0;">Found</h3>
            <h1 style="margin: 0.5rem 0;">{metrics['found']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #ef4444; margin: 0;">Missing</h3>
            <h1 style="margin: 0.5rem 0;">{metrics['missing']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #f59e0b; margin: 0;">Review</h3>
            <h1 style="margin: 0.5rem 0;">{metrics['needs_review']}</h1>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <h3 style="color: #dc2626; margin: 0;">Critical Missing</h3>
            <h1 style="margin: 0.5rem 0;">{metrics['critical_missing']}</h1>
        </div>
        """, unsafe_allow_html=True)

def render_checklist_item(item: ChecklistItem):
    """Render individual checklist item"""
    status_colors = {
        'found': '#10b981',
        'missing': '#ef4444',
        'needs_review': '#f59e0b',
        'conflict': '#dc2626',
        'pending': '#9ca3af'
    }
    
    status_icons = {
        'found': '✅',
        'missing': '❌',
        'needs_review': '⚠️',
        'conflict': '🔴',
        'pending': '⏳'
    }
    
    with st.container():
        col1, col2, col3 = st.columns([0.5, 3, 1])
        
        with col1:
            st.markdown(f"<h2>{status_icons[item.status]}</h2>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div style="padding: 0.5rem 0;">
                <h4 style="margin: 0; color: #1f2937;">{item.label}</h4>
                <p style="margin: 0.25rem 0; color: #6b7280; font-size: 0.9rem;">{item.description}</p>
                <div style="margin-top: 0.5rem;">
                    <span style="background: {status_colors[item.status]}; color: white; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.85rem;">
                        {item.status.upper()}
                    </span>
                    <span style="background: #e5e7eb; color: #4b5563; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.85rem; margin-left: 0.5rem;">
                        Severity: {item.severity}
                    </span>
                    <span style="background: #e5e7eb; color: #4b5563; padding: 0.25rem 0.5rem; border-radius: 4px; font-size: 0.85rem; margin-left: 0.5rem;">
                        Confidence: {item.confidence:.0%}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if st.button("View Details", key=f"view_{item.id}"):
                st.session_state[f"expand_{item.id}"] = not st.session_state.get(f"expand_{item.id}", False)
        
        # Expandable details
        if st.session_state.get(f"expand_{item.id}", False):
            st.markdown("---")
            
            # Evidence
            if item.evidence:
                st.markdown("**📄 Evidence Found:**")
                for evidence in item.evidence[:2]:
                    st.markdown(f"""
                    <div class="evidence-box">
                        <strong>Document:</strong> {evidence.get('doc_name', 'Unknown')}<br>
                        <strong>Extract:</strong> {evidence.get('text', '')[:300]}...
                    </div>
                    """, unsafe_allow_html=True)
            
            # Value extracted
            if item.value:
                st.markdown(f"**📊 Extracted Value:** `{item.value}`")
            
            # Suggestion
            if item.suggestion and item.status in ["missing", "needs_review"]:
                st.markdown("**💡 Suggested Action:**")
                st.info(item.suggestion)

def render_category_analysis(checklist_items: List[ChecklistItem], metrics: Dict):
    """Render category-wise analysis"""
    st.markdown("### 📊 Category-wise Analysis")
    
    categories = {}
    for item in checklist_items:
        if item.category not in categories:
            categories[item.category] = []
        categories[item.category].append(item)
    
    for category, items in categories.items():
        cat_metrics = metrics['categories'].get(category, {})
        
        with st.expander(
            f"{category} ({cat_metrics.get('found', 0)}/{cat_metrics.get('total', 0)} compliant)",
            expanded=False
        ):
            for item in items:
                render_checklist_item(item)

def export_results(documents: List[Document], checklist_items: List[ChecklistItem], metrics: Dict):
    """Export results to various formats"""
    st.markdown("### 📥 Export Results")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Excel export
        if st.button("📊 Export to Excel", type="primary", use_container_width=True):
            df_data = []
            for item in checklist_items:
                df_data.append({
                    'Category': item.category,
                    'Subcategory': item.subcategory,
                    'Item': item.label,
                    'Status': item.status,
                    'Severity': item.severity,
                    'Confidence': f"{item.confidence:.0%}",
                    'Value': str(item.value) if item.value else '',
                    'Suggestion': item.suggestion if item.suggestion else ''
                })
            
            df = pd.DataFrame(df_data)
            excel_buffer = io.BytesIO()
            df.to_excel(excel_buffer, index=False, engine='openpyxl')
            excel_buffer.seek(0)
            
            st.download_button(
                label="Download Excel Report",
                data=excel_buffer,
                file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )
    
    with col2:
        # JSON export
        if st.button("📄 Export to JSON", type="secondary", use_container_width=True):
            export_data = {
                'timestamp': datetime.now().isoformat(),
                'metrics': metrics,
                'documents': [asdict(d) for d in documents],
                'checklist_items': [asdict(item) for item in checklist_items]
            }
            
            json_str = json.dumps(export_data, indent=2, default=str)
            st.download_button(
                label="Download JSON Report",
                data=json_str,
                file_name=f"compliance_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )
    
    with col3:
        # Summary PDF (placeholder - would need additional library)
        if st.button("📑 Generate Summary", type="secondary", use_container_width=True):
            summary = f"""
            TENDER COMPLIANCE REPORT
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
            
            EXECUTIVE SUMMARY
            ================
            Compliance Score: {metrics['compliance_score']}%
            Total Items Checked: {metrics['total_items']}
            Items Found Compliant: {metrics['found']}
            Items Missing: {metrics['missing']}
            Items Needing Review: {metrics['needs_review']}
            Critical Items Missing: {metrics['critical_missing']}
            
            DOCUMENTS ANALYZED
            ==================
            """
            
            for doc in documents:
                summary += f"\n- {doc.name} (Type: {doc.doc_type}, Pages: {doc.pages})"
            
            summary += "\n\nCRITICAL ISSUES\n===============\n"
            
            critical_issues = [item for item in checklist_items 
                             if item.severity == "critical" and item.status == "missing"]
            
            for item in critical_issues:
                summary += f"\n❌ {item.label}\n   {item.description}\n"
            
            st.download_button(
                label="Download Summary",
                data=summary,
                file_name=f"compliance_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain"
            )

# ===================== Main Application =====================

def main():
    """Main application entry point"""
    
    # Initialize session state
    if 'config' not in st.session_state:
        st.session_state.config = Config()
    
    if 'analysis_complete' not in st.session_state:
        st.session_state.analysis_complete = False
    
    # Render header
    render_header()
    
    # Sidebar configuration
    with st.sidebar:
        st.markdown("## ⚙️ Configuration")
        
        # API Key input
        api_key = st.text_input(
            "OpenAI API Key",
            type="password",
            value=st.session_state.config.openai_api_key,
            help="Enter your OpenAI API key"
        )
        
        if api_key:
            st.session_state.config.openai_api_key = api_key
        
        st.markdown("---")
        
        # Advanced settings
        with st.expander("Advanced Settings"):
            st.session_state.config.chunk_size = st.slider(
                "Chunk Size",
                min_value=500,
                max_value=2000,
                value=st.session_state.config.chunk_size,
                step=100
            )
            
            st.session_state.config.chunk_overlap = st.slider(
                "Chunk Overlap",
                min_value=50,
                max_value=500,
                value=st.session_state.config.chunk_overlap,
                step=50
            )
            
            st.session_state.config.temperature = st.slider(
                "AI Temperature",
                min_value=0.0,
                max_value=1.0,
                value=st.session_state.config.temperature,
                step=0.1
            )
        
        st.markdown("---")
        
        # Info section
        st.markdown("""
        ### 📖 How to Use
        1. Enter your OpenAI API key
        2. Upload tender documents
        3. Click 'Analyze Documents'
        4. Review compliance results
        5. Export reports as needed
        
        ### 📄 Supported Documents
        - Notice Inviting Tender (NIT)
        - General/Special Conditions
        - Technical Specifications
        - Bill of Quantities (BoQ)
        - Drawings
        - Quality Plans
        - Make Lists
        """)
    
    # Main content area
    if not st.session_state.config.openai_api_key:
        st.warning("⚠️ Please enter your OpenAI API key in the sidebar to continue.")
        st.stop()
    
    # File upload section
    st.markdown("### 📁 Upload Tender Documents")
    
    uploaded_files = st.file_uploader(
        "Select tender package documents",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload all tender documents including NIT, GCC, Specifications, BoQ, Drawings, etc."
    )
    
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} documents uploaded")
        
        # Display uploaded files
        with st.expander("View uploaded documents"):
            for file in uploaded_files:
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.text(f"📄 {file.name}")
                with col2:
                    st.text(f"{file.size / 1024:.1f} KB")
        
        # Analyze button
        if st.button("🔍 Analyze Documents", type="primary", use_container_width=True):
            try:
                # Initialize OpenAI client
                client = OpenAI(api_key=st.session_state.config.openai_api_key)
                
                # Initialize analyzer
                analyzer = ComplianceAnalyzer(client)
                
                # Run analysis
                with st.spinner("Analyzing tender documents... This may take a few minutes."):
                    documents, checklist_items, metrics = analyzer.analyze_tender_package(uploaded_files)
                
                # Store results in session state
                st.session_state.documents = documents
                st.session_state.checklist_items = checklist_items
                st.session_state.metrics = metrics
                st.session_state.analysis_complete = True
                
                st.success("✅ Analysis complete!")
                st.balloons()
                
            except Exception as e:
                st.error(f"❌ Analysis failed: {str(e)}")
                st.stop()
    
    # Display results if analysis is complete
    if st.session_state.analysis_complete:
        st.markdown("---")
        st.markdown("## 📊 Compliance Analysis Results")
        
        # Display metrics
        render_metrics(st.session_state.metrics)
        
        st.markdown("---")
        
        # Filter options
        col1, col2, col3 = st.columns(3)
        
        with col1:
            status_filter = st.multiselect(
                "Filter by Status",
                options=['found', 'missing', 'needs_review', 'conflict'],
                default=['missing', 'needs_review']
            )
        
        with col2:
            severity_filter = st.multiselect(
                "Filter by Severity",
                options=['critical', 'high', 'medium', 'low'],
                default=['critical', 'high']
            )
        
        with col3:
            category_filter = st.multiselect(
                "Filter by Category",
                options=list(set(item.category for item in st.session_state.checklist_items)),
                default=[]
            )
        
        # Apply filters
        filtered_items = st.session_state.checklist_items
        
        if status_filter:
            filtered_items = [item for item in filtered_items if item.status in status_filter]
        
        if severity_filter:
            filtered_items = [item for item in filtered_items if item.severity in severity_filter]
        
        if category_filter:
            filtered_items = [item for item in filtered_items if item.category in category_filter]
        
        # Display filtered results
        st.markdown(f"### 🔍 Showing {len(filtered_items)} items")
        
        # Category-wise display
        render_category_analysis(filtered_items, st.session_state.metrics)
        
        st.markdown("---")
        
        # Export section
        export_results(
            st.session_state.documents,
            st.session_state.checklist_items,
            st.session_state.metrics
        )
        
        # Clear analysis button
        st.markdown("---")
        if st.button("🔄 Start New Analysis", type="secondary"):
            st.session_state.analysis_complete = False
            st.session_state.documents = []
            st.session_state.checklist_items = []
            st.session_state.metrics = {}
            st.rerun()

if __name__ == "__main__":
    main()
