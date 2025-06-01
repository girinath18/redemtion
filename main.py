import os
import json
import time
from typing import Dict, Any, List, Optional
import google.generativeai as genai
from jsonschema import validate, ValidationError
import logging
from pathlib import Path
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Form, Header, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedTextToJSONMapper:
    def __init__(self, api_key: str, model_name: str = "gemini-1.5-flash"):
        """
        Initialize the Enhanced TextToJSONMapper with Gemini API
        """
        if not api_key or api_key == "GOOGLE_AI_API_KEY":
            raise ValueError("Please provide a valid Google AI API key")
        
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel(model_name)
        
        # Configure generation parameters for JSON output
        self.generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            top_p=0.9,
            top_k=1,
            max_output_tokens=8192,
            response_mime_type="application/json"
        )
    
    def load_text_file(self, file_path: str) -> str:
        """Load content from text file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            logger.info(f"Successfully loaded {len(content)} characters from {file_path}")
            return content
        except FileNotFoundError:
            logger.error(f"File not found: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {file_path}: {str(e)}")
            raise
    
    def clean_json_response(self, response_text: str) -> str:
        """Clean and fix common JSON issues in the response"""
        try:
            # Remove any markdown code blocks
            if '```json' in response_text:
                response_text = response_text.split('```json')[1].split('```')[0]
            elif '```' in response_text:
                response_text = response_text.split('```')[1].split('```')[0]
            
            # Remove any text before the first {
            start_idx = response_text.find('{')
            if start_idx > 0:
                response_text = response_text[start_idx:]
            
            # Remove any text after the last }
            end_idx = response_text.rfind('}')
            if end_idx > 0:
                response_text = response_text[:end_idx + 1]
            
            response_text = response_text.strip()
            return response_text
            
        except Exception as e:
            logger.warning(f"Error cleaning JSON response: {str(e)}")
            return response_text
    
    def intelligent_chunk_text(self, text: str, max_chars: int = 15000) -> List[str]:
        """
        Intelligently split text into chunks, preserving item boundaries
        """
        if len(text) <= max_chars:
            return [text]
        
        chunks = []
        
        # Try to find natural break points (like item separators)
        # Look for patterns that might indicate new items
        item_patterns = [
    '\n\nItem',
    '\nS.No',
    '\nSL NO',
    r'\n\d+\.',  # numbered items
    '\nDescription:',
    '\nProduct:',
    '\nHSN',
    '\nQuantity:'
]
        
        # Split by lines first
        lines = text.split('\n')
        current_chunk = ""
        
        for i, line in enumerate(lines):
            # Check if adding this line would exceed the limit
            if len(current_chunk + line + '\n') > max_chars:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = line + '\n'
                else:
                    # Single line is too long, force split
                    chunks.append(line[:max_chars])
                    current_chunk = line[max_chars:] + '\n'
            else:
                current_chunk += line + '\n'
        
        if current_chunk.strip():
            chunks.append(current_chunk.strip())
        
        logger.info(f"Split text into {len(chunks)} chunks")
        return chunks
    
    def create_comprehensive_prompt(self, text_content: str, chunk_index: int = 0, total_chunks: int = 1) -> str:
        """Create a comprehensive prompt that doesn't truncate text"""
        
        chunk_info = f" (Chunk {chunk_index + 1} of {total_chunks})" if total_chunks > 1 else ""
        
        prompt = f"""Extract ALL invoice/export document data{chunk_info} from the text and return ONLY valid JSON in this exact format:

{{
  "Header": [
    {{
      "Branch": null,
      "BranchCode": "",
      "BuyerName": null,
      "BuyerOrderNo": null,
      "CartonNo": null,
      "Client": "",
      "Commission": null,
      "CommonItem": null,
      "CompanyID": null,
      "ConsigneeAdd1": "",
      "ConsigneeAdd2": "",
      "ConsigneeAdd3": "",
      "ConsigneeAdd4": "",
      "ConsigneeName": "",
      "CountryOfDischarge": "",
      "Currency": "",
      "Discount": null,
      "ExporterName": "",
      "FileName": null,
      "Freight": null,
      "GarmentsIGSTPercentage": null,
      "HangerIGSTPercentage": null,
      "IEC": "",
      "Insurance": null,
      "InvoiceDate": "",
      "InvoiceNo": "",
      "InvoiceResponseTime": "",
      "InvoiceStartTime": "",
      "InvoiceValue": "",
      "JobNo": null,
      "JobStatus": null,
      "JobType": "",
      "KimballNo": null,
      "NotifyAddress1": null,
      "NotifyAddress2": null,
      "NotifyAddress3": null,
      "NotifyPartyName": null,
      "OrderNo": null,
      "OtherDed": null,
      "OtherReference": null,
      "PageCount": null,
      "PaymentPeriod": "",
      "PdfClientName": "",
      "PdfCount": null,
      "PortOfDischarge": "",
      "PortOfFinalDestination": null,
      "QtyCode": null,
      "SchemeCode": null,
      "Status": "",
      "Terms": null,
      "TermsOfPayment": "",
      "TotalCBM": null,
      "TotalCarton": null,
      "TotalGrossWeight": null,
      "TotalNetWeight": null,
      "UserID": "",
      "WorkingPeriod": null
    }}
  ],
  "ItemsDetails": [
    {{
      "Amount": "",
      "ExtraItemDesc": null,
      "ExtraQuantity": null,
      "HSNCode": "",
      "IGSTAmount": null,
      "InfoQty": null,
      "InfoUnitPrice": null,
      "ItemCountry": "",
      "ItemDesc": "",
      "ItemQTYCode": null,
      "Itemslno": null,
      "NetWeight": null,
      "Quantity": "",
      "Rate": "",
      "TaxableAmount": null
    }}
  ]
}}

CRITICAL INSTRUCTIONS:
- Extract ALL items found in the text, do not skip any
- For multi-chunk processing, focus on items in this chunk
- Use NULL for missing string values
- Use null for missing numeric/optional values
- Ensure proper JSON syntax with double quotes
- Return ONLY the JSON, no explanations
- For any kind of amount in invoice use USD instead of INR
- PaymentPeriod No need to print entire strings only the integer
- For the PdfClientName use the which company's invoice is about 
- For terms of payment use the Incoterms value from the invoice
- HS Code should be 8 digits
- For ItemCountry use the  Country of Origin in the invoice, mapp correctly item country of origin to the correct item beacuse the alignment is important may confusing
- For Itemslno use the Sr.No. of the item in the invoice

Text to analyze:
{text_content}

JSON:"""
        return prompt
    
    def merge_json_results(self, json_results: List[Dict[Any, Any]]) -> Dict[Any, Any]:
        """
        Merge multiple JSON results from chunked processing
        """
        if not json_results:
            return None
        
        if len(json_results) == 1:
            return json_results[0]
        
        # Start with the first result
        merged_result = json_results[0].copy()
        
        # Merge additional chunks
        for result in json_results[1:]:
            if "ItemsDetails" in result and result["ItemsDetails"]:
                # Extend items from subsequent chunks
                merged_result["ItemsDetails"].extend(result["ItemsDetails"])
            
            # Update header if it has more complete information
            if "Header" in result and result["Header"]:
                for key, value in result["Header"][0].items():
                    # Fixed: Check if value exists and is not None, and handle both string and non-string values
                    if value is not None and str(value).strip() and not merged_result["Header"][0].get(key):
                        merged_result["Header"][0][key] = value
        
        logger.info(f"Merged {len(json_results)} chunks into final result with {len(merged_result.get('ItemsDetails', []))} items")
        return merged_result
    
    def process_single_chunk(self, text_content: str, json_schema: Dict[Any, Any], 
                           chunk_index: int = 0, total_chunks: int = 1, 
                           max_retries: int = 3) -> Optional[Dict[Any, Any]]:
        """
        Process a single chunk of text
        """
        prompt = self.create_comprehensive_prompt(text_content, chunk_index, total_chunks)
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Processing chunk {chunk_index + 1}/{total_chunks}, attempt {attempt + 1}")
                
                response = self.model.generate_content(
                    prompt,
                    generation_config=self.generation_config
                )
                
                if not response.text:
                    logger.warning(f"Empty response for chunk {chunk_index + 1}, attempt {attempt + 1}")
                    continue
                
                # Clean and parse the response
                clean_response = self.clean_json_response(response.text)
                
                try:
                    json_data = json.loads(clean_response)
                    
                    # Validate structure
                    if isinstance(json_data, dict) and "Header" in json_data and "ItemsDetails" in json_data:
                        items_count = len(json_data.get("ItemsDetails", []))
                        logger.info(f"Successfully processed chunk {chunk_index + 1} with {items_count} items")
                        return json_data
                    else:
                        logger.warning(f"Invalid JSON structure for chunk {chunk_index + 1}")
                        
                except json.JSONDecodeError as e:
                    logger.error(f"JSON parsing error for chunk {chunk_index + 1}: {str(e)}")
                    logger.debug(f"Clean response preview: {clean_response[:500]}...")
                
            except Exception as e:
                logger.error(f"API request failed for chunk {chunk_index + 1}: {str(e)}")
                
            # Wait before retry
            if attempt < max_retries - 1:
                wait_time = (attempt + 1) * 2
                logger.info(f"Waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
        
        logger.error(f"Failed to process chunk {chunk_index + 1} after all attempts")
        return None
    
    def map_text_to_json_comprehensive(self, text_content: str, json_schema: Dict[Any, Any]) -> Optional[Dict[Any, Any]]:
        """
        Comprehensive mapping that handles large texts by chunking
        """
        # Determine if we need to chunk the text
        max_single_chunk_size = 15000  # Increased from 6000
        
        if len(text_content) <= max_single_chunk_size:
            # Process as single chunk
            logger.info("Processing as single chunk")
            return self.process_single_chunk(text_content, json_schema)
        else:
            # Process in multiple chunks
            logger.info(f"Text is {len(text_content)} characters, chunking required")
            chunks = self.intelligent_chunk_text(text_content, max_single_chunk_size)
            
            chunk_results = []
            for i, chunk in enumerate(chunks):
                result = self.process_single_chunk(chunk, json_schema, i, len(chunks))
                if result:
                    chunk_results.append(result)
                
                # Add delay between chunks
                if i < len(chunks) - 1:
                    time.sleep(2)
            
            if chunk_results:
                return self.merge_json_results(chunk_results)
            else:
                logger.error("All chunks failed to process")
                return None
    
    def process_file_comprehensive(self, input_file: str, output_file: str, json_schema: Dict[Any, Any]) -> bool:
        """
        Comprehensive file processing that handles large files
        """
        try:
            # Load text content
            text_content = self.load_text_file(input_file)
            logger.info(f"Loaded file with {len(text_content)} characters")
            
            # Map to JSON using comprehensive method
            mapped_data = self.map_text_to_json_comprehensive(text_content, json_schema)
            
            if mapped_data is None:
                logger.error("Failed to map text to JSON")
                return False
            
            # Save output
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(mapped_data, f, indent=2, ensure_ascii=False)
            
            # Log statistics
            total_items = len(mapped_data.get('ItemsDetails', []))
            logger.info(f"Successfully saved {total_items} items to {output_file}")
            return True
            
        except Exception as e:
            logger.error(f"Processing failed: {str(e)}")
            return False


class PDFProcessor:
    """Process PDF files to extract text using LlamaIndex"""
    
    def __init__(self, groq_api_key: str):
        """Initialize PDF processor with GROQ API key"""
        self.groq_api_key = groq_api_key
        os.environ["GROQ_API_KEY"] = groq_api_key
        
        # Import and setup LlamaIndex components
        try:
            from llama_index.core import SimpleDirectoryReader
            from llama_index.core import Settings
            from llama_index.llms.groq import Groq
            from llama_index.embeddings.huggingface import HuggingFaceEmbedding
            
            # Setup LLM and embeddings
            llm = Groq(model="llama-3.3-70b-versatile")
            Settings.llm = llm
            Settings.embed_model = HuggingFaceEmbedding()
            
            self.SimpleDirectoryReader = SimpleDirectoryReader
            logger.info("PDF processor initialized successfully")
            
        except ImportError as e:
            logger.error(f"Failed to import LlamaIndex components: {str(e)}")
            raise
    
    def process_pdf_to_text(self, pdf_path: str, output_text_path: str) -> bool:
        """Process PDF file and extract text"""
        try:
            # Check if PDF file exists
            if not os.path.exists(pdf_path):
                logger.error(f"PDF file not found: {pdf_path}")
                return False
            
            # Load the document
            documents = self.SimpleDirectoryReader(input_files=[pdf_path]).load_data()
            logger.info(f"Loaded {len(documents)} document(s) from PDF")
            
            # Create a single text file from the loaded documents
            with open(output_text_path, "w", encoding="utf-8") as f:
                for doc in documents:
                    f.write(doc.text)
                    f.write("\n")  # Add a newline between documents for clarity
            
            logger.info(f"Successfully extracted text to {output_text_path}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to process PDF: {str(e)}")
            return False


def setup_directories():
    """Create necessary directories if they don't exist"""
    directories = ["input", "output", "temp"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
    logger.info("Directories setup completed")


app = FastAPI()

# Allow CORS for testing (optional, can be removed in prod)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

AUTH_TOKEN = "A9q34SbyLnknR7fhwUKPDI1OSLTit5Xu"

@app.post("/process-invoice")
async def process_invoice(
    file: UploadFile = File(...),
    job_type: str = Form(...),
    authorization: str = Header(None)
):
    # Auth check
    if authorization != AUTH_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid token.")

    # Validate job_type
    if job_type not in ["export", "import"]:
        raise HTTPException(status_code=400, detail="Invalid job_type. Must be 'export' or 'import'.")

    # Save uploaded PDF to temp dir
    try:
        setup_directories()
        pdf_path = f"temp/{file.filename}"
        with open(pdf_path, "wb") as f:
            f.write(await file.read())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save uploaded file: {str(e)}")

    # Set up environment variables
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    GOOGLE_AI_API_KEY = os.getenv("GOOGLE_AI_API_KEY")
    if not GROQ_API_KEY or not GOOGLE_AI_API_KEY:
        raise HTTPException(status_code=500, detail="Missing GROQ_API_KEY or GOOGLE_AI_API_KEY in environment.")

    # File paths
    TEXT_FILE = f"temp/{file.filename}.txt"
    OUTPUT_FILE = f"temp/{file.filename}.json"

    # Step 1: Extract text from PDF
    try:
        pdf_processor = PDFProcessor(GROQ_API_KEY)
        if not pdf_processor.process_pdf_to_text(pdf_path, TEXT_FILE):
            raise HTTPException(status_code=500, detail="Failed to extract text from PDF.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PDF processing error: {str(e)}")

    # Step 2: Process text to JSON
    try:
        INVOICE_SCHEMA = {
            "type": "object",
            "properties": {
                "Header": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Branch": {"type": ["string", "null"]},
                            "BranchCode": {"type": "string"},
                            "BuyerName": {"type": ["string", "null"]},
                            "BuyerOrderNo": {"type": ["string", "null"]},
                            "CartonNo": {"type": ["string", "null"]},
                            "Client": {"type": "string"},
                            "Commission": {"type": ["number", "null"]},
                            "CommonItem": {"type": ["string", "null"]},
                            "CompanyID": {"type": ["string", "null"]},
                            "ConsigneeAdd1": {"type": "string"},
                            "ConsigneeAdd2": {"type": "string"},
                            "ConsigneeAdd3": {"type": "string"},
                            "ConsigneeAdd4": {"type": "string"},
                            "ConsigneeName": {"type": "string"},
                            "CountryOfDischarge": {"type": "string"},
                            "Currency": {"type": "string"},
                            "Discount": {"type": ["number", "null"]},
                            "ExporterName": {"type": "string"},
                            "FileName": {"type": ["string", "null"]},
                            "Freight": {"type": ["number", "null"]},
                            "GarmentsIGSTPercentage": {"type": ["number", "null"]},
                            "HangerIGSTPercentage": {"type": ["number", "null"]},
                            "IEC": {"type": "string"},
                            "Insurance": {"type": ["number", "null"]},
                            "InvoiceDate": {"type": "string"},
                            "InvoiceNo": {"type": "string"},
                            "InvoiceResponseTime": {"type": "string"},
                            "InvoiceStartTime": {"type": "string"},
                            "InvoiceValue": {"type": "string"},
                            "JobNo": {"type": ["string", "null"]},
                            "JobStatus": {"type": ["string", "null"]},
                            "JobType": {"type": "string"},
                            "KimballNo": {"type": ["string", "null"]},
                            "NotifyAddress1": {"type": ["string", "null"]},
                            "NotifyAddress2": {"type": ["string", "null"]},
                            "NotifyAddress3": {"type": ["string", "null"]},
                            "NotifyPartyName": {"type": ["string", "null"]},
                            "OrderNo": {"type": ["string", "null"]},
                            "OtherDed": {"type": ["number", "null"]},
                            "OtherReference": {"type": ["string", "null"]},
                            "PageCount": {"type": ["number", "null"]},
                            "PaymentPeriod": {"type": "string"},
                            "PdfClientName": {"type": "string"},
                            "PdfCount": {"type": ["number", "null"]},
                            "PortOfDischarge": {"type": "string"},
                            "PortOfFinalDestination": {"type": ["string", "null"]},
                            "QtyCode": {"type": ["string", "null"]},
                            "SchemeCode": {"type": ["string", "null"]},
                            "Status": {"type": "string"},
                            "Terms": {"type": ["string", "null"]},
                            "TermsOfPayment": {"type": "string"},
                            "TotalCBM": {"type": ["number", "null"]},
                            "TotalCarton": {"type": ["number", "null"]},
                            "TotalGrossWeight": {"type": ["number", "null"]},
                            "TotalNetWeight": {"type": ["number", "null"]},
                            "UserID": {"type": "string"},
                            "WorkingPeriod": {"type": ["string", "null"]}
                        },
                        "required": []
                    }
                },
                "ItemsDetails": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "Amount": {"type": "string"},
                            "ExtraItemDesc": {"type": ["string", "null"]},
                            "ExtraQuantity": {"type": ["number", "null"]},
                            "HSNCode": {"type": "string"},
                            "IGSTAmount": {"type": ["number", "null"]},
                            "InfoQty": {"type": ["number", "null"]},
                            "InfoUnitPrice": {"type": ["number", "null"]},
                            "ItemCountry": {"type": "string"},
                            "ItemDesc": {"type": "string"},
                            "ItemQTYCode": {"type": ["string", "null"]},
                            "Itemslno": {"type": ["number", "null"]},
                            "NetWeight": {"type": ["number", "null"]},
                            "Quantity": {"type": "string"},
                            "Rate": {"type": "string"},
                            "TaxableAmount": {"type": ["number", "null"]}
                        },
                        "required": []
                    }
                }
            },
            "required": ["Header", "ItemsDetails"]
        }
        mapper = EnhancedTextToJSONMapper(api_key=GOOGLE_AI_API_KEY)
        success = mapper.process_file_comprehensive(TEXT_FILE, OUTPUT_FILE, INVOICE_SCHEMA)
        if not success:
            raise HTTPException(status_code=500, detail="Failed to process invoice text to JSON.")
        with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
            result = json.load(f)
        return JSONResponse(content=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Text-to-JSON processing error: {str(e)}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "api":
        uvicorn.run("main:app", host="0.0.0.0", port=8777, reload=True)
    else:
        main()