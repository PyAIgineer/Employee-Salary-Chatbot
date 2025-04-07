import os
import re
import json
import shutil
import tempfile
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader, Docx2txtLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import logging
import watchdog.observers
import watchdog.events
from pathlib import Path
import time

load_dotenv()

app = FastAPI()

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
db_connection = None
chat_histories = {}  # Store chat history per user
vector_stores = {}  # Store vector database per user
current_employee_registry = {}  # Store employee_id, name, email mappings

class DatabaseConfig(BaseModel):
    host: str
    port: str
    user: str
    password: str
    database: str

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    employee_id: str

class DocumentQuery(BaseModel):
    query: str
    employee_id: str

class EmployeeVerification(BaseModel):
    employee_id: str

class EmployeeInfo(BaseModel):
    uuid: str
    name: Optional[str] = None
    email: Optional[str] = None

def init_database(config: DatabaseConfig) -> SQLDatabase:
    db_uri = f"mysql+mysqlconnector://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db):
    template = """
    You are a salary query assistant for employees. You are interacting with a user who is asking you questions about their salary information.
    Based on the table schema below, write a SQL query that would answer the user's question.
    Only return information for the employee with ID '{employee_id}'. Always include a filter for employee_id in your SQL queries.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: Why was my salary lower in February?
    SQL Query: SELECT * FROM salary_records WHERE employee_id = '{employee_id}' AND month = 'February' ORDER BY date DESC LIMIT 1;
    
    Question: What deductions were applied to my salary last month?
    SQL Query: SELECT deduction_type, amount FROM salary_deductions WHERE employee_id = '{employee_id}' AND date >= DATE_SUB(CURDATE(), INTERVAL 1 MONTH) ORDER BY date DESC;
    
    Your turn:
    
    Question: {question}
    SQL Query:
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize ChatGroq with supported model
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
    
    def get_schema(_):
        return db.get_table_info()
    
    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def get_response(user_query: str, db: SQLDatabase, employee_id: str):
    # Get user-specific chat history
    chat_history = chat_histories.get(employee_id, [])
    
    sql_chain = get_sql_chain(db)
    
    # Convert chat history to format expected by the chain
    formatted_history = []
    for msg in chat_history:
        if msg.role == "ai":
            formatted_history.append(AIMessage(content=msg.content))
        else:
            formatted_history.append(HumanMessage(content=msg.content))
    
    template = """
    You are a salary query assistant for employees. You help employees understand their salary information.
    Based on the schema, query, and SQL response below, provide a clear explanation of the salary information.
    
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
    
    Provide a helpful, informative response that explains the salary information in simple terms.
    If the data shows variations in salary, explain possible reasons (like deductions, bonuses, leaves taken, etc.).
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize ChatGroq with supported model
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
    
    chain = (
        RunnablePassthrough.assign(
            query=lambda x: sql_chain.invoke({
                "question": x["question"],
                "chat_history": formatted_history,
                "employee_id": employee_id
            })
        ).assign(
            schema=lambda _: db.get_table_info(),
            response=lambda vars: db.run(vars["query"]),
        )
        | prompt
        | llm
        | StrOutputParser()
    )
    
    return chain.invoke({
        "question": user_query,
        "chat_history": formatted_history,
        "employee_id": employee_id
    })

# Load employee registry from JSON file or create it if it doesn't exist
def load_employee_registry():
    registry_path = "employee_registry.json"
    if os.path.exists(registry_path):
        with open(registry_path, "r") as f:
            return json.load(f)
    else:
        # Create a sample registry with some example employees
        sample_registry = {
            "EMP001": {"name": "John Doe", "email": "john.doe@company.com"},
            "EMP002": {"name": "Jane Smith", "email": "jane.smith@company.com"},
            "EMP003": {"name": "Alex Johnson", "email": "alex.johnson@company.com"}
        }
        with open(registry_path, "w") as f:
            json.dump(sample_registry, f)
        return sample_registry

# Save employee registry
def save_employee_registry(registry):
    with open("employee_registry.json", "w") as f:
        json.dump(registry, f)

# Extract employee info from uploaded document content
def extract_employee_info(content):
    # Look for patterns like "Name: John Doe", "Employee ID: EMP123", "Email: example@company.com"
    name_match = re.search(r"(?:Name|Employee Name|Full Name):\s*([A-Za-z\s]+)", content, re.IGNORECASE)
    id_match = re.search(r"(?:Employee ID|ID|Employee Number):\s*([A-Za-z0-9]+)", content, re.IGNORECASE)
    email_match = re.search(r"(?:Email|E-mail):\s*([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})", content, re.IGNORECASE)
    
    info = {}
    if name_match:
        info["name"] = name_match.group(1).strip()
    if id_match:
        info["id"] = id_match.group(1).strip()
    if email_match:
        info["email"] = email_match.group(1).strip()
    
    return info if info else None

# Function to get uuid by employee details
def get_uuid_by_details(info, employee_registry):
    # Try to match by ID, email, or name
    if "id" in info:
        for uuid, details in employee_registry.items():
            if details.get("employee_id", "").lower() == info["id"].lower():
                return uuid
    
    if "email" in info:
        for uuid, details in employee_registry.items():
            if details.get("email", "").lower() == info["email"].lower():
                return uuid
    
    if "name" in info:
        for uuid, details in employee_registry.items():
            if details.get("name", "").lower() == info["name"].lower():
                return uuid
    
    return None

# Initialize vector database for a specific employee
def init_vector_db(employee_id):
    # Use HuggingFace embeddings (free to use)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create persistent directory for this employee's vector store
    persist_dir = f"chroma_db/{employee_id}"
    os.makedirs(persist_dir, exist_ok=True)
    
    # Initialize Chroma vector store
    return Chroma(persist_directory=persist_dir, embedding_function=embeddings)

# Function to process documents with employee metadata
def process_document(file_path, file_type, employee_id):
    try:
        # Choose loader based on file type
        if file_type == "pdf":
            loader = PyPDFLoader(file_path)
        elif file_type == "csv":
            loader = CSVLoader(file_path)
        else:  # Default to text
            loader = TextLoader(file_path)
        
        # Load documents
        documents = loader.load()
        
        # Add employee_id to metadata
        for doc in documents:
            doc.metadata['employee_id'] = employee_id
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        
        # Get or create employee-specific vector store
        if employee_id not in vector_stores:
            vector_stores[employee_id] = init_vector_db(employee_id)
        
        # Add to vector store
        vector_stores[employee_id].add_documents(splits)
        return len(splits)
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise e

# Function to get response from document vector store
def get_document_response(query: str, requesting_employee_id: str, target_employee_id: Optional[str] = None):
    # Default to the requesting employee's ID if target is not specified
    employee_id_to_query = target_employee_id or requesting_employee_id
    
    # Access control check
    if target_employee_id and target_employee_id != requesting_employee_id:
        return "Access Denied: You can only access your own salary information."
    
    # Make sure the employee's vector store exists
    if employee_id_to_query not in vector_stores:
        vector_stores[employee_id_to_query] = init_vector_db(employee_id_to_query)
    
    # Query the vector store
    results = vector_stores[employee_id_to_query].similarity_search(query, k=3)
    
    # Format context from results
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Create prompt for LLM
    template = """
    You are a salary query assistant for employees. You help employees understand their salary and leave information.
    
    Context from documents:
    {context}
    
    User question: {question}
    Employee ID: {employee_id}
    
    Based on the context provided, answer the employee's question about their salary or leave information.
    If the context doesn't contain enough information to answer the question fully, suggest what other information might be needed.
    Remember that employees should only be able to access their own salary information.
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Initialize LLM model
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
    
    chain = (
        prompt 
        | llm 
        | StrOutputParser()
    )
    
    return chain.invoke({
        "context": context,
        "question": query,
        "employee_id": employee_id_to_query
    })

def identify_employees_with_llm(content, employee_registry):
    """
    Use LLM to identify employees mentioned in document content.
    Returns a list of employee UUIDs that are mentioned in the document.
    """
    # Initialize LLM
    llm = ChatGroq(
        api_key=os.getenv("GROQ_API_KEY"),
        model="llama-3.3-70b-versatile"
    )
    
    # Create a list of employees for the LLM to reference
    employee_list = []
    for uuid, details in employee_registry.items():
        employee_info = {
            "uuid": uuid,
            "name": details.get("name", ""),
            "email": details.get("email", ""),
            "employee_id": details.get("employee_id", "")
        }
        employee_list.append(employee_info)
    
    # Create prompt for the LLM
    template = """
    You are an employee document analyzer. Your job is to identify which employees are mentioned in a document.
    
    Below is a list of employees with their details:
    {employee_list}
    
    And here is the document content:
    {document_content}
    
    Task: Identify which employees from the list are mentioned in the document content.
    An employee is considered mentioned if their name, email, or employee ID appears in the document.
    
    Return your answer as a JSON list of employee UUIDs only. For example:
    ["EMP001", "EMP003"]
    
    If no employees are mentioned, return an empty list:
    []
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    
    # Create parser to extract JSON response
    def parse_employee_uuids(text):
        try:
            # Extract JSON array from the text (if there's explanatory text around it)
            matches = re.search(r'\[.*\]', text, re.DOTALL)
            if matches:
                json_str = matches.group(0)
            else:
                json_str = text
                
            # Parse the JSON
            employee_uuids = json.loads(json_str)
            return employee_uuids
        except Exception as e:
            logging.error(f"Error parsing LLM response: {e}")
            logging.error(f"Response was: {text}")
            return []
    
    # Chain components together
    chain = (
        prompt 
        | llm 
        | StrOutputParser()
        | parse_employee_uuids
    )
    
    try:
        # Invoke the chain
        result = chain.invoke({
            "employee_list": json.dumps(employee_list, indent=2),
            "document_content": content[:5000]  # Limit to first 5000 chars to stay within context window
        })
        
        return result
    except Exception as e:
        logging.error(f"Error in LLM employee identification: {e}")
        # Fall back to regex method if LLM fails
        return []

@app.post("/chat")
async def chat(request: ChatRequest):
    global db_connection
    
    if not request.employee_id:
        raise HTTPException(status_code=400, detail="Employee ID is required")
    
    # Create user-specific chat history if it doesn't exist
    if request.employee_id not in chat_histories:
        chat_histories[request.employee_id] = []
    
    # Add user message to chat history
    chat_histories[request.employee_id].append(ChatMessage(role="user", content=request.message))
    
    # Check if database is connected
    if not db_connection:
        # Try to connect using environment variables
        try:
            db_config = DatabaseConfig(
                host=os.getenv("DB_HOST"),
                port=os.getenv("DB_PORT"),
                user=os.getenv("DB_USER"),
                password=os.getenv("DB_PASSWORD"),
                database=os.getenv("DB_NAME")
            )
            db_connection = init_database(db_config)
        except Exception as e:
            # If database connection fails, but we have documents, we can still proceed
            if request.employee_id not in vector_stores or vector_stores[request.employee_id]._collection.count() == 0:
                raise HTTPException(status_code=500, detail="Database connection failed and no document data available.")
    
    try:
        # Check if this is a salary-related query
        query_keywords = request.message.lower()
        is_salary_query = any(keyword in query_keywords for keyword in 
                             ["salary", "pay", "deduction", "bonus", "tax", "earnings", "payment"])
        
        # Determine which source to use
        if db_connection and is_salary_query:
            # Use SQL database for salary queries
            response = get_response(request.message, db_connection, request.employee_id)
        elif request.employee_id in vector_stores and vector_stores[request.employee_id]._collection.count() > 0:
            # Use vector store for document queries
            response = get_document_response(request.message, request.employee_id)
        else:
            response = "I don't have the necessary data to answer your question. Please upload your salary documents."
        
        # Add AI response to chat history
        ai_message = ChatMessage(role="ai", content=response)
        chat_histories[request.employee_id].append(ai_message)
        
        return {
            "response": response,
            "role": "ai"
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/chat-history/{employee_id}")
async def get_chat_history(employee_id: str):
    if employee_id not in chat_histories:
        return {"chat_history": []}
    return {"chat_history": chat_histories[employee_id]}

@app.post("/clear-history/{employee_id}")
async def clear_chat_history(employee_id: str):
    if employee_id in chat_histories:
        chat_histories[employee_id] = []
    return {"message": "Chat history cleared"}

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    # Redirect to login page
    return RedirectResponse(url="/login")

@app.get("/login", response_class=HTMLResponse)
async def get_login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
async def get_dashboard(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/verify-employee")
async def verify_employee(verification: EmployeeVerification):
    # Load employee registry
    employee_registry = load_employee_registry()
    
    # Check if employee ID exists
    if verification.employee_id in employee_registry:
        return {"verified": True, "employee_info": employee_registry[verification.employee_id]}
    else:
        # Strict verification - only allow UUIDs that exist in the registry
        return {"verified": False, "message": "Invalid UUID. Please try again."}

@app.post("/register-employee")
async def register_employee(employee: EmployeeInfo):
    employee_registry = load_employee_registry()
    
    # Add or update employee in registry
    if employee.uuid not in employee_registry:
        employee_registry[employee.uuid] = {}
    
    if employee.name:
        employee_registry[employee.uuid]["name"] = employee.name
    if employee.email:
        employee_registry[employee.uuid]["email"] = employee.email
        
    # Save updated registry
    save_employee_registry(employee_registry)
    
    return {"status": "success", "message": "Employee information registered"}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...), employee_id: Optional[str] = Form(None)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    temp_path = None
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            # Write the uploaded file content to the temporary file
            contents = await file.read()
            temp.write(contents)
            temp_path = temp.name
        
        # Extract text for employee identification
        file_extension = file.filename.split(".")[-1].lower()
        
        # If no employee_id is provided, try to extract it from the document
        extracted_uuid = None
        if not employee_id:
            # Extract content from file for employee identification
            content = ""
            if file_extension == "pdf":
                from PyPDF2 import PdfReader
                reader = PdfReader(temp_path)
                for page in reader.pages:
                    content += page.extract_text()
            elif file_extension == "csv":
                import csv
                with open(temp_path, 'r') as csv_file:
                    csv_reader = csv.reader(csv_file)
                    for row in csv_reader:
                        content += " ".join(row) + "\n"
            else:  # Assume text file
                with open(temp_path, 'r') as text_file:
                    content = text_file.read()
            
            # Extract employee info
            employee_info = extract_employee_info(content)
            
            if employee_info:
                # Get UUID from registry
                employee_registry = load_employee_registry()
                extracted_uuid = get_uuid_by_details(employee_info, employee_registry)
            
            if not extracted_uuid:
                raise HTTPException(status_code=400, detail="Could not identify employee from document and no employee ID provided")
            
            employee_id = extracted_uuid
        
        # Determine file type from extension
        file_type = "pdf" if file_extension == "pdf" else "csv" if file_extension == "csv" else "text"
        
        # Process the document
        chunks_added = process_document(temp_path, file_type, employee_id)
        
        return {"message": f"Document uploaded and processed successfully. Added {chunks_added} chunks to knowledge base for employee {employee_id}."}
    except Exception as e:
        print(f"Error processing uploaded file: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")
    finally:
        # Clean up the temporary file with proper error handling
        if temp_path:
            try:
                if os.path.exists(temp_path):
                    os.unlink(temp_path)
            except Exception as e:
                print(f"Warning - could not delete temporary file: {str(e)}")

@app.post("/query-documents")
async def query_documents(query: DocumentQuery):
    if not query.employee_id:
        raise HTTPException(status_code=400, detail="Employee ID is required")
    
    try:
        # Get employee-specific vector store
        if query.employee_id not in vector_stores:
            vector_stores[query.employee_id] = init_vector_db(query.employee_id)
        
        # Query the vector store
        results = vector_stores[query.employee_id].similarity_search(query.query, k=5)
        
        # Format results
        formatted_results = []
        for doc in results:
            formatted_results.append({
                "content": doc.page_content,
                "metadata": doc.metadata
            })
        
        return {"results": formatted_results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))