import os
import json
import re
from typing import List, Optional, Dict
from fastapi import FastAPI, HTTPException, Request, UploadFile, File, Response, Cookie, Depends
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
import tempfile
import shutil
import logging
import glob

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

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
vector_store = None  # Store vector database

# Employee chat history dictionary - store by employee UUID
employee_chat_histories = {}

# Active sessions store - simple in-memory storage
active_sessions = {}

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
    employee_id: Optional[str] = None

class DocumentQuery(BaseModel):
    query: str
    employee_id: Optional[str] = None

class EmployeeVerification(BaseModel):
    employee_id: str

class EmployeeInfo(BaseModel):
    name: str
    email: str

def get_employee_registry() -> Dict[str, EmployeeInfo]:
    """Load employee registry from JSON file."""
    try:
        with open("employee_registry.json", "r") as f:
            registry = json.load(f)
            return registry
    except (FileNotFoundError, json.JSONDecodeError) as e:
        logger.error(f"Error loading employee registry: {str(e)}")
        return {}

# def extract_employee_info_from_content(content: str, registry: Dict) -> Optional[str]:
#     """
#     Extract employee UUID from document content by matching
#     names or emails found in the registry.
#     """
#     # Create lookup dictionaries for reverse mapping
#     name_to_uuid = {info['name'].lower(): uuid for uuid, info in registry.items()}
#     email_to_uuid = {info['email'].lower(): uuid for uuid, info in registry.items()}
    
#     # Look for names and emails in content
#     for name, uuid in name_to_uuid.items():
#         if name in content.lower():
#             return uuid
    
#     for email, uuid in email_to_uuid.items():
#         if email in content.lower():
#             return uuid
    
#     # Try to find employee IDs or UUIDs mentioned directly
#     for uuid in registry.keys():
#         if uuid.lower() in content.lower():
#             return uuid
    
#     return None

def identify_employee_with_llm(content, registry):
    """Use LLM to identify employee from document content by finding name or email"""
    template = """
    You are a document analyzer. Extract the employee name or email from this document content.
    
    Document content:
    {content}
    
    Known employees (only for reference):
    {employee_list}
    
    INSTRUCTIONS:
    1. Look for patterns like "Employee Name: [Name]" or similar employee name indicators
    2. Look for email addresses that might belong to employees
    3. DO NOT look for or return employee IDs - they won't be in the documents
    4. Return ONLY the full employee name or email you found, exactly as written in the document
    5. If you find multiple names/emails, return only the first one
    6. If you can't identify any employee name or email, return "UNKNOWN"
    """
    
    # Format employee list for prompt
    employee_list = "\n".join([
        f"Name: {info['name']}, Email: {info['email']}" 
        for uuid, info in registry.items()
    ])
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatGroq(api_key=os.getenv("GROQ_API_KEY"), model="llama-3.3-70b-versatile")
    chain = prompt | llm | StrOutputParser()
    
    try:
        response = chain.invoke({
            "content": content,
            "employee_list": employee_list
        })
        
        # Clean the response
        response = response.strip()
        
        # If we got back a name or email, match it with registry
        if response != "UNKNOWN":
            # Try to match extracted name/email with registry
            for uuid, info in registry.items():
                if (
                    response.lower() in info['name'].lower() or 
                    info['name'].lower() in response.lower() or
                    response.lower() in info['email'].lower()
                ):
                    return uuid
        
        return None
    except Exception as e:
        logger.error(f"Error identifying employee with LLM: {str(e)}")
        return None

def verify_employee_uuid(employee_id: str) -> Optional[EmployeeInfo]:
    """Verify if employee UUID exists in registry."""
    registry = get_employee_registry()
    employee_data = registry.get(employee_id)
    if employee_data:
        return EmployeeInfo(**employee_data)
    return None

async def get_current_employee(session_id: str = Cookie(None)):
    """Get current employee ID from session cookie."""
    if not session_id or session_id not in active_sessions:
        raise HTTPException(
            status_code=401,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return active_sessions[session_id]

def init_database(config: DatabaseConfig) -> SQLDatabase:
    """Initialize SQL database connection."""
    db_uri = f"mysql+mysqlconnector://{config.user}:{config.password}@{config.host}:{config.port}/{config.database}"
    return SQLDatabase.from_uri(db_uri)

def get_sql_chain(db, employee_id: str):
    """Create SQL generation chain with employee ID filter."""
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

def get_response(user_query: str, db: SQLDatabase, employee_id: str, chat_history: List[ChatMessage]):
    """Generate a response using SQL database."""
    sql_chain = get_sql_chain(db, employee_id)
    
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
    
    IMPORTANT: The query already includes the employee_id filter for security purposes. 
    DO NOT mention any employee IDs in your response or comment on any ID matching issues.
    If the result is empty, simply state that no data was found for the specified criteria (like time period),
    without mentioning IDs or ID mismatches.
    
    Focus only on the actual salary data in the response, not on how the query is filtered.
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

def get_document_response(query: str, employee_id: str, chat_history: List[ChatMessage]):
    """Generate a response using document vectors."""
    global vector_store
    
    # Query the collection for the specific employee
    try:
        # First try to query the employee's specific collection
        results = vector_store.similarity_search(
            query, 
            k=3,
            filter={"employee_id": employee_id}
        )
        
        # If no results found, try without filter but post-filter manually
        if not results:
            logger.info(f"No direct results for employee {employee_id}, trying broader search")
            all_results = vector_store.similarity_search(query, k=10)
            results = [doc for doc in all_results if doc.metadata.get('employee_id') == employee_id]
    except Exception as e:
        logger.error(f"Error querying vector store: {str(e)}")
        return "I encountered an error retrieving document information."

    # If no results after filtering, return appropriate message
    if not results:
        return "I couldn't find any relevant documents for your query. Could you please try a different question?"
    
    # Format context from results
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Convert chat history to format expected by the chain
    formatted_history = []
    for msg in chat_history:
        if msg.role == "ai":
            formatted_history.append(AIMessage(content=msg.content))
        else:
            formatted_history.append(HumanMessage(content=msg.content))
    
    # Create prompt for LLM
    template = """
    You are a salary query assistant for employees. You help employees understand their salary and leave information.
    
    Context from documents:
    {context}
    
    Conversation History: {chat_history}
    User question: {question}

    Based on the context provided, answer the employee's question about their salary or leave information.
    
    IMPORTANT: Do NOT mention any employee IDs or filtering mechanisms in your response.
    Never discuss ID matching issues. The context is already filtered to only show information 
    relevant to the current user.
    
    If the context doesn't contain enough information to answer the question fully, just acknowledge that
    and suggest alternative information they might want to ask about.
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
        "chat_history": formatted_history,
        "question": query
    })

def load_and_process_employee_data():
    """
    Load employee data from data directory, chunk it, and save to vector store.
    Folder structure: data/{month}/{salary_details|leave_details}/files
    """
    global vector_store
    registry = get_employee_registry()
    
    if not registry:
        logger.error("Employee registry not available - cannot process data")
        return
    
    # Set up text splitter for document chunking
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    # Create a directory to track processed files
    os.makedirs("processed_files", exist_ok=True)
    processed_files_path = "processed_files/processed.json"
    
    # Load list of already processed files
    processed_files = set()
    if os.path.exists(processed_files_path):
        try:
            with open(processed_files_path, 'r') as f:
                processed_files = set(json.load(f))
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Could not load processed files list, starting fresh")
            processed_files = set()
    
    # Find all data files in the data directory structure
    try:
        all_files = []
        
        # Walk through the data directory to find all files
        for month_dir in glob.glob("data/*/"):
            month_name = os.path.basename(os.path.dirname(month_dir))
            logger.info(f"Processing month directory: {month_name}")
            
            # Check for salary_details and leave_details subdirectories
            for category in ["salary_details", "leave_details"]:
                category_path = os.path.join(month_dir, category)
                
                if os.path.exists(category_path):
                    # Get all files in this category directory
                    for file_path in glob.glob(f"{category_path}/*"):
                        if os.path.isfile(file_path) and file_path not in processed_files:
                            all_files.append((file_path, month_name, category))

            # Also check for files directly in the month directory
            for file_path in glob.glob(f"{month_dir}*.doc*"):  # Match .doc and .docx
                if os.path.isfile(file_path) and file_path not in processed_files:
                    # Determine category from filename
                    filename = os.path.basename(file_path).lower()
                    if "leave" in filename:
                        category = "leave_details"
                    elif "salary" in filename:
                        category = "salary_details"
                    else:
                        category = "other"
                    all_files.append((file_path, month_name, category))
        
        logger.info(f"Found {len(all_files)} new files to process")
        
        # Process each file
        docs_by_employee = {}
        
        for file_path, month, category in all_files:
            try:
                file_ext = os.path.splitext(file_path)[1].lower()
                
                # Load document based on file type
                if file_ext == '.txt':
                    loader = TextLoader(file_path)
                    docs = loader.load()
                elif file_ext == '.pdf':
                    loader = PyPDFLoader(file_path)
                    docs = loader.load()
                elif file_ext in ['.csv', '.tsv']:
                    loader = CSVLoader(file_path)
                    docs = loader.load()
                elif file_ext in ['.doc', '.docx']:
                    loader = Docx2txtLoader(file_path)
                    docs = loader.load()
                else:
                    logger.warning(f"Unsupported file type: {file_path}")
                    continue
                
                # Split documents into chunks
                chunks = text_splitter.split_documents(docs)
                
                # Determine employee_id for each chunk
                for chunk in chunks:
                    # Add month and category metadata
                    chunk.metadata["month"] = month
                    chunk.metadata["category"] = category
                    chunk.metadata["source"] = os.path.basename(file_path)
                    
                    # Extract employee ID from content
                    employee_id = identify_employee_with_llm(chunk.page_content, registry)
                    
                    if employee_id:
                        chunk.metadata["employee_id"] = employee_id
                        
                        # Group by employee
                        if employee_id not in docs_by_employee:
                            docs_by_employee[employee_id] = []
                        docs_by_employee[employee_id].append(chunk)
                    else:
                        logger.warning(f"Could not identify employee for chunk from {file_path}")
                
                # Mark file as processed
                processed_files.add(file_path)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
        
        # Add all documents to vector store by employee
        for employee_id, docs in docs_by_employee.items():
            logger.info(f"Adding {len(docs)} documents for employee {employee_id}")
            try:
                vector_store.add_documents(docs)
            except Exception as e:
                logger.error(f"Error adding documents for employee {employee_id}: {str(e)}")
        
        # Save updated processed files list
        with open(processed_files_path, 'w') as f:
            json.dump(list(processed_files), f)
            
        logger.info("Data processing complete")
        
    except Exception as e:
        logger.error(f"Error in load_and_process_employee_data: {str(e)}")

@app.post("/verify-employee")
async def verify_employee(verification: EmployeeVerification, response: Response):
    """Verify employee UUID and create session."""
    employee_id = verification.employee_id
    employee_info = verify_employee_uuid(employee_id)
    
    if employee_info:
        # Create a simple session ID
        session_id = os.urandom(16).hex()
        
        # Store the session with the employee ID
        active_sessions[session_id] = employee_id
        
        # Set session ID in cookie
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            max_age=28800,  # 8 hours in seconds
            samesite="lax"
        )
        
        # Initialize chat history for this employee if not exists
        if employee_id not in employee_chat_histories:
            employee_chat_histories[employee_id] = []
        
        return {
            "verified": True,
            "employee_info": {
                "name": employee_info.name,
                "email": employee_info.email
            }
        }
    else:
        return {"verified": False, "message": "Invalid employee UUID"}

@app.post("/chat")
async def chat(request: ChatRequest, employee_id: str = Depends(get_current_employee)):
    """Process chat message and generate response for authenticated employee."""
    global db_connection, vector_store
    
    # Validate that the employee can access only their own information
    if request.employee_id and request.employee_id != employee_id:
        return {
            "response": "Access Denied. You can only view your own information.",
            "role": "ai"
        }
    
    # Initialize chat history for this employee if not exists
    if employee_id not in employee_chat_histories:
        employee_chat_histories[employee_id] = []
    
    chat_history = employee_chat_histories[employee_id]
    
    # Add user message to chat history
    chat_history.append(ChatMessage(role="user", content=request.message))
    
    try:
        # Step 1: First, check if this query contains employee identifiers or names other than the current user
        query = request.message.lower()
        employee_registry = get_employee_registry()
        
        # Check if the query mentions other employees - very basic check
        # In a real system, this would be more sophisticated
        for other_id, other_info in employee_registry.items():
            if other_id != employee_id and (
                other_id.lower() in query or 
                other_info['name'].lower() in query or 
                other_info['email'].lower() in query
            ):
                response = "Access Denied. You can only view your own information."
                chat_history.append(ChatMessage(role="ai", content=response))
                return {"response": response, "role": "ai"}
        
        # Step 2: Process query with both databases
        response_db = None
        response_docs = None
        
        # Try MySQL for structured data if connected
        if db_connection:
            try:
                response_db = get_response(request.message, db_connection, employee_id, chat_history)
            except Exception as e:
                logger.error(f"Error with SQL response: {str(e)}")
                response_db = "Could not retrieve information from the database."
        else:
            logger.warning("Database connection not available")
        
        # Try vector store for document-based data
        if vector_store and vector_store._collection.count() > 0:
            try:
                response_docs = get_document_response(request.message, employee_id, chat_history)
            except Exception as e:
                logger.error(f"Error with document response: {str(e)}")
                response_docs = None
        else:
            logger.warning("Vector store not properly initialized or empty")
        
        # Combine responses intelligently
        if response_db and "Access Denied" in response_db:
            # If SQL access was denied, respect that
            response = response_db
        elif response_db and response_docs:
            # If we have useful info from both sources, combine them
            response = f"{response_db}\n\nAdditional information from your documents:\n{response_docs}"
        elif response_db:
            response = response_db
        elif response_docs:
            response = response_docs
        else:
            response = "I don't have the necessary data to answer your question."
        
        # Add AI response to chat history
        chat_history.append(ChatMessage(role="ai", content=response))
        
        # Update the employee's chat history
        employee_chat_histories[employee_id] = chat_history
        
        return {
            "response": response,
            "role": "ai"
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        
        error_msg = f"Error processing your request: {str(e)}"
        chat_history.append(ChatMessage(role="ai", content=error_msg))
        return {"response": error_msg, "role": "ai"}

@app.post("/logout")
async def logout(response: Response, session_id: str = Cookie(None)):
    """Clear session to log out employee."""
    if session_id and session_id in active_sessions:
        del active_sessions[session_id]
    
    response.delete_cookie(key="session_id")
    return {"message": "Logged out successfully"}

@app.get("/chat-history")
async def get_chat_history(employee_id: str = Depends(get_current_employee)):
    """Get chat history for authenticated employee."""
    if employee_id in employee_chat_histories:
        return {"chat_history": employee_chat_histories[employee_id]}
    return {"chat_history": []}

@app.post("/clear-history")
async def clear_chat_history(employee_id: str = Depends(get_current_employee)):
    """Clear chat history for authenticated employee."""
    if employee_id in employee_chat_histories:
        employee_chat_histories[employee_id] = []
    return {"message": "Chat history cleared"}

# Scheduled job to refresh employee data
@app.post("/refresh-employee-data")
async def refresh_employee_data():
    """Admin endpoint to refresh employee data."""
    load_and_process_employee_data()
    return {"message": "Employee data refresh initiated"}

# Initialize vector database
def init_vector_db():
    """Initialize Chroma vector database."""
    # Use HuggingFace embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create persistent directory for vector store
    os.makedirs("chroma_db", exist_ok=True)
    
    # Initialize Chroma vector store
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global vector_store, db_connection
    
    # Initialize vector store
    vector_store = init_vector_db()
    logger.info("Vector store initialized")
    
    # Process employee data and add to vector store
    load_and_process_employee_data()
    
    # Connect to MySQL database (if configured)
    try:
        db_config = DatabaseConfig(
            host=os.getenv("DB_HOST", "localhost"),
            port=os.getenv("DB_PORT", "3306"),
            user=os.getenv("DB_USER", "root"),
            password=os.getenv("DB_PASSWORD", ""),
            database=os.getenv("DB_NAME", "employee_db")
        )
        db_connection = init_database(db_config)
        logger.info("Database connected on startup.")
    except Exception as e:
        logger.error(f"Failed to connect to database: {str(e)}")
        db_connection = None

@app.get("/", response_class=HTMLResponse)
async def get_login_page(request: Request):
    """Serve login page."""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/chat", response_class=HTMLResponse)
async def get_chat_page(request: Request, employee_id: str = Depends(get_current_employee)):
    """Serve chat page for authenticated users."""
    return templates.TemplateResponse("chat.html", {"request": request})

@app.get("/user-info")
async def get_user_info(employee_id: str = Depends(get_current_employee)):
    """Get current authenticated user info."""
    employee_info = verify_employee_uuid(employee_id)
    if employee_info:
        return {
            "name": employee_info.name,
            "email": employee_info.email,
            "employee_id": employee_id
        }
    raise HTTPException(status_code=404, detail="Employee not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)