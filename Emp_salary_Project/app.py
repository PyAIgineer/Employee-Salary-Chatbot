import os
from typing import List, Optional
from fastapi import FastAPI, HTTPException, Request, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import tempfile
import shutil

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
chat_history = []  # Store chat history
vector_store = None  # Store vector database
employee_id = None  # Store current employee ID for filtering queries

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
    global chat_history
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

@app.post("/chat")
async def chat(request: ChatRequest):
    global db_connection, chat_history, vector_store, employee_id
    
    # Add user message to chat history
    chat_history.append(ChatMessage(role="user", content=request.message))
    
    # Check if we have either a database connection or documents indexed
    if not db_connection and (vector_store is None or vector_store._collection.count() == 0):
        raise HTTPException(status_code=400, detail="Either connect to a database or upload documents first")
    
    try:
        # Check if this is a salary-related query
        query_keywords = request.message.lower()
        is_salary_query = any(keyword in query_keywords for keyword in 
                             ["salary", "pay", "deduction", "bonus", "tax", "earnings", "payment"])
        
        # For salary queries, we need employee ID
        if is_salary_query and not request.employee_id:
            raise HTTPException(status_code=400, detail="Please provide an employee ID for salary-related queries")
        
        # Store employee ID for this session if provided
        if request.employee_id:
            employee_id = request.employee_id
        
        # Determine which source to use
        if db_connection and is_salary_query:
            # Use SQL database for salary queries
            response = get_response(request.message, db_connection, employee_id)
        elif vector_store and vector_store._collection.count() > 0:
            # Use vector store for document queries
            response = get_document_response(request.message, request.employee_id)
        else:
            response = "I don't have the necessary data to answer your question."
        
        # Add AI response to chat history
        ai_message = ChatMessage(role="ai", content=response)
        chat_history.append(ai_message)
        
        return {
            "response": response,
            "role": "ai"
        }
    except Exception as e:
        if isinstance(e, HTTPException):
            raise e
        raise HTTPException(status_code=500, detail=str(e))
        
        # Add AI response to chat history
        ai_message = ChatMessage(role="ai", content=response)
        chat_history.append(ai_message)
        
        return {
            "response": response,
            "role": "ai"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# function to get response from document vector store
def get_document_response(query: str, employee_id: str = None):
    global vector_store
    
    # Query the vector store
    results = vector_store.similarity_search(query, k=3)
    
    # Filter results by employee_id if provided
    if employee_id:
        filtered_results = [doc for doc in results if doc.metadata.get('employee_id') == employee_id]
        if filtered_results:  # Only use filtered results if any match
            results = filtered_results
    
    # Format context from results
    context = "\n\n".join([doc.page_content for doc in results])
    
    # Create prompt for LLM
    template = """
    You are a salary query assistant for employees. You help employees understand their salary and leave information.
    
    Context from documents:
    {context}
    
    User question: {question}
    Employee ID (if provided): {employee_id}
    
    Based on the context provided, answer the employee's question. 
    If the question is about salary and no employee ID is provided, explain that an employee ID is needed.
    If the context doesn't contain enough information to answer the question fully, suggest what other information might be needed.
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
        "employee_id": employee_id or "Not provided"
    })

@app.get("/chat-history")
async def get_chat_history():
    return {"chat_history": chat_history}

@app.post("/clear-history")
async def clear_chat_history():
    global chat_history
    chat_history = []
    return {"message": "Chat history cleared"}

@app.get("/", response_class=HTMLResponse)
async def get_home(request: Request):
    print("Home route accessed")  # Add this line for debugging
    return templates.TemplateResponse("index.html", {"request": request})

# Initialize vector database
def init_vector_db():
    # Use HuggingFace embeddings (free to use)
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    # Create persistent directory for vector store
    os.makedirs("chroma_db", exist_ok=True)
    
    # Initialize Chroma vector store
    return Chroma(persist_directory="chroma_db", embedding_function=embeddings)

# function to process documents with employee metadata
def process_document(file_path, file_type, employee_id=None):
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
        
        # Add employee_id to metadata if available
        if employee_id:
            for doc in documents:
                doc.metadata['employee_id'] = employee_id
        
        # Split documents into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        splits = text_splitter.split_documents(documents)
        
        # Add to vector store
        global vector_store
        if vector_store is None:
            vector_store = init_vector_db()
        
        vector_store.add_documents(splits)
        return len(splits)
    except Exception as e:
        print(f"Error processing document: {str(e)}")
        raise e

@app.on_event("startup")
async def startup_event():
    global vector_store, db_connection
    # Initialize vector store if it doesn't exist
    if vector_store is None:
        vector_store = init_vector_db()
    
    # Automatically connect to the database using credentials from .env
    db_config = DatabaseConfig(
        host=os.getenv("DB_HOST"),
        port=os.getenv("DB_PORT"),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME")
    )
    db_connection = init_database(db_config)
    print("Database connected on startup.")

@app.post("/verify-employee")
async def verify_employee(verification: EmployeeVerification):
    """
    For test version - always verify employee IDs as valid
    
    In production, you would implement actual verification logic:
    - Check database for employee existence
    - Validate credentials
    - Apply authentication rules
    """
    # For testing, accept any employee ID
    return {"verified": True}

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded")
    
    temp_path = None    
    try:
        # Create a temporary file
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            # Write the uploaded file content to the temporary file
            shutil.copyfileobj(file.file, temp)
            temp_path = temp.name
        
        # Make sure to close the file handle
        await file.close()
        
        # Determine file type from extension
        file_extension = file.filename.split(".")[-1].lower()
        file_type = "pdf" if file_extension == "pdf" else "csv" if file_extension == "csv" else "text"
        
        # Process the document
        chunks_added = process_document(temp_path, file_type)
        
        return {"message": f"HR document uploaded and processed successfully. Added {chunks_added} chunks to knowledge base."}
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
                # Continue execution, don't raise exception

@app.post("/query-documents")
async def query_documents(query: DocumentQuery):
    global vector_store
    
    if vector_store is None:
        raise HTTPException(status_code=400, detail="Vector database not initialized")
    
    try:
        employee_id_filter = query.employee_id or ""
        
        # Query the vector store
        results = vector_store.similarity_search(query.query, k=5)
        
        # Filter results by employee_id if provided
        if employee_id_filter:
            results = [doc for doc in results if doc.metadata.get('employee_id') == employee_id_filter]
        
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

# Add this block at the end of your script
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)