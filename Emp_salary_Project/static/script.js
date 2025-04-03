document.addEventListener('DOMContentLoaded', function() {
    const dbForm = document.getElementById('db-form');
    const chatForm = document.getElementById('chat-form');
    const userInput = document.getElementById('user-input');
    const chatMessages = document.getElementById('chat-messages');
    const connectionStatus = document.getElementById('connection-status');
    const clearHistoryBtn = document.getElementById('clear-history');
    const sendButton = chatForm.querySelector('button[type="submit"]');
    const uploadForm = document.getElementById('upload-form');
    const queryForm = document.getElementById('query-form');
    const employeeIdInput = document.getElementById('employee-id');
    const verifyEmployeeBtn = document.getElementById('verify-employee');
    const employeeStatus = document.getElementById('employee-status');
    const currentEmployeeIdDisplay = document.getElementById('current-employee-id');
    const logoutBtn = document.getElementById('logout-btn');
    
    let connected = true; // Set to true since we're connecting automatically
    let documentsUploaded = false;
    let employeeVerified = false;
    let currentEmployeeId = '';
    
    // Show connection status as connected initially
    connectionStatus.textContent = 'Connected automatically';
    connectionStatus.className = 'mt-3 text-center connected';
    
    // Add system message about connection
    addMessage('Database connected automatically. Please verify your employee ID to start querying.', 'system');
    
    // Load chat history on page load
    loadChatHistory();
    
    // Verify employee
    verifyEmployeeBtn && verifyEmployeeBtn.addEventListener('click', async function() {
        const employeeId = employeeIdInput.value.trim();
        
        if (!employeeId) {
            employeeStatus.textContent = 'Please enter an Employee ID';
            employeeStatus.className = 'mt-2 text-danger';
            return;
        }
        
        try {
            const response = await fetch('/verify-employee', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ employee_id: employeeId })
            });
            
            const data = await response.json();
            
            if (response.ok && data.verified) {
                employeeStatus.textContent = 'Employee ID Verified';
                employeeStatus.className = 'mt-2 text-success';
                employeeVerified = true;
                currentEmployeeId = employeeId;
                
                // Update employee ID display
                if (currentEmployeeIdDisplay) {
                    currentEmployeeIdDisplay.textContent = employeeId;
                }
                
                // Enable chat since database is connected automatically
                userInput.disabled = false;
                sendButton.disabled = false;
                
                // Add system message
                addMessage(`Employee ID ${employeeId} verified. You can now ask questions about your salary information.`, 'system');
            } else {
                employeeStatus.textContent = 'Employee ID Verification Failed';
                employeeStatus.className = 'mt-2 text-danger';
                employeeVerified = false;
            }
        } catch (error) {
            employeeStatus.textContent = 'Verification Error: ' + error.message;
            employeeStatus.className = 'mt-2 text-danger';
            employeeVerified = false;
        }
    });
    
    // Make database form optional since we connect automatically
    dbForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const dbConfig = {
            host: document.getElementById('host').value,
            port: document.getElementById('port').value,
            user: document.getElementById('user').value,
            password: document.getElementById('password').value,
            database: document.getElementById('database').value
        };
        
        try {
            // Show loading message
            connectionStatus.textContent = 'Connecting...';
            connectionStatus.className = 'mt-3 text-center';
            
            const response = await fetch('/connect', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(dbConfig)
            });
            
            const data = await response.json();
            
            if (response.ok) {
                connectionStatus.textContent = 'Connected';
                connectionStatus.className = 'mt-3 text-center connected';
                connected = true;
                
                // Add system message
                addMessage(`Manually connected to ${dbConfig.database} database`, 'system');
            } else {
                connectionStatus.textContent = 'Connection Failed: ' + data.detail;
                connectionStatus.className = 'mt-3 text-center disconnected';
            }
        } catch (error) {
            connectionStatus.textContent = 'Connection Error: ' + error.message;
            connectionStatus.className = 'mt-3 text-center disconnected';
        }
    });
    
    // Send chat message
    chatForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const message = userInput.value.trim();
        const employeeId = currentEmployeeId || employeeIdInput.value.trim();
        
        if (!message) return;
        
        // Check if employee ID is provided
        if (!employeeId) {
            addMessage('Please verify your Employee ID first', 'system');
            return;
        }
        
        // Add user message to UI
        addMessage(message, 'user');
        userInput.value = '';
        
        try {
            const response = await fetch('/chat', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    message: message, 
                    employee_id: employeeId 
                })
            });
            
            if (response.ok) {
                const data = await response.json();
                addMessage(data.response, 'ai');
            } else {
                const error = await response.json();
                addMessage('Error: ' + error.detail, 'system');
            }
        } catch (error) {
            addMessage('Error: ' + error.message, 'system');
        }
    });
    
    // Clear chat history
    clearHistoryBtn.addEventListener('click', async function() {
        try {
            const response = await fetch('/clear-history', {
                method: 'POST'
            });
            
            if (response.ok) {
                chatMessages.innerHTML = '';
                addMessage('Chat history cleared', 'system');
            }
        } catch (error) {
            addMessage('Error clearing history: ' + error.message, 'system');
        }
    });
    
    // Logout functionality
    logoutBtn.addEventListener('click', function() {
        currentEmployeeId = '';
        employeeVerified = false;
        if (currentEmployeeIdDisplay) {
            currentEmployeeIdDisplay.textContent = 'Not verified';
        }
        userInput.disabled = true;
        sendButton.disabled = true;
        
        // Reset employee ID input
        if (employeeIdInput) {
            employeeIdInput.value = '';
        }
        
        // Add system message
        addMessage('Logged out. Please verify your employee ID to continue.', 'system');
    });
    
    // Upload document with employee ID
    uploadForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const fileInput = document.getElementById('document');
        const file = fileInput.files[0];
        const employeeId = currentEmployeeId || employeeIdInput.value.trim();
        
        if (!file) {
            addMessage('Please select a file to upload', 'system');
            return;
        }
        
        if (!employeeId) {
            addMessage('Please verify your Employee ID before uploading documents', 'system');
            return;
        }
        
        const formData = new FormData();
        formData.append('file', file);
        formData.append('employee_id', employeeId);  // Add employee ID to the form data
        
        try {
            // Show loading message
            const loadingId = addMessage(`Uploading ${file.name} for employee ${employeeId}...`, 'system');
            
            const response = await fetch('/upload-document', {
                method: 'POST',
                body: formData
            });
            
            // Remove loading message
            removeMessage(loadingId);
            
            if (response.ok) {
                const data = await response.json();
                addMessage(data.message, 'system');
                
                // Set documents uploaded flag
                documentsUploaded = true;
                
                // Clear file input
                fileInput.value = '';
            } else {
                const error = await response.json();
                addMessage('Upload Error: ' + error.detail, 'system');
            }
        } catch (error) {
            addMessage('Upload Error: ' + error.message, 'system');
        }
    });
    
    // Query documents with employee ID filter
    queryForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const queryInput = document.getElementById('query');
        const query = queryInput.value.trim();
        const employeeId = currentEmployeeId || employeeIdInput.value.trim();
        
        if (!query) {
            addMessage('Please enter a search query', 'system');
            return;
        }
        
        try {
            // Show loading message
            const loadingId = addMessage(`Searching for: "${query}"...`, 'system');
            
            const response = await fetch('/query-documents', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ 
                    query: query,
                    employee_id: employeeId  // Include employee ID for filtering
                })
            });
            
            // Remove loading message
            removeMessage(loadingId);
            
            if (response.ok) {
                const data = await response.json();
                
                if (data.results && data.results.length > 0) {
                    // Add search results header
                    addMessage(`Found ${data.results.length} results for "${query}":`, 'system');
                    
                    // Add each result
                    data.results.forEach((result, index) => {
                        const resultDiv = document.createElement('div');
                        resultDiv.className = 'search-result';
                        
                        const contentDiv = document.createElement('div');
                        contentDiv.className = 'search-result-content';
                        contentDiv.textContent = result.content;
                        
                        const metadataDiv = document.createElement('div');
                        metadataDiv.className = 'search-result-metadata';
                        metadataDiv.textContent = JSON.stringify(result.metadata);
                        
                        resultDiv.appendChild(contentDiv);
                        resultDiv.appendChild(metadataDiv);
                        
                        const messageDiv = document.createElement('div');
                        messageDiv.className = 'message ai';
                        messageDiv.appendChild(resultDiv);
                        
                        chatMessages.appendChild(messageDiv);
                    });
                } else {
                    addMessage(`No results found for "${query}"`, 'system');
                }
                
                // Clear query input
                queryInput.value = '';
            } else {
                const error = await response.json();
                addMessage('Search Error: ' + error.detail, 'system');
            }
        } catch (error) {
            addMessage('Search Error: ' + error.message, 'system');
        }
    });
    
    // Load chat history
    async function loadChatHistory() {
        try {
            const response = await fetch('/chat-history');
            
            if (response.ok) {
                const data = await response.json();
                
                // Clear existing messages
                chatMessages.innerHTML = '';
                
                // Add welcome message
                addMessage('Welcome to the Salary Information Assistant! Please verify your employee ID to start asking questions.', 'system');
                
                // Add messages from history
                if (data.chat_history && data.chat_history.length > 0) {
                    data.chat_history.forEach(msg => {
                        addMessage(msg.content, msg.role);
                    });
                }
            }
        } catch (error) {
            console.error('Error loading chat history:', error);
        }
    }
    
    // Helper function to remove a message by ID
    function removeMessage(id) {
        const message = document.getElementById(id);
        if (message) {
            message.remove();
        }
    }
    
    // Helper function that returns the ID of the added message
    function addMessage(content, role) {
        const messageDiv = document.createElement('div');
        const messageId = 'msg-' + Date.now();
        messageDiv.id = messageId;
        messageDiv.className = `message ${role}`;
        messageDiv.innerHTML = content;
        chatMessages.appendChild(messageDiv);
        chatMessages.scrollTop = chatMessages.scrollHeight;
        return messageId;
    }
});