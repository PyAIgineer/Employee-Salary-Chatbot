document.addEventListener('DOMContentLoaded', function() {
    const loginForm = document.getElementById('login-form');
    const loginStatus = document.getElementById('login-status');
    const uuidInput = document.getElementById('employee-uuid');
    
    // Clear any previous UUID in the input field and prevent autofill
    uuidInput.value = '';
    
    // Add random name attribute to prevent browser autofill
    uuidInput.setAttribute('name', 'uuid_' + Math.random().toString(36).substring(2, 11));
    
    // Clear any existing session data
    sessionStorage.clear();
    
    // Use a timeout to ensure the field is cleared after any browser autofill
    setTimeout(() => {
        uuidInput.value = '';
    }, 100);

    loginForm.addEventListener('submit', async function(e) {
        e.preventDefault();
        
        const uuid = uuidInput.value.trim();
        
        if (!uuid) {
            displayLoginStatus('Please enter your UUID', 'error');
            return;
        }
        
        try {
            // Show loading message
            displayLoginStatus('Verifying...', 'loading');
            
            const response = await fetch('/verify-employee', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ employee_id: uuid })
            });
            
            const data = await response.json();
            
            if (response.ok && data.verified) {
                // Store UUID in session storage
                sessionStorage.setItem('employee_uuid', uuid);
                
                // Store employee name if available
                if (data.employee_info && data.employee_info.name) {
                    sessionStorage.setItem('employee_name', data.employee_info.name);
                }
                
                // Show success message briefly before redirecting
                displayLoginStatus(`Login successful! Redirecting...`, 'success');
                
                // Redirect to main page after a brief delay
                setTimeout(() => {
                    window.location.href = '/dashboard';
                }, 1000);
            } else {
                displayLoginStatus(data.message || 'Invalid UUID. Please try again.', 'error');
                // Clear the input on failed login for security
                uuidInput.value = '';
            }
        } catch (error) {
            displayLoginStatus('Login Error: ' + error.message, 'error');
            // Clear the input on error for security
            uuidInput.value = '';
        }
    });
    
    function displayLoginStatus(message, type) {
        loginStatus.textContent = message;
        
        // Clear any existing classes
        loginStatus.className = 'mt-3 text-center';
        
        // Add appropriate class based on status type
        switch(type) {
            case 'success':
                loginStatus.classList.add('text-success');
                break;
            case 'error':
                loginStatus.classList.add('text-danger');
                break;
            case 'loading':
                loginStatus.classList.add('text-primary');
                break;
        }
    }
    
    // Focus on the input field but clear it first
    uuidInput.focus();
    uuidInput.value = '';
});