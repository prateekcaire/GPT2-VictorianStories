// Establish WebSocket connection
const socket = new WebSocket('wss://u3z4o1jl33.execute-api.us-west-2.amazonaws.com/prod');

// Connection opened
socket.addEventListener('open', (event) => {
    console.log('Connected to WebSocket server');
});

// Listen for messages
socket.addEventListener('message', (event) => {
    console.log('Message from server:', event.data);
    // Handle the message here (e.g., update UI)
});

// Connection closed
socket.addEventListener('close', (event) => {
    console.log('Disconnected from WebSocket server');
});

// Handle any errors
socket.addEventListener('error', (event) => {
    console.error('WebSocket error:', event);
});

// Function to send a message to the server
function sendMessage(message) {
    if (socket.readyState === WebSocket.OPEN) {
        socket.send(JSON.stringify(message));
    } else {
        console.error('WebSocket is not open. ReadyState:', socket.readyState);
    }
}

// Example usage:
// sendMessage({ action: 'sendmessage', prompt: 'Hello, GPT!', max_tokens: 100 });