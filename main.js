function analyze() {
    const text = document.getElementById('text-input').value;  // Get text from textarea
    
    // Ensure text is not empty
    if (!text) {
        alert("Please enter some text.");
        return;
    }

    // Send the text to the Flask backend via a POST request
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/x-www-form-urlencoded'  // Make sure it's form-urlencoded
        },
        body: new URLSearchParams({
            'text': text  // Ensure the 'text' field is being sent
        })
    })
    .then(response => response.json())
    .then(data => {
        // Show results
        document.getElementById('results').innerHTML = `Prediction: ${data.label}<br>Confidence: ${data.confidence}`;
    })
    .catch(error => {
        console.error("Error:", error);
    });
}
