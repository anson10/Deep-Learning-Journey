document.getElementById('prediction-form').addEventListener('submit', async function (event) {
    event.preventDefault();

    const tv = parseFloat(document.getElementById('tv').value);
    const radio = parseFloat(document.getElementById('radio').value);
    const newspaper = parseFloat(document.getElementById('newspaper').value);

    const data = [
        {
            TV: tv,
            radio: radio,
            newspaper: newspaper
        }
    ];

    const resultElement = document.getElementById('result');
    const outputContainer = document.getElementById('output');

    // Display a loading message
    resultElement.innerText = 'Calculating...';

    try {
        const response = await fetch('http://127.0.0.1:5000/predict', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify(data),
        });

        const result = await response.json();
        resultElement.innerText = `Prediction: ${result.prediction}`;
        
        // Add a success animation
        outputContainer.style.borderColor = '#4CAF50';
        outputContainer.style.backgroundColor = '#eaffea';
    } catch (error) {
        console.error('Error:', error);
        resultElement.innerText = 'Error: Unable to get prediction.';
        
        // Add an error animation
        outputContainer.style.borderColor = '#FF0000';
        outputContainer.style.backgroundColor = '#ffeaea';
    }
});
