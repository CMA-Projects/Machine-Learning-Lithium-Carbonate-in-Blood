document.getElementById('inputForm').addEventListener('submit', function(e) {
    e.preventDefault();

    // Collect the user input values into different constants
    const inputValue1 = parseFloat(document.getElementById('input1').value);
    const inputValue2 = parseFloat(document.getElementById('input2').value);
    const inputValue3 = parseFloat(document.getElementById('input3').value);
    const inputValue4 = parseFloat(document.getElementById('input4').value);

    // Convert the user input values into a 2D array to use in the NN model
    const data = {
        inputs: [inputValue1, inputValue2, inputValue3, inputValue4]
    };

    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json'
        },
        body: JSON.stringify(data)
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            document.getElementById('result').innerText = 'Error: ' + data.error;
        } else {
            document.getElementById('result').innerText = 'Prediction: ' + data.prediction;
        }
    })
    .catch(error => {
        console.error('Error:', error);
        document.getElementById('result').innerText = 'Error outputting results';
    });
});
