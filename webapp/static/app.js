document.getElementById('floodForm').addEventListener('submit', function(e) {
    e.preventDefault();
    const rainfall = document.getElementById('rainfall').value;
    const riverLevel = document.getElementById('river_level').value;

    fetch('/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ rainfall: rainfall, river_level: riverLevel })
    })
    .then(response => response.json())
    .then(data => {
        document.getElementById('result').innerText = `Flood Risk: ${data['Flood Risk']}`;
    });
});
