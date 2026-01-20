document.getElementById("checkBtn").addEventListener("click", () => {
  const message = document.getElementById("message").value;
  const result = document.getElementById("result");
  const confidence = document.getElementById("confidence");

  if (message.trim() === "") {
    result.innerText = "Please enter a message";
    confidence.innerText = "";
    return;
  }

  result.innerText = "Checking...";
  confidence.innerText = "";

  fetch("http://127.0.0.1:5000/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ message: message })
  })
  .then(response => response.json())
  .then(data => {
    result.innerText = "Result: " + data.prediction;
    // backend returns `probability` as a float in [0,1]
    confidence.innerText = "Confidence: " + Math.round(data.probability * 100) + "%";
  })
  .catch(error => {
    result.innerText = "Backend not running!";
    confidence.innerText = "";
    console.error('predict error', error);
  });
});
