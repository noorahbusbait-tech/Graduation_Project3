
// Dummy data (replace with model output)
const days = ["Day 1","Day 2","Day 3","Day 4","Day 5","Day 6","Day 7",
              "Day 8","Day 9","Day 10","Day 11","Day 12","Day 13","Day 14"];

const losData = [5,6,5.5,6.2,6.8,7,6.5,6.7,7.2,7.5,7.1,6.9,7.3,7.6];
const occupancyData = [70,72,75,78,80,82,85,87,88,90,92,91,93,95];

// Length of Stay Chart
new Chart(document.getElementById("losChart"), {
    type: "line",
    data: {
        labels: days,
        datasets: [{
            label: "Length of Stay (Days)",
            data: losData,
            tension: 0.3
        }]
    }
});

// Bed Occupancy Chart
new Chart(document.getElementById("occupancyChart"), {
    type: "line",
    data: {
        labels: days,
        datasets: [{
            label: "Bed Occupancy (%)",
            data: occupancyData,
            tension: 0.3
        }]
    }
});

// Example Prediction Outputs
document.getElementById("patientStatus").innerText =
    "Stable (Low Risk of Readmission)";

document.getElementById("occupancyPrediction").innerText =
    "Expected to reach 95% capacity within 2 weeks";

document.getElementById("predOccupancy").innerText = "95%";
document.getElementById("predLOS").innerText = "7.6";
document.getElementById("predRisk").innerText = "12 patients";
document.getElementById("predConfidence").innerText = "89%";

function logout(){
    localStorage.clear();
    window.location="login.php";
}



// load your JSON file
fetch("finaloccupancy.json")
  .then(response => response.json())
  .then(data => {

    // 🔹 1. Fill table-like results
    let box = document.getElementById("occupancyResults");

    data.forEach(row => {
      box.innerHTML += `
        <p>${row.Date} → ${row.Tuned_Predicted_Occupancy}</p>
      `;
    });

    // 🔹 2. Prepare data for chart
    const labels = data.map(d => d.Date);
    const values = data.map(d => d.Tuned_Predicted_Occupancy);

    // 🔹 3. Draw chart
    new Chart(document.getElementById("occupancyChart"), {
      type: "line",
      data: {
        labels: labels,
        datasets: [{
          label: "Bed Occupancy",
          data: values,
          borderWidth: 2
        }]
      }
    });

    // 🔹 4. Example patient prediction (dummy for now)
    document.getElementById("patientPrediction").innerHTML +=
      "<p>Stable (Low Risk)</p>";

  })
  .catch(err => {
    console.error(err);
    document.getElementById("forecastStatus").innerText =
      "Model Status: Error loading data";
  });
