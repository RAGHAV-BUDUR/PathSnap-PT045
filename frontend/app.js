/**
 * PathSnap Frontend Application Logic
 * Team: PT-045 | P4: Parth Pandurang Kale (PES1UG24AM184)
 */

let API_BASE = "http://localhost:8000";
let selectedTripId = null;

// DOM Elements
const apiUrlInput = document.getElementById("apiUrlInput");
const btnConnect = document.getElementById("btnConnect");
const statusDot = document.getElementById("statusDot");
const tripGrid = document.getElementById("tripGrid");
const selectedTripLabel = document.getElementById("selectedTripLabel");
const btnRunInference = document.getElementById("btnRunInference");

const resultsCard = document.getElementById("resultsCard");
const resTripId = document.getElementById("resTripId");
const resAcc = document.getElementById("resAcc");
const resAccDiff = document.getElementById("resAccDiff");
const resCont = document.getElementById("resCont");
const resContDiff = document.getElementById("resContDiff");
const resObs = document.getElementById("resObs");
const resArcs = document.getElementById("resArcs");
const resTime = document.getElementById("resTime");
const resViterbiTime = document.getElementById("resViterbiTime");
const mapFrame = document.getElementById("mapFrame");
const btnOpenMapTab = document.getElementById("btnOpenMapTab");
const benchmarkTableBody = document.getElementById("benchmarkTableBody");

// Initialize
document.addEventListener("DOMContentLoaded", () => {
  // Check if API_BASE is in current host
  if (window.location.port === "8000") {
    API_BASE = `${window.location.protocol}//${window.location.host}`;
    apiUrlInput.value = API_BASE;
  }

  btnConnect.addEventListener("click", () => {
    API_BASE = apiUrlInput.value.trim().replace(/\/$/, "");
    checkConnection();
  });

  btnRunInference.addEventListener("click", runInference);

  checkConnection();
  loadBenchmarks();
});

async function checkConnection() {
  statusDot.className = "status-indicator";
  try {
    const res = await fetch(`${API_BASE}/`);
    if (res.ok) {
      statusDot.className = "status-indicator online";
      loadTrips();
    } else {
      statusDot.className = "status-indicator offline";
    }
  } catch (err) {
    statusDot.className = "status-indicator offline";
    console.error("API connection failed:", err);
    tripGrid.innerHTML = `<div class="error-text" style="color:#dc2626; padding:10px;">Could not connect to Laptop 1 API at ${API_BASE}. Make sure uvicorn is running.</div>`;
  }
}

async function loadTrips() {
  try {
    const res = await fetch(`${API_BASE}/api/trips`);
    const data = await res.json();
    renderTrips(data.trips);
  } catch (err) {
    console.error("Failed to load trips:", err);
  }
}

function renderTrips(trips) {
  tripGrid.innerHTML = "";
  if (!trips || trips.length === 0) {
    tripGrid.innerHTML = "<div>No trips available</div>";
    return;
  }

  trips.forEach((trip) => {
    const chip = document.createElement("div");
    chip.className = "trip-chip";
    chip.innerHTML = `
      <div class="trip-chip-id">${trip.trip_id}</div>
      <div class="trip-chip-obs">${trip.num_observations} pts</div>
    `;

    chip.addEventListener("click", () => {
      document.querySelectorAll(".trip-chip").forEach((c) => c.classList.remove("selected"));
      chip.classList.add("selected");
      selectedTripId = trip.trip_id;
      selectedTripLabel.innerHTML = `Selected Trip: <strong>${trip.trip_id}</strong> (${trip.num_observations} observations, ${trip.duration_seconds}s)`;
      btnRunInference.disabled = false;
    });

    tripGrid.appendChild(chip);
  });
}

async function runInference() {
  if (!selectedTripId) return;

  btnRunInference.disabled = true;
  btnRunInference.innerHTML = `<span class="spinner"></span> Processing on Laptop 1...`;

  try {
    const res = await fetch(`${API_BASE}/api/match`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ trip_id: selectedTripId })
    });

    if (!res.ok) {
      throw new Error(`Server returned HTTP ${res.status}`);
    }

    const data = await res.json();
    displayResults(data);
  } catch (err) {
    alert("Inference failed: " + err.message);
    console.error(err);
  } finally {
    btnRunInference.disabled = false;
    btnRunInference.innerHTML = `
      <svg width="18" height="18" fill="none" stroke="currentColor" viewBox="0 0 24 24"><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M14.752 11.168l-3.197-2.132A1 1 0 0010 9.87v4.263a1 1 0 001.555.832l3.197-2.132a1 1 0 000-1.664z"></path><path stroke-linecap="round" stroke-linejoin="round" stroke-width="2" d="M21 12a9 9 0 11-18 0 9 9 0 0118 0z"></path></svg>
      Run Pipeline & Evaluate
    `;
  }
}

function displayResults(data) {
  resultsCard.style.display = "block";
  resTripId.textContent = data.trip_id;

  resAcc.textContent = `${data.observation_top1_accuracy}%`;
  const diffSign = data.accuracy_improvement >= 0 ? "+" : "";
  resAccDiff.textContent = `Raw Clf: ${data.raw_classifier_accuracy}% (${diffSign}${data.accuracy_improvement}%)`;

  resCont.textContent = `${data.route_continuity}%`;
  resContDiff.textContent = `Raw Clf: ${data.raw_classifier_continuity}%`;

  resObs.textContent = `${data.num_observations}`;
  resArcs.textContent = `${data.predicted_route_arc_count} predicted segments`;

  resTime.textContent = `${data.total_inference_time_ms} ms`;
  resViterbiTime.textContent = `Viterbi: ${data.viterbi_decoding_time_ms} ms`;

  // Set map iframe
  const mapUrl = `${API_BASE}${data.map_url}`;
  mapFrame.src = mapUrl;
  btnOpenMapTab.href = mapUrl;

  // Scroll to results smoothly
  resultsCard.scrollIntoView({ behavior: "smooth" });
}

async function loadBenchmarks() {
  try {
    const res = await fetch(`${API_BASE}/api/models`);
    if (!res.ok) return;
    const data = await res.json();
    renderBenchmarkTable(data.model_comparison);
  } catch (err) {
    console.warn("Could not load benchmarks from API:", err);
  }
}

function renderBenchmarkTable(models) {
  if (!models || models.length === 0) return;
  benchmarkTableBody.innerHTML = "";

  models.forEach((m) => {
    const tr = document.createElement("tr");
    if (m.Model === "Random Forest") {
      tr.className = "highlight-row";
    }
    tr.innerHTML = `
      <td><strong>${m.Model}</strong> ${m.Model === "Random Forest" ? '<span class="badge badge-success">Chosen</span>' : ''}</td>
      <td>${m['Algorithm Family']}</td>
      <td><code>${m['Key Configuration']}</code></td>
      <td><strong>${m['CV Macro F1-score']}</strong></td>
      <td>${m['Top-1 Accuracy']}</td>
      <td>${m['Route continuity']}</td>
      <td>${m['Student-Test Macro F1-score']}</td>
      <td>${m['Log Loss']}</td>
      <td>${m['Inference Time']}</td>
    `;
    benchmarkTableBody.appendChild(tr);
  });
}
