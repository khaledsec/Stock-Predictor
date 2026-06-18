// --- 1. Configuration & DOM Elements ---
const API_BASE_URL = "http://localhost:8000";

// Chart & Prediction Elements
const chartCanvas = document.getElementById('stockChart').getContext('2d');
const predictionText = document.getElementById('prediction-text');
const loader = document.getElementById('loader');
const apiStatus = document.getElementById('api-status');
let myStockChart;

// Dropdown & News Elements
const tickerSelect = document.getElementById('ticker-select');
const trainButton = document.getElementById('train-button');
const trainingStatus = document.getElementById('training-status');
const newsBox = document.getElementById('news-box');

// Metrics Card
const metricsBox = document.getElementById('metrics-box');
const maeText = document.getElementById('mae-text');
const rmseText = document.getElementById('rmse-text');
const mapeText = document.getElementById('mape-text');

// --- 2. Initial Setup & Logout ---
window.onload = () => {
    // Check if user is logged in
    const username = sessionStorage.getItem('username');
    if (username) {
        const welcomeElement = document.getElementById('welcome-msg');
        if (welcomeElement) {
            welcomeElement.innerText = `Welcome, ${username}!`;
        }
    }
    setStatus("Select a stock from the dropdown and click 'Fetch Live Data & Train Model'.", false);
};

window.logout = function() {
    sessionStorage.clear();
    window.location.href = 'login.html';
}

// --- 3. UI Helpers ---
function setStatus(message, isError = false) {
    apiStatus.innerHTML = message;
    apiStatus.className = isError 
        ? "mt-6 p-4 rounded-lg text-center bg-red-900 text-red-200"
        : "mt-6 p-4 rounded-lg text-center bg-blue-900 text-blue-200";
}

function setTrainingStatus(message, type = 'training') {
    if (type === 'success') {
        trainingStatus.innerHTML = message;
    } else if (type === 'error') {
        trainingStatus.innerHTML = `<p class="status-error">${message}</p>`;
    } else {
        trainingStatus.innerHTML = `<p class="status-training">${message}</p>`;
    }
    trainingStatus.className = `mt-4 text-center`;
}

function displayMetrics(metrics) {
    if (metrics) {
        if(maeText) maeText.innerText = metrics.mae;
        if(rmseText) rmseText.innerText = metrics.rmse || "N/A";
        if(mapeText) mapeText.innerText = metrics.mape || "N/A";
        metricsBox.classList.remove('hidden');
    } else {
        metricsBox.classList.add('hidden');
    }
}

// --- 4. Prediction, Chart, & News Logic ---
function renderChart(data) {
    // Zoom into the most recent window so today's price is clearly visible
    const ZOOM_DAYS = 90;
    const zoomed = data.length > ZOOM_DAYS ? data.slice(-ZOOM_DAYS) : data;

    const labels = zoomed.map(item => item.date);
    const realPrices = zoomed.map(item => item.real);
    const predictedPrices = zoomed.map(item => item.predicted);
    const lastIndex = realPrices.length - 1;

    if (myStockChart) myStockChart.destroy();

    myStockChart = new Chart(chartCanvas, {
        type: 'line',
        data: {
            labels: labels,
            datasets: [
                {
                    label: 'Real Prices',
                    data: realPrices,
                    borderColor: '#3498db',
                    backgroundColor: 'rgba(52, 152, 219, 0.1)',
                    fill: false,
                    borderWidth: 2,
                    pointRadius: (ctx) => ctx.dataIndex === lastIndex ? 5 : 0,
                    pointBackgroundColor: '#3498db'
                },
                { 
                    label: 'Predicted Prices', 
                    data: predictedPrices, 
                    borderColor: '#e67e22', 
                    backgroundColor: 'rgba(230, 126, 34, 0.1)', 
                    fill: false, 
                    borderWidth: 2, 
                    pointRadius: 0, 
                    borderDash: [5, 5] 
                }
            ]
        },
        options: { 
            responsive: true, 
            maintainAspectRatio: false,
            scales: {
                x: { type: 'time', time: { unit: 'month' }, grid: { color: 'rgba(255, 255, 255, 0.1)' }, ticks: { color: 'rgb(209, 213, 219)' } },
                y: { grid: { color: 'rgba(255, 255, 255, 0.1)' }, ticks: { color: 'rgb(209, 213, 219)', callback: (value) => '$' + value } }
            },
            plugins: { legend: { display: true, labels: { color: 'rgb(209, 213, 219)' } } }
        }
    });
}

// Fetch Latest News Function
async function fetchNews(ticker) {
    newsBox.innerHTML = '<p class="text-gray-400">Loading latest news...</p>';
    try {
        const response = await fetch(`${API_BASE_URL}/news/${ticker}`);
        const data = await response.json();
        
        if (data.news && data.news.length > 0) {
            newsBox.innerHTML = data.news.map(n => `
                <div class="border-b border-gray-700 pb-3 mb-3">
                    <a href="${n.link}" target="_blank" class="text-blue-400 hover:text-blue-300 hover:underline font-semibold block transition">${n.title}</a>
                    <span class="text-xs text-gray-500 mt-1 block">Source: ${n.publisher}</span>
                </div>
            `).join('');
        } else { 
            newsBox.innerHTML = '<p class="text-gray-500">No recent news found for this stock.</p>'; 
        }
    } catch (error) { 
        newsBox.innerHTML = '<p class="text-red-400">Failed to load news.</p>'; 
    }
}

async function fetchChartData() {
    try {
        const response = await fetch(`${API_BASE_URL}/all_predictions`);
        if (response.ok) {
            const data = await response.json();
            if(data.length > 0) renderChart(data);
            setStatus(`<strong>Connection successful.</strong><br>Successfully fetched and trained live data.`, false);
        }
    } catch (error) { 
        setStatus(`<strong>Error:</strong> Cannot fetch chart data.`, true);
    }
}

async function fetchPrediction() {
    loader.style.display = 'block';
    predictionText.innerHTML = '<p class="text-lg text-gray-400">Loading prediction...</p>';
    
    try {
        const response = await fetch(`${API_BASE_URL}/predict_next_day`);
        if (!response.ok) throw new Error('Failed to get prediction.');
        
        const data = await response.json();
        loader.style.display = 'none';
        predictionText.innerHTML = `<div class="text-5xl font-bold text-green-400">$${data.prediction.toFixed(2)}</div>`;
    } catch (error) {
        loader.style.display = 'none';
        predictionText.innerHTML = `<div class="text-lg font-medium text-red-400">Error fetching prediction.</div>`;
    }
}

// Handle the new Dropdown API Training
async function handleFetchAndTrain() {
    const ticker = tickerSelect.value;
    if (!ticker) return;

    // Reset UI state
    trainButton.disabled = true;
    trainButton.innerHTML = "Fetching Live Data...";
    setTrainingStatus(`Downloading last 5 years of ${ticker} data & training...`, 'training');
    
    if (myStockChart) myStockChart.destroy();
    displayMetrics(null);
    loader.style.display = 'block';
    predictionText.innerHTML = '<p class="text-lg text-gray-400">Waiting for new model...</p>';
    setStatus(`Connecting to Yahoo Finance for ${ticker}...`, false);

    // Fetch News in the background
    fetchNews(ticker);

    try {
        // Post to the new train_ticker endpoint
        const response = await fetch(`${API_BASE_URL}/train_ticker`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ ticker: ticker })
        });
        
        const result = await response.json();

        if (response.ok) {
            let historyHtml = `<p class="status-success">${result.message}</p>`; 

            

            setTrainingStatus(historyHtml, 'success');
            displayMetrics(result.metrics);
            
            // Fetch updated chart and next-day prediction
            await fetchChartData();
            await fetchPrediction();
        } else {
            throw new Error(result.detail);
        }
    } catch (error) {
        setTrainingStatus(error.message, 'error');
        setStatus(`<strong>Training Failed.</strong><br>${error.message}`, true);
        loader.style.display = 'none';
        predictionText.innerHTML = `<div class="text-lg font-medium text-red-400">Failed</div>`;
    } finally {
        trainButton.disabled = false;
        trainButton.innerHTML = "Fetch Live Data & Train Model";
    }
}

// --- 5. Event Listeners ---
if (trainButton) {
    trainButton.addEventListener('click', handleFetchAndTrain);
}
