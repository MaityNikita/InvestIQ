import re

with open(r"c:\Users\nikit\OneDrive\Desktop\mini project\graph.html", "r", encoding="utf-8") as f:
    content = f.read()

# We need to inject the data definition and fetch loop at the beginning of the script tag, and modify the chart definitions.
script_start = """  <script>
    // ══════════════════════════════════════════════════
    //  LIVE DATA INTEGRATION
    // ══════════════════════════════════════════════════
    const STOCKS = [
      { name: 'Tata Consultancy', symbol: 'TCS.NS', sector: 'IT', base: 3912, rb: 0.35 },
      { name: 'Infosys Limited', symbol: 'INFY.NS', sector: 'IT', base: 1543, rb: 0.52 },
      { name: 'Wipro Limited', symbol: 'WIPRO.NS', sector: 'IT', base: 567, rb: 0.58 },
      { name: 'HCL Technologies', symbol: 'HCLTECH.NS', sector: 'IT', base: 1830, rb: 0.40 },
      { name: 'Tech Mahindra', symbol: 'TECHM.NS', sector: 'IT', base: 1346, rb: 0.50 }
    ];
    let S = {};
    const N = 40;
    const rnd = (a, b) => Math.random() * (b - a) + a;
    const API = window.location.hostname === '127.0.0.1' || window.location.hostname === 'localhost' ? `${window.location.protocol}//${window.location.host}` : '';

    function initData() {
      STOCKS.forEach(s => {
        let p = s.base, h = [];
        for(let i=0; i<N; i++) { p += rnd(-s.base * .008, s.base * .009); h.push(p); }
        S[s.symbol] = { ...s, price: p, hist: h, risk: Math.min(100, Math.max(8, Math.round(s.rb * 100 + rnd(-10, 10)))) };
      });
    }
    initData();

    async function fetchLiveData() {
      try {
        const res = await fetch(`${API}/api/finnhub/all-quotes`);
        const data = await res.json();
        if (!data.error && !data.fallback && Object.keys(data).length > 0) {
          Object.entries(data).forEach(([ticker, info]) => {
            if (S[ticker]) {
              S[ticker].price = info.price;
              S[ticker].hist.push(info.price);
              if (S[ticker].hist.length > N) S[ticker].hist.shift();
            }
          });
        }
      } catch (e) {
        // Fallback simulation
        STOCKS.forEach(s => {
          const st = S[s.symbol];
          st.price += rnd(-st.base * 0.005, st.base * 0.006);
          st.hist.push(st.price);
          if (st.hist.length > N) st.hist.shift();
          st.risk = Math.min(100, Math.max(5, st.risk + rnd(-1.8, 1.8)));
        });
      }
      if (typeof updateCharts === 'function') updateCharts();
    }
    setInterval(fetchLiveData, 15000);

"""

script_end = """
    function updateCharts() {
      if(window.lineChart) {
        window.lineChart.data.datasets[0].data = S['TCS.NS'].hist;
        window.lineChart.data.datasets[1].data = S['INFY.NS'].hist;
        window.lineChart.data.datasets[2].data = S['WIPRO.NS'].hist;
        window.lineChart.data.datasets[3].data = S['HCLTECH.NS'].hist;
        window.lineChart.data.datasets[4].data = S['TECHM.NS'].hist;
        window.lineChart.data.labels = Array.from({length: S['TCS.NS'].hist.length}, (_, i) => i);
        window.lineChart.update('none');
      }
      if(window.barChart) {
        window.barChart.data.datasets[0].data = [S['TCS.NS'].price, S['INFY.NS'].price, S['WIPRO.NS'].price, S['HCLTECH.NS'].price, S['TECHM.NS'].price];
        window.barChart.update('none');
      }
      if(window.riskRadar) {
        window.riskRadar.data.datasets[0].data[0] = S['TCS.NS'].risk;
        window.riskRadar.data.datasets[1].data[0] = S['INFY.NS'].risk;
        window.riskRadar.data.datasets[2].data[0] = S['WIPRO.NS'].risk;
        window.riskRadar.update('none');
      }
      if(window.volatilityBar) {
        window.volatilityBar.data.datasets[0].data = [S['TCS.NS'].risk, S['INFY.NS'].risk, S['WIPRO.NS'].risk, S['HCLTECH.NS'].risk, S['TECHM.NS'].risk];
        window.volatilityBar.update('none');
      }
    }
    setTimeout(updateCharts, 50);
  </script>
"""

# Replace script start
content = content.replace("  <script>", script_start)

# Define charts as global variables
content = content.replace("new Chart(document.getElementById('lineChart')", "window.lineChart = new Chart(document.getElementById('lineChart')")
content = content.replace("new Chart(document.getElementById('barChart')", "window.barChart = new Chart(document.getElementById('barChart')")
content = content.replace("new Chart(document.getElementById('perfLineChart')", "window.perfLineChart = new Chart(document.getElementById('perfLineChart')")
content = content.replace("new Chart(document.getElementById('riskRadar')", "window.riskRadar = new Chart(document.getElementById('riskRadar')")
content = content.replace("new Chart(document.getElementById('volatilityBar')", "window.volatilityBar = new Chart(document.getElementById('volatilityBar')")
content = content.replace("new Chart(document.getElementById('sectorDonut')", "window.sectorDonut = new Chart(document.getElementById('sectorDonut')")
content = content.replace("new Chart(document.getElementById('sectorBar')", "window.sectorBar = new Chart(document.getElementById('sectorBar')")

# We don't overwrite datasets inline because our updateCharts function will update them automatically!
# This is much safer than heavy regex replacements of arrays in code.

# Replace script end
content = content.replace("  </script>", script_end)

with open(r"c:\Users\nikit\OneDrive\Desktop\mini project\graph.html", "w", encoding="utf-8") as f:
    f.write(content)
print("Updated graph.html")
