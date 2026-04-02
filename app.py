"""
=============================================================================
  app.py — Flask REST API
  AI Financial Advisory Chatbot — InvestIQ
=============================================================================
  Endpoints:
    GET  /                              — health check
    GET  /api/predict                   — stock price forecast
    GET  /api/recommend                 — personalised financial advice
    GET  /api/compare                   — YoY comparison data (JSON)
    POST /api/chat                      — chatbot NLP response
    POST /api/auth/register             — register a new user
    POST /api/auth/login                — login & get JWT
    GET  /api/auth/me                   — verify token / get profile
    POST /api/collect-email             — store visitor email (GDPR consent)

  Run:
      pip install flask flask-cors PyJWT werkzeug
      python app.py
=============================================================================
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import traceback
import uuid
import jwt
from datetime import datetime, timezone, timedelta
from werkzeug.security import generate_password_hash, check_password_hash
from functools import wraps

app = Flask(__name__, static_folder=".", static_url_path="")
CORS(app, supports_credentials=True)  # Allow frontend to call the API

# ── Auth Config ───────────────────────────────────────────────────────────────
JWT_SECRET  = os.environ.get("IQ_JWT_SECRET", "investiq-dev-secret-change-in-prod")
JWT_EXPIRY  = timedelta(days=7)
USERS_FILE  = os.path.join(os.path.dirname(__file__), "users.json")
EMAILS_FILE = os.path.join(os.path.dirname(__file__), "visitor_emails.json")


# ── JSON file helpers ─────────────────────────────────────────────────────────
def _load_json(path):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


# ── JWT helpers ───────────────────────────────────────────────────────────────
def _make_token(user_id, email, name):
    payload = {
        "sub":   user_id,
        "email": email,
        "name":  name,
        "iat":   datetime.now(timezone.utc),
        "exp":   datetime.now(timezone.utc) + JWT_EXPIRY,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm="HS256")


def _decode_token(token):
    return jwt.decode(token, JWT_SECRET, algorithms=["HS256"])


def _get_bearer():
    auth = request.headers.get("Authorization", "")
    if auth.startswith("Bearer "):
        return auth[7:]
    return None

# ── Lazy imports (avoid long startup) ─────────────────────────────────────────
predictor   = None
rec_engine  = None
chatbot_nlp = None


def _get_predictor():
    global predictor
    if predictor is None:
        import predictor as _p
        predictor = _p
    return predictor


def _get_rec_engine():
    global rec_engine
    if rec_engine is None:
        import recommendation_engine as _r
        rec_engine = _r
    return rec_engine


# ═══════════════════════════════════════════════════════════════════════════════
#  ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/")
def index():
    """Serve the landing page."""
    return send_from_directory(".", "index.html")


@app.route("/live")
def live_page():
    """Serve the live dashboard."""
    return send_from_directory(".", "live.html")


@app.route("/settings")
def settings_page():
    """Serve the settings page."""
    return send_from_directory(".", "settings.html")


@app.route("/chatbot")
def chatbot_page():
    return send_from_directory(".", "chatbot.html")


@app.route("/graph")
def graph_page():
    return send_from_directory(".", "graph.html")


@app.route("/health")
def health():
    return jsonify({"status": "ok", "service": "InvestIQ API"})


# ═══════════════════════════════════════════════════════════════════════════════
#  AUTH ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/api/auth/register", methods=["POST"])
def api_register():
    """
    POST /api/auth/register
    Body: {"name": "Nikit", "email": "x@y.com", "password": "secret"}
    Returns: {"token": "...", "user": {id, name, email}}
    """
    body     = request.get_json(force=True, silent=True) or {}
    name     = (body.get("name")     or "").strip()
    email    = (body.get("email")    or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not name or not email or not password:
        return jsonify({"error": "Name, email, and password are required."}), 400
    if len(password) < 6:
        return jsonify({"error": "Password must be at least 6 characters."}), 400

    users = _load_json(USERS_FILE)
    if any(u["email"] == email for u in users):
        return jsonify({"error": "An account with this email already exists."}), 409

    user = {
        "id":            str(uuid.uuid4()),
        "name":          name,
        "email":         email,
        "password_hash": generate_password_hash(password),
        "created_at":    datetime.now(timezone.utc).isoformat(),
    }
    users.append(user)
    _save_json(USERS_FILE, users)

    # Also auto-save their email to visitor_emails list
    _collect_email(email, source="registered", name=name)

    token = _make_token(user["id"], user["email"], user["name"])
    return jsonify({"token": token, "user": {"id": user["id"], "name": name, "email": email}}), 201


@app.route("/api/auth/login", methods=["POST"])
def api_login():
    """
    POST /api/auth/login
    Body: {"email": "x@y.com", "password": "secret"}
    Returns: {"token": "...", "user": {id, name, email}}
    """
    body     = request.get_json(force=True, silent=True) or {}
    email    = (body.get("email")    or "").strip().lower()
    password = (body.get("password") or "").strip()

    if not email or not password:
        return jsonify({"error": "Email and password are required."}), 400

    users = _load_json(USERS_FILE)
    user  = next((u for u in users if u["email"] == email), None)
    if not user or not check_password_hash(user["password_hash"], password):
        return jsonify({"error": "Invalid email or password."}), 401

    token = _make_token(user["id"], user["email"], user["name"])
    return jsonify({"token": token, "user": {"id": user["id"], "name": user["name"], "email": email}})


@app.route("/api/auth/me", methods=["GET"])
def api_me():
    """
    GET /api/auth/me
    Header: Authorization: Bearer <token>
    Returns user profile if token is valid.
    """
    token = _get_bearer()
    if not token:
        return jsonify({"error": "No token provided."}), 401
    try:
        payload = _decode_token(token)
        return jsonify({"user": {"id": payload["sub"], "name": payload["name"], "email": payload["email"]}})
    except jwt.ExpiredSignatureError:
        return jsonify({"error": "Token expired."}), 401
    except jwt.InvalidTokenError:
        return jsonify({"error": "Invalid token."}), 401


# ═══════════════════════════════════════════════════════════════════════════════
#  EMAIL COLLECTION
# ═══════════════════════════════════════════════════════════════════════════════

def _collect_email(email, source="guest", name=None):
    """Internal helper to append an email entry."""
    entries = _load_json(EMAILS_FILE)
    # Avoid duplicate guest entries (registered users may already be present)
    if not any(e["email"] == email for e in entries):
        entry = {
            "email":      email,
            "source":     source,
            "timestamp":  datetime.now(timezone.utc).isoformat(),
        }
        if name:
            entry["name"] = name
        entries.append(entry)
        _save_json(EMAILS_FILE, entries)


@app.route("/api/collect-email", methods=["POST"])
def api_collect_email():
    """
    POST /api/collect-email
    Body: {"email": "x@y.com", "consent": true}
    GDPR-compliant email capture endpoint. Works for both guests and logged-in users.
    """
    body    = request.get_json(force=True, silent=True) or {}
    email   = (body.get("email")   or "").strip().lower()
    consent = body.get("consent", False)
    name    = (body.get("name")    or "").strip() or None

    if not email:
        return jsonify({"error": "Email is required."}), 400
    if not consent:
        return jsonify({"error": "Consent is required."}), 400
    if "@" not in email or "." not in email.split("@")[-1]:
        return jsonify({"error": "Please enter a valid email address."}), 400

    # Determine if this is a registered user by checking their JWT (optional)
    source = "guest"
    token  = _get_bearer()
    if token:
        try:
            payload = _decode_token(token)
            source  = "registered"
            name    = name or payload.get("name")
        except Exception:
            pass

    _collect_email(email, source=source, name=name)
    return jsonify({"success": True, "message": "Thank you! Your email has been saved."})


# ── 1. STOCK PREDICTION ───────────────────────────────────────────────────────

@app.route("/api/predict")
def api_predict():
    """
    GET /api/predict?ticker=TCS.NS&days=30&start=2018-01-01&end=2024-12-31

    Returns JSON forecast dict.
    """
    ticker = request.args.get("ticker", "TCS.NS")
    days   = int(request.args.get("days",  30))
    start  = request.args.get("start", "2018-01-01")
    end    = request.args.get("end",   "2024-12-31")

    try:
        result = _get_predictor().predict(
            ticker=ticker,
            start=start,
            end=end,
            forecast_days=days,
            verbose=False,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc),
                        "traceback": traceback.format_exc()}), 500


# ── 2. PERSONALISED RECOMMENDATION ───────────────────────────────────────────

@app.route("/api/recommend")
def api_recommend():
    """
    GET /api/recommend?age=30&income=80000&expenses=50000
                      &savings=200000&horizon=10&risk=moderate

    Returns personalised financial advice JSON.
    """
    params = {
        "age":      int(request.args.get("age",       30)),
        "income":   float(request.args.get("income",   80_000)),
        "expenses": float(request.args.get("expenses", 50_000)),
        "savings":  float(request.args.get("savings",  2_00_000)),
        "horizon":  int(request.args.get("horizon",    10)),
        "risk":     request.args.get("risk", "moderate"),
    }
    try:
        rec = _get_rec_engine().generate_from_params(**params)
        return jsonify(rec)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── 3. YoY COMPARISON DATA ────────────────────────────────────────────────────

@app.route("/api/compare")
def api_compare():
    """
    GET /api/compare?start=2018&end=2024

    Returns annual returns for IT sector companies as JSON.
    """
    start_yr = int(request.args.get("start", 2018))
    end_yr   = int(request.args.get("end",   2024))

    try:
        from data_ingestion import download_data, DEFAULT_COMPANIES
        import pandas as pd

        data = download_data(verbose=False)
        series_dict = {}
        for name, df in data.items():
            annual = {}
            for yr in range(start_yr, end_yr + 1):
                yr_df = df[df.index.year == yr]["Close"].dropna()
                if len(yr_df) >= 2:
                    ret = (yr_df.iloc[-1] - yr_df.iloc[0]) / yr_df.iloc[0] * 100
                    annual[str(yr)] = round(float(ret), 2)
            series_dict[name] = annual

        return jsonify(series_dict)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── 4. CHATBOT NLP ────────────────────────────────────────────────────────────

@app.route("/api/chat", methods=["POST"])
def api_chat():
    """
    POST /api/chat
    Body: {"message": "What is a SIP?"}

    Returns {"response": "...", "category": "investment"}
    """
    body    = request.get_json(force=True, silent=True) or {}
    message = body.get("message", "").strip()

    if not message:
        return jsonify({"error": "No message provided"}), 400

    response, category = _rule_nlp(message)
    return jsonify({"response": response, "category": category})


# ═══════════════════════════════════════════════════════════════════════════════
#  RULE-BASED NLP (backend copy — also replicated client-side in chatbot.html)
# ═══════════════════════════════════════════════════════════════════════════════

def _rule_nlp(msg: str):
    m = msg.lower()

    # ── Greetings ──────────────────────────────────────────────────────────
    if any(w in m for w in ["hi", "hello", "hey", "good morning", "good evening"]):
        return ("Hello! 👋 I'm InvestIQ AI, your personal financial advisor. "
                "I can help you with stocks, mutual funds, budgeting, risk analysis, "
                "and investment recommendations. What would you like to explore?",
                "greeting")

    # ── Stock basics ───────────────────────────────────────────────────────
    if "what is a stock" in m or "what are stocks" in m:
        return ("📈 A **stock** (or share) represents a unit of ownership in a company. "
                "When you buy a stock, you become a part-owner (shareholder) and can "
                "profit through price appreciation and dividends. Indian stocks are "
                "traded on NSE (National Stock Exchange) and BSE (Bombay Stock Exchange).",
                "stocks")

    if "nse" in m or "bse" in m:
        return ("🏛️ **NSE** (National Stock Exchange) and **BSE** (Bombay Stock Exchange) "
                "are India's two premier stock exchanges. NSE is known for its NIFTY 50 "
                "index, while BSE is home to the SENSEX. Both are regulated by SEBI.",
                "stocks")

    if "dividend" in m:
        return ("💰 **Dividends** are a portion of a company's profits distributed to "
                "shareholders. In India, dividends are taxed at your income slab rate. "
                "Look for companies with consistent dividend-paying track records — "
                "TCS, Infosys, and HCL Tech are known for reliable dividends.",
                "stocks")

    if "stock" in m or "share" in m or "equity" in m:
        return ("📊 **Equities** (stocks/shares) offer higher long-term returns but come "
                "with market volatility. The NIFTY 50 has delivered ~12–14% CAGR over "
                "the past decade. Ideal for investors with a 5+ year horizon.",
                "stocks")

    # ── Mutual Funds ───────────────────────────────────────────────────────
    if "mutual fund" in m or "mf" in m:
        return ("🏦 **Mutual Funds** pool money from many investors to invest in a "
                "diversified portfolio of stocks, bonds, or other assets. They are "
                "managed by professional fund managers. In India, they are regulated "
                "by SEBI. Categories include: Equity, Debt, Hybrid, and Index Funds.",
                "investment")

    if "sip" in m:
        return ("📅 **SIP (Systematic Investment Plan)** lets you invest a fixed amount "
                "monthly in a mutual fund — as low as ₹500/month. SIPs average out "
                "market volatility through rupee-cost averaging, making them ideal for "
                "long-term wealth creation. Starting early maximises compounding benefits.",
                "investment")

    if "etf" in m:
        return ("📦 **ETFs (Exchange-Traded Funds)** track an index (like NIFTY 50) and "
                "trade on exchanges like stocks. They offer low cost (~0.1–0.2% expense "
                "ratio), diversification, and liquidity. NIFTY BeES and GOLDBEES are "
                "popular ETFs in India.",
                "investment")

    if "index fund" in m or "index  " in m:
        return ("📉 **Index Funds** passively track a market index (e.g., NIFTY 50, "
                "SENSEX). They have very low expense ratios (~0.1%) and consistently "
                "outperform most actively managed funds over 10+ years. Ideal for "
                "passive investors.",
                "investment")

    if "bond" in m or "g-sec" in m or "government bond" in m:
        return ("📜 **Bonds** are debt instruments — you lend money to governments or "
                "corporations in return for periodic interest payments. Government "
                "Securities (G-Secs) are the safest. Sovereign Gold Bonds (SGBs) offer "
                "2.5% p.a. interest + gold price appreciation — tax-free on maturity.",
                "investment")

    if "ppf" in m or "public provident fund" in m:
        return ("🔒 **PPF (Public Provident Fund)** is a government-backed savings scheme "
                "offering 7.1% p.a. interest, fully tax-free (EEE — Exempt-Exempt-Exempt). "
                "Lock-in: 15 years. Maximum: ₹1.5L/year. Excellent for conservative "
                "investors seeking safe, tax-efficient returns.",
                "investment")

    if "elss" in m or "tax sav" in m or "80c" in m:
        return ("🧾 **ELSS (Equity Linked Savings Scheme)** qualifies for ₹1.5L deduction "
                "under Section 80C, saving up to ₹46,800 in taxes. Lock-in: 3 years "
                "(shortest among 80C options). Offers equity market returns (~12–15% "
                "CAGR historically). Best of both worlds — tax saving + wealth creation.",
                "investment")

    if "invest" in m:
        return ("💼 Smart investing starts with defining your **goal, horizon, and risk "
                "appetite**. General rule: 100 minus your age = equity allocation %. "
                "Diversify across equity, debt, and gold. Start early — ₹5,000/month "
                "at 12% CAGR for 30 years grows to ₹1.7 crore!",
                "investment")

    # ── Risk ───────────────────────────────────────────────────────────────
    if "risk" in m or "volatile" in m or "volatility" in m:
        return ("⚠️ **Investment risk** is the possibility of losing value. Key types:\n"
                "• **Market risk** — overall market decline\n"
                "• **Credit risk** — bond issuer defaults\n"
                "• **Liquidity risk** — can't sell when needed\n"
                "• **Inflation risk** — returns don't beat inflation\n\n"
                "Diversification across asset classes is the best risk management tool.",
                "risk")

    if "diversif" in m or "portfolio" in m or "allocation" in m:
        return ("🗂️ **Diversification** spreads investments across assets to reduce risk. "
                "A balanced portfolio for a moderate-risk investor:\n"
                "• 55% Equity (index funds + direct stocks)\n"
                "• 30% Debt (bonds, PPF, FDs)\n"
                "• 8% Gold (SGBs or Gold ETFs)\n"
                "• 7% Cash / Liquid Fund\n\n"
                "Rebalance every 6–12 months to maintain target allocation.",
                "risk")

    if "hedge" in m or "hedging" in m:
        return ("🛡️ **Hedging** protects your portfolio against downside risk. Common "
                "strategies:\n• Buy put options on index positions\n"
                "• Allocate to gold (negatively correlated to equities)\n"
                "• Hold short-duration debt during uncertain markets\n"
                "• Invest in defensive sectors (FMCG, Pharma) for stability.",
                "risk")

    # ── Market Analysis ─────────────────────────────────────────────────────
    if "bull" in m and "market" in m:
        return ("🐂 A **bull market** is a sustained period of rising stock prices "
                "(typically 20%+ gain from recent lows). Bull markets are driven by "
                "strong GDP growth, low unemployment, and investor optimism. "
                "India's NIFTY 50 has been broadly bullish from 2020 to 2024.",
                "market")

    if "bear" in m and "market" in m:
        return ("🐻 A **bear market** is defined as a 20%+ drop from recent highs. "
                "It often signals economic slowdown or recession. Strategy during "
                "bear markets: continue SIPs (buy cheap), avoid panic selling, "
                "and consider increasing debt allocation temporarily.",
                "market")

    if "p/e" in m or "pe ratio" in m or "price to earning" in m:
        return ("📐 The **P/E Ratio** (Price-to-Earnings) measures how much investors "
                "pay per rupee of earnings. NIFTY 50 P/E of 20–22 is historically fair; "
                ">25 suggests overvaluation; <18 suggests undervaluation. "
                "Compare within the same sector for meaningful insights.",
                "market")

    if "market cap" in m or "capitalisation" in m or "capitalization" in m:
        return ("🏢 **Market Capitalisation** = Share Price × Total Shares. Categories:\n"
                "• Large-cap: >₹20,000 Cr (TCS, Infosys) — stable, lower risk\n"
                "• Mid-cap: ₹5,000–20,000 Cr — growth potential\n"
                "• Small-cap: <₹5,000 Cr — high risk, high reward",
                "market")

    if "inflation" in m:
        return ("📊 **Inflation** erodes purchasing power over time. India's CPI inflation "
                "averages ~5–6% p.a. Your investments must beat this to grow wealth. "
                "Equities historically return 12–14% CAGR — well above inflation. "
                "FDs and savings accounts often fail to beat inflation after tax.",
                "market")

    if "nifty" in m or "sensex" in m:
        return ("📈 **NIFTY 50** tracks the top 50 companies on NSE. "
                "**SENSEX** tracks the top 30 on BSE. Both are key Indian market "
                "benchmarks. NIFTY has returned ~12% CAGR over the past 20 years, "
                "making it an excellent long-term investment via index funds.",
                "market")

    # ── Personal Finance ────────────────────────────────────────────────────
    if "budget" in m or "budgeting" in m or "expenses" in m:
        return ("💳 **50-30-20 Budgeting Rule** (recommended for India):\n"
                "• 50% → Needs (rent, groceries, utilities, EMIs)\n"
                "• 30% → Wants (dining, entertainment, travel)\n"
                "• 20% → Savings & Investments (SIPs, PPF, emergency fund)\n\n"
                "Track spending with apps like Money View, Walnut, or ET Money.",
                "personal_finance")

    if "savings" in m or "save" in m or "emergency" in m:
        return ("🏦 **Emergency Fund** is your financial safety net — 6 months of "
                "living expenses in a liquid instrument (liquid mutual fund or "
                "savings account). Build this before any long-term investing. "
                "After that, automate savings via SIPs to stay consistent.",
                "personal_finance")

    if "emi" in m or "loan" in m or "debt" in m:
        return ("🏠 **EMI (Equated Monthly Instalment)** tip: Keep total EMIs below "
                "40% of take-home income. Prioritise high-interest debt (personal loans, "
                "credit cards at 18–24% p.a.) over low-interest loans (home loans at "
                "8–9% p.a.). Never invest in equities with borrowed money.",
                "personal_finance")

    if "tax" in m:
        return ("🧾 **Key Indian Tax Saving Options (FY 2025–26):**\n"
                "• Section 80C (₹1.5L): ELSS, PPF, EPF, NSC, Tax-saver FD\n"
                "• Section 80D (₹25K–75K): Health insurance premium\n"
                "• HRA exemption for salaried employees\n"
                "• LTCG on equities: 10% above ₹1L gain\n"
                "• STCG on equities: 15%\n\n"
                "Consider the New Tax Regime if deductions are <₹3.75L.",
                "personal_finance")

    if "retire" in m or "retirement" in m:
        return ("🎯 **Retirement Planning** — the 25x rule: you need 25× your annual "
                "expenses saved to retire comfortably. For ₹6L/year expenses → ₹1.5 Cr "
                "corpus. Invest in NPS (Tier-I) for additional 80CCD(1B) benefit of "
                "₹50,000 deduction. Start early — time is your greatest asset!",
                "personal_finance")

    # ── AI & This System ────────────────────────────────────────────────────
    if "how" in m and ("predict" in m or "forecast" in m or "work" in m):
        return ("🤖 **InvestIQ uses 4 ML models:**\n"
                "1. **LSTM (Bidirectional)** — deep learning on price sequences\n"
                "2. **XGBoost** — gradient boosting with early stopping\n"
                "3. **Random Forest** — ensemble of decision trees\n"
                "4. **Polynomial Regression** — baseline trend model\n\n"
                "An **ensemble** weighted average (LSTM 40% + XGBoost 35% + RF 25%) "
                "gives the final forecast. Features include RSI, MACD, Bollinger Bands, "
                "OBV, and 30+ more technical indicators.",
                "ai")

    if "accuracy" in m or "how accurate" in m:
        return ("📐 **Model Accuracy** varies by stock and time period. Typical results:\n"
                "• LSTM: R² ~0.92–0.97, MAPE ~2–4%\n"
                "• XGBoost: R² ~0.95–0.98, MAPE ~1.5–3%\n"
                "• Random Forest: R² ~0.93–0.97, MAPE ~2–4%\n\n"
                "⚠️ Note: Past accuracy doesn't guarantee future performance. "
                "Always combine AI predictions with your own due diligence.",
                "ai")

    if "ai" in m or "machine learning" in m or "model" in m:
        return ("🧠 InvestIQ's AI engine is built on **Python** with TensorFlow/Keras "
                "(LSTM), XGBoost, and scikit-learn. It processes 30+ technical "
                "indicators per stock and provides a weighted ensemble forecast with "
                "confidence scores. The system can be retrained with fresh data anytime.",
                "ai")

    # ── App usage ────────────────────────────────────────────────────────────
    if "help" in m or "what can you do" in m or "what do you do" in m:
        return ("🌟 **I can help you with:**\n"
                "• Stock analysis & price forecasts (TCS, Infosys, Wipro…)\n"
                "• Mutual fund & SIP guidance\n"
                "• Risk profiling & portfolio allocation\n"
                "• Budgeting & personal finance tips\n"
                "• Market concepts (P/E, bull/bear, inflation)\n"
                "• Tax-saving investment strategies\n"
                "• Retirement planning\n\n"
                "Just type your question or use the quick-reply buttons below!",
                "help")

    if "thank" in m or "bye" in m or "goodbye" in m:
        return ("😊 You're welcome! Remember: **consistent investing beats market timing**. "
                "Come back anytime for financial insights. Happy investing! 🚀",
                "farewell")

    # ── Fallback ─────────────────────────────────────────────────────────────
    return ("🤔 I didn't fully understand that. Try asking about:\n"
            "stocks, mutual funds, SIPs, risk, budgeting, tax saving, "
            "retirement, market trends, or how our AI predictions work.\n\n"
            "Or type **help** to see everything I can do!",
            "unknown")


# ═══════════════════════════════════════════════════════════════════════════════
#  LIVE DATA API ENDPOINTS
# ═══════════════════════════════════════════════════════════════════════════════

# All tracked stocks (matches the live dashboard)
LIVE_TICKERS = {
    "Tata Consultancy":  "TCS.NS",
    "Infosys Limited":   "INFY.NS",
    "Wipro Limited":     "WIPRO.NS",
    "HCL Technologies":  "HCLTECH.NS",
    "Reliance Ind.":     "RELIANCE.NS",
    "HDFC Bank":         "HDFC.NS",
    "ICICI Bank":        "ICICIBANK.NS",
    "Bajaj Finance":     "BAJFINANCE.NS",
    "ITC Ltd":           "ITC.NS",
    "Maruti Suzuki":     "MARUTI.NS",
}

INDEX_TICKERS = {
    "NIFTY 50":    "^NSEI",
    "SENSEX":      "^BSESN",
    "BANK NIFTY":  "^NSEBANK",
    "India VIX":   "^INDIAVIX",
}


@app.route("/api/live-quotes")
def api_live_quotes():
    """
    GET /api/live-quotes
    Returns current prices + day change for all tracked stocks.
    Uses yfinance fast_info for speed; falls back to simulated data.
    """
    try:
        import yfinance as yf
        results = {}
        for name, ticker in LIVE_TICKERS.items():
            try:
                t = yf.Ticker(ticker)
                info = t.fast_info
                price = float(info.get("lastPrice", 0) or info.get("regularMarketPrice", 0) or 0)
                prev  = float(info.get("previousClose", price) or price)
                if price == 0:
                    hist = t.history(period="2d")
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                        prev  = float(hist["Close"].iloc[0]) if len(hist) > 1 else price
                pct = ((price - prev) / prev * 100) if prev else 0
                results[ticker] = {
                    "name": name,
                    "symbol": ticker,
                    "price": round(price, 2),
                    "prevClose": round(prev, 2),
                    "changePct": round(pct, 2),
                }
            except Exception:
                pass
        if not results:
            return jsonify({"error": "No data available", "fallback": True}), 200
        return jsonify(results)
    except ImportError:
        return jsonify({"error": "yfinance not installed", "fallback": True}), 200
    except Exception as exc:
        return jsonify({"error": str(exc), "fallback": True}), 200


@app.route("/api/market-indices")
def api_market_indices():
    """
    GET /api/market-indices
    Returns current values for NIFTY 50, SENSEX, BANK NIFTY, India VIX.
    """
    try:
        import yfinance as yf
        results = {}
        for name, ticker in INDEX_TICKERS.items():
            try:
                t = yf.Ticker(ticker)
                info = t.fast_info
                price = float(info.get("lastPrice", 0) or info.get("regularMarketPrice", 0) or 0)
                prev  = float(info.get("previousClose", price) or price)
                if price == 0:
                    hist = t.history(period="2d")
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                        prev  = float(hist["Close"].iloc[0]) if len(hist) > 1 else price
                pct = ((price - prev) / prev * 100) if prev else 0
                results[name] = {
                    "symbol": ticker,
                    "value": round(price, 2),
                    "prevClose": round(prev, 2),
                    "changePct": round(pct, 2),
                }
            except Exception:
                pass
        if not results:
            return jsonify({"error": "No data", "fallback": True}), 200
        return jsonify(results)
    except ImportError:
        return jsonify({"error": "yfinance not installed", "fallback": True}), 200
    except Exception as exc:
        return jsonify({"error": str(exc), "fallback": True}), 200


@app.route("/api/search-stocks")
def api_search_stocks():
    """
    GET /api/search-stocks?q=tcs
    Searches tracked stocks by name or symbol. Returns matching results with live prices.
    """
    query = (request.args.get("q", "") or "").strip().lower()
    if not query:
        return jsonify([])

    # Extended stock list for broader search
    ALL_STOCKS = {
        **LIVE_TICKERS,
        "Tech Mahindra":     "TECHM.NS",
        "Sun Pharma":        "SUNPHARMA.NS",
        "Tata Motors":       "TATAMOTORS.NS",
        "L&T":               "LT.NS",
        "Axis Bank":         "AXISBANK.NS",
        "Kotak Bank":        "KOTAKBANK.NS",
        "SBI":               "SBIN.NS",
        "Asian Paints":      "ASIANPAINT.NS",
        "Hindustan Unilever": "HINDUNILVR.NS",
        "Bharti Airtel":     "BHARTIARTL.NS",
        "Power Grid":        "POWERGRID.NS",
        "Titan Company":     "TITAN.NS",
        "UltraTech Cement":  "ULTRACEMCO.NS",
        "Nestle India":      "NESTLEIND.NS",
        "Adani Ports":       "ADANIPORTS.NS",
    }

    matches = []
    for name, ticker in ALL_STOCKS.items():
        if query in name.lower() or query in ticker.lower():
            matches.append({"name": name, "symbol": ticker})

    # Try to fetch live prices for matches
    try:
        import yfinance as yf
        for m in matches[:10]:  # limit to 10
            try:
                t = yf.Ticker(m["symbol"])
                info = t.fast_info
                price = float(info.get("lastPrice", 0) or info.get("regularMarketPrice", 0) or 0)
                prev  = float(info.get("previousClose", price) or price)
                if price == 0:
                    hist = t.history(period="1d")
                    if not hist.empty:
                        price = float(hist["Close"].iloc[-1])
                pct = ((price - prev) / prev * 100) if prev else 0
                m["price"] = round(price, 2)
                m["changePct"] = round(pct, 2)
            except Exception:
                m["price"] = 0
                m["changePct"] = 0
    except ImportError:
        pass

    return jsonify(matches[:10])


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*55)
    print("  🚀  InvestIQ API — starting on http://localhost:5000")
    print("="*55)
    print("  Endpoints:")
    print("    GET  /api/predict?ticker=TCS.NS&days=30")
    print("    GET  /api/recommend?age=30&income=80000&risk=moderate")
    print("    GET  /api/compare?start=2018&end=2024")
    print("    GET  /api/live-quotes")
    print("    GET  /api/market-indices")
    print("    GET  /api/search-stocks?q=tcs")
    print("    POST /api/chat  (body: {\"message\": \"...\"})")
    print("="*55 + "\n")
    app.run(debug=True, port=5000, host="0.0.0.0")

