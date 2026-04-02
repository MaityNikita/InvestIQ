// ══════════════════════════════════════════════════
//  InvestIQ Chatbot Brain — Comprehensive Knowledge
// ══════════════════════════════════════════════════
const BRAIN = {};

// ── Stock-specific analysis using live data ──
BRAIN.analyzeStock = function(query, STOCKS, S, HOLDINGS, fmt){
  const found = STOCKS.find(s=> s.name.toLowerCase().includes(query) || s.symbol.toLowerCase().includes(query));
  if(!found) return null;
  const st = S[found.symbol];
  const pct = ((st.price-found.base)/found.base*100).toFixed(2);
  const isUp = parseFloat(pct) >= 0;
  const riskLevel = st.risk >= 70 ? '🔴 High Risk' : st.risk >= 45 ? '🟡 Moderate Risk' : '🟢 Low Risk';
  const h = st.hist.slice(-10);
  const trend5 = h.length>=5 ? ((h[h.length-1]-h[h.length-5])/h[h.length-5]*100).toFixed(2) : '0';
  const vol = Math.abs(Math.max(...h)-Math.min(...h))/h[0]*100;
  const holding = HOLDINGS.find(x=>x.symbol===found.symbol);
  let holdingInfo = '';
  if(holding){
    const pnl = (st.price - holding.avgBuy) * holding.qty;
    const pnlP = ((st.price - holding.avgBuy)/holding.avgBuy*100).toFixed(2);
    holdingInfo = `\n\n💼 **Your Position:** ${holding.qty} shares @ ₹${holding.avgBuy}\n📊 P&L: **${pnl>=0?'+':''}₹${fmt(Math.abs(pnl))}** (${pnl>=0?'+':''}${pnlP}%)`;
  }
  let signal = '';
  if(st.risk >= 65 && parseFloat(pct) > 10) signal = '\n\n🔔 **Signal:** Consider booking partial profits — elevated risk with strong gains.';
  else if(st.risk < 35 && parseFloat(pct) < -3) signal = '\n\n🔔 **Signal:** Potential buying opportunity — low risk with a temporary dip.';
  else if(isUp && st.risk < 45) signal = '\n\n🔔 **Signal:** Strong position — low risk with positive momentum. Hold or accumulate.';
  else if(!isUp && st.risk >= 55) signal = '\n\n🔔 **Signal:** Caution advised — negative momentum with elevated risk.';

  return `**${found.name}** (${found.symbol}) — ${found.sector}\n\n` +
    `💰 Current Price: **₹${fmt(st.price)}**\n` +
    `${isUp?'📈':'📉'} Day Change: **${isUp?'+':''}${pct}%**\n` +
    `📊 5-Tick Trend: **${parseFloat(trend5)>=0?'+':''}${trend5}%**\n` +
    `🎯 Risk Score: **${Math.round(st.risk)}/100** ${riskLevel}\n` +
    `📉 Volatility: **${vol.toFixed(2)}%** (recent range)` +
    holdingInfo + signal;
};

// ── Comprehensive Knowledge Base ──
BRAIN.KB = [
  // Stock specific - any stock name mentioned
  {match: s => {
    const stockNames = ['tcs','infosys','wipro','hcl','reliance','hdfc','icici','bajaj','itc','maruti',
      'tata','infy','nifty bank'];
    return stockNames.some(n => s.includes(n));
  }, handler: (m, ctx) => {
    const q = m.replace(/what|is|the|current|situation|of|explain|tell|me|about|show|how|status|analysis|analyze|stock/gi,'').trim();
    const result = BRAIN.analyzeStock(q, ctx.STOCKS, ctx.S, ctx.HOLDINGS, ctx.fmt);
    if(result) return result;
    return null; // fall through
  }},

  // Intraday / Day Trading
  {match: s => /intraday|day.?trad|scalp/i.test(s),
   resp: `**📊 Intraday Trading Explained**\n\nIntraday (day) trading means buying and selling stocks **within the same trading day** — no positions are held overnight.\n\n**Key Concepts:**\n• **Intraday Momentum** — the directional force of price movement during the day, shown in the chart above\n• **Volume** — higher volume = stronger price moves\n• **Support/Resistance** — price levels where stocks tend to bounce or reverse\n• **VWAP** (Volume Weighted Avg Price) — institutional benchmark; price above VWAP = bullish\n\n**Strategies:**\n1. **Breakout Trading** — enter when price crosses a key resistance with volume\n2. **Mean Reversion** — buy dips to VWAP, sell rallies above it\n3. **Momentum Scalping** — ride strong 5-min trends, exit on reversal\n\n**Risk Management:**\n• Never risk >2% of capital per trade\n• Always use stop-loss orders\n• Avoid trading during 9:15-9:30 (high volatility) unless experienced\n\n⚠️ Intraday trading is high-risk. 90% of retail day traders lose money. Consider swing trading or investing instead.`},

  // Swing Trading
  {match: s => /swing.?trad/i.test(s),
   resp: `**🔄 Swing Trading**\n\nSwing trading holds stocks for **2-15 days** to capture short-term price swings.\n\n**How it works:**\n1. Identify stocks with clear trend (use 20-day & 50-day moving averages)\n2. Enter on pullbacks to support or breakouts above resistance\n3. Hold for the expected swing (typically 3-8% move)\n4. Exit with trailing stop-loss\n\n**Best indicators:** RSI, MACD, Bollinger Bands, Volume\n**Capital needed:** ₹50,000+ recommended\n**Time commitment:** 30 mins/day for analysis\n\n✅ Lower risk than intraday, higher returns than pure investing if done right.`},

  // Options & Futures
  {match: s => /option|call.?put|strike|expiry|futures|derivative|f\&o/i.test(s),
   resp: `**📋 Options & Futures (F&O)**\n\n**Futures:** A contract to buy/sell an asset at a predetermined price on a future date. Used for hedging or speculation. Lot sizes vary (NIFTY = 25 units).\n\n**Options:**\n• **Call Option** — right to BUY at strike price (bullish bet)\n• **Put Option** — right to SELL at strike price (bearish bet / insurance)\n• **Premium** — price you pay for the option\n• **Strike Price** — the target price in the contract\n• **Expiry** — options expire on last Thursday of each month\n\n**Common Strategies:**\n• **Covered Call** — hold stock + sell call = earn premium income\n• **Protective Put** — hold stock + buy put = insurance against downfall\n• **Bull Call Spread** — buy lower strike call + sell higher strike call\n• **Iron Condor** — profit from low volatility\n\n⚠️ Options lose value daily (theta decay). Only trade F&O after thorough education. SEBI requires ₹10L+ net worth for options trading.`},

  // Technical Analysis
  {match: s => /technical.?analy|rsi|macd|bollinger|moving.?average|support|resistance|candle|chart.?pattern|indicator/i.test(s),
   resp: `**📐 Technical Analysis**\n\nTechnical analysis studies **price patterns and indicators** to predict future movements.\n\n**Key Indicators:**\n• **RSI** (Relative Strength Index): >70 = overbought (sell signal), <30 = oversold (buy signal)\n• **MACD** (Moving Average Convergence Divergence): signal line crossover = trend change\n• **Bollinger Bands**: price touching upper band = overbought, lower band = oversold\n• **Moving Averages**: 50-day MA crossing above 200-day MA = "Golden Cross" (bullish)\n• **Volume**: confirms price moves — high volume breakout = genuine\n\n**Chart Patterns:**\n• **Head & Shoulders** — reversal pattern (bearish after uptrend)\n• **Double Bottom** — reversal pattern (bullish after downtrend)\n• **Cup & Handle** — continuation pattern (bullish)\n• **Triangle** (ascending/descending) — breakout pattern\n\n**Support & Resistance:**\n• **Support** — price level where buying interest emerges (floor)\n• **Resistance** — price level where selling pressure emerges (ceiling)\n• Broken resistance becomes new support and vice versa\n\nInvestIQ uses **30+ technical indicators** including RSI, MACD, OBV, and Bollinger Bands in its LSTM model.`},

  // Fundamental Analysis
  {match: s => /fundamental|p\/e|pe.?ratio|eps|revenue|earning|balance.?sheet|valuation|intrinsic|book.?value|roe|roce|debt.?to.?equity/i.test(s),
   resp: `**📊 Fundamental Analysis**\n\nFundamental analysis evaluates a company's **financial health and intrinsic value**.\n\n**Key Metrics:**\n• **P/E Ratio** = Price / Earnings per Share. NIFTY avg ~22. Below 15 = potentially undervalued\n• **EPS** (Earnings Per Share) = Net Profit / Total Shares. Higher = more profitable\n• **ROE** (Return on Equity) = Net Income / Shareholder Equity. >15% = excellent\n• **ROCE** (Return on Capital Employed) = EBIT / Capital Employed. >20% = superb\n• **Debt-to-Equity** = Total Debt / Total Equity. <1 preferred, <0.5 = conservative\n• **P/B Ratio** = Price / Book Value. <1 = trading below asset value\n\n**What to check:**\n1. Revenue growth (consistent 10%+ YoY is strong)\n2. Profit margins (stable or expanding)\n3. Cash flow (positive free cash flow is essential)\n4. Promoter holding (>50% and increasing = good sign)\n5. Debt levels (declining debt is bullish)\n\n💡 Combine fundamental + technical analysis for the best results.`},

  // SIP
  {match: s => /sip|systematic.?invest/i.test(s),
   resp: `**💰 SIP (Systematic Investment Plan)**\n\nSIP invests a **fixed amount every month** into mutual funds automatically.\n\n**Why SIP works:**\n• **Rupee Cost Averaging** — buy more units when prices are low, fewer when high\n• **Compounding** — ₹10,000/month at 12% for 20 years = **₹99.9 Lakhs** (invested only ₹24L)\n• **Discipline** — automates investing, removes emotional decisions\n• **Flexible** — start with just ₹500/month\n\n**Best SIP strategy:**\n1. Choose 2-3 diversified equity mutual funds\n2. Set up auto-debit on salary day\n3. Increase SIP by 10% every year (step-up SIP)\n4. Never stop SIP during market crashes — that's when you get the best prices!\n\n**Top SIP categories:**\n• Large Cap Index Fund (NIFTY 50) — safest\n• Flexi Cap Fund — mix of large/mid/small\n• ELSS — tax saving + equity growth\n\n📌 Rule of thumb: Invest at least **20% of your income** via SIP.`},

  // Mutual Funds
  {match: s => /mutual.?fund|nav|amc|expense.?ratio|direct.?plan|regular.?plan/i.test(s),
   resp: `**📦 Mutual Funds Guide**\n\nMutual funds pool money from investors to invest in diversified portfolios managed by professionals.\n\n**Types:**\n• **Equity Funds** — invest in stocks (12-15% CAGR historically)\n• **Debt Funds** — invest in bonds (6-8% returns, lower risk)\n• **Hybrid Funds** — mix of equity + debt\n• **Index Funds** — passively track NIFTY/SENSEX (lowest cost)\n• **ELSS** — tax-saving equity funds (3-year lock-in)\n\n**Key terms:**\n• **NAV** = Net Asset Value (price per unit)\n• **Expense Ratio** — annual fee (lower = better; <0.5% for index funds)\n• **Direct Plan** — buy directly from AMC (no commission, ~1% cheaper)\n• **Regular Plan** — bought via distributor (higher expense ratio)\n\n**How to choose:**\n1. Always pick **Direct Growth** plans\n2. Check 5-year and 10-year returns\n3. Compare expense ratios\n4. Look for consistency across market cycles\n\n💡 For beginners: Start with a NIFTY 50 Index Fund SIP.`},

  // ETF
  {match: s => /etf|exchange.?traded/i.test(s),
   resp: `**📊 ETFs (Exchange-Traded Funds)**\n\nETFs track an index and trade on stock exchanges like regular shares.\n\n**Popular Indian ETFs:**\n• **NIFTY BeES** — tracks NIFTY 50 (expense ratio ~0.05%)\n• **GOLDBEES** — tracks gold prices\n• **BANKBEES** — tracks Bank NIFTY\n• **MON100** — tracks NASDAQ 100 (US tech exposure)\n\n**ETF vs Mutual Fund:**\n• ETFs trade real-time on NSE/BSE; MFs settle at day-end NAV\n• ETFs have lower expense ratios\n• ETFs need a demat account; MFs don't\n• MFs have SIP option; ETFs need manual buying\n\n💡 Best of both worlds: SIP into an index mutual fund for convenience, or buy ETFs for lower costs.`},

  // IPO
  {match: s => /ipo|initial.?public|listing|grey.?market/i.test(s),
   resp: `**🏷️ IPO (Initial Public Offering)**\n\nAn IPO is when a private company first sells shares to the public on the stock exchange.\n\n**How to apply:**\n1. Open a demat account (Zerodha, Groww, Upstox)\n2. Apply via UPI-based ASBA (amount blocked, not debited)\n3. Lot sizes usually ₹14,000-₹15,000 for retail investors\n4. Allotment is via lottery if oversubscribed\n\n**How to evaluate an IPO:**\n• Check P/E vs industry peers\n• Revenue growth trend (3+ years)\n• Promoter track record and use of IPO proceeds\n• Grey Market Premium (GMP) — indicates market sentiment\n• Avoid IPOs where promoters are selling large stakes (OFS)\n\n**Key tips:**\n• Don't invest based on hype alone\n• Check DRHP (Draft Red Herring Prospectus) on SEBI website\n• Not all IPOs give listing gains — many trade below issue price\n\n⚠️ IPO investing is speculative. Use only 5-10% of your portfolio for IPOs.`},

  // Market indices explanation
  {match: s => /nifty|sensex|bank.?nifty|vix|index|indices|benchmark/i.test(s),
   resp: `**📈 Indian Market Indices**\n\n**NIFTY 50:** Top 50 companies on NSE by market cap. India's primary benchmark. 12% CAGR over 20 years.\n\n**SENSEX:** Top 30 companies on BSE (oldest exchange, est. 1875). Closely tracks NIFTY.\n\n**BANK NIFTY:** Top 12 banking stocks. More volatile than NIFTY — popular for F&O trading.\n\n**India VIX:** Measures expected market **volatility** over next 30 days.\n• VIX < 13 = calm market (good for investing)\n• VIX 13-20 = normal volatility\n• VIX > 20 = fear/uncertainty (caution)\n• VIX > 30 = extreme fear (potential buying opportunity!)\n\n**Live on your dashboard:**\nCheck the stats cards at the top — they show real-time values for all four indices with % changes. Click any card to see its intraday momentum chart.`},

  // Bull/Bear market
  {match: s => /bull|bear|market.?sentiment|market.?crash|correction|rally/i.test(s),
   resp: `**🐂 Bull vs 🐻 Bear Markets**\n\n**Bull Market:** Sustained rise of 20%+ from recent lows\n• Driven by GDP growth, low interest rates, corporate earnings\n• Strategy: Stay invested, increase equity allocation\n\n**Bear Market:** Sustained fall of 20%+ from recent highs\n• Triggered by recession fears, rate hikes, global shocks\n• Strategy: Continue SIPs, buy quality stocks at discounts\n\n**Correction:** A 10-20% drop (healthy and normal)\n**Rally:** A sharp short-term rise (may or may not sustain)\n**Crash:** A sudden 20%+ drop over days/weeks\n\n**What to do during a crash:**\n1. Don't panic sell — history shows markets always recover\n2. Have cash reserves ready to deploy\n3. Increase SIP amounts if possible\n4. Focus on quality large-caps with strong fundamentals\n\n📌 Time in the market > Timing the market.`},

  // Dividend
  {match: s => /dividend|yield|payout|ex.?date/i.test(s),
   resp: `**💵 Dividends Explained**\n\nDividends are a share of company profits distributed to shareholders.\n\n**Key Terms:**\n• **Dividend Yield** = Annual Dividend / Share Price × 100. >3% = good yield stock\n• **Ex-Date** — buy before this date to receive the dividend\n• **Record Date** — date to check shareholder eligibility\n• **Payout Ratio** — % of profits paid as dividends. 30-50% is healthy\n\n**Top Dividend Stocks in India:**\n• ITC (yield ~3.5%), Coal India (~7%), Power Grid (~5%), ONGC (~5%), Hindustan Zinc (~6%)\n\n**Tax on Dividends:** Taxed at your income tax slab rate (since 2020). TDS of 10% if dividend > ₹5,000/year.\n\n💡 Dividend stocks are great for passive income but don't chase yield alone — check if the company can sustain its payout.`},

  // Investment basics
  {match: s => /invest|beginner|start|where.?to.?begin|first.?time|new.?to/i.test(s),
   resp: `**💼 Investment Guide for Beginners**\n\n**Step 1: Build an Emergency Fund**\n6 months of expenses in a liquid fund or savings account\n\n**Step 2: Get Insurance**\nTerm life insurance (₹1 Cr cover for ~₹12,000/year at age 25) + Health insurance (₹10L family floater)\n\n**Step 3: Start Investing**\n• **Safest start:** NIFTY 50 Index Fund SIP — ₹5,000/month\n• **Tax saving:** ELSS fund (Section 80C benefit)\n• **Gold exposure:** Sovereign Gold Bonds (2.5% interest + gold appreciation)\n\n**Asset Allocation (age-based):**\n• 20s: 80% equity, 15% debt, 5% gold\n• 30s: 70% equity, 20% debt, 10% gold\n• 40s: 55% equity, 35% debt, 10% gold\n\n**Golden Rules:**\n1. Start early — compounding is magical\n2. Never invest borrowed money\n3. Diversify across asset classes\n4. Stay consistent — don't stop SIPs during crashes\n5. Review portfolio every 6 months\n\n📌 ₹10,000/month at 12% for 25 years = **₹1.89 Crore!**`},

  // Risk management
  {match: s => /risk|volatil|safe|danger|protect|hedge|diversif|allocat/i.test(s),
   handler: (m, ctx) => {
     const risky = ctx.STOCKS.filter(s=> ctx.S[s.symbol].risk >= 55).sort((a,b)=> ctx.S[b.symbol].risk - ctx.S[a.symbol].risk);
     const safe = ctx.STOCKS.filter(s=> ctx.S[s.symbol].risk < 55).sort((a,b)=> ctx.S[a.symbol].risk - ctx.S[b.symbol].risk);
     return `**⚠️ Risk Analysis (Live Dashboard)**\n\n` +
       (risky.length ? `🔴 **High Risk Stocks:** ${risky.map(s=>`${s.name.split(' ')[0]} (${Math.round(ctx.S[s.symbol].risk)}/100)`).join(', ')}\n\n` : '') +
       (safe.length ? `🟢 **Low Risk Stocks:** ${safe.map(s=>`${s.name.split(' ')[0]} (${Math.round(ctx.S[s.symbol].risk)}/100)`).join(', ')}\n\n` : '') +
       `**Risk Management Principles:**\n` +
       `• Never put >10% of portfolio in a single stock\n` +
       `• Use stop-loss orders (7-8% below buy price)\n` +
       `• Diversify across sectors: IT, Finance, FMCG, Pharma, Energy\n` +
       `• Hold 20-30% in debt instruments for stability\n` +
       `• Rebalance every 6 months`;
   }},

  // P&L / Portfolio
  {match: s => /p.?&.?l|pnl|profit|loss|portfolio|holding|position|my stock|my invest/i.test(s),
   handler: (m, ctx) => {
     let totalInv=0, totalCur=0;
     const lines = ctx.HOLDINGS.map(h=>{
       const st=ctx.S[h.symbol]; if(!st) return '';
       const stock = ctx.STOCKS.find(s=>s.symbol===h.symbol);
       const pnl = (st.price - h.avgBuy) * h.qty;
       const pnlP = ((st.price - h.avgBuy)/h.avgBuy*100).toFixed(2);
       totalInv += h.avgBuy*h.qty; totalCur += st.price*h.qty;
       const isUp = pnl >= 0;
       return `• **${stock.name}**: ₹${ctx.fmt(st.price)} | ${isUp?'+':''}₹${ctx.fmt(Math.abs(pnl))} (${isUp?'+':''}${pnlP}%)`;
     });
     const totalPnl = totalCur - totalInv;
     const isUp = totalPnl >= 0;
     return `**💼 Portfolio Summary (Live)**\n\n${lines.join('\n')}\n\n` +
       `📊 Total Invested: **₹${ctx.fmt(totalInv)}**\n` +
       `📈 Current Value: **₹${ctx.fmt(totalCur)}**\n` +
       `${isUp?'✅':'❌'} Total P&L: **${isUp?'+':'−'}₹${ctx.fmt(Math.abs(totalPnl))}** (${((totalPnl/totalInv)*100).toFixed(2)}%)`;
   }},

  // Trending / Top performers
  {match: s => /trend|top|best|worst|mover|gainer|loser|perform/i.test(s),
   handler: (m, ctx) => {
     const sorted = [...ctx.STOCKS].sort((a,b)=>{
       return ((ctx.S[b.symbol].price-b.base)/b.base) - ((ctx.S[a.symbol].price-a.base)/a.base);
     });
     const gainers = sorted.slice(0,5).map((s,i)=>{
       const pct = ((ctx.S[s.symbol].price-s.base)/s.base*100).toFixed(2);
       return `${i+1}. **${s.name}** — ₹${ctx.fmt(ctx.S[s.symbol].price)} (${parseFloat(pct)>=0?'+':''}${pct}%)`;
     });
     const losers = sorted.slice(-3).reverse().map((s,i)=>{
       const pct = ((ctx.S[s.symbol].price-s.base)/s.base*100).toFixed(2);
       return `${i+1}. **${s.name}** — ₹${ctx.fmt(ctx.S[s.symbol].price)} (${pct}%)`;
     });
     return `**📈 Top Gainers:**\n${gainers.join('\n')}\n\n**📉 Top Losers:**\n${losers.join('\n')}`;
   }},

  // Tax
  {match: s => /tax|80c|ltcg|stcg|section|deduct|elss/i.test(s),
   resp: `**🧾 Tax & Investments (India FY 2025-26)**\n\n**Capital Gains Tax:**\n• **LTCG** (held >1 year): 10% above ₹1 lakh gains (equity)\n• **STCG** (held <1 year): 15% on equity gains\n• Debt funds: Taxed at slab rate\n\n**Tax Saving Instruments (Section 80C — ₹1.5L limit):**\n• **ELSS Mutual Funds** — best option (3-year lock-in, ~12-15% returns)\n• **PPF** — 7.1% guaranteed, 15-year lock-in, EEE status\n• **EPF** — employee provident fund\n• **Tax-saver FD** — 5-year lock-in, ~7% returns\n• **NPS** — extra ₹50,000 under 80CCD(1B)\n\n**Section 80D:** Health insurance premium — ₹25,000 (self) + ₹25,000 (parents)\n\n💡 **Pro tip:** Max out ELSS first (best risk-adjusted after-tax returns), then PPF, then NPS for additional deduction.`},

  // Retirement
  {match: s => /retire|pension|nps|fire|financial.?freedom|corpus/i.test(s),
   resp: `**🎯 Retirement Planning**\n\n**The 25x Rule:** You need 25× your annual expenses to retire.\n• Monthly expenses ₹50,000 → Annual ₹6L → Corpus needed: **₹1.5 Crore**\n\n**The 4% Rule:** Withdraw 4% of corpus annually — it should last 30+ years.\n\n**How to build your corpus:**\n1. **NPS** — extra ₹50,000 tax deduction, equity exposure up to 75%\n2. **PPF** — safe 7.1% component\n3. **Equity SIPs** — growth engine (12-15% CAGR)\n4. **EPF** — employer contribution (free money!)\n\n**FIRE (Financial Independence, Retire Early):**\n• Save 50-70% of income\n• Invest aggressively in equity (80%+)\n• Target 50x annual expenses for early retirement\n\n📌 Starting a ₹15,000 SIP at age 25 at 12% CAGR = **₹2.8 Crore** by age 50!`},

  // Budget / Personal Finance
  {match: s => /budget|save|saving|expense|emergency|income|spend|money.?manage/i.test(s),
   resp: `**💳 Personal Finance Essentials**\n\n**50-30-20 Rule:**\n• 50% → Needs (rent, food, EMIs, insurance)\n• 30% → Wants (dining, travel, shopping)\n• 20% → Savings & Investments (SIPs, PPF, FDs)\n\n**Emergency Fund:** Keep 6 months of expenses in liquid fund/savings account.\n\n**Financial Milestones:**\n• Age 25: Emergency fund + term insurance + start SIP\n• Age 30: ₹10L+ invested + health insurance + own equity portfolio\n• Age 35: ₹25L+ invested + home down payment\n• Age 40: ₹1Cr+ invested + kid's education fund\n\n**Apps for tracking:** Money View, Walnut, ET Money, INDmoney\n\n📌 **Golden Rule:** Pay yourself first — automate investments on salary day.`},

  // Gold
  {match: s => /gold|sgb|sovereign|yellow.?metal/i.test(s),
   resp: `**🥇 Gold as an Investment**\n\n**Options to invest in gold:**\n• **Sovereign Gold Bonds (SGBs)** — BEST option. 2.5% annual interest + gold price appreciation. Tax-free on maturity (8 years). Issued by RBI.\n• **Gold ETFs** — trade on exchange, no interest, expense ratio ~0.5%\n• **Digital Gold** — buy from apps, stored in vaults\n• **Physical Gold** — jewelry/coins (making charges, storage issues)\n\n**Why gold?**\n• Hedge against inflation and currency depreciation\n• Negatively correlated to equities (goes up when stocks fall)\n• Safe haven during global crises\n\n**Allocation:** 5-15% of portfolio in gold\n\n💡 Always prefer SGBs > Gold ETFs > Digital Gold > Physical Gold.`},

  // Real Estate
  {match: s => /real.?estate|property|reit|house|home.?loan|rent.?vs/i.test(s),
   resp: `**🏠 Real Estate Investment**\n\n**Direct Property:**\n• High capital requirement (₹30L+ in metros)\n• Illiquid — hard to sell quickly\n• Maintenance and tenant issues\n• Returns: 6-8% appreciation + 2-3% rental yield\n\n**REITs (Real Estate Investment Trusts):**\n• Buy real estate exposure like a stock on NSE\n• Mindspace REIT, Embassy REIT, Brookfield REIT available in India\n• Minimum investment: ~₹300-400\n• Dividend yield: 5-7% annually\n• Highly liquid — sell anytime\n\n**Rent vs Buy?**\nIf rent < 3% of property value annually → renting is financially better. Invest the difference in equity SIPs.\n\n💡 For real estate exposure without hassles, prefer REITs over physical property.`},

  // Crypto
  {match: s => /crypto|bitcoin|ethereum|blockchain|web3/i.test(s),
   resp: `**₿ Cryptocurrency**\n\n**In India:**\n• 30% flat tax on crypto gains (no offset with losses)\n• 1% TDS on transactions above ₹10,000\n• Not illegal but not legal tender either\n\n**Major cryptocurrencies:**\n• **Bitcoin (BTC)** — digital gold, store of value\n• **Ethereum (ETH)** — smart contract platform\n\n**Should you invest?**\n• Extremely volatile (50%+ swings common)\n• Unregulated — no SEBI/RBI protection\n• Only invest what you can afford to lose completely\n• Maximum 5% of total portfolio\n\n⚠️ InvestIQ focuses on **stock market and traditional investments** where our AI models provide data-driven insights. For crypto, do extensive independent research.`},

  // Compare stocks
  {match: s => /compare|vs|versus|better|which.?one|choose.?between/i.test(s),
   handler: (m, ctx) => {
     const words = m.toLowerCase().split(/\s+/);
     const found = [];
     ctx.STOCKS.forEach(s => {
       if(words.some(w => s.name.toLowerCase().includes(w) || s.symbol.toLowerCase().replace('.ns','').includes(w)))
         found.push(s);
     });
     if(found.length >= 2){
       const s1 = found[0], s2 = found[1];
       const st1 = ctx.S[s1.symbol], st2 = ctx.S[s2.symbol];
       const pct1 = ((st1.price-s1.base)/s1.base*100).toFixed(2);
       const pct2 = ((st2.price-s2.base)/s2.base*100).toFixed(2);
       return `**📊 ${s1.name} vs ${s2.name}**\n\n` +
         `| Metric | ${s1.name.split(' ')[0]} | ${s2.name.split(' ')[0]} |\n` +
         `|--------|-------|-------|\n` +
         `| Price | ₹${ctx.fmt(st1.price)} | ₹${ctx.fmt(st2.price)} |\n` +
         `| Change | ${pct1}% | ${pct2}% |\n` +
         `| Risk | ${Math.round(st1.risk)}/100 | ${Math.round(st2.risk)}/100 |\n` +
         `| Sector | ${s1.sector} | ${s2.sector} |\n\n` +
         `${st1.risk < st2.risk ? s1.name.split(' ')[0] + ' is **lower risk**' : s2.name.split(' ')[0] + ' is **lower risk**'}. ` +
         `${parseFloat(pct1) > parseFloat(pct2) ? s1.name.split(' ')[0] + ' has **better performance** today.' : s2.name.split(' ')[0] + ' has **better performance** today.'}`;
     }
     return null;
   }},

  // SEBI / regulations
  {match: s => /sebi|regulation|insider|trading.?rule|demat|broker/i.test(s),
   resp: `**⚖️ SEBI & Market Regulations**\n\n**SEBI** (Securities & Exchange Board of India) regulates the Indian stock market.\n\n**Key rules:**\n• **Demat account** mandatory for trading (hold shares electronically)\n• **T+1 settlement** — stocks settle 1 day after trade\n• **Insider trading** is illegal — heavy fines and jail time\n• **Circuit breakers** — trading halts when NIFTY moves 10/15/20% in a day\n• **Margin requirements** — can't over-leverage beyond SEBI limits\n\n**Best discount brokers:** Zerodha, Groww, Upstox, Angel One, Dhan\n• Equity delivery: ₹0 brokerage\n• Intraday/F&O: ₹20 per order\n\n**Account types:**\n• **Demat** — holds your shares\n• **Trading** — execute buy/sell orders\n• **3-in-1** — demat + trading + bank (offered by ICICI, HDFC, Kotak)`},

  // Sector analysis
  {match: s => /sector|it.?sector|bank.?sector|fmcg|pharma|auto|energy|sector.?rotat/i.test(s),
   resp: `**🏭 Sector Analysis (Indian Market)**\n\n**IT (TCS, Infosys, Wipro, HCL):**\nExport-driven, benefits from weak rupee. Defensive sector. Revenue in USD.\n\n**Banking (HDFC, ICICI, SBI, Kotak):**\nDriven by credit growth and interest rates. Biggest NIFTY weight.\n\n**FMCG (ITC, HUL, Nestle):**\nDefensive — stable demand regardless of market. Lower growth but consistent.\n\n**Auto (Maruti, Tata Motors, M&M):**\nCyclical — tied to economic growth and consumer spending.\n\n**Pharma (Sun Pharma, Dr Reddy's):**\nDefensive — counter-cyclical. Benefits during health crises.\n\n**Energy (Reliance, ONGC, Power Grid):**\nAffected by oil prices and government policy.\n\n**Sector Rotation Strategy:**\n• During bull markets → overweight cyclicals (auto, banking)\n• During bear markets → overweight defensives (FMCG, pharma, IT)\n\nOn your dashboard, you can see stocks from IT, Finance, FMCG, Energy, and Auto sectors.`},

  // AI / ML predictions
  {match: s => /ai|machine.?learn|lstm|model|xgboost|prediction|how.?work|algorithm|neural|deep.?learn|shap/i.test(s),
   resp: `**🤖 InvestIQ AI Engine**\n\nOur prediction system uses **4 ML models** in an ensemble:\n\n**1. Bidirectional LSTM** (40% weight)\n• Deep learning on price sequences\n• Captures long-term temporal patterns\n• Input: 60-day price windows\n\n**2. XGBoost** (35% weight)\n• Gradient boosted decision trees\n• Best for tabular financial data\n• Handles missing data gracefully\n\n**3. Random Forest** (25% weight)\n• Ensemble of 500+ decision trees\n• Robust against overfitting\n\n**Features used (30+):**\nRSI, MACD, Bollinger Bands, OBV, ATR, Moving Averages (5/10/20/50/200-day), Volume ratios, Price momentum, Stochastic oscillator, Williams %R, and more.\n\n**SHAP Analysis:**\nWe use SHAP (SHapley Additive exPlanations) to explain WHY each prediction is made — showing which features drove the decision.\n\n**Accuracy:** R² ~0.92-0.98, MAPE ~1.5-4% depending on stock and timeframe.\n\n⚠️ ML predictions are probabilistic estimates, not guarantees. Always combine with fundamental analysis.`},

  // Greetings
  {match: s => /^(hi|hello|hey|good|namaste|hola)[\s!.,]*/i.test(s),
   resp: `Hello! 👋 I'm your **InvestIQ AI Assistant** — your personal stock market expert!\n\nI can help you with:\n📊 **Live stock analysis** — prices, risk, and trends from the dashboard\n📈 **Trading concepts** — intraday, swing trading, options, futures\n📚 **Investment education** — SIP, mutual funds, ETFs, bonds\n🧮 **Financial planning** — tax saving, retirement, budgeting\n🤖 **AI predictions** — LSTM forecasts for any tracked stock\n\nJust ask anything about stocks or investing!`},

  // Thank you / goodbye
  {match: s => /thank|bye|goodbye|see you|take care/i.test(s),
   resp: `You're welcome! 😊 Remember:\n\n📌 **Invest regularly** (SIP > lump sum for most people)\n📌 **Stay patient** (compounding needs time)\n📌 **Never panic sell** (crashes are temporary, quality is permanent)\n\nCome back anytime for market insights. Happy investing! 🚀`},

  // Help
  {match: s => /help|what can|menu|commands|features/i.test(s),
   resp: `**🌟 InvestIQ AI — Complete Guide**\n\n**📊 Live Data (uses your dashboard):**\n• "What is TCS price?" — live stock analysis\n• "Risk analysis" — all stocks risk breakdown\n• "Show my P&L" — portfolio summary\n• "What's trending?" — top gainers & losers\n• "Compare TCS vs Infosys" — side by side\n\n**🔮 Predictions:**\n• "Predict TCS" — 7-day ML forecast\n\n**📚 Investment Topics:**\n• SIP, mutual funds, ETFs, bonds, gold, IPO\n• Options & futures, technical analysis\n• Fundamental analysis, P/E ratio, ROE\n• Tax saving, Section 80C, LTCG/STCG\n• Retirement planning, FIRE\n• Budgeting, emergency fund\n• Real estate, REITs, crypto\n• Sector analysis, market indices\n• Bull/bear markets, dividends\n• SEBI regulations, demat accounts\n\nJust ask naturally — I understand questions like a human! 🧠`},
];

// ── Main lookup function ──
BRAIN.getResponse = async function(input, ctx){
  const m = input.toLowerCase();

  // Try knowledge base entries
  for(const entry of BRAIN.KB){
    if(typeof entry.match === 'function' ? entry.match(m) : entry.match.test(m)){
      if(entry.handler){
        const result = entry.handler(m, ctx);
        if(result) return result;
      }
      if(entry.resp) return entry.resp;
    }
  }

  // Prediction queries — call API
  if(m.includes('predict') || m.includes('forecast') || m.includes('future')){
    const tickerMatch = m.match(/(?:predict|forecast).*?([a-z]+)/i);
    let ticker = 'TCS.NS';
    if(tickerMatch){
      const q = tickerMatch[1].toLowerCase();
      const found = ctx.STOCKS.find(s=> s.name.toLowerCase().includes(q) || s.symbol.toLowerCase().includes(q));
      if(found) ticker = found.symbol;
    }
    try{
      const res = await fetch(`${ctx.API}/api/predict?ticker=${ticker}&days=7`);
      const data = await res.json();
      if(data.predictions && data.predictions.length){
        const stock = ctx.STOCKS.find(s=>s.symbol===ticker);
        const name = stock ? stock.name : ticker;
        return `**🔮 Prediction for ${name}** (next 7 days)\n\n${data.predictions.map((p,i)=>`Day ${i+1}: ₹${p.predicted_price.toFixed(2)}`).join('\n')}\n\n⚠️ ML predictions are not financial advice.`;
      }
    }catch(e){}
    return `I tried to fetch predictions for **${ticker}** but the model isn't available right now. The prediction engine requires trained model files.`;
  }

  // Fallback — try the backend chatbot
  try{
    const res = await fetch(`${ctx.API}/api/chat`, {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({message: input})
    });
    const data = await res.json();
    if(data.reply && !data.reply.includes("didn't fully understand")) return data.reply;
  }catch(e){}

  // Smart fallback — try to find relevant topic
  const topics = [
    {keys:['stock','share','equity','buy','sell','trade'], suggest:'Try asking about a specific stock (e.g. "TCS analysis") or trading concept (e.g. "what is intraday trading")'},
    {keys:['money','finance','wealth','bank','account'], suggest:'Try asking about investing basics, SIPs, mutual funds, or budgeting tips'},
    {keys:['market','index','economy'], suggest:'Try asking about NIFTY, SENSEX, bull/bear markets, or sector analysis'},
  ];
  for(const t of topics){
    if(t.keys.some(k=>m.includes(k))) return `🤔 I'm not sure about that specific question, but I have extensive knowledge on the topic.\n\n${t.suggest}\n\nOr type **help** to see everything I can answer!`;
  }

  return `🤔 I'm a stock market & investment AI expert. I can answer detailed questions about:\n\n📊 **Stocks & Trading** — prices, analysis, intraday, options\n📈 **Investing** — SIP, mutual funds, ETFs, gold, real estate\n💰 **Finance** — tax saving, retirement, budgeting\n🤖 **AI Predictions** — ML forecasts for tracked stocks\n\nTry rephrasing your question or type **help** for the full list!`;
};
