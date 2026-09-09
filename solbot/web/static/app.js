/* Dashboard live updates: polls the read-only JSON API and re-renders.
 * Control actions are plain form posts, so they work without JS too.
 */
(function () {
  "use strict";

  var POLL_MS = 5000;
  var lastEventId = 0;
  var charts = {};

  /* ---------- theme toggle ---------- */
  var toggle = document.getElementById("themeToggle");
  var stored = null;
  try { stored = localStorage.getItem("solbot-theme"); } catch (e) { /* private mode */ }
  if (stored) document.documentElement.setAttribute("data-theme", stored);
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = document.documentElement.getAttribute("data-theme") === "dark" ? "light" : "dark";
      document.documentElement.setAttribute("data-theme", next);
      try { localStorage.setItem("solbot-theme", next); } catch (e) { /* ignore */ }
      refreshCharts();
    });
  }

  function get(url) {
    return fetch(url, { credentials: "same-origin" }).then(function (r) {
      if (r.status === 401) { window.location.href = "/login"; throw new Error("unauthenticated"); }
      if (!r.ok) throw new Error("HTTP " + r.status);
      return r.json();
    });
  }

  function el(id) { return document.getElementById(id); }
  function money(v) { return (v < 0 ? "-$" : "$") + Math.abs(v).toFixed(2); }

  // Same "how many decimals actually mean something" scaling charts.js's own
  // (private, unexported) fmtPrice uses for the axis labels - duplicated
  // rather than reaching into that module's closure for one small function.
  function fmtLivePrice(v) {
    if (v === null || v === undefined || !isFinite(v) || v === 0) return "—";
    var abs = Math.abs(v);
    if (abs >= 1000) return "$" + v.toFixed(0);
    if (abs >= 1) return "$" + v.toFixed(2);
    if (abs >= 0.01) return "$" + v.toFixed(4);
    return "$" + v.toPrecision(3);
  }
  function pct(v) { return (v >= 0 ? "+" : "") + v.toFixed(2) + "%"; }
  function cls(v) { return v > 0 ? "pos" : (v < 0 ? "neg" : "dim"); }
  function esc(s) {
    return String(s === null || s === undefined ? "" : s)
      .replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  /* ---------- status ---------- */
  function loadStatus() {
    return get("/api/status").then(function (d) {
      var hb = el("workerStatus");
      if (hb) {
        var alive = d.worker && d.worker.alive;
        hb.innerHTML = '<span class="dot ' + (alive ? "dot-on" : "dot-off") + '"></span>' +
          (alive ? "Worker alive (cycle " + esc(d.worker.cycle || 0) + ")"
                 : "Worker not responding" + (d.worker && d.worker.age_seconds !== null
                     ? " (" + Math.round(d.worker.age_seconds) + "s ago)" : ""));
      }

      var budget = el("scanBudget");
      if (budget && d.scan_budget && d.scan_budget.message) {
        budget.textContent = d.scan_budget.message;
        budget.className = d.scan_budget.sustainable ? "hint" : "hint warn";
      }

      var primary = null;
      (d.instances || []).forEach(function (i) {
        if (i.instance === (d.mode === "live" ? "live" : "paper")) primary = i;
      });
      if (primary) {
        setText("statBalance", money(primary.balance));
        setText("statDeployed", money(primary.deployed));
        setText("statOpen", String(primary.open_positions));
        var p = primary.performance || {};
        setText("statWinRate", ((p.win_rate || 0) * 100).toFixed(1) + "%");
        setText("statTrades", String(p.trades || 0));
        setText("statPnl", money(p.total_pnl || 0));
        var pnlEl = el("statPnl");
        if (pnlEl) pnlEl.className = "value " + cls(p.total_pnl || 0);
        setText("statPf", p.profit_factor === null ? "∞" : (p.profit_factor || 0).toFixed(2));
        setText("statFees", money(p.total_fees || 0));
        setText("statAvgWin", money(p.avg_win || 0));
        setText("statAvgLoss", money(p.avg_loss || 0));
        setText("statDrawdown", ((p.max_drawdown || 0) * 100).toFixed(1) + "%");
      }
      return d;
    });
  }

  function setText(id, text) { var n = el(id); if (n) n.textContent = text; }

  /* ---------- positions ---------- */
  function loadPositions() {
    var host = el("positionsBody");
    var chartHost = el("positionCharts");
    if (!host && !chartHost) return Promise.resolve();

    return get("/api/positions").then(function (rows) {
      if (host) {
        if (!rows.length) {
          host.innerHTML = '<tr><td colspan="9" class="empty">No open positions</td></tr>';
        } else {
          host.innerHTML = rows.map(function (p) {
            var u = p.unrealised_usd;
            return "<tr>" +
              '<td><span class="tag tag-' + esc(p.instance) + '">' + esc(p.instance) + "</span></td>" +
              "<td><strong>" + esc(p.symbol || p.mint.slice(0, 8)) + "</strong></td>" +
              '<td class="num">' + esc(fmtP(p.entry_price)) + "</td>" +
              '<td class="num">' + (p.current_price ? esc(fmtP(p.current_price)) : "—") + "</td>" +
              '<td class="num">' + money(p.size_usd) + "</td>" +
              '<td class="num ' + (u === null ? "dim" : cls(u)) + '">' +
                (u === null ? "—" : money(u) + " (" + pct(p.unrealised_pct) + ")") + "</td>" +
              '<td class="num neg">' + esc(fmtP(p.hard_stop)) + "</td>" +
              '<td class="num warn">' + (p.trailing_stop ? esc(fmtP(p.trailing_stop)) : "—") + "</td>" +
              '<td><form method="post" action="/positions/' + p.id + '/close" ' +
                'onsubmit="return confirm(\'Close this position at market?\')">' +
                '<button class="btn-sm">Close</button></form></td>' +
              "</tr>";
          }).join("");
        }
      }
      if (chartHost) renderPositionCharts(chartHost, rows);
    });
  }

  function fmtP(v) {
    v = Number(v);
    if (!isFinite(v)) return "—";
    if (Math.abs(v) >= 1) return "$" + v.toFixed(4);
    return "$" + v.toPrecision(4);
  }

  /* Every open position renders as its own clearly labelled chart, wrapping
   * to as many rows as needed - there is no fixed position-count cap. */
  function renderPositionCharts(host, rows) {
    if (!rows.length) {
      host.innerHTML = '<div class="panel"><div class="empty">' +
        "Charts appear here when a position is open</div></div>";
      charts = {};
      return;
    }
    var wanted = rows.map(function (p) { return "pos-" + p.id; });
    if (Object.keys(charts).sort().join() !== wanted.slice().sort().join()) {
      host.innerHTML = rows.map(function (p) {
        return '<div class="panel">' +
          '<div class="panel-head"><h2>' + esc(p.symbol || p.mint.slice(0, 8)) +
          ' <span class="tag tag-' + esc(p.instance) + '">' + esc(p.instance) + "</span></h2>" +
          '<span class="dim mono">' + esc(p.mint.slice(0, 10)) + "…</span></div>" +
          '<div class="chart-box"><canvas id="chart-pos-' + p.id + '"></canvas></div>' +
          '<div class="chart-legend">' +
          '<span class="legend-entry">entry</span>' +
          '<span class="legend-stop">hard stop</span>' +
          '<span class="legend-trail">trailing stop</span>' +
          '<span class="legend-target">target</span></div></div>';
      }).join("");
      charts = {};
      rows.forEach(function (p) { charts["pos-" + p.id] = true; });
    }

    var t = window.SolChart.theme();
    rows.forEach(function (p) {
      get("/api/candles/" + encodeURIComponent(p.mint) + "?limit=120").then(function (d) {
        var canvas = el("chart-pos-" + p.id);
        if (!canvas) return;
        var levels = [
          { value: p.entry_price, color: t.green, label: "entry", dash: [2, 2] },
          { value: p.hard_stop, color: t.red, label: "stop" },
          { value: p.take_profit, color: t.accent, label: "target" }
        ];
        if (p.trailing_stop) {
          levels.push({ value: p.trailing_stop, color: t.amber, label: "trail" });
        }
        window.SolChart.candles(canvas, d.candles, {
          height: 300,
          levels: levels,
          markers: [{ time: p.entry_ts, price: p.entry_price, color: t.green, shape: "up" }]
        });
      }).catch(function () { /* transient */ });
    });
  }

  /* ---------- market chart (always visible, dropdown over the universe) ---------- */
  var marketUniverse = [];        // [{mint, symbol, label}], sorted by volume desc (server order)
  var marketMint = null;          // currently displayed mint
  var marketLastKey = null;       // mint+interval last rendered, to know when to reset zoom/pan
  var marketUniverseLoaded = false;
  var MARKET_CHART_HEIGHT = 600;  // the dashboard's dominant panel, not one among equals
  var MARKET_RSI_HEIGHT = 110;

  /* ---- timeframe/range controls: interval matches solbot/web/api.py's
   * CHART_INTERVALS exactly (only intervals that roll up cleanly from the
   * 1-minute base); range picks how many candles at that interval covers
   * roughly the named wall-clock span, capped at the server's 1000 limit -
   * persisted in localStorage the same way the indicator picker is. ---- */
  var TIMEFRAME_STORAGE_KEY = "solbot-market-timeframe";
  var INTERVAL_SECONDS = { "1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400 };
  var RANGE_DAYS = { "1D": 1, "1W": 7, "1M": 30, "3M": 90, "1Y": 365, "All": 100000 };
  var marketInterval = "1m";
  var marketRange = "1W";

  (function loadTimeframeSettings() {
    var saved = null;
    try { saved = JSON.parse(localStorage.getItem(TIMEFRAME_STORAGE_KEY) || "null"); } catch (e) { /* ignore */ }
    if (saved && INTERVAL_SECONDS[saved.interval]) marketInterval = saved.interval;
    if (saved && RANGE_DAYS[saved.range]) marketRange = saved.range;
  })();

  function saveTimeframeSettings() {
    try {
      localStorage.setItem(TIMEFRAME_STORAGE_KEY, JSON.stringify({ interval: marketInterval, range: marketRange }));
    } catch (e) { /* private mode */ }
  }

  function marketLimit() {
    var seconds = INTERVAL_SECONDS[marketInterval] || 60;
    var days = RANGE_DAYS[marketRange] || 7;
    return Math.max(20, Math.min(1000, Math.ceil((days * 86400) / seconds)));
  }

  function syncTimeframeButtons() {
    var intervalGroup = el("marketIntervalGroup");
    var rangeGroup = el("marketRangeGroup");
    if (intervalGroup) {
      var ibtns = intervalGroup.querySelectorAll("button");
      for (var i = 0; i < ibtns.length; i++) {
        ibtns[i].classList.toggle("active", ibtns[i].getAttribute("data-interval") === marketInterval);
      }
    }
    if (rangeGroup) {
      var rbtns = rangeGroup.querySelectorAll("button");
      for (var j = 0; j < rbtns.length; j++) {
        rbtns[j].classList.toggle("active", rbtns[j].getAttribute("data-range") === marketRange);
      }
    }
  }

  (function wireTimeframeButtons() {
    var intervalGroup = el("marketIntervalGroup");
    var rangeGroup = el("marketRangeGroup");
    if (intervalGroup) {
      intervalGroup.addEventListener("click", function (ev) {
        var btn = ev.target.closest && ev.target.closest("button[data-interval]");
        if (!btn) return;
        marketInterval = btn.getAttribute("data-interval");
        saveTimeframeSettings();
        syncTimeframeButtons();
        loadMarketChart().catch(noop);
      });
    }
    if (rangeGroup) {
      rangeGroup.addEventListener("click", function (ev) {
        var btn = ev.target.closest && ev.target.closest("button[data-range]");
        if (!btn) return;
        marketRange = btn.getAttribute("data-range");
        saveTimeframeSettings();
        syncTimeframeButtons();
        loadMarketChart().catch(noop);
      });
    }
    syncTimeframeButtons();
  })();

  function marketLabel(row) {
    return (row.symbol || row.mint.slice(0, 8)) + " — " + row.mint.slice(0, 6) + "…";
  }

  function loadMarketUniverse() {
    var select = el("marketSymbolSelect");
    if (!select) return Promise.resolve();
    return get("/api/universe").then(function (rows) {
      marketUniverse = rows.map(function (r) {
        return { mint: r.mint, symbol: r.symbol, label: marketLabel(r) };
      });
      select.innerHTML = marketUniverse.map(function (r) {
        return '<option value="' + esc(r.mint) + '">' + esc(r.label) + "</option>";
      }).join("");
      marketUniverseLoaded = true;
      // Default to the top-ranked (highest 24h volume) coin once, and whenever
      // the previously-selected mint drops out of the universe.
      var stillPresent = marketMint && marketUniverse.some(function (r) { return r.mint === marketMint; });
      if (!stillPresent && marketUniverse.length) {
        marketMint = marketUniverse[0].mint;
      }
      if (marketMint) select.value = marketMint;
    });
  }

  var REGIME_NAMES = ["cluster 0", "cluster 1", "cluster 2", "cluster 3", "cluster 4", "cluster 5"];

  function renderRegimeReadout(enabled, regime) {
    var out = el("marketRegimeReadout");
    if (!out) return;
    if (!enabled) {
      out.textContent = "";
      return;
    }
    if (!regime || !regime.current) {
      out.textContent = "fuzzy regime: no discovered model for this coin yet";
      return;
    }
    var parts = Object.keys(regime.current)
      .map(function (k) { return { k: Number(k), v: regime.current[k] }; })
      .sort(function (a, b) { return b.v - a.v; })
      .map(function (p) { return (REGIME_NAMES[p.k] || ("cluster " + p.k)) + " " + (p.v * 100).toFixed(0) + "%"; });
    out.textContent = "fuzzy regime (live): " + parts.join(" · ");
  }

  /* ---- indicator picker: persisted in localStorage so it survives a reload,
   * same spirit as the theme toggle above. ---- */
  var INDICATOR_STORAGE_KEY = "solbot-market-indicators";
  var DEFAULT_INDICATORS = { sma: false, smaPeriod: 20, ema: false, emaPeriod: 9,
    bb: false, bbPeriod: 20, rsi: false, rsiPeriod: 14, volume: true };

  function loadIndicatorSettings() {
    var saved = null;
    try { saved = JSON.parse(localStorage.getItem(INDICATOR_STORAGE_KEY) || "null"); } catch (e) { /* ignore */ }
    var settings = {};
    Object.keys(DEFAULT_INDICATORS).forEach(function (k) {
      settings[k] = (saved && saved[k] !== undefined) ? saved[k] : DEFAULT_INDICATORS[k];
    });
    return settings;
  }

  function saveIndicatorSettings(settings) {
    try { localStorage.setItem(INDICATOR_STORAGE_KEY, JSON.stringify(settings)); } catch (e) { /* private mode */ }
  }

  function readIndicatorFormInto(settings) {
    var map = {
      sma: "indSMA", smaPeriod: "indSMAPeriod", ema: "indEMA", emaPeriod: "indEMAPeriod",
      bb: "indBB", bbPeriod: "indBBPeriod", rsi: "indRSI", rsiPeriod: "indRSIPeriod",
      volume: "indVolume"
    };
    Object.keys(map).forEach(function (k) {
      var node = el(map[k]);
      if (!node) return;
      settings[k] = node.type === "checkbox" ? node.checked : (parseInt(node.value, 10) || DEFAULT_INDICATORS[k]);
    });
    return settings;
  }

  function applyIndicatorSettingsToForm(settings) {
    var map = {
      sma: "indSMA", smaPeriod: "indSMAPeriod", ema: "indEMA", emaPeriod: "indEMAPeriod",
      bb: "indBB", bbPeriod: "indBBPeriod", rsi: "indRSI", rsiPeriod: "indRSIPeriod",
      volume: "indVolume"
    };
    Object.keys(map).forEach(function (k) {
      var node = el(map[k]);
      if (!node) return;
      if (node.type === "checkbox") node.checked = !!settings[k]; else node.value = settings[k];
    });
  }

  function indicatorQuery(settings) {
    var parts = [];
    if (settings.sma) parts.push("sma=" + settings.smaPeriod);
    if (settings.ema) parts.push("ema=" + settings.emaPeriod);
    if (settings.bb) parts.push("bbands=" + settings.bbPeriod);
    if (settings.rsi) parts.push("rsi=" + settings.rsiPeriod);
    return parts.length ? "&" + parts.join("&") : "";
  }

  function loadMarketChart() {
    var canvas = el("marketChart");
    var hint = el("marketChartHint");
    if (!canvas) return Promise.resolve();

    var afterUniverse = marketUniverseLoaded ? Promise.resolve() : loadMarketUniverse();
    return afterUniverse.then(function () {
      if (!marketMint) {
        if (hint) hint.textContent = "No coins in the universe yet — try Refresh universe.";
        renderRegimeReadout(false, null);
        setText("marketPrice", "");
        return;
      }
      if (hint) {
        var row = marketUniverse.find(function (r) { return r.mint === marketMint; });
        hint.textContent = row ? (row.symbol || marketMint) + " · " + marketUniverse.length +
          " coins in the current universe" : "";
      }
      var toggle = el("marketRegimeToggle");
      var withRegime = toggle && toggle.checked;
      var settings = readIndicatorFormInto(loadIndicatorSettings());
      var url = "/api/candles/" + encodeURIComponent(marketMint) +
        "?limit=" + marketLimit() + "&interval=" + encodeURIComponent(marketInterval) +
        (withRegime ? "&regime=1" : "") + indicatorQuery(settings);
      return get(url).then(function (d) {
        var ind = d.indicators || {};
        setText("marketPrice", fmtLivePrice(d.current_price));
        var resetZoom = marketLastKey !== (marketMint + "|" + marketInterval);
        marketLastKey = marketMint + "|" + marketInterval;
        window.SolChart.candles(canvas, d.candles, {
          height: MARKET_CHART_HEIGHT,
          regime: withRegime && d.regime ? d.regime.history : null,
          volume: settings.volume,
          overlays: { sma: ind.sma, ema: ind.ema, bbands: ind.bbands },
          enableZoomPan: true,
          resetZoom: resetZoom,
          currentPrice: d.current_price
        });
        renderRegimeReadout(withRegime, d.regime);

        var rsiBox = el("marketRsiBox"), rsiCanvas = el("marketRsiChart");
        if (rsiBox && rsiCanvas) {
          if (settings.rsi && ind.rsi) {
            rsiBox.style.display = "";
            window.SolChart.rsi(rsiCanvas, d.candles, ind.rsi.values, { height: MARKET_RSI_HEIGHT });
          } else {
            rsiBox.style.display = "none";
          }
        }
      });
    });
  }

  (function wireMarketSymbolSelect() {
    var select = el("marketSymbolSelect");
    if (!select) return;
    select.addEventListener("change", function () {
      if (select.value && select.value !== marketMint) {
        marketMint = select.value;
        loadMarketChart().catch(noop);
      }
    });
  })();

  (function wireMarketRegimeToggle() {
    var toggle = el("marketRegimeToggle");
    if (!toggle) return;
    toggle.addEventListener("change", function () { loadMarketChart().catch(noop); });
  })();

  (function wireIndicatorPicker() {
    var picker = el("indicatorPicker");
    if (!picker) return;
    applyIndicatorSettingsToForm(loadIndicatorSettings());
    picker.addEventListener("change", function () {
      var settings = readIndicatorFormInto(loadIndicatorSettings());
      saveIndicatorSettings(settings);
      loadMarketChart().catch(noop);
    });
    // A period number input firing on every keystroke would refetch mid-typing;
    // "change" alone already covers blur/enter for number inputs, so no extra
    // debounce is needed here.
  })();

  /* ---------- equity ---------- */
  function loadEquity() {
    var canvas = el("equityChart");
    if (!canvas) return Promise.resolve();
    return get("/api/equity?days=30").then(function (series) {
      window.SolChart.lines(canvas, series, { height: 240 });
      var legend = el("equityLegend");
      if (legend) {
        legend.innerHTML = Object.keys(series).map(function (n) {
          return '<span class="legend-' + esc(n) + '" style="color:var(--' +
            (n === "live" ? "red" : n === "paper" ? "accent" : "purple") + ')">' + esc(n) + "</span>";
        }).join("");
      }
    });
  }

  /* ---------- event feed ---------- */
  function loadEvents() {
    var host = el("eventFeed");
    if (!host) return Promise.resolve();
    return get("/api/events?limit=80").then(function (rows) {
      if (!rows.length) {
        host.innerHTML = '<div class="empty">No activity yet</div>';
        return;
      }
      lastEventId = rows[0].id;
      host.innerHTML = rows.map(function (e) {
        var d = new Date(e.ts * 1000);
        var time = String(d.getHours()).padStart(2, "0") + ":" +
                   String(d.getMinutes()).padStart(2, "0") + ":" +
                   String(d.getSeconds()).padStart(2, "0");
        return '<div class="feed-item feed-' + esc(e.level) + '">' +
          '<span class="feed-time">' + time + "</span>" +
          '<span class="feed-cat">' + esc(e.category) + "</span>" +
          '<span class="feed-msg">' + esc(e.message) + "</span></div>";
      }).join("");
    });
  }

  /* ---------- entry-decision explainability ---------- */
  var reviewRows = [];

  function loadReview() {
    var body = el("reviewRows");
    if (!body) return Promise.resolve();
    return get("/api/review").then(function (d) {
      var market = d.market || {};
      setText("reviewMarket",
        market.tone ? "market: " + market.tone + " · " +
          Math.round((market.breadth || 0) * 100) + "% of coins up" : "—");

      reviewRows = d.recent || [];
      if (!reviewRows.length) {
        body.innerHTML = "<tr><td colspan='5' class='empty'>No entry decisions recorded yet</td></tr>";
        clear("reviewDetail");
        return;
      }
      // Most recent first - the API already orders that way, this just
      // makes it explicit rather than relying on insertion order.
      var rows = reviewRows.slice().reverse();
      body.innerHTML = rows.map(function (r, i) {
        var detail = r.detail || {};
        var decision = detail.decision || {};
        var signal = detail.signal || {};
        var approved = r.decision === "approve";
        var d = new Date(r.ts * 1000);
        var time = String(d.getHours()).padStart(2, "0") + ":" +
          String(d.getMinutes()).padStart(2, "0") + ":" + String(d.getSeconds()).padStart(2, "0");
        return "<tr class='review-row' data-idx='" + i + "' style='cursor:pointer'>" +
          "<td class='mono nowrap'>" + time + "</td>" +
          "<td>" + esc((detail.token && detail.token.symbol) || r.mint || "") + "</td>" +
          "<td class='" + (approved ? "pos" : "neg") + "'>" + esc(r.decision || "") + "</td>" +
          "<td class='num'>" + (r.conviction != null ? r.conviction.toFixed(2) : "—") + "</td>" +
          "<td class='hint' style='max-width:340px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap'>" +
          esc(signal.why || decision.rationale || "") + "</td></tr>";
      }).join("");
    });
  }

  document.addEventListener("click", function (ev) {
    var row = ev.target.closest && ev.target.closest("#reviewRows tr[data-idx]");
    if (!row) return;
    var idx = Number(row.getAttribute("data-idx"));
    var sorted = reviewRows.slice().reverse();
    var r = sorted[idx];
    var box = el("reviewDetail");
    if (!r || !box) return;
    var detail = r.detail || {};
    var decision = detail.decision || {};
    var signal = detail.signal || {};
    var token = detail.token || {};
    var market = detail.market || {};
    var lines = [
      "<strong>" + esc(token.symbol || r.mint || "") + "</strong> — " +
        esc((r.decision || "").toUpperCase()) +
        (decision.source ? " (" + esc(decision.source) + ")" : ""),
      esc(decision.rationale || signal.why || "no rationale recorded"),
      "RSI " + (signal.rsi != null ? signal.rsi : "—") +
        " · momentum " + (signal.momentum_pct != null ? signal.momentum_pct + "%" : "—") +
        " · volume " + (signal.volume_vs_average != null ? signal.volume_vs_average + "x avg" : "—") +
        " · ATR " + (signal.atr_pct != null ? signal.atr_pct + "%" : "—") +
        " · efficiency " + (signal.efficiency_ratio != null ? signal.efficiency_ratio : "—") +
        " · regime " + esc(signal.regime || "—") +
        " · " + (signal.higher_timeframes_agreeing != null ? signal.higher_timeframes_agreeing : "—") + " timeframe(s) agreeing",
      "market: " + esc(market.tone || "—") + " · reward/risk " +
        (signal.reward_risk != null ? signal.reward_risk : "—") + " · conviction " +
        (decision.conviction != null ? decision.conviction : "—"),
    ];
    box.innerHTML = lines.map(function (l) { return "<div>" + l + "</div>"; }).join("");
  });

  /* ---------- progress (backtest page, settings page) ---------- */
  function loadProgress() {
    if (!el("pullProgress") && !el("backtestProgress") && !el("regimePassProgress") &&
        !el("runpodBenchmarkProgress")) {
      return Promise.resolve();
    }
    return get("/api/progress").then(function (d) {
      bar("pull", d.historical_pull);
      bar("dailyPull", d.daily_incremental_pull);
      bar("backtest", d.backtest);
      bar("regimePass", d.regime_pass);
      bar("runpodBenchmark", d.runpod_benchmark);
      // Real, billed pods get launched per tier - keep the button from
      // being clicked again mid-run, the same way a double-click on any
      // other "launch something real" action in this dashboard is avoided.
      var benchBtn = el("runpodBenchmarkBtn");
      if (benchBtn && d.runpod_benchmark) {
        var running = d.runpod_benchmark.status === "running";
        benchBtn.disabled = running;
        benchBtn.textContent = running ? "Running…" : "Run benchmark now";
      }
      // Same double-submission guard as the RunPod benchmark button: a full
      // Binance.US pull is a long, rate-limited job, so clicking it again
      // mid-run would just queue a second one on top of the first.
      var pullBtn = el("pullBtn");
      var pullStopBtn = el("pullStopBtn");
      if (pullBtn && d.historical_pull) {
        var pullRunning = d.historical_pull.status === "running";
        pullBtn.disabled = pullRunning;
        pullBtn.textContent = pullRunning ? "Running…" : "Start historical pull";
        if (pullStopBtn) pullStopBtn.disabled = !pullRunning;
      }
      var cov = el("candleCoverage");
      if (cov && d.candle_coverage) {
        var c = d.candle_coverage;
        cov.textContent = (c.tokens || 0) + " coins, " + (c.candles || 0).toLocaleString() +
          " candles, " + (c.days || 0) + " days (" +
          ((c.disk && c.disk.megabytes) || 0).toLocaleString() + " MB on disk)";
      }
    });
  }

  // settings.html opts out of the dashboard's general live-polling loop
  // (data-live="off" - it's a form page, not a live view), so the RunPod
  // benchmark's progress bar needs its own small trigger to actually poll,
  // reusing loadProgress()/bar() rather than duplicating their logic.
  (function wireRunpodBenchmarkProgress() {
    if (!el("runpodBenchmarkProgress")) return;
    loadProgress().catch(noop);
    setInterval(function () { loadProgress().catch(noop); }, POLL_MS);
  })();

  (function wireRunpodBenchmarkForm() {
    var form = el("runpodBenchmarkForm");
    var btn = el("runpodBenchmarkBtn");
    if (!form || !btn) return;
    form.addEventListener("submit", function () {
      // Immediate feedback before the first poll tick lands; loadProgress()
      // takes over (and can re-enable it) once the job actually starts.
      btn.disabled = true;
      btn.textContent = "Running…";
    });
  })();

  function bar(prefix, p) {
    if (!p) return;
    var wrap = el(prefix + "Progress");
    if (!wrap) return;
    var fill = el(prefix + "Bar");
    var label = el(prefix + "Label");
    if (fill) fill.style.width = (p.percent || 0) + "%";
    if (label) {
      label.textContent = p.status === "running"
        ? (p.message || "") + " — " + (p.percent || 0) + "%"
        : (p.status === "idle" ? "Not run yet" : p.status + ": " + (p.message || ""));
    }
    wrap.style.display = "";

    // A dedicated, styled error line (prefix + "Error") - a stopped-with-
    // failure job is worth calling out distinctly from the ordinary
    // "cancelled by operator" or "done" cases the label line already covers.
    var errEl = el(prefix + "Error");
    if (errEl) {
      if (p.status === "failed") {
        errEl.textContent = "Error: " + (p.message || "the pull stopped unexpectedly");
        errEl.style.display = "";
      } else {
        errEl.style.display = "none";
      }
    }

    // A prominent, live-updating time-remaining readout (prefix + "EtaBox"/
    // "EtaValue") - derived from actual observed throughput (elapsed time
    // vs. done/total) rather than the static upper-bound estimate shown
    // before the job starts, so it tightens up as real progress comes in.
    var etaBox = el(prefix + "EtaBox");
    var etaValue = el(prefix + "EtaValue");
    if (etaBox && etaValue) {
      var done = p.done || 0, total = p.total || 0;
      if (p.status === "running" && done > 0 && total > done && p.started_at) {
        var elapsedS = (Date.now() / 1000) - p.started_at;
        var remainingS = Math.max(0, (elapsedS / done) * (total - done));
        etaValue.textContent = "~" + formatDuration(remainingS) + " remaining (" +
          done + " of " + total + ")";
        etaBox.style.display = "";
      } else {
        etaBox.style.display = "none";
      }
    }
  }

  function formatDuration(seconds) {
    seconds = Math.max(0, Math.round(seconds));
    var h = Math.floor(seconds / 3600);
    var m = Math.floor((seconds % 3600) / 60);
    var s = seconds % 60;
    if (h > 0) return h + "h " + m + "m";
    if (m > 0) return m + "m " + s + "s";
    return s + "s";
  }

  /* ---------- backfill timing estimate (backtest page) ---------- */
  function loadBackfillEstimate() {
    var out = el("pullEstimate");
    if (!out) return Promise.resolve();
    return get("/api/backfill/estimate").then(function (d) {
      if (d.error) { out.textContent = d.error; return; }
      var hours = d.hours_estimate;
      if (hours === undefined || hours === null) { out.textContent = "—"; return; }
      out.textContent = "Upper bound: ~" + hours.toFixed(1) + "h for " + (d.tokens || 0) +
        " Binance.US pairs, assuming every one goes back the full " + d.assumed_years +
        " years (" + d.calls.toLocaleString() + " API calls at the current binance_rps). " +
        "Most pairs' real history is much shorter than that.";
      out.className = "hint";
    });
  }

  (function wirePullButton() {
    if (!el("pullEstimate")) return;
    loadBackfillEstimate().catch(noop);
    var form = el("pullForm");
    var btn = el("pullBtn");
    if (form && btn) {
      form.addEventListener("submit", function () {
        btn.disabled = true;
        btn.textContent = "Running…";
      });
    }
    var stopForm = el("pullStopForm");
    var stopBtn = el("pullStopBtn");
    if (stopForm && stopBtn) {
      stopForm.addEventListener("submit", function () {
        stopBtn.disabled = true;
        stopBtn.textContent = "Stopping…";
      });
    }
  })();

  /* ---------- per-coin candle coverage (backtest page) ---------- */
  var coverageData = [];
  var coverageSort = { key: "candles", dir: "desc" };

  function loadCoverage() {
    var body = el("coverageRows");
    if (!body) return Promise.resolve();
    return get("/api/candles/coverage").then(function (rows) {
      coverageData = rows || [];
      setText("coverageSummary", coverageData.length + " coin" +
        (coverageData.length === 1 ? "" : "s") + " with history on disk");
      renderCoverage();
    });
  }

  function renderCoverage() {
    var body = el("coverageRows");
    if (!body) return;
    if (!coverageData.length) {
      body.innerHTML = "<tr><td colspan='5' class='empty'>No candle history on disk yet</td></tr>";
      return;
    }
    var key = coverageSort.key, dir = coverageSort.dir === "asc" ? 1 : -1;
    var field = { symbol: "symbol", candles: "bars", first_ts: "first_ts", last_ts: "last_ts", days: "days" }[key] || "bars";
    var sorted = coverageData.slice().sort(function (a, b) {
      var av = a[field], bv = b[field];
      if (typeof av === "string" || typeof bv === "string") {
        return dir * String(av || "").localeCompare(String(bv || ""));
      }
      return dir * ((av || 0) - (bv || 0));
    });
    body.innerHTML = sorted.map(function (r) {
      return "<tr><td class='mono'>" + esc(r.symbol) + "</td>" +
        "<td class='num'>" + (r.bars || 0).toLocaleString() + "</td>" +
        "<td class='nowrap'>" + (r.first_ts ? fmtTime(r.first_ts) : "—") + "</td>" +
        "<td class='nowrap'>" + (r.last_ts ? fmtTime(r.last_ts) : "—") + "</td>" +
        "<td class='num'>" + (r.days || 0) + "</td></tr>";
    }).join("");
  }

  document.addEventListener("click", function (ev) {
    var th = ev.target.closest && ev.target.closest("th.sortable[data-sort]");
    var coverageBody = el("coverageRows");
    if (!th || !coverageBody) return;
    var table = th.closest("table");
    if (!table || !table.contains(coverageBody)) return;
    var key = th.getAttribute("data-sort");
    var same = coverageSort.key === key;
    coverageSort = { key: key, dir: same && coverageSort.dir === "desc" ? "asc" : "desc" };
    var headers = th.parentElement.querySelectorAll("th.sortable");
    for (var i = 0; i < headers.length; i++) headers[i].classList.remove("sort-asc", "sort-desc");
    th.classList.add(coverageSort.dir === "asc" ? "sort-asc" : "sort-desc");
    renderCoverage();
  });

  /* ---------- walk-forward tab ---------- */
  var wfFeedSince = 0;
  var wfRunId = null;
  var mcData = null;
  // null = always show the latest run (default, live-following); set by
  // clicking a row in "Recent runs" to pin the view to one historical run
  // instead - historical browsability alongside the live one.
  var wfSelectedRunId = null;

  function pctText(v, digits) {
    if (v === null || v === undefined) return "—";
    return (v * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }
  function signedPct(v, digits) {
    if (v === null || v === undefined) return "—";
    return (v >= 0 ? "+" : "") + (v * 100).toFixed(digits === undefined ? 1 : digits) + "%";
  }

  function loadWalkForward() {
    if (!el("wfFeed")) return Promise.resolve();
    var url = "/api/walkforward" + (wfSelectedRunId ? "?run=" + wfSelectedRunId : "");
    return get(url).then(function (d) {
      var run = d.run;
      renderRunList(d.runs || [], run ? run.run_id : null);
      if (!run) { setText("wfStatus", "no runs yet"); return; }

      // A new run resets the feed cursor so a fresh search does not append to
      // the previous one's narration.
      if (wfRunId !== run.run_id) { wfRunId = run.run_id; wfFeedSince = 0; clear("wfFeed"); }

      var cov = run.coverage || {};
      setText("wfRunLabel",
        "run " + run.run_id + (run.label ? " · " + run.label : "") +
        (cov.symbols ? " · " + cov.symbols + " coins, " + (cov.days || 0) + " days" : "") +
        (wfSelectedRunId ? " · viewing history, not the live run" : ""));
      setText("wfStatus", run.status === "running"
        ? "running — last update " + Math.round(run.age_seconds) + "s ago"
        : run.status);

      var s = run.summary || {};
      var wfe = s.walk_forward_efficiency;
      setText("wfEfficiency", wfe === undefined || wfe === null ? "—" : wfe.toFixed(2));
      var eff = el("wfEfficiency");
      if (eff) eff.className = "value " + (wfe >= 0.5 ? "pos" : (wfe > 0 ? "warn" : "dim"));
      setText("wfEfficiencySub", s.counted_windows
        ? "out-of-sample kept " + Math.round((wfe || 0) * 100) + "% of in-sample"
        : "out-of-sample against in-sample");

      setText("wfWindows", (s.profitable_windows || 0) + " / " + (s.counted_windows || 0));
      setText("wfWindowsSub", s.counted_windows
        ? Math.round((s.profitable_share || 0) * 100) + "% profitable"
        : "no window reached the trade floor");
      setText("wfMeanReturn", signedPct(s.mean_window_return, 2));
      var mean = el("wfMeanReturn");
      if (mean) mean.className = "value " + cls(s.mean_window_return || 0);
      setText("wfStdev", "std dev " + pctText(s.stdev_window_return, 2));

      renderFlags(s);
      renderWindows(s);
      renderBestParams(s.best_params);
      renderMonteCarlo(run.monte_carlo);
      renderStress(run.stress);
      return loadWalkForwardFeed(run.run_id);
    });
  }

  function renderFlags(s) {
    var box = el("wfFlags");
    if (!box) return;
    var flags = [];
    if (s.overfit) {
      flags.push(["neg", "OVERFIT — window returns swing more than they average"]);
    }
    if (s.fragile) {
      flags.push(["neg", "FRAGILE — failed the crash replay: " +
        (s.fragile_windows || []).join(", ")]);
    }
    if (s.accepted) {
      flags.push(["pos", "Accepted — met every walk-forward, Monte Carlo and stress gate"]);
    }
    (s.reasons || []).forEach(function (r) { flags.push(["warn", "Not accepted: " + r]); });
    if (!flags.length && s.counted_windows) flags.push(["dim", "No flags raised."]);

    box.innerHTML = flags.map(function (f) {
      return '<span class="mode ' + f[0] + '">' + esc(f[1]) + "</span>";
    }).join("");
  }

  function renderWindows(s) {
    var body = el("wfWindowRows");
    if (!body) return;
    var windows = s.windows_detail || [];
    if (!windows.length) {
      // Browsing a different run must not leave the previous one's rows on
      // screen just because this one's summary carries counts but no detail.
      body.innerHTML = "<tr><td colspan='7' class='empty'>No windows reported yet</td></tr>";
      return;
    }
    body.innerHTML = windows.map(function (w, i) {
      var m = w.oos_metrics || {};
      var verdict = !w.counted
        ? '<span class="dim">not enough trades</span>'
        : (w.profitable ? '<span class="pos">held up</span>'
                        : '<span class="neg">did not hold up</span>');
      return "<tr><td>" + (i + 1) + "</td><td class='mono nowrap'>" + esc(w.is_label || "") +
        "</td><td class='mono nowrap'>" + esc(w.oos_label || "") +
        "</td><td class='num'>" + (m.trades || 0) +
        "</td><td class='num " + cls(m.total_return || 0) + "'>" + signedPct(m.total_return, 2) +
        "</td><td class='num'>" + pctText(m.win_rate, 0) +
        "</td><td>" + verdict + "</td></tr>";
    }).join("");
  }

  function renderBestParams(params) {
    var box = el("wfBestParams");
    if (!box) return;
    var keys = params ? Object.keys(params) : [];
    if (!keys.length) {
      box.innerHTML = '<span class="hint">No accepted parameter set for this run yet.</span>';
      return;
    }
    keys.sort();
    box.innerHTML = keys.map(function (k) {
      var v = params[k];
      return '<span class="hint" style="font-size:13px">' + esc(k) +
        ' <span class="mono" style="color:var(--text)">' +
        esc(typeof v === "number" ? String(v) : JSON.stringify(v)) + "</span></span>";
    }).join("");
  }

  function renderRunList(runs, activeRunId) {
    var body = el("wfRunRows");
    if (!body) return;
    if (!runs.length) {
      body.innerHTML = "<tr><td colspan='5' class='empty'>No runs yet</td></tr>";
      return;
    }
    body.innerHTML = runs.map(function (r) {
      var wfe = r.walk_forward_efficiency;
      var active = r.run_id === activeRunId;
      return "<tr class='run-row" + (active ? " active" : "") + "' data-run-id='" + r.run_id +
        "' style='cursor:pointer" + (active ? ";font-weight:600" : "") + "'>" +
        "<td class='mono'>" + r.run_id + (r.label ? " · " + esc(r.label) : "") + "</td>" +
        "<td class='nowrap'>" + fmtTime(r.started_at) + "</td>" +
        "<td class='" + (r.status === "done" ? "pos" : (r.status === "failed" ? "neg" : "dim")) +
        "'>" + esc(r.status) + "</td>" +
        "<td class='num'>" + (r.profitable_windows != null ? r.profitable_windows + "/" + (r.counted_windows || 0) : "—") + "</td>" +
        "<td class='num'>" + (wfe !== undefined && wfe !== null ? wfe.toFixed(2) : "—") + "</td>" +
        "</tr>";
    }).join("");
  }

  document.addEventListener("click", function (ev) {
    var row = ev.target.closest && ev.target.closest("#wfRunRows tr[data-run-id]");
    if (!row) return;
    var runId = Number(row.getAttribute("data-run-id"));
    if (!runId) return;
    wfSelectedRunId = runId;
    loadWalkForward().catch(noop);
  });

  function renderMonteCarlo(mc) {
    if (!mc) return;
    mcData = mc;
    setText("mcMedianReturn", signedPct(mc.median_return, 2));
    setText("mcMeanReturn", signedPct(mc.mean_return, 2));
    setText("mcStdevReturn", pctText(mc.stdev_return, 2));
    setText("mcBestReturn", signedPct(mc.best_return, 2));
    setText("mcWorstReturn", signedPct(mc.worst_return, 2));
    setText("mcP5Return", signedPct(mc.p5_return, 2));
    setText("mcP95Return", signedPct(mc.p95_return, 2));
    setText("mcMedianDd", pctText(mc.median_max_drawdown, 2));
    setText("mcP5Dd", pctText(mc.p5_max_drawdown, 2));
    setText("mcHistoricalDd", pctText(mc.historical_max_drawdown, 2));
    setText("mcLossProb", pctText(mc.probability_of_loss, 1));
    setText("mcRuinProb", pctText(mc.risk_of_ruin, 2));
    setText("wfP5Drawdown", pctText(mc.p5_max_drawdown, 1));
    var exec = mc.execution || {};
    setText("mcSource", (mc.iterations || 0).toLocaleString() + " runs · " +
      (exec.source === "observed"
        ? exec.samples + " observed fills"
        : "assumed execution costs"));
    drawMonteCarlo();
  }

  function drawMonteCarlo() {
    var canvas = el("mcChart");
    if (canvas && mcData) {
      var hist = mcData.drawdown_histogram ||
        (mcData.histograms && mcData.histograms.drawdown);
      if (hist) {
        window.SolChart.histogram(canvas, hist, {
          height: 220,
          marker: mcData.p5_max_drawdown,
          median: mcData.median_max_drawdown,
          format: function (v) { return (v * 100).toFixed(0) + "%"; },
          empty: "No out-of-sample trades to resample yet"
        });
      }
    }
    var pathsCanvas = el("mcPathsChart");
    if (pathsCanvas && mcData) {
      window.SolChart.paths(pathsCanvas, mcData.paths || [], {
        height: 200,
        empty: "No simulated paths yet"
      });
    }
  }

  function renderStress(rows) {
    var body = el("wfStressRows");
    if (!body || !rows || !rows.length) return;
    body.innerHTML = rows.map(function (r) {
      return "<tr><td class='mono'>" + esc(r.name) + "</td><td>" +
        (r.passed ? '<span class="pos">survived</span>' : '<span class="neg">FAILED</span>') +
        "</td><td class='hint'>" + esc(r.note || "") + "</td></tr>";
    }).join("");
  }

  function loadWalkForwardFeed(runId) {
    return get("/api/walkforward/feed?run=" + runId + "&since_id=" + wfFeedSince)
      .then(function (d) {
        var box = el("wfFeed");
        if (!box || !d.lines || !d.lines.length) return;
        if (box.querySelector(".empty")) box.innerHTML = "";
        d.lines.forEach(function (line) {
          if (line.id > wfFeedSince) wfFeedSince = line.id;
          var div = document.createElement("div");
          div.className = "feed-item feed-" + esc(line.level);
          div.innerHTML =
            '<span class="feed-time">' + fmtTime(line.ts) + "</span>" +
            '<span class="feed-msg">' + esc(line.message) + "</span>";
          box.appendChild(div);
        });
        while (box.children.length > 400) box.removeChild(box.firstChild);
        box.scrollTop = box.scrollHeight;
      });
  }

  function loadDrift() {
    if (!el("driftRows")) return Promise.resolve();
    return get("/api/drift?limit=200").then(function (d) {
      var box = el("driftRows");
      var latest = d.latest || {};
      var names = Object.keys(latest);
      box.innerHTML = names.length
        ? names.map(function (n) {
            var s = latest[n];
            var klass = s.status === "drifting" ? "neg"
              : (s.status === "watch" ? "warn" : (s.status === "ok" ? "pos" : "dim"));
            return '<div class="kv"><span>' + esc(n) + '</span><span class="' + klass +
              '">' + esc(s.status) + "</span></div>" +
              '<p class="hint">' + esc((s.detail && s.detail.message) || "") + "</p>";
          }).join("")
        : '<p class="hint">No drift samples yet — a backtest and some live trades are needed first.</p>';

      var series = {};
      (d.samples || []).forEach(function (s) {
        if (!series[s.instance]) series[s.instance] = [];
        series[s.instance].push({ time: s.ts, value: s.live_expectancy });
      });
      var canvas = el("driftChart");
      if (canvas && Object.keys(series).length) {
        window.SolChart.lines(canvas, series, { height: 180 });
      }

      renderDriftCompare(latest, names);
    });
  }

  function renderDriftCompare(latest, names) {
    var body = el("driftCompareRows");
    if (!body) return;
    var usable = names.filter(function (n) {
      var det = latest[n].detail || {};
      return (det.live_trades || 0) > 0 && det.expected && det.expected.trades > 0;
    });
    if (!usable.length) {
      body.innerHTML = "<tr><td colspan='7' class='empty'>No comparable data yet</td></tr>";
      return;
    }
    body.innerHTML = usable.map(function (n) {
      var det = latest[n].detail || {};
      var exp = det.expected || {};
      var wrGap = cls((det.live_win_rate || 0) - (exp.win_rate || 0));
      var expGap = cls((det.live_expectancy || 0) - (exp.expectancy || 0));
      var pfGap = cls((det.live_profit_factor || 0) - (exp.profit_factor || 0));
      return "<tr><td>" + esc(n) + "</td>" +
        "<td class='num " + wrGap + "'>" + pctText(det.live_win_rate, 1) + "</td>" +
        "<td class='num dim'>" + pctText(exp.win_rate, 1) + "</td>" +
        "<td class='num " + expGap + "'>$" + (det.live_expectancy || 0).toFixed(2) + "</td>" +
        "<td class='num dim'>$" + (exp.expectancy || 0).toFixed(2) + "</td>" +
        "<td class='num " + pfGap + "'>" + (det.live_profit_factor || 0).toFixed(2) + "</td>" +
        "<td class='num dim'>" + (exp.profit_factor || 0).toFixed(2) + "</td></tr>";
    }).join("");
  }

  function loadParamSync() {
    if (!el("psBundleRows")) return Promise.resolve();
    return get("/api/paramsync").then(function (d) {
      var last = d.last_bundle || {};
      setText("psLastPull", last.ts
        ? fmtTime(last.ts) + " · " + String(last.source || "")
        : "never");

      var bundles = d.bundles || [];
      if (bundles.length) {
        el("psBundleRows").innerHTML = bundles.map(function (b) {
          var klass = b.status === "promoted" ? "pos"
            : (b.status === "rejected" ? "neg" : (b.status === "shadow" ? "warn" : "dim"));
          return "<tr><td class='mono'>" + esc(b.fingerprint) + "</td><td class='" + klass +
            "'>" + esc(b.status) + "</td><td class='nowrap'>" + fmtTime(b.received_at) +
            "</td><td class='hint'>" + esc(b.note || "") + "</td></tr>";
        }).join("");
      }

      var promotions = d.promotions || [];
      if (promotions.length) {
        el("psPromotionRows").innerHTML = promotions.map(function (p) {
          return "<tr><td class='nowrap'>" + fmtTime(p.ts) + "</td><td class='" +
            (p.promoted ? "pos" : "dim") + "'>" +
            (p.promoted ? "promoted" : "held") + "</td><td class='hint'>" +
            esc(p.reason) + "</td></tr>";
        }).join("");
      }
    });
  }

  /* ---------- parameter sensitivity ---------- */
  var sensitivityData = {};

  function loadSensitivity() {
    var body = el("sensitivityRows");
    if (!body) return Promise.resolve();
    return get("/api/sensitivity").then(function (d) {
      sensitivityData = d.axes || {};
      setText("sensitivitySource", (d.entries_used || 0) + " kept combination" +
        (d.entries_used === 1 ? "" : "s") + " in the library");

      var ranked = d.ranked || [];
      if (!ranked.length) {
        body.innerHTML = "<tr><td colspan='4' class='empty'>Not enough library history yet</td></tr>";
        clear("sensitivityDetail");
        return;
      }
      body.innerHTML = ranked.map(function (axis) {
        var a = sensitivityData[axis];
        return "<tr class='sensitivity-row' data-axis='" + esc(axis) + "' style='cursor:pointer'>" +
          "<td class='mono'>" + esc(axis) + "</td>" +
          "<td class='mono'>" + esc(String(a.best_value)) + "</td>" +
          "<td class='num'>" + a.score_range.toFixed(4) + "</td>" +
          "<td class='num'>" + a.buckets.length + "</td></tr>";
      }).join("");
    });
  }

  document.addEventListener("click", function (ev) {
    var row = ev.target.closest && ev.target.closest("#sensitivityRows tr[data-axis]");
    if (!row) return;
    var axis = row.getAttribute("data-axis");
    var a = sensitivityData[axis];
    var box = el("sensitivityDetail");
    if (!a || !box) return;
    var buckets = a.buckets.slice().sort(function (x, y) { return y.mean_score - x.mean_score; });
    box.innerHTML = "<strong>" + esc(axis) + "</strong> by mean library score: " +
      buckets.map(function (b) {
        return esc(String(b.value)) + " (" + b.mean_score.toFixed(3) + ", n=" + b.count + ")";
      }).join(" · ");
  });

  /* ---------- WF/MC storage browser (spec 6c) ---------- */
  function loadWfmcStorage() {
    if (!el("wfmcStorageRows")) return Promise.resolve();
    return get("/api/wfmc/storage").then(function (d) {
      setText("storageRetention", "kept " + (d.retention_days || 0) + " days");
      var runs = d.runs || [];
      if (!runs.length) return;
      el("wfmcStorageRows").innerHTML = runs.map(function (r) {
        var klass = r.status === "done" ? "pos" : (r.status === "failed" ? "neg" : "dim");
        return "<tr><td class='mono'>" + r.id + (r.label ? " · " + esc(r.label) : "") +
          "</td><td class='nowrap'>" + fmtTime(r.started_at) +
          "</td><td class='nowrap'>" + (r.finished_at ? fmtTime(r.finished_at) : "—") +
          "</td><td class='" + klass + "'>" + esc(r.status) +
          "</td><td class='mono dim'>" + esc(r.bundle || "") + "</td></tr>";
      }).join("");
    });
  }

  function fmtTime(ts) {
    var d = new Date(ts * 1000);
    return d.toLocaleString(undefined, {
      month: "short", day: "numeric", hour: "2-digit", minute: "2-digit"
    });
  }

  function clear(id) { var e = el(id); if (e) e.innerHTML = ""; }

  /* ---------- key rotation ---------- */
  document.addEventListener("submit", function (ev) {
    var form = ev.target;
    if (!form.classList || !form.classList.contains("key-form")) return;
    ev.preventDefault();

    var provider = form.getAttribute("data-provider");
    var input = form.querySelector("input[name=key]");
    var status = form.querySelector(".key-status");
    var button = form.querySelector("button");
    if (!input.value.trim()) { status.textContent = "Enter a key first."; return; }

    button.disabled = true;
    status.className = "key-status hint";
    status.innerHTML = '<span class="spinner">◌</span> Validating against the ' + esc(provider) + " API…";

    fetch("/api/keys/" + encodeURIComponent(provider), {
      method: "POST",
      credentials: "same-origin",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ key: input.value.trim() })
    }).then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
      .then(function (res) {
        button.disabled = false;
        if (res.ok && res.body.ok) {
          status.className = "key-status hint pos";
          status.textContent = "✓ " + res.body.message + " In effect: " + res.body.in_effect_masked;
          input.value = "";
          var masked = form.querySelector(".key-masked");
          if (masked) masked.textContent = res.body.in_effect_masked;
        } else {
          status.className = "key-status hint neg";
          // `message` is generic boilerplate ("failed validation and was not
          // saved"); `error` is the actual reason (the exception RunPod/the
          // provider raised) - show both, or the operator has no way to
          // tell a bad key apart from RunPod being unreachable.
          var reason = res.body.error ? " — " + res.body.error : "";
          status.textContent = "✗ " + (res.body.message || "Validation failed") + reason +
            (res.body.in_effect_masked ? " (still using " + res.body.in_effect_masked + ")" : "");
        }
      }).catch(function (e) {
        button.disabled = false;
        status.className = "key-status hint neg";
        status.textContent = "✗ Request failed: " + e.message;
      });
  });

  /* ---------- bulk data token ---------- */
  var genToken = el("genBulkToken");
  if (genToken) {
    genToken.addEventListener("click", function () {
      if (!confirm(
        "Generate a new bulk data token?\n\n" +
        "The current token stops working immediately. Any optimizer still using it " +
        "will not be able to download history until you update SOLOPT_TOKEN on that machine."
      )) return;

      var status = el("bulkTokenStatus");
      var value = el("bulkTokenValue");
      genToken.disabled = true;
      status.className = "hint";
      status.innerHTML = '<span class="spinner">◌</span> Generating…';

      fetch("/api/keys/bulk_data/generate", { method: "POST", credentials: "same-origin" })
        .then(function (r) { return r.json().then(function (j) { return { ok: r.ok, body: j }; }); })
        .then(function (res) {
          genToken.disabled = false;
          if (res.ok && res.body.ok) {
            status.className = "hint pos";
            status.textContent = "✓ " + res.body.message;
            value.textContent = res.body.token;
          } else {
            status.className = "hint neg";
            status.textContent = "✗ " + (res.body.error || "Could not generate a token");
          }
        })
        .catch(function (e) {
          genToken.disabled = false;
          status.className = "hint neg";
          status.textContent = "✗ " + e.message;
        });
    });
  }

  /* ---------- kill switch two-step confirm ---------- */
  document.addEventListener("submit", function (ev) {
    var form = ev.target;
    if (!form.classList || !form.classList.contains("kill-form")) return;
    if (!confirm(
      "Engage the KILL SWITCH?\n\n" +
      "This halts all trading immediately and does NOT auto-resume — it stays off " +
      "until you re-enable it here.\n\n" +
      "Open positions are not force-closed; they remain for manual closing."
    )) {
      ev.preventDefault();
    }
  });

  /* ---------- loop ---------- */
  function refreshCharts() {
    loadPositions().catch(noop);
    loadEquity().catch(noop);
    loadMarketChart().catch(noop);
    drawMonteCarlo();
  }
  function noop() {}

  function tick() {
    loadStatus().catch(noop);
    loadPositions().catch(noop);
    loadEvents().catch(noop);
    loadReview().catch(noop);
    loadEquity().catch(noop);
    loadMarketChart().catch(noop);
    loadProgress().catch(noop);
    loadWalkForward().catch(noop);
    loadDrift().catch(noop);
    loadParamSync().catch(noop);
    loadWfmcStorage().catch(noop);
    loadSensitivity().catch(noop);
    loadCoverage().catch(noop);
  }

  if (document.body.dataset.live !== "off") {
    tick();
    setInterval(tick, POLL_MS);
    var resizeTimer;
    window.addEventListener("resize", function () {
      clearTimeout(resizeTimer);
      resizeTimer = setTimeout(refreshCharts, 250);
    });
  }
})();
