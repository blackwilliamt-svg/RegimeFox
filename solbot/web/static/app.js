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

  /* ---------- market chart (always visible, symbol search over the universe) ---------- */
  var marketUniverse = [];        // [{mint, symbol, label}], sorted by volume desc (server order)
  var marketMint = null;          // currently displayed mint
  var marketUniverseLoaded = false;

  function marketLabel(row) {
    return (row.symbol || row.mint.slice(0, 8)) + " — " + row.mint.slice(0, 6) + "…";
  }

  function loadMarketUniverse() {
    var input = el("marketSymbolInput");
    if (!input) return Promise.resolve();
    return get("/api/universe").then(function (rows) {
      marketUniverse = rows.map(function (r) {
        return { mint: r.mint, symbol: r.symbol, label: marketLabel(r) };
      });
      var list = el("marketSymbolDatalist");
      if (list) {
        list.innerHTML = marketUniverse.map(function (r) {
          return '<option value="' + esc(r.label) + '">';
        }).join("");
      }
      marketUniverseLoaded = true;
      // Default to the top-ranked (highest 24h volume) coin once, and whenever
      // the previously-selected mint drops out of the universe.
      var stillPresent = marketMint && marketUniverse.some(function (r) { return r.mint === marketMint; });
      if (!stillPresent && marketUniverse.length) {
        marketMint = marketUniverse[0].mint;
        input.value = marketUniverse[0].label;
      }
    });
  }

  function resolveMarketSymbolInput() {
    var input = el("marketSymbolInput");
    if (!input) return;
    var typed = input.value.trim().toLowerCase();
    if (!typed) return;
    var hit = marketUniverse.find(function (r) {
      return r.label.toLowerCase() === typed || (r.symbol || "").toLowerCase() === typed;
    });
    if (hit && hit.mint !== marketMint) {
      marketMint = hit.mint;
      loadMarketChart();
    }
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

  function loadMarketChart() {
    var canvas = el("marketChart");
    var hint = el("marketChartHint");
    if (!canvas) return Promise.resolve();

    var afterUniverse = marketUniverseLoaded ? Promise.resolve() : loadMarketUniverse();
    return afterUniverse.then(function () {
      if (!marketMint) {
        if (hint) hint.textContent = "No coins in the universe yet — try Refresh universe.";
        renderRegimeReadout(false, null);
        return;
      }
      if (hint) {
        var row = marketUniverse.find(function (r) { return r.mint === marketMint; });
        hint.textContent = row ? (row.symbol || marketMint) + " · " + marketUniverse.length +
          " coins in the current universe" : "";
      }
      var toggle = el("marketRegimeToggle");
      var withRegime = toggle && toggle.checked;
      var url = "/api/candles/" + encodeURIComponent(marketMint) + "?limit=200" +
        (withRegime ? "&regime=1" : "");
      return get(url).then(function (d) {
        window.SolChart.candles(canvas, d.candles, {
          height: 300,
          regime: withRegime && d.regime ? d.regime.history : null
        });
        renderRegimeReadout(withRegime, d.regime);
      });
    });
  }

  (function wireMarketSearch() {
    var input = el("marketSymbolInput");
    if (!input) return;
    input.addEventListener("change", resolveMarketSymbolInput);
    input.addEventListener("blur", resolveMarketSymbolInput);
  })();

  (function wireMarketRegimeToggle() {
    var toggle = el("marketRegimeToggle");
    if (!toggle) return;
    toggle.addEventListener("change", function () { loadMarketChart().catch(noop); });
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

  /* ---------- progress (backtest page) ---------- */
  function loadProgress() {
    if (!el("pullProgress") && !el("backtestProgress") && !el("regimePassProgress")) {
      return Promise.resolve();
    }
    return get("/api/progress").then(function (d) {
      bar("pull", d.historical_pull);
      bar("dailyPull", d.daily_incremental_pull);
      bar("backtest", d.backtest);
      bar("regimePass", d.regime_pass);
      var cov = el("candleCoverage");
      if (cov && d.candle_coverage) {
        var c = d.candle_coverage;
        cov.textContent = (c.tokens || 0) + " coins, " + (c.candles || 0).toLocaleString() +
          " candles, " + (c.days || 0) + " days (" +
          ((c.disk && c.disk.megabytes) || 0).toLocaleString() + " MB on disk)";
      }
    });
  }

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
  }

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
    });
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
          status.textContent = "✗ " + (res.body.message || res.body.error || "Validation failed") +
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
    loadEquity().catch(noop);
    loadMarketChart().catch(noop);
    loadProgress().catch(noop);
    loadWalkForward().catch(noop);
    loadDrift().catch(noop);
    loadParamSync().catch(noop);
    loadWfmcStorage().catch(noop);
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
