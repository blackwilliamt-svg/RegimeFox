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

  /* Two open positions render as two clearly labelled charts side by side. */
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
    if (!el("pullProgress") && !el("backtestProgress")) return Promise.resolve();
    return get("/api/progress").then(function (d) {
      bar("pull", d.historical_pull);
      bar("backtest", d.backtest);
      var b = el("budgetUsed");
      if (b && d.birdeye_budget) {
        var used = d.birdeye_budget.units || 0;
        var limit = d.birdeye_budget.limit || 0;
        b.textContent = used.toLocaleString() + " / " + limit.toLocaleString() + " CU" +
          (limit ? " (" + ((used / limit) * 100).toFixed(1) + "%)" : "");
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
  }
  function noop() {}

  function tick() {
    loadStatus().catch(noop);
    loadPositions().catch(noop);
    loadEvents().catch(noop);
    loadEquity().catch(noop);
    loadProgress().catch(noop);
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
