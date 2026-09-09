/* Lightweight canvas charts: candlesticks with trade overlays, and line charts.
 *
 * Written locally rather than pulled from a CDN for two reasons: the dashboard
 * ships a strict Content-Security-Policy that forbids third-party scripts, and
 * a purpose-built renderer keeps the droplet's CPU out of chart work entirely
 * (all drawing happens in the browser). Swap in Lightweight-Charts by vendoring
 * the file into static/ and reimplementing the two render functions below.
 */
(function (global) {
  "use strict";

  function css(name) {
    return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  }

  function theme() {
    return {
      text: css("--text") || "#e6edf3",
      muted: css("--muted") || "#8b949e",
      border: css("--border") || "#2a313c",
      panel: css("--panel") || "#161b22",
      green: css("--green") || "#3fb950",
      red: css("--red") || "#f85149",
      amber: css("--amber") || "#d29922",
      accent: css("--accent") || "#4f9cf9",
      purple: css("--purple") || "#a371f7"
    };
  }

  function setupCanvas(canvas, height) {
    var dpr = global.devicePixelRatio || 1;
    var width = canvas.parentElement.clientWidth || 600;
    canvas.width = Math.floor(width * dpr);
    canvas.height = Math.floor(height * dpr);
    canvas.style.width = width + "px";
    canvas.style.height = height + "px";
    var ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);
    return { ctx: ctx, width: width, height: height };
  }

  function niceTicks(min, max, count) {
    if (!isFinite(min) || !isFinite(max) || min === max) return [min];
    var span = max - min;
    var step = Math.pow(10, Math.floor(Math.log10(span / count)));
    var err = (span / count) / step;
    if (err >= 7.5) step *= 10; else if (err >= 3) step *= 5; else if (err >= 1.5) step *= 2;
    var out = [];
    for (var v = Math.ceil(min / step) * step; v <= max; v += step) out.push(v);
    return out;
  }

  function fmtPrice(v) {
    if (v === 0) return "0";
    var abs = Math.abs(v);
    if (abs >= 1000) return v.toFixed(0);
    if (abs >= 1) return v.toFixed(2);
    if (abs >= 0.01) return v.toFixed(4);
    return v.toPrecision(3);
  }

  function fmtTime(ts) {
    var d = new Date(ts * 1000);
    return String(d.getHours()).padStart(2, "0") + ":" + String(d.getMinutes()).padStart(2, "0");
  }

  function fmtDate(ts) {
    var d = new Date(ts * 1000);
    return (d.getMonth() + 1) + "/" + d.getDate();
  }

  /* ------------------------------------------------------------------ */
  /* Candlestick chart with entry/exit markers and stop levels           */
  /* ------------------------------------------------------------------ */
  function candles(canvas, data, opts) {
    opts = opts || {};
    var t = theme();
    var height = opts.height || 300;
    var s = setupCanvas(canvas, height);
    var ctx = s.ctx;

    if (!data || !data.length) {
      ctx.fillStyle = t.muted;
      ctx.font = "13px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No candle data yet", s.width / 2, height / 2);
      return;
    }

    var regimeBand = (opts.regime && opts.regime.length) ? 8 : 0;
    var showVolume = opts.volume !== false;
    var padL = 8, padR = 62, padT = 10 + regimeBand, padB = 22;
    var volH = showVolume ? Math.round(height * 0.18) : 0;
    var plotH = height - padT - padB - volH;
    var plotW = s.width - padL - padR;

    var levels = [];
    if (opts.levels) {
      opts.levels.forEach(function (l) { if (l && isFinite(l.value) && l.value > 0) levels.push(l); });
    }

    var overlays = opts.overlays || {};

    var lo = Infinity, hi = -Infinity, maxVol = 0;
    data.forEach(function (c) {
      if (c.low < lo) lo = c.low;
      if (c.high > hi) hi = c.high;
      if (c.volume > maxVol) maxVol = c.volume;
    });
    levels.forEach(function (l) {
      if (l.value < lo) lo = l.value;
      if (l.value > hi) hi = l.value;
    });
    // Overlay lines (SMA/EMA/Bollinger) can run outside the candle high/low
    // range (a band widens past a spike, an EMA lags through one) - fold
    // them into the autoscale so they never get silently clipped off.
    function widenForSeries(values) {
      (values || []).forEach(function (v) {
        if (v === null || v === undefined || !isFinite(v)) return;
        if (v < lo) lo = v;
        if (v > hi) hi = v;
      });
    }
    (overlays.sma || []).forEach(function (o) { widenForSeries(o.values); });
    (overlays.ema || []).forEach(function (o) { widenForSeries(o.values); });
    if (overlays.bbands) { widenForSeries(overlays.bbands.upper); widenForSeries(overlays.bbands.lower); }

    var pad = (hi - lo) * 0.06 || hi * 0.02 || 1;
    lo -= pad; hi += pad;

    function x(i) { return padL + (i + 0.5) * (plotW / data.length); }
    function y(v) { return padT + plotH - ((v - lo) / (hi - lo)) * plotH; }

    /* grid + price axis */
    ctx.font = "10px ui-monospace, monospace";
    ctx.textBaseline = "middle";
    niceTicks(lo, hi, 5).forEach(function (v) {
      var py = y(v);
      ctx.strokeStyle = t.border;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(padL, py);
      ctx.lineTo(padL + plotW, py);
      ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = t.muted;
      ctx.textAlign = "left";
      ctx.fillText(fmtPrice(v), padL + plotW + 6, py);
    });

    /* volume */
    if (showVolume) {
      var volTop = padT + plotH + 6;
      data.forEach(function (c, i) {
        if (!maxVol) return;
        var h = (c.volume / maxVol) * (volH - 6);
        ctx.fillStyle = c.close >= c.open ? t.green : t.red;
        ctx.globalAlpha = 0.25;
        var w = Math.max(1, plotW / data.length - 1.5);
        ctx.fillRect(x(i) - w / 2, volTop + (volH - 6 - h), w, h);
        ctx.globalAlpha = 1;
      });
    }

    /* candles */
    var cw = Math.max(1, plotW / data.length - 1.5);
    data.forEach(function (c, i) {
      var up = c.close >= c.open;
      var color = up ? t.green : t.red;
      var cx = x(i);
      ctx.strokeStyle = color;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(cx, y(c.high));
      ctx.lineTo(cx, y(c.low));
      ctx.stroke();
      var yo = y(c.open), yc = y(c.close);
      var top = Math.min(yo, yc);
      var bh = Math.max(1, Math.abs(yc - yo));
      ctx.fillStyle = color;
      ctx.fillRect(cx - cw / 2, top, cw, bh);
    });

    /* indicator overlays: SMA/EMA lines, Bollinger Bands (dashboard step 2).
     * A null entry (not-enough-warm-up-yet, or a gap) breaks the stroke into
     * a new segment rather than drawing a straight line across the hole. */
    function strokeSeries(values, color, dash) {
      if (!values || !values.length) return;
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.4;
      ctx.setLineDash(dash || []);
      ctx.beginPath();
      var drawing = false;
      for (var i = 0; i < values.length && i < data.length; i++) {
        var v = values[i];
        if (v === null || v === undefined || !isFinite(v)) { drawing = false; continue; }
        var px = x(i), py = y(v);
        if (!drawing) { ctx.moveTo(px, py); drawing = true; } else { ctx.lineTo(px, py); }
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }

    if (overlays.bbands) {
      var bb = overlays.bbands;
      strokeSeries(bb.upper, t.purple, [3, 3]);
      strokeSeries(bb.lower, t.purple, [3, 3]);
      strokeSeries(bb.mid, t.purple);
    }
    var smaColors = [t.accent, t.amber];
    (overlays.sma || []).forEach(function (o, i) { strokeSeries(o.values, smaColors[i % smaColors.length]); });
    var emaColors = [t.green, t.red];
    (overlays.ema || []).forEach(function (o, i) { strokeSeries(o.values, emaColors[i % emaColors.length], [6, 3]); });

    /* horizontal levels: hard stop, trailing stop, target, entry */
    levels.forEach(function (l) {
      var py = y(l.value);
      ctx.strokeStyle = l.color || t.muted;
      ctx.lineWidth = 1;
      ctx.setLineDash(l.dash || [4, 3]);
      ctx.beginPath();
      ctx.moveTo(padL, py);
      ctx.lineTo(padL + plotW, py);
      ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = l.color || t.muted;
      ctx.textAlign = "left";
      ctx.font = "9px ui-monospace, monospace";
      ctx.fillText(l.label || "", padL + 3, py - 5);
    });

    /* entry / exit markers */
    (opts.markers || []).forEach(function (m) {
      var idx = -1, best = Infinity;
      data.forEach(function (c, i) {
        var d = Math.abs(c.time - m.time);
        if (d < best) { best = d; idx = i; }
      });
      if (idx < 0) return;
      var cx = x(idx), cy = y(m.price);
      ctx.fillStyle = m.color || t.accent;
      ctx.beginPath();
      if (m.shape === "down") {
        ctx.moveTo(cx, cy - 9); ctx.lineTo(cx - 5, cy - 17); ctx.lineTo(cx + 5, cy - 17);
      } else {
        ctx.moveTo(cx, cy + 9); ctx.lineTo(cx - 5, cy + 17); ctx.lineTo(cx + 5, cy + 17);
      }
      ctx.closePath();
      ctx.fill();
    });

    /* time axis */
    ctx.fillStyle = t.muted;
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    var every = Math.max(1, Math.floor(data.length / 6));
    for (var i = 0; i < data.length; i += every) {
      ctx.fillText(fmtTime(data[i].time), x(i), height - padB + 6);
    }

    /* fuzzy-regime overlay (fuzzy-regime section, step 6): a thin strip along
     * the top, one segment per bar, colored by that bar's dominant regime
     * cluster - a quick "where has this coin's character been" read at a
     * glance, with the exact membership breakdown left to the readout text
     * beside the chart rather than crowding it onto the strip itself. */
    if (regimeBand) {
      var byTs = {};
      opts.regime.forEach(function (r) { byTs[r.ts] = r; });
      var clusterColors = [t.accent, t.green, t.amber, t.purple, t.red, t.muted];
      var stripY = padT - regimeBand;
      var sw = Math.max(1, plotW / data.length);
      data.forEach(function (c, i) {
        var r = byTs[c.time];
        if (!r) return;
        ctx.fillStyle = clusterColors[r.dominant % clusterColors.length];
        ctx.fillRect(x(i) - sw / 2, stripY, sw, regimeBand - 1);
      });
    }
  }

  /* ------------------------------------------------------------------ */
  /* RSI sub-panel - a thin 0-100 line chart sharing the price chart's bar
   * spacing (same `data` array) so the two line up visually when stacked. */
  /* ------------------------------------------------------------------ */
  function rsi(canvas, data, values, opts) {
    opts = opts || {};
    var t = theme();
    var height = opts.height || 90;
    var s = setupCanvas(canvas, height);
    var ctx = s.ctx;

    if (!data || !data.length || !values || !values.length) {
      return;
    }

    var padL = 8, padR = 62, padT = 8, padB = 16;
    var plotH = height - padT - padB;
    var plotW = s.width - padL - padR;
    function x(i) { return padL + (i + 0.5) * (plotW / data.length); }
    function y(v) { return padT + plotH - (v / 100) * plotH; }

    [30, 50, 70].forEach(function (v) {
      var py = y(v);
      ctx.strokeStyle = t.border;
      ctx.globalAlpha = v === 50 ? 0.35 : 0.5;
      ctx.setLineDash(v === 50 ? [2, 3] : []);
      ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(padL + plotW, py); ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
      ctx.fillStyle = t.muted;
      ctx.font = "9px ui-monospace, monospace";
      ctx.textAlign = "left";
      ctx.textBaseline = "middle";
      ctx.fillText(String(v), padL + plotW + 6, py);
    });

    ctx.strokeStyle = t.accent;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    var drawing = false;
    for (var i = 0; i < values.length && i < data.length; i++) {
      var v = values[i];
      if (v === null || v === undefined || !isFinite(v)) { drawing = false; continue; }
      var px = x(i), py = y(v);
      if (!drawing) { ctx.moveTo(px, py); drawing = true; } else { ctx.lineTo(px, py); }
    }
    ctx.stroke();

    ctx.fillStyle = t.muted;
    ctx.font = "10px ui-monospace, monospace";
    ctx.textAlign = "left";
    ctx.textBaseline = "top";
    ctx.fillText("RSI", padL, 2);
  }

  /* ------------------------------------------------------------------ */
  /* Multi-series line chart (equity curves)                            */
  /* ------------------------------------------------------------------ */
  function lines(canvas, series, opts) {
    opts = opts || {};
    var t = theme();
    var height = opts.height || 240;
    var s = setupCanvas(canvas, height);
    var ctx = s.ctx;

    var names = Object.keys(series).filter(function (k) { return series[k] && series[k].length; });
    if (!names.length) {
      ctx.fillStyle = t.muted;
      ctx.font = "13px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText("No equity history yet", s.width / 2, height / 2);
      return;
    }

    var colors = { live: t.red, paper: t.accent, shadow: t.purple };
    var padL = 8, padR = 62, padT = 12, padB = 22;
    var plotW = s.width - padL - padR;
    var plotH = height - padT - padB;

    var tMin = Infinity, tMax = -Infinity, vMin = Infinity, vMax = -Infinity;
    names.forEach(function (n) {
      series[n].forEach(function (p) {
        if (p.time < tMin) tMin = p.time;
        if (p.time > tMax) tMax = p.time;
        if (p.value < vMin) vMin = p.value;
        if (p.value > vMax) vMax = p.value;
      });
    });
    if (tMax === tMin) tMax = tMin + 1;
    var vPad = (vMax - vMin) * 0.08 || Math.abs(vMax) * 0.05 || 1;
    vMin -= vPad; vMax += vPad;

    function x(ts) { return padL + ((ts - tMin) / (tMax - tMin)) * plotW; }
    function y(v) { return padT + plotH - ((v - vMin) / (vMax - vMin)) * plotH; }

    ctx.font = "10px ui-monospace, monospace";
    ctx.textBaseline = "middle";
    niceTicks(vMin, vMax, 4).forEach(function (v) {
      var py = y(v);
      ctx.strokeStyle = t.border;
      ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(padL + plotW, py); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = t.muted;
      ctx.textAlign = "left";
      ctx.fillText("$" + fmtPrice(v), padL + plotW + 6, py);
    });

    names.forEach(function (n) {
      var pts = series[n];
      ctx.strokeStyle = colors[n] || t.accent;
      ctx.lineWidth = 1.8;
      ctx.beginPath();
      pts.forEach(function (p, i) {
        var px = x(p.time), py = y(p.value);
        if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    });

    ctx.fillStyle = t.muted;
    ctx.textAlign = "center";
    ctx.textBaseline = "top";
    for (var i = 0; i <= 4; i++) {
      var ts = tMin + (tMax - tMin) * (i / 4);
      ctx.fillText(fmtDate(ts), padL + plotW * (i / 4), height - padB + 6);
    }
  }

  /* ---------- histogram ----------
   * Used for the Monte Carlo drawdown distribution. The point of the chart is
   * the tail, not the mode, so the 5%-worst-case marker is drawn on top of the
   * bars rather than left to a legend.
   */
  function histogram(canvas, data, opts) {
    opts = opts || {};
    var t = theme();
    var height = opts.height || 220;
    var s = setupCanvas(canvas, height);
    var ctx = s.ctx;

    var edges = (data && data.edges) || [];
    var counts = (data && data.counts) || [];
    if (edges.length < 2 || !counts.length) {
      ctx.fillStyle = t.muted;
      ctx.font = "13px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(opts.empty || "No distribution yet", s.width / 2, height / 2);
      return;
    }

    var padL = 8, padR = 8, padT = 10, padB = 26;
    var plotW = s.width - padL - padR;
    var plotH = height - padT - padB;
    var maxCount = 0;
    counts.forEach(function (c) { if (c > maxCount) maxCount = c; });
    if (!maxCount) maxCount = 1;

    var lo = edges[0], hi = edges[edges.length - 1];
    if (hi === lo) hi = lo + 1;
    function x(v) { return padL + ((v - lo) / (hi - lo)) * plotW; }

    var barW = Math.max(1, plotW / counts.length - 1);
    for (var i = 0; i < counts.length; i++) {
      var h = (counts[i] / maxCount) * plotH;
      var past = opts.marker !== undefined && edges[i] >= opts.marker;
      ctx.fillStyle = past ? t.red : t.accent;
      ctx.globalAlpha = past ? 0.85 : 0.55;
      ctx.fillRect(x(edges[i]), padT + plotH - h, barW, h);
    }
    ctx.globalAlpha = 1;

    [["median", opts.median, t.green], ["p5", opts.marker, t.red]].forEach(function (m) {
      if (m[1] === undefined || m[1] === null) return;
      var px = x(m[1]);
      if (px < padL || px > padL + plotW) return;
      ctx.strokeStyle = m[2];
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(px, padT);
      ctx.lineTo(px, padT + plotH);
      ctx.stroke();
      ctx.setLineDash([]);
    });

    ctx.fillStyle = t.muted;
    ctx.font = "11px system-ui, sans-serif";
    ctx.textBaseline = "top";
    var fmt = opts.format || function (v) { return v.toFixed(2); };
    for (var k = 0; k <= 4; k++) {
      var v = lo + (hi - lo) * (k / 4);
      ctx.textAlign = k === 0 ? "left" : (k === 4 ? "right" : "center");
      ctx.fillText(fmt(v), x(v), height - padB + 8);
    }
  }

  /* ---------- paths ----------
   * Several equity curves over a shared step axis (trade index, not wall-clock
   * time) - the Monte Carlo worst-case drawdown-curve view (spec 6a). The
   * worst path is drawn last and thickest so it reads clearly on top of the
   * others rather than getting lost among nine similar lines.
   */
  function paths(canvas, series, opts) {
    opts = opts || {};
    var t = theme();
    var height = opts.height || 200;
    var s = setupCanvas(canvas, height);
    var ctx = s.ctx;

    if (!series || !series.length) {
      ctx.fillStyle = t.muted;
      ctx.font = "13px system-ui, sans-serif";
      ctx.textAlign = "center";
      ctx.fillText(opts.empty || "No simulated paths yet", s.width / 2, height / 2);
      return;
    }

    var padL = 8, padR = 46, padT = 10, padB = 20;
    var plotW = s.width - padL - padR;
    var plotH = height - padT - padB;
    var steps = series[0].length;

    var vMin = Infinity, vMax = -Infinity;
    series.forEach(function (path) {
      path.forEach(function (v) { if (v < vMin) vMin = v; if (v > vMax) vMax = v; });
    });
    if (vMax === vMin) vMax = vMin + 0.01;
    var pad = (vMax - vMin) * 0.06;
    vMin -= pad; vMax += pad;

    function x(i) { return padL + (i / Math.max(1, steps - 1)) * plotW; }
    function y(v) { return padT + plotH - ((v - vMin) / (vMax - vMin)) * plotH; }

    ctx.font = "10px ui-monospace, monospace";
    ctx.textBaseline = "middle";
    niceTicks(vMin, vMax, 4).forEach(function (v) {
      var py = y(v);
      ctx.strokeStyle = t.border;
      ctx.globalAlpha = 0.5;
      ctx.beginPath(); ctx.moveTo(padL, py); ctx.lineTo(padL + plotW, py); ctx.stroke();
      ctx.globalAlpha = 1;
      ctx.fillStyle = t.muted;
      ctx.textAlign = "left";
      ctx.fillText((v * 100).toFixed(0) + "%", padL + plotW + 6, py);
    });

    // Baseline at 100% (starting balance), so a path's dip below it is visible
    // at a glance rather than inferred from the axis labels.
    if (vMin < 1 && vMax > 1) {
      ctx.strokeStyle = t.muted;
      ctx.globalAlpha = 0.6;
      ctx.setLineDash([3, 3]);
      ctx.beginPath(); ctx.moveTo(padL, y(1)); ctx.lineTo(padL + plotW, y(1)); ctx.stroke();
      ctx.setLineDash([]);
      ctx.globalAlpha = 1;
    }

    // Backend sorts worst-first; draw it last (on top) so it reads clearly
    // over the other nine rather than getting drawn-under and obscured.
    var ordered = series.slice(1).map(function (path) { return { path: path, worst: false }; });
    ordered.push({ path: series[0], worst: true });
    ordered.forEach(function (entry) {
      ctx.strokeStyle = entry.worst ? t.red : t.amber;
      ctx.globalAlpha = entry.worst ? 1 : 0.45;
      ctx.lineWidth = entry.worst ? 2 : 1;
      ctx.beginPath();
      entry.path.forEach(function (v, k) {
        var px = x(k), py = y(v);
        if (k === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
      });
      ctx.stroke();
    });
    ctx.globalAlpha = 1;
  }

  global.SolChart = {
    candles: candles, lines: lines, histogram: histogram, paths: paths, rsi: rsi, theme: theme
  };
})(window);
