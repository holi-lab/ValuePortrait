/* Shared helpers and chart builders for the Value Portrait project page. Plain JS, no dependencies. */
(function () {
  var VP = window.VP = {};
  VP.PVQ = ['Universalism', 'Benevolence', 'Conformity', 'Tradition', 'Security', 'Power', 'Achievement', 'Hedonism', 'Stimulation', 'Self_Direction'];
  VP.HI = ['Self_Transcendence', 'Conservation', 'Self_Enhancement', 'Openness_to_Change'];
  VP.BFI = ['Openness', 'Conscientiousness', 'Extraversion', 'Agreeableness', 'Neuroticism'];
  VP.ABBR = { Universalism: 'Uni', Benevolence: 'Ben', Conformity: 'Con', Tradition: 'Tra', Security: 'Sec', Power: 'Pow', Achievement: 'Ach', Hedonism: 'Hed', Stimulation: 'Sti', Self_Direction: 'Sel',
    Self_Transcendence: 'ST', Conservation: 'CO', Self_Enhancement: 'SE', Openness_to_Change: 'OC',
    Openness: 'Ope', Conscientiousness: 'Csc', Extraversion: 'Ext', Agreeableness: 'Agr', Neuroticism: 'Neu' };
  // Higher-order group used only for colouring the ten values (Hedonism drawn with Openness to Change).
  VP.GROUP = { Universalism: 'st', Benevolence: 'st', Conformity: 'co', Tradition: 'co', Security: 'co', Power: 'se', Achievement: 'se', Hedonism: 'oc', Stimulation: 'oc', Self_Direction: 'oc',
    Self_Transcendence: 'st', Conservation: 'co', Self_Enhancement: 'se', Openness_to_Change: 'oc' };
  VP.GROUP_COLOR = { st: 'var(--c1)', se: 'var(--c2)', co: 'var(--c3)', oc: 'var(--c4)' };
  VP.GROUP_NAME = { st: 'Self-Transcendence', se: 'Self-Enhancement', co: 'Conservation', oc: 'Openness to Change' };
  VP.LIKERT = ['Not like me at all', 'Not like me', 'A little like me', 'Somewhat like me', 'Like me', 'Very much like me'];
  VP.FAMILIES = ['OpenAI', 'Anthropic', 'Google', 'Qwen', 'Mistral', 'Llama', 'DeepSeek', 'xAI', 'Qwen2.5-Instruct', 'DeepSeek-R1-Distill-Qwen', 'Gemma3-it'];
  VP.SRC = { Reddit: 'Reddit (AITA)', DearAbby: 'Dear Abby', ShareGPT: 'ShareGPT', LMSYS: 'LMSYS' };

  VP.name = function (d) { return ({ Self_Direction: 'Self-Direction', Self_Transcendence: 'Self-Transcendence', Self_Enhancement: 'Self-Enhancement', Openness_to_Change: 'Openness to Change' })[d] || d; };
  VP.dimColor = function (d) { var g = VP.GROUP[d]; return g ? VP.GROUP_COLOR[g] : 'var(--c1)'; };
  VP.fmt = function (n, d) { if (n === null || n === undefined || isNaN(n)) return '—'; var s = Math.abs(n).toFixed(d); return (n < 0 ? '−' : '') + s; };
  VP.fmtS = function (n, d) { if (n === null || n === undefined || isNaN(n)) return '—'; var s = Math.abs(n).toFixed(d); return (n < 0 ? '−' : '+') + s; };
  VP.esc = function (s) { return String(s === null || s === undefined ? '' : s).replace(/[&<>"']/g, function (c) { return ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[c]; }); };
  VP.param = function (k) { var m = new RegExp('[?&]' + k + '=([^&#]*)').exec(location.search); return m ? decodeURIComponent(m[1].replace(/\+/g, ' ')) : null; };
  VP.setParam = function (obj) { var u = new URL(location.href); Object.keys(obj).forEach(function (k) { if (obj[k] === null || obj[k] === undefined || obj[k] === '') u.searchParams.delete(k); else u.searchParams.set(k, obj[k]); }); history.replaceState(null, '', u.toString()); };
  VP.mean = function (a) { var s = 0, n = 0; a.forEach(function (v) { if (v !== null && v !== undefined && !isNaN(v)) { s += v; n++; } }); return n ? s / n : null; };
  VP.variance = function (a) { var m = VP.mean(a); if (m === null) return null; var s = 0, n = 0; a.forEach(function (v) { if (v !== null && v !== undefined) { s += (v - m) * (v - m); n++; } }); return s / n; };

  // DOM helpers ---------------------------------------------------------------
  VP.h = function (tag, attrs, children) {
    var e = document.createElement(tag);
    if (attrs) Object.keys(attrs).forEach(function (k) {
      var v = attrs[k];
      if (k === 'class') e.className = v;
      else if (k === 'html') e.innerHTML = v;
      else if (k === 'text') e.textContent = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(e.style, v);
      else if (k.slice(0, 2) === 'on') e.addEventListener(k.slice(2), v);
      else if (v !== null && v !== undefined && v !== false) e.setAttribute(k, v === true ? '' : v);
    });
    if (children !== undefined && children !== null) (Array.isArray(children) ? children : [children]).forEach(function (c) {
      if (c === null || c === undefined || c === false) return;
      e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    });
    return e;
  };
  var SVGNS = 'http://www.w3.org/2000/svg';
  VP.s = function (tag, attrs, parent) { var e = document.createElementNS(SVGNS, tag); if (attrs) Object.keys(attrs).forEach(function (k) { e.setAttribute(k, attrs[k]); }); if (parent) parent.appendChild(e); return e; };
  VP.clear = function (e) { while (e.firstChild) e.removeChild(e.firstChild); return e; };
  var cache = {};
  VP.get = function (url) { if (!cache[url]) cache[url] = fetch(url).then(function (r) { if (!r.ok) throw new Error(url + ' ' + r.status); return r.json(); }); return cache[url]; };
  VP.compact = function () { return window.matchMedia && window.matchMedia('(max-width: 640px)').matches; };

  // Tooltip (delegated, works for HTML and SVG nodes carrying data-tip) -------
  var tip;
  function tipEl() { if (!tip) { tip = document.getElementById('tip') || document.body.appendChild(VP.h('div', { class: 'tip', id: 'tip', hidden: true })); } return tip; }
  function moveTip(e) { var t = tipEl(); var x = e.clientX + 14, y = e.clientY + 14; var r = t.getBoundingClientRect(); if (x + r.width > window.innerWidth - 8) x = e.clientX - r.width - 10; if (y + r.height > window.innerHeight - 8) y = e.clientY - r.height - 10; t.style.left = x + 'px'; t.style.top = y + 'px'; }
  document.addEventListener('mouseover', function (e) { var n = e.target.closest && e.target.closest('[data-tip]'); if (!n) return; var t = tipEl(); t.innerHTML = n.getAttribute('data-tip'); t.hidden = false; moveTip(e); });
  document.addEventListener('mousemove', function (e) { if (tip && !tip.hidden) moveTip(e); });
  document.addEventListener('mouseout', function (e) { var n = e.target.closest && e.target.closest('[data-tip]'); if (n && tip) tip.hidden = true; });

  // Nav: mark current page --------------------------------------------------------
  document.addEventListener('DOMContentLoaded', function () {
    var here = location.pathname.split('/').pop() || 'index.html';
    document.querySelectorAll('.topnav a').forEach(function (a) { var href = a.getAttribute('href').split('#')[0]; if (href === here || (here === '' && href === 'index.html')) a.setAttribute('aria-current', 'page'); });
    var copy = document.getElementById('copyBib');
    if (copy) copy.addEventListener('click', function () {
      var txt = document.getElementById('bib').textContent;
      function done(ok) { copy.textContent = ok ? 'Copied' : 'Select and copy'; setTimeout(function () { copy.textContent = 'Copy'; }, 1800); }
      if (navigator.clipboard && navigator.clipboard.writeText) navigator.clipboard.writeText(txt).then(function () { done(true); }, function () { done(false); }); else done(false);
    });
  });

  // Generic table ------------------------------------------------------------------
  VP.table = function (o) {
    var t = VP.h('table', { class: 'data' + (o.cls ? ' ' + o.cls : '') });
    if (o.head) { var tr = VP.h('tr'); o.head.forEach(function (hd, i) { tr.appendChild(VP.h('th', { class: (i === 0 || (o.left && o.left.indexOf(i) >= 0)) ? 'l' : '', html: hd })); }); t.appendChild(VP.h('thead', null, tr)); }
    var tb = VP.h('tbody');
    o.rows.forEach(function (r) {
      var cls = r.cls || ''; var cells = r.cells || r;
      var tr = VP.h('tr', { class: cls });
      cells.forEach(function (c, i) {
        var isObj = c && typeof c === 'object' && !(c instanceof Node);
        var v = isObj ? c.v : c; var cc = (i === 0 || (o.left && o.left.indexOf(i) >= 0)) ? 'l' : 'num';
        if (isObj && c.cls) cc += ' ' + c.cls;
        if (typeof v === 'number') { var txt = VP.fmtS ? (o.signed ? VP.fmtS(v, o.dec === undefined ? 2 : o.dec) : VP.fmt(v, o.dec === undefined ? 2 : o.dec)) : v; if (v < 0) cc += ' neg'; tr.appendChild(VP.h('td', { class: cc, text: txt })); }
        else if (v instanceof Node) tr.appendChild(VP.h('td', { class: cc }, v));
        else tr.appendChild(VP.h('td', { class: cc, html: v }));
      });
      tb.appendChild(tr);
    });
    t.appendChild(tb);
    return t;
  };

  // Dot strip: one row per dimension, one dot per model ----------------------------
  VP.dotStrip = function (svg, o) {
    VP.clear(svg);
    var compact = VP.compact();
    var W = compact ? 460 : 780, LW = compact ? 104 : 150, RW = 18, ROW = compact ? 30 : 34, TOP = 30, PAD = 10;
    var H = TOP + o.rows.length * ROW + PAD;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var x0 = LW, x1 = W - RW, dom = o.domain;
    var sx = function (v) { return x0 + (v - dom[0]) / (dom[1] - dom[0]) * (x1 - x0); };
    var ticks = o.ticks || [-1, -0.5, 0, 0.5, 1];
    ticks.forEach(function (t) {
      if (t < dom[0] || t > dom[1]) return;
      VP.s('line', { x1: sx(t), x2: sx(t), y1: TOP - 6, y2: H - PAD, class: t === 0 ? 'zero' : 'grid' }, svg);
      var tx = VP.s('text', { x: sx(t), y: TOP - 12, 'text-anchor': 'middle', class: 'axis' }, svg); tx.textContent = VP.fmtS(t, 1).replace('+0.0', '0').replace('−0.0', '0');
    });
    if (o.axisLabel) { var al = VP.s('text', { x: x1, y: H - 1, 'text-anchor': 'end', class: 'axis' }, svg); al.textContent = o.axisLabel; }
    o.rows.forEach(function (r, i) {
      var y = TOP + i * ROW + ROW / 2;
      VP.s('line', { x1: x0, x2: x1, y1: y, y2: y, class: 'grid' }, svg);
      if (r.color) VP.s('rect', { x: 4, y: y - 5, width: 10, height: 10, class: 'swatch', style: 'fill:' + r.color }, svg);
      var lb = VP.s('text', { x: LW - 12, y: y + 4.5, 'text-anchor': 'end', class: 'lbl' }, svg); lb.textContent = compact ? (r.short || r.label) : r.label;
      if (o.means && o.means[r.key] !== undefined && o.means[r.key] !== null) {
        VP.s('line', { x1: sx(o.means[r.key]), x2: sx(o.means[r.key]), y1: y - 9, y2: y + 9, class: 'mean' }, svg).setAttribute('data-tip', 'Mean of all models: ' + VP.fmtS(o.means[r.key], 2));
      }
      var pts = o.points.map(function (p, j) { return { p: p, v: p.vals[r.key], j: j }; }).filter(function (q) { return q.v !== null && q.v !== undefined; });
      pts.sort(function (a, b) { return a.v - b.v; });
      var sel = [];
      pts.forEach(function (q, k) {
        var st = o.state ? o.state(q.p) : '';
        var dy = ((k % 5) - 2) * 3.2;
        var c = VP.s('circle', { cx: sx(Math.max(dom[0], Math.min(dom[1], q.v))), cy: y + dy, r: st ? 5 : 3.6, class: 'dot' + (st ? ' ' + st : '') }, svg);
        c.setAttribute('data-tip', '<b>' + VP.esc(q.p.name) + '</b> · ' + VP.esc(q.p.family) + '<br>' + VP.esc(r.label) + ': ' + VP.fmtS(q.v, 2));
        if (o.onClick) c.addEventListener('click', function () { o.onClick(q.p); });
        if (st === 'sel') sel.push(c);
      });
      sel.forEach(function (c) { svg.appendChild(c); });
    });
  };

  // Horizontal diverging bars --------------------------------------------------------
  VP.bars = function (svg, o) {
    VP.clear(svg);
    var compact = VP.compact();
    var W = o.width || (compact ? 460 : 700), LW = compact ? 104 : 150, RW = 44, ROW = o.row || 28, TOP = 26, PAD = 8, BH = o.barH || 16;
    var H = TOP + o.rows.length * ROW + PAD;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var maxAbs = o.max || Math.max.apply(null, o.rows.map(function (r) { return Math.max(Math.abs(r.value || 0), Math.abs(r.ghost || 0)); }).concat([0.3]));
    var dom = o.domain || [-maxAbs, maxAbs];
    var x0 = LW, x1 = W - RW; var sx = function (v) { return x0 + (v - dom[0]) / (dom[1] - dom[0]) * (x1 - x0); };
    var ticks = o.ticks; if (!ticks) { var step = maxAbs > 0.8 ? 0.5 : 0.25; ticks = []; for (var t = -Math.ceil(maxAbs / step) * step; t <= maxAbs + 1e-9; t += step) ticks.push(Math.round(t * 100) / 100); }
    ticks.forEach(function (t) { if (t < dom[0] || t > dom[1]) return; VP.s('line', { x1: sx(t), x2: sx(t), y1: TOP - 6, y2: H - PAD, class: t === 0 ? 'zero' : 'grid' }, svg); var tx = VP.s('text', { x: sx(t), y: TOP - 12, 'text-anchor': 'middle', class: 'axis' }, svg); tx.textContent = t === 0 ? '0' : VP.fmtS(t, 2).replace(/0+$/, '').replace(/\.$/, ''); });
    o.rows.forEach(function (r, i) {
      var y = TOP + i * ROW + ROW / 2;
      var g = VP.s('g', { class: 'row' }, svg);
      var hit = VP.s('rect', { x: 0, y: y - ROW / 2, width: W, height: ROW, class: 'rowhit' }, g);
      if (r.color && o.swatch !== false) VP.s('rect', { x: 4, y: y - 5, width: 10, height: 10, class: 'swatch', style: 'fill:' + r.color }, g);
      var lb = VP.s('text', { x: LW - 12, y: y + 4.5, 'text-anchor': 'end', class: 'lbl' + (r.sel ? ' sel' : '') }, g); lb.textContent = compact ? (r.short || r.label) : r.label;
      if (r.sel) lb.setAttribute('font-weight', '700');
      var v = r.value;
      if (v !== null && v !== undefined) {
        var a = sx(Math.min(0, v)), b = sx(Math.max(0, v));
        VP.s('rect', { x: a, y: y - BH / 2, width: Math.max(1, b - a), height: BH, class: 'bar', style: 'fill:' + (r.color || (v < 0 ? 'var(--neg)' : 'var(--pos)')) + (r.dim ? ';opacity:.45' : '') }, g);
        var vt = VP.s('text', { x: v < 0 ? a - 5 : b + 5, y: y + 4, 'text-anchor': v < 0 ? 'end' : 'start', class: 'val' }, g); vt.textContent = VP.fmtS(v, 2);
      }
      if (r.ghost !== null && r.ghost !== undefined) {
        var ga = sx(Math.min(0, r.ghost)), gb = sx(Math.max(0, r.ghost));
        VP.s('rect', { x: ga, y: y - BH / 2 - 2, width: Math.max(1, gb - ga), height: BH + 4, class: 'ghost' }, g);
      }
      if (r.tip) g.setAttribute('data-tip', r.tip);
      if (o.onClick) { g.style.cursor = 'pointer'; g.addEventListener('click', function () { o.onClick(r); }); }
    });
    if (o.legend) {
      var lx = LW; o.legend.forEach(function (l) { var gg = VP.s('g', {}, svg); if (l.ghost) VP.s('rect', { x: lx, y: H - 4, width: 14, height: 8, class: 'ghost' }, gg); else VP.s('rect', { x: lx, y: H - 4, width: 14, height: 8, class: 'swatch', style: 'fill:' + l.color }, gg); var tt = VP.s('text', { x: lx + 18, y: H + 4, class: 'legend' }, gg); tt.textContent = l.label; lx += 18 + l.label.length * 6.6 + 16; });
      svg.setAttribute('viewBox', '0 0 ' + W + ' ' + (H + 12));
    }
  };

  // Grouped horizontal bars (k bars per row) ---------------------------------------
  VP.groupBars = function (svg, o) {
    VP.clear(svg);
    var compact = VP.compact();
    var k = o.groups.length, BH = o.barH || 9, GAP = 2;
    var W = o.width || 470, LW = o.labelW || (compact ? 80 : 110), RW = 40, ROW = k * (BH + GAP) + 8, TOP = 44, PAD = 8;
    var H = TOP + o.rows.length * ROW + PAD;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var maxAbs = o.max || 0.8; var dom = [-maxAbs, maxAbs];
    var x0 = LW, x1 = W - RW; var sx = function (v) { return x0 + (v - dom[0]) / (dom[1] - dom[0]) * (x1 - x0); };
    if (o.title) { var tt = VP.s('text', { x: LW, y: 14, class: 'lbl', 'font-weight': '700' }, svg); tt.textContent = o.title; }
    var ticks = o.ticks || [-0.5, 0, 0.5];
    ticks.forEach(function (t) { if (t < dom[0] || t > dom[1]) return; VP.s('line', { x1: sx(t), x2: sx(t), y1: TOP - 4, y2: H - PAD, class: t === 0 ? 'zero' : 'grid' }, svg); var tx = VP.s('text', { x: sx(t), y: TOP - 8, 'text-anchor': 'middle', class: 'axis' }, svg); tx.textContent = t === 0 ? '0' : VP.fmtS(t, 1); });
    o.rows.forEach(function (r, i) {
      var y0 = TOP + i * ROW + 4;
      var lb = VP.s('text', { x: LW - 10, y: y0 + (ROW - 8) / 2 + 4, 'text-anchor': 'end', class: 'lbl' }, svg); lb.textContent = compact ? (r.short || r.label) : r.label;
      o.groups.forEach(function (gr, j) {
        var v = o.data[i][j]; if (v === null || v === undefined) return;
        var y = y0 + j * (BH + GAP);
        var a = sx(Math.max(dom[0], Math.min(0, v))), b = sx(Math.min(dom[1], Math.max(0, v)));
        var rect = VP.s('rect', { x: a, y: y, width: Math.max(1, b - a), height: BH, class: 'bar', style: 'fill:' + gr.color }, svg);
        rect.setAttribute('data-tip', '<b>' + VP.esc(gr.label) + '</b> · ' + VP.esc(r.label) + ': ' + VP.fmtS(v, 2));
        var hit = VP.s('rect', { x: x0, y: y - 1, width: x1 - x0, height: BH + 2, fill: 'transparent' }, svg); hit.setAttribute('data-tip', rect.getAttribute('data-tip'));
      });
    });
  };

  // Rating strip: models on a 1–6 Likert axis ------------------------------------------
  VP.ratingStrip = function (svg, o) {
    VP.clear(svg);
    var compact = VP.compact();
    var W = compact ? 460 : 760, L = 16, R = 16, TOP = 14, H = 64;
    svg.setAttribute('viewBox', '0 0 ' + W + ' ' + H);
    var sx = function (v) { return L + (v - 1) / 5 * (W - L - R); };
    var SHORT = ['Not at all', 'Not like me', 'A little like me', 'Somewhat like me', 'Like me', 'Very much like me'];
    for (var t = 1; t <= 6; t++) { VP.s('line', { x1: sx(t), x2: sx(t), y1: TOP, y2: TOP + 26, class: 'grid' }, svg); var tx = VP.s('text', { x: sx(t), y: H - 4, 'text-anchor': t === 1 ? 'start' : (t === 6 ? 'end' : 'middle'), class: 'axis' }, svg); tx.textContent = compact ? String(t) : (t + ' ' + SHORT[t - 1]); }
    VP.s('line', { x1: sx(1), x2: sx(6), y1: TOP + 13, y2: TOP + 13, class: 'grid' }, svg);
    if (o.mean !== null && o.mean !== undefined) { var m = VP.s('line', { x1: sx(o.mean), x2: sx(o.mean), y1: TOP - 2, y2: TOP + 28, class: 'mean' }, svg); m.setAttribute('data-tip', 'Mean over models: ' + o.mean.toFixed(2)); }
    var pts = o.points.filter(function (p) { return p.v !== null && p.v !== undefined; }).sort(function (a, b) { return a.v - b.v; });
    var sel = [];
    pts.forEach(function (p, k) {
      var dy = ((k % 5) - 2) * 4.2;
      var c = VP.s('circle', { cx: sx(p.v), cy: TOP + 13 + dy, r: p.sel ? 5.5 : 4, class: 'dot' + (p.sel ? ' sel' : '') }, svg);
      c.setAttribute('data-tip', '<b>' + VP.esc(p.name) + '</b><br>' + p.v.toFixed(2) + ' (mean of 6 prompts)');
      if (o.onClick) c.addEventListener('click', function () { o.onClick(p); });
      if (p.sel) sel.push(c);
    });
    sel.forEach(function (c) { svg.appendChild(c); });
  };

  // Heat-tinted table -------------------------------------------------------------------
  VP.tint = function (v, max, mode) {
    if (v === null || v === undefined || isNaN(v)) return '';
    var a = Math.min(1, Math.abs(v) / max) * 55;
    var col = (mode === 'seq') ? 'var(--pos)' : (v < 0 ? 'var(--neg)' : 'var(--pos)');
    return 'background: color-mix(in srgb, ' + col + ' ' + a.toFixed(0) + '%, transparent);';
  };
  VP.heat = function (table, o) {
    VP.clear(table);
    var thead = VP.h('thead'), tr = VP.h('tr');
    tr.appendChild(VP.h('th', { class: 'l', html: o.corner || '' }));
    o.cols.forEach(function (c, j) {
      var th = VP.h('th', { class: 'sortable' + (o.sortCol === j ? ' sorted' : ''), html: c.label, title: c.title || '' });
      if (o.onSort) th.addEventListener('click', function () { o.onSort(j); });
      tr.appendChild(th);
    });
    (o.extra || []).forEach(function (x) { var th = VP.h('th', { class: 'sortable' + (o.sortCol === 'x' + x.key ? ' sorted' : ''), html: x.label, title: x.title || '' }); if (o.onSort) th.addEventListener('click', function () { o.onSort('x' + x.key); }); tr.appendChild(th); });
    thead.appendChild(tr); table.appendChild(thead);
    var tb = VP.h('tbody');
    o.rows.forEach(function (r) {
      if (r.group) { var g = VP.h('tr', { class: 'group' }); g.appendChild(VP.h('td', { class: 'l', colspan: 1 + o.cols.length + (o.extra || []).length, html: r.group })); tb.appendChild(g); return; }
      var row = VP.h('tr', { class: (o.onRow ? 'clickable' : '') + (r.sel ? ' sel' : '') });
      row.appendChild(VP.h('td', { class: 'l', html: r.label }));
      var vals = r.cells.filter(function (v) { return v !== null && v !== undefined; });
      var mx = Math.max.apply(null, vals), mn = Math.min.apply(null, vals);
      r.cells.forEach(function (v) {
        var cls = 'num'; if (o.marks && vals.length > 1) { if (v === mx) cls += ' mark max'; else if (v === mn) cls += ' mark min'; }
        var td = VP.h('td', { class: cls, style: VP.tint(v, o.max, o.mode), text: v === null || v === undefined ? '—' : (o.signed ? VP.fmtS(v, o.dec || 2) : VP.fmt(v, o.dec || 2)) });
        row.appendChild(td);
      });
      (r.extra || []).forEach(function (x) { row.appendChild(VP.h('td', { class: 'num', html: x })); });
      if (o.onRow) row.addEventListener('click', function () { o.onRow(r); });
      tb.appendChild(row);
    });
    table.appendChild(tb);
  };

  // Correlation block for one response ----------------------------------------------------
  VP.corrList = function (obj, dims, opts) {
    var wrap = VP.h('div');
    var entries = dims.map(function (d) { return { d: d, r: obj[d][0], p: obj[d][1] }; });
    if (opts && opts.sort) entries.sort(function (a, b) { return Math.abs(b.r) - Math.abs(a.r); });
    entries.forEach(function (e) {
      var sig = Math.abs(e.r) >= 0.3 && e.p < 0.05;
      var row = VP.h('div', { class: 'corr' + (sig ? ' sig' : '') });
      row.appendChild(VP.h('span', { class: 'cl', text: VP.name(e.d) }));
      var bar = VP.h('span', { class: 'bar' });
      var w = Math.min(50, Math.abs(e.r) / 0.6 * 50);
      bar.appendChild(VP.h('i', { class: e.r < 0 ? 'neg' : 'pos', style: e.r < 0 ? ('right:50%;width:' + w + '%') : ('left:50%;width:' + w + '%') }));
      row.appendChild(bar);
      row.appendChild(VP.h('span', { class: 'cv', text: VP.fmtS(e.r, 2) }));
      row.appendChild(VP.h('span', { class: 'cp', text: e.p < 0.001 ? 'p<.001' : ('p=' + e.p.toFixed(3).replace(/^0/, '')) }));
      wrap.appendChild(row);
    });
    return wrap;
  };
  VP.sigTags = function (resp, opts) {
    var tags = [];
    [['pvq', VP.PVQ, ''], ['bfi', VP.BFI, ' bfi']].forEach(function (fam) {
      fam[1].forEach(function (d) { var r = resp[fam[0]][d][0], p = resp[fam[0]][d][1]; if (Math.abs(r) >= 0.3 && p < 0.05) tags.push({ fam: fam[0], d: d, r: r, cls: fam[2] }); });
    });
    tags.sort(function (a, b) { return Math.abs(b.r) - Math.abs(a.r); });
    return tags;
  };
})();
