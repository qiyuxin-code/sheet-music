/**
 * 手绘任意音符草图 → 识别类型与音高 → 规范化为标准五线谱音符 + 简谱。
 */
(function (global) {
  const LAYOUT = {
    width: 960,
    pad_x: 48,
    pad_y: 56,
    line_spacing: 11,
    grand_staff_gap: 16,
    system_gap: 40,
  };

  const BASE_CLEF = { treble: [2, 4], bass: [4, 2] };
  const JIANPU_BASE_OCTAVE = 4;

  let noteIdSeq = 1;

  function buildDefaultStaves() {
    const cfg = LAYOUT;
    const sp = cfg.line_spacing;
    const staffH = sp * 4;
    let y = cfg.pad_y;
    const systems = [{ index: 0, staves: [] }];
    const registry = [];

    function addStaff(clef, key) {
      const lines = [0, 1, 2, 3, 4].map((k) => Math.round(y + k * sp));
      systems[0].staves.push({ key, clef, lines });
      registry.push({ key, clef, lines, top: lines[0], bottom: lines[4] });
      y += staffH;
    }

    addStaff('treble', '0_0');
    y += cfg.grand_staff_gap;
    addStaff('bass', '0_1');
    const height = Math.round(y + cfg.pad_y);

    return { systems, registry, height, layout: { ...cfg } };
  }

  function staffStep(lines, y) {
    const sorted = [...lines].sort((a, b) => a - b);
    const n = sorted.length;
    if (n < 2) return 0;
    const meanGap = (sorted[n - 1] - sorted[0]) / (n - 1);
    let pos;
    if (y <= sorted[0]) pos = (y - sorted[0]) / meanGap;
    else if (y >= sorted[n - 1]) pos = n - 1 + (y - sorted[n - 1]) / meanGap;
    else {
      let i = 0;
      for (let j = 0; j < n - 1; j++) {
        if (sorted[j] <= y) i = j;
      }
      pos = i + (y - sorted[i]) / (sorted[i + 1] - sorted[i]);
    }
    return Math.round(2 * ((n - 1) - pos));
  }

  function yFromStaffStep(lines, step) {
    const sorted = [...lines].sort((a, b) => a - b);
    const meanGap = (sorted[4] - sorted[0]) / 4;
    const pos = 4 - step / 2;
    if (pos <= 0) return sorted[0] + pos * meanGap;
    if (pos >= 4) return sorted[4] + (pos - 4) * meanGap;
    const i = Math.floor(pos);
    const frac = pos - i;
    return sorted[i] + frac * (sorted[i + 1] - sorted[i]);
  }

  function pitchFromY(clef, lines, y) {
    const ref = BASE_CLEF[clef] || BASE_CLEF.treble;
    const step = staffStep(lines, y);
    const total = ref[0] + step;
    const letter = ((total % 7) + 7) % 7;
    const octave = ref[1] + Math.floor(total / 7);
    return { letter, octave };
  }

  function toJianpu(letter, octave) {
    return {
      digit: (letter % 7) + 1,
      dots: octave - JIANPU_BASE_OCTAVE,
      prefix: '',
    };
  }

  function nearestStaff(registry, y) {
    let best = null;
    let bd = Infinity;
    for (const st of registry) {
      const c = (st.top + st.bottom) / 2;
      const d = Math.abs(y - c);
      if (d < bd) {
        bd = d;
        best = st;
      }
    }
    return best;
  }

  function snapX(x) {
    const grid = LAYOUT.line_spacing * 0.65;
    return Math.round(x / grid) * grid;
  }

  function defaultStemUp(lines, y) {
    const mid = (lines[0] + lines[4]) / 2;
    return y >= mid;
  }

  function bboxOf(points) {
    let minX = Infinity;
    let maxX = -Infinity;
    let minY = Infinity;
    let maxY = -Infinity;
    for (const p of points) {
      minX = Math.min(minX, p.x);
      maxX = Math.max(maxX, p.x);
      minY = Math.min(minY, p.y);
      maxY = Math.max(maxY, p.y);
    }
    return { minX, maxX, minY, maxY, w: maxX - minX, h: maxY - minY };
  }

  function strokeFillRatio(points, box) {
    const cols = 14;
    const rows = 14;
    if (box.w < 1 || box.h < 1) return 0.5;
    const grid = new Uint8Array(cols * rows);
    const r = Math.max(1.2, LAYOUT.line_spacing * 0.12);
    for (const p of points) {
      const gx = Math.min(cols - 1, Math.max(0, Math.floor(((p.x - box.minX) / box.w) * cols)));
      const gy = Math.min(rows - 1, Math.max(0, Math.floor(((p.y - box.minY) / box.h) * rows)));
      grid[gy * cols + gx] = 1;
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) {
        const nx = gx + dx;
        const ny = gy + dy;
        if (nx >= 0 && nx < cols && ny >= 0 && ny < rows) grid[ny * cols + nx] = 1;
      }
    }
    let filled = 0;
    for (let i = 0; i < grid.length; i++) if (grid[i]) filled++;
    const perimeter = points.length > 8 ? estimatePerimeterFill(points, box) : 0;
    const density = filled / grid.length;
    return Math.max(0, Math.min(1, density * 1.15 - perimeter * 0.25));
  }

  function estimatePerimeterFill(points, box) {
    const cx = (box.minX + box.maxX) / 2;
    const cy = (box.minY + box.maxY) / 2;
    let edge = 0;
    for (const p of points) {
      const dx = (p.x - cx) / (box.w / 2 || 1);
      const dy = (p.y - cy) / (box.h / 2 || 1);
      if (Math.abs(dx) > 0.55 || Math.abs(dy) > 0.55) edge++;
    }
    return edge / points.length;
  }

  function clusterStemX(points, box) {
    const sp = LAYOUT.line_spacing;
    if (box.h < sp * 1.2) return null;
    const bins = new Map();
    for (const p of points) {
      const k = Math.round(p.x / (sp * 0.35));
      bins.set(k, (bins.get(k) || 0) + 1);
    }
    let bestK = null;
    let bestN = 0;
    for (const [k, n] of bins) {
      if (n > bestN) {
        bestN = n;
        bestK = k;
      }
    }
    if (bestN < points.length * 0.28) return null;
    const stemX = bestK * sp * 0.35;
    const onStem = points.filter((p) => Math.abs(p.x - stemX) <= sp * 0.45);
    const span = onStem.length ? Math.max(...onStem.map((p) => p.y)) - Math.min(...onStem.map((p) => p.y)) : 0;
    if (span < sp * 1.5) return null;
    return stemX;
  }

  function headCenterFromStroke(points, box, stemX) {
    const sp = LAYOUT.line_spacing;
    let pool = points;
    if (stemX != null) {
      pool = points.filter((p) => Math.abs(p.x - stemX) > sp * 0.35);
      if (pool.length < 3) pool = points;
    }
    let sx = 0;
    let sy = 0;
    for (const p of pool) {
      sx += p.x;
      sy += p.y;
    }
    return { x: sx / pool.length, y: sy / pool.length };
  }

  function detectFlag(points, box, stemX, headY) {
    const sp = LAYOUT.line_spacing;
    if (stemX == null) return false;
    const side = points.filter(
      (p) => p.x > stemX + sp * 0.15 && Math.abs(p.y - headY) < sp * 2.2,
    );
    if (side.length < 3) return false;
    const dy = Math.max(...side.map((p) => p.y)) - Math.min(...side.map((p) => p.y));
    const dx = Math.max(...side.map((p) => p.x)) - Math.min(...side.map((p) => p.x));
    return dx > sp * 0.35 && dy > sp * 0.25;
  }

  /**
   * 从手绘笔画推断音符类型与符头中心（未吸附）。
   * @returns {{ x, y, kind, filled, hasStem, stemUp } | null}
   */
  function classifyStroke(points, registry, hintKind) {
    if (!points || points.length < 2) return null;
    const box = bboxOf(points);
    const sp = LAYOUT.line_spacing;
    const travel =
      Math.hypot(points[points.length - 1].x - points[0].x, points[points.length - 1].y - points[0].y);

    if (travel < 5 && box.w < sp * 0.5 && box.h < sp * 0.5) {
      const p = points[0];
      const st = nearestStaff(registry, p.y);
      const stemUp = st ? defaultStemUp(st.lines, p.y) : true;
      const kind = hintKind || 'quarter';
      return {
        x: p.x,
        y: p.y,
        kind,
        filled: kind === 'quarter' || kind === 'eighth',
        hasStem: kind !== 'whole',
        stemUp,
      };
    }

    const stemX = clusterStemX(points, box);
    const head = headCenterFromStroke(points, box, stemX);
    const fillRatio = strokeFillRatio(points, box);
    const tall = box.h > box.w * 1.25 && box.h > sp * 1.1;
    const wide = box.w > sp * 1.35;
    const hasFlag = detectFlag(points, box, stemX, head.y);
    const hasStem = stemX != null || tall;

    let kind = hintKind || 'quarter';
    if (!hintKind) {
      if (hasFlag && fillRatio > 0.35) kind = 'eighth';
      else if (!hasStem && wide && fillRatio < 0.42) kind = 'whole';
      else if (fillRatio < 0.38) kind = hasStem ? 'half' : 'whole';
      else kind = 'quarter';
    }

    let stemUp = true;
    if (stemX != null && tall) {
      const ys = points.filter((p) => Math.abs(p.x - stemX) <= sp * 0.5).map((p) => p.y);
      if (ys.length) {
        const stemTop = Math.min(...ys);
        const stemBot = Math.max(...ys);
        stemUp = head.y > (stemTop + stemBot) / 2;
      }
    } else {
      const st = nearestStaff(registry, head.y);
      stemUp = st ? defaultStemUp(st.lines, head.y) : true;
    }

    return {
      x: head.x,
      y: head.y,
      kind,
      filled: kind === 'quarter' || kind === 'eighth',
      hasStem: kind !== 'whole',
      stemUp,
    };
  }

  function normalizeNote(raw, registry) {
    const st = registry.find((s) => s.key === raw.staffKey) || nearestStaff(registry, raw.y);
    if (!st) return null;
    const step = staffStep(st.lines, raw.y);
    const y = yFromStaffStep(st.lines, step);
    const x = snapX(raw.x);
    const pitch = pitchFromY(st.clef, st.lines, y);
    const jp = toJianpu(pitch.letter, pitch.octave);
    const kind = raw.kind || (raw.filled === false ? 'half' : 'quarter');
    const stemUp = raw.stemUp !== undefined ? raw.stemUp : defaultStemUp(st.lines, y);
    return {
      id: raw.id || noteIdSeq++,
      staffKey: st.key,
      x: Math.max(LAYOUT.pad_x + 12, Math.min(LAYOUT.width - LAYOUT.pad_x - 12, x)),
      y: Math.round(y),
      kind,
      filled: kind === 'quarter' || kind === 'eighth',
      stemUp,
      letter: pitch.letter,
      octave: pitch.octave,
      digit: jp.digit,
      dots: jp.dots,
      prefix: jp.prefix,
    };
  }

  function buildLabels(notes, registry) {
    const sp = LAYOUT.line_spacing;
    const byStaff = new Map();
    for (const n of notes) {
      if (!byStaff.has(n.staffKey)) byStaff.set(n.staffKey, []);
      byStaff.get(n.staffKey).push(n);
    }
    const labels = [];
    for (const [key, group] of byStaff) {
      const st = registry.find((s) => s.key === key);
      if (!st) continue;
      const baselineY = st.top - sp * 0.55;
      const sorted = [...group].sort((a, b) => a.x - b.x);
      for (const n of sorted) {
        labels.push({
          x: n.x,
          y: baselineY,
          digit: n.digit,
          dots: n.dots,
          prefix: n.prefix,
        });
      }
    }
    return labels;
  }

  function notesToScene(notes, stavesMeta) {
    const sceneNotes = notes.map((n) => ({
      x: n.x,
      y: n.y,
      kind: n.kind,
      filled: n.filled,
      stemUp: n.stemUp,
    }));
    return {
      width: LAYOUT.width,
      height: stavesMeta.height,
      layout: stavesMeta.layout,
      systems: stavesMeta.systems,
      notes: sceneNotes,
      labels: buildLabels(notes, stavesMeta.registry),
    };
  }

  function hitNote(notes, px, py, spacing) {
    const rw = spacing * 0.65;
    const rh = spacing * 0.55;
    for (let i = notes.length - 1; i >= 0; i--) {
      const n = notes[i];
      const dx = (px - n.x) / rw;
      const dy = (py - n.y) / rh;
      if (dx * dx + dy * dy <= 1.35) return n;
      const stemX = n.x + spacing * 0.55 * 0.82;
      const stemLen = spacing * 3.5;
      const y0 = n.stemUp ? n.y - rh : n.y + rh;
      const y1 = n.stemUp ? y0 - stemLen : y0 + stemLen;
      const sy = Math.min(y0, y1);
      const ey = Math.max(y0, y1);
      if (px >= stemX - 4 && px <= stemX + 4 && py >= sy && py <= ey) return n;
    }
    return null;
  }

  function createEditor(canvas, opts = {}) {
    const stavesMeta = buildDefaultStaves();
    let notes = [];
    let mode = 'draw';
    let strokeHint = null;
    let drag = null;
    let selectedId = null;
    let sketchPoints = [];
    let isDrawing = false;
    const onChange = opts.onChange || (() => {});

    function scene() {
      return notesToScene(notes, stavesMeta);
    }

    function redraw() {
      if (global.SheetCanvas) {
        global.SheetCanvas.drawScene(canvas, scene(), {
          sketch: isDrawing ? sketchPoints : null,
        });
      }
      onChange(scene());
    }

    function commitStroke(points) {
      const inferred = classifyStroke(points, stavesMeta.registry, strokeHint);
      if (!inferred) return;
      const st = nearestStaff(stavesMeta.registry, inferred.y);
      const n = normalizeNote(
        {
          staffKey: st ? st.key : '0_0',
          x: inferred.x,
          y: inferred.y,
          kind: inferred.kind,
          filled: inferred.filled,
          stemUp: inferred.stemUp,
        },
        stavesMeta.registry,
      );
      if (!n) return;
      notes.push(n);
      selectedId = n.id;
    }

    function removeNote(id) {
      notes = notes.filter((n) => n.id !== id);
      if (selectedId === id) selectedId = null;
      redraw();
    }

    function clearAll() {
      notes = [];
      selectedId = null;
      redraw();
    }

    function removeLast() {
      if (!notes.length) return;
      notes.pop();
      selectedId = null;
      redraw();
    }

    function removeSelected() {
      if (selectedId != null) removeNote(selectedId);
      else removeLast();
    }

    function canvasCoords(evt) {
      const rect = canvas.getBoundingClientRect();
      const sx = parseFloat(canvas.style.width) || rect.width;
      const sy = parseFloat(canvas.style.height) || rect.height;
      return {
        x: ((evt.clientX - rect.left) / rect.width) * sx,
        y: ((evt.clientY - rect.top) / rect.height) * sy,
      };
    }

    function onPointerDown(evt) {
      if (evt.button !== 0) return;
      evt.preventDefault();
      const { x, y } = canvasCoords(evt);
      const sp = LAYOUT.line_spacing;
      const hit = hitNote(notes, x, y, sp);
      if (mode === 'erase') {
        if (hit) removeNote(hit.id);
        return;
      }
      if (hit) {
        selectedId = hit.id;
        drag = {
          id: hit.id,
          startX: x,
          startY: y,
          noteX: hit.x,
          noteY: hit.y,
        };
        return;
      }
      isDrawing = true;
      sketchPoints = [{ x, y }];
      redraw();
    }

    function onPointerMove(evt) {
      const { x, y } = canvasCoords(evt);
      if (isDrawing) {
        const last = sketchPoints[sketchPoints.length - 1];
        if (Math.hypot(x - last.x, y - last.y) >= 1.5) {
          sketchPoints.push({ x, y });
          redraw();
        }
        return;
      }
      if (!drag) return;
      const idx = notes.findIndex((n) => n.id === drag.id);
      if (idx < 0) return;
      const st = nearestStaff(stavesMeta.registry, y);
      const prev = notes[idx];
      notes[idx] = normalizeNote(
        {
          id: prev.id,
          staffKey: st ? st.key : prev.staffKey,
          x: drag.noteX + (x - drag.startX),
          y: drag.noteY + (y - drag.startY),
          kind: prev.kind,
          filled: prev.filled,
          stemUp: prev.stemUp,
        },
        stavesMeta.registry,
      );
      redraw();
    }

    function onPointerUp() {
      if (isDrawing) {
        commitStroke(sketchPoints);
        isDrawing = false;
        sketchPoints = [];
        redraw();
        return;
      }
      drag = null;
    }

    canvas.addEventListener('mousedown', onPointerDown);
    window.addEventListener('mousemove', onPointerMove);
    window.addEventListener('mouseup', onPointerUp);
    canvas.addEventListener('contextmenu', (e) => {
      e.preventDefault();
      const { x, y } = canvasCoords(e);
      const hit = hitNote(notes, x, y, LAYOUT.line_spacing);
      if (hit) removeNote(hit.id);
    });

    function setMode(m) {
      mode = m;
      isDrawing = false;
      sketchPoints = [];
    }
    function setStrokeHint(kind) {
      strokeHint = kind;
    }

    function exportPng() {
      if (!global.SheetCanvas) return '';
      return global.SheetCanvas.sceneToDataUrl(scene());
    }

    redraw();

    return {
      getScene: scene,
      setMode,
      setStrokeHint,
      clearAll,
      exportPng,
      getNotes: () => notes.slice(),
      removeLast,
      removeSelected,
      classifyStroke: (pts) => classifyStroke(pts, stavesMeta.registry, strokeHint),
    };
  }

  global.SheetEditor = {
    LAYOUT,
    buildDefaultStaves,
    staffStep,
    pitchFromY,
    toJianpu,
    classifyStroke,
    notesToScene,
    createEditor,
  };
})(typeof window !== 'undefined' ? window : globalThis);
