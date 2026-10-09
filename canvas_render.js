/**
 * 规范化五线谱 Canvas 渲染：统一线距谱表 + 标准音符 + 简谱数字。
 */
(function (global) {
  const JIANPU_COLOR = '#16a34a';
  const STAFF_COLOR = '#1f2937';
  const NOTE_COLOR = '#111827';

  function drawStaffLines(ctx, lines, padX, width) {
    ctx.save();
    ctx.strokeStyle = STAFF_COLOR;
    ctx.lineWidth = 1;
    ctx.beginPath();
    const x0 = padX;
    const x1 = width - padX;
    for (const y of lines) {
      ctx.moveTo(x0, y);
      ctx.lineTo(x1, y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawLabel(ctx, lab, fontSize) {
    fontSize = lab.font_size || fontSize;
    const text = `${lab.prefix || ''}${lab.digit}`;
    ctx.save();
    ctx.fillStyle = lab.color || JIANPU_COLOR;
    ctx.font = `600 ${fontSize}px system-ui, -apple-system, "PingFang SC", sans-serif`;
    ctx.textAlign = 'center';
    ctx.textBaseline = 'bottom';
    const x = lab.x;
    const baseY = lab.y;
    ctx.fillText(text, x, baseY);
    const w = ctx.measureText(text).width;
    const dotR = Math.max(1.5, fontSize * 0.1);
    const cx = x + w * 0.15;
    if (lab.dots > 0) {
      for (let k = 0; k < lab.dots; k++) {
        ctx.beginPath();
        ctx.arc(cx, baseY - fontSize - 2 - k * (dotR * 2 + 3), dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    } else if (lab.dots < 0) {
      for (let k = 0; k < -lab.dots; k++) {
        ctx.beginPath();
        ctx.arc(cx, baseY + 4 + k * (dotR * 2 + 3), dotR, 0, Math.PI * 2);
        ctx.fill();
      }
    }
    ctx.restore();
  }

  function drawNoteHead(ctx, x, y, rw, rh, filled) {
    ctx.beginPath();
    ctx.ellipse(x, y, rw, rh, 0, 0, Math.PI * 2);
    if (!filled) {
      ctx.lineWidth = 1.25;
      ctx.strokeStyle = NOTE_COLOR;
      ctx.stroke();
    } else {
      ctx.fillStyle = NOTE_COLOR;
      ctx.fill();
    }
  }

  function drawStemAndFlag(ctx, note, spacing, rw, rh) {
    const kind = note.kind || 'quarter';
    if (kind === 'whole') return;
    const stemLen = spacing * 3.5;
    const stemUp = note.stemUp !== undefined ? note.stemUp : true;
    const stemX = xStem(note.x, rw, stemUp);
    const yAttach = stemUp ? note.y - rh * 0.85 : note.y + rh * 0.85;
    const yEnd = stemUp ? yAttach - stemLen : yAttach + stemLen;
    ctx.strokeStyle = NOTE_COLOR;
    ctx.lineWidth = 1.15;
    ctx.beginPath();
    ctx.moveTo(stemX, yAttach);
    ctx.lineTo(stemX, yEnd);
    ctx.stroke();
    if (kind === 'eighth') {
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      if (stemUp) {
        ctx.moveTo(stemX, yEnd);
        ctx.quadraticCurveTo(stemX + spacing * 0.9, yEnd + spacing * 0.35, stemX + spacing * 0.75, yEnd + spacing * 1.1);
      } else {
        ctx.moveTo(stemX, yEnd);
        ctx.quadraticCurveTo(stemX + spacing * 0.9, yEnd - spacing * 0.35, stemX + spacing * 0.75, yEnd - spacing * 1.1);
      }
      ctx.stroke();
    }
  }

  function xStem(noteX, rw, stemUp) {
    return noteX + rw * 0.82;
  }

  function drawNote(ctx, note, spacing) {
    ctx.save();
    const kind = note.kind || (note.filled === false ? 'half' : 'quarter');
    const scale = kind === 'whole' ? 1.12 : 1;
    const rw = spacing * 0.55 * scale;
    const rh = spacing * 0.42 * scale;
    const filled = kind === 'quarter' || kind === 'eighth';
    drawNoteHead(ctx, note.x, note.y, rw, rh, filled);
    drawStemAndFlag(ctx, note, spacing, rw, rh);
    ctx.restore();
  }

  function drawSketch(ctx, points) {
    if (!points || points.length < 2) return;
    ctx.save();
    ctx.strokeStyle = 'rgba(59, 130, 246, 0.75)';
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.moveTo(points[0].x, points[0].y);
    for (let i = 1; i < points.length; i++) {
      ctx.lineTo(points[i].x, points[i].y);
    }
    ctx.stroke();
    ctx.restore();
  }

  function drawScene(canvas, scene, opts = {}) {
    const dpr = opts.dpr || window.devicePixelRatio || 1;
    const w = scene.width || 960;
    const h = scene.height || 400;
    canvas.width = Math.round(w * dpr);
    canvas.height = Math.round(h * dpr);
    canvas.style.width = `${w}px`;
    canvas.style.height = `${h}px`;
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(0, 0, w, h);

    const spacing = (scene.layout && scene.layout.line_spacing) || 11;
    const padX = (scene.layout && scene.layout.pad_x) || 48;
    for (const sys of scene.systems || []) {
      for (const st of sys.staves || []) {
        drawStaffLines(ctx, st.lines, padX, w);
      }
    }
    for (const note of scene.notes || []) {
      drawNote(ctx, note, spacing);
    }
    const fontSize = Math.round(spacing * 1.35);
    const labels = (scene.labels || []).slice().sort((a, b) => a.x - b.x);
    for (const lab of labels) {
      drawLabel(ctx, lab, fontSize);
    }
    if (opts.sketch && opts.sketch.length) {
      drawSketch(ctx, opts.sketch);
    }
    if (typeof opts.overlay === 'function') {
      opts.overlay(ctx, w, h);
    }
    return { width: w, height: h };
  }

  function sceneToDataUrl(scene, opts = {}) {
    const c = document.createElement('canvas');
    drawScene(c, scene, opts);
    return c.toDataURL('image/jpeg', opts.quality || 0.92);
  }

  global.SheetCanvas = { drawScene, sceneToDataUrl, drawNote, drawSketch };
})(typeof window !== 'undefined' ? window : globalThis);
