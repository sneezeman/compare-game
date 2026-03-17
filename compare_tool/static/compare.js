// Compare Game — Frontend Logic

const state = {
    // Multi-select: [{expId, filename, numEpochs, width, height}, ...]
    selected: [],
    roi: null,  // single ROI string "x,y,w,h" applied to all GIFs
    rO: 0.5,
    // Tournament
    sessionId: null,
    currentPair: null,
    showingSide: 'left',
    metricsA: null,
    metricsB: null,
    spaceCount: 0,
    spaceTimer: null,
    // Internal
    _roiMouseDown: null,
    _roiMouseMove: null,
    _roiMouseUp: null,
};

// =========================================================================
// Utilities
// =========================================================================

function metricColor(t) {
    // t in [0, 1]: 0 = worst (red), 1 = best (green)
    // Interpolate from #ff4444 (red) through #ffcc44 (yellow) to #44dd44 (green)
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.5) {
        const s = t * 2; // 0..1 for red-to-yellow
        r = 255;
        g = Math.round(100 + 155 * s);
        b = 68;
    } else {
        const s = (t - 0.5) * 2; // 0..1 for yellow-to-green
        r = Math.round(255 - 187 * s);
        g = Math.round(255 - 34 * s);
        b = 68;
    }
    return `rgb(${r},${g},${b})`;
}

// =========================================================================
// Phase management
// =========================================================================

function showPhase(phase) {
    document.querySelectorAll('.phase').forEach(el => el.classList.remove('active'));
    document.getElementById(`phase-${phase}`).classList.add('active');
}

// =========================================================================
// Phase 1: Upload & Selection
// =========================================================================

function initUpload() {
    const zone = document.getElementById('upload-zone');
    const fileInput = document.getElementById('file-input');

    zone.addEventListener('click', () => fileInput.click());

    zone.addEventListener('dragover', e => {
        e.preventDefault();
        zone.classList.add('dragover');
    });
    zone.addEventListener('dragleave', () => zone.classList.remove('dragover'));

    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('dragover');
        for (const f of e.dataTransfer.files) {
            uploadFile(f);
        }
    });

    fileInput.addEventListener('change', () => {
        for (const f of fileInput.files) {
            uploadFile(f);
        }
        fileInput.value = '';
    });

    loadPreloaded();
}

async function loadPreloaded() {
    try {
        const resp = await fetch('/api/experiments');
        if (!resp.ok) return;
        const data = await resp.json();
        const section = document.getElementById('preloaded-section');
        const list = document.getElementById('preloaded-list');

        if (data.experiments.length === 0) {
            section.style.display = 'none';
            return;
        }

        section.style.display = 'block';
        list.innerHTML = '';
        data.experiments.forEach(exp => {
            const parts = exp.filename.split('/');
            const viewFile = parts.pop();
            const dirName = parts.join('/');
            const div = document.createElement('div');
            div.className = 'upload-item';
            div.innerHTML = `
                <label class="checkbox-label">
                    <input type="checkbox" class="exp-checkbox" data-exp-id="${exp.exp_id}"
                           data-num-epochs="${exp.num_epochs}" data-width="${exp.width}"
                           data-height="${exp.height}" data-filename="${exp.filename}">
                </label>
                <div class="exp-info-text">
                    <span class="filename">${viewFile}</span>
                    <span class="info">${dirName}</span>
                    <span class="info">${exp.num_epochs} epochs, ${exp.width}x${exp.height}</span>
                </div>
            `;
            list.appendChild(div);
        });

        updateStartButton();
        list.addEventListener('change', updateStartButton);
    } catch (e) {
        // ignore
    }
}

function updateStartButton() {
    const checked = document.querySelectorAll('.exp-checkbox:checked');
    const btn = document.getElementById('start-selected-btn');
    btn.disabled = checked.length === 0;
    btn.textContent = checked.length === 0
        ? 'Select GIFs to compare'
        : `Continue with ${checked.length} GIF${checked.length > 1 ? 's' : ''}`;
}

async function uploadFile(file) {
    if (!file.name.toLowerCase().endsWith('.gif')) return;
    const formData = new FormData();
    formData.append('file', file);

    const resp = await fetch('/api/upload', { method: 'POST', body: formData });
    const data = await resp.json();
    if (data.error) { alert(data.error); return; }

    const list = document.getElementById('upload-list');
    const div = document.createElement('div');
    div.className = 'upload-item';
    div.innerHTML = `
        <label class="checkbox-label">
            <input type="checkbox" class="exp-checkbox" data-exp-id="${data.exp_id}"
                   data-num-epochs="${data.num_epochs}" data-width="${data.width}"
                   data-height="${data.height}" data-filename="${data.filename}" checked>
        </label>
        <div class="exp-info-text">
            <span class="filename">${data.filename}</span>
            <span class="info">${data.num_epochs} epochs, ${data.width}x${data.height}</span>
        </div>
    `;
    list.appendChild(div);
    list.addEventListener('change', updateStartButton);
    updateStartButton();
}

async function startWithSelected() {
    const checked = document.querySelectorAll('.exp-checkbox:checked');
    if (checked.length === 0) return;

    const selected = [];
    checked.forEach(cb => {
        selected.push({
            expId: cb.dataset.expId,
            filename: cb.dataset.filename,
            numEpochs: parseInt(cb.dataset.numEpochs),
            width: parseInt(cb.dataset.width),
            height: parseInt(cb.dataset.height),
        });
    });

    state.selected = selected;
    state.roi = null;

    // Pre-load all selected GIFs (triggers lazy loading on server)
    const btn = document.getElementById('start-selected-btn');
    btn.disabled = true;
    btn.textContent = 'Loading GIFs...';

    try {
        // Trigger lazy load by fetching the first frame of each
        const responses = await Promise.all(selected.map(s =>
            fetch(`/api/first_frame/${s.expId}`)
        ));
        for (const r of responses) {
            if (!r.ok) {
                alert(`Error loading GIF: ${r.status}`);
                btn.disabled = false;
                updateStartButton();
                return;
            }
        }
    } catch (e) {
        alert('Error loading GIFs: ' + e.message);
        btn.disabled = false;
        updateStartButton();
        return;
    }

    btn.disabled = false;
    updateStartButton();
    showPhase('roi');
    initROISelection();
}

// =========================================================================
// Phase 2: ROI Selection (cycles through each selected GIF)
// =========================================================================

function initROISelection() {
    const container = document.getElementById('roi-container');
    const baseImg = document.getElementById('roi-base-img');
    const overlayImg = document.getElementById('roi-overlay-img');
    const canvas = document.getElementById('roi-canvas');
    const ctx = canvas.getContext('2d');

    // Use first selected GIF for ROI drawing (same ROI applies to all)
    const current = state.selected[0];

    document.getElementById('roi-title').textContent = 'Select region of interest';
    document.getElementById('roi-info').textContent = '';
    document.getElementById('start-tournament-btn').disabled = true;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Restore previous ROI if re-entering
    if (state.roi) {
        document.getElementById('start-tournament-btn').disabled = false;
        const [x, y, w, h] = state.roi.split(',').map(Number);
        document.getElementById('roi-info').textContent = `ROI: ${w}x${h} at (${x}, ${y})`;
    }

    function syncCanvasSize() {
        canvas.width = baseImg.naturalWidth;
        canvas.height = baseImg.naturalHeight;
        canvas.style.width = baseImg.offsetWidth + 'px';
        canvas.style.height = baseImg.offsetHeight + 'px';
    }

    function onImageReady() {
        requestAnimationFrame(() => {
            syncCanvasSize();
            overlayImg.src = `/api/variability/${current.expId}`;
            if (state.roi) {
                const [x, y, w, h] = state.roi.split(',').map(Number);
                ctx.strokeStyle = '#4ecdc4';
                ctx.lineWidth = 3;
                ctx.setLineDash([8, 4]);
                ctx.strokeRect(x, y, w, h);
                ctx.fillStyle = 'rgba(78, 205, 196, 0.15)';
                ctx.fillRect(x, y, w, h);
            }
        });
    }

    // Load image — use a new Image() to guarantee onload fires
    const loader = new Image();
    loader.onload = () => {
        baseImg.src = loader.src;
        onImageReady();
    };
    loader.src = `/api/first_frame/${current.expId}`;

    window.onresize = syncCanvasSize;

    const toggle = document.getElementById('heatmap-toggle');
    toggle.checked = true;
    toggle.onchange = () => {
        overlayImg.style.display = toggle.checked ? 'block' : 'none';
    };

    const roInput = document.getElementById('ro-input');
    roInput.value = state.rO;
    roInput.onchange = () => { state.rO = parseFloat(roInput.value) || 0.5; };

    // ROI drawing
    let drawing = false;
    let startX, startY;

    function imgCoords(e) {
        const rect = baseImg.getBoundingClientRect();
        const scaleX = baseImg.naturalWidth / rect.width;
        const scaleY = baseImg.naturalHeight / rect.height;
        const x = Math.max(0, Math.min(baseImg.naturalWidth, (e.clientX - rect.left) * scaleX));
        const y = Math.max(0, Math.min(baseImg.naturalHeight, (e.clientY - rect.top) * scaleY));
        return { x, y };
    }

    if (state._roiMouseDown) {
        container.removeEventListener('mousedown', state._roiMouseDown);
        document.removeEventListener('mousemove', state._roiMouseMove);
        document.removeEventListener('mouseup', state._roiMouseUp);
    }

    state._roiMouseDown = e => {
        if (!container.contains(e.target)) return;
        e.preventDefault();
        const pos = imgCoords(e);
        startX = pos.x;
        startY = pos.y;
        drawing = true;
    };

    state._roiMouseMove = e => {
        if (!drawing) return;
        e.preventDefault();
        const pos = imgCoords(e);
        const x = Math.min(startX, pos.x);
        const y = Math.min(startY, pos.y);
        const w = Math.abs(pos.x - startX);
        const h = Math.abs(pos.y - startY);

        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.strokeStyle = '#4ecdc4';
        ctx.lineWidth = 3;
        ctx.setLineDash([8, 4]);
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = 'rgba(78, 205, 196, 0.15)';
        ctx.fillRect(x, y, w, h);
    };

    state._roiMouseUp = e => {
        if (!drawing) return;
        drawing = false;
        const pos = imgCoords(e);
        const x = Math.round(Math.min(startX, pos.x));
        const y = Math.round(Math.min(startY, pos.y));
        const w = Math.round(Math.abs(pos.x - startX));
        const h = Math.round(Math.abs(pos.y - startY));

        if (w > 10 && h > 10) {
            state.roi = `${x},${y},${w},${h}`;
            document.getElementById('roi-info').textContent = `ROI: ${w}x${h} at (${x}, ${y})`;
            document.getElementById('start-tournament-btn').disabled = false;
        }
    };

    container.addEventListener('mousedown', state._roiMouseDown);
    document.addEventListener('mousemove', state._roiMouseMove);
    document.addEventListener('mouseup', state._roiMouseUp);
}

async function startTournament() {
    const expsConfig = state.selected.map(s => ({ exp_id: s.expId }));

    const resp = await fetch('/api/tournament/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ experiments: expsConfig, roi: state.roi, r_o: state.rO }),
    });
    const data = await resp.json();

    if (data.error) { alert(data.error); return; }

    state.sessionId = data.session_id;
    state.currentPair = data.pair;  // {left: {exp_id, epoch, label, index}, right: {...}}
    showPhase('tournament');
    updateProgress(data.progress);
    buildComparisonImages();
    initTournamentUI();
    loadComparison();
}

// =========================================================================
// Phase 3: Tournament
// =========================================================================

function buildComparisonImages() {
    const container = document.getElementById('comparison-display');
    container.innerHTML = `
        <div id="epoch-label"></div>
        <div id="comparison-images">
            <div class="comp-view">
                <img id="comp-img" class="comparison-img" alt="Loading..." onclick="toggleZoom(this)">
            </div>
        </div>`;
    applyImgSize();
}

function applyImgSize() {
    const slider = document.getElementById('img-size-slider');
    if (!slider) return;
    const px = slider.value;
    document.querySelectorAll('.comparison-img').forEach(img => {
        img.style.width = px + 'px';
        img.style.maxWidth = px + 'px';
    });
}

function toggleZoom(img) {
    img.classList.toggle('zoomed');
}

function initTournamentUI() {
    state.showingSide = 'left';
    state.spaceCount = 0;
    document.removeEventListener('keydown', handleTournamentKey);
    document.addEventListener('keydown', handleTournamentKey);

    const slider = document.getElementById('img-size-slider');
    const sizeLabel = document.getElementById('img-size-label');
    slider.oninput = () => {
        const px = slider.value;
        sizeLabel.textContent = px + 'px';
        applyImgSize();
    };
}

function handleTournamentKey(e) {
    if (!document.getElementById('phase-tournament').classList.contains('active')) return;

    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        switchTo('left');
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        switchTo('right');
    } else if (e.key === ' ') {
        e.preventDefault();
        state.spaceCount++;
        if (state.spaceTimer) clearTimeout(state.spaceTimer);
        if (state.spaceCount >= 2) {
            state.spaceCount = 0;
            submitChoice(state.showingSide);
        } else {
            state.spaceTimer = setTimeout(() => { state.spaceCount = 0; }, 400);
        }
    } else if (e.key === 'Enter') {
        e.preventDefault();
        submitChoice('tie');
    } else if (e.ctrlKey && e.key === 'z') {
        e.preventDefault();
        undoChoice();
    }
}

function switchTo(side) {
    state.showingSide = side;
    if (!state.currentPair) return;

    const candidate = state.currentPair[side];
    if (!candidate) { console.error('No candidate for side', side, state.currentPair); return; }
    const img = document.getElementById('comp-img');
    const roiParam = state.roi ? `?roi=${state.roi}` : '';
    img.src = `/api/frame/${candidate.exp_id}/${candidate.epoch}${roiParam}`;
    img.className = 'comparison-img ' + (side === 'left' ? 'active-left' : 'active-right');
    applyImgSize();

    const label = document.getElementById('epoch-label');
    const ll = state.currentPair.left.label;
    const rl = state.currentPair.right.label;
    if (side === 'left') {
        label.innerHTML = `<span class="label-left">${ll}</span> <span style="color:#666">vs</span> <span style="color:#666">${rl}</span>`;
    } else {
        label.innerHTML = `<span style="color:#666">${ll}</span> <span style="color:#666">vs</span> <span class="label-right">${rl}</span>`;
    }

    highlightMetrics(side);
}

async function loadComparison() {
    if (!state.currentPair) return;
    const left = state.currentPair.left;
    const right = state.currentPair.right;
    if (!left || !right) { console.error('Invalid pair', state.currentPair); return; }

    // Show left image immediately (don't wait for metrics)
    switchTo('left');

    const roiParam = state.roi ? `&roi=${state.roi}` : '';
    try {
        const [respA, respB] = await Promise.all([
            fetch(`/api/metrics/${left.exp_id}/${left.epoch}?r_o=${state.rO}${roiParam}`).then(r => r.json()),
            fetch(`/api/metrics/${right.exp_id}/${right.epoch}?r_o=${state.rO}${roiParam}`).then(r => r.json()),
        ]);

        state.metricsA = respA.metrics;
        state.metricsB = respB.metrics;

        renderMetricsTable();
    } catch (e) {
        console.error('Error loading metrics', e);
    }
}

function renderMetricsTable() {
    const table = document.getElementById('metrics-table');
    const ll = state.currentPair.left.label;
    const rl = state.currentPair.right.label;
    const thead = `<tr><th>Metric</th><th class="label-left">${ll}</th><th class="label-right">${rl}</th></tr>`;

    let rows = '';
    for (let i = 0; i < state.metricsA.length; i++) {
        const a = state.metricsA[i];
        const b = state.metricsB[i];
        const hib = a.higher_is_better;
        const aWins = hib ? a.value > b.value : a.value < b.value;
        const bWins = hib ? b.value > a.value : b.value < a.value;

        const aClass = aWins ? 'metric-winner' : (bWins ? 'metric-loser' : '');
        const bClass = bWins ? 'metric-winner' : (aWins ? 'metric-loser' : '');
        const rowClass = a.name === 'NHWTSE' ? 'metric-highlight' : '';

        rows += `<tr class="${rowClass}">
            <td>${a.name}</td>
            <td class="${aClass}">${a.value.toFixed(4)}</td>
            <td class="${bClass}">${b.value.toFixed(4)}</td>
        </tr>`;
    }

    table.innerHTML = thead + rows;
}

function highlightMetrics(side) {
    const ths = document.querySelectorAll('#metrics-table th');
    if (ths.length >= 3) {
        ths[1].style.opacity = side === 'left' ? '1' : '0.5';
        ths[2].style.opacity = side === 'right' ? '1' : '0.5';
    }
}

async function submitChoice(winner) {
    const resp = await fetch(`/api/tournament/${state.sessionId}/choice`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ winner }),
    });
    const data = await resp.json();
    updateProgress(data.progress);

    if (data.done) {
        showResults(data.ranking, data.top3);
        return;
    }

    state.currentPair = data.pair;
    state.spaceCount = 0;
    loadComparison();
}

async function undoChoice() {
    const resp = await fetch(`/api/tournament/${state.sessionId}/undo`, { method: 'POST' });
    const data = await resp.json();
    if (!data.undone) return;

    updateProgress(data.progress);
    state.currentPair = data.pair;
    state.spaceCount = 0;
    loadComparison();
}

function updateProgress(progress) {
    const bar = document.getElementById('progress-bar');
    const text = document.getElementById('progress-text');
    const pct = Math.min(100, (progress.current / progress.total) * 100);
    bar.style.width = pct + '%';
    text.textContent = `Comparison ${progress.current + 1} of ~${progress.total}`;
}

// =========================================================================
// Phase 4: Results
// =========================================================================

async function showResults(ranking, top3) {
    showPhase('results');
    document.removeEventListener('keydown', handleTournamentKey);

    // ranking and top3 are now arrays of {exp_id, epoch, label} objects
    const podium = document.getElementById('results-podium');
    podium.innerHTML = '';
    const medals = ['1st', '2nd', '3rd'];
    for (let i = 0; i < top3.length; i++) {
        const div = document.createElement('div');
        div.className = 'podium-item';
        div.innerHTML = `<div class="rank">${medals[i]}</div><div class="epoch-num">${top3[i].full_label || top3[i].label}</div>`;
        podium.appendChild(div);
    }

    // Full ranking with metrics
    const table = document.getElementById('full-ranking-table');
    const roiParam = state.roi ? `&roi=${state.roi}` : '';

    const allMetrics = [];
    for (const c of ranking) {
        const resp = await fetch(`/api/metrics/${c.exp_id}/${c.epoch}?r_o=${state.rO}${roiParam}`);
        const data = await resp.json();
        allMetrics.push(data.metrics);
    }

    let headers = '<tr><th>Rank</th><th>Candidate</th>';
    const numMetrics = allMetrics.length > 0 ? allMetrics[0].length : 0;
    if (numMetrics > 0) {
        allMetrics[0].forEach(m => { headers += `<th>${m.name}</th>`; });
    }
    headers += '</tr>';

    const metricMins = [];
    const metricMaxs = [];
    for (let mi = 0; mi < numMetrics; mi++) {
        let min = Infinity, max = -Infinity;
        for (const em of allMetrics) {
            if (em[mi].value < min) min = em[mi].value;
            if (em[mi].value > max) max = em[mi].value;
        }
        metricMins.push(min);
        metricMaxs.push(max);
    }

    let rows = '';
    for (let i = 0; i < ranking.length; i++) {
        rows += `<tr><td>${i + 1}</td><td>${ranking[i].full_label || ranking[i].label}</td>`;
        allMetrics[i].forEach((m, mi) => {
            const range = metricMaxs[mi] - metricMins[mi];
            const t = range > 0 ? (m.value - metricMins[mi]) / range : 0.5;
            const color = metricColor(t);
            rows += `<td style="color:${color}">${m.value.toFixed(4)}</td>`;
        });
        rows += '</tr>';
    }

    table.innerHTML = headers + rows;

    state.lastResults = { ranking, top3, allMetrics, timestamp: new Date().toISOString() };
}

function exportResults() {
    if (!state.lastResults) return;
    const r = state.lastResults;
    const lines = [];
    lines.push(`Compare Game — Tournament Results`);
    lines.push(`Date: ${r.timestamp}`);
    lines.push(`GIFs: ${state.selected.map(s => s.filename).join(', ')}`);
    lines.push(`OTF radius: ${state.rO}`);
    lines.push(`ROI: ${state.roi || 'none'}`);
    lines.push('');
    lines.push(`Top 3: ${r.top3.map(c => c.full_label || c.label).join(', ')}`);
    lines.push('');

    // TSV table
    const metricNames = r.allMetrics[0].map(m => m.name);
    lines.push(['Rank', 'Candidate', ...metricNames].join('\t'));
    for (let i = 0; i < r.ranking.length; i++) {
        const vals = r.allMetrics[i].map(m => m.value.toFixed(6));
        lines.push([i + 1, r.ranking[i].full_label || r.ranking[i].label, ...vals].join('\t'));
    }

    const blob = new Blob([lines.join('\n')], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `compare_results_${r.timestamp.slice(0, 10)}.tsv`;
    a.click();
    URL.revokeObjectURL(url);
}

function rerunWithNewROI() {
    state.roiIndex = 0;
    state.roi = null;
    initROISelection();
    showPhase('roi');
}

function rerunSameROI() {
    startTournament();
}

// =========================================================================
// Init
// =========================================================================

document.addEventListener('DOMContentLoaded', () => {
    showPhase('upload');
    initUpload();
});
