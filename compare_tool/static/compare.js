// Compare Game — Frontend Logic

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}

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
    submitting: false,
    // Internal
    _roiMouseDown: null,
    _roiMouseMove: null,
    _roiMouseUp: null,
};

// =========================================================================
// Utilities
// =========================================================================

function formatDirName(dirPath) {
    // Parse naming convention:
    // Training tile: YY031A_HT_100nm_T000_0001_rec_
    // Finetuning:    YY031A_HT_100nm_T001_0001_rec__from_YY031A_HT_100nm_T000_0001_rec__140
    const name = dirPath.split('/').pop() || dirPath;

    const ftMatch = name.match(/(.+?)_(T\d+)(?:_\d+)?_rec__from_(.+?)_(T\d+)(?:_\d+)?_rec__(\d+)/);
    if (ftMatch) {
        const [, sample, tomo, srcTomo, , srcEpoch] = ftMatch;
        return `<span class="dir-tomo">${escapeHtml(tomo)}</span> <span class="dir-arrow">\u2190</span> <span class="dir-src">${escapeHtml(srcTomo)}@${escapeHtml(srcEpoch)}</span> <span class="dir-sample">${escapeHtml(sample)}</span>`;
    }

    const trainMatch = name.match(/(.+?)_(T\d+)(?:_\d+)?_rec_?$/);
    if (trainMatch) {
        const [, sample, tomo] = trainMatch;
        return `<span class="dir-tomo">${escapeHtml(tomo)}</span> <span class="dir-type">training</span> <span class="dir-sample">${escapeHtml(sample)}</span>`;
    }

    return escapeHtml(dirPath);
}

function shortExpLabel(filename) {
    // Extract a readable tile label from a full path like
    // 'ls3639/NM0029/finetune/NM0029_HT_100nm_T008_0001_rec__from_..._T000_0001_rec__140/all_epochs_view1_141-150.gif'
    // -> 'T008←T000@140 view1'
    const parts = filename.replace(/\\/g, '/').split('/');
    const gifName = parts[parts.length - 1] || filename;
    const dirName = parts.length >= 2 ? parts[parts.length - 2] : '';

    const vm = gifName.match(/view(\d+)/);
    const view = vm ? ` view${vm[1]}` : '';

    const ft = dirName.match(/_?(T\d+)(?:_\d+)?_rec__from_.+?_(T\d+)(?:_\d+)?_rec__(\d+)/);
    if (ft) return `${ft[1]}←${ft[2]}@${ft[3]}${view}`;

    const tr = dirName.match(/_?(T\d+)(?:_\d+)?_rec_?$/);
    if (tr) return `${tr[1]}${view}`;

    return gifName;
}

function formatIsoDate(iso) {
    // Turn '2026-04-14T18:58:41.123' into '2026/04/14 18:58'
    if (!iso) return '';
    const d = new Date(iso);
    if (isNaN(d)) return iso;
    const pad = n => String(n).padStart(2, '0');
    return `${d.getFullYear()}/${pad(d.getMonth() + 1)}/${pad(d.getDate())} ${pad(d.getHours())}:${pad(d.getMinutes())}`;
}

function metricColor(t) {
    // t in [0, 1]: 0 = worst (red), 1 = best (green)
    t = Math.max(0, Math.min(1, t));
    let r, g, b;
    if (t < 0.5) {
        const s = t * 2;
        r = 255;
        g = Math.round(100 + 155 * s);
        b = 68;
    } else {
        const s = (t - 0.5) * 2;
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
    loadPreloaded();
    loadPastResults();
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

        // Group experiments by directory path
        const tree = {};
        data.experiments.forEach(exp => {
            const parts = exp.filename.split('/');
            const viewFile = parts.pop();
            const dirPath = parts.join('/') || '.';
            if (!tree[dirPath]) tree[dirPath] = [];
            tree[dirPath].push({ ...exp, viewFile });
        });

        // Sort GIFs within each directory by filename
        for (const dir in tree) {
            tree[dir].sort((a, b) => a.viewFile.localeCompare(b.viewFile));
        }

        // Sort directories
        const sortedDirs = Object.keys(tree).sort();

        sortedDirs.forEach(dirPath => {
            const dirDiv = document.createElement('div');
            dirDiv.className = 'dir-group';

            // Directory header with toggle and select-all
            const header = document.createElement('div');
            header.className = 'dir-header';
            const dirId = 'dir-' + dirPath.replace(/[^a-zA-Z0-9]/g, '_');
            header.innerHTML = `
                <span class="dir-toggle" onclick="toggleDir('${escapeHtml(dirId)}')">&#9660;</span>
                <label class="checkbox-label">
                    <input type="checkbox" class="dir-checkbox" data-dir="${escapeHtml(dirId)}"
                        onchange="toggleDirCheckboxes('${escapeHtml(dirId)}', this.checked)">
                </label>
                <span class="dir-name" onclick="toggleDir('${escapeHtml(dirId)}')">${formatDirName(dirPath)}</span>
                <span class="dir-raw-path">${escapeHtml(dirPath)}</span>
            `;
            dirDiv.appendChild(header);

            // GIF list within directory
            const contents = document.createElement('div');
            contents.className = 'dir-contents';
            contents.id = dirId;

            tree[dirPath].forEach(exp => {
                const doneClass = exp.has_results ? ' upload-item-done' : '';
                const doneBy = exp.done_by || '';
                const doneBadge = exp.has_results
                    ? `<span class="done-badge">${doneBy ? 'done by ' + escapeHtml(doneBy) : 'done'}</span>`
                    : '';
                const div = document.createElement('div');
                div.className = 'upload-item' + doneClass;
                div.innerHTML = `
                    <label class="checkbox-label">
                        <input type="checkbox" class="exp-checkbox" data-exp-id="${escapeHtml(exp.exp_id)}"
                               data-num-epochs="${exp.num_epochs}" data-width="${exp.width}"
                               data-height="${exp.height}" data-filename="${escapeHtml(exp.filename)}"
                               data-dir="${escapeHtml(dirId)}">
                    </label>
                    <div class="exp-info-text">
                        <span class="filename">${escapeHtml(exp.viewFile)} ${doneBadge}</span>
                        <span class="info">${exp.num_epochs} epochs, ${exp.width}x${exp.height}</span>
                    </div>
                `;
                contents.appendChild(div);
            });

            // Auto-collapse folders where every experiment is done
            const allDone = tree[dirPath].some(exp => exp.has_results);
            if (allDone) {
                contents.classList.add('collapsed');
                header.querySelector('.dir-toggle').innerHTML = '&#9654;';
            }

            dirDiv.appendChild(contents);
            list.appendChild(dirDiv);
        });

        updateStartButton();
        list.addEventListener('change', (e) => {
            if (e.target.classList.contains('exp-checkbox')) {
                syncDirCheckbox(e.target.dataset.dir);
            }
            updateStartButton();
        });
    } catch (e) {
        // ignore
    }
}

function togglePastResults() {
    const list = document.getElementById('past-results-list');
    const toggle = document.getElementById('past-results-toggle');
    if (list.style.display === 'none') {
        list.style.display = 'block';
        toggle.innerHTML = '&#9660;';
        // Refresh on expand — results may have changed since page load
        loadPastResults();
    } else {
        list.style.display = 'none';
        toggle.innerHTML = '&#9654;';
    }
}

function toggleDir(dirId) {
    const contents = document.getElementById(dirId);
    const toggle = contents.parentElement.querySelector('.dir-toggle');
    if (contents.classList.contains('collapsed')) {
        contents.classList.remove('collapsed');
        toggle.innerHTML = '&#9660;';
    } else {
        contents.classList.add('collapsed');
        toggle.innerHTML = '&#9654;';
    }
}

function toggleDirCheckboxes(dirId, checked) {
    const contents = document.getElementById(dirId);
    contents.querySelectorAll('.exp-checkbox').forEach(cb => { cb.checked = checked; });
    updateStartButton();
}

function syncDirCheckbox(dirId) {
    if (!dirId) return;
    const contents = document.getElementById(dirId);
    const all = contents.querySelectorAll('.exp-checkbox');
    const checkedCount = contents.querySelectorAll('.exp-checkbox:checked').length;
    const dirCb = contents.parentElement.querySelector('.dir-checkbox');
    dirCb.checked = checkedCount === all.length;
    dirCb.indeterminate = checkedCount > 0 && checkedCount < all.length;
}

async function loadPastResults() {
    try {
        const resp = await fetch('/api/past-results');
        if (!resp.ok) return;
        const data = await resp.json();
        const section = document.getElementById('past-results-section');
        const list = document.getElementById('past-results-list');

        if (!data.results || data.results.length === 0) {
            section.style.display = 'none';
            return;
        }

        section.style.display = 'block';
        list.innerHTML = '';
        data.results.forEach(r => {
            const div = document.createElement('div');
            div.className = 'past-result-item';
            const dateStr = r.date ? formatIsoDate(r.date) : r.filename;
            // Pull tile info (e.g. T008←T000@140) from each GIF path
            const gifsStr = r.gifs.map(g => shortExpLabel(g)).filter((v, i, a) => a.indexOf(v) === i).join(', ');
            div.innerHTML = `
                <div class="past-result-info">
                    <span class="past-result-date">${escapeHtml(dateStr)}</span>
                    <span class="past-result-gifs">${escapeHtml(gifsStr)}</span>
                    <span class="past-result-top3">${escapeHtml(r.top3 || '')}</span>
                </div>
                <a href="/api/past-results/${escapeHtml(r.filename)}" class="btn btn-secondary" style="padding: 4px 12px; font-size: 0.8em;">Download</a>
            `;
            list.appendChild(div);
        });
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

async function startWithSelected() {
    const checked = document.querySelectorAll('.exp-checkbox:checked');
    if (checked.length === 0) return;

    const userName = document.getElementById('user-name').value.trim();
    if (!userName) {
        alert('Please enter your name before starting.');
        document.getElementById('user-name').focus();
        return;
    }
    state.userName = userName;

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
    showEpochConfig();
}

// =========================================================================
// Phase 1.5: Epoch Configuration
// =========================================================================

function generateLabels(numEpochs, start, end, step, rawFirst) {
    if (start === null || isNaN(start)) {
        return Array.from({length: numEpochs}, (_, i) => `Ep.${i + 1}`);
    }

    const effectiveEpochs = numEpochs - (rawFirst ? 1 : 0);
    if ((step === null || isNaN(step) || step < 1) && end !== null && !isNaN(end) && effectiveEpochs > 1) {
        step = Math.floor((end - start) / (effectiveEpochs - 1));
    }
    if (step === null || isNaN(step) || step < 1) step = 1;

    const labels = [];
    for (let i = 0; i < numEpochs; i++) {
        if (rawFirst && i === 0) {
            labels.push('RAW');
        } else {
            const epochIdx = i - (rawFirst ? 1 : 0);
            labels.push(`Ep.${start + epochIdx * step}`);
        }
    }
    return labels;
}

async function showEpochConfig() {
    showPhase('epoch-config');

    const resp = await fetch('/api/epoch-config');
    const data = await resp.json();

    const form = document.getElementById('epoch-config-form');

    // Build per-GIF config rows
    let html = '';
    const anyDetected = state.selected.some(s => {
        const c = data.experiments[s.expId]?.epoch_config || {};
        return c.source === 'cli' || c.source === 'filename';
    });
    if (anyDetected) {
        html += '<p class="center-row" style="color: #4ecdc4; margin-bottom: 12px;">Pre-filled from detected values</p>';
    }

    state.selected.forEach((s, i) => {
        const expConfig = data.experiments[s.expId]?.epoch_config || {};
        const gifName = s.filename.split('/').pop();
        const sourceText = expConfig.source || '';
        html += `
            <div class="epoch-config-card">
                <div class="epoch-config-gif-name">${escapeHtml(gifName)}</div>
                <div class="epoch-config-row">
                    <label>Start: <input type="number" class="ec-start" data-idx="${i}" value="${expConfig.start ?? ''}" placeholder="e.g. 100"></label>
                    <label>End: <input type="number" class="ec-end" data-idx="${i}" value="${expConfig.end ?? ''}" placeholder="optional"></label>
                    <label>Step: <input type="number" class="ec-step" data-idx="${i}" value="${expConfig.step ?? ''}" placeholder="1" min="1"></label>
                    <label class="checkbox-label" style="gap: 6px;">
                        <input type="checkbox" class="ec-raw" data-idx="${i}" ${expConfig.raw_first ? 'checked' : ''}>
                        RAW
                    </label>
                </div>
                <div class="preview-row ec-preview" data-idx="${i}"></div>
            </div>
        `;
    });
    html += '<div id="epoch-preview"></div>';
    form.innerHTML = html;

    const updatePreview = () => {
        state.selected.forEach((s, i) => {
            const start = getVal(`.ec-start[data-idx="${i}"]`);
            const end = getVal(`.ec-end[data-idx="${i}"]`);
            const step = getVal(`.ec-step[data-idx="${i}"]`);
            const rawFirst = document.querySelector(`.ec-raw[data-idx="${i}"]`).checked;
            const labels = generateLabels(s.numEpochs, start, end, step, rawFirst);
            const display = labels.length > 10
                ? labels.slice(0, 4).join(', ') + ', \u2026, ' + labels.slice(-2).join(', ')
                : labels.join(', ');
            document.querySelector(`.ec-preview[data-idx="${i}"]`).textContent = display;
        });
    };

    function getVal(selector) {
        const el = document.querySelector(selector);
        return el && el.value ? parseInt(el.value) : null;
    }

    form.addEventListener('input', updatePreview);
    form.addEventListener('change', updatePreview);
    updatePreview();
}

async function submitEpochConfig() {
    const config = {};
    let anyConfig = false;
    state.selected.forEach((s, i) => {
        const start = getVal(`.ec-start[data-idx="${i}"]`);
        const end = getVal(`.ec-end[data-idx="${i}"]`);
        const step = getVal(`.ec-step[data-idx="${i}"]`);
        const rawFirst = document.querySelector(`.ec-raw[data-idx="${i}"]`).checked;
        if (start !== null) {
            config[s.expId] = { start, end, step, raw_first: rawFirst };
            anyConfig = true;
        }
    });

    function getVal(selector) {
        const el = document.querySelector(selector);
        return el && el.value ? parseInt(el.value) : null;
    }

    if (anyConfig) {
        await fetch('/api/epoch-config', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ experiments: config }),
        });
    }

    showPhase('roi');
    initROISelection();
}

function skipEpochConfig() {
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

    // Clean up old resize listener
    if (state._resizeHandler) {
        window.removeEventListener('resize', state._resizeHandler);
    }

    // Load ROI suggestions
    loadROISuggestions(current.expId);

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

    state._resizeHandler = syncCanvasSize;
    window.addEventListener('resize', syncCanvasSize);

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

async function loadROISuggestions(expId) {
    const container = document.getElementById('roi-suggestions');
    try {
        const resp = await fetch(`/api/roi-suggestions/${expId}`);
        const data = await resp.json();
        if (!data.suggestions || data.suggestions.length === 0) {
            container.innerHTML = '';
            return;
        }
        let html = '<span style="color: #888; font-size: 0.9em;">Suggested ROIs: </span>';
        data.suggestions.forEach((roi, i) => {
            html += `<button class="btn btn-secondary" style="padding: 4px 10px; font-size: 0.8em; margin: 2px;"
                onclick="applyROISuggestion(${roi.x},${roi.y},${roi.w},${roi.h})">
                ${roi.w}x${roi.h} (score: ${roi.score.toFixed(2)})
            </button>`;
        });
        container.innerHTML = html;
    } catch (e) {
        container.innerHTML = '';
    }
}

function applyROISuggestion(x, y, w, h) {
    state.roi = `${x},${y},${w},${h}`;
    document.getElementById('roi-info').textContent = `ROI: ${w}x${h} at (${x}, ${y})`;
    document.getElementById('start-tournament-btn').disabled = false;

    // Draw the ROI on canvas
    const canvas = document.getElementById('roi-canvas');
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = '#4ecdc4';
    ctx.lineWidth = 3;
    ctx.setLineDash([8, 4]);
    ctx.strokeRect(x, y, w, h);
    ctx.fillStyle = 'rgba(78, 205, 196, 0.15)';
    ctx.fillRect(x, y, w, h);
}

// =========================================================================
// Phase 2.5: Pre-filter
// =========================================================================

let prefilterCandidates = [];  // sorted by score (best first)
let pfIdxLeft = 0;   // index into full list (left viewer)
let pfIdxRight = 0;  // index into full list (right viewer)
let pfImgSize = 500;

async function showPrefilter() {
    showPhase('prefilter');
    document.getElementById('pf-img-left').src = '';
    document.getElementById('pf-img-right').src = '';
    document.getElementById('pf-label-left').textContent = 'Computing metrics...';
    document.getElementById('pf-label-right').textContent = '';

    try {
        const expsConfig = state.selected.map(s => ({ exp_id: s.expId }));
        const resp = await fetch('/api/prefilter', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ experiments: expsConfig, roi: state.roi, r_o: state.rO }),
        });
        const data = await resp.json();
        if (!resp.ok || data.error) {
            document.getElementById('pf-label-left').innerHTML =
                `<span style="color:#e94560">Error: ${data.error || resp.status}. Use "Skip" to proceed.</span>`;
            return;
        }

        prefilterCandidates = data.candidates;
        // No pre-filled exclusions — user decides
        prefilterCandidates.forEach(c => { c._excluded = false; });

        pfIdxLeft = 0;
        pfIdxRight = prefilterCandidates.length - 1;  // start at worst
        renderPrefilterViewer();
    } catch (e) {
        document.getElementById('pf-label-left').innerHTML =
            `<span style="color:#e94560">Failed: ${e.message}. Use "Skip".</span>`;
    }
}

function pfRenderSide(side, idx) {
    const n = prefilterCandidates.length;
    if (n === 0) return;
    const c = prefilterCandidates[idx];
    const roiParam = state.roi ? `&roi=${state.roi}` : '';

    document.getElementById(`pf-img-${side}`).src =
        `/api/frame/${c.exp_id}/${c.epoch}?w=${pfImgSize}${roiParam}`;
    document.getElementById(`pf-label-${side}`).innerHTML =
        `<strong>${escapeHtml(c.label)}</strong> <span class="pf-score">#${idx + 1} / ${n} &bull; score: ${c.score.toFixed(2)}</span>`;
    document.getElementById(`pf-pos-${side}`).textContent = `${idx + 1} / ${n}`;

    // Highlight border if excluded
    const wrap = document.getElementById(`pf-img-${side}`).parentElement;
    wrap.classList.toggle('pf-excluded-img', !!c._excluded);
}

function renderPrefilterViewer() {
    pfRenderSide('left', pfIdxLeft);
    pfRenderSide('right', pfIdxRight);

    // Sync checkbox with right side's candidate
    const cb = document.getElementById('pf-exclude-cb');
    cb.checked = prefilterCandidates[pfIdxRight]._excluded;

    updatePrefilterCount();
}

function pfNav(side, delta) {
    const n = prefilterCandidates.length;
    if (side === 'left') {
        pfIdxLeft = Math.max(0, Math.min(n - 1, pfIdxLeft + delta));
    } else {
        pfIdxRight = Math.max(0, Math.min(n - 1, pfIdxRight + delta));
    }
    renderPrefilterViewer();
}

function pfExcludeToggle() {
    const cb = document.getElementById('pf-exclude-cb');
    prefilterCandidates[pfIdxRight]._excluded = cb.checked;
    renderPrefilterViewer();
}

function resizePrefilter(val) {
    pfImgSize = parseInt(val);
    document.getElementById('pf-size-label').textContent = val + 'px';
    renderPrefilterViewer();
}

function updatePrefilterCount() {
    const cut = prefilterCandidates.filter(c => c._excluded).length;
    const kept = prefilterCandidates.length - cut;
    document.getElementById('prefilter-keep-count').textContent = kept;
    document.getElementById('pf-cut-n').textContent = cut;
    document.getElementById('pf-total-n').textContent = prefilterCandidates.length;
}

function skipPrefilter() {
    // Clear exclusions and start tournament
    prefilterCandidates.forEach(c => { c._excluded = false; });
    startTournament();
}

function applyPrefilter() {
    startTournament();
}

async function startTournament() {
    if (state.submitting) return;
    state.submitting = true;
    try {
        const expsConfig = state.selected.map(s => ({ exp_id: s.expId }));

        // Build exclusion list from pre-filter
        const exclude = prefilterCandidates
            .filter(c => c._excluded)
            .map(c => ({ exp_id: c.exp_id, epoch: c.epoch }));

        const resp = await fetch('/api/tournament/start', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ experiments: expsConfig, roi: state.roi, r_o: state.rO, user_name: state.userName, exclude }),
        });
        if (!resp.ok) { alert('Server error: ' + resp.status); return; }
        const data = await resp.json();

        if (data.error) { alert(data.error); return; }

        state.sessionId = data.session_id;
        state.currentPair = data.pair;

        // Handle immediate completion (e.g., 1 candidate after RAW exclusion)
        if (data.done) {
            showResults(data.ranking, data.top3, data.all_metrics, data.save_path, data.confidence);
            return;
        }

        showPhase('tournament');
        updateProgress(data.progress);
        buildComparisonImages();
        initTournamentUI();
        loadComparison();
    } catch (e) {
        alert('Network error: ' + e.message);
    } finally {
        state.submitting = false;
    }
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
    if (state.submitting) return;

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
    } else if (e.key === 'Escape') {
        e.preventDefault();
        finishEarly();
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
    const ll = escapeHtml(state.currentPair.left.label);
    const rl = escapeHtml(state.currentPair.right.label);
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

function metricRow(a, b) {
    const hib = a.higher_is_better;
    const aWins = hib ? a.value > b.value : a.value < b.value;
    const bWins = hib ? b.value > a.value : b.value < a.value;
    const maxVal = Math.max(Math.abs(a.value), Math.abs(b.value), 1e-10);
    const relDiff = Math.abs(a.value - b.value) / maxVal;
    const intensity = Math.min(1, Math.sqrt(relDiff * 5));
    const winColor = `rgba(78, 255, 78, ${0.15 + intensity * 0.85})`;
    const loseColor = `rgba(136, 136, 136, ${0.4 + (1 - intensity) * 0.6})`;
    const aStyle = aWins ? `color:${winColor};font-weight:700` : (bWins ? `color:${loseColor}` : '');
    const bStyle = bWins ? `color:${winColor};font-weight:700` : (aWins ? `color:${loseColor}` : '');
    return { aStyle, bStyle, aWins, bWins };
}

function renderMetricsTable() {
    const table = document.getElementById('metrics-table');
    const ll = escapeHtml(state.currentPair.left.label);
    const rl = escapeHtml(state.currentPair.right.label);
    const thead = `<tr><th>Metric</th><th class="label-left">${ll}</th><th class="label-right">${rl}</th></tr>`;

    // Group metrics by cluster
    const grouped = {};   // group name -> [{a, b, index}]
    const standalone = []; // [{a, b, index}]
    for (let i = 0; i < state.metricsA.length; i++) {
        const a = state.metricsA[i];
        const b = state.metricsB[i];
        if (a.group) {
            if (!grouped[a.group]) grouped[a.group] = [];
            grouped[a.group].push({ a, b, i });
        } else {
            standalone.push({ a, b, i });
        }
    }

    let rows = '';

    // Render grouped metrics
    for (const [groupName, members] of Object.entries(grouped)) {
        const rep = members.find(m => m.a.representative) || members[0];
        const others = members.filter(m => m !== rep);
        const groupId = 'mg-' + groupName.replace(/[^a-zA-Z0-9]/g, '_');

        // Count how many metrics agree on which side wins
        let leftWins = 0, rightWins = 0;
        members.forEach(m => {
            const { aWins, bWins } = metricRow(m.a, m.b);
            if (aWins) leftWins++;
            if (bWins) rightWins++;
        });
        const consensus = leftWins > rightWins ? 'left' : (rightWins > leftWins ? 'right' : 'tie');
        const consensusLabel = consensus === 'tie' ? 'tie' : `${Math.max(leftWins, rightWins)}/${members.length}`;

        // Group header (clickable to expand/collapse)
        rows += `<tr class="metric-group-header" onclick="toggleMetricGroup('${groupId}')">
            <td><span class="mg-toggle" id="${groupId}-toggle">&#9654;</span> ${escapeHtml(groupName)}</td>
            <td colspan="2" style="text-align:center;color:#666;font-size:0.85em">${consensusLabel} agree</td>
        </tr>`;

        // Representative metric (always visible)
        const r = metricRow(rep.a, rep.b);
        rows += `<tr>
            <td>${escapeHtml(rep.a.name)}</td>
            <td style="${r.aStyle}">${rep.a.value.toFixed(4)}</td>
            <td style="${r.bStyle}">${rep.b.value.toFixed(4)}</td>
        </tr>`;

        // Other members (hidden by default)
        others.forEach(m => {
            const r2 = metricRow(m.a, m.b);
            rows += `<tr class="metric-group-member" data-group="${groupId}">
                <td>${escapeHtml(m.a.name)}</td>
                <td style="${r2.aStyle}">${m.a.value.toFixed(4)}</td>
                <td style="${r2.bStyle}">${m.b.value.toFixed(4)}</td>
            </tr>`;
        });
    }

    // Render standalone metrics
    standalone.forEach(({ a, b }) => {
        const r = metricRow(a, b);
        rows += `<tr>
            <td>${escapeHtml(a.name)}</td>
            <td style="${r.aStyle}">${a.value.toFixed(4)}</td>
            <td style="${r.bStyle}">${b.value.toFixed(4)}</td>
        </tr>`;
    });

    table.innerHTML = thead + rows;
}

function toggleMetricGroup(groupId) {
    const members = document.querySelectorAll(`[data-group="${groupId}"]`);
    const toggle = document.getElementById(groupId + '-toggle');
    const expanded = members[0] && members[0].classList.contains('expanded');
    members.forEach(m => m.classList.toggle('expanded'));
    toggle.innerHTML = expanded ? '&#9654;' : '&#9660;';
}

function highlightMetrics(side) {
    const table = document.getElementById('metrics-table');
    table.classList.remove('highlight-left', 'highlight-right');
    if (side) table.classList.add('highlight-' + side);
}

async function submitChoice(winner) {
    if (state.submitting) return;
    state.submitting = true;
    try {
        const resp = await fetch(`/api/tournament/${state.sessionId}/choice`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ winner }),
        });
        if (!resp.ok) { alert('Server error: ' + resp.status); return; }
        const data = await resp.json();
        updateProgress(data.progress);

        // Show tie explanation if present
        const tieEl = document.getElementById('tie-explanation');
        if (data.tie_explanation) {
            tieEl.textContent = data.tie_explanation;
            tieEl.style.display = 'block';
            setTimeout(() => { tieEl.style.display = 'none'; }, 3000);
        } else {
            tieEl.style.display = 'none';
        }

        if (data.done) {
            showResults(data.ranking, data.top3, data.all_metrics, data.save_path, data.confidence);
            return;
        }

        state.currentPair = data.pair;
        state.spaceCount = 0;
        loadComparison();
    } catch (e) {
        alert('Network error: ' + e.message);
    } finally {
        state.submitting = false;
    }
}

async function finishEarly() {
    if (state.submitting) return;
    if (!state.sessionId) return;
    if (!confirm('Finish the tournament early with current rankings?')) return;

    state.submitting = true;
    try {
        const resp = await fetch(`/api/tournament/${state.sessionId}/finish`, { method: 'POST' });
        if (!resp.ok) { alert('Server error: ' + resp.status); return; }
        const data = await resp.json();
        if (data.done) {
            showResults(data.ranking, data.top3, data.all_metrics, data.save_path, data.confidence);
        }
    } catch (e) {
        alert('Network error: ' + e.message);
    } finally {
        state.submitting = false;
    }
}

async function undoChoice() {
    if (state.submitting) return;
    state.submitting = true;
    try {
        const resp = await fetch(`/api/tournament/${state.sessionId}/undo`, { method: 'POST' });
        if (!resp.ok) { alert('Server error: ' + resp.status); return; }
        const data = await resp.json();
        if (!data.undone) return;

        updateProgress(data.progress);
        state.currentPair = data.pair;
        state.spaceCount = 0;
        loadComparison();
    } catch (e) {
        alert('Network error: ' + e.message);
    } finally {
        state.submitting = false;
    }
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

async function showResults(ranking, top3, serverMetrics, savePath, confidence) {
    showPhase('results');
    document.removeEventListener('keydown', handleTournamentKey);

    const podium = document.getElementById('results-podium');
    podium.innerHTML = '';
    const medals = ['1st', '2nd', '3rd'];
    for (let i = 0; i < top3.length; i++) {
        const div = document.createElement('div');
        div.className = 'podium-item';
        div.innerHTML = `<div class="rank">${medals[i]}</div><div class="epoch-num">${escapeHtml(top3[i].label)}</div>`;
        podium.appendChild(div);
    }

    // Use server-provided metrics or fetch them
    let allMetrics = serverMetrics;
    if (!allMetrics) {
        allMetrics = [];
        const roiParam = state.roi ? `&roi=${state.roi}` : '';
        for (const c of ranking) {
            const resp = await fetch(`/api/metrics/${c.exp_id}/${c.epoch}?r_o=${state.rO}${roiParam}`);
            const data = await resp.json();
            allMetrics.push(data.metrics);
        }
    }

    const table = document.getElementById('full-ranking-table');

    let headers = '<tr><th>Rank</th><th>Candidate</th>';
    const numMetrics = allMetrics.length > 0 ? allMetrics[0].length : 0;
    if (numMetrics > 0) {
        allMetrics[0].forEach(m => { headers += `<th>${escapeHtml(m.name)}</th>`; });
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
        rows += `<tr><td>${i + 1}</td><td>${escapeHtml(ranking[i].label)}</td>`;
        allMetrics[i].forEach((m, mi) => {
            const range = metricMaxs[mi] - metricMins[mi];
            const t = range > 0 ? (m.value - metricMins[mi]) / range : 0.5;
            const color = metricColor(t);
            rows += `<td style="color:${color}">${m.value.toFixed(4)}</td>`;
        });
        rows += '</tr>';
    }

    table.innerHTML = headers + rows;

    // Show save notification
    const existingNote = document.querySelector('.save-notification');
    if (existingNote) existingNote.remove();
    if (savePath) {
        const saveNote = document.createElement('p');
        saveNote.className = 'save-notification';
        saveNote.textContent = `Results auto-saved to ${savePath}`;
        const resultsSection = document.getElementById('results-section');
        resultsSection.insertBefore(saveNote, resultsSection.children[1]);
    }

    state.lastResults = { ranking, top3, allMetrics, timestamp: new Date().toISOString() };
}

function exportResults() {
    if (!state.lastResults) return;
    const r = state.lastResults;
    const lines = [];
    lines.push(`Compare Game \u2014 Tournament Results`);
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
    setTimeout(() => URL.revokeObjectURL(url), 1000);
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
