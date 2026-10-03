const state = { videos: [], analyses: [], page: 1, pageSize: 10 };
const $ = (id) => document.getElementById(id);

async function loadData() {
  const [videosResponse, analysesResponse] = await Promise.all([fetch('/api/videos'), fetch('/api/analyses')]);
  state.videos = await videosResponse.json();
  state.analyses = await analysesResponse.json();
  renderAll();
}

function analysisFor(video) {
  return state.analyses.find((item) => item.video === video.name && item.category === video.category);
}

function displayStatus(status) {
  return status === 'EM_ANALISE' ? 'EM ANÁLISE' : status;
}

function formatDateTime(value) {
  if (!value || value === '--') return '--';
  const match = value.match(/^(\d{4})-(\d{2})-(\d{2})\s+(\d{2}:\d{2})/);
  return match ? `${match[3]}/${match[2]}/${match[1]} ${match[4]}` : value;
}

function renderAll() {
  $('total-videos').textContent = state.videos.length;
  $('analyzed-videos').textContent = state.analyses.filter((item) => item.status === 'ANALISADO').length;
  const alerts = state.analyses.filter((item) => item.status === 'ANALISADO' && item.classification === 'FURTO');
  $('total-alerts').textContent = alerts.length;
  const average = state.analyses.length ? state.analyses.reduce((sum, item) => sum + Number(item.risk || 0), 0) / state.analyses.length : 0;
  $('avg-risk').textContent = `${average.toFixed(1)}%`;
  $('alert-count').textContent = alerts.length;
  renderVideos();
  renderAnalyses();
  renderAlerts(alerts);
}

function renderVideos() {
  const totalPages = Math.max(1, Math.ceil(state.videos.length / state.pageSize));
  state.page = Math.min(state.page, totalPages);
  const start = (state.page - 1) * state.pageSize;
  const videos = state.videos.slice(start, start + state.pageSize);
  $('video-list').innerHTML = videos.map((video) => {
    const analysis = analysisFor(video);
    const status = analysis ? analysis.status : 'PENDENTE';
    const analyzed = status === 'ANALISADO';
    const result = analyzed ? (analysis.classification === 'FURTO' ? 'SUSPEITA' : 'SEM SUSPEITA') : '--';
    const date = analyzed ? formatDateTime(analysis.updated_at) : '--';
    const user = analyzed ? (analysis.analyzed_by || '--') : '--';
    const action = analyzed || status === 'EM_ANALISE' ? '' : `<button class="analyze-button" data-id="${encodeURIComponent(video.id)}" title="Analisar vídeo">Analisar</button>`;
    const statusClass = status === 'ANALISADO' ? 'done' : status === 'EM_ANALISE' ? 'running' : 'pending';
    return `<tr class="video-row" data-id="${encodeURIComponent(video.id)}"><td class="video-play-cell"><span class="video-play-icon">▶</span></td><td><strong>${video.name}</strong></td><td><span class="analysis-flag ${statusClass}">${displayStatus(status)}</span></td><td>${result}</td><td>${date}</td><td>${user}</td><td>${action}</td></tr>`;
  }).join('') || '<tr><td colspan="7" class="muted">Nenhum vídeo encontrado.</td></tr>';
  $('page-number').textContent = state.page;
  $('page-summary').textContent = `${state.videos.length ? start + 1 : 0}-${Math.min(start + state.pageSize, state.videos.length)} de ${state.videos.length}`;
  $('previous-page').disabled = state.page <= 1;
  $('next-page').disabled = state.page >= totalPages;
  $('first-page').disabled = state.page <= 1;
  $('last-page').disabled = state.page >= totalPages;
  document.querySelectorAll('.video-row').forEach((row) => row.addEventListener('click', (event) => { if (!event.target.closest('.analyze-button')) playVideo(decodeURIComponent(row.dataset.id)); }));
  document.querySelectorAll('.analyze-button').forEach((button) => button.addEventListener('click', (event) => { event.stopPropagation(); analyzeVideo(decodeURIComponent(button.dataset.id), button); }));
}

function focusTableHeader() {
  $('video-table-header').scrollIntoView({ behavior: 'smooth', block: 'start' });
  $('video-table-header').focus({ preventScroll: true });
}

async function analyzeVideo(videoId, button) {
  playVideo(videoId);
  button.disabled = true;
  button.textContent = 'Analisando...';
  const response = await fetch('/api/analyze', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({video_id: videoId}) });
  if (!response.ok) { button.disabled = false; button.textContent = 'Analisar'; }
}

function renderAnalyses() {
  $('analysis-table').innerHTML = state.analyses.length
    ? state.analyses.map((item) => `<tr><td>${item.video}</td><td><span class="badge ${item.classification === 'FURTO' ? 'danger' : 'safe'}">${item.classification}</span></td><td>${Number(item.risk || 0).toFixed(1)}%</td><td>${(Number(item.suspicious_frames || 0) * 100).toFixed(1)}%</td><td>${displayStatus(item.status)}</td></tr>`).join('')
    : '<tr><td colspan="5" class="empty-table-message">Nenhuma análise registrada.</td></tr>';
}

function renderAlerts(alerts) {
  $('alerts-list').innerHTML = alerts.map((item) => `<button class="alert-item" data-video="${item.video}" data-category="${item.category}"><span class="alert-symbol">!</span><span><strong>Suspeita detectada</strong><small>${item.video} · risco ${Number(item.risk || 0).toFixed(1)}%</small></span><span class="alert-time">${item.updated_at.slice(11)}</span></button>`).join('') || '<p class="muted">Nenhuma ocorrência crítica.</p>';
  document.querySelectorAll('.alert-item[data-video]').forEach((alert) => alert.addEventListener('click', () => {
    const video = state.videos.find((item) => item.name === alert.dataset.video && item.category === alert.dataset.category);
    if (video) playVideo(video.id);
  }));
}

function playVideo(videoId) {
  const video = state.videos.find((item) => item.id === videoId);
  if (!video) return;
  state.currentVideo = video;
  $('custom-controls').hidden = false;
  $('player-status-row').hidden = false;
  $('source-controls-row').hidden = false;
  showVideoLoading('Carregando vídeo...');
  $('stage-empty').hidden = true;
  setVideoSource();
  $('video-player').load();
  $('video-player').play().catch(() => {});
  $('player-title').textContent = video.name;
  $('player-category').textContent = video.category.toUpperCase();
  updateAnalysisFlags(video);
  $('source-status').textContent = $('annotated-toggle').checked ? 'COM MARCAÇÕES' : 'ORIGINAL';
  $('player-panel').scrollIntoView({ behavior: 'smooth', block: 'start' });
  $('video-player').focus({ preventScroll: true });
}

function updateAnalysisFlags(video) {
  const processing = $('processing-status');
  const classification = $('classification-status');
  const analyzeButton = $('analyze-current-button');
  const analysis = analysisFor(video);
  classification.hidden = true;
  analyzeButton.hidden = true;
  processing.className = 'analysis-flag';
  if (!analysis) { processing.textContent = 'PENDENTE'; processing.classList.add('pending'); analyzeButton.hidden = false; return; }
  if (analysis.status === 'EM_ANALISE') { processing.textContent = 'EM ANÁLISE'; processing.classList.add('running'); return; }
  if (analysis.status === 'ANALISADO') {
    processing.textContent = 'ANALISADO'; processing.classList.add('done'); classification.hidden = false;
    classification.textContent = analysis.classification === 'FURTO' ? 'SUSPEITA' : 'SEM SUSPEITA';
    classification.className = `analysis-flag ${analysis.classification === 'FURTO' ? 'suspicious' : 'safe'}`;
    return;
  }
  processing.textContent = analysis.status || 'ERRO'; processing.classList.add('pending');
}

async function analyzeCurrentVideo() {
  if (!state.currentVideo) return;
  const button = $('analyze-current-button');
  button.disabled = true;
  button.innerHTML = '<span class="play-icon">▶</span><span>Enviando...</span>';
  const response = await fetch('/api/analyze', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({video_id: state.currentVideo.id}) });
  button.disabled = false; button.innerHTML = '<span class="play-icon">▶</span><span>Analisar Vídeo</span>';  
}

function setVideoSource() {
  if (!state.currentVideo) return;
  const source = $('annotated-toggle').checked ? state.currentVideo.annotated_url : state.currentVideo.url;
  $('video-player').src = `${source}?v=${Date.now()}`;
}

function toggleAnnotatedVideo() {
  if (!state.currentVideo) return;
  const wasPlaying = !$('video-player').paused;
  const position = $('video-player').currentTime;
  showVideoLoading($('annotated-toggle').checked ? 'Gerando marcações...' : 'Carregando vídeo original...');
  setVideoSource();
  $('video-player').load();
  $('source-status').textContent = $('annotated-toggle').checked ? 'GERANDO MARCAÇÕES' : 'ORIGINAL';
  $('video-player').addEventListener('loadedmetadata', () => { $('video-player').currentTime = position; $('source-status').textContent = $('annotated-toggle').checked ? 'MARCAÇÕES PRONTAS' : 'ORIGINAL'; if (wasPlaying) $('video-player').play().catch(() => {}); }, { once: true });
}

function showVideoLoading(message) { $('video-loading').querySelector('span:last-child').textContent = message; $('video-loading').hidden = false; }
function hideVideoLoading() { $('video-loading').hidden = true; }

$('video-player').addEventListener('error', () => { if (!$('video-player').error) return; hideVideoLoading(); $('stage-empty').hidden = false; $('stage-empty').textContent = 'Não foi possível reproduzir este arquivo neste navegador.'; $('source-status').textContent = 'ERRO DE LEITURA'; });
$('video-player').addEventListener('loadedmetadata', hideVideoLoading);
$('video-player').addEventListener('canplay', hideVideoLoading);
$('first-page').addEventListener('click', () => { state.page = 1; renderVideos(); focusTableHeader(); });
$('previous-page').addEventListener('click', () => { if (state.page > 1) { state.page--; renderVideos(); focusTableHeader(); } });
$('next-page').addEventListener('click', () => { if (state.page < Math.ceil(state.videos.length / state.pageSize)) { state.page++; renderVideos(); focusTableHeader(); } });
$('last-page').addEventListener('click', () => { state.page = Math.max(1, Math.ceil(state.videos.length / state.pageSize)); renderVideos(); focusTableHeader(); });
$('page-size').addEventListener('change', (event) => { state.pageSize = Number(event.target.value); state.page = 1; renderVideos(); focusTableHeader(); });
$('annotated-toggle').addEventListener('change', toggleAnnotatedVideo);
$('analyze-current-button').addEventListener('click', analyzeCurrentVideo);
$('view-mode-toggle').addEventListener('click', () => { const grid = $('content-grid'); const theater = grid.classList.toggle('theater-mode'); grid.classList.toggle('normal-mode', !theater); const button = $('view-mode-toggle'); button.title = theater ? 'Voltar ao modo normal' : 'Entrar no modo teatro'; button.setAttribute('aria-label', button.title); });
$('play-toggle').addEventListener('click', () => { const video = $('video-player'); if (video.paused) video.play().catch(() => {}); else video.pause(); });
$('video-player').addEventListener('play', () => { $('play-toggle').textContent = '❚❚'; });
$('video-player').addEventListener('pause', () => { $('play-toggle').textContent = '▶'; });
$('video-player').addEventListener('loadedmetadata', () => { $('timeline').max = Number.isFinite($('video-player').duration) ? $('video-player').duration : 100; $('timeline').value = 0; });
$('video-player').addEventListener('timeupdate', () => { const video = $('video-player'); $('timeline').value = Math.min(video.currentTime, Number($('timeline').max)); const format = (seconds) => `${String(Math.floor(seconds / 60)).padStart(2, '0')}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`; $('time-label').textContent = `${format(video.currentTime)} / ${format(video.duration || 0)}`; });
$('video-player').addEventListener('ended', () => { $('timeline').value = $('timeline').max; $('play-toggle').textContent = '▶'; });
$('timeline').addEventListener('input', (event) => { $('video-player').currentTime = Number(event.target.value); });

const socket = io();
socket.on('connect', () => { $('socket-status').textContent = 'WebSocket conectado'; });
socket.on('disconnect', () => { $('socket-status').textContent = 'WebSocket desconectado'; });
socket.on('analysis_update', (analysis) => { const index = state.analyses.findIndex((item) => item.id === analysis.id); if (index >= 0) state.analyses[index] = analysis; else state.analyses.unshift(analysis); if (state.currentVideo && state.currentVideo.name === analysis.video && state.currentVideo.category === analysis.category) updateAnalysisFlags(state.currentVideo); renderAll(); });
loadData();
