import '/static/vendor/model-viewer.min.js';

const element = id => document.getElementById(id);
const state = {jobs: [], selected: null, modelJob: null, modelUrl: null, uploadUrl: null, timer: null, busy: false, limit: 50, selectionVersion: 0};
const labels = {queued: '대기 중', running: '생성 중', complete: '완료', failed: '실패', cancelled: '취소됨'};
const stages = {
  initializing: '모델 준비', preprocess_and_camera: '1 / 9 · 이미지 전처리 및 카메라 추정',
  sparse_structure: '2 / 9 · Sparse structure 생성', lr_conditioning_and_shape: '3 / 9 · 기본 형상 생성',
  coordinate_subdivision: '4 / 9 · 공간 좌표 확장', hr_conditioning_and_shape: '5 / 9 · 고해상도 형상 생성',
  shape_decode_and_mesh: '6 / 9 · 형상 디코딩 및 메시 정리', texture_sampling: '7 / 9 · 텍스처 생성',
  texture_decode_and_export: '8 / 9 · PBR 텍스처 베이크 및 출력', normal_correction: '9 / 9 · 노멀 교정', validating: '최종 GLB 검증'
};

function node(tag, text, className) {
  const result = document.createElement(tag);
  if (text !== undefined) result.textContent = text;
  if (className) result.className = className;
  return result;
}

async function api(path, options = {}) {
  const response = await fetch(`/api/${path}`, options);
  if (!response.ok) {
    const error = await response.json().catch(() => ({detail: `HTTP ${response.status}`}));
    throw new Error(typeof error.detail === 'string' ? error.detail : JSON.stringify(error.detail));
  }
  return response;
}

function duration(seconds) {
  if (seconds === null || seconds === undefined) return '—';
  const rounded = Math.max(0, Math.floor(seconds));
  return `${Math.floor(rounded / 60)}m ${String(rounded % 60).padStart(2, '0')}s`;
}

function renderJobs() {
  const list = element('job-list');
  if (!state.jobs.length) {
    list.replaceChildren(node('div', '아직 생성한 에셋이 없습니다.', 'history-empty'));
    return;
  }
  list.replaceChildren(...state.jobs.map(job => {
    const button = node('button', undefined, `job-row${job.id === state.selected ? ' selected' : ''}`);
    button.type = 'button';
    button.setAttribute('aria-pressed', String(job.id === state.selected));
    const info = node('div', undefined, 'job-info');
    info.append(node('strong', job.name), node('small', `${job.options.resolution} GRID · ${job.options.target_faces.toLocaleString()} TARGET · SEED ${job.options.seed}`));
    button.append(node('span', '◇', 'job-icon'), info,
      node('span', new Date(job.created_at * 1000).toLocaleTimeString('ko-KR', {hour: '2-digit', minute: '2-digit'}), 'job-date'),
      node('span', labels[job.status] || job.status, `badge ${job.status}`));
    button.addEventListener('click', () => selectJob(job.id));
    return button;
  }));
}

function clearModel() {
  element('viewer').removeAttribute('src');
  element('viewer').hidden = true;
  element('viewer-empty').hidden = false;
  if (state.modelUrl) URL.revokeObjectURL(state.modelUrl);
  state.modelUrl = null;
  state.modelJob = null;
}

async function loadModel(job) {
  const version = state.selectionVersion;
  state.modelJob = job.id;
  element('viewer-message').textContent = '실제 GLB와 PBR 텍스처를 불러오는 중…';
  try {
    const response = await api(`jobs/${job.id}/model`);
    const blob = await response.blob();
    if (version !== state.selectionVersion) return;
    clearModel();
    state.modelJob = job.id;
    state.modelUrl = URL.createObjectURL(blob);
    element('viewer').src = state.modelUrl;
    element('viewer').hidden = false;
    element('viewer-empty').hidden = true;
  } catch (error) {
    if (version === state.selectionVersion) {
      state.modelJob = null;
      element('viewer-message').textContent = `뷰어 오류: ${error.message}`;
    }
  }
}

async function renderSelection() {
  const job = state.jobs.find(item => item.id === state.selected);
  if (!job) return;
  element('asset-title').textContent = job.name;
  element('asset-status').textContent = labels[job.status] || job.status;
  element('asset-status').className = `badge ${job.status}`;
  element('metric-faces').textContent = job.summary?.triangles?.toLocaleString() || '—';
  element('metric-size').textContent = job.summary?.bytes ? `${(job.summary.bytes / 1024 ** 2).toFixed(2)} MiB` : '—';
  element('metric-normals').textContent = job.summary?.normal_correction ? '교정 완료' : '자동 교정';
  element('metric-time').textContent = duration(job.summary?.elapsed_seconds ?? (job.started_at ? (job.finished_at || Date.now() / 1000) - job.started_at : null));
  element('download-model').disabled = job.status !== 'complete';
  element('download-report').disabled = job.status !== 'complete';
  element('job-progress').hidden = job.status === 'complete';
  element('stage-label').textContent = job.error || stages[job.stage] || labels[job.status] || job.status;
  element('cancel-job').hidden = job.status !== 'queued';
  if (job.status === 'complete' && state.modelJob !== job.id) await loadModel(job);
  if (element('job-logs').closest('details').open) {
    try {
      const response = await api(`jobs/${job.id}/logs`);
      const logText = await response.text();
      if (state.selected === job.id) element('job-logs').textContent = logText;
    } catch (error) {
      element('job-logs').textContent = error.message;
    }
  }
}

async function selectJob(jobId) {
  state.selectionVersion += 1;
  state.selected = jobId;
  clearModel();
  element('viewer-message').textContent = '';
  element('job-logs').textContent = '로그를 불러오는 중…';
  renderJobs();
  await renderSelection();
}

async function refresh() {
  try {
    const [health, listing] = await Promise.all([api('health').then(response => response.json()), api(`jobs?limit=${state.limit}`).then(response => response.json())]);
    state.jobs = listing.jobs;
    const counts = listing.counts;
    element('connection-dot').className = health.worker_alive ? '' : 'offline';
    element('connection-text').textContent = health.worker_alive ? '워커 연결됨' : '워커 연결 대기';
    element('resolution-1024').disabled = !health.allow_1024;
    element('resolution-1024').textContent = health.allow_1024 ? '1024 · 고메모리' : '1024 · 서버 허용 필요';
    element('job-count').textContent = Object.values(counts).reduce((total, value) => total + value, 0);
    element('queue-summary').textContent = `${counts.running || 0}개 생성 중 · ${counts.queued || 0}개 대기`;
    element('load-more').hidden = state.jobs.length < state.limit || state.limit >= 100;
    renderJobs();
    if (!state.selected && state.jobs.length) await selectJob(state.jobs[0].id);
    else await renderSelection();
    if (!health.viewer_ready) element('viewer-message').textContent = '서버에서 npm run build --prefix web을 실행해 뷰어를 준비하세요.';
  } catch (error) {
    element('connection-dot').className = 'offline';
    element('connection-text').textContent = '연결 끊김 · 재시도 중';
    element('form-message').textContent = error.message;
  } finally {
    clearTimeout(state.timer);
    state.timer = setTimeout(refresh, 2500);
  }
}

element('generate-form').addEventListener('submit', async event => {
  event.preventDefault();
  if (state.busy) return;
  const image = element('image').files[0];
  if (!image) return;
  state.busy = true;
  element('submit-job').disabled = true;
  const message = element('form-message');
  message.classList.remove('success');
  message.textContent = '이미지 업로드 중…';
  try {
    const form = new FormData();
    form.append('image', image);
    form.append('options', JSON.stringify({seed: Number(element('seed').value), resolution: Number(element('resolution').value),
      target_faces: Number(element('target-faces').value), texture_size: Number(element('texture-size').value),
      max_faces: element('strict-cap').checked ? Number(element('max-faces').value) : 0,
      remove_background: element('remove-background').checked}));
    const response = await api('jobs', {method: 'POST', body: form});
    const job = await response.json();
    state.jobs.unshift(job);
    await selectJob(job.id);
    message.classList.add('success');
    message.textContent = '대기열에 추가했습니다. 창을 닫아도 생성은 계속됩니다.';
    clearTimeout(state.timer);
    await refresh();
  } catch (error) {
    message.textContent = error.message;
  } finally {
    state.busy = false;
    element('submit-job').disabled = false;
  }
});

function previewUpload() {
  const file = element('image').files[0];
  if (!file) return;
  if (file.size > 16 * 1024 ** 2) {
    element('image').value = '';
    element('form-message').textContent = '16 MiB 이하의 이미지를 선택하세요.';
    return;
  }
  if (state.uploadUrl) URL.revokeObjectURL(state.uploadUrl);
  state.uploadUrl = URL.createObjectURL(file);
  element('upload-preview').src = state.uploadUrl;
  element('upload-preview').hidden = false;
  element('upload-prompt').hidden = true;
  element('upload-name').hidden = false;
  element('upload-name').textContent = file.name;
}

element('image').addEventListener('change', previewUpload);
for (const type of ['dragenter', 'dragover']) element('dropzone').addEventListener(type, event => {event.preventDefault(); element('dropzone').classList.add('dragging');});
for (const type of ['dragleave', 'drop']) element('dropzone').addEventListener(type, event => {event.preventDefault(); element('dropzone').classList.remove('dragging');});
element('dropzone').addEventListener('drop', event => {
  if (!event.dataTransfer.files.length) return;
  const transfer = new DataTransfer();
  transfer.items.add(event.dataTransfer.files[0]);
  element('image').files = transfer.files;
  previewUpload();
});
element('strict-cap').addEventListener('change', () => {element('cap-field').hidden = !element('strict-cap').checked;});
element('random-seed').addEventListener('click', () => {element('seed').value = Math.min(crypto.getRandomValues(new Uint32Array(1))[0], 4294967292);});
element('rotate').addEventListener('click', () => {
  const rotate = !element('viewer').autoRotate;
  element('viewer').autoRotate = rotate;
  element('rotate').setAttribute('aria-pressed', String(rotate));
  element('rotate').textContent = rotate ? '자동 회전' : '회전 멈춤';
});
element('reset-camera').addEventListener('click', () => {element('viewer').cameraOrbit = '180deg 80deg 105%'; element('viewer').fieldOfView = 'auto';});
element('viewer').addEventListener('load', () => {element('viewer-message').textContent = '';});
element('viewer').addEventListener('error', () => {element('viewer-message').textContent = 'GLB 로딩에 실패했습니다. 다운로드 후 확인하세요.';});
element('job-logs').closest('details').addEventListener('toggle', () => {if (element('job-logs').closest('details').open) renderSelection();});
element('cancel-job').addEventListener('click', async () => {
  try {await api(`jobs/${state.selected}/cancel`, {method: 'POST'}); clearTimeout(state.timer); await refresh();}
  catch (error) {element('form-message').textContent = error.message;}
});
element('load-more').addEventListener('click', () => {state.limit = 100; clearTimeout(state.timer); refresh();});

async function download(kind) {
  const jobId = state.selected;
  try {
    const response = await api(`jobs/${jobId}/${kind}`);
    const url = URL.createObjectURL(await response.blob());
    const anchor = node('a');
    anchor.href = url;
    anchor.download = `pixal3d-${jobId}.${kind === 'model' ? 'glb' : 'json'}`;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    setTimeout(() => URL.revokeObjectURL(url), 60000);
  } catch (error) {element('form-message').textContent = error.message;}
}
element('download-model').addEventListener('click', () => download('model'));
element('download-report').addEventListener('click', () => download('report'));
refresh();
