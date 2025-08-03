from __future__ import annotations
import io, os, uuid, json, hashlib
from datetime import datetime
from zipfile import ZipFile, ZIP_DEFLATED

from flask import Flask, request, jsonify, send_file, render_template_string, url_for
import av
from PIL import Image
import pandas as pd
from pymediainfo import MediaInfo
import cv2
import numpy as np

# ----------------------------- paths & config --------------------------------------
APP_DIR    = os.path.abspath(os.path.dirname(__file__))
DATA_DIR   = os.path.join(APP_DIR, "data")
FRAMES_DIR = os.path.join(DATA_DIR, "frames")
VIDEOS_DIR = os.path.join(DATA_DIR, "videos")
CATEGORIES_FILE = os.path.join(DATA_DIR, 'categories.json')
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(VIDEOS_DIR, exist_ok=True)

# ----------------------------- State Management (In-Memory / JSON) -------------------
VIDEOS_SESSIONS = {}
FRAMES_BY_VIDEO = {}

def get_default_category():
    return {"id": "default", "name": "Não categorizado", "color": "#6b7280"}

def load_categories_from_file():
    try:
        with open(CATEGORIES_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        cats = [get_default_category()]
        save_categories_to_file(cats)
        return cats

def save_categories_to_file(categories_list):
    with open(CATEGORIES_FILE, 'w', encoding='utf-8') as f:
        json.dump(categories_list, f, indent=2, ensure_ascii=False)

# ----------------------------- App Setup -----------------------------------
app = Flask(__name__)

# ------------------------ Image Processing Pipeline --------------------------
def apply_filter_pipeline(image_bytes, filters_array):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if img is None: return None

    for f in filters_array:
        if not f.get('enabled'): continue
        
        name = f.get('name')
        if name == 'brightness_contrast':
            alpha = 1.0 + (f.get('contrast', 0) / 100.0) # Contrast
            beta = f.get('brightness', 0) # Brightness
            img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        
        elif name == 'white_balance':
            # Simple Gray World algorithm
            result = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            avg_a = np.average(result[:, :, 1])
            avg_b = np.average(result[:, :, 2])
            result[:, :, 1] = result[:, :, 1] - ((avg_a - 128) * (result[:, :, 0] / 255.0) * 1.1)
            result[:, :, 2] = result[:, :, 2] - ((avg_b - 128) * (result[:, :, 0] / 255.0) * 1.1)
            img = cv2.cvtColor(result, cv2.COLOR_LAB2BGR)

        elif name == 'clahe':
            clip_limit = f.get('clipLimit', 2.0)
            grid_size = f.get('gridSize', 8)
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(grid_size,grid_size))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            
    _, img_encoded = cv2.imencode('.png', img)
    return img_encoded.tobytes()

# ------------------------ Helpers -------------------------------------
def extract_exact_frame(video_path: str, ts: float, out_path: str) -> None:
    try:
        container = av.open(video_path)
        vstream    = container.streams.video[0]
        pts = int(ts / vstream.time_base)
        container.seek(pts, any_frame=False, stream=vstream, backward=True)
        for frm in container.decode(vstream):
            if frm.pts >= pts:
                img: Image.Image = frm.to_image()
                os.makedirs(os.path.dirname(out_path), exist_ok=True)
                img.save(out_path)
                break
    finally:
        if 'container' in locals() and container:
            container.close()

# ----------------------------- Front-end Templates ------------------------
HTML = """
<!doctype html><html lang=pt><head><meta charset=utf-8>
<title>Bardin Video Annotator v8.7</title>
<script src=https://cdn.tailwindcss.com></script>
<script src="https://cdn.jsdelivr.net/npm/sortablejs@latest/Sortable.min.js"></script>
<style>
  #mainLayout { display: grid; grid-template-rows: 1fr auto; }
  #mainPanels { display: grid; grid-template-columns: 320px 1fr 320px; transition: grid-template-columns 0.3s ease; }
  .panel.collapsed .panel-header-title { display: none; }
  .panel.collapsed .panel-header { justify-content: center; border-bottom: none; }
  .filter-item { cursor: grab; }
  .filter-item:active { cursor: grabbing; }
</style>
</head>
<body class="bg-gray-200 flex flex-col h-screen overflow-hidden">
<header class="bg-indigo-700 text-white p-4 text-xl font-semibold flex-shrink-0 z-10">Bardin Video Annotator</header>
<div id="mainLayout" class="flex-grow min-h-0">
  <div id="mainPanels" class="min-h-0">
    <aside id="leftPanel" class="panel flex flex-col bg-white shadow-lg">
      <h3 class="panel-header font-medium p-2 border-b flex justify-between items-center flex-shrink-0">
        <span class="panel-header-title">Controles</span>
        <button id="toggleLeft" class="px-2 py-1 hover:bg-gray-200 rounded">◀</button>
      </h3>
      <div class="panel-content p-4 overflow-y-auto">
        <input id=videoFile type=file accept="video/*" class="block w-full text-sm text-gray-900 border border-gray-300 rounded-lg cursor-pointer bg-gray-50 focus:outline-none mb-4">
        <div id=ctl class="hidden space-y-3">
          <button id="infoBtn" class="bg-gray-200 hover:bg-gray-300 text-gray-800 px-3 py-1 rounded w-full text-sm disabled:opacity-50" disabled>Informações do arquivo</button>
          <span class="text-sm font-mono block pt-2" id=ts>t=0.000s</span>
          <div class="space-x-1"><span class="text-sm mr-2">Velocidade:</span><button class=sBtn data-s=.25>0.25×</button><button class=sBtn data-s=.5>0.5×</button><button class=sBtn data-s=1>1×</button><button class=sBtn data-s=1.5>1.5×</button><button class=sBtn data-s=2>2×</button></div>
          <div class="space-x-1"><span class="text-sm mr-2">Frame:</span><button id=prev class="px-2">◀</button><button id=next class="px-2">▶</button></div>
          <div class="space-y-1 pt-2">
            <label for="activeCategorySelect" class="text-sm font-medium">Categoria para Captura:</label>
            <select id="activeCategorySelect" class="w-full bg-gray-50 border border-gray-300 text-gray-900 text-sm rounded-lg focus:ring-indigo-500 focus:border-indigo-500 block p-2"></select>
          </div>
          <button id=cap class="mt-2 bg-indigo-600 text-white px-3 py-1 rounded w-full">Capturar Frame</button>
        </div>
      </div>
    </aside>
    <div id="playerWrapper" class="bg-black flex items-center justify-center min-w-0">
      <p id="videoPrompt" class="text-gray-400 text-lg">Selecione um vídeo no painel à esquerda.</p>
    </div>
    <aside id="rightPanel" class="panel flex flex-col bg-white shadow-lg">
      <h3 class="panel-header font-medium p-2 border-b flex justify-between items-center flex-shrink-0">
        <span class="panel-header-title">Gerenciar/Exportar</span>
        <button id="toggleRight" class="px-2 py-1 hover:bg-gray-200 rounded">▶</button>
      </h3>
      <div class="panel-content p-4 overflow-y-auto flex flex-col">
        <div>
            <h4 class="text-sm font-medium">Nova Categoria:</h4>
            <div class=flex mb-4><input id=newCat class="flex-1 border px-2 rounded-l-md" placeholder="Nome…"><button id=addCat class="bg-green-600 text-white px-3 rounded-r-md">+</button></div>
            <h4 class="text-sm font-medium border-t pt-3 mt-3">Categorias Existentes:</h4>
            <ul id="categoryList" class="text-sm space-y-1 h-24 overflow-y-auto"></ul>
        </div>
        <div class="grid grid-cols-2 gap-2 mt-auto border-t pt-4">
          <button id="saveCatsBtn" class="bg-teal-600 hover:bg-teal-700 text-white px-2 py-1 rounded text-sm disabled:opacity-50" disabled>Salvar Categorias</button>
          <button id="loadCatsBtn" class="bg-teal-600 hover:bg-teal-700 text-white px-2 py-1 rounded text-sm">Carregar Categorias</button>
          <button id="saveGalleryBtn" class="bg-sky-600 hover:bg-sky-700 text-white px-2 py-1 rounded text-sm disabled:opacity-50" disabled>Salvar Galeria</button>
          <button id="loadGalleryBtn" class="bg-sky-600 hover:bg-sky-700 text-white px-2 py-1 rounded text-sm disabled:opacity-50" disabled>Carregar Galeria</button>
          <button id="expZip" class="bg-amber-500 hover:bg-amber-600 text-white px-2 py-1 rounded text-sm disabled:opacity-50" disabled>Exportar ZIP</button>
          <button id="expCsv" class="bg-amber-500 hover:bg-amber-600 text-white px-2 py-1 rounded text-sm disabled:opacity-50" disabled>Exportar CSV</button>
        </div>
        <input type="file" id="categoryFileInput" class="hidden" accept=".json"><input type="file" id="galleryFileInput" class="hidden" accept=".json">
      </div>
    </aside>
  </div>
  <aside id="bottomPanel" class="panel flex flex-col bg-gray-100 border-t-2">
    <h3 class="font-medium p-2 bg-white border-b flex justify-between items-center cursor-pointer flex-shrink-0" id="toggleBottomHeader">
      <div class="flex items-center gap-4">
        <span>Galeria de Frames</span>
        <div id="galleryControls" class="flex gap-2 items-center">
            <label for="categoryFilter" class="text-sm font-normal">Filtrar:</label>
            <select id="categoryFilter" class="bg-gray-200 rounded px-2 py-1 text-sm"></select>
        </div>
      </div>
      <span id="toggleBottomIcon">▼</span>
    </h3>
    <div class="panel-content p-4 overflow-y-auto bg-white" id="galleryWrapper">
        <div id="galleryGrid" class="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5 gap-4"></div>
    </div>
  </aside>
</div>
<div id="infoModal" class="fixed inset-0 bg-black/60 p-4 z-50 hidden items-center justify-center">
  <div class="bg-white rounded-lg shadow-xl p-6 w-full max-w-3xl flex flex-col">
    <div class="flex justify-between items-center mb-4 flex-shrink-0">
      <h2 class="text-xl font-bold text-gray-800">Informações do Arquivo de Mídia</h2>
      <button id="closeInfoModal" class="text-3xl text-gray-500 hover:text-gray-800">&times;</button>
    </div>
    <pre id="infoContent" class="bg-gray-100 p-4 rounded text-sm overflow-auto flex-grow whitespace-pre-wrap" style="max-height: 65vh; min-height: 200px;"></pre>
    <div class="mt-4 flex justify-end gap-2 flex-shrink-0">
      <button id="copyInfoBtn" class="bg-blue-500 hover:bg-blue-600 text-white px-4 py-2 rounded text-sm">Copiar</button>
      <button id="saveInfoBtn" class="bg-green-500 hover:bg-green-600 text-white px-4 py-2 rounded text-sm">Salvar em Arquivo</button>
    </div>
  </div>
</div>
<div id="imageModal" class="fixed inset-0 bg-black/80 p-4 z-50 hidden items-center justify-center" onclick="if(event.target===this) this.style.display='none'">
  <div class="bg-white rounded-lg shadow-xl w-full max-w-6xl flex overflow-hidden" style="height: 90vh;">
      <div class="flex-grow flex items-center justify-center bg-gray-800 p-2 relative"><img id="modalImage" src="" alt="Frame em tela cheia" class="max-w-full max-h-full object-contain"><div id="loadingIndicator" class="absolute text-white hidden">Processando...</div></div>
      <div class="w-64 flex-shrink-0 p-4 border-l overflow-y-auto">
          <h3 class="font-bold mb-4 text-gray-800">Ajustes de Imagem</h3>
          <div id="filter-list" class="space-y-2"></div>
      </div>
      <button id="closeImageModal" class="absolute top-2 right-2 text-white bg-black/50 rounded-full p-1 text-2xl leading-none">&times;</button>
  </div>
</div>

<script>
// STATE & CONFIG
let vId=null, selCat=null, player=null, cats={}, allFrames = [], currentVideoFilename = null, videoFps = 30, currentModalFrameId = null;
const $ = id => document.getElementById(id);
const L_WIDTH = '320px', R_WIDTH = '320px', COLLAPSED_WIDTH = '48px';
const speeds = [0.25, 0.5, 1.0, 1.5, 2.0];

// LAYOUT LOGIC
function updateGridLayout() { const isLeftCollapsed = $('leftPanel').classList.contains('collapsed'), isRightCollapsed = $('rightPanel').classList.contains('collapsed'); const col1 = isLeftCollapsed ? COLLAPSED_WIDTH : L_WIDTH, col3 = isRightCollapsed ? COLLAPSED_WIDTH : R_WIDTH; $('mainPanels').style.gridTemplateColumns = `${col1} 1fr ${col3}`; }
$('toggleLeft').onclick = () => { $('leftPanel').classList.toggle('collapsed'); $('leftPanel').querySelector('.panel-content').classList.toggle('hidden'); $('toggleLeft').innerHTML = $('leftPanel').classList.contains('collapsed') ? '▶' : '◀'; updateGridLayout(); };
$('toggleRight').onclick = () => { $('rightPanel').classList.toggle('collapsed'); $('rightPanel').querySelector('.panel-content').classList.toggle('hidden'); $('toggleRight').innerHTML = $('rightPanel').classList.contains('collapsed') ? '◀' : '▶'; updateGridLayout(); };
$('toggleBottomHeader').onclick = () => { $('bottomPanel').classList.toggle('collapsed'); $('galleryWrapper').classList.toggle('hidden'); $('toggleBottomIcon').innerHTML = $('bottomPanel').classList.contains('collapsed') ? '▲' : '▼'; };

// DATA & CORE APP LOGIC
async function initData() {
    try {
        const catsData = await fetch('/categories').then(r => r.json());
        cats = {};
        catsData.forEach(c => { cats[c.id] = { id: c.id, name: c.name }; });
        allFrames = vId ? await fetch(`/frames/${vId}`).then(r => r.json()) : [];
        updateAllUI();
    } catch(err) { console.error("Falha ao inicializar dados:", err); alert("Falha ao carregar dados do servidor."); }
}
function updateAllUI() { renderCategoryList(); populateCategoryFilter(); populateActiveCategorySelect(); renderGallery(); }

// <<< MODIFICADO >>> Lógica de upload alterada para resetar categorias e permitir re-seleção do mesmo arquivo.
$('videoFile').onchange=async e=>{
  const f=e.target.files[0]; if(!f) return;
  
  $('playerWrapper').innerHTML = '<p class="text-white">Reiniciando ambiente...</p>';
  
  // 1. Reseta as categorias no servidor para o estado padrão.
  await fetch('/categories/reset', { method: 'POST' });

  // 2. Envia o novo vídeo.
  const fd=new FormData(); fd.append('video',f);
  $('playerWrapper').innerHTML = '<p class="text-white">Enviando vídeo...</p>';
  const {id, filename, fps}=await fetch('/upload',{method:'POST',body:fd}).then(r=>r.json());
  
  // 3. Configura o novo estado da sessão.
  vId=id;
  currentVideoFilename = filename;
  videoFps = fps || 30;
  
  // 4. Configura o player e a UI.
  $('playerWrapper').innerHTML = `<video id="player" src="/video/${vId}" controls autoplay class="w-full h-full"></video>`;
  player = $('player');
  player.ontimeupdate=()=>{ const frameNum = Math.floor(player.currentTime * videoFps) + 1; ts.textContent = `t=${player.currentTime.toFixed(3)}s (frame ${frameNum})`; };
  player.onratechange=()=> updateSpeedButtons(player.playbackRate);
  $('ctl').classList.remove('hidden');
  ['infoBtn', 'saveCatsBtn', 'saveGalleryBtn', 'loadGalleryBtn', 'expZip', 'expCsv'].forEach(id => $(id).disabled = false);
  
  // 5. Inicializa os dados (agora com categorias resetadas) e atualiza a UI.
  await initData();
  updateSpeedButtons(1.0);

  // 6. Limpa o valor do input para permitir que o evento 'onchange' dispare novamente com o mesmo arquivo.
  e.target.value = null;
};

function updateSpeedButtons(newSpeed) {
    document.querySelectorAll('.sBtn').forEach(b => {
        if (Math.abs(parseFloat(b.dataset.s) - newSpeed) < 0.01) { b.classList.add('font-bold', 'text-indigo-700'); }
        else { b.classList.remove('font-bold', 'text-indigo-700'); }
    });
}

document.querySelectorAll('.sBtn').forEach(b=>b.onclick=()=>{if(player){ const rate=parseFloat(b.dataset.s); player.playbackRate=rate;}});
prev.onclick=()=>{if(!player) return; player.pause(); player.currentTime=Math.max(0,player.currentTime-1/videoFps)};
next.onclick=()=>{if(!player) return; player.pause(); player.currentTime+=1/videoFps};
$('activeCategorySelect').onchange = (e) => { selCat = e.target.value; };

cap.onclick=async()=>{
  if(!player || !selCat){alert('Player não está pronto ou categoria não foi selecionada.'); return;}
  const body={video_id:vId,cat_id:selCat,ts:player.currentTime};
  const newFrame=await fetch('/frame',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)}).then(r=>r.json());
  allFrames.push(newFrame);
  renderGallery();
};

async function handleApiCall(url, options, successMessage, errorMessage) {
    try {
        const response = await fetch(url, options);
        const result = await response.json();
        if (!response.ok) throw new Error(result.error || 'Erro desconhecido no servidor.');
        if(successMessage) alert(successMessage(result));
        return result;
    } catch (error) { console.error(errorMessage, error); alert(`${errorMessage}: ${error.message}`); return null; }
}

// CATEGORY MANAGEMENT
addCat.onclick = async () => { const newCatName = $('newCat').value.trim(); if (!newCatName) return; const newCat = await handleApiCall('/cat', { method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ name: newCatName }) }, null, 'Não foi possível criar a categoria'); if (newCat) { cats[newCat.id] = { id: newCat.id, name: newCat.name }; $('newCat').value = ''; updateAllUI(); }};
function renderCategoryList() {
    const list = $('categoryList'); list.innerHTML = '';
    Object.values(cats).sort((a,b) => a.name.localeCompare(b.name)).forEach(cat => {
        const li = document.createElement('li'); li.className = 'flex justify-between items-center group py-1'; li.id = `cat-list-item-${cat.id}`;
        const nameSpan = document.createElement('span'); nameSpan.textContent = cat.name; nameSpan.className = 'flex-grow'; li.appendChild(nameSpan);
        if (cat.name !== 'Não categorizado') {
            const controlsDiv = document.createElement('div'); controlsDiv.className = 'opacity-0 group-hover:opacity-100 transition-opacity';
            const editBtn = document.createElement('button'); editBtn.title = "Editar"; editBtn.innerHTML = '✏️'; editBtn.className = 'text-xs p-1'; editBtn.onclick = () => toggleCategoryEdit(cat.id, true);
            const deleteBtn = document.createElement('button'); deleteBtn.title = "Excluir"; deleteBtn.innerHTML = '🗑️'; deleteBtn.className = 'text-xs p-1 text-red-500'; deleteBtn.onclick = () => deleteCategory(cat.id);
            controlsDiv.append(editBtn, deleteBtn); li.appendChild(controlsDiv);
        }
        list.appendChild(li);
    });
}
function toggleCategoryEdit(catId, isEditing) {
    const li = $(`cat-list-item-${catId}`);
    if (isEditing) {
        const originalName = cats[catId].name;
        li.innerHTML = `<input type="text" value="${originalName}" class="flex-grow border rounded px-1 text-sm bg-white"><div class="flex"><button onclick="saveCategoryEdit('${catId}')" title="Salvar" class="text-xs p-1">✔️</button><button onclick="renderCategoryList()" title="Cancelar" class="text-xs p-1">❌</button></div>`;
        li.querySelector('input').focus();
    } else { renderCategoryList(); }
}
async function saveCategoryEdit(catId) {
    const input = $(`cat-list-item-${catId}`).querySelector('input'), newName = input.value.trim();
    if (!newName || newName === cats[catId].name) { renderCategoryList(); return; }
    const updatedCat = await handleApiCall(`/cat/${catId}`, { method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ name: newName }) }, null, 'Falha ao editar categoria');
    if (updatedCat) { cats[updatedCat.id].name = updatedCat.name; updateAllUI(); } else { renderCategoryList(); }
}
async function deleteCategory(catId) {
    if (!confirm(`Tem certeza que deseja excluir a categoria "${cats[catId].name}"? Todos os frames associados serão movidos para "Não categorizado".`)) return;
    const result = await handleApiCall(`/cat/${catId}`, { method: 'DELETE' }, null, 'Falha ao excluir categoria');
    if (result) { delete cats[catId]; allFrames.forEach(frame => { if (frame.cat_id === catId) frame.cat_id = result.default_cat_id; }); updateAllUI(); }
}
$('saveCatsBtn').onclick = () => { if (!currentVideoFilename) { alert("Carregue um vídeo primeiro."); return; } const baseName = currentVideoFilename.split('.').slice(0, -1).join('.'); window.location.href = `/categories/export?video_name=${encodeURIComponent(baseName)}`; };
$('loadCatsBtn').onclick = () => { $('categoryFileInput').click(); };
$('categoryFileInput').onchange = async (e) => { const file = e.target.files[0]; if (!file) return; const formData = new FormData(); formData.append('file', file); const result = await handleApiCall('/categories/import', { method: 'POST', body: formData }, res => `Importação concluída!\\nImportadas: ${res.imported}\\nIgnoradas: ${res.skipped}`, 'Falha ao carregar categorias'); if (result) await initData(); e.target.value = null; };

// GALLERY IMPORT/EXPORT
$('saveGalleryBtn').onclick = () => { if(vId) window.location.href = `/gallery/export/${vId}`; };
$('loadGalleryBtn').onclick = () => { if(vId) $('galleryFileInput').click(); };
$('galleryFileInput').onchange = async (e) => {
    const file = e.target.files[0]; if (!file || !vId) return;
    if (!confirm("Atenção: Isto substituirá TODAS as anotações do vídeo atual. Deseja continuar?")) { e.target.value = null; return; }
    const formData = new FormData(); formData.append('file', file);
    const result = await handleApiCall(`/gallery/import/${vId}`, { method: 'POST', body: formData }, res => `Importação concluída!\\n${res.imported} frames foram carregados.`, 'Falha ao carregar galeria');
    if (result) await initData();
    e.target.value = null;
};

// UI RENDERING
function populateSelectWithOptions(select, keepValue, addAll) {
    const currentVal = keepValue ? select.value : null; select.innerHTML = '';
    if (addAll) select.add(new Option('Todas as Categorias', 'all'));
    const sorted = Object.values(cats).sort((a,b) => (a.name === "Não categorizado") ? -1 : (b.name === "Não categorizado") ? 1 : a.name.localeCompare(b.name));
    let defaultId = null;
    sorted.forEach(cat => { if (cat.name === 'Não categorizado') defaultId = cat.id; select.add(new Option(cat.name, cat.id)); });
    if (keepValue && cats[currentVal]) select.value = currentVal; else if (!addAll) select.value = defaultId;
    return defaultId;
}
function populateCategoryFilter() { populateSelectWithOptions($('categoryFilter'), true, true); }
function populateActiveCategorySelect() { const defaultId = populateSelectWithOptions($('activeCategorySelect'), true, false); if (!selCat || !cats[selCat]) selCat = defaultId; }
$('categoryFilter').onchange = () => renderGallery();
function renderGallery() { const grid = $('galleryGrid'), selectedCid = $('categoryFilter').value; grid.innerHTML = ''; allFrames.filter(f => selectedCid === 'all' || f.cat_id === selectedCid).sort((a, b) => a.ts - b.ts).forEach(f => grid.appendChild(createFrameCard(f))); }
function createFrameCard(frame) {
    const card = document.createElement('div'); card.className = 'gallery-card bg-white rounded-lg shadow p-2 flex flex-col border';
    const imgBtn = document.createElement('button'); imgBtn.innerHTML = `<img src="${frame.img_url}" class="rounded w-full object-cover aspect-video mb-2" loading="lazy">`;
    imgBtn.onclick = () => { if (player) { player.currentTime = frame.ts; player.pause(); } };
    const tsLabel = document.createElement('p'); tsLabel.className = 'text-xs text-gray-600 mb-1'; tsLabel.textContent = `t: ${frame.ts.toFixed(3)}s`;
    const catSelect = document.createElement('select'); catSelect.className = "text-xs p-1 rounded border w-full mb-2";
    populateSelectWithOptions(catSelect, false, false); catSelect.value = frame.cat_id;
    catSelect.onchange = (e) => changeFrameCategory(frame.id, e.target.value);
    const noteArea = document.createElement('textarea'); noteArea.className = "text-xs p-1 mt-1 rounded border w-full h-20 bg-gray-50";
    noteArea.value = frame.note || ''; noteArea.onblur = (e) => updateFrameNote(frame.id, e.target.value);
    const btnContainer = document.createElement('div'); btnContainer.className = 'mt-auto flex gap-1';
    const openBtn = document.createElement('button'); openBtn.className = 'w-1/2 bg-blue-500 hover:bg-blue-600 text-white text-xs py-1 px-2 rounded';
    openBtn.textContent = 'Abrir'; openBtn.onclick = () => openImageModal(frame.id);
    const delBtn = document.createElement('button'); delBtn.className = 'w-1/2 bg-red-500 hover:bg-red-600 text-white text-xs py-1 px-2 rounded';
    delBtn.textContent = 'Deletar'; delBtn.onclick = () => deleteFrame(frame.id);
    btnContainer.append(openBtn, delBtn);
    card.append(imgBtn, tsLabel, catSelect, noteArea, btnContainer); return card;
}
async function updateFrameNote(fid, note) { const frame = allFrames.find(f => f.id === fid); if (frame) frame.note = note; try { await fetch(`/frame/${fid}`, { method: 'PUT', headers: {'Content-Type': 'application/json'}, body: JSON.stringify({ note }) }); } catch (error) { console.error("Error saving note:", error); } }
async function changeFrameCategory(fid, cid) {
    const result = await handleApiCall(`/frame/${fid}/change_category`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ new_cat_id: cid }) }, null, "Não foi possível alterar a categoria");
    if(result) { const frame = allFrames.find(f => f.id === fid); if (frame) { frame.cat_id = cid; $('categoryFilter').value = 'all'; renderGallery(); }}
}
async function deleteFrame(fid) {
    if (!confirm('Tem certeza que deseja deletar este frame?')) return;
    const res = await handleApiCall('/frame/' + fid, { method: 'DELETE' }, null, "Falha ao deletar o frame.");
    if (res && res.ok) { allFrames = allFrames.filter(f => f.id !== fid); renderGallery(); }
}
expZip.onclick=()=>vId && (window.location='/export/zip/'+vId);
expCsv.onclick=()=>vId && (window.location='/export/csv/'+vId);

// MODALS (Info & Image)
const infoModal = $('infoModal');
$('infoBtn').onclick = async () => { if (!vId) return; infoModal.style.display = 'flex'; const content = $('infoContent'); content.textContent = 'Carregando...'; const result = await handleApiCall(`/mediainfo/${vId}`, {}, null, "Falha ao obter informações do vídeo"); if(result) content.textContent = result.info; };
$('closeInfoModal').onclick = () => infoModal.style.display = 'none';
$('copyInfoBtn').onclick = (e) => { navigator.clipboard.writeText($('infoContent').textContent); const btn = e.target; btn.textContent = 'Copiado!'; setTimeout(() => { btn.textContent = 'Copiar'; }, 2000); };
$('saveInfoBtn').onclick = () => { const text = $('infoContent').textContent; const blob = new Blob([text], { type: 'text/plain' }); const url = URL.createObjectURL(blob); const a = document.createElement('a'); a.href = url; const baseName = currentVideoFilename.split('.').slice(0, -1).join('.'); a.download = `mediainfo_${baseName}.txt`; document.body.appendChild(a); a.click(); document.body.removeChild(a); URL.revokeObjectURL(url); };

const imageModal = $('imageModal'), modalImage = $('modalImage'), loadingIndicator = $('loadingIndicator');
$('closeImageModal').onclick = () => imageModal.style.display = 'none';

function openImageModal(frameId) {
    currentModalFrameId = frameId;
    const frame = allFrames.find(f => f.id === frameId);
    if (!frame) return;
    renderFilterControls(frame.filters);
    updateModalImage();
    imageModal.style.display = 'flex';
}
function updateModalImage() {
    const frame = allFrames.find(f => f.id === currentModalFrameId); if (!frame) return;
    const activeFilters = frame.filters.filter(f => f.enabled);
    loadingIndicator.style.display = 'block';
    if (activeFilters.length > 0) {
        const filtersQuery = encodeURIComponent(JSON.stringify(activeFilters));
        modalImage.src = `/frame_image_processed/${frame.path}?filters=${filtersQuery}`;
    } else {
        modalImage.src = frame.img_url;
    }
    modalImage.onload = () => loadingIndicator.style.display = 'none';
}

// FILTER LOGIC
function createFilterControl(filter) {
    const el = document.createElement('div');
    el.className = 'p-2 border rounded filter-item';
    el.dataset.name = filter.name;
    let controlsHtml = '';
    if(filter.name === 'brightness_contrast'){
        controlsHtml = `
            <div class="space-y-1"><label class="text-xs">Brilho</label><input type="range" min="-100" max="100" value="${filter.brightness}" data-param="brightness" class="w-full filter-slider"><span class="text-xs">${filter.brightness}</span></div>
            <div class="space-y-1"><label class="text-xs">Contraste</label><input type="range" min="-100" max="100" value="${filter.contrast}" data-param="contrast" class="w-full filter-slider"><span class="text-xs">${filter.contrast}</span></div>`;
    } else if (filter.name === 'clahe') {
        controlsHtml = `
            <div class="space-y-1"><label class="text-xs">Limite Contraste</label><input type="range" min="1" max="40" value="${filter.clipLimit}" step="0.5" data-param="clipLimit" class="w-full filter-slider"><span class="text-xs">${filter.clipLimit}</span></div>
            <div class="space-y-1"><label class="text-xs">Tam. Grade</label><input type="range" min="2" max="16" value="${filter.gridSize}" data-param="gridSize" class="w-full filter-slider"><span class="text-xs">${filter.gridSize}</span></div>`;
    }
    el.innerHTML = `
        <div class="flex items-center justify-between">
            <label class="text-sm font-medium flex items-center gap-2"><span class="cursor-grab">☰</span><input type="checkbox" data-param="enabled" ${filter.enabled ? 'checked' : ''}>${filter.label}</label>
        </div>
        <div class="pl-6 pt-1 filter-controls ${filter.enabled ? '' : 'hidden'}">${controlsHtml}</div>`;
    return el;
}
function renderFilterControls(filters) {
    const container = $('filter-list');
    container.innerHTML = '';
    filters.forEach(f => container.appendChild(createFilterControl(f)));
    container.querySelectorAll('input').forEach(input => input.oninput = applyFilters);
    new Sortable(container, {
        animation: 150,
        handle: '.cursor-grab',
        onEnd: () => applyFilters(true) // Pass true to indicate reorder
    });
}
async function applyFilters(isReorder = false) {
    const frame = allFrames.find(f => f.id === currentModalFrameId); if (!frame) return;
    const newFilters = [];
    $('filter-list').querySelectorAll('.filter-item').forEach(item => {
        const name = item.dataset.name;
        const newFilterState = { name: name, label: frame.filters.find(f=>f.name===name).label };
        item.querySelectorAll('input').forEach(input => {
            const param = input.dataset.param;
            const value = input.type === 'checkbox' ? input.checked : (input.type === 'range' ? parseFloat(input.value) : input.value);
            newFilterState[param] = value;
            if (input.type === 'range') input.nextElementSibling.textContent = value;
            if(param==='enabled') item.querySelector('.filter-controls').classList.toggle('hidden', !value);
        });
        newFilters.push(newFilterState);
    });
    frame.filters = newFilters;
    try {
      await fetch(`/frame/${currentModalFrameId}/filters`, {
          method: 'PUT',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(newFilters)
      });
    } catch (err) {
      console.error("Falha ao salvar os filtros no servidor:", err);
    }
    updateModalImage();
}

// KEYBOARD SHORTCUTS
document.addEventListener('keydown', (e) => {
    const activeEl = document.activeElement;
    if (activeEl && (activeEl.tagName === 'INPUT' || activeEl.tagName === 'TEXTAREA')) return;
    if (infoModal.style.display === 'flex' || imageModal.style.display === 'flex') return;
    if (!player) return;
    let preventDefault = true;
    switch(e.key.toLowerCase()) {
        case 'arrowright': next.click(); break;
        case 'arrowleft': prev.click(); break;
        case ' ': if (player.paused) player.play(); else player.pause(); break;
        case 'enter': cap.click(); break;
        case 'l': let u = speeds.indexOf(player.playbackRate); if (u < 0) u = 2; if (u < speeds.length - 1) player.playbackRate = speeds[u + 1]; break;
        case 'j': let d = speeds.indexOf(player.playbackRate); if (d < 0) d = 2; if (d > 0) player.playbackRate = speeds[d - 1]; break;
        case 'k': player.pause(); player.playbackRate = 1.0; break;
        default: preventDefault = false; break;
    }
    if (preventDefault) e.preventDefault();
});

updateGridLayout();
initData();
</script></body></html>
"""

# ------------------------------ Routes (Python Backend) ------------------------------------
@app.route("/")
def index(): return render_template_string(HTML)

@app.route("/categories")
def get_categories(): return jsonify(load_categories_from_file())

@app.route("/cat", methods=["POST"])
def add_category():
    name = request.get_json()['name'].strip()
    if not name: return jsonify({"error": "Nome não pode ser vazio."}), 400
    cats = load_categories_from_file()
    if any(c['name'] == name for c in cats): return jsonify({"error": f"Categoria '{name}' já existe."}), 409
    new_cat = {"id": uuid.uuid4().hex, "name": name, "color": "#4f46e5"}
    cats.append(new_cat)
    save_categories_to_file(cats)
    return jsonify(new_cat), 201

@app.route("/cat/<cat_id>", methods=["PUT"])
def update_category(cat_id):
    new_name = request.get_json().get("name").strip()
    if not new_name: return jsonify({"error": "O nome não pode ser vazio."}), 400
    cats = load_categories_from_file()
    target_cat = next((c for c in cats if c['id'] == cat_id), None)
    if not target_cat: return "Categoria não encontrada", 404
    if target_cat['name'] == "Não categorizado": return jsonify({"error": "A categoria padrão não pode ser editada."}), 403
    if any(c['name'] == new_name and c['id'] != cat_id for c in cats): return jsonify({"error": f"A categoria '{new_name}' já existe."}), 409
    target_cat['name'] = new_name
    save_categories_to_file(cats)
    return jsonify(target_cat)

@app.route("/cat/<cat_id>", methods=["DELETE"])
def delete_category(cat_id):
    cats = load_categories_from_file()
    cat_to_delete = next((c for c in cats if c['id'] == cat_id), None)
    if not cat_to_delete: return "Categoria não encontrada", 404
    if cat_to_delete['name'] == "Não categorizado": return jsonify({"error": "A categoria padrão não pode ser excluída."}), 403
    default_id = get_default_category()['id']
    for frames in FRAMES_BY_VIDEO.values():
        for frame in frames:
            if frame['cat_id'] == cat_id: frame['cat_id'] = default_id
    new_cats = [c for c in cats if c['id'] != cat_id]
    save_categories_to_file(new_cats)
    return jsonify({"ok": True, "default_cat_id": default_id})

@app.route("/categories/export")
def export_categories():
    video_name = request.args.get('video_name', 'padrao')
    cats = [c for c in load_categories_from_file() if c['name'] != "Não categorizado"]
    buffer = io.BytesIO(json.dumps(cats, indent=2).encode('utf-8'))
    buffer.seek(0)
    return send_file(buffer, as_attachment=True, download_name=f"categorias_{video_name}.json", mimetype='application/json')

@app.route("/categories/import", methods=["POST"])
def import_categories():
    file = request.files.get('file');
    if not file or not file.filename.endswith('.json'): return jsonify({"error": "Arquivo inválido."}), 400
    try:
        new_cats_data, cats, cat_names = json.load(file.stream), load_categories_from_file(), {c['name'] for c in load_categories_from_file()}
        if not isinstance(new_cats_data, list): return jsonify({"error": "O JSON deve ser uma lista."}), 400
        imported_count, skipped_count = 0, 0
        for cat_info in new_cats_data:
            name = cat_info.get("name", "").strip()
            if not name or name in cat_names: skipped_count += 1; continue
            cats.append({"id": uuid.uuid4().hex, "name": name, "color": cat_info.get("color", "#4f46e5")})
            cat_names.add(name); imported_count += 1
        save_categories_to_file(cats)
        return jsonify({"imported": imported_count, "skipped": skipped_count})
    except Exception as e: return jsonify({"error": f"Falha ao processar: {e}"}), 500

# <<< ADICIONADO >>> Rota para resetar as categorias para o estado padrão.
@app.route("/categories/reset", methods=["POST"])
def reset_categories():
    try:
        default_cats = [get_default_category()]
        save_categories_to_file(default_cats)
        return jsonify(default_cats)
    except Exception as e:
        app.logger.error(f"Falha ao resetar categorias: {e}")
        return jsonify({"error": "Não foi possível resetar as categorias."}), 500

@app.route("/gallery/export/<vid>")
def export_gallery(vid):
    if vid not in VIDEOS_SESSIONS: return "Vídeo não encontrado", 404
    frames, cats_map = FRAMES_BY_VIDEO.get(vid, []), {c['id']: c for c in load_categories_from_file()}
    gallery_data = [{"ts": f["ts"], "cat_name": cats_map.get(f["cat_id"], {}).get("name"), "note": f["note"], "filters": f.get("filters", [])} for f in frames]
    buffer = io.BytesIO(json.dumps(gallery_data, indent=2).encode('utf-8'))
    buffer.seek(0)
    original_name = "_".join(os.path.splitext(os.path.basename(VIDEOS_SESSIONS[vid]['filename']))[0].split('_')[1:])
    return send_file(buffer, as_attachment=True, download_name=f"galeria_{original_name}.json", mimetype='application/json')

@app.route("/gallery/import/<vid>", methods=["POST"])
def import_gallery(vid):
    if vid not in VIDEOS_SESSIONS: return "Vídeo não encontrado", 404
    file = request.files.get('file');
    if not file or not file.filename.endswith('.json'): return jsonify({"error": "Arquivo inválido."}), 400
    try:
        if vid in FRAMES_BY_VIDEO:
            for frame in FRAMES_BY_VIDEO[vid]:
                if os.path.exists(frame['fpath']): os.remove(frame['fpath'])
        FRAMES_BY_VIDEO[vid] = []
        data, cats_by_name, default_cat = json.load(file.stream), {c['name']: c for c in load_categories_from_file()}, get_default_category()
        if not isinstance(data, list): return jsonify({"error": "JSON deve ser uma lista."}), 400
        video_info, original_user_filename = VIDEOS_SESSIONS[vid], "_".join(os.path.splitext(os.path.basename(VIDEOS_SESSIONS[vid]['filename']))[0].split('_')[1:])
        imported_frames = []
        for idx, frame_info in enumerate(data):
            ts, note, cat_name, filters = frame_info.get("ts"), frame_info.get("note", ""), frame_info.get("cat_name"), frame_info.get("filters", [])
            if ts is None: continue
            category = cats_by_name.get(cat_name, default_cat)
            frame_filename = f"{original_user_filename}_frame{idx + 1}_ts{f'{ts:.3f}'.replace('.', '_')}.png"
            fpath = os.path.join(FRAMES_DIR, frame_filename)
            extract_exact_frame(video_info['filepath'], ts, fpath)
            new_frame = {"id": uuid.uuid4().hex, "video_id": vid, "cat_id": category['id'], "ts": ts, "path": frame_filename, "fpath": fpath, "note": note, "filters": filters, "img_url": url_for('serve_frame_image_by_path', frame_path=frame_filename)}
            imported_frames.append(new_frame)
        FRAMES_BY_VIDEO[vid] = imported_frames
        return jsonify({"imported": len(imported_frames)})
    except Exception as e: return jsonify({"error": f"Falha ao processar: {e}"}), 500

@app.route("/upload", methods=["POST"])
def upload():
    f = request.files.get('video');
    if not f or not f.filename: return jsonify({"error": "No file"}), 400
    vid, filepath = uuid.uuid4().hex, os.path.join(VIDEOS_DIR, f"{uuid.uuid4().hex}_{f.filename}")
    f.save(filepath)
    fps = 30.0
    try:
        with av.open(filepath) as container:
            stream = container.streams.video[0]
            if stream.average_rate: fps = float(stream.average_rate)
    except Exception as e: app.logger.error(f"Não foi possível determinar o FPS para {filepath}: {e}")
    VIDEOS_SESSIONS[vid] = {'id': vid, 'filename': f.filename, 'filepath': filepath, 'fps': fps}
    FRAMES_BY_VIDEO[vid] = []
    return jsonify({"id": vid, "filename": f.filename, "fps": fps})

@app.route("/video/<vid>")
def serve_video(vid):
    if vid not in VIDEOS_SESSIONS: return "Vídeo não encontrado", 404
    return send_file(VIDEOS_SESSIONS[vid]['filepath'])

@app.route("/frame", methods=["POST"])
def save_frame():
    d = request.get_json(); vid, ts, cid = d['video_id'], float(d['ts']), d.get('cat_id')
    if vid not in VIDEOS_SESSIONS: return "Vídeo não encontrado", 404
    category = next((c for c in load_categories_from_file() if c['id'] == cid), get_default_category())
    frame_num = len(FRAMES_BY_VIDEO.get(vid, [])) + 1
    default_note = f"Frame: {frame_num}, Tempo: {ts:.3f}s"
    video_info = VIDEOS_SESSIONS[vid]
    original_user_filename = "_".join(os.path.splitext(os.path.basename(video_info['filename']))[0].split('_')[1:])
    frame_filename = f"{original_user_filename}_frame{frame_num}_ts{f'{ts:.3f}'.replace('.', '_')}.png"
    fpath = os.path.join(FRAMES_DIR, frame_filename)
    extract_exact_frame(video_info['filepath'], ts, fpath)
    default_filters = [
        {"name": "brightness_contrast", "label": "Brilho/Contraste", "enabled": False, "brightness": 0, "contrast": 0},
        {"name": "clahe", "label": "CLAHE", "enabled": False, "clipLimit": 2.0, "gridSize": 8},
        {"name": "white_balance", "label": "Balanço de Branco", "enabled": False}
    ]
    new_frame = {"id": uuid.uuid4().hex, "video_id": vid, "cat_id": category['id'], "ts": ts, "path": frame_filename, "fpath": fpath, "note": default_note, "filters": default_filters, "cat_name": category['name'], "img_url": url_for('serve_frame_image_by_path', frame_path=frame_filename)}
    FRAMES_BY_VIDEO.setdefault(vid, []).append(new_frame)
    return jsonify(new_frame)

@app.route("/frame/<fid>", methods=["PUT"])
def update_note(fid):
    new_note = request.get_json().get("note", "")
    for frames in FRAMES_BY_VIDEO.values():
        for frame in frames:
            if frame['id'] == fid: frame['note'] = new_note; return jsonify({"ok": True})
    return "Frame não encontrado", 404

@app.route("/frame/<fid>", methods=["DELETE"])
def delete_frame(fid):
    for vid, frames in FRAMES_BY_VIDEO.items():
        frame_to_delete = next((f for f in frames if f['id'] == fid), None)
        if frame_to_delete:
            if os.path.exists(frame_to_delete['fpath']): os.remove(frame_to_delete['fpath'])
            FRAMES_BY_VIDEO[vid] = [f for f in frames if f['id'] != fid]
            return jsonify({"ok": True})
    return "Frame não encontrado", 404
    
@app.route("/frame_image/<path:frame_path>")
def serve_frame_image_by_path(frame_path):
    fpath = os.path.join(FRAMES_DIR, frame_path)
    if not os.path.exists(fpath): return "Arquivo não encontrado", 404
    return send_file(fpath)
    
@app.route("/frame_image_processed/<path:frame_path>")
def serve_processed_frame_image(frame_path):
    fpath = os.path.join(FRAMES_DIR, frame_path)
    if not os.path.exists(fpath): return "Arquivo não encontrado", 404
    try:
        filters_json = request.args.get('filters', '[]')
        filters_array = json.loads(filters_json)
        if not filters_array: return send_file(fpath)
        with open(fpath, "rb") as f:
            img_bytes = f.read()
        processed_bytes = apply_filter_pipeline(img_bytes, filters_array)
        if processed_bytes is None: return "Falha ao processar imagem", 500
        return send_file(io.BytesIO(processed_bytes), mimetype='image/png')
    except Exception as e:
        app.logger.error(f"Erro ao processar imagem: {e}")
        return "Erro de processamento", 500

@app.route("/frames/<vid>")
def get_frames(vid): return jsonify(FRAMES_BY_VIDEO.get(vid, []))

@app.route("/frame/<fid>/change_category", methods=["POST"])
def change_frame_category(fid):
    new_cat_id = request.get_json().get('new_cat_id')
    if not any(c['id'] == new_cat_id for c in load_categories_from_file()): return "Categoria não encontrada", 404
    for frames in FRAMES_BY_VIDEO.values():
        for frame in frames:
            if frame['id'] == fid: frame['cat_id'] = new_cat_id; return jsonify({"ok": True})
    return "Frame não encontrado", 404

@app.route("/frame/<fid>/filters", methods=["PUT"])
def update_frame_filters(fid):
    filters_data = request.get_json()
    if not isinstance(filters_data, list):
        return jsonify({"error": "Dados de filtro inválidos."}), 400
    for frames in FRAMES_BY_VIDEO.values():
        for frame in frames:
            if frame['id'] == fid:
                frame['filters'] = filters_data
                return jsonify({"ok": True})
    return "Frame não encontrado", 404

@app.route("/export/zip/<vid>")
def export_zip(vid):
    if vid not in VIDEOS_SESSIONS: return "Vídeo não encontrado", 404
    video_info, frames = VIDEOS_SESSIONS[vid], FRAMES_BY_VIDEO.get(vid, [])
    cats_map = {c['id']: c for c in load_categories_from_file()}
    original_name = "_".join(os.path.splitext(os.path.basename(video_info['filename']))[0].split('_')[1:])
    buf = io.BytesIO()
    with ZipFile(buf, 'w', ZIP_DEFLATED) as zf:
        for r in frames:
            if not os.path.exists(r['fpath']): continue
            cat_name = cats_map.get(r['cat_id'], {}).get("name", "sem_categoria")
            arcname = os.path.join(cat_name, r['path'])
            active_filters = [f for f in r.get('filters', []) if f.get('enabled')]
            if active_filters:
                try:
                    with open(r['fpath'], "rb") as f:
                        img_bytes = f.read()
                    processed_bytes = apply_filter_pipeline(img_bytes, active_filters)
                    if processed_bytes: zf.writestr(arcname, processed_bytes)
                except Exception as e:
                    app.logger.error(f"Falha ao processar e adicionar o frame {r['path']} ao zip: {e}")
            else:
                zf.write(r['fpath'], arcname=arcname)
    buf.seek(0)
    return send_file(buf, download_name=f"imagens_{original_name}.zip", as_attachment=True)

@app.route("/export/csv/<vid>")
def export_csv(vid):
    if vid not in VIDEOS_SESSIONS: return "Vídeo não encontrado", 404
    video_info, frames = VIDEOS_SESSIONS[vid], FRAMES_BY_VIDEO.get(vid, [])
    cats_map = {c['id']: c for c in load_categories_from_file()}
    original_name = "_".join(os.path.splitext(os.path.basename(video_info['filename']))[0].split('_')[1:])
    rows = [{"category": cats_map.get(r['cat_id'], {}).get("name", "sem_categoria"), "timestamp": r['ts'], "file": r['path'], "note": r['note']} for r in frames]
    csv_io = io.StringIO(); pd.DataFrame(rows).to_csv(csv_io, index=False); csv_io.seek(0)
    return send_file(io.BytesIO(csv_io.getvalue().encode('utf-8')), download_name=f"relatorio_{original_name}.csv", as_attachment=True)
    
@app.route("/mediainfo/<vid>")
def get_mediainfo(vid):
    if vid not in VIDEOS_SESSIONS: return "Vídeo não encontrado", 404
    filepath = VIDEOS_SESSIONS[vid]['filepath']
    sha512_hash = hashlib.sha512()
    try:
        with open(filepath, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha512_hash.update(byte_block)
        hash_hex = sha512_hash.hexdigest()
    except IOError as e:
        app.logger.error(f"Não foi possível ler o arquivo para hash: {e}")
        hash_hex = "Erro ao ler o arquivo para calcular o hash."
    try:
        media_info = MediaInfo.parse(filepath)
        info_text = [f"SHA-512: {hash_hex}", ""]
        for track in media_info.tracks:
            info_text.append(f"--- {track.track_type} ---")
            track_data = track.to_data()
            for key, value in sorted(track_data.items()):
                if key not in ['track_type', 'streamorder', 'track_id']:
                     info_text.append(f"{key.replace('_',' ').title():>25}: {value}")
            info_text.append("")
        return jsonify({"info": "\n".join(info_text)})
    except Exception as e:
        app.logger.error(f"Erro ao obter MediaInfo: {e}")
        return jsonify({"error": f"Não foi possível obter os metadados do vídeo: {e}"}), 500

# ----------------------------- Run ----------------------------------------
if __name__ == "__main__":
    app.run(debug=True, threaded=False)