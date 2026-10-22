/* ============================================================
   MiniMax-H3 Video WebUI 逻辑（原生 JS，直连引擎 API）
   六模式联动：t2v / i2v / fl2v / ref2v / ref2va / video_edit
   方案管理（A/B/C/D）+ 双层交叉过滤（Tab→范式 × 方案→精度）
   设置区（模式/方案/变体/加载） ↔ 参数区（模式上传/提示词/视频参数/高级） ↔ 结果区（SSE + 播放器）
   ============================================================ */
"use strict";

const API = "";  // 同源直连（引擎自身挂载 /ui）

// ---------- 状态 ----------
const state = {
  mode: "t2v",
  i2vRole: "first_frame",
  i2vImage: null,       // { name, path, localUrl }
  fl2vFirst: null,      // { name, path, localUrl }
  fl2vLast: null,       // { name, path, localUrl }
  ref2vImages: [],      // [{ name, path, localUrl }] ≤9
  ref2vaImages: [],     // [{ name, path, localUrl }] ≤9（多模态参考-图）
  ref2vaVideos: [],     // [{ name, path, localUrl }] ≤3（多模态参考-视频）
  ref2vaAudios: [],     // [{ name, path, localUrl }] ≤3（多模态参考-音频）
  editVideo: null,      // { name, path, localUrl }
  variants: [],         // capabilities.variants
  schemes: [],          // /schemes.schemes
  defaultSchemeId: "",
  loadedSchemeId: "",
  generating: false,
  loading: false,
};

// ---------- DOM ----------
const $ = (id) => document.getElementById(id);

// SSE 阶段 → 中文（stage 定义见 ui.py generate_video + progress_reporter.py）
const STAGE_TEXT = {
  "video.start": "开始生成",
  "video.model_ready": "模型就绪",
  "video.prompt_processing": "处理提示词",
  "video.generating": "生成视频帧",
  "video.vae_decoding": "VAE 解码视频",
  "video.audio_decoding": "解码音频",
  "video.saving": "保存产物",
  "completed": "完成",
};

// 模式提示（对应 Gradio ui.py 五 Tab + ref2va 多模态参考）
const MODE_HINTS = {
  t2v: "T2V 文生视频：仅根据文本描述生成视频（FL2VA 范式，含原生音频）",
  i2v: "I2V 图生视频：上传 1 张图片作为首帧或尾帧（FL2VA 范式）",
  fl2v: "首尾帧生视频：上传首帧和尾帧，模型生成两帧之间的过渡（FL2VA 范式）",
  ref2v: "多宫格生视频：上传最多 9 张参考图，提示词用 <Subject N> 引用（Ref2VA 范式）",
  ref2va: "多模态参考：图（≤9）/视频（≤3）/音频（≤3）混合参考，按组合自动判定场景（对口型/音色/动作参考，Ref2VA 范式，含音频）",
  video_edit: "视频编辑：上传源视频 + 编辑指令（Ref2VA 范式，支持 <Video N>/<Audio N> 标签）",
};

const MODE_CN = {
  t2v: "文生视频",
  i2v: "图生视频",
  fl2v: "首尾帧",
  ref2v: "多宫格",
  ref2va: "多模态",
  video_edit: "视频编辑",
};

// Tab → 范式映射（对应 ui.py _TAB_TO_PARADIGM）
const TAB_TO_PARADIGM = {
  t2v: "FL2VA",
  i2v: "FL2VA",
  fl2v: "FL2VA",
  ref2v: "Ref2VA",
  ref2va: "Ref2VA",
  video_edit: "Ref2VA",
};

// 方案联动表（对应 ui.py _SCHEME_LINKAGE；与 schemes/*.yaml 实际配置对齐）
const SCHEME_LINKAGE = {
  T4: { offload: "L2", steps: 4,  sage: false, lora: true,  fbc: false, blockswap: false },
  T8: { offload: "L2", steps: 8,  sage: false, lora: true,  fbc: false, blockswap: false },
  A:  { offload: "L2", steps: 30, sage: true,  lora: false, fbc: true,  blockswap: false },
  B:  { offload: "L2", steps: 8,  sage: true,  lora: true,  fbc: false, blockswap: false },
  C:  { offload: "L3", steps: 8,  sage: true,  lora: true,  fbc: true,  blockswap: false },
  D:  { offload: "L0", steps: 30, sage: false, lora: false, fbc: false, blockswap: false },
};

// 方案适用场景说明（对应实测基准：T8=235.8s / T4=87.9s / A=406.0s / D 需 ≥80GB）
const SCHEME_NOTES = {
  T4: "快速预览（4 步，速度最快，画质略降）",
  T8: "默认均衡（8 步，速度与画质平衡）",
  A: "高质量（30 步 + FBC + Sage，细节最佳，速度慢）",
  B: "均衡加速（8 步 + Sage + LoRA）",
  C: "低显存激进（L3 激进卸载 + 全加速）",
  D: "全 GPU 极致质量（30 步，L0 全显存）",
};

// 高级设置选项说明（对应 ui.py acceleration_override）
const ADVANCED_HELP = [
  ["SageAttention", "1.5-2× 加速", "B / C / A", "高效注意力替代标准注意力，显存占用更低，画质基本无损"],
  ["LightX2V LoRA", "6-12× 加速", "T4 / T8 / B / C", "蒸馏 LoRA，推理步数可从 30 降到 8 甚至 4，画质略降"],
  ["FirstBlockCache", "~1.5× 加速", "A / C", "缓存首个 block 计算结果，跳过重复计算"],
  ["BlockSwap", "降低显存峰值", "C（低显存）", "显存不足时启用，与 L3 激进 offload 搭配"],
  ["LoRA 权重", "风格强度 0-2", "全部", "调节生成风格贴合度，默认 1.0"],
  ["CPU Offload", "显存分级卸载", "全部", "L0 全 GPU（≥80GB）/ L1 TE 卸载（≥48GB）/ L2 层级别（≥24GB）/ L3 激进（<24GB）"],
];

// ---------- 帮助弹窗（模态框）：问号按钮 → 表格说明 ----------
function helpTableHtml(heads, rows) {
  return `<table class="help-table">
    <thead><tr>${heads.map((h) => `<th>${h}</th>`).join("")}</tr></thead>
    <tbody>${rows.map((r) => `<tr>${r.map((c) => `<td>${c}</td>`).join("")}</tr>`).join("")}</tbody>
  </table>`;
}

// 方案参数表（SCHEME_LINKAGE 固定列 + state.schemes 实时显存/说明合并）
function schemeHelpTableHtml() {
  const heads = ["方案", "推理步数", "CPU Offload", "Sage", "LoRA", "FBC", "BlockSwap", "适用场景 / 显存"];
  const rows = Object.keys(SCHEME_LINKAGE).map((id) => {
    const lk = SCHEME_LINKAGE[id];
    const s = state.schemes.find((x) => x.id.toUpperCase() === id);
    const vram = s?.vram_required
      || (lk.offload === "L0" ? "≥80GB" : lk.offload === "L3" ? "<24GB" : "≥24GB");
    return [
      id,
      lk.steps,
      lk.offload,
      lk.sage ? "✓" : "—",
      lk.lora ? "✓" : "—",
      lk.fbc ? "✓" : "—",
      lk.blockswap ? "✓" : "—",
      `${SCHEME_NOTES[id] || ""}（显存 ${vram}）`,
    ];
  });
  return helpTableHtml(heads, rows);
}

// 高级设置表（加速选项 / LoRA / Offload 说明）
function advancedHelpTableHtml() {
  return helpTableHtml(["选项", "加速效果", "适用方案", "说明"], ADVANCED_HELP);
}

function openHelpDialog(title, bodyHtml) {
  $("help-modal-title").textContent = title;
  $("help-modal-body").innerHTML = bodyHtml;
  $("help-modal").style.display = "flex";
}

function closeHelpDialog() {
  $("help-modal").style.display = "none";
}

// ============================================================
// 初始化
// ============================================================
document.addEventListener("DOMContentLoaded", () => {
  bindModeSeg();
  bindRoleSeg();
  bindSliders();
  bindUploads();
  bindButtons();
  refreshAll();
  setInterval(refreshStatus, 15000);  // 状态轮询（空闲时）
});

// ============================================================
// 设置区：模式切换（五模式联动显隐 + 双层过滤 + 不匹配提示）
// ============================================================
function bindModeSeg() {
  document.querySelectorAll("#mode-seg .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#mode-seg .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.mode = btn.dataset.mode;
      $("mode-hint").textContent = MODE_HINTS[state.mode];
      // 模式专属上传区联动显隐（对应 Gradio 五 Tab + ref2va 多模态）
      $("i2v-field").style.display = state.mode === "i2v" ? "" : "none";
      $("fl2v-field").style.display = state.mode === "fl2v" ? "" : "none";
      $("ref2v-field").style.display = state.mode === "ref2v" ? "" : "none";
      $("ref2va-field").style.display = state.mode === "ref2va" ? "" : "none";
      $("edit-field").style.display = state.mode === "video_edit" ? "" : "none";
      // 提示词语义联动
      $("prompt-label").innerHTML = state.mode === "video_edit"
        ? '编辑指令 <span class="req">（必填）</span>'
        : '正向提示词 <span class="req">（必填）</span>';
      $("footer-mode").textContent = `模式：${MODE_CN[state.mode]}`;
      // v1.3：切换工作流 → 自动跳到该工作流的推荐最佳步数（未手动改时默认）
      applyRecommendedSteps(state.mode);
      // Tab 切换 → 双层过滤变体 + 已加载模型范式不匹配提示
      refreshVariantSelect();
      checkParadigmMismatch();
      // ref2va：进入时刷新场景判定，离开时恢复生成按钮
      updateRef2vaSceneHint();
      if (state.mode !== "ref2va" && !state.generating) $("btn-generate").disabled = false;
    });
  });
  $("footer-mode").textContent = "模式：文生视频";
}

// I2V 图片角色（首帧 / 尾帧）
function bindRoleSeg() {
  document.querySelectorAll("#i2v-role-seg .seg-btn").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#i2v-role-seg .seg-btn").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      state.i2vRole = btn.dataset.role;
    });
  });
}

// 已加载模型与当前 Tab 范式不匹配时提示（对应 ui.py on_tab_change_filter T4.3）
function checkParadigmMismatch() {
  const paradigm = TAB_TO_PARADIGM[state.mode];
  fetch(`${API}/status`)
    .then((r) => r.json())
    .then((st) => {
      const loaded = st.loaded_variant || "";
      if (!loaded || !paradigm) return;
      const loadedParadigm = loaded.includes("Ref2VA") ? "Ref2VA"
        : loaded.includes("FL2VA") ? "FL2VA" : "";
      if (loadedParadigm && loadedParadigm !== paradigm) {
        showError(
          `当前已加载模型 ${loaded}（${loadedParadigm}）不适用于 ${MODE_CN[state.mode]} 功能，` +
          `需加载 ${paradigm} 范式模型。请在「模型变体」下拉框选择对应的 [${paradigm}] 模型后重新生成。`
        );
      }
    })
    .catch(() => {});  // /status 不可达时静默跳过
}

// ============================================================
// 参数区：滑块联动
// ============================================================
function bindSliders() {
  const pairs = [
    ["duration-range", "duration-val", (v) => v + "s"],
    ["fps-range", "fps-val", (v) => v],
    ["steps-range", "steps-val", (v) => v],
    ["guidance-range", "guidance-val", (v) => parseFloat(v).toFixed(1)],
    ["adv-lora-scale", "adv-lora-scale-val", (v) => parseFloat(v).toFixed(1)],
    ["lora-weight", "lora-weight-val", (v) => parseFloat(v).toFixed(1)],
  ];
  for (const [id, valId, fmt] of pairs) {
    $(id).addEventListener("input", (e) => { $(valId).textContent = fmt(e.target.value); });
  }
}

// ============================================================
// 参数区：图片/视频上传（五模式联动）
// ============================================================
function bindUploads() {
  bindFileDropzone("i2v-dropzone", "i2v-file-input", (f) => uploadSingle("i2vImage", f));
  bindFileDropzone("fl2v-first-dropzone", "fl2v-first-input", (f) => uploadSingle("fl2vFirst", f));
  bindFileDropzone("fl2v-last-dropzone", "fl2v-last-input", (f) => uploadSingle("fl2vLast", f));
  bindFileDropzone("ref2v-dropzone", "ref2v-file-input", uploadRef2vBatch);
  bindFileDropzone("ref2va-img-dropzone", "ref2va-img-input",
    (f) => uploadRef2vaBatch(f, "ref2vaImages", "image/", /\.(png|jpe?g|webp|gif|bmp)$/i, 9));
  bindFileDropzone("ref2va-vid-dropzone", "ref2va-vid-input",
    (f) => uploadRef2vaBatch(f, "ref2vaVideos", "video/", /\.(mp4|mov|avi|mkv|webm)$/i, 3));
  bindFileDropzone("ref2va-aud-dropzone", "ref2va-aud-input",
    (f) => uploadRef2vaBatch(f, "ref2vaAudios", "audio/", /\.(mp3|wav|flac|aac|ogg|m4a|wma)$/i, 3));
  bindFileDropzone("edit-dropzone", "edit-file-input", (f) => uploadSingle("editVideo", f));
}

function bindFileDropzone(dzId, fiId, handler) {
  const dz = $(dzId);
  const fi = $(fiId);
  dz.addEventListener("click", () => fi.click());
  dz.addEventListener("dragover", (e) => { e.preventDefault(); dz.classList.add("dragover"); });
  dz.addEventListener("dragleave", () => dz.classList.remove("dragover"));
  dz.addEventListener("drop", (e) => {
    e.preventDefault();
    dz.classList.remove("dragover");
    handler(e.dataTransfer.files);
  });
  fi.addEventListener("change", () => handler(fi.files));
}

// 单文件上传（I2V 图 / FL2V 首尾帧 / 编辑源视频）
async function uploadSingle(key, file) {
  if (!file) return;
  if (state[key]) { showError("已有文件，请先移除后再上传"); return; }
  const localUrl = URL.createObjectURL(file);
  state[key] = { name: file.name, path: "", localUrl };
  const chipBox = key === "i2vImage" ? "i2v-chip-box"
    : key === "fl2vFirst" ? "fl2v-first-chip-box"
    : key === "fl2vLast" ? "fl2v-last-chip-box"
    : "edit-chip-box";
  renderUploadChip(chipBox, file.name, "上传中…", () => removeSingle(key));
  try {
    const data = await uploadFile(file);
    state[key].path = data.path;
    renderUploadChip(chipBox, file.name, fmtSize(data.size ?? file.size), () => removeSingle(key));
  } catch (err) {
    removeSingle(key);
    showError(`文件上传失败：${err.message}`);
  }
}

function removeSingle(key) {
  if (state[key]?.localUrl?.startsWith("blob:")) URL.revokeObjectURL(state[key].localUrl);
  state[key] = null;
  const chipBox = key === "i2vImage" ? "i2v-chip-box"
    : key === "fl2vFirst" ? "fl2v-first-chip-box"
    : key === "fl2vLast" ? "fl2v-last-chip-box"
    : "edit-chip-box";
  $(chipBox).innerHTML = "";
  const fiId = key === "i2vImage" ? "i2v-file-input"
    : key === "fl2vFirst" ? "fl2v-first-input"
    : key === "fl2vLast" ? "fl2v-last-input"
    : "edit-file-input";
  $(fiId).value = "";
}

// Ref2V 多图上传（≤9 张，可多选批量）
async function uploadRef2vBatch(fileList) {
  const files = Array.from(fileList || []).filter((f) => f.type.startsWith("image/") || /\.(png|jpe?g|webp|gif|bmp)$/i.test(f.name));
  if (!files.length) return;
  if (state.ref2vImages.length + files.length > 9) {
    showError("参考图最多 9 张（超出部分忽略）");
  }
  for (const file of files.slice(0, 9 - state.ref2vImages.length)) {
    const localUrl = URL.createObjectURL(file);
    const entry = { name: file.name, path: "", localUrl };
    state.ref2vImages.push(entry);
    renderRef2vChips();
    try {
      const data = await uploadFile(file);
      entry.path = data.path;
      entry.size = data.size ?? file.size;
    } catch (err) {
      state.ref2vImages = state.ref2vImages.filter((e) => e !== entry);
      if (entry.localUrl.startsWith("blob:")) URL.revokeObjectURL(entry.localUrl);
      showError(`参考图 ${file.name} 上传失败：${err.message}`);
    }
    renderRef2vChips();
  }
}

function removeRef2vAt(idx) {
  const entry = state.ref2vImages[idx];
  if (!entry) return;
  if (entry.localUrl?.startsWith("blob:")) URL.revokeObjectURL(entry.localUrl);
  state.ref2vImages.splice(idx, 1);
  renderRef2vChips();
}

function renderRef2vChips() {
  const box = $("ref2v-chip-box");
  box.innerHTML = "";
  state.ref2vImages.forEach((entry, idx) => {
    const chip = document.createElement("span");
    chip.className = "audio-chip";
    const sizeText = entry.path ? fmtSize(entry.size ?? 0) : "上传中…";
    chip.innerHTML = `
      <span class="fc-name" title="${entry.name}">#${idx + 1} ${entry.name}</span>
      <span style="color:#1a7f37">${sizeText}</span>
      <button class="fc-rm" title="移除">×</button>`;
    chip.querySelector(".fc-rm").addEventListener("click", () => removeRef2vAt(idx));
    box.appendChild(chip);
  });
  $("ref2v-file-input").value = "";
}

// ============================================================
// Ref2VA 多模态参考：批量上传（图 ≤9 / 视频 ≤3 / 音频 ≤3）+ 场景自动判定
// ============================================================
// 三类参考的 UI 配置（列表 key / chip 容器 / 计数徽标 / 标签前缀 / 徽标颜色）
const REF2VA_ZONES = {
  ref2vaImages: { chipBox: "ref2va-img-chip-box", countId: "ref2va-img-count", max: 9, label: "Picture", color: "#2563eb" },
  ref2vaVideos: { chipBox: "ref2va-vid-chip-box", countId: "ref2va-vid-count", max: 3, label: "Video", color: "#9333ea" },
  ref2vaAudios: { chipBox: "ref2va-aud-chip-box", countId: "ref2va-aud-count", max: 3, label: "Audio", color: "#d97706" },
};

async function uploadRef2vaBatch(fileList, listKey, mimePrefix, nameRegex, maxCount) {
  const files = Array.from(fileList || []).filter(
    (f) => f.type.startsWith(mimePrefix) || nameRegex.test(f.name)
  );
  if (!files.length) return;
  const list = state[listKey];
  if (list.length + files.length > maxCount) {
    showError(`该类参考最多 ${maxCount} 个（超出部分忽略）`);
  }
  for (const file of files.slice(0, maxCount - list.length)) {
    const localUrl = URL.createObjectURL(file);
    const entry = { name: file.name, path: "", localUrl };
    list.push(entry);
    renderRef2vaChips();
    try {
      const data = await uploadFile(file);
      entry.path = data.path;
      entry.size = data.size ?? file.size;
    } catch (err) {
      state[listKey] = list.filter((e) => e !== entry);
      if (entry.localUrl.startsWith("blob:")) URL.revokeObjectURL(entry.localUrl);
      showError(`参考文件 ${file.name} 上传失败：${err.message}`);
    }
    renderRef2vaChips();
  }
  updateRef2vaSceneHint();
}

function removeRef2vaAt2(listKey, idx) {
  const list = state[listKey];
  const entry = list[idx];
  if (!entry) return;
  if (entry.localUrl?.startsWith("blob:")) URL.revokeObjectURL(entry.localUrl);
  list.splice(idx, 1);
  renderRef2vaChips();
  updateRef2vaSceneHint();
}

function renderRef2vaChips() {
  for (const [listKey, zone] of Object.entries(REF2VA_ZONES)) {
    const box = $(zone.chipBox);
    box.innerHTML = "";
    state[listKey].forEach((entry, idx) => {
      const chip = document.createElement("span");
      chip.className = "audio-chip";
      const sizeText = entry.path ? fmtSize(entry.size ?? 0) : "上传中…";
      chip.innerHTML = `
        <span class="fc-name" title="${entry.name}"><span class="chip-badge" style="border-color:${zone.color};color:${zone.color}">&lt;${zone.label} ${idx + 1}&gt;</span> ${entry.name}</span>
        <span style="color:#1a7f37">${sizeText}</span>
        <button class="fc-rm" title="移除">×</button>`;
      chip.querySelector(".fc-rm").addEventListener("click", () => removeRef2vaAt2(listKey, idx));
      box.appendChild(chip);
    });
    const countEl = $(zone.countId);
    countEl.textContent = `${state[listKey].length}/${zone.max}`;
    countEl.classList.toggle("full", state[listKey].length >= zone.max);
  }
}

// 场景自动判定（R1：纯音频禁用生成；图+音频 = 对口型）
function updateRef2vaSceneHint() {
  const hint = $("ref2va-scene-hint");
  if (state.mode !== "ref2va") { hint.style.display = "none"; return; }
  const nI = state.ref2vaImages.filter((e) => e.path).length;
  const nV = state.ref2vaVideos.filter((e) => e.path).length;
  const nA = state.ref2vaAudios.filter((e) => e.path).length;
  const nIAll = state.ref2vaImages.length, nVAll = state.ref2vaVideos.length, nAAll = state.ref2vaAudios.length;

  let text, cls, disable = false;
  if (!nIAll && !nVAll && !nAAll) {
    text = "请上传参考：图片提供身份，视频提供动作，音频提供说话内容与音色"; cls = "info";
  } else if (nIAll + nVAll + nAAll !== nI + nV + nA) {
    text = "⏳ 部分参考文件上传中，请稍候…"; cls = "info";
  } else if (!nI && !nV && nA) {
    text = "⛔ 音频参考不能单独使用，请至少配对 1 张图片或 1 个视频"; cls = "warn"; disable = true;
  } else if (!nI && !nV && !nA) {
    text = "请上传参考（至少 1 张图片或 1 个视频）"; cls = "info"; disable = true;
  } else if (nI && nV && nA) {
    text = `✨ 已识别为<b>全要素参考</b>：${nI} 图提供身份 + ${nV} 视频提供动作 + ${nA} 音频提供音色与内容`; cls = "info";
  } else if (nI && nV) {
    text = `✨ 已识别为<b>身份保持 + 动作迁移</b>：${nI} 图提供身份 + ${nV} 视频提供动作`; cls = "info";
  } else if (nI && nA) {
    text = `✨ 已识别为<b>对口型</b>场景：人物图提供身份，音频提供说话内容与音色（建议时长取音频时长）`; cls = "info";
  } else if (nV && nA) {
    text = `✨ 已识别为<b>动作参考 + 音色替换</b>：视频提供动作，音频提供新音色与内容`; cls = "info";
  } else if (nI) {
    text = `✨ 已识别为<b>多图参考</b>：${nI} 张图片（等价于多宫格场景）`; cls = "info";
  } else {
    text = `✨ 已识别为<b>视频参考</b>：${nV} 个视频（保留帧率与音轨）`; cls = "info";
  }
  hint.className = `scene-hint ${cls}`;
  hint.innerHTML = text;
  hint.style.display = "";
  // 生成按钮联动禁用（R1）
  $("btn-generate").disabled = disable;
}

// 对口型一键提示词模板（图 + 音频场景）
function applyLipsyncTemplate() {
  $("prompt-input").value =
    "画面中的人物用参考音频的声音自然地说出这段话，口型与语音精准对齐，表情自然，头部动作轻微。";
}

async function uploadFile(file) {
  const fd = new FormData();
  fd.append("file", file);
  const r = await fetch(`${API}/upload`, { method: "POST", body: fd });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

function renderUploadChip(boxId, name, sizeText, onRemove) {
  const box = $(boxId);
  box.innerHTML = `
    <span class="audio-chip">
      <span class="fc-name" title="${name}">📎 ${name}</span>
      <span style="color:#1a7f37">${sizeText}</span>
      <button class="fc-rm" title="移除">×</button>
    </span>`;
  box.querySelector(".fc-rm").addEventListener("click", onRemove);
}

function fmtSize(bytes) {
  if (bytes >= 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + " MB";
  return Math.round(bytes / 1024) + " KB";
}

// ============================================================
// 设置区：方案管理 + 变体双层过滤
// ============================================================
async function refreshAll() {
  await refreshStatus();
  await refreshCapabilities();
  await refreshSchemes();
}

async function refreshCapabilities() {
  try {
    const r = await fetch(`${API}/capabilities`);
    const cap = await r.json();
    state.variants = cap.variants || [];
    refreshVariantSelect();
    renderLoraSelect();
  } catch {
    $("mode-hint").textContent = "capabilities 获取失败（引擎未启动？）";
  }
}

// 双层交叉过滤（对应 ui.py cross_filter_variants）：
// 维度 1：当前 Tab → 范式（FL2VA / Ref2VA）
// 维度 2：当前方案 → applicable_variants 精度白名单
function filterVariants() {
  const paradigm = TAB_TO_PARADIGM[state.mode];
  const sid = $("scheme-select").value;
  let applicable = new Set();
  if (sid) {
    const scheme = state.schemes.find((s) => s.id === sid);
    if (scheme?.applicable_variants) applicable = new Set(scheme.applicable_variants);
  }
  return state.variants.filter((v) => {
    if (!v.available) return false;
    if (paradigm && v.base_model && v.base_model !== paradigm) return false;
    if (applicable.size && !applicable.has(v.name)) return false;
    return true;
  });
}

function refreshVariantSelect() {
  const sel = $("variant-select");
  const cur = sel.value;
  const filtered = filterVariants();
  sel.innerHTML = "";
  if (!filtered.length) {
    sel.innerHTML = `<option value="">该模式/方案暂无可用模型，请到引擎详情页下载</option>`;
    return;
  }
  for (const v of filtered) {
    const opt = document.createElement("option");
    opt.value = v.name;
    const resolution = v.resolution || "未知";
    const audioTag = v.supports_audio ? "含音频" : "无音频";
    opt.textContent = `[${v.base_model || "?"}] ${v.name} (${resolution}, ${v.precision || "?"}, ${audioTag})`;
    sel.appendChild(opt);
  }
  if (cur && filtered.some((v) => v.name === cur)) sel.value = cur;
  else sel.value = filtered[0].name;
}

async function refreshSchemes() {
  loadRecommended();   // v1.3：拉取工作流推荐步数表（异步，失败静默）
  try {
    const r = await fetch(`${API}/schemes`);
    const data = await r.json();
    state.schemes = data.schemes || [];
    state.defaultSchemeId = data.default_scheme_id || "";
    state.loadedSchemeId = data.loaded_scheme_id || "";
    const sel = $("scheme-select");
    const cur = sel.value;
    sel.innerHTML = "";
    for (const s of state.schemes) {
      const opt = document.createElement("option");
      opt.value = s.id;
      opt.textContent = `${s.id} - ${s.name}（${s.precision}）`;
      sel.appendChild(opt);
    }
    // 默认选中：已加载方案 > 默认方案 > 第一个
    const selected = state.loadedSchemeId || state.defaultSchemeId || (state.schemes[0]?.id ?? "");
    if (selected) sel.value = selected;
    else if (cur) sel.value = cur;
    updateSchemeDesc();
    // 已加载方案 → 回显联动；否则默认/首个方案 → 联动最佳参数（需求：切换方案步数自动跳）
    if (state.loadedSchemeId) applySchemeLinkage(state.loadedSchemeId, false);
    else applySchemeLinkage(sel.value, false);
  } catch {
    $("scheme-desc").textContent = "方案列表获取失败（引擎未启动？）";
  }
}

// 方案描述（对应 ui.py update_scheme_description）
function updateSchemeDesc() {
  const sid = $("scheme-select").value;
  const s = state.schemes.find((x) => x.id === sid);
  if (!s) { $("scheme-desc").textContent = "—"; return; }
  $("scheme-desc").textContent =
    `${s.name} · ${s.description} · 精度 ${s.precision} · 显存 ${s.vram_required} · offload ${s.offload_level}` +
    (state.loadedSchemeId === sid ? ` ·【当前已加载】` : "");
}

// 方案联动（对应 ui.py on_scheme_change_linkage）：
// 过滤变体 + offload/steps/加速方案默认值
function applySchemeLinkage(sid, setVariant = true) {
  const linkage = SCHEME_LINKAGE[sid?.toUpperCase()];
  if (!linkage) return;
  $("offload-select").value = linkage.offload;
  $("steps-range").value = linkage.steps;
  $("steps-val").textContent = linkage.steps;
  $("adv-sage").checked = linkage.sage;
  $("adv-lora").checked = linkage.lora;
  $("adv-fbc").checked = linkage.fbc;
  $("adv-blockswap").checked = linkage.blockswap;
  if (setVariant) refreshVariantSelect();
}

// v1.3：各生成方式推荐最佳步数表（来自引擎 GET /recommended，config.RECOMMENDED_STEPS）
// 切换工作流时自动跳到对应推荐值；用户手动改滑块后以滑块实值提交（手动优先）。
function applyRecommendedSteps(mode) {
  const rec = state.recommendedSteps?.[mode];
  if (!rec) return;
  $("steps-range").value = rec;
  $("steps-val").textContent = rec;
}
async function loadRecommended() {
  state.recommendedSteps = state.recommendedSteps || {};
  state.recommendedTable = state.recommendedTable || [];
  try {
    const r = await fetch(`${API}/recommended`);
    if (!r.ok) return;
    const j = await r.json();
    state.recommendedSteps = j.recommended_steps || {};
    state.recommendedTable = j.recommended_table || [];
  } catch { /* 引擎未启动时忽略，保留旧值 */ }
}

// 推荐步数参考表（来自 engine GET /recommended → state.recommendedTable）
// 数据源：config.RECOMMENDED_TABLE（单一来源，含模型归属/中文方式/步数/理由）
function recommendedStepsHelpTableHtml() {
  const heads = ["模型", "生成方式", "推荐步数", "理由"];
  const rows = (state.recommendedTable && state.recommendedTable.length
    ? state.recommendedTable
    : Object.keys(state.recommendedSteps || {}).map((mode) => ({
        model: "-", label: mode, steps: state.recommendedSteps[mode], reason: "-",
      }))
  ).map((row) => [
    row.model || "-",
    row.label || row.mode || "-",
    `<b>${row.steps ?? "-"}</b>`,
    row.reason || "-",
  ]);
  const extra = "<p style='margin:10px 0 0;font-size:12px;color:var(--muted, #888)'>"
    + "推荐值据实测基准与网上建议整理（4 步=仅连通链路必糊 / 8 步=清晰基线 ≈13min / 30 步=最佳画质 ≈50min）。"
    + "选择生成方式即自动跳到对应推荐值，手动改滑块后以手动值为准提交。</p>";
  return helpTableHtml(heads, rows) + extra;
}

async function saveDefaultScheme() {
  const sid = $("scheme-select").value;
  if (!sid) { showError("请先选择方案"); return; }
  try {
    const r = await fetch(`${API}/schemes/default`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ scheme_id: sid }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
    const data = await r.json();
    state.defaultSchemeId = sid;
    $("scheme-desc").textContent = `✓ ${data.message || `默认方案已设置为 ${sid}`}`;
  } catch (err) {
    showError(`保存默认方案失败：${err.message}`);
  }
}

// ============================================================
// 设置区：加载 / 卸载 / 高级设置 / LoRA
// ============================================================
function bindButtons() {
  $("btn-refresh-status").addEventListener("click", refreshAll);
  $("btn-scheme-save").addEventListener("click", saveDefaultScheme);
  $("scheme-select").addEventListener("change", () => {
    updateSchemeDesc();
    applySchemeLinkage($("scheme-select").value);
  });
  $("btn-load").addEventListener("click", loadModel);
  $("btn-unload").addEventListener("click", unloadModel);
  $("btn-apply-accel").addEventListener("click", applyAccelerationOverride);
  $("btn-lora-load").addEventListener("click", loraLoad);
  $("btn-lora-unload").addEventListener("click", loraUnload);
  $("btn-generate").addEventListener("click", generate);
  $("btn-lipsync-tpl").addEventListener("click", () => {
    applyLipsyncTemplate();
    $("prompt-input").focus();
  });
  // 问号帮助弹窗
  $("btn-help-scheme").addEventListener("click", () =>
    openHelpDialog("加载方案参数说明（什么方案用什么参数）", schemeHelpTableHtml()));
  $("btn-help-advanced").addEventListener("click", (e) => {
    e.preventDefault();   // 阻止 summary 默认行为，避免触发 details 折叠
    e.stopPropagation();
    openHelpDialog("高级设置参数说明（加速选项 / LoRA / Offload）", advancedHelpTableHtml());
  });
  $("btn-help-steps").addEventListener("click", () =>
    openHelpDialog("各生成方式推荐步数参考表", recommendedStepsHelpTableHtml()));
  $("btn-help-close").addEventListener("click", closeHelpDialog);
  $("btn-help-done").addEventListener("click", closeHelpDialog);
}

// /load 请求体（cpu_offload_level + 可选 acceleration_override）
function buildLoadBody(withOverride) {
  const body = {
    variant: $("variant-select").value || null,
    scheme_id: $("scheme-select").value || null,
    inputs: { cpu_offload_level: $("offload-select").value },
  };
  if (withOverride) {
    body.inputs.acceleration_override = {
      sage_attention: { enabled: $("adv-sage").checked },
      lora: $("adv-lora").checked
        ? { enabled: true, scale: parseFloat($("adv-lora-scale").value) }
        : null,
      first_block_cache: { enabled: $("adv-fbc").checked },
      block_swap: { enabled: $("adv-blockswap").checked },
    };
  }
  return body;
}

async function loadModel() {
  if (state.loading) return;
  const variant = $("variant-select").value;
  if (!variant) { showError("请先选择模型变体"); return; }
  const sid = $("scheme-select").value;
  // 后端按默认方案加载（_resolve_scheme_for_variant），先把选中的方案设为默认再加载
  if (sid) {
    try {
      await fetch(`${API}/schemes/default`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scheme_id: sid }),
      });
      state.defaultSchemeId = sid;
    } catch (err) {
      showError(`设置默认方案失败：${err.message}`);
    }
  }
  state.loading = true;
  $("btn-load").disabled = true;
  $("btn-load").textContent = "加载中…";
  try {
    await postLoad(buildLoadBody(false));
    await waitLoadComplete();  // /load fire-and-forget，轮询直到 ready
  } catch (err) {
    showError(`模型加载失败：${err.message}`);
  } finally {
    state.loading = false;
    $("btn-load").disabled = false;
    $("btn-load").textContent = "加载模型";
    refreshStatus();
    refreshSchemes();
  }
}

async function unloadModel() {
  try {
    const r = await fetch(`${API}/unload`, { method: "POST" });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
  } catch (err) {
    showError(`模型卸载失败：${err.message}`);
  } finally {
    refreshStatus();
    refreshSchemes();
  }
}

// 加速方案覆盖：unload → load（带 acceleration_override）（对应 ui.py apply_acceleration_override）
async function applyAccelerationOverride() {
  if (state.loading) return;
  const variant = $("variant-select").value;
  if (!variant) { showError("请先选择模型变体"); return; }
  state.loading = true;
  $("btn-apply-accel").disabled = true;
  $("accel-status").textContent = "正在卸载…";
  try {
    await fetch(`${API}/unload`, { method: "POST" });
    $("accel-status").textContent = "正在用新加速方案加载…";
    await postLoad(buildLoadBody(true));
    await waitLoadComplete();
    $("accel-status").textContent = "✓ 加速方案已应用";
  } catch (err) {
    $("accel-status").textContent = `失败: ${err.message}`;
    showError(`应用加速方案失败：${err.message}`);
  } finally {
    state.loading = false;
    $("btn-apply-accel").disabled = false;
    refreshStatus();
    refreshSchemes();
  }
}

// LoRA 下拉（capabilities.variants[].available_loras 或全量 loras 字段）
function renderLoraSelect() {
  const sel = $("lora-select");
  sel.innerHTML = "";
  const names = new Set();
  for (const v of state.variants) {
    for (const l of v.available_loras || v.loras || []) {
      const n = typeof l === "string" ? l : l.name;
      if (n) names.add(n);
    }
  }
  if (!names.size) {
    sel.innerHTML = `<option value="">（无可用 LoRA）</option>`;
    return;
  }
  for (const n of names) {
    const opt = document.createElement("option");
    opt.value = n;
    opt.textContent = n;
    sel.appendChild(opt);
  }
}

async function loraLoad() {
  const name = $("lora-select").value;
  if (!name) { showError("请先选择 LoRA"); return; }
  try {
    const r = await fetch(`${API}/lora/load`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name, weight: parseFloat($("lora-weight").value) }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
    $("lora-status").textContent = `✓ 已加载 ${name}`;
  } catch (err) {
    $("lora-status").textContent = `失败: ${err.message}`;
    showError(`LoRA 加载失败：${err.message}`);
  }
}

async function loraUnload() {
  const name = $("lora-select").value;
  if (!name) { showError("请先选择 LoRA"); return; }
  try {
    const r = await fetch(`${API}/lora/unload`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name }),
    });
    if (!r.ok) {
      const e = await r.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${r.status}`);
    }
    $("lora-status").textContent = `✓ 已卸载 ${name}`;
  } catch (err) {
    $("lora-status").textContent = `失败: ${err.message}`;
    showError(`LoRA 卸载失败：${err.message}`);
  }
}

// ============================================================
// 状态轮询
// ============================================================
async function refreshStatus() {
  if (state.generating) return null;
  try {
    const r = await fetch(`${API}/status`);
    const st = await r.json();
    const badge = $("state-badge");
    badge.className = "badge " + (st.state || "");
    badge.textContent = {
      idle: "空闲", loading: "加载中", ready: "就绪",
      inferring: "推理中", error: "异常",
    }[st.state] || st.state;
    const loaded = st.loaded_variant || "";
    const vram = st.extra?.vram_used || st.vram_used || "—";
    const offload = st.extra?.offload_level || "—";
    const dev = st.extra?.device_name || st.extra?.device || "—";
    $("state-detail").textContent =
      `${loaded || "未加载"} · offload ${offload} · 显存 ${vram} · ${dev}`;
    return st;
  } catch {
    $("state-badge").className = "badge";
    $("state-badge").textContent = "未连接";
    $("state-detail").textContent = "引擎不可达（127.0.0.1:7006）";
    return null;
  }
}

// ============================================================
// 结果区：生成（auto-load + SSE 流）
// ============================================================
async function generate() {
  if (state.generating) return;

  // ----- 1. 校验（对应 ui.py generate_video 的模式校验） -----
  const prompt = $("prompt-input").value.trim();
  if (!prompt) {
    showError(state.mode === "video_edit" ? "请输入编辑指令" : "请输入正向提示词");
    return;
  }
  if (state.mode === "i2v" && !state.i2vImage?.path) { showError("I2V 模式需要上传图片"); return; }
  if (state.mode === "fl2v") {
    if (!state.fl2vFirst?.path || !state.fl2vLast?.path) { showError("首尾帧模式需要上传首帧和尾帧图片"); return; }
  }
  if (state.mode === "ref2v") {
    if (!state.ref2vImages.length || state.ref2vImages.some((e) => !e.path)) {
      showError("多宫格模式需要至少 1 张已上传完成的参考图"); return;
    }
  }
  if (state.mode === "ref2va") {
    const nI = state.ref2vaImages.filter((e) => e.path).length;
    const nV = state.ref2vaVideos.filter((e) => e.path).length;
    const nA = state.ref2vaAudios.filter((e) => e.path).length;
    const nAll = state.ref2vaImages.length + state.ref2vaVideos.length + state.ref2vaAudios.length;
    if (nAll !== nI + nV + nA) { showError("部分参考文件仍在上传中，请稍候"); return; }
    if (!nI && !nV && !nA) { showError("多模态模式需要至少 1 个图片或视频参考"); return; }
    if (!nI && !nV && nA) { showError("音频参考不能单独使用，请至少配对 1 张图片或 1 个视频"); return; }
  }
  if (state.mode === "video_edit" && !state.editVideo?.path) { showError("视频编辑模式需要上传源视频"); return; }

  const variant = $("variant-select").value;
  if (!variant) { showError("请先选择模型变体"); return; }

  // ----- 2. auto-load（对齐 ui.py：未加载自动 /load，variant 不匹配自动切换） -----
  const st = await refreshStatus();
  if (st) {
    if (st.state === "error") {
      showError("引擎处于错误状态，请到引擎详情页查看错误详情或重启引擎后重试");
      return;
    }
    const loaded = st.loaded_variant || "";
    try {
      if (st.state !== "ready" || !loaded) {
        setProgress("正在加载模型…（视频模型较大，可能需要数分钟）", 0);
        await postLoad(buildLoadBody(false));
        await waitLoadComplete();
      } else if (loaded !== variant) {
        setProgress(`正在切换模型 ${loaded} → ${variant}…`, 0);
        await fetch(`${API}/unload`, { method: "POST" });
        await postLoad(buildLoadBody(false));
        await waitLoadComplete();
      }
    } catch (err) {
      showError(`模型加载/切换失败：${err.message}`);
      return;
    }
  }

  // ----- 3. 构造请求体（对应 ui.py generate_video L1781-1798） -----
  const [width, height] = $("resolution-select").value.split("x").map(Number);
  const inputs = {
    prompt,
    negative_prompt: $("negative-input").value.trim(),
    task_type: state.mode,
    duration: parseInt($("duration-range").value, 10),
    fps: parseInt($("fps-range").value, 10),
    width,
    height,
    num_inference_steps: parseInt($("steps-range").value, 10),
    guidance_scale: parseFloat($("guidance-range").value),
    seed: parseInt($("seed-input").value, 10) || -1,
    cpu_offload_level: $("offload-select").value,
  };
  const images = [];
  const videos = [];
  const audios = [];
  if (state.mode === "i2v") {
    images.push({ path: state.i2vImage.path, role: state.i2vRole });
  } else if (state.mode === "fl2v") {
    images.push({ path: state.fl2vFirst.path, role: "first_frame" });
    images.push({ path: state.fl2vLast.path, role: "last_frame" });
  } else if (state.mode === "ref2v") {
    state.ref2vImages.forEach((e, i) => images.push({ path: e.path, role: `reference_${i + 1}` }));
  } else if (state.mode === "ref2va") {
    // 多模态参考：图（≤9）→ 视频（≤3）→ 音频（≤3），按上传序传递。
    // 引擎侧固定该顺序构造 references 列表（prompt 标签 <Picture i>/<Video k>/<Audio j>
    // 编号与 rotary clock 排布由该顺序决定——"不同顺序就是不同请求"）
    state.ref2vaImages.forEach((e, i) => images.push({ path: e.path, role: `reference_${i + 1}` }));
    state.ref2vaVideos.forEach((e, i) => videos.push({ path: e.path, role: `reference_${i + 1}` }));
    state.ref2vaAudios.forEach((e, i) => audios.push({ path: e.path, role: `reference_${i + 1}` }));
  } else if (state.mode === "video_edit") {
    videos.push({ path: state.editVideo.path, role: "source_video" });
  }

  // 不携带 task_id：直连链路契约（引擎以 request.task_id 是否非空区分
  // gateway 转发/直连）。引擎会自动生成 task_id 上报任务中心全生命周期。
  const body = { variant, inputs, images, videos, audios };

  // ----- 4. SSE 流式推理 -----
  state.generating = true;
  $("btn-generate").disabled = true;
  $("btn-generate").textContent = "生成中…";
  $("progress-wrap").style.display = "flex";
  $("progress-fill").style.width = "0%";
  $("progress-text").textContent = "连接引擎…";
  $("error-box").style.display = "none";
  $("state-badge").className = "badge inferring";
  $("state-badge").textContent = "推理中";

  try {
    const resp = await fetch(`${API}/infer`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!resp.ok) {
      const e = await resp.json().catch(() => ({}));
      throw new Error(e.detail || `HTTP ${resp.status}`);
    }

    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    let finalResult = null;
    let finalTiming = null;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const lines = buf.split("\n");
      buf = lines.pop();  // 末行可能不完整，留在缓冲
      for (const line of lines) {
        if (!line.startsWith("data:")) continue;
        let frame;
        try { frame = JSON.parse(line.slice(5)); } catch { continue; }
        const r = handleFrame(frame);
        if (r) { finalResult = r.result; finalTiming = r.timing; }
      }
    }

    if (finalResult) {
      showResult(finalResult, finalTiming);
    } else {
      throw new Error("SSE 流结束但未收到结果帧");
    }
  } catch (err) {
    showError(`生成失败：${err.message}`);
  } finally {
    state.generating = false;
    $("btn-generate").disabled = false;
    $("btn-generate").textContent = "生成视频";
    $("progress-wrap").style.display = "none";
    refreshStatus();
  }
}

async function postLoad(body) {
  const r = await fetch(`${API}/load`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) {
    const e = await r.json().catch(() => ({}));
    throw new Error(e.detail || `HTTP ${r.status}`);
  }
  return r.json();
}

// /load fire-and-forget 轮询（对应 ui.py _wait_for_load_complete：
// 2s 间隔，600s 超时；ready 结束；error/中断抛错）
async function waitLoadComplete() {
  const maxWait = 600;
  const pollInterval = 2000;
  let waited = 0;
  while (waited < maxWait) {
    await new Promise((res) => setTimeout(res, pollInterval));
    waited += pollInterval / 1000;
    let st;
    try {
      const r = await fetch(`${API}/status`);
      st = await r.json();
    } catch { continue; }  // 轮询失败继续重试
    if (st.state === "ready") {
      setProgress(`模型加载完成（${waited}s），开始推理…`, 0);
      return;
    }
    if (st.state === "error") {
      throw new Error(`模型加载失败（${waited}s）：${st.error || st.error_message || "未知错误"}`);
    }
    if (st.state === "idle" || st.state === "stopped") {
      throw new Error(`模型加载被中断（${waited}s，state=${st.state}）`);
    }
    setProgress(`正在加载模型… ${waited}s`, 0);
  }
  throw new Error(`模型加载超时（${maxWait}s）`);
}

function setProgress(text, pct) {
  $("progress-wrap").style.display = "flex";
  $("progress-fill").style.width = `${pct}%`;
  $("progress-text").textContent = text;
}

// 处理一帧，返回最终结果（若有）
function handleFrame(f) {
  if (f.stage === "error" || f.stage === "video.error") {
    throw new Error(f.error?.message || "引擎错误");
  }
  const pct = Math.round((f.progress || 0) * 100);
  $("progress-fill").style.width = `${pct}%`;
  let stageText = STAGE_TEXT[f.stage] || f.stage;
  // 生成视频帧阶段带步数（video.generating 含 step/total_steps）
  if (f.stage === "video.generating" && f.total_steps) {
    stageText += ` ${f.step || 0}/${f.total_steps}`;
  }
  $("progress-text").textContent = `${stageText} ${pct}%`;
  if ((f.stage === "completed" || f.stage === "video.completed") && f.result) {
    return { result: f.result, timing: f.timing };
  }
  return null;
}

function showResult(result, timing) {
  const view = $("result-view");
  const t = Date.now();  // 防缓存
  // result: {video_url, audio_url, video_only_url?, duration, mode, offload_level, is_placeholder}
  const videoUrl = result.video_url ? result.video_url + "?t=" + t : "";
  const audioUrl = result.audio_url ? result.audio_url + "?t=" + t : "";
  const videoOnlyUrl = result.video_only_url ? result.video_only_url + "?t=" + t : "";

  let html = "";
  if (videoUrl) {
    html += `
      <div class="video-result">
        <span class="media-label">生成的视频（${MODE_CN[result.mode] || result.mode} 模式，含音频）</span>
        <video src="${videoUrl}" controls autoplay></video>
      </div>`;
  }
  if (audioUrl) {
    html += `
      <div class="audio-result">
        <span class="media-label">独立音频轨</span>
        <audio src="${audioUrl}" controls></audio>
      </div>`;
  }
  view.innerHTML = html || `<p class="placeholder">未收到视频产物</p>`;

  const chips = [
    `<span class="meta-chip">模式 ${MODE_CN[result.mode] || result.mode}</span>`,
    `<span class="meta-chip">时长 ${result.duration?.toFixed?.(1) ?? result.duration ?? "—"}s</span>`,
    `<span class="meta-chip">offload ${result.offload_level || $("offload-select").value}</span>`,
    `<span class="meta-chip">seed ${$("seed-input").value}</span>`,
    `<span class="meta-chip">分辨率 ${$("resolution-select").value}</span>`,
    `<span class="meta-chip">总耗时 ${timing?.total_seconds?.toFixed(1) ?? "—"}s</span>`,
  ];
  if (result.is_placeholder) chips.push(`<span class="meta-chip" style="color:var(--warning)">⚠️ placeholder 实现</span>`);
  if (videoUrl) chips.push(`<span class="meta-chip"><a href="${videoUrl}" download>下载视频</a></span>`);
  if (videoOnlyUrl) chips.push(`<span class="meta-chip"><a href="${videoOnlyUrl}" download>下载纯视频</a></span>`);
  if (audioUrl) chips.push(`<span class="meta-chip"><a href="${audioUrl}" download>下载音频</a></span>`);
  $("result-meta").innerHTML = chips.join("");
}

// ---------- 错误提示（不自动消失，手动关闭） ----------
function showError(msg) {
  $("error-text").textContent = msg;
  $("error-box").style.display = "block";
}
