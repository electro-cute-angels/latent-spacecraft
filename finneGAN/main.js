import * as THREE from 'three'
import { ImprovedNoise } from 'three/examples/jsm/math/ImprovedNoise.js'
import { EffectComposer } from 'three/examples/jsm/postprocessing/EffectComposer.js'
import { RenderPass } from 'three/examples/jsm/postprocessing/RenderPass.js'
import { UnrealBloomPass } from 'three/examples/jsm/postprocessing/UnrealBloomPass.js'
import { FilmPass } from 'three/examples/jsm/postprocessing/FilmPass.js'
import { AfterimagePass } from 'three/examples/jsm/postprocessing/AfterimagePass.js'
import { RenderPixelatedPass } from 'three/examples/jsm/postprocessing/RenderPixelatedPass'
import { LUTPass } from 'three/examples/jsm/postprocessing/LUTPass.js'
import { LUTCubeLoader } from 'three/examples/jsm/loaders/LUTCubeLoader.js'
import { ScanPixelPass } from './ScanPixelPass.js';
import { FontLoader } from 'three/examples/jsm/loaders/FontLoader.js'
import { TextGeometry } from 'three/examples/jsm/geometries/TextGeometry.js'



// RECAP: 5 CLOUD LAYERS >> 5 CONVOLUTIONAL LAYERS UPCONV1 >>> UPCONV5

// GANs GENERATION !!!!!!!!!!!!
//  EXAMPLE OF CATEGORICAL CODE, ONLY 0 OR 1! const code = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1];

const SERVER_URL = 'http://localhost:5000'; // or your server's IP/domain

// Generate audio for a specific layer from external project
async function generateLayer(layerName, categoricalCode, serverUrl = SERVER_URL) {
  const response = await fetch(`${serverUrl}/generate_layer_file`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      categorical_code: categoricalCode, // Array of 16 binary values [0,1,0,1,...]
      layer: layerName                   // e.g. "upconv1", "upconv2", "upconv3", "upconv4", "upconv5"
    })
  });

  if (!response.ok) {
    const err = await response.json();
    throw new Error(err.error || 'Generation failed');
  }

  const data = await response.json();
  console.log('[Transcription] generateLayer response', layerName, data)
  // data.file = "/static/batch/layer_upconv0_req_123456_abc.wav"
  
  // Convert relative path to full URL
  const fullAudioUrl = `${serverUrl}${data.file}`;
  return { url: fullAudioUrl, transcription: data.transcription || null };
}

const AUDIO_LAYER_NAMES = ['upconv1', 'upconv2', 'upconv3', 'upconv4', 'upconv5'];

const transcriptionPanel = document.getElementById('transcription-inner')

const TRANSCRIPTION_CFG = {
  windowSeconds: 0.001,
  threshold: 0.005,
  charIntervalMs: 20,
  simple: true,
  wordStretch: 2.0,
}

if (!window.__TranscriptionSettings) {
  window.__TranscriptionSettings = {
    get config() {
      return { ...TRANSCRIPTION_CFG }
    },
    setWindowSeconds(value) {
      if (typeof value === 'number' && Number.isFinite(value) && value > 0.01) {
        TRANSCRIPTION_CFG.windowSeconds = value
      }
    },
    setThreshold(value) {
      if (typeof value === 'number' && Number.isFinite(value) && value >= 0) {
        TRANSCRIPTION_CFG.threshold = value
      }
    },
    setCharInterval(ms) {
      if (typeof ms === 'number' && Number.isFinite(ms) && ms >= 0) {
        TRANSCRIPTION_CFG.charIntervalMs = ms
      }
    },
    setSimpleMode(flag) {
      TRANSCRIPTION_CFG.simple = Boolean(flag)
    },
    setWordStretch(value) {
      if (typeof value === 'number' && Number.isFinite(value) && value > 0) {
        TRANSCRIPTION_CFG.wordStretch = value
      }
    },
  }
}

function formatLatentVector(vector) {
  const values = Array.isArray(vector) && vector.length
    ? vector
    : ['?']
  return `[${values.map((value) => String(value)).join(', ')}]`
}

function buildActivityMap(audioBuffer, config = TRANSCRIPTION_CFG) {
  if (!audioBuffer) return [' ']

  const windowSeconds = Math.max(0.01, Number(config.windowSeconds) || TRANSCRIPTION_CFG.windowSeconds)
  const threshold = typeof config.threshold === 'number' ? config.threshold : TRANSCRIPTION_CFG.threshold

  const sampleRate = audioBuffer.sampleRate || 44100
  const windowSamples = Math.max(1, Math.floor(sampleRate * windowSeconds))
  const totalSamples = audioBuffer.length
  const channels = audioBuffer.numberOfChannels

  if (!totalSamples || !channels) return [' ']

  const timeline = []
  const channelData = new Array(channels)
  for (let c = 0; c < channels; c++) {
    channelData[c] = audioBuffer.getChannelData(c)
  }

  for (let start = 0; start < totalSamples; start += windowSamples) {
    const end = Math.min(totalSamples, start + windowSamples)
    let energy = 0
    let count = 0

    for (let c = 0; c < channels; c++) {
      const data = channelData[c]
      for (let i = start; i < end; i++) {
        const sample = data[i]
        energy += sample * sample
      }
      count += end - start
    }

    const rms = count > 0 ? Math.sqrt(energy / count) : 0
    //timeline.push(rms > threshold ? '⚫ ' : ' ') //⚫
  }

  if (!timeline.length) return [' ']

  return timeline
}

// Map word-level transcript timings into a spaced string aligned with analysis windows.
function buildTimedWordSequence(words, totalDuration, slotCount) {
  if (!Array.isArray(words) || !words.length) return ''

  const duration = typeof totalDuration === 'number' && totalDuration > 0
    ? totalDuration
    : null
  const slotSeconds = 0.04
  const NBSP = ' \u00A0 '

  let lastTimestamp = 0
  for (const entry of words) {
    const start = typeof entry?.start === 'number' ? Math.max(0, entry.start) : null
    const end = typeof entry?.end === 'number' ? Math.max(start ?? 0, entry.end) : null
    if (end !== null) {
      lastTimestamp = Math.max(lastTimestamp, end)
    } else if (start !== null) {
      lastTimestamp = Math.max(lastTimestamp, start)
    }
  }

  const timelineDuration = duration || (lastTimestamp || words.length * slotSeconds)
  const minSlots = Math.ceil(Math.max(timelineDuration, slotSeconds) / slotSeconds) + 4
  const requestedSlots = typeof slotCount === 'number' && slotCount > 0 ? slotCount : 0
  const totalSlots = Math.max(minSlots, requestedSlots)

  let result = ''
  let currentSlot = 0

  for (const entry of words) {
    const text = typeof entry?.text === 'string' ? entry.text.trim() : ''
    if (!text) continue

    const wordWidth = text.length + 1 // include trailing nbsp delimiter

    let targetSlot = currentSlot
    const start = typeof entry?.start === 'number' ? Math.max(0, entry.start) : null
    if (start !== null) {
      targetSlot = Math.round(start / slotSeconds)
    }

    targetSlot = Math.max(targetSlot, currentSlot)
    targetSlot = Math.min(targetSlot, Math.max(0, totalSlots - wordWidth))

    const gap = Math.max(0, targetSlot - currentSlot)
    if (gap > 0) {
      result += NBSP.repeat(gap)
      currentSlot += gap
    }

    result += text
    result += NBSP
    currentSlot = Math.min(totalSlots, targetSlot + wordWidth)
  }

  return result.replace(new RegExp(`${NBSP}+$`), '')
}

const textStreamRegistry = new WeakMap()

// Animate arbitrary character sequences so each segment reveals over time.
function streamCharacters(target, content, config = TRANSCRIPTION_CFG) {
  if (!target) return Promise.resolve()

  const previous = textStreamRegistry.get(target)
  if (previous?.timer) {
    clearTimeout(previous.timer)
  }

  const characters = Array.isArray(content)
    ? [...content]
    : String(content ?? '').split('')

  target.textContent = ''

  if (!characters.length) {
    textStreamRegistry.delete(target)
    return Promise.resolve()
  }

  const interval = Math.max(0, Number(config.charIntervalMs) || 0)

  return new Promise((resolve) => {
    const state = { timer: null }
    textStreamRegistry.set(target, state)

    const step = () => {
      target.textContent += characters.shift()

      if (!characters.length) {
        textStreamRegistry.delete(target)
        resolve()
        return
      }

      state.timer = window.setTimeout(step, interval)
    }

    step()
  })
}

function appendTranscriptionEntry(layerIndex, latentVector, audioBuffer, transcriptionData, options = {}) {
  if (!transcriptionPanel) return

  if (!transcriptionPanel.dataset.autogenerated) {
    transcriptionPanel.textContent = ''
    transcriptionPanel.dataset.autogenerated = 'true'
  }

  const entry = document.createElement('span')
  entry.className = 'transcription-entry'

  const latentSpan = document.createElement('span')
  latentSpan.className = 'latent-vector'

  const upconvSpan = document.createElement('span')
  upconvSpan.className = 'upconv'
  const layerLabel = AUDIO_LAYER_NAMES[layerIndex] ?? `layer-${layerIndex}`

  const activitySpan = document.createElement('span')
  activitySpan.className = 'transcription-activity'
  const simpleMode = options.simple ?? TRANSCRIPTION_CFG.simple
  const includeLatent = options.includeLatent !== false
  const arrowDirection = options.arrowDirection
  const timeline = !simpleMode ? buildActivityMap(audioBuffer) : null
  let activityContent = timeline || ' '

  if (simpleMode && layerLabel !== 'upconv5') {
    activitySpan.classList.add('transcription-arrow')
    activityContent = arrowDirection === 'down' ? '↓' : '↑'
  }

  if (layerLabel === 'upconv5') {
    const segmentWords = Array.isArray(transcriptionData?.segments)
      ? transcriptionData.segments.flatMap((segment) => Array.isArray(segment?.words) ? segment.words : [])
      : []
    const wordSource = Array.isArray(transcriptionData?.words) && transcriptionData.words.length
      ? transcriptionData.words
      : segmentWords
    const totalDuration = typeof transcriptionData?.duration === 'number'
      ? transcriptionData.duration
      : audioBuffer?.duration
    if (simpleMode) {
      activityContent = buildTimedWordSequence(wordSource, totalDuration, 60) || activityContent || ' '
    } else {
      const wordSequence = buildTimedWordSequence(wordSource, totalDuration, timeline?.length || 0)
      if (wordSequence) {
        activityContent = wordSequence
      }
    }
    if (simpleMode && (!activityContent || !String(activityContent).trim())) {
      activityContent = '⚫'
    }
  }

  if (includeLatent) {
    entry.appendChild(latentSpan)
  }
  entry.appendChild(upconvSpan)
  entry.appendChild(activitySpan)

  transcriptionPanel.appendChild(entry)
  transcriptionPanel.appendChild(document.createTextNode(' '))

  if (transcriptionMover?.sync) {
    transcriptionMover.sync()
  }

  const animate = async () => {
    try {
      if (includeLatent) {
        await streamCharacters(latentSpan, formatLatentVector(latentVector))
      }
      await streamCharacters(upconvSpan, ` ${String.fromCharCode(0x2021)}${layerLabel}`)
      const timelineChars = Array.isArray(activityContent)
        ? [' ', ...activityContent]
        : ` ${activityContent}`
      await streamCharacters(activitySpan, timelineChars)
    } finally {
      if (transcriptionMover?.sync) {
        transcriptionMover.sync()
      }
    }
  }

  animate().catch((err) => console.warn('Transcription animation failed', err))
}

const audioManager = (() => {
  let audioCtx = null;
  let masterGain = null;
  const fadeSeconds = 1.8;
  let geometry = [];
  let activeLayerIndex = -1;
  let traversalDirection = 0;
  let traversalCode = null;
  let traversalId = 0;
  let traversalRender = null;
  let firstAscentPending = true;

  const layerState = AUDIO_LAYER_NAMES.map(() => ({
    code: null,
    buffer: null,
    source: null,
    gainNode: null,
    loading: null,
    stopTimer: null,
    targetGain: 0,
    entryRendered: false,
    transcription: null,
    pendingSimple: null,
  }));

  function generateRandomCode() {
    return Array.from({ length: 16 }, () => (Math.random() < 0.5 ? 0 : 1));
  }

  function currentDirectionLabel(direction = traversalDirection) {
    if (direction > 0) return 'ascent';
    if (direction < 0) return 'descent';
    return 'idle';
  }

  function beginTraversal(directionHint) {
    const newDirection = directionHint || traversalDirection || 1;
    traversalDirection = newDirection;
    traversalId += 1;
    traversalCode = generateRandomCode();
    traversalRender = { latentRendered: false };
    console.log('[Transcription] traversal start', traversalId, currentDirectionLabel(newDirection), 'code', JSON.stringify(traversalCode));

    layerState.forEach((state) => {
      if (state.stopTimer) {
        clearTimeout(state.stopTimer);
        state.stopTimer = null;
      }
      if (state.source) {
        try { state.source.stop(); } catch (err) {}
        state.source.disconnect();
        state.source = null;
      }
      state.buffer = null;
      state.code = null;
      state.transcription = null;
      state.entryRendered = false;
      state.loading = null;
      state.pendingSimple = null;
    });
  }

  function ensureTraversal(directionHint) {
    if (!traversalCode) {
      beginTraversal(directionHint);
    } else if (!traversalRender) {
      traversalRender = { latentRendered: false };
    }
  }

  function logSimpleVisit(index, directionSign) {
    if (!TRANSCRIPTION_CFG.simple) return;
    const state = layerState[index];
    if (!state) return;

    if (!traversalRender) {
      traversalRender = { latentRendered: false };
    }

    const arrowDir = directionSign < 0 ? 'down' : 'up';
    if (!state.buffer) {
      state.pendingSimple = { direction: directionSign };
      return;
    }

    const includeLatent = !traversalRender.latentRendered;
    appendTranscriptionEntry(index, state.code || traversalCode || [], state.buffer, state.transcription, {
      simple: true,
      includeLatent,
      arrowDirection: arrowDir,
    });
    state.pendingSimple = null;
    if (includeLatent) {
      traversalRender.latentRendered = true;
    }
  }

  async function requestLayer(index) {
    const ctx = ensureContext();
    ensureTraversal(traversalDirection || 1);

    const state = layerState[index];
    if (state.buffer && state.code === traversalCode) {
      return state.loading?.promise || Promise.resolve(state.buffer);
    }

    if (state.loading && state.loading.traversalId === traversalId) {
      return state.loading.promise;
    }

    state.code = traversalCode;
    const requestTraversalId = traversalId;
    const layerName = AUDIO_LAYER_NAMES[index];

    console.log('[Transcription] request layer', layerName, 'traversal', requestTraversalId, currentDirectionLabel(), 'code', JSON.stringify(state.code));

    const promise = (async () => {
      try {
        const { url, transcription } = await generateLayer(layerName, state.code);
        if (requestTraversalId !== traversalId) return;

        state.transcription = transcription || null;
        const res = await fetch(url);
        const arrayBuffer = await res.arrayBuffer();
        if (requestTraversalId !== traversalId) return;

        const audioBuffer = await ctx.decodeAudioData(arrayBuffer);
        if (requestTraversalId !== traversalId) return;

        state.buffer = audioBuffer;
        startLayerPlayback(index);
        if (requestTraversalId === traversalId) {
          if (TRANSCRIPTION_CFG.simple) {
            const requestedDirection = state.pendingSimple?.direction ?? (traversalDirection < 0 ? -1 : 1);
            logSimpleVisit(index, requestedDirection);
          } else if (!state.entryRendered) {
            appendTranscriptionEntry(index, state.code || [], audioBuffer, state.transcription);
            state.entryRendered = true;
          }
        }
      } catch (err) {
        if (requestTraversalId === traversalId) {
          console.warn(`Audio layer ${layerName} failed`, err);
          state.code = null;
          state.transcription = null;
        }
      } finally {
        if (state.loading && state.loading.traversalId === requestTraversalId) {
          state.loading = null;
        }
      }
    })();

    state.loading = { traversalId: requestTraversalId, promise };
    return promise;
  }

  function ensureContext() {
    if (!audioCtx) {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
      masterGain = audioCtx.createGain();
      masterGain.gain.value = 0.9;
      masterGain.connect(audioCtx.destination);
    }
    return audioCtx;
  }

  function startLayerPlayback(index) {
    if (!audioCtx || audioCtx.state === 'closed') return;
    const state = layerState[index];
    if (!state.buffer) return;

    if (state.source) {
      try { state.source.stop(); } catch (err) {}
      state.source.disconnect();
      state.source = null;
    }

    const source = audioCtx.createBufferSource();
    source.buffer = state.buffer;
    source.loop = true;

    const gain = state.gainNode || audioCtx.createGain();
    if (!state.gainNode) {
      gain.gain.value = 0;
      gain.connect(masterGain);
      state.gainNode = gain;
    }

    source.connect(gain);
    source.start();
    source.onended = () => {
      if (state.source === source) state.source = null;
    };

    state.source = source;
    if (state.stopTimer) {
      clearTimeout(state.stopTimer);
      state.stopTimer = null;
    }
  }

  function setLayerTarget(index, targetGain) {
    if (!audioCtx) return;
    const state = layerState[index];
    if (!state.gainNode) return;

    state.targetGain = targetGain;

    const now = audioCtx.currentTime;
    const currentValue = state.gainNode.gain.value;
    state.gainNode.gain.cancelScheduledValues(now);
    state.gainNode.gain.setValueAtTime(currentValue, now);

    const rampDuration = targetGain < currentValue ? Math.min(fadeSeconds, 1.1) : fadeSeconds;
    state.gainNode.gain.linearRampToValueAtTime(targetGain, now + rampDuration);

    if (targetGain <= 0.001 && state.source) {
      if (state.stopTimer) {
        clearTimeout(state.stopTimer);
      }
      state.stopTimer = setTimeout(() => {
        if (state.source && state.targetGain <= 0.001) {
          try { state.source.stop(); } catch (err) {}
          state.source.disconnect();
          state.source = null;
        }
        state.stopTimer = null;
      }, Math.ceil((rampDuration + 0.2) * 1000));
    } else if (state.stopTimer) {
      clearTimeout(state.stopTimer);
      state.stopTimer = null;
    }
  }

  function determineLayerIndex(cameraY) {
    if (!geometry.length) return -1;

    // Prefer current layer if still within bounds to prevent jitter at borders
    if (activeLayerIndex >= 0 && activeLayerIndex < geometry.length) {
      const current = geometry[activeLayerIndex];
      const margin = current.halfHeight * 0.15;
      const bottom = current.center - current.halfHeight - margin;
      const top = current.center + current.halfHeight + margin;
      if (cameraY >= bottom && cameraY <= top) {
        return activeLayerIndex;
      }
    }

    for (let i = 0; i < geometry.length; i++) {
      const layer = geometry[i];
      if (cameraY >= layer.center - layer.halfHeight && cameraY <= layer.center + layer.halfHeight) {
        return i;
      }
    }

    // Fallback to nearest layer center when outside explicit bounds
    let closest = 0;
    let closestDist = Math.abs(cameraY - geometry[0].center);
    for (let i = 1; i < geometry.length; i++) {
      const dist = Math.abs(cameraY - geometry[i].center);
      if (dist < closestDist) {
        closest = i;
        closestDist = dist;
      }
    }
    return closest;
  }

  function update(cameraY) {
    if (!audioCtx || audioCtx.state !== 'running') return;
    if (!geometry.length) return;

    const nextIndex = determineLayerIndex(cameraY);
    if (nextIndex === -1) return;

    if (nextIndex !== activeLayerIndex) {
      const prevIndex = activeLayerIndex;
      const topIndex = geometry.length - 1;
      const bottomIndex = 0;
      const direction = prevIndex === -1 ? 0 : Math.sign(nextIndex - prevIndex);

      if (prevIndex === -1 && !traversalCode) {
        const hint = direction || traversalDirection || 1;
        ensureTraversal(hint);
        if (direction !== 0) {
          traversalDirection = direction;
        }
      } else if (prevIndex === topIndex && direction < 0) {
        beginTraversal(-1);
      } else if (prevIndex === bottomIndex && direction > 0) {
        if (firstAscentPending) {
          firstAscentPending = false;
          traversalDirection = direction || 1;
        } else {
          beginTraversal(1);
        }
      } else if (direction !== 0 && direction !== traversalDirection) {
        traversalDirection = direction;
      }

      activeLayerIndex = nextIndex;
      if (TRANSCRIPTION_CFG.simple) {
        const visitDirection = direction !== 0
          ? direction
          : (traversalDirection === 0 ? 1 : traversalDirection);
        logSimpleVisit(activeLayerIndex, visitDirection);
      }
      const state = layerState[activeLayerIndex];
      const hasActiveLoading = state.loading && state.loading.traversalId === traversalId;
      if ((!state.buffer || state.code !== traversalCode) && !hasActiveLoading) {
        state.buffer = state.code === traversalCode ? state.buffer : null;
        requestLayer(activeLayerIndex);
      }
    }

    layerState.forEach((state, index) => {
      const isActive = index === activeLayerIndex;
      const hasActiveLoading = state.loading && state.loading.traversalId === traversalId;
      if (state.code !== traversalCode && state.buffer) {
        if (state.source) {
          try { state.source.stop(); } catch (err) {}
          state.source.disconnect();
          state.source = null;
        }
        state.buffer = null;
        state.pendingSimple = null;
      }
      if (isActive && (!state.buffer || state.code !== traversalCode) && !hasActiveLoading) {
        requestLayer(index);
      }
      if (state.buffer && !state.source) {
        startLayerPlayback(index);
      }
      if (state.gainNode) {
        setLayerTarget(index, isActive ? 1 : 0);
      }
    });
  }

  function resume() {
    const ctx = ensureContext();
    if (ctx.state === 'suspended') {
      return ctx.resume();
    }
    return Promise.resolve();
  }

  function setLayerGeometry(data) {
    geometry = data.slice().sort((a, b) => a.center - b.center);
    layerState.forEach((state) => {
      if (state.stopTimer) {
        clearTimeout(state.stopTimer);
        state.stopTimer = null;
      }
      if (state.source) {
        try { state.source.stop(); } catch (err) {}
        state.source.disconnect();
        state.source = null;
      }
      state.buffer = null;
      state.code = null;
      state.loading = null;
      state.entryRendered = false;
      state.transcription = null;
      state.pendingSimple = null;
    });
    traversalId += 1;
    traversalCode = null;
    traversalDirection = 0;
    traversalRender = null;
    firstAscentPending = true;
    activeLayerIndex = -1;
  }

  return { resume, update, setLayerGeometry };
})();

/*async function getUpconv3Audio() {
  const code = [0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1];
  
  try {
  const { url: audioUrl } = await generateLayer('upconv5', code, 'http://localhost:5000');
  console.log('Audio URL:', audioUrl);
    
    // Play in browser
    const audio = new Audio(audioUrl);
    audio.play();
    
    // Or download the file
  const response = await fetch(audioUrl);
    const blob = await response.blob();
    const downloadUrl = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = downloadUrl;
    a.download = 'upconv3_audio.wav';
    //a.click();
    
    return audioUrl;
  } catch (error) {
    console.error('Error generating layer:', error);
  }
}
getUpconv3Audio();*/





let geoCursorMesh;
let geoCursorMesh2;




// ============== RENDERER / SCENE / CAMERA ==============
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true})
const PIXEL_RATIO = 1
//renderer.domElement.style.cssText = 'filter: blur(0px);';
renderer.setPixelRatio(Math.min(window.devicePixelRatio, PIXEL_RATIO))
renderer.setSize(window.innerWidth, window.innerHeight)
renderer.outputColorSpace = THREE.SRGBColorSpace
renderer.setAnimationLoop(animate)
document.getElementById('app').appendChild(renderer.domElement)
renderer.domElement.style = "position:fixed; left:0; top:0; z-index:1; mix-blend-mode:exclusion; filter: blur(0px)";


if (!renderer.capabilities.isWebGL2) { alert('Needs WebGL2 (Data3DTexture).'); throw new Error('WebGL2 req') }

const scene = new THREE.Scene()
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 400)
// Camera only moves on Y. Fixed X/Z so FOV/distance stay constant (no stretching).
camera.position.set(0, 0, 6)

// Simple sky dome
{
  const canvas = document.createElement('canvas'); canvas.width = 1; canvas.height = 64
  const ctx = canvas.getContext('2d')
  const g = ctx.createLinearGradient(0,0,0,64)
  g.addColorStop(0.0, '#0b2b48'); g.addColorStop(0.5, '#16588c'); g.addColorStop(1.0, '#5f8fbf')
  ctx.fillStyle = g; ctx.fillRect(0,0,1,64)
  const skyMap = new THREE.CanvasTexture(canvas); skyMap.colorSpace = THREE.SRGBColorSpace
  scene.add(new THREE.Mesh(new THREE.SphereGeometry(80,32,16),
    new THREE.MeshBasicMaterial({ map: skyMap, side: THREE.BackSide })))
}

// ================== 3D NOISE ==================
const VOLUME_SIZE = 160 // lean & fast; raise later if you want
const volumeTexture = (() => {
  const s = VOLUME_SIZE, data = new Uint8Array(s*s*s)
  const perlin = new ImprovedNoise(), v = new THREE.Vector3(), scale = 0.05
  let i = 0
  for (let z=0; z<s; z++) for (let y=0; y<s; y++) for (let x=0; x<s; x++) {
    const d = 1.0 - v.set(x,y,z).subScalar(s/2).divideScalar(s).length()
    const n = perlin.noise(x*scale/1.5, y*scale, z*scale/1.5)
    data[i++] = (128 + 128*n) * d * d
  }
  const tex = new THREE.Data3DTexture(data, s, s, s)
  tex.format = THREE.RedFormat; tex.minFilter = THREE.LinearFilter; tex.magFilter = THREE.LinearFilter
  tex.unpackAlignment = 1; tex.needsUpdate = true
  return tex
})()

// ================== SHADER (single-sample march, fast) ==================
const VERT = /* glsl */`
in vec3 position;
uniform mat4 modelMatrix, modelViewMatrix, projectionMatrix;
uniform vec3 cameraPos;
out vec3 vOrigin;
out vec3 vDirection;
void main(){
  vec4 mv = modelViewMatrix * vec4(position, 1.0);
  vOrigin = (inverse(modelMatrix) * vec4(cameraPos, 1.0)).xyz;
  vDirection = position - vOrigin;
  gl_Position = projectionMatrix * mv;
}
`
const FRAG = /* glsl */`
precision highp float;
precision highp sampler3D;

in vec3 vOrigin;
in vec3 vDirection;
out vec4 color;

uniform sampler3D map;
uniform vec3 base;
uniform float threshold, range, opacity, frame;

uniform float uGain;
uniform vec3  uTint;
uniform vec3  uOffset;     // wind (unbounded; we tile with fract)
uniform float uMinSteps, uMaxSteps, uStepDensity, uNearFade, uLOD;

// NEW:
uniform float uFeather;       // 0..0.5 (box space)
uniform float uEdgeNoiseAmp;  // 0..~0.2
uniform float uEdgeNoiseFreq; // e.g. 3.0

uint wang_hash(uint seed){
  seed = (seed ^ 61u) ^ (seed >> 16u);
  seed *= 9u; seed = seed ^ (seed >> 4u);
  seed *= 0x27d4eb2du; seed = seed ^ (seed >> 15u);
  return seed;
}
float randomFloat(inout uint seed){ return float(wang_hash(seed)) / 4294967296.; }

vec2 hitBox(vec3 o, vec3 d){
  vec3 bmin = vec3(-0.5), bmax = vec3(0.5);
  vec3 inv = 1.0 / d;
  vec3 t0s = (bmin - o) * inv, t1s = (bmax - o) * inv;
  vec3 tsm = min(t0s, t1s), tbg = max(t0s, t1s);
  float t0 = max(tsm.x, max(tsm.y, tsm.z));
  float t1 = min(tbg.x, min(tbg.y, tbg.z));
  return vec2(t0, t1);
}

float sampleVolume(vec3 q){ return texture(map, fract(q)).r; }

vec4 linearToSRGB(vec4 v){
  vec3 a = pow(v.rgb, vec3(0.41666)) * 1.055 - vec3(0.055);
  vec3 b = v.rgb * 12.92;
  vec3 m = mix(a, b, step(v.rgb, vec3(0.0031308)));
  return vec4(m, v.a);
}

// --- NEW: edge feather factor (1.0 middle, fades to 0 near caps) ---
float edgeFeather(vec3 objP, vec3 qTex){
  // objP is in box space: [-0.5, 0.5] on each axis
  float halfH = 0.5;
  float fromCap = halfH - abs(objP.y); // distance to closest cap in box space
  float f = smoothstep(0.0, uFeather, fromCap);

  // Add a little noise to avoid a perfectly flat transition band
  float n = sampleVolume(qTex * uEdgeNoiseFreq) - 0.5;
  f = smoothstep(0.0, uFeather, fromCap + n * uEdgeNoiseAmp);

  return clamp(f, 0.0, 1.0);
}

void main(){
  vec3 rd = normalize(vDirection);
  vec2 bounds = hitBox(vOrigin, rd);
  if(bounds.x > bounds.y) discard;

  bounds.x = max(bounds.x, 0.002);
  float travel = max(0.0, bounds.y - bounds.x);
  if(travel <= 0.0005) discard;

  float N = clamp(travel * uStepDensity * mix(0.6, 1.0, uLOD), uMinSteps, uMaxSteps);
  float delta = travel / N;

  uint seed = uint(gl_FragCoord.x)*1973u + uint(gl_FragCoord.y)*9277u + uint(frame)*26699u;
  vec3 size = vec3(textureSize(map, 0));

  vec3 p = vOrigin + bounds.x * rd;
  p += rd * (randomFloat(seed)*2.0 - 1.0) * (1.0/size);

  vec4 ac = vec4(base, 0.0);
  float safety = smoothstep(uNearFade, uNearFade*2.0, travel);

  for(float i=0.0; i<1024.0; i++){
    if(i >= N) break;

    vec3 q = p + 0.5 + uOffset; // texture space (plus wind)
    float d = sampleVolume(q);
    d = smoothstep(threshold - range, threshold + range, d) * opacity * safety;

    // --- NEW: multiply by edge feather ---
    float fEdge = edgeFeather(p, q);
    d *= fEdge;

    float ton = 0.35 + 0.65 * d;

    d *= uGain;
    ac.rgb += (1.0 - ac.a) * d * (ton * uTint);
    ac.a   += (1.0 - ac.a) * d;

    if(ac.a >= 0.985) break;
    p += rd * delta;
  }

  color = linearToSRGB(ac);
  if(color.a == 0.0) discard;
}
`

function makeCloudMaterial(tex) {
  return new THREE.RawShaderMaterial({
    glslVersion: THREE.GLSL3,
    uniforms: {
      base:        { value: new THREE.Color(0x7d8ea4) },
      map:         { value: tex },
      cameraPos:   { value: new THREE.Vector3() },
      threshold:   { value: 0.25 },
      opacity:     { value: 0.24 },
      range:       { value: 0.10 },
      frame:       { value: 0.0 },

      uGain:       { value: 1.0 },
      uTint:       { value: new THREE.Color(0xffffff) },
      uOffset:     { value: new THREE.Vector3(0,0,0) },

      uFeather:       { value: 0.10 }, // 0..0.5 in box-space (0.18 ≈ 36% of half-height)
      uEdgeNoiseAmp:  { value: 0.06 }, // how much noise perturbs the feather
      uEdgeNoiseFreq: { value: 3.0 },

      // adaptive marching
      uMinSteps:   { value: 28.0 },
      uMaxSteps:   { value: 68.0 },
      uStepDensity:{ value: 34.0 },
      uNearFade:   { value: 0.10 },
      uLOD:        { value: 1.0 }
    },
    vertexShader: VERT,
    fragmentShader: FRAG,
    side: THREE.BackSide,
    transparent: true,
    depthWrite: false
  })
}

// ================== LAYERS ==================
function makeLayer(yCenter, thickness, gain, tint, thr=0.25, op=0.24, rng=0.10) {
  const geo = new THREE.BoxGeometry(1,1,1)
  const mat = makeCloudMaterial(volumeTexture)
  mat.uniforms.uGain.value = gain
  mat.uniforms.uTint.value = new THREE.Color(tint)
  mat.uniforms.threshold.value = thr
  mat.uniforms.opacity.value   = op
  mat.uniforms.range.value     = rng

  const mesh = new THREE.Mesh(geo, mat)
  mesh.position.set(0, yCenter, 0) // fixed Z=0 (distance to camera is constant)
  mesh.scale.set(1, Math.max(0.001, thickness), 1)
  mesh.userData.thickness = mesh.scale.y
  return mesh
}

function scaleLayerToViewport(mesh) {
  // Face camera (but camera rotation is fixed; this is cheap)
  mesh.quaternion.copy(camera.quaternion)
  // Constant frustum width: camera Z and FOV never change
  const dist = Math.abs(camera.position.z - mesh.position.z) // fixed ~6
  const halfH = Math.tan(THREE.MathUtils.degToRad(camera.fov * 0.5)) * dist
  const halfW = halfH * camera.aspect
  const width = (halfW * 2) * 1.2 //HORIZONTAL DIMENSION OF THE STACK
  mesh.scale.x = width
  mesh.scale.z = width
  mesh.scale.y = mesh.userData.thickness
}

const layers = []
const world = {
  totalHeight: 60.0,  // TOTAL HEIGHT
  gap: 0,          // small spacing between slabs
  wind: new THREE.Vector3(0.008, 0.00, 0.008) // gentle drift
}
function rebuildLayers() {
  layers.forEach(m => scene.remove(m)); layers.length = 0
  const N = 5
  const slice = world.totalHeight / N
  const thickness = Math.max(0.001, slice - world.gap)
  const y0 = -world.totalHeight * 0.5 + slice * 0.5

  const gains = [1.50, 1.12, 0.80, 0.80, 0.32]
  const tints = ['#2e2e2eff', '#e8f2ff', '#d9ecff', '#cbe6ff', '#bfe2ff']
  const thr   = [0.15, 0.20, 0.26, 0.34, 0.44]
  const opac  = [0.03, 0.07, 0.13, 0.18, 0.0]
  const rng   = [0.12, 0.11, 0.10, 0.09, 0.08]

  for (let i=0; i<N; i++){
    const y = y0 + i * slice
    const layer = makeLayer(y, thickness, gains[i], tints[i], thr[i], opac[i], rng[i])
    // gentle per-layer LOD: top layers cheaper
    const lod = 1.0 - (i/(N-1))*0.35
    layer.material.uniforms.uLOD.value = lod
    scene.add(layer); layers.push(layer)
  }

  // Start **at the bottom (densest)**
  const bounds = getVerticalBounds()
  camera.position.y = bounds.min + 0.2

  audioManager.setLayerGeometry(layers.map((layer) => ({
    center: layer.position.y,
    halfHeight: layer.scale.y * 0.5
  })))
}
rebuildLayers()

function getVerticalBounds() {
  const half = world.totalHeight * 0.5
  // small margins so edges aren’t visible
  return { min: -half*0.9, max: half*0.98 }
}

// ================== POST FX ==================
const composer = new EffectComposer(renderer)
composer.addPass(new RenderPass(scene, camera))
const bloomPass = new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.1, 0.2, 0.12)
//composer.addPass(bloomPass)
composer.addPass(new RenderPixelatedPass(4, scene, camera));

const filmPass = new FilmPass(0.05, true);
composer.addPass(filmPass);

//composer.addPass(new AfterimagePass(0.99));


const lutLoader = new LUTCubeLoader();
lutLoader.load('/LUTs/Thermal Monochrome 06 BMD Film ACIDBITE.cube', (result) => {
    const lutTexture = result.texture3D;
    const lutPass = new LUTPass({ lut: lutTexture, intensity: 1 });
    composer.addPass(lutPass);
});

// ============== OVERLAYED SCENE: ConcatenationGAN Scene A replica ==============
// An orthographic scene with post-processing/LUTs/meshes, rendered on top
// without disturbing Finnegan's main scene.
{
  const frustumSizeA = 30;
  let aspectA = window.innerWidth / window.innerHeight;

  const sceneA = new THREE.Scene();
  sceneA.background = new THREE.Color(0x383838);

  const cameraA = new THREE.OrthographicCamera(
    (frustumSizeA * aspectA) / -2,
    (frustumSizeA * aspectA) / 2,
    frustumSizeA / 2,
    frustumSizeA / -2,
    0.1,
    1000
  );
  cameraA.position.set(0, 20, 0);
  cameraA.lookAt(0, 0, 0);

  const rendererA = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  rendererA.setSize(window.innerWidth, window.innerHeight);
  rendererA.domElement.style.cssText = 'position:fixed; left:0; top:0; z-index:9; mix-blend-mode:difference; pointer-events:none;';
  document.body.appendChild(rendererA.domElement);

  const composerA = new EffectComposer(rendererA);
  composerA.addPass(new RenderPass(sceneA, cameraA));
  composerA.addPass(new FilmPass(100, 0.01, 1000, false));
  composerA.addPass(new RenderPixelatedPass(4, sceneA, cameraA));
  composerA.addPass(new FilmPass(0.2, 0, 648, false));
  composerA.addPass(new AfterimagePass(0.998));
  composerA.addPass(new FilmPass(0.4, 0.1, 1, false));
  // LUT on top scene (mirroring concatenationgan-new Scene A usage)
  const lutLoaderA = new LUTCubeLoader();
  lutLoaderA.load('/LUTs/Thermal Monochrome 06 BMD Film ACIDBITE.cube', (result) => {
    const lutTexture = result.texture3D;
    const lutPass = new LUTPass({ lut: lutTexture, intensity: 1 });
    composerA.addPass(lutPass);
  });

  // === Math Scene overlay (Scene B clone) ===
  const sceneMath = new THREE.Scene();
  let aspectMath = aspectA;
  const cameraMath = new THREE.OrthographicCamera(
    (frustumSizeA * aspectMath) / -2,
    (frustumSizeA * aspectMath) / 2,
    frustumSizeA / 2,
    frustumSizeA / -2,
    0.1,
    1000
  );
  cameraMath.position.set(0, 20, 0);
  cameraMath.lookAt(0, 0, 0);

  //sceneMath.background = new THREE.Color(0XFFFFFF);

  const rendererMath = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  rendererMath.setSize(window.innerWidth, window.innerHeight);
  rendererMath.domElement.style.cssText = 'position:fixed; left:0; top:0; z-index:2; mix-blend-mode:difference; pointer-events:none; filter:invert(0)';
  document.body.appendChild(rendererMath.domElement);

  const composerMath = new EffectComposer(rendererMath);
  composerMath.addPass(new RenderPass(sceneMath, cameraMath));
  composerMath.addPass(new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 10, 0.4, 0.2));
  composerMath.addPass(new RenderPixelatedPass(1, sceneMath, cameraMath));
  composerMath.addPass(new UnrealBloomPass(new THREE.Vector2(window.innerWidth, window.innerHeight), 0.5, 0.4, 0.2));
  composerMath.addPass(new AfterimagePass(0.996));
  composerMath.addPass(new FilmPass(0.2, 0, 100, false));
  const lutLoaderMath = new LUTCubeLoader();
  lutLoaderMath.load('/LUTs/Thermal Monochrome 01 BMD Film ACIDBITE.cube', (result) => {
    const lutTexture = result.texture3D;
    const lutPass = new LUTPass({ lut: lutTexture, intensity: 1 });
    composerMath.addPass(lutPass);
  });

  const sphereMathGeometry = new THREE.SphereGeometry(12, 200, 200);
  const sphereMathMaterial = new THREE.MeshBasicMaterial({ color: 0xffffff, wireframe: false });
  const sphereMath = new THREE.Mesh(sphereMathGeometry, sphereMathMaterial);
  sphereMath.position.set(0, -200, 0);
  sceneMath.add(sphereMath);


  ///////geo-cursor!!!!!
  function createGeoCursor() {
        const shape = new THREE.Shape();
        const points = 4;
        const size = 1.3;
        const curveFactor = 0.8;
        const PI2 = Math.PI * 2;
        const vertices = [];
        for (let i = 0; i < points; i++) {
          const angle = i * (PI2 / points);
          vertices.push({ x: Math.cos(angle) * size, y: Math.sin(angle) * size });
        }
        shape.moveTo(vertices[0].x, vertices[0].y);
        for (let i = 0; i < points; i++) {
          const current = vertices[i];
          const next = vertices[(i + 1) % points];
          const midX = (current.x + next.x) / 2;
          const midY = (current.y + next.y) / 2;
          const toCenterX = -midX;
          const toCenterY = -midY;
          const dist = Math.sqrt(toCenterX**2 + toCenterY**2);
          const controlX = midX + (toCenterX / dist) * curveFactor * size;
          const controlY = midY + (toCenterY / dist) * curveFactor * size;
          shape.quadraticCurveTo(controlX, controlY, next.x, next.y);
        }
        return new THREE.ShapeGeometry(shape);
      }
  
      const cursorMaterialFlute = new THREE.MeshBasicMaterial({ color: 0xffffff});
      geoCursorMesh = new THREE.Mesh(createGeoCursor(), cursorMaterialFlute);
      geoCursorMesh.rotation.x = -Math.PI / 2;
      geoCursorMesh.position.set(0, 1, 0);
      geoCursorMesh.scale.set(1.8,5.2,1.8);
      sceneA.add(geoCursorMesh);


      let geoCursor2;
          const TEXT_SETTINGS = {
            text: '‡‡',//◊‡
            color: 0x000000,
            metalness: 1,
            roughness: 1,
            inflate: 0,
            size: 3,
            depth: 10
          };
      
          const fontLoader = new FontLoader();
          let cachedFont = null;
          let fontLoading = false;
      
      
          const textMaterial = new THREE.MeshPhysicalMaterial({
            color: TEXT_SETTINGS.color,
            metalness: TEXT_SETTINGS.metalness,
            roughness: TEXT_SETTINGS.roughness,
            clearcoat: 0,
            clearcoatRoughness: 0.1
          });
      
       function buildTextMesh() {
            const construct = (font) => {
      
              geoCursor2 = new TextGeometry(TEXT_SETTINGS.text, {
                font,
                size: TEXT_SETTINGS.size,
                depth: TEXT_SETTINGS.depth,
                curveSegments: 1,
                bevelEnabled: false,
                bevelThickness: 0.06,
                bevelSize: 0.02,
                bevelSegments: 1
              });
      
              geoCursor2.center();
      
              textMaterial.color.set(TEXT_SETTINGS.color);
              textMaterial.metalness = TEXT_SETTINGS.metalness;
              textMaterial.roughness = TEXT_SETTINGS.roughness;
      
              let geoMaterial = new THREE.MeshBasicMaterial( { color: 0xFFFFFF, wireframe: false } );
      
              geoCursorMesh2 = new THREE.Mesh(geoCursor2, geoMaterial);
              geoCursorMesh2.rotation.x = -Math.PI / 2;
              geoCursorMesh2.position.set(0, 10, 12);
              geoCursorMesh2.scale.set(1,1.5,1);
              sceneA.add(geoCursorMesh2);
              console.log(geoCursorMesh2);
            };
      
            if (cachedFont) {
              construct(cachedFont);
              return;
            }
      
            if (fontLoading) return;
            fontLoading = true;
      
            fontLoader.load('/riccardo-metahaven.json', (font) => {
              cachedFont = font;
              fontLoading = false;
              construct(font);
            });
          }
              buildTextMesh();






  // Helpers/meshes similar to concatenation Scene A (using the same styling)
  const PolarGridHelperFL   = new THREE.PolarGridHelper(4000, 200, 200, 300, '#000000', '#000000');
  PolarGridHelperFL.rotation.y = Math.PI / 2; PolarGridHelperFL.position.set(100, -1, 0); sceneA.add(PolarGridHelperFL);
  const PolarGridHelper_rFL = new THREE.PolarGridHelper(4000, 200, 100, 300, '#000000', '#000000');
  PolarGridHelper_rFL.rotation.y = Math.PI / 2; PolarGridHelper_rFL.position.set(0, -1, 100); sceneA.add(PolarGridHelper_rFL);
  const PolarGridHelper2FL  = new THREE.PolarGridHelper(4000, 300, 300, 300, '#000000', '#000000');
  PolarGridHelper2FL.rotation.y = Math.PI / 2; PolarGridHelper2FL.position.set(-100, -2, 0); sceneA.add(PolarGridHelper2FL);

  const PolarGridHelper     = new THREE.PolarGridHelper(400, 200, 200, 3, '#202020', '#202020');
  PolarGridHelper.rotation.y = Math.PI / 2; PolarGridHelper.position.set(100, -1, 0); sceneA.add(PolarGridHelper);
  const PolarGridHelper_r   = new THREE.PolarGridHelper(400, 200, 100, 3, '#000000', '#202020');
  PolarGridHelper_r.rotation.y = Math.PI / 2; PolarGridHelper_r.position.set(0, -1, 100); sceneA.add(PolarGridHelper_r);
  const PolarGridHelper2    = new THREE.PolarGridHelper(100, 300, 300, 10, '#202020', '#202020');
  PolarGridHelper2.rotation.y = Math.PI / 2; PolarGridHelper2.position.set(-100, -2, 0); sceneA.add(PolarGridHelper2);

  // Cursor-following scan lines (like concatenationgan-new Scene A)
  const lineMatA = new THREE.LineBasicMaterial({ color: 0xffffff });
  const lineXGeomA = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(-100, 0.1, 0), new THREE.Vector3(100, 0.1, 0)
  ]);
  const lineYGeomA = new THREE.BufferGeometry().setFromPoints([
    new THREE.Vector3(0, 0.1, -100), new THREE.Vector3(0, 0.1, 100)
  ]);
  const lineXA = new THREE.Line(lineXGeomA, lineMatA);
  const lineYA = new THREE.Line(lineYGeomA, lineMatA);
  sceneA.add(lineXA)
  //sceneA.add(lineYA);

  // Raycaster for cursor mapping to ground plane
  const raycasterA = new THREE.Raycaster();
  const mouseA = new THREE.Vector2();
  const groundA = new THREE.Mesh(new THREE.PlaneGeometry(1000, 1000), new THREE.MeshBasicMaterial({ visible: false }));
  groundA.rotation.x = -Math.PI / 2;
  groundA.position.y = 0;
  sceneA.add(groundA);

  // Smoothed cursor position in Scene A
  const cursorPosA = new THREE.Vector3(0, 0.2, 0);
  const targetCursorA = new THREE.Vector3().copy(cursorPosA);
  const SMOOTH_A = 0.06; let DAMPING_A = 0;

  // Track mouse and project with cameraA
  document.addEventListener('mousemove', (event) => {
    mouseA.x = (event.clientX / window.innerWidth) * 2 - 1;
    mouseA.y = -(event.clientY / window.innerHeight) * 2 + 1;
    raycasterA.setFromCamera(mouseA, cameraA);
    const is = raycasterA.intersectObject(groundA);
    if (is.length > 0) {
      targetCursorA.copy(is[0].point);
      targetCursorA.y = 0.2;
    }
  });

  const clockA = new THREE.Clock();

  // === Continuous vertical scanning horizontal line ===
  let scanLineSpeed = 3.0; // units per second (adjustable)
  let scanLineZ = -frustumSizeA / 2; // start at bottom edge
  const scanLineGeom = new THREE.BufferGeometry();
  const scanLineMat = new THREE.LineBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.4 });
  const scanLine = new THREE.Line(scanLineGeom, scanLineMat);
  //sceneA.add(scanLine);

  function updateScanLineGeometry() {
    const left = (frustumSizeA * aspectA) / -2;
    const right = (frustumSizeA * aspectA) / 2;
    const positions = new Float32Array([
      left,  0.11, scanLineZ,
      right, 0.11, scanLineZ
    ]);
    scanLineGeom.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    scanLineGeom.computeBoundingSphere();
  }
  updateScanLineGeometry();

  // Runtime control hook
  window.setSceneAScanSpeed = (v) => { if (typeof v === 'number') scanLineSpeed = v; };

  // Resize hook for overlay scene
  const _origOnResize = onResize;
  function onResizeA(){
    // keep original resize behavior
    _origOnResize();
    // update overlay scene
    aspectA = window.innerWidth / window.innerHeight;
    cameraA.left   = (frustumSizeA * aspectA) / -2;
    cameraA.right  = (frustumSizeA * aspectA) /  2;
    cameraA.top    =  frustumSizeA / 2;
    cameraA.bottom = -frustumSizeA / 2;
    cameraA.updateProjectionMatrix();
    rendererA.setSize(window.innerWidth, window.innerHeight);
    composerA.setSize(window.innerWidth, window.innerHeight);
    updateScanLineGeometry();

    aspectMath = aspectA;
    cameraMath.left   = (frustumSizeA * aspectMath) / -2;
    cameraMath.right  = (frustumSizeA * aspectMath) /  2;
    cameraMath.top    =  frustumSizeA / 2;
    cameraMath.bottom = -frustumSizeA / 2;
    cameraMath.updateProjectionMatrix();
    rendererMath.setSize(window.innerWidth, window.innerHeight);
    composerMath.setSize(window.innerWidth, window.innerHeight);
  }
  window.removeEventListener('resize', onResize);
  window.addEventListener('resize', onResizeA);

  // Render hook: extend existing loop to render top scene too
  const _origAnimate = animate;
  function animateWrapper(){
    _origAnimate();

    // Animate grid helpers (similar to concatenationgan-new)
    PolarGridHelper_r.rotation.z += 0.02;
    PolarGridHelper2.rotation.y += 0.02;
    PolarGridHelper.rotation.z += 0.02;
    PolarGridHelper_r.scale.z += 0.02;
    PolarGridHelper2.scale.z += 0.02;
    PolarGridHelper.scale.z += 0.02;

    // Smooth cursor motion and update scan lines
    const dtA = Math.min(0.05, clockA.getDelta());
    const lerpA = 1 - Math.pow(1 - SMOOTH_A, dtA * 60);
    if (DAMPING_A <= 0) {
      cursorPosA.lerp(targetCursorA, lerpA);
    } else {
      cursorPosA.x = THREE.MathUtils.damp(cursorPosA.x, targetCursorA.x, DAMPING_A, dtA);
      cursorPosA.y = THREE.MathUtils.damp(cursorPosA.y, targetCursorA.y, DAMPING_A, dtA);
      cursorPosA.z = THREE.MathUtils.damp(cursorPosA.z, targetCursorA.z, DAMPING_A, dtA);
    }
    lineXA.position.set(cursorPosA.x, 0.05, cursorPosA.z);
    //lineYA.position.set(cursorPosA.x, 0.05, cursorPosA.z);

    if(geoCursorMesh!=undefined){
    geoCursorMesh.position.y = 1;
    geoCursorMesh.position.x = 0;
    geoCursorMesh.position.z = cursorPosA.z;
    if(geoCursorMesh2!=undefined){
      geoCursorMesh2.position.y = 2;
      geoCursorMesh2.position.x = 0;
      geoCursorMesh2.position.z = cursorPosA.z + 1;
    }
  }

    // Move scan line bottom -> top; loop when passing top edge
    scanLineZ -= scanLineSpeed * dtA;
    if (scanLineZ < -frustumSizeA / 2) {
      scanLineZ = frustumSizeA / 2; // reset to top then descend back through update
    }
    updateScanLineGeometry();

    composerA.render();
    composerMath.render();
  }
  // Replace animation loop
  renderer.setAnimationLoop(animateWrapper);
}

// ================== INPUT: VERTICAL ONLY ==================
let pointerY = 0 // +1 (cursor bottom) .. -1 (cursor top)
let scrollImpulse = 0
const canvas = renderer.domElement
canvas.style.touchAction = 'none' // avoid browser gestures

canvas.addEventListener('pointermove', (e) => {
  const ny = (e.clientY / window.innerHeight - 0.5) * 2 // -1 center→top, +1 center→bottom
  // We want: cursor DOWN => move DOWN → positive ny should produce downward motion
  pointerY = THREE.MathUtils.clamp(ny, -1, 1)
}, { passive: true })

canvas.addEventListener('wheel', (e) => {
  e.preventDefault()
  // Wheel down (positive deltaY) => go DOWN
  scrollImpulse += THREE.MathUtils.clamp(e.deltaY * 0.002, -1, 1)
}, { passive: false })

canvas.addEventListener('contextmenu', (e) => e.preventDefault())

const unlockAudio = () => {
  audioManager.resume().catch((err) => {
    console.warn('Audio context resume failed', err)
  })
}

canvas.addEventListener('pointerdown', unlockAudio, { once: true })
canvas.addEventListener('touchstart', unlockAudio, { once: true })
window.addEventListener('keydown', unlockAudio, { once: true })

// ================== RESIZE ==================
function onResize(){
  camera.aspect = window.innerWidth / window.innerHeight
  camera.updateProjectionMatrix()
  renderer.setSize(window.innerWidth, window.innerHeight)
  composer.setSize(window.innerWidth, window.innerHeight)
}
window.addEventListener('resize', onResize)

const transcriptionMover = (() => {
  const primary = document.getElementById('transcription-space')
  if (!primary) return null

  primary.classList.add('transcription-runner')

  const ensurePanel = (container, removeId = false) => {
    const panel = container.querySelector('.transcription-panel')
      || container.querySelector('#transcription-inner')
    if (panel) {
      panel.classList.add('transcription-panel')
      if (removeId) panel.removeAttribute('id')
    }
  }

  ensurePanel(primary, false)

  let secondary = document.querySelector('.transcription-runner--clone')
  if (!secondary) {
    secondary = primary.cloneNode(true)
    secondary.removeAttribute('id')
    ensurePanel(secondary, true)
    secondary.classList.add('transcription-runner--clone')
    primary.insertAdjacentElement('afterend', secondary)
  } else {
    ensurePanel(secondary, true)
    if (secondary.previousElementSibling !== primary) {
      primary.insertAdjacentElement('afterend', secondary)
    }
  }

  const instances = [primary, secondary]
  instances.forEach((node) => {
    node.classList.add('transcription-runner')
    node.style.willChange = 'transform'
  })

  const syncPanels = () => {
    const sourcePanel = primary.querySelector('.transcription-panel')
    if (!sourcePanel) return

    const markup = sourcePanel.innerHTML
    for (let i = 1; i < instances.length; i += 1) {
      const clonePanel = instances[i].querySelector('.transcription-panel')
      if (clonePanel && clonePanel.innerHTML !== markup) {
        clonePanel.innerHTML = markup
      }
    }
  }

  const state = {
    instances,
    y: 0,
    viewportHeight: window.innerHeight,
    elementHeight: primary.getBoundingClientRect().height || primary.offsetHeight || 0,
    bottomStart: 0,
    travelHeight: 0,
    speedPxPerSec: 0,
    sweepSeconds: 40,
  }

  function applyPositions() {
    const travel = state.travelHeight
    state.instances.forEach((node, idx) => {
      const offset = state.y + idx * travel
      node.style.transform = `translate3d(0, ${offset}px, 0)`
    })
  }

  function updateMetrics() {
    state.viewportHeight = window.innerHeight
    state.elementHeight = state.instances[0].getBoundingClientRect().height || primary.offsetHeight || 0
    state.bottomStart = Math.max(0, state.viewportHeight - state.elementHeight)
    state.travelHeight = Math.max(1, state.bottomStart + state.elementHeight)
    const duration = Math.max(0.25, state.sweepSeconds)
    state.speedPxPerSec = state.travelHeight / duration
    state.y = THREE.MathUtils.clamp(state.y, -state.elementHeight, state.bottomStart)
    applyPositions()
  }

  function update(dt) {
    if (!state.instances.length || !state.travelHeight) return

    state.y -= state.speedPxPerSec * dt
    if (state.y <= -state.elementHeight) {
      state.y += state.travelHeight
    }

    applyPositions()
  }

  function setDuration(seconds) {
    if (typeof seconds === 'number' && Number.isFinite(seconds) && seconds > 0.2) {
      state.sweepSeconds = seconds
      updateMetrics()
    }
  }

  function setSpeed(pxPerSec) {
    if (typeof pxPerSec === 'number' && Number.isFinite(pxPerSec) && pxPerSec >= 0) {
      state.speedPxPerSec = pxPerSec
      state.sweepSeconds = pxPerSec > 0
        ? Math.max(0.25, state.travelHeight / pxPerSec)
        : state.sweepSeconds
    }
  }

  function resetToBottom() {
    state.y = state.bottomStart
    applyPositions()
  }

  syncPanels()
  updateMetrics()
  resetToBottom()

  window.addEventListener('resize', updateMetrics)

  const api = {
    update,
    duration: setDuration,
    speed: setSpeed,
    recalc: updateMetrics,
    reset: resetToBottom,
    sync: syncPanels,
  }

  window.__TranscriptionSpace = api

  return api
})()

// ================== LOOP ==================
const clock = new THREE.Clock()
function animate() {
  const dt = Math.min(0.05, clock.getDelta())
  const t  = clock.elapsedTime

  // Camera vertical velocity from pointer position + scroll impulse
  // pointerY>0 means cursor below center => move DOWN (decrease Y)
  const SPEED = 2 // units/sec at full input
  const DAMP  = 0.88
  let vy = (pointerY * SPEED) + scrollImpulse
  scrollImpulse *= DAMP

  const bounds = getVerticalBounds()
  camera.position.y = THREE.MathUtils.clamp(camera.position.y - vy * dt, bounds.min, bounds.max)

  // Look forward at a point aligned with camera Y (no roll/yaw control)
  camera.lookAt(0, camera.position.y, 0)

  // Wind (endless via fract() in shader)
  const windStep = world.wind.clone().multiplyScalar(dt)

  // Keep slabs full-width & update uniforms
  for (const mesh of layers) {
    scaleLayerToViewport(mesh)
    const u = mesh.material.uniforms
    u.cameraPos.value.copy(camera.position)
    u.frame.value += 1.0
    u.uOffset.value.add(windStep)
  }

  audioManager.update(camera.position.y)
  if (transcriptionMover) {
    transcriptionMover.update(dt)
  }

  composer.render()

}