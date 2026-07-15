import * as paper from 'paper';

import { SVGUtils } from './utils/svgUtils.js';
import { FileUtils } from './utils/fileUtils.js';
import { PoseIllustration } from './illustrationGen/illustration.js';
import { Skeleton, HAND_PART_NAMES, HAND_BONE_CONNECTIONS, buildDefaultHandKeypoints } from './illustrationGen/skeleton.js';
const paperLib = paper.default || paper;

const CANVAS_WIDTH = 513;
const CANVAS_HEIGHT = 513;
const WAIST_CAMERA_ZOOM = 2.05;
const WAIST_CAMERA_TARGET_X = 0.5;
const WAIST_CAMERA_TARGET_HIP_Y = 1.03;
const MIC_LEVEL_THRESHOLD = 0.012;
const MIC_END_SILENCE_MS = 900;
const MIC_MIN_VOICE_MS = 220;
const MIC_MAX_RECORDING_MS = 12000;

const BUILTIN_AVATARS = {
  signing: './resources/illustration/signing.svg',
  boy: './resources/illustration/boy.svg',
  girl: './resources/illustration/girl.svg',
  abstract: './resources/illustration/abstract.svg',
  blathers: './resources/illustration/blathers.svg',
  'tom-nook': './resources/illustration/tom-nook.svg',
};
const DEFAULT_AVATAR = 'signing';

const REQUIRED_POSE_PARTS = [
  'nose',
  'leftEye',
  'rightEye',
  'leftEar',
  'rightEar',
  'leftShoulder',
  'rightShoulder',
  'leftElbow',
  'rightElbow',
  'leftWrist',
  'rightWrist',
  'leftHip',
  'rightHip',
  'leftKnee',
  'rightKnee',
  'leftAnkle',
  'rightAnkle',
];
const HAND_LANDMARK_IDS = Array.from({ length: 21 }, (_, idx) => idx);
const HAND_CONNECTIONS = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [5, 9], [9, 10], [10, 11], [11, 12],
  [9, 13], [13, 14], [14, 15], [15, 16],
  [13, 17], [17, 18], [18, 19], [19, 20],
  [0, 17],
];
const HAND_PALM_IDS = [0, 5, 9, 13, 17];
const HAND_TIP_IDS = new Set([4, 8, 12, 16, 20]);
const HAND_MIN_SCORE = 0.18;
const HAND_DETAIL_SCALE = 1.6;
const HAND_LINE_WIDTH = 2.25;
const HAND_JOINT_RADIUS = 1.9;
const HAND_TIP_RADIUS = 2.7;
const SHOW_HAND_OVERLAYS = false;
// The generic landmark rig does not blend with the built-in SVG avatars.
// Keep their native hands until a dedicated, skinned signing-hand asset exists.
const ENABLE_PROCEDURAL_HAND_RIG = false;
const HAND_STYLES = {
  left: {
    stroke: '#80513c',
    fill: 'rgba(247, 202, 170, 0.72)',
    joint: '#c48266',
    tip: '#d88f82',
  },
  right: {
    stroke: '#704838',
    fill: 'rgba(244, 190, 160, 0.72)',
    joint: '#ba765f',
    tip: '#d5847c',
  },
};

let canvasScope = null;
let skeleton = null;
let illustration = null;
let sequence = null;
let currentFrame = 0;
let playing = false;
let rafId = null;
let lastTick = 0;
let speedMultiplier = 1.0;
let micStream = null;
let micAudioCtx = null;
let micSourceNode = null;
let micProcessorNode = null;
let micChunks = [];
let isMicRecording = false;
let micVoiceDetected = false;
let micSilenceMs = 0;
let micVoiceMs = 0;
let micRecordedMs = 0;
let micAutoStopping = false;
let micStopping = false;
let handRigReady = false;

const el = {
  textPrompt: null,
  generateBtn: null,
  micStartBtn: null,
  micStopBtn: null,
  sequenceFile: null,
  avatarSelect: null,
  avatarFile: null,
  playBtn: null,
  resetBtn: null,
  speed: null,
  speedLabel: null,
  frameSlider: null,
  loopCheck: null,
  status: null,
};

function setStatus(message) {
  if (el.status) {
    el.status.textContent = message;
  }
}

function setMicButtons(recording) {
  if (!el.micStartBtn || !el.micStopBtn) return;
  el.micStartBtn.disabled = !!recording;
  el.micStopBtn.disabled = !recording;
}

function getFps() {
  if (!sequence || !sequence.meta || !sequence.meta.fps) {
    return 30;
  }
  return Math.max(1, Number(sequence.meta.fps) || 30);
}

function normalizePose(pose) {
  if (!pose || !pose.keypoints || !pose.keypoints.length) {
    return null;
  }
  const byPart = {};
  pose.keypoints.forEach((kp) => {
    if (!kp || !kp.part || !kp.position) return;
    byPart[kp.part] = kp;
  });
  const keypoints = REQUIRED_POSE_PARTS.map((part) => {
    const kp = byPart[part];
    if (kp && kp.position) {
      return {
        part: part,
        score: Number(kp.score || 1.0),
        position: {
          x: Number(kp.position.x || 0),
          y: Number(kp.position.y || 0),
        },
      };
    }
    return {
      part: part,
      score: 0,
      position: { x: 0, y: 0 },
    };
  });
  return {
    score: Number(pose.score || 1.0),
    keypoints: keypoints,
  };
}

function normalizeFace(face) {
  if (!face || !face.positions || !face.positions.length) {
    return null;
  }
  return {
    faceInViewConfidence: Number(face.faceInViewConfidence || 1.0),
    positions: face.positions.map((v) => Number(v || 0)),
  };
}

function normalizeHand(hand) {
  if (!hand || !hand.keypoints || !hand.keypoints.length) {
    return null;
  }
  const byId = {};
  hand.keypoints.forEach((kp) => {
    const landmarkId = Number(kp && (kp.landmarkId ?? kp.landmark_id));
    if (!Number.isFinite(landmarkId) || !kp || !kp.position) return;
    byId[landmarkId] = {
      landmarkId: landmarkId,
      score: Number(kp.score || 0),
      position: {
        x: Number(kp.position.x || 0),
        y: Number(kp.position.y || 0),
      },
    };
  });
  const keypoints = HAND_LANDMARK_IDS.map((landmarkId) => {
    const kp = byId[landmarkId];
    if (kp) return kp;
    return {
      landmarkId: landmarkId,
      score: 0,
      position: { x: 0, y: 0 },
    };
  });
  return {
    score: Number(hand.score || 0),
    keypoints: keypoints,
  };
}

function normalizeSequence(payload) {
  if (!payload || !payload.frames || !Array.isArray(payload.frames)) {
    throw new Error('Invalid sequence JSON: missing frames array');
  }

  const out = {
    meta: {
      fps: payload.meta && payload.meta.fps ? Number(payload.meta.fps) : 30,
      canvasWidth: payload.meta && payload.meta.canvasWidth ? Number(payload.meta.canvasWidth) : CANVAS_WIDTH,
      canvasHeight: payload.meta && payload.meta.canvasHeight ? Number(payload.meta.canvasHeight) : CANVAS_HEIGHT,
    },
    frames: [],
  };

  payload.frames.forEach((frame) => {
    if (!frame) return;
    const pose = normalizePose(frame.pose);
    if (!pose) return;
    const hands = frame.hands || {};
    out.frames.push({
      pose: pose,
      face: normalizeFace(frame.face),
      hands: {
        left: normalizeHand(hands.left || frame.leftHand || frame.left_hand),
        right: normalizeHand(hands.right || frame.rightHand || frame.right_hand),
      },
    });
  });

  if (!out.frames.length) {
    throw new Error('Sequence has no valid frames');
  }

  return out;
}

function cloneColorForScope(scope, value) {
  if (!value) return null;
  if (typeof value.clone === 'function') {
    return value.clone();
  }
  return new scope.Color(value);
}

function inferHandRigStyle(svgScope, wristPoint, side) {
  const fallback = HAND_STYLES[side] || HAND_STYLES.left;
  const items = svgScope.project.getItems({ recursive: true }).filter((item) => {
    if (!item || !item.parent || !item.parent.name || !item.parent.name.startsWith('illustration')) return false;
    return SVGUtils.isPath(item) || SVGUtils.isShape(item);
  });

  let best = null;
  let bestDistance = Infinity;
  items.forEach((item) => {
    if (!item.bounds || !item.bounds.width || !item.bounds.height) return;
    const color = item.fillColor || item.strokeColor;
    if (!color) return;
    const distance = item.bounds.center.getDistance(wristPoint);
    if (distance < bestDistance) {
      bestDistance = distance;
      best = item;
    }
  });

  const fillColor = cloneColorForScope(svgScope, best && best.fillColor ? best.fillColor : fallback.fill) || new svgScope.Color(fallback.fill);
  const strokeColor = cloneColorForScope(svgScope, best && best.strokeColor ? best.strokeColor : fallback.stroke) || new svgScope.Color(fallback.stroke);
  if (fillColor.alpha === undefined) {
    fillColor.alpha = 1;
  }
  fillColor.alpha = 0.98;
  if (typeof fillColor.brightness === 'number') {
    fillColor.brightness = Math.min(0.96, Math.max(0.76, fillColor.brightness + 0.08));
  }
  if (typeof fillColor.saturation === 'number') {
    fillColor.saturation = Math.min(0.92, Math.max(0.25, fillColor.saturation + 0.10));
  }
  if (typeof strokeColor.brightness === 'number') {
    strokeColor.brightness = Math.max(0, strokeColor.brightness * 0.5);
  }
  if (typeof strokeColor.saturation === 'number') {
    strokeColor.saturation = Math.min(1, strokeColor.saturation + 0.15);
  }
  const haloColor = new svgScope.Color(1, 1, 1, 0.94);
  const outlineColor = strokeColor.clone();
  const palmColor = fillColor.clone();
  palmColor.alpha = 0.94;
  if (typeof palmColor.brightness === 'number') {
    palmColor.brightness = Math.min(0.98, palmColor.brightness + 0.04);
  }
  const segmentColor = fillColor.clone();
  segmentColor.alpha = 1;
  if (typeof segmentColor.brightness === 'number') {
    segmentColor.brightness = Math.max(0.52, segmentColor.brightness - 0.08);
  }
  if (typeof segmentColor.saturation === 'number') {
    segmentColor.saturation = Math.min(1, segmentColor.saturation + 0.08);
  }
  const jointColor = segmentColor.clone();
  const tipColor = outlineColor.clone();
  return { fillColor, strokeColor, haloColor, outlineColor, palmColor, segmentColor, jointColor, tipColor };
}

function addHandKeypointMarker(svgScope, group, name, point) {
  const marker = new svgScope.Shape.Circle({
    center: point,
    radius: 1.5,
    fillColor: new svgScope.Color(0, 0, 0, 0),
    strokeColor: null,
  });
  marker.name = name;
  group.addChild(marker);
}

function getHandSegmentWidth(fromId, toId, baseWidth) {
  const palmBones = new Set(['0-5', '0-9', '0-13', '0-17', '5-9', '9-13', '13-17']);
  const key = `${Math.min(fromId, toId)}-${Math.max(fromId, toId)}`;
  if (palmBones.has(key)) {
    return baseWidth * 0.86;
  }
  const tipIds = new Set([4, 8, 12, 16, 20]);
  if (tipIds.has(toId)) {
    return baseWidth * 0.48;
  }
  if (tipIds.has(fromId)) {
    return baseWidth * 0.42;
  }
  return baseWidth * 0.62;
}

function addTaperedFingerSegment(svgScope, group, fromPoint, toPoint, style, width) {
  const direction = toPoint.subtract(fromPoint);
  if (direction.length < 0.1) return;
  const normal = direction.normalize().rotate(90);
  const startRadius = Math.max(1.5, width * 0.54);
  const endRadius = Math.max(1.2, width * 0.40);
  const segment = new svgScope.Path({
    closed: true,
    fillColor: style.segmentColor.clone(),
    strokeColor: null,
  });
  segment.add(fromPoint.add(normal.multiply(startRadius)));
  segment.add(toPoint.add(normal.multiply(endRadius)));
  segment.add(toPoint.subtract(normal.multiply(endRadius)));
  segment.add(fromPoint.subtract(normal.multiply(startRadius)));
  group.addChild(segment);
}

function addNaturalPalm(svgScope, group, points, style, baseWidth) {
  const palm = new svgScope.Path({
    closed: true,
    fillColor: style.palmColor.clone(),
    strokeColor: null,
  });
  points.forEach((point) => palm.add(point));
  palm.smooth({ type: 'continuous' });
  group.addChild(palm);
}

function addRoundedFingertip(svgScope, group, tipPoint, previousPoint, style, baseWidth) {
  const direction = tipPoint.subtract(previousPoint);
  if (direction.length < 0.1) return;
  const radius = Math.max(2, baseWidth * 0.20);
  const fingertip = new svgScope.Path.Circle({
    center: tipPoint,
    radius: radius * 1.12,
    fillColor: style.segmentColor.clone(),
    strokeColor: null,
  });
  group.addChild(fingertip);
}

function addSoftHandJoint(svgScope, group, point, style, baseWidth) {
  const joint = new svgScope.Path.Circle({
    center: point,
    radius: Math.max(1.9, baseWidth * 0.28),
    fillColor: style.segmentColor.clone(),
    strokeColor: null,
  });
  group.addChild(joint);
}

function addWristBlend(svgScope, group, wristPoint, style, baseWidth) {
  const blend = new svgScope.Path.Circle({
    center: wristPoint,
    radius: Math.max(3, baseWidth * 0.62),
    fillColor: style.palmColor.clone(),
    strokeColor: null,
  });
  group.addChild(blend);
}

function buildProceduralHandRig(svgScope, side, wristPoint, elbowPoint) {
  const keypoints = buildDefaultHandKeypoints(side, wristPoint, elbowPoint);
  const names = HAND_PART_NAMES[side];
  const style = inferHandRigStyle(svgScope, wristPoint, side);
  const group = new svgScope.Group();
  group.name = `illustration_generated_${side}_hand`;

  names.forEach((name) => {
    addHandKeypointMarker(svgScope, group, name, keypoints[name].position);
  });

  const forearmLength = Math.max(wristPoint.getDistance(elbowPoint), 24);
  const baseWidth = Math.max(7, forearmLength * 0.13);
  const palmPoints = [0, 5, 9, 13, 17].map((idx) => keypoints[names[idx]].position);
  addWristBlend(svgScope, group, keypoints[names[0]].position, style, baseWidth);
  addNaturalPalm(svgScope, group, palmPoints, style, baseWidth);

  HAND_BONE_CONNECTIONS.forEach(([fromId, toId]) => {
    addTaperedFingerSegment(
      svgScope,
      group,
      keypoints[names[fromId]].position,
      keypoints[names[toId]].position,
      style,
      getHandSegmentWidth(fromId, toId, baseWidth),
    );
  });

  [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15, 17, 18, 19].forEach((jointId) => {
    addSoftHandJoint(svgScope, group, keypoints[names[jointId]].position, style, baseWidth);
  });

  [[4, 3], [8, 7], [12, 11], [16, 15], [20, 19]].forEach(([tipId, previousId]) => {
    addRoundedFingertip(
      svgScope,
      group,
      keypoints[names[tipId]].position,
      keypoints[names[previousId]].position,
      style,
      baseWidth,
    );
  });

  return group;
}

function attachProceduralHandRig(svgScope, rigSkeleton) {
  const leftWrist = rigSkeleton.bLeftElbowLeftWrist.kp1.position;
  const leftElbow = rigSkeleton.bLeftElbowLeftWrist.kp0.position;
  const rightWrist = rigSkeleton.bRightElbowRightWrist.kp1.position;
  const rightElbow = rigSkeleton.bRightElbowRightWrist.kp0.position;
  buildProceduralHandRig(svgScope, 'left', leftWrist, leftElbow);
  buildProceduralHandRig(svgScope, 'right', rightWrist, rightElbow);
}

function updateFrameSlider() {
  const max = sequence ? Math.max(sequence.frames.length - 1, 0) : 0;
  el.frameSlider.max = String(max);
  el.frameSlider.value = String(Math.min(currentFrame, max));
}

function clearCanvas() {
  if (canvasScope) {
    canvasScope.project.clear();
    canvasScope.view.update();
  }
}

function getPosePart(pose, partName) {
  if (!pose || !pose.keypoints || !pose.keypoints.length) return null;
  for (let i = 0; i < pose.keypoints.length; i += 1) {
    const kp = pose.keypoints[i];
    if (kp && kp.part === partName && kp.position) {
      return kp.position;
    }
  }
  return null;
}

function getHipAnchor(pose) {
  const leftHip = getPosePart(pose, 'leftHip');
  const rightHip = getPosePart(pose, 'rightHip');
  if (leftHip && rightHip) {
    return {
      x: (leftHip.x + rightHip.x) * 0.5,
      y: (leftHip.y + rightHip.y) * 0.5,
    };
  }
  return leftHip || rightHip || null;
}

function applyWaistCameraFraming(pose) {
  if (!canvasScope || !canvasScope.project || !canvasScope.project.activeLayer) return;
  const anchor = getHipAnchor(pose) || { x: CANVAS_WIDTH * 0.5, y: CANVAS_HEIGHT * 0.72 };
  const targetX = CANVAS_WIDTH * WAIST_CAMERA_TARGET_X;
  const targetY = CANVAS_HEIGHT * WAIST_CAMERA_TARGET_HIP_Y;
  const layer = canvasScope.project.activeLayer;
  layer.scale(WAIST_CAMERA_ZOOM, new canvasScope.Point(anchor.x, anchor.y));
  layer.translate(targetX - anchor.x, targetY - anchor.y);
}

function getHandPoint(hand, landmarkId) {
  if (!hand || !hand.keypoints || !hand.keypoints.length) return null;
  const kp = hand.keypoints[landmarkId];
  if (!kp || !kp.position || Number(kp.score || 0) <= 0) return null;
  return kp.position;
}

function projectHandPoint(rawPoint, rawWrist, targetWrist) {
  return new canvasScope.Point(
    Number(targetWrist.x) + (Number(rawPoint.x) - Number(rawWrist.x)) * HAND_DETAIL_SCALE,
    Number(targetWrist.y) + (Number(rawPoint.y) - Number(rawWrist.y)) * HAND_DETAIL_SCALE,
  );
}

function getHandOverlayWidth(fromId, toId) {
  if (fromId === 0 || toId === 0) return HAND_LINE_WIDTH * 1.32;
  if (HAND_TIP_IDS.has(toId)) return HAND_LINE_WIDTH * 0.68;
  if ([1, 5, 9, 13, 17].includes(fromId)) return HAND_LINE_WIDTH * 1.02;
  return HAND_LINE_WIDTH * 0.84;
}

function drawHandConnection(a, b, style, width) {
  new canvasScope.Path({
    segments: [a, b],
    strokeColor: 'rgba(255, 255, 255, 0.9)',
    strokeWidth: width + 1.35,
    strokeCap: 'round',
    strokeJoin: 'round',
  });
  new canvasScope.Path({
    segments: [a, b],
    strokeColor: style.stroke,
    strokeWidth: width + 0.65,
    strokeCap: 'round',
    strokeJoin: 'round',
  });
  new canvasScope.Path({
    segments: [a, b],
    strokeColor: style.joint,
    strokeWidth: width,
    strokeCap: 'round',
    strokeJoin: 'round',
  });
}

function drawHandJoint(point, landmarkId, style) {
  const isTip = HAND_TIP_IDS.has(landmarkId);
  const radius = isTip ? HAND_TIP_RADIUS : HAND_JOINT_RADIUS;
  new canvasScope.Path.Circle({
    center: point,
    radius: radius + 0.7,
    fillColor: 'rgba(255, 255, 255, 0.9)',
    strokeColor: null,
  });
  new canvasScope.Path.Circle({
    center: point,
    radius: radius,
    fillColor: isTip ? style.tip : style.joint,
    strokeColor: style.stroke,
    strokeWidth: 0.85,
  });
}

function drawHandDetails(frame, side) {
  const hand = frame && frame.hands ? frame.hands[side] : null;
  if (!hand || Number(hand.score || 0) < HAND_MIN_SCORE) return;

  const rawWrist = getHandPoint(hand, 0);
  if (!rawWrist) return;

  const poseWrist = getPosePart(frame.pose, side === 'left' ? 'leftWrist' : 'rightWrist') || rawWrist;
  const projected = HAND_LANDMARK_IDS.map((landmarkId) => {
    const rawPoint = getHandPoint(hand, landmarkId);
    if (!rawPoint) return null;
    return projectHandPoint(rawPoint, rawWrist, poseWrist);
  });

  const visibleCount = projected.filter(Boolean).length;
  if (visibleCount < 6) return;

  const style = HAND_STYLES[side] || HAND_STYLES.left;
  if (HAND_PALM_IDS.every((landmarkId) => projected[landmarkId])) {
    [
      { fill: 'rgba(255, 255, 255, 0.82)', stroke: 'rgba(255, 255, 255, 0.82)', width: 2.4 },
      { fill: style.fill, stroke: style.stroke, width: 0.85 },
    ].forEach((palmStyle) => {
      const palm = new canvasScope.Path({
        closed: true,
        fillColor: palmStyle.fill,
        strokeColor: palmStyle.stroke,
        strokeWidth: palmStyle.width,
      });
      HAND_PALM_IDS.forEach((landmarkId) => {
        palm.add(projected[landmarkId]);
      });
      palm.smooth({ type: 'continuous' });
    });
  }

  HAND_CONNECTIONS.forEach(([fromId, toId]) => {
    if (!projected[fromId] || !projected[toId]) return;
    drawHandConnection(
      projected[fromId],
      projected[toId],
      style,
      getHandOverlayWidth(fromId, toId),
    );
  });

  projected.forEach((point, landmarkId) => {
    if (!point) return;
    drawHandJoint(point, landmarkId, style);
  });
}

function renderHandOverlays(frame) {
  if (!frame) return;
  drawHandDetails(frame, 'left');
  drawHandDetails(frame, 'right');
}

function renderFrame(frameIndex) {
  if (!sequence || !illustration || !skeleton) return;
  if (frameIndex < 0 || frameIndex >= sequence.frames.length) return;

  const frame = sequence.frames[frameIndex];
  if (!frame || !frame.pose) return;

  skeleton.reset();
  canvasScope.project.clear();
  illustration.updateSkeleton(frame.pose, frame.face || null, frame.hands || null);
  illustration.draw();
  if (SHOW_HAND_OVERLAYS && !handRigReady) {
    renderHandOverlays(frame);
  }
  applyWaistCameraFraming(frame.pose);
  canvasScope.view.update();
}

function stopPlayback() {
  playing = false;
  lastTick = 0;
  if (rafId) {
    cancelAnimationFrame(rafId);
    rafId = null;
  }
  el.playBtn.textContent = 'Play';
}

function stepForward(stepCount) {
  if (!sequence) return;
  const maxIdx = sequence.frames.length - 1;
  let next = currentFrame + stepCount;
  if (next > maxIdx) {
    if (el.loopCheck.checked) {
      next = next % (maxIdx + 1);
    } else {
      next = maxIdx;
      stopPlayback();
    }
  }
  currentFrame = next;
  updateFrameSlider();
  renderFrame(currentFrame);
}

function animate(ts) {
  if (!playing || !sequence) return;
  if (!lastTick) {
    lastTick = ts;
  }
  const targetFps = getFps() * speedMultiplier;
  const frameDuration = 1000.0 / Math.max(1, targetFps);
  const elapsed = ts - lastTick;
  if (elapsed >= frameDuration) {
    const steps = Math.max(1, Math.floor(elapsed / frameDuration));
    lastTick = ts;
    stepForward(steps);
  }
  rafId = requestAnimationFrame(animate);
}

function startPlayback() {
  if (!sequence || !sequence.frames.length) {
    setStatus('Load sequence JSON first.');
    return;
  }
  if (playing) return;
  playing = true;
  lastTick = 0;
  el.playBtn.textContent = 'Pause';
  rafId = requestAnimationFrame(animate);
}

function togglePlayback() {
  if (playing) {
    stopPlayback();
  } else {
    startPlayback();
  }
}

async function loadSVG(target) {
  const svgScope = await SVGUtils.importSVG(target);
  handRigReady = false;
  skeleton = new Skeleton(svgScope);
  if (ENABLE_PROCEDURAL_HAND_RIG) {
    try {
      attachProceduralHandRig(svgScope, skeleton);
      handRigReady = true;
    } catch (err) {
      handRigReady = false;
      console.warn('Procedural hand rig disabled:', err);
    }
  }
  illustration = new PoseIllustration(canvasScope);
  illustration.bindSkeleton(skeleton, svgScope);
  if (sequence && sequence.frames.length) {
    renderFrame(currentFrame);
  } else {
    canvasScope.project.clear();
    illustration.drawRestPose();
    applyWaistCameraFraming(null);
    canvasScope.view.update();
  }
}

async function loadBuiltInAvatar(key) {
  const url = BUILTIN_AVATARS[key];
  if (!url) {
    throw new Error(`Unknown avatar key: ${key}`);
  }
  const res = await fetch(url, { cache: 'no-store' });
  if (!res.ok) {
    throw new Error(`Failed to fetch avatar (${res.status})`);
  }
  const svgText = await res.text();
  await loadSVG(svgText);
}

async function readFileText(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = reject;
    reader.readAsText(file);
  });
}

function mergeFloat32Chunks(chunks) {
  if (!chunks || !chunks.length) return new Float32Array(0);
  let totalLength = 0;
  chunks.forEach((chunk) => {
    totalLength += chunk.length;
  });
  const out = new Float32Array(totalLength);
  let offset = 0;
  chunks.forEach((chunk) => {
    out.set(chunk, offset);
    offset += chunk.length;
  });
  return out;
}

function downsampleBuffer(source, inputRate, targetRate) {
  if (!source || !source.length) return new Float32Array(0);
  if (inputRate === targetRate) return source;
  if (inputRate < targetRate) {
    throw new Error(`Input sample rate ${inputRate} is lower than target ${targetRate}.`);
  }

  const ratio = inputRate / targetRate;
  const newLength = Math.max(1, Math.round(source.length / ratio));
  const out = new Float32Array(newLength);

  let sourceOffset = 0;
  for (let i = 0; i < newLength; i += 1) {
    const nextSourceOffset = Math.min(source.length, Math.round((i + 1) * ratio));
    let sum = 0;
    let count = 0;
    for (let j = sourceOffset; j < nextSourceOffset; j += 1) {
      sum += source[j];
      count += 1;
    }
    out[i] = count > 0 ? sum / count : 0;
    sourceOffset = nextSourceOffset;
  }

  return out;
}

function encodeWavPcm16(samples, sampleRate) {
  const bytesPerSample = 2;
  const blockAlign = bytesPerSample;
  const byteRate = sampleRate * blockAlign;
  const dataSize = samples.length * bytesPerSample;
  const buffer = new ArrayBuffer(44 + dataSize);
  const view = new DataView(buffer);

  const writeString = (offset, text) => {
    for (let i = 0; i < text.length; i += 1) {
      view.setUint8(offset + i, text.charCodeAt(i));
    }
  };

  writeString(0, 'RIFF');
  view.setUint32(4, 36 + dataSize, true);
  writeString(8, 'WAVE');
  writeString(12, 'fmt ');
  view.setUint32(16, 16, true); // PCM chunk size
  view.setUint16(20, 1, true); // PCM format
  view.setUint16(22, 1, true); // mono
  view.setUint32(24, sampleRate, true);
  view.setUint32(28, byteRate, true);
  view.setUint16(32, blockAlign, true);
  view.setUint16(34, 16, true); // bits per sample
  writeString(36, 'data');
  view.setUint32(40, dataSize, true);

  let offset = 44;
  for (let i = 0; i < samples.length; i += 1) {
    let s = samples[i];
    if (s > 1) s = 1;
    if (s < -1) s = -1;
    const pcm = s < 0 ? s * 0x8000 : s * 0x7fff;
    view.setInt16(offset, pcm, true);
    offset += 2;
  }

  return buffer;
}

async function cleanupMicCapture() {
  isMicRecording = false;
  micVoiceDetected = false;
  micSilenceMs = 0;
  micVoiceMs = 0;
  micRecordedMs = 0;
  micAutoStopping = false;

  if (micProcessorNode) {
    micProcessorNode.onaudioprocess = null;
    try {
      micProcessorNode.disconnect();
    } catch (err) {
      // no-op
    }
    micProcessorNode = null;
  }

  if (micSourceNode) {
    try {
      micSourceNode.disconnect();
    } catch (err) {
      // no-op
    }
    micSourceNode = null;
  }

  if (micStream) {
    try {
      micStream.getTracks().forEach((track) => track.stop());
    } catch (err) {
      // no-op
    }
    micStream = null;
  }

  if (micAudioCtx) {
    try {
      await micAudioCtx.close();
    } catch (err) {
      // no-op
    }
    micAudioCtx = null;
  }

  micChunks = [];
  setMicButtons(false);
}

async function startMicCapture() {
  if (isMicRecording || micStopping) return;
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    setStatus('Microphone is not supported in this browser.');
    return;
  }

  try {
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: {
        channelCount: 1,
        noiseSuppression: true,
        echoCancellation: true,
      },
    });
    const Ctx = window.AudioContext || window.webkitAudioContext;
    micAudioCtx = new Ctx();
    micSourceNode = micAudioCtx.createMediaStreamSource(micStream);
    micProcessorNode = micAudioCtx.createScriptProcessor(4096, 1, 1);
    micChunks = [];
    isMicRecording = true;
    micVoiceDetected = false;
    micSilenceMs = 0;
    micVoiceMs = 0;
    micRecordedMs = 0;
    micAutoStopping = false;

    micProcessorNode.onaudioprocess = (event) => {
      if (!isMicRecording) return;
      const input = event.inputBuffer.getChannelData(0);
      const chunk = new Float32Array(input);
      micChunks.push(chunk);

      const sr = micAudioCtx ? micAudioCtx.sampleRate : 48000;
      const chunkMs = (chunk.length / Math.max(1, sr)) * 1000.0;
      micRecordedMs += chunkMs;

      let sumSq = 0;
      for (let i = 0; i < chunk.length; i += 1) {
        const v = chunk[i];
        sumSq += v * v;
      }
      const rms = Math.sqrt(sumSq / Math.max(1, chunk.length));
      if (rms >= MIC_LEVEL_THRESHOLD) {
        micVoiceDetected = true;
        micVoiceMs += chunkMs;
        micSilenceMs = 0;
      } else if (micVoiceDetected) {
        micSilenceMs += chunkMs;
      }

      const shouldAutoStopBySilence =
        micVoiceDetected &&
        micVoiceMs >= MIC_MIN_VOICE_MS &&
        micSilenceMs >= MIC_END_SILENCE_MS;
      const shouldAutoStopByMaxLen = micRecordedMs >= MIC_MAX_RECORDING_MS;

      if (!micAutoStopping && (shouldAutoStopBySilence || shouldAutoStopByMaxLen)) {
        micAutoStopping = true;
        setStatus('Speech end detected. Transcribing...');
        void stopMicCaptureAndTranscribe('auto');
      }
    };

    micSourceNode.connect(micProcessorNode);
    micProcessorNode.connect(micAudioCtx.destination);

    setMicButtons(true);
    setStatus('Listening... it will auto-stop after you finish speaking.');
  } catch (err) {
    await cleanupMicCapture();
    setStatus(`Microphone error: ${err.message}`);
  }
}

async function stopMicCaptureAndTranscribe(trigger = 'manual') {
  if (micStopping) return;
  if (!isMicRecording) return;

  micStopping = true;
  isMicRecording = false;
  setMicButtons(false);
  if (trigger !== 'auto') {
    setStatus('Transcribing speech...');
  }

  const inputRate = micAudioCtx ? Math.floor(micAudioCtx.sampleRate || 48000) : 48000;
  const captured = mergeFloat32Chunks(micChunks);
  await cleanupMicCapture();

  if (!captured.length) {
    setStatus('No audio captured.');
    return;
  }

  try {
    const targetRate = 16000;
    const downsampled = downsampleBuffer(captured, inputRate, targetRate);
    const wavBuffer = encodeWavPcm16(downsampled, targetRate);
    const res = await fetch('/api/transcribe_wav', {
      method: 'POST',
      headers: {
        'Content-Type': 'audio/wav',
      },
      body: wavBuffer,
    });
    const data = await res.json();
    if (!res.ok || !data.ok) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    const text = String(data.text || '').trim();
    if (text) {
      el.textPrompt.value = text;
      setStatus(`Voice recognized: "${text}". Generating...`);
      await onGenerateFromText();
    } else {
      setStatus('No speech recognized. Try again with a louder/clearer voice.');
    }
  } catch (err) {
    setStatus(`Transcription failed: ${err.message}`);
  } finally {
    micStopping = false;
  }
}

function loadSequencePayload(payload) {
  sequence = normalizeSequence(payload);
  currentFrame = 0;
  updateFrameSlider();
  renderFrame(currentFrame);
  const hasHandDetail = sequence.frames.some((frame) => {
    const hands = frame && frame.hands ? frame.hands : {};
    return ['left', 'right'].some((side) => hands[side] && Number(hands[side].score || 0) >= HAND_MIN_SCORE);
  });
  setStatus(`Loaded ${sequence.frames.length} frames @ ${getFps()} FPS${hasHandDetail ? ' with detailed hands' : ''}`);
}

async function onSequenceFileChange(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  try {
    const text = await readFileText(file);
    const parsed = JSON.parse(text);
    loadSequencePayload(parsed);
  } catch (err) {
    stopPlayback();
    setStatus(`Failed to load sequence: ${err.message}`);
  }
}

async function onGenerateFromText() {
  const raw = (el.textPrompt.value || '').trim();
  if (!raw) {
    setStatus('Type text first.');
    return;
  }
  stopPlayback();
  setStatus('Generating sequence from text...');
  try {
    const url = `/api/generate_sequence?text=${encodeURIComponent(raw)}`;
    const res = await fetch(url);
    const data = await res.json();
    if (!res.ok || !data.ok || !data.payload) {
      throw new Error(data.error || `HTTP ${res.status}`);
    }
    loadSequencePayload(data.payload);
    startPlayback();
  } catch (err) {
    setStatus(`Generate failed: ${err.message}`);
  }
}

async function onAvatarFileChange(event) {
  const file = event.target.files && event.target.files[0];
  if (!file) return;
  try {
    const svgText = await readFileText(file);
    await loadSVG(svgText);
    setStatus('Custom SVG avatar loaded.');
  } catch (err) {
    setStatus(`Failed to load avatar SVG: ${err.message}`);
  }
}

async function init() {
  el.textPrompt = document.getElementById('textPrompt');
  el.generateBtn = document.getElementById('generateBtn');
  el.micStartBtn = document.getElementById('micStartBtn');
  el.micStopBtn = document.getElementById('micStopBtn');
  el.sequenceFile = document.getElementById('sequenceFile');
  el.avatarSelect = document.getElementById('avatarSelect');
  el.avatarFile = document.getElementById('avatarFile');
  el.playBtn = document.getElementById('playBtn');
  el.resetBtn = document.getElementById('resetBtn');
  el.speed = document.getElementById('speed');
  el.speedLabel = document.getElementById('speedLabel');
  el.frameSlider = document.getElementById('frameSlider');
  el.loopCheck = document.getElementById('loopCheck');
  el.status = document.getElementById('status');

  const canvas = document.querySelector('.illustration-canvas');
  canvas.width = CANVAS_WIDTH;
  canvas.height = CANVAS_HEIGHT;
  canvasScope = paperLib;
  canvasScope.setup(canvas);

  Object.keys(BUILTIN_AVATARS).forEach((name) => {
    const option = document.createElement('option');
    option.value = name;
    option.textContent = name;
    el.avatarSelect.appendChild(option);
  });
  el.avatarSelect.value = DEFAULT_AVATAR;

  el.sequenceFile.addEventListener('change', onSequenceFileChange);
  el.generateBtn.addEventListener('click', onGenerateFromText);
  if (el.micStartBtn) {
    el.micStartBtn.addEventListener('click', startMicCapture);
  }
  if (el.micStopBtn) {
    el.micStopBtn.addEventListener('click', stopMicCaptureAndTranscribe);
  }
  el.avatarFile.addEventListener('change', onAvatarFileChange);
  el.playBtn.addEventListener('click', togglePlayback);

  el.textPrompt.addEventListener('keydown', (event) => {
    if (event.key === 'Enter') {
      onGenerateFromText();
    }
  });

  el.resetBtn.addEventListener('click', () => {
    stopPlayback();
    currentFrame = 0;
    updateFrameSlider();
    if (sequence) renderFrame(currentFrame);
  });

  el.speed.addEventListener('input', () => {
    speedMultiplier = Number(el.speed.value || 1.0);
    el.speedLabel.textContent = `${speedMultiplier.toFixed(2)}x`;
  });

  el.frameSlider.addEventListener('input', () => {
    if (!sequence) return;
    currentFrame = Number(el.frameSlider.value || 0);
    renderFrame(currentFrame);
  });

  el.avatarSelect.addEventListener('change', async () => {
    const key = el.avatarSelect.value;
    if (!BUILTIN_AVATARS[key]) return;
    await loadBuiltInAvatar(key);
    setStatus(`Avatar loaded: ${key}`);
  });

  // Same UX as original repo: drag/drop SVG file anywhere in page.
  FileUtils.setDragDropHandler(async (svgText) => {
    await loadSVG(svgText);
    setStatus('Custom SVG avatar loaded by drag & drop.');
  });

  await loadBuiltInAvatar(DEFAULT_AVATAR);
  setStatus('Signing avatar loaded. Type text or load sequence JSON to start.');
  setMicButtons(false);
}

window.addEventListener('error', (event) => {
  setStatus(`Runtime error: ${event.message}`);
});

window.addEventListener('unhandledrejection', (event) => {
  const msg = event && event.reason && event.reason.message ? event.reason.message : String(event.reason);
  setStatus(`Promise error: ${msg}`);
});

window.addEventListener('beforeunload', () => {
  cleanupMicCapture();
});

window.onload = () => {
  init().catch((err) => {
    setStatus(`Init failed: ${err.message}`);
  });
};
