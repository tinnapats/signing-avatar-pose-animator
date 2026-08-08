export const REQUIRED_POSE_PARTS = [
  'nose', 'leftEye', 'rightEye', 'leftEar', 'rightEar',
  'leftShoulder', 'rightShoulder', 'leftElbow', 'rightElbow',
  'leftWrist', 'rightWrist', 'leftHip', 'rightHip',
  'leftKnee', 'rightKnee', 'leftAnkle', 'rightAnkle',
] as const;

export const HAND_LANDMARK_IDS = Array.from({ length: 21 }, (_, index) => index);

export interface Position2D { x: number; y: number; }
export interface Position3D extends Position2D { z: number; }
export interface PoseKeypoint { part: string; score: number; position: Position2D; }
export interface HandKeypoint { landmarkId: number; score: number; position: Position3D; }
export interface NormalizedHand { score: number; keypoints: HandKeypoint[]; observed: boolean; flipHeld: boolean; }
export interface NormalizedSequence {
  meta: { fps: number; canvasWidth: number; canvasHeight: number };
  frames: Array<{
    pose: { score: number; keypoints: PoseKeypoint[] };
    face: { faceInViewConfidence: number; positions: number[] } | null;
    hands: { left: NormalizedHand | null; right: NormalizedHand | null };
  }>;
}

type UnknownRecord = Record<string, unknown>;
const record = (value: unknown): UnknownRecord | null => value !== null && typeof value === 'object' ? value as UnknownRecord : null;
const numberValue = (value: unknown, fallback = 0): number => {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
};

function normalizePose(value: unknown): NormalizedSequence['frames'][number]['pose'] | null {
  const pose = record(value);
  const rawKeypoints = Array.isArray(pose?.keypoints) ? pose.keypoints : [];
  if (!rawKeypoints.length) return null;
  const byPart = new Map<string, UnknownRecord>();
  rawKeypoints.forEach((raw) => {
    const keypoint = record(raw);
    const part = typeof keypoint?.part === 'string' ? keypoint.part : '';
    if (part && keypoint && record(keypoint.position)) byPart.set(part, keypoint);
  });
  return {
    score: numberValue(pose?.score, 1),
    keypoints: REQUIRED_POSE_PARTS.map((part): PoseKeypoint => {
      const keypoint = byPart.get(part);
      const position = record(keypoint?.position);
      return { part, score: keypoint ? numberValue(keypoint.score, 1) : 0, position: { x: numberValue(position?.x), y: numberValue(position?.y) } };
    }),
  };
}

function normalizeFace(value: unknown): NormalizedSequence['frames'][number]['face'] {
  const face = record(value);
  if (!Array.isArray(face?.positions) || !face.positions.length) return null;
  return { faceInViewConfidence: numberValue(face.faceInViewConfidence, 1), positions: face.positions.map((item) => numberValue(item)) };
}

function normalizeHand(value: unknown): NormalizedHand | null {
  const hand = record(value);
  const rawKeypoints = Array.isArray(hand?.keypoints) ? hand.keypoints : [];
  if (!rawKeypoints.length) return null;
  const byId = new Map<number, HandKeypoint>();
  rawKeypoints.forEach((raw) => {
    const keypoint = record(raw);
    const position = record(keypoint?.position);
    const landmarkId = numberValue(keypoint?.landmarkId ?? keypoint?.landmark_id, Number.NaN);
    if (!Number.isInteger(landmarkId) || landmarkId < 0 || landmarkId > 20 || !position) return;
    byId.set(landmarkId, {
      landmarkId,
      score: numberValue(keypoint?.score),
      position: { x: numberValue(position.x), y: numberValue(position.y), z: numberValue(position.z) },
    });
  });
  return {
    score: numberValue(hand?.score),
    keypoints: HAND_LANDMARK_IDS.map((landmarkId) => byId.get(landmarkId) ?? { landmarkId, score: 0, position: { x: 0, y: 0, z: 0 } }),
    observed: Boolean(hand?.observed),
    flipHeld: Boolean(hand?.flipHeld),
  };
}

export function normalizeSequence(value: unknown, canvasWidth = 513, canvasHeight = 513): NormalizedSequence {
  const payload = record(value);
  if (!Array.isArray(payload?.frames)) throw new Error('Invalid sequence JSON: missing frames array');
  const meta = record(payload.meta);
  const frames: NormalizedSequence['frames'] = [];
  payload.frames.forEach((rawFrame) => {
    const frame = record(rawFrame);
    if (!frame) return;
    const pose = normalizePose(frame.pose);
    if (!pose) return;
    const hands = record(frame.hands);
    frames.push({
      pose,
      face: normalizeFace(frame.face),
      hands: {
        left: normalizeHand(hands?.left ?? frame.leftHand ?? frame.left_hand),
        right: normalizeHand(hands?.right ?? frame.rightHand ?? frame.right_hand),
      },
    });
  });
  if (!frames.length) throw new Error('Sequence has no valid frames');
  return {
    meta: {
      fps: numberValue(meta?.fps, 30),
      canvasWidth: numberValue(meta?.canvasWidth, canvasWidth),
      canvasHeight: numberValue(meta?.canvasHeight, canvasHeight),
    },
    frames,
  };
}