import argparse
import asyncio
import os
import json
import re
import math
import ssl
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import holoviews as hv

# The bundled Windows Python runtime cannot load the Windows certificate store
# while Tornado imports.  Panel only needs these defaults for optional HTTPS;
# this app serves plain HTTP on localhost.  Scope the fallback to Panel's
# import, then restore the normal HTTPS behaviour for the rest of the app.
_create_default_ssl_context = ssl.create_default_context


def _local_server_ssl_context(purpose=ssl.Purpose.SERVER_AUTH, *args, **kwargs):
    protocol = (
        ssl.PROTOCOL_TLS_CLIENT
        if purpose == ssl.Purpose.SERVER_AUTH
        else ssl.PROTOCOL_TLS_SERVER
    )
    context = ssl.SSLContext(protocol)
    if purpose == ssl.Purpose.SERVER_AUTH:
        context.check_hostname = False
        context.verify_mode = ssl.CERT_NONE
    return context


ssl.create_default_context = _local_server_ssl_context
try:
    import panel as pn
finally:
    ssl.create_default_context = _create_default_ssl_context

# Discover available keypoint CSV files recursively under DATA_DIR
DATA_DIR = Path(r"SLclean")

def discover_clips() -> Dict[str, Path]:
    files = {}
    if DATA_DIR.exists():
        for p in DATA_DIR.rglob("*.csv"):
            # Use path relative to DATA_DIR as the selection key
            try:
                key = str(p.relative_to(DATA_DIR)).replace('\\', '/')
            except Exception:
                key = str(p)
            files[key] = p
    return files

CLIPS = discover_clips()

# Optional lexicon mapping: word -> list of file keys (relative CSV paths under DATA_DIR)
def load_lexicon() -> Dict[str, List[str]]:
    path = DATA_DIR / 'lexicon.json'
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    out: Dict[str, List[str]] = {}
    for k, v in data.items():
        key = str(k).strip().lower()
        if isinstance(v, str):
            out[key] = [v.strip().lower()]
        elif isinstance(v, list):
            out[key] = [str(x).strip().lower() for x in v]
    return out

LEXICON = load_lexicon()

# Helpers to load and concatenate multiple CSVs into a continuous sequence
EXPECTED_COLS = ['frame', 'part', 'landmark_id', 'x', 'y']

# ปรับค่าเริ่มต้นเพื่อความลื่นไหลมากขึ้น
DEFAULT_PAUSE_FRAMES = 0  # เพิ่มจาก 1 เป็น 3
DEFAULT_TARGET_FPS = 1200 # target playback frames per second
DEFAULT_INTERVAL_SEC = 1 / DEFAULT_TARGET_FPS
DEFAULT_INTERVAL_MS = max(1, int(round(1000.0 / DEFAULT_TARGET_FPS)))
UPSAMPLE_FACTOR = 7  # เพิ่มจาก 60 เป็น 80
SMOOTH_WINDOW = 25  # เพิ่มจาก 9 เป็น 15 สำหรับความนุ่มนวลมากขึ้น

# เปิดใช้งาน transitions เพื่อความต่อเนื่อง
ENABLE_TRANSITIONS = True
TRANSITION_FRAMES = 0  # เพิ่มจาก 0 เป็น 10
TRANSITION_PARTS: Tuple[str, ...] = ('left_hand', 'right_hand', 'pose')  # กำหนดส่วนที่จะทำ transition
TRANSITION_ENFORCE_BONE = True  # เปลี่ยนเป็น True

TRANSITION_SMOOTH_WINDOW = 5  # เพิ่มจาก 1 เป็น 7
TRANSITION_SMOOTH_FACTOR = 5  # เพิ่มจาก 0 เป็น 5


# Ensure Panel/Tornado use a selector loop on Windows (needed for Tornado <=6.1)
if os.name == 'nt':
    try:
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
    except AttributeError:
        pass

def _sanitize_clip_df(cdf: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in EXPECTED_COLS if c in cdf.columns]
    if len(cols) < len(EXPECTED_COLS):
        return pd.DataFrame(columns=EXPECTED_COLS)
    cdf = cdf[EXPECTED_COLS].copy()
    # Coerce types to reduce memory and avoid concat alignment issues
    for col, dtype in [('frame', 'int32'), ('landmark_id', 'int32')]:
        cdf[col] = pd.to_numeric(cdf[col], errors='coerce').astype(dtype, copy=False)
    for col in ['x', 'y']:
        cdf[col] = pd.to_numeric(cdf[col], errors='coerce').astype('float32', copy=False)
    cdf['part'] = cdf['part'].astype('string')
    cdf = cdf.dropna(subset=EXPECTED_COLS)
    return cdf

def load_clip_df(key: str) -> pd.DataFrame:
    p = CLIPS.get(key)
    if p is None:
        return pd.DataFrame(columns=EXPECTED_COLS)
    try:
        cdf = pd.read_csv(str(p))
    except Exception:
        return pd.DataFrame(columns=EXPECTED_COLS)
    return _sanitize_clip_df(cdf)

# เพิ่มฟังก์ชันสำหรับสร้าง transition ระหว่างคลิป
def synth_transition(
    clip1: pd.DataFrame,
    clip2: pd.DataFrame,
    n_frames: int = 10,
    parts: Tuple[str, ...] = (),
    enforce_bone: bool = False,
    smooth_factor: int = 5,
    smooth_window: int = 7,
) -> pd.DataFrame:
    """สร้าง transition frames ระหว่าง 2 คลิป"""
    if clip1.empty or clip2.empty or n_frames <= 0:
        return pd.DataFrame(columns=EXPECTED_COLS)
    
    # ใช้ frame สุดท้ายของ clip1 และ frame แรกของ clip2
    last_frame1 = clip1['frame'].max()
    first_frame2 = clip2['frame'].min()
    
    end_state = clip1[clip1['frame'] == last_frame1].copy()
    start_state = clip2[clip2['frame'] == first_frame2].copy()
    
    if parts:
        end_state = end_state[end_state['part'].isin(parts)]
        start_state = start_state[start_state['part'].isin(parts)]
    
    # หา landmarks ที่มีทั้งสองฝั่ง
    end_keys = set(zip(end_state['part'], end_state['landmark_id']))
    start_keys = set(zip(start_state['part'], start_state['landmark_id']))
    common_keys = end_keys & start_keys
    
    if not common_keys:
        return pd.DataFrame(columns=EXPECTED_COLS)
    
    trans_frames = []
    for i in range(1, n_frames + 1):
        # ใช้ sigmoid curve สำหรับ interpolation ที่นุ่มนวล
        t = i / (n_frames + 1)
        # Sigmoid smooth step
        smooth_t = 3 * t**2 - 2 * t**3  # Smoothstep function
        
        frame_data = []
        for (part, lid) in common_keys:
            end_pt = end_state[(end_state['part'] == part) & (end_state['landmark_id'] == lid)]
            start_pt = start_state[(start_state['part'] == part) & (start_state['landmark_id'] == lid)]
            
            if end_pt.empty or start_pt.empty:
                continue
            
            x1, y1 = end_pt.iloc[0][['x', 'y']]
            x2, y2 = start_pt.iloc[0][['x', 'y']]
            
            # Interpolate ด้วย smooth curve
            new_x = x1 + (x2 - x1) * smooth_t
            new_y = y1 + (y2 - y1) * smooth_t
            
            frame_data.append({
                'frame': i - 1,
                'part': part,
                'landmark_id': lid,
                'x': new_x,
                'y': new_y
            })
        
        if frame_data:
            trans_frames.append(pd.DataFrame(frame_data))
    
    if not trans_frames:
        return pd.DataFrame(columns=EXPECTED_COLS)
    
    result = pd.concat(trans_frames, ignore_index=True)
    return result[EXPECTED_COLS]

def build_sequence_from_tokens(keys: List[str], pause_frames: int = DEFAULT_PAUSE_FRAMES) -> pd.DataFrame:
    seq_parts = []
    frame_offset = 0
    prev_norm = None

    for key in keys:
        cdf = load_clip_df(key)
        if cdf.empty:
            continue

        norm_clip = cdf.copy()
        base0 = int(norm_clip['frame'].min()) if not norm_clip.empty else 0
        norm_clip['frame'] = norm_clip['frame'] - base0
        norm_clip = norm_clip.sort_values(['frame', 'part', 'landmark_id']).reset_index(drop=True)
        norm_clip['frame'] = norm_clip['frame'].astype('int32')
        norm_clip['landmark_id'] = norm_clip['landmark_id'].astype('int32')
        norm_clip['x'] = norm_clip['x'].astype('float32')
        norm_clip['y'] = norm_clip['y'].astype('float32')
        norm_clip['part'] = norm_clip['part'].astype('string')

        if (
            ENABLE_TRANSITIONS
            and TRANSITION_FRAMES > 0
            and prev_norm is not None
            and not prev_norm.empty
        ):
            trans_df = synth_transition(
                prev_norm,
                norm_clip,
                n_frames=TRANSITION_FRAMES,
                parts=TRANSITION_PARTS,
                enforce_bone=TRANSITION_ENFORCE_BONE,
                smooth_factor=TRANSITION_SMOOTH_FACTOR,
                smooth_window=TRANSITION_SMOOTH_WINDOW,
            )
            if not trans_df.empty:
                trans_df = trans_df.sort_values(['frame', 'part', 'landmark_id']).reset_index(drop=True)
                trans_df['frame'] = trans_df['frame'].astype('int32')
                trans_df['landmark_id'] = trans_df['landmark_id'].astype('int32')
                trans_df['x'] = trans_df['x'].astype('float32')
                trans_df['y'] = trans_df['y'].astype('float32')
                trans_df['part'] = trans_df['part'].astype('string')
                trans_df['frame'] = trans_df['frame'] + frame_offset
                seq_parts.append(trans_df)
                frame_offset = int(trans_df['frame'].max()) + 1

        shifted_clip = norm_clip.copy()
        shifted_clip['frame'] = shifted_clip['frame'] + frame_offset
        seq_parts.append(shifted_clip)

        last_frame = int(shifted_clip['frame'].max()) if not shifted_clip.empty else frame_offset - 1
        frame_offset = last_frame + 1

        if pause_frames > 0 and not shifted_clip.empty:
            last_rows = shifted_clip[shifted_clip['frame'] == last_frame]
            for offset in range(1, pause_frames + 1):
                hold = last_rows.copy()
                hold['frame'] = last_frame + offset
                seq_parts.append(hold)
            frame_offset = last_frame + pause_frames + 1

        prev_norm = norm_clip.copy()

    if not seq_parts:
        return pd.DataFrame(columns=EXPECTED_COLS)

    combined = pd.concat(seq_parts, ignore_index=True, copy=False)
    return combined[EXPECTED_COLS]


def _normalize_frames(seq_df: pd.DataFrame) -> pd.DataFrame:
    if seq_df.empty:
        return seq_df
    seq_df = seq_df.sort_values(['frame', 'part', 'landmark_id']).copy()
    base = int(seq_df['frame'].min())
    seq_df['frame'] = seq_df['frame'].astype('int32') - base
    return seq_df


def _upsample_sequence(seq_df: pd.DataFrame, factor: int) -> pd.DataFrame:
    """ปรับปรุง upsampling ให้ลื่นขึ้นด้วย spline interpolation"""
    if factor <= 1 or seq_df.empty:
        return seq_df
    seq_df = seq_df.copy()
    seq_df['frame'] = seq_df['frame'].astype('int32') * factor
    groups = []
    for (part, landmark_id), group in seq_df.groupby(['part', 'landmark_id'], sort=False):
        g = group.sort_values('frame').set_index('frame')
        start = int(g.index.min())
        end = int(g.index.max())
        full_index = range(start, end + 1)
        g = g.reindex(full_index)
        
        # ใช้ cubic spline แทน linear interpolation สำหรับความลื่นไหล
        if len(g) > 3:  # cubic ต้องการอย่างน้อย 4 จุด
            g[['x', 'y']] = g[['x', 'y']].interpolate(method='cubic')
        else:
            g[['x', 'y']] = g[['x', 'y']].interpolate(method='linear')
        
        g[['x', 'y']] = g[['x', 'y']].ffill().bfill()
        g['part'] = part
        g['landmark_id'] = landmark_id
        groups.append(g.reset_index().rename(columns={'index': 'frame'}))
    out = pd.concat(groups, ignore_index=True)
    return out[['frame', 'part', 'landmark_id', 'x', 'y']]


def _smooth_sequence(seq_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """ปรับปรุง smoothing ด้วย Gaussian window"""
    if window <= 1 or seq_df.empty:
        return seq_df
    seq_df = seq_df.sort_values(['part', 'landmark_id', 'frame'])

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        g = group.set_index('frame')
        # ใช้ Gaussian window สำหรับ smoothing ที่ดีกว่า
        g[['x', 'y']] = g[['x', 'y']].rolling(
            window=window, 
            center=True, 
            min_periods=1,
            win_type='gaussian'
        ).mean(std=window/4)  # std กำหนดความกว้างของ Gaussian
        return g.reset_index()

    smoothed = seq_df.groupby(['part', 'landmark_id'], sort=False, group_keys=False).apply(_apply)
    return smoothed[['frame', 'part', 'landmark_id', 'x', 'y']]


def prepare_sequence(seq_df: pd.DataFrame, upsample_factor: int = UPSAMPLE_FACTOR, smooth_window: int = SMOOTH_WINDOW) -> pd.DataFrame:
    if seq_df.empty:
        return seq_df
    upsample_factor = max(1, int(upsample_factor))
    smooth_window = max(1, int(smooth_window))
    if smooth_window % 2 == 0:
        smooth_window += 1
    seq_df = _normalize_frames(seq_df)
    seq_df = _upsample_sequence(seq_df, upsample_factor)
    seq_df = _smooth_sequence(seq_df, smooth_window)
    seq_df = _normalize_frames(seq_df)
    seq_df = seq_df.sort_values(['frame', 'part', 'landmark_id']).reset_index(drop=True)
    seq_df = seq_df.astype({'frame': 'int32', 'landmark_id': 'int32', 'x': 'float32', 'y': 'float32'})
    seq_df['part'] = seq_df['part'].astype('string')
    return seq_df

def _normalize_name(name: str) -> str:
    # Get filename without path
    name = Path(name).name
    # Remove extension (.csv)
    if '.' in name:
        name = name.rsplit('.', 1)[0]
    name = name.lower()
    # Remove unwanted suffixes
    for suf in ["_holistic_keypoints", "-holistic_keypoints", "_keypoints", "-keypoints"]:
        if name.endswith(suf):
            name = name[: -len(suf)]
            break
    # Replace special chars with space
    for ch in ['_', '-']:
        name = name.replace(ch, ' ')
    return name.strip()


def _build_search_index() -> List[Tuple[str, str]]:
    out = []
    for key in CLIPS.keys():
        base = Path(key).name
        out.append((key, _normalize_name(base)))
    return out


SEARCH_INDEX = _build_search_index()

# Do NOT auto-load all clips by default to avoid high memory usage.
df = pd.DataFrame(columns=EXPECTED_COLS)
CURRENT_RAW_DF = pd.DataFrame(columns=EXPECTED_COLS)

connections_by_part = {
    'face': [
        [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109, 10],
        [336, 296, 334, 293, 300, 276, 283, 282, 295, 285],
        [70, 63, 105, 66, 107, 55, 65, 52, 53, 46],
        [168, 6, 197, 195, 5, 4],
        [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
        [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],
        [78, 95, 88, 178, 87, 14, 317, 402, 318, 324, 308, 78],
    ],
    'pose': [
        [11, 12, 13, 14, 12, 11, 23, 24, 12],
        [11, 13, 15, 17, 19, 15, 21],
        [12, 14, 16, 18, 20, 16, 22],
        [23, 25, 27, 29, 31, 27, 29, 31],
        [24, 26, 28, 30, 32, 28, 30, 32],
    ],
    'left_hand': [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
        [5, 9, 13, 17],
    ],
    'right_hand': [
        [0, 1, 2, 3, 4],
        [0, 5, 6, 7, 8],
        [0, 9, 10, 11, 12],
        [0, 13, 14, 15, 16],
        [0, 17, 18, 19, 20],
        [5, 9, 13, 17],
    ],
}

# --- AVATAR STYLING & RENDERING ---

# --- AVATAR STYLING & RENDERING ---

# Define styles for the cartoon avatar
AVATAR_STYLES = {
    'skin': '#F5D0B0',      # Skin tone
    'shirt': '#3A86FF',     # Shirt color
    'pants': '#3A3A3A',     # Pants color
    'hair': '#2C2C2C',      # Hair color
    'face_border': '#E0B090',
    'arm_width': 30,        # Bigger arms (was 15)
    'leg_width': 20,        # Thicker legs
    'neck': '#E5C0A0',      # Slightly darker skin for neck
}

frames = sorted(df['frame'].unique().tolist())

# Helper to extract points for a specific list of landmarks
def get_points(part_df, landmark_ids):
    sub = part_df[part_df['landmark_id'].isin(landmark_ids)]
    if sub.empty:
        return np.array([])
    # Sort by the order in landmark_ids
    order_map = {lid: i for i, lid in enumerate(landmark_ids)}
    sub = sub.copy()
    sub['_ord'] = sub['landmark_id'].map(order_map)
    sub = sub.sort_values('_ord')
    return sub[['x', 'y']].to_numpy()

def build_paths_for_part(part_df: pd.DataFrame, connections):
    paths = []
    for conn in connections:
        ordered = list(dict.fromkeys(conn))
        sub = part_df[part_df['landmark_id'].isin(ordered)].copy()
        if sub.empty:
            continue
        order_map = {lid: i for i, lid in enumerate(ordered)}
        sub['_ord'] = sub['landmark_id'].map(order_map)
        sub = sub.sort_values('_ord')
        arr = sub[['x', 'y']].to_numpy()
        if arr.size:
            paths.append(arr)
    return paths

def render_index(i: int):
    i = int(i)
    if not frames:
        return hv.Curve([])
    i = max(0, min(i, len(frames) - 1))
    frame_val = frames[i]
    fdf = df[df['frame'] == frame_val]
    
    elements = []
    
    # --- 0. NECK (Behind everything) ---
    pose_df = fdf[fdf['part'] == 'pose']
    face_df = fdf[fdf['part'] == 'face']
    
    if not pose_df.empty and not face_df.empty:
        # Neck: Shoulders (11, 12) -> Jaw (152, 148, 176? Let's use 152 as chin)
        # We need a width for the neck. Let's approximate points between shoulder and chin
        shoulders = get_points(pose_df, [11, 12])
        chin = get_points(face_df, [152]) # Chin tip
        
        if shoulders.shape[0] == 2 and chin.shape[0] == 1:
            s1, s2 = shoulders[0], shoulders[1]
            c = chin[0]
            # Simple trapezoid neck
            neck_width = (s2[0] - s1[0]) * 0.4
            
            # Interpolate neck base on shoulders
            n1 = s1 + (s2 - s1) * 0.4
            n2 = s1 + (s2 - s1) * 0.6
            
            # Neck top (near chin/jaw)
            # Find jaw corners roughly (Face 58, 288 or nearby). Let's just go straight up from shoulders to jaw height
            elements.append(hv.Polygons([np.array([n1, n2, [n2[0], c[1]], [n1[0], c[1]]])], label='Neck').opts(
                color=AVATAR_STYLES['neck'], line_color=None
            ))

    # --- 1. TORSO (Shirt) ---
    if not pose_df.empty:
        torso_pts = get_points(pose_df, [11, 12, 24, 23]) 
        if torso_pts.shape[0] == 4:
            elements.append(hv.Polygons([torso_pts], label='Torso').opts(
                color=AVATAR_STYLES['shirt'], line_color=None
            ))
            
    # --- 2. FACE & FEATURES ---
    if not face_df.empty:
        # HAIR (Background layer behind face or top layer? Let's simply offset the forehead up)
        # Upper face contour indices (approximate top half)
        face_contour_ids = connections_by_part['face'][0]
        # Top half are usually the middle indices of the contour loop or specific forehead landmarks (10 is top)
        # Let's simple create a "Helmet" of hair based on the top of the head
        
        face_pts = get_points(face_df, face_contour_ids)
        if face_pts.size > 0:
            # HAIR HELMET (Solid shape behind face)
            hair_pts = face_pts.copy()
            center = np.mean(face_pts, axis=0)
            
            # Vector from center
            diff = hair_pts - center
            
            # Style: Widen slightly everywhere
            diff[:, 0] *= 1.15
            
            # Vertical: Stretch TOP parts UP significantly (Volume)
            # Y < center means Top in screen coords
            mask_top = diff[:, 1] < 0
            diff[mask_top, 1] *= 2.2 
            
            # Vertical: Taper BOTTOM slightly so it doesn't look like a huge beard
            mask_bottom = diff[:, 1] > 0
            diff[mask_bottom, 1] *= 0.9
            
            hair_poly = center + diff
            
            # Render as solid Polygon BEHIND face
            elements.append(hv.Polygons([hair_poly]).opts(color=AVATAR_STYLES['hair'], line_color=None))
        
        # Base Face
        if face_pts.size > 0:
            elements.append(hv.Polygons([face_pts], label='Face').opts(
                color=AVATAR_STYLES['skin'], line_color=AVATAR_STYLES['face_border'], line_width=2
            ))
        
        # Features (Brows, Eyes, Mouth)
        for i in [1, 2]: # Brows
            brow_pts = build_paths_for_part(face_df, [connections_by_part['face'][i]])
            if brow_pts:
                elements.append(hv.Path(brow_pts, label='Brow').opts(color='#5D4037', line_width=2))
        
        # Eyes
        eye_r_pts = get_points(face_df, connections_by_part['face'][4]) 
        if eye_r_pts.size > 0:
             elements.append(hv.Polygons([eye_r_pts], label='EyeR').opts(color='white', line_color='black'))
             cx, cy = np.mean(eye_r_pts, axis=0) # Pupil
             elements.append(hv.Points([(cx, cy)]).opts(color='black', size=3))

        eye_l_pts = get_points(face_df, connections_by_part['face'][5])
        if eye_l_pts.size > 0:
             elements.append(hv.Polygons([eye_l_pts], label='EyeL').opts(color='white', line_color='black'))
             cx, cy = np.mean(eye_l_pts, axis=0)
             elements.append(hv.Points([(cx, cy)]).opts(color='black', size=3))

        # Mouth
        mouth_pts = get_points(face_df, connections_by_part['face'][6])
        if mouth_pts.size > 0:
             elements.append(hv.Polygons([mouth_pts], label='Mouth').opts(color='#D81B60', line_color='#880E4F'))

    # --- 3. LIMBS & JOINTS ---
    if not pose_df.empty:
        # Joints Data
        joints = {
            'shoulder_l': 11, 'shoulder_r': 12,
            'elbow_l': 13, 'elbow_r': 14,
            # 'wrist_l': 15, 'wrist_r': 16, # Hands cover wrists
            'hip_l': 23, 'hip_r': 24,
            'knee_l': 25, 'knee_r': 26,
            'ankle_l': 27, 'ankle_r': 28
        }
        
        joint_pts = {}
        for name, lid in joints.items():
            pts = get_points(pose_df, [lid])
            if pts.size > 0:
                joint_pts[name] = pts[0]

        # Draw Arms (Sleeves!)
        # Split into Upper Arm (Shirt) and Forearm (Skin)
        # Left Arm
        l_upper = get_points(pose_df, [11, 13])
        l_lower = get_points(pose_df, [13, 15])
        if l_upper.size > 0:
            elements.append(hv.Path([l_upper], label='ArmL_Up').opts(color=AVATAR_STYLES['shirt'], line_width=AVATAR_STYLES['arm_width'], line_cap='round'))
        if l_lower.size > 0:
            elements.append(hv.Path([l_lower], label='ArmL_Low').opts(color=AVATAR_STYLES['skin'], line_width=AVATAR_STYLES['arm_width'], line_cap='round'))

        # Right Arm
        r_upper = get_points(pose_df, [12, 14])
        r_lower = get_points(pose_df, [14, 16])
        if r_upper.size > 0:
            elements.append(hv.Path([r_upper], label='ArmR_Up').opts(color=AVATAR_STYLES['shirt'], line_width=AVATAR_STYLES['arm_width'], line_cap='round'))
        if r_lower.size > 0:
            elements.append(hv.Path([r_lower], label='ArmR_Low').opts(color=AVATAR_STYLES['skin'], line_width=AVATAR_STYLES['arm_width'], line_cap='round'))
            
        # Draw Legs
        left_leg = get_points(pose_df, [23, 25, 27]) 
        right_leg = get_points(pose_df, [24, 26, 28])
        if left_leg.size > 0:
            elements.append(hv.Path([left_leg], label='LegL').opts(color=AVATAR_STYLES['pants'], line_width=AVATAR_STYLES['leg_width'], line_cap='round'))
        if right_leg.size > 0:
            elements.append(hv.Path([right_leg], label='LegR').opts(color=AVATAR_STYLES['pants'], line_width=AVATAR_STYLES['leg_width'], line_cap='round'))

        # Draw Joint Patches
        for j in ['shoulder_l', 'shoulder_r']:
            if j in joint_pts:
                elements.append(hv.Points([joint_pts[j]]).opts(color=AVATAR_STYLES['shirt'], size=AVATAR_STYLES['arm_width']))
        for j in ['hip_l', 'hip_r']:
            if j in joint_pts:
                elements.append(hv.Points([joint_pts[j]]).opts(color=AVATAR_STYLES['pants'], size=AVATAR_STYLES['leg_width'])) 
        for j in ['knee_l', 'knee_r']:
            if j in joint_pts:
                elements.append(hv.Points([joint_pts[j]]).opts(color=AVATAR_STYLES['pants'], size=AVATAR_STYLES['leg_width']))
        for j in ['elbow_l', 'elbow_r']:
            if j in joint_pts:
                elements.append(hv.Points([joint_pts[j]]).opts(color=AVATAR_STYLES['skin'], size=AVATAR_STYLES['arm_width']))

    # --- 4. HANDS (Realism) ---
    for hand_part in ['left_hand', 'right_hand']:
        hdf = fdf[fdf['part'] == hand_part]
        if not hdf.empty:
            # 1. Palm (Wrist 0 -> Knuckles 5, 9, 13, 17)
            # We want a solid shape for the hand palm
            palm_ids = [0, 5, 9, 13, 17, 0] 
            palm_pts = get_points(hdf, palm_ids)
            if palm_pts.size > 2:
                elements.append(hv.Polygons([palm_pts]).opts(
                    color=AVATAR_STYLES['skin'], line_width=0
                ))

            # 2. Fingers (Thicker fleshy lines)
            conns = connections_by_part[hand_part]
            finger_paths = build_paths_for_part(hdf, conns)
            if finger_paths:
                elements.append(hv.Path(finger_paths).opts(
                    color=AVATAR_STYLES['skin'], line_width=8, line_cap='round', line_join='round'
                ))
            
            # 3. Knuckles/Joints (Smoothness)
            # Add points at all key joints to round off connections
            elements.append(hv.Points(hdf, kdims=['x', 'y']).opts(
                color=AVATAR_STYLES['skin'], size=8
            ))


    if not elements:
        return hv.Curve([])
        
    overlay = hv.Overlay(elements)
    # Responsive sizing
    return overlay.opts(
        responsive=True, 
        aspect=1.0,  # Maintain aspect ratio
        xlim=(0, 1), 
        ylim=(1, 0), 
        xaxis=None, 
        yaxis=None,
        toolbar=None
    )



# --- VOSK & THREADING SETUP ---
import threading
import zipfile
from urllib.request import urlretrieve

# Global state for speech recognition
SPEECH_STATE = {
    'is_listening': False,
    'text': '',
    'status': 'Ready',
    'stop_event': threading.Event()
}

def download_vosk_model(model_name="vosk-model-small-en-us-0.15"):
    if Path(model_name).exists():
        return model_name
    if Path("model").exists():
        return "model"
        
    print(f"Model not found. Downloading {model_name}...")
    url = f"https://alphacephei.com/vosk/models/{model_name}.zip"
    zip_path = f"{model_name}.zip"
    try:
        urlretrieve(url, zip_path)
        print("Unzipping model...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(".")
        os.remove(zip_path)
        print("Model downloaded successfully.")
        return model_name
    except Exception as e:
        print(f"Failed to download model: {e}")
        return None

def threaded_listen():
    """Runs in a separate thread to avoid blocking the UI"""
    SPEECH_STATE['status'] = 'Initializing...'
    
    try:
        from vosk import Model, KaldiRecognizer
        import pyaudio
    except ImportError:
        SPEECH_STATE['status'] = 'Error: Missing Libs'
        SPEECH_STATE['is_listening'] = False
        return

    model_path = "model"
    if not Path(model_path).exists():
        # Try to find the downloaded folder
        found = False
        for p in Path('.').glob('vosk-model-*'):
            if p.is_dir():
                model_path = str(p)
                found = True
                break
        if not found:
            # Attempt auto-download
            downloaded = download_vosk_model()
            if downloaded:
                model_path = downloaded
            else:
                SPEECH_STATE['status'] = 'Error: No Model'
                SPEECH_STATE['is_listening'] = False
                return

    try:
        SPEECH_STATE['status'] = 'Listening...'
        print(f"Loading model from {model_path}...")
        model = Model(str(model_path))
        rec = KaldiRecognizer(model, 16000)
        
        pa = pyaudio.PyAudio()
        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=8000)
        stream.start_stream()
        
        # Listen until stop event or silence timeout
        count = 0
        max_chunks = 50 # approx 10-15 seconds
        
        while SPEECH_STATE['is_listening'] and count < max_chunks:
            if SPEECH_STATE['stop_event'].is_set():
                break
                
            data = stream.read(4000, exception_on_overflow=False)
            if len(data) == 0:
                break
                
            if rec.AcceptWaveform(data):
                import json as _json
                res = _json.loads(rec.Result())
                txt = res.get('text', '')
                if txt:
                    SPEECH_STATE['text'] = txt
                    print(f"Heard: {txt}")
                    break
            count += 1
            
        # Check partial result if no full result
        if not SPEECH_STATE['text']:
            import json as _json
            res = _json.loads(rec.FinalResult())
            SPEECH_STATE['text'] = res.get('text', '')

        stream.stop_stream()
        stream.close()
        pa.terminate()
        
    except Exception as e:
        print(f"Worker Error: {e}")
        SPEECH_STATE['status'] = 'Error'
    
    SPEECH_STATE['is_listening'] = False
    SPEECH_STATE['status'] = 'Done'


def build_panel_app():
    hv.extension('bokeh')
    pn.extension(sizing_mode='stretch_both')

    # Make plots responsive to container size
    hv.opts.defaults(
        hv.opts.Overlay(responsive=True, min_height=400, show_legend=False),
        hv.opts.Path(responsive=True),
        hv.opts.Points(responsive=True),
    )

    # Type words, match to CSV basenames under DATA_DIR
    text = pn.widgets.TextInput(name='พิมพ์คำศัพท์ (คั่นด้วยช่องว่าง)', placeholder='เช่น: hello thankyou หรือคำในชื่อไฟล์')
    status = pn.pane.Markdown('')
    
    # Voice Button
    voice_btn = pn.widgets.Button(name='🎤 กดเพื่อพูด', button_type='primary', width=300)
    
    def on_voice_process():
        """Periodic callback to check thread status"""
        if SPEECH_STATE['is_listening']:
            # Still listening
            voice_btn.name = f"🔴 {SPEECH_STATE['status']} (กดเพื่อหยุด)"
            voice_btn.button_type = 'warning'
        else:
            # Finished listening
            voice_btn.name = '🎤 กดเพื่อพูด'
            voice_btn.button_type = 'primary'
            
            # If we got text, update input
            if SPEECH_STATE['text']:
                print(f"Updating text input: {SPEECH_STATE['text']}")
                text.value = SPEECH_STATE['text']
                SPEECH_STATE['text'] = '' # Clear after use
                
    # Add periodic callback to UI via onload
    def _init_periodic():
        try:
            pn.state.add_periodic_callback(on_voice_process, period=500)
        except Exception as e:
            print(f"Warning: could not add periodic callback: {e}")

    def _on_voice_click(event):
        if not SPEECH_STATE['is_listening']:
            # Start
            SPEECH_STATE['is_listening'] = True
            SPEECH_STATE['text'] = ''
            SPEECH_STATE['stop_event'].clear()
            SPEECH_STATE['status'] = 'Starting...'
            t = threading.Thread(target=threaded_listen)
            t.daemon = True
            t.start()
        else:
            # Stop
            SPEECH_STATE['stop_event'].set()
            SPEECH_STATE['is_listening'] = False
            voice_btn.name = 'Stopping...'

    voice_btn.on_click(_on_voice_click)

    # ลดค่า interval เพื่อเล่นเร็วขึ้นและลื่นขึ้น
    # Calculate initial interval and step based on DEFAULT_TARGET_FPS
    initial_fps = max(1, int(DEFAULT_TARGET_FPS))
    # Enforce minimum interval of 20ms to avoid browser throttling
    MIN_INTERVAL_MS = 20
    initial_step = int(math.ceil((initial_fps / 1000.0) * MIN_INTERVAL_MS))
    initial_interval = MIN_INTERVAL_MS

    # Custom Player Implementation (Only Play Button)
    # Using IntSlider as the source of truth for the current frame
    frame_slider = pn.widgets.IntSlider(name='Frame', start=0, end=max(len(frames) - 1, 0), value=0, visible=False)
    play_btn = pn.widgets.Button(name='▶ Play', button_type='primary', width=100, align='center')

    # State for playback control
    playback = {
        'interval': initial_interval,
        'step': initial_step,
        'running': False
    }

    def advance_frame():
        if not playback['running']:
            return
        
        current = frame_slider.value
        end = frame_slider.end
        
        # If we are already at the end, stop
        if current >= end:
            stop_playing()
            return
            
        step = playback['step']
        new_val = current + step
        if new_val > end:
            new_val = end
        frame_slider.value = new_val

    # Periodic callback
    anim_cb = pn.state.add_periodic_callback(advance_frame, period=initial_interval, start=False)

    def start_playing():
        # Check if we should restart from beginning
        if frame_slider.value >= frame_slider.end:
            frame_slider.value = 0
        
        playback['running'] = True
        play_btn.name = '⏸ Pause'
        play_btn.button_type = 'warning'
        
        # Update callback period in case it changed
        anim_cb.period = playback['interval']
        if not anim_cb.running:
            anim_cb.start()

    def stop_playing():
        playback['running'] = False
        play_btn.name = '▶ Play'
        play_btn.button_type = 'primary'
        if anim_cb.running:
            anim_cb.stop()

    def toggle_play(event):
        if playback['running']:
            stop_playing()
        else:
            start_playing()

    play_btn.on_click(toggle_play)

    view = pn.bind(render_index, i=frame_slider)

    state = {'raw': CURRENT_RAW_DF.copy()}

    def _apply_playback_rate(target_fps):
        fps = max(1, int(target_fps))
        
        # Enforce minimum interval to avoid browser throttling (e.g. 1ms is too fast)
        min_interval = 20 
        
        # Calculate step size to maintain target FPS at this interval
        # Frames per ms = fps / 1000
        # Frames per interval = (fps / 1000) * min_interval
        step = int(math.ceil((fps / 1000.0) * min_interval))
        interval_ms = min_interval
        
        # Update playback state
        playback['interval'] = interval_ms
        playback['step'] = step
        
        # Update running callback if needed
        if playback['running']:
            anim_cb.period = interval_ms

    def apply_sequence():
        global df, frames
        raw_df = state['raw']
        if raw_df.empty:
            df = pd.DataFrame(columns=EXPECTED_COLS)
            frames = []
            frame_slider.end = 0
            frame_slider.value = 0
            stop_playing()
            return
        prepared = prepare_sequence(raw_df, UPSAMPLE_FACTOR, SMOOTH_WINDOW)
        df = prepared
        frames = sorted(df['frame'].unique().tolist())
        frame_slider.end = max(len(frames) - 1, 0)
        frame_slider.value = 0  # รีเซ็ตไปเฟรมแรก
        _apply_playback_rate(DEFAULT_TARGET_FPS)
        
        # Auto-play
        start_playing()

    def set_sequence(raw_df: pd.DataFrame):
        global CURRENT_RAW_DF
        CURRENT_RAW_DF = raw_df.copy()
        state['raw'] = raw_df.copy()
        apply_sequence()

    # Using default FPS and smoothing values

    _apply_playback_rate(DEFAULT_TARGET_FPS)

    def _on_input_change(event):
        raw = (event.new or '').strip().lower()


        # แยกคำและตัด is, am, are ออกตามหลัก gloss
        words = re.split(r"\s+", raw)
        filtered_words = [w for w in words if w and w not in ['is', 'am', 'are']]

        # รวมคำที่เหลือกลับเป็นประโยค
        search_term = ' '.join(filtered_words)

        keys = []
        missing_terms = []

        # ค้นหาคำเต็มก่อน (เช่น "thank you")
        exact_matches = [k for (k, norm) in SEARCH_INDEX if norm == search_term]
        if exact_matches:
            keys.append(exact_matches[0])
        else:
            # ถ้าไม่พบคำเต็ม ลองแยกคำและค้นหาทีละคำ
            for w in filtered_words:
                word_matches = [k for (k, norm) in SEARCH_INDEX if norm == w]
                if word_matches:
                    keys.append(word_matches[0])
                else:
                    missing_terms.append(w)

        if not keys:
            missing_text = ' '.join(missing_terms) if missing_terms else '(ว่าง)'
            status.object = (
                f"❌ ไม่พบไฟล์ที่ตรงกับคำ: {missing_text}\n"
                f"ไฟล์ทั้งหมด: {len(CLIPS)}"
            )
            return

        # แสดงสถานะ "กำลังสร้าง animation"
        status.object = '⏳ กำลังสร้างแอนิเมชัน...'
        
        new_df = build_sequence_from_tokens(keys)
        set_sequence(new_df)

        if missing_terms:
            status.object = f"⚠️ ไม่พบคำ: {' '.join(missing_terms)} | ✅ เล่นแอนิเมชันแล้ว"
        else:
            status.object = '✅ เล่นแอนิเมชันแล้ว'

    quit_btn = pn.widgets.Button(name='Quit', button_type='danger')

    def _quit(_):
        os._exit(0)

    quit_btn.on_click(_quit)

    text.param.watch(_on_input_change, 'value')

    apply_sequence()

    pn.state.onload(_init_periodic)

    # --- CENTERED CLEAN UI ---
    
    # Header / Title Area
    header = pn.Row(
        pn.layout.HSpacer(),
        pn.pane.Markdown("# 🎭 Avatar Animation", styles={'font-size': '24px', 'font-weight': 'bold'}),
        pn.layout.HSpacer(),
    )

    # Control Bar (Input + Play)
    control_bar = pn.Row(
        text,
        voice_btn,
        play_btn,
        align='center',
        sizing_mode='scale_width',
        max_width=800
    )
    
    status_bar = pn.Row(
        status,
        align='center'
    )

    # Main Layout
    # Use a Column with alignment to center everything
    main_col = pn.Column(
        header,
        pn.layout.Divider(),
        # The View (Avatar) - make it occupy good space
        pn.Column(
            view, 
            height=600, 
            sizing_mode='stretch_both', 
            align='center',
            css_classes=['avatar-container']
        ),
        pn.layout.Divider(),
        control_bar,
        status_bar,
        pn.layout.Divider(),
        quit_btn,
        sizing_mode='stretch_both',
        align='center' 
    )

    # Use VanillaTemplate or just render the main column full page
    template = pn.template.VanillaTemplate(
        title='Avatar Animation',
        theme='default'
    )
    template.main.append(main_col)
    
    return template

def main():
    parser = argparse.ArgumentParser(description='Keypoints Animation (HoloViews + Panel)')
    parser.add_argument('--serve', action='store_true', help='Run a live app with a Player that auto-plays')
    parser.add_argument('--files', type=str, default='', help='Relative CSV paths separated by space (e.g. "a/b/clip1.csv clip2.csv")')
    args = parser.parse_args()

    # Default: serve if possible; else export
    if args.files:
        parts = [t for t in args.files.split() if t]
        existing = set(CLIPS.keys())
        selected = []
        # Resolve each input as exact key or basename (unique)
        basename_to_keys = {}
        for k in existing:
            basename_to_keys.setdefault(Path(k).name.lower(), set()).add(k)
        for p in parts:
            k = p.replace('\\', '/')
            if k in existing:
                selected.append(k)
            else:
                base = Path(k).name.lower()
                matches = basename_to_keys.get(base, set())
                if len(matches) == 1:
                    selected.append(next(iter(matches)))
        if selected:
            raw_seq = build_sequence_from_tokens(selected)
            if not raw_seq.empty:
                globals()['CURRENT_RAW_DF'] = raw_seq.copy()
                seq_df = prepare_sequence(raw_seq)
                globals()['df'] = seq_df
                globals()['frames'] = sorted(seq_df['frame'].unique().tolist())

    try:
        # Serve the app on a random free port and open browser
        # Note: on Windows, sometimes threaded=True helps if there are event loop issues
        pn.serve(build_panel_app, show=True, port=0, start=True, threaded=False)
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f'Panel server failed to start: {exc}')


if __name__ == '__main__':
    main()
