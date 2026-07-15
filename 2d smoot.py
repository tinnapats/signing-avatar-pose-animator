import argparse
import asyncio
import os
import json
import base64
import re
import math
import socket
import ssl
from html import escape as _html_escape
from pathlib import Path
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import holoviews as hv

# Panel imports Tornado, which loads Windows certificate defaults during import.
# The bundled Python runtime cannot load that store; this local app serves HTTP
# only, so provide temporary TLS defaults for the import and restore HTTPS
# behaviour immediately afterwards.
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

BASE_DIR = Path(__file__).resolve().parent
POSE_ANIMATOR_DIR = Path(os.environ.get("POSE_ANIMATOR_DIR", BASE_DIR / "pose-animator")).expanduser()
VENDOR_DIR = POSE_ANIMATOR_DIR / "vendor"
DATA_DIR = Path(os.environ.get("SL_DATA_DIR", BASE_DIR / "SLclean")).expanduser()

def build_panel_static_dirs() -> Dict[str, str]:
    static_dirs: Dict[str, str] = {}
    if VENDOR_DIR.exists():
        static_dirs["three"] = os.fspath(VENDOR_DIR)
    if POSE_ANIMATOR_DIR.exists():
        static_dirs["threepanel"] = os.fspath(POSE_ANIMATOR_DIR)
    return static_dirs

PANEL_STATIC_DIRS = build_panel_static_dirs()

def build_websocket_origins(port: int) -> List[str]:
    override = os.environ.get("PANEL_WEBSOCKET_ORIGIN", "").strip()
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    if port <= 0:
        return ["*"]
    hosts = {
        "localhost",
        "127.0.0.1",
        socket.gethostname(),
        socket.getfqdn(),
    }
    return [f"{host}:{port}" for host in hosts if host]

# Allow HTML with scripts (needed for Three.js pane)
pn.config.sanitize_html = False
pn.config.static_dirs = getattr(pn.config, "static_dirs", {})
pn.config.static_dirs.update(PANEL_STATIC_DIRS)
# Cached Three.js bundle for inline use.
_THREE_JS_CACHE = None

def _load_three_js() -> str:
    global _THREE_JS_CACHE
    if _THREE_JS_CACHE is not None:
        return _THREE_JS_CACHE
    three_path = VENDOR_DIR / "three.min.js"
    try:
        _THREE_JS_CACHE = three_path.read_text(encoding="utf-8")
    except Exception:
        _THREE_JS_CACHE = ""
    return _THREE_JS_CACHE
# Discover available keypoint CSV files recursively under DATA_DIR

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

# เธเธฃเธฑเธเธเนเธฒเน€เธฃเธดเนเธกเธ•เนเธเน€เธเธทเนเธญเธเธงเธฒเธกเธฅเธทเนเธเนเธซเธฅเธกเธฒเธเธเธถเนเธ
DEFAULT_PAUSE_FRAMES = 0  # เน€เธเธดเนเธกเธเธฒเธ 1 เน€เธเนเธ 3
DEFAULT_TARGET_FPS = 120 # target playback frames per second
DEFAULT_INTERVAL_SEC = 1 / DEFAULT_TARGET_FPS
DEFAULT_INTERVAL_MS = max(1, int(round(1000.0 / DEFAULT_TARGET_FPS)))
UPSAMPLE_FACTOR = 7  # เน€เธเธดเนเธกเธเธฒเธ 60 เน€เธเนเธ 80
SMOOTH_WINDOW = 25  # เน€เธเธดเนเธกเธเธฒเธ 9 เน€เธเนเธ 15 เธชเธณเธซเธฃเธฑเธเธเธงเธฒเธกเธเธธเนเธกเธเธงเธฅเธกเธฒเธเธเธถเนเธ

# เน€เธเธดเธ”เนเธเนเธเธฒเธ transitions เน€เธเธทเนเธญเธเธงเธฒเธกเธ•เนเธญเน€เธเธทเนเธญเธ
ENABLE_TRANSITIONS = True
TRANSITION_FRAMES = 0  # เน€เธเธดเนเธกเธเธฒเธ 0 เน€เธเนเธ 10
TRANSITION_PARTS: Tuple[str, ...] = ('left_hand', 'right_hand', 'pose')  # เธเธณเธซเธเธ”เธชเนเธงเธเธ—เธตเนเธเธฐเธ—เธณ transition
TRANSITION_ENFORCE_BONE = True  # เน€เธเธฅเธตเนเธขเธเน€เธเนเธ True

TRANSITION_SMOOTH_WINDOW = 5  # เน€เธเธดเนเธกเธเธฒเธ 1 เน€เธเนเธ 7
TRANSITION_SMOOTH_FACTOR = 5  # เน€เธเธดเนเธกเธเธฒเธ 0 เน€เธเนเธ 5



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

# เน€เธเธดเนเธกเธเธฑเธเธเนเธเธฑเธเธชเธณเธซเธฃเธฑเธเธชเธฃเนเธฒเธ transition เธฃเธฐเธซเธงเนเธฒเธเธเธฅเธดเธ
def synth_transition(
    clip1: pd.DataFrame,
    clip2: pd.DataFrame,
    n_frames: int = 10,
    parts: Tuple[str, ...] = (),
    enforce_bone: bool = False,
    smooth_factor: int = 5,
    smooth_window: int = 7,
) -> pd.DataFrame:
    """เธชเธฃเนเธฒเธ transition frames เธฃเธฐเธซเธงเนเธฒเธ 2 เธเธฅเธดเธ"""
    if clip1.empty or clip2.empty or n_frames <= 0:
        return pd.DataFrame(columns=EXPECTED_COLS)
    
    # เนเธเน frame เธชเธธเธ”เธ—เนเธฒเธขเธเธญเธ clip1 เนเธฅเธฐ frame เนเธฃเธเธเธญเธ clip2
    last_frame1 = clip1['frame'].max()
    first_frame2 = clip2['frame'].min()
    
    end_state = clip1[clip1['frame'] == last_frame1].copy()
    start_state = clip2[clip2['frame'] == first_frame2].copy()
    
    if parts:
        end_state = end_state[end_state['part'].isin(parts)]
        start_state = start_state[start_state['part'].isin(parts)]
    
    # เธซเธฒ landmarks เธ—เธตเนเธกเธตเธ—เธฑเนเธเธชเธญเธเธเธฑเนเธ
    end_keys = set(zip(end_state['part'], end_state['landmark_id']))
    start_keys = set(zip(start_state['part'], start_state['landmark_id']))
    common_keys = end_keys & start_keys
    
    if not common_keys:
        return pd.DataFrame(columns=EXPECTED_COLS)
    
    trans_frames = []
    for i in range(1, n_frames + 1):
        # เนเธเน sigmoid curve เธชเธณเธซเธฃเธฑเธ interpolation เธ—เธตเนเธเธธเนเธกเธเธงเธฅ
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
            
            # Interpolate เธ”เนเธงเธข smooth curve
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
    """เธเธฃเธฑเธเธเธฃเธธเธ upsampling เนเธซเนเธฅเธทเนเธเธเธถเนเธเธ”เนเธงเธข spline interpolation"""
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
        
        # เนเธเน cubic spline เนเธ—เธ linear interpolation เธชเธณเธซเธฃเธฑเธเธเธงเธฒเธกเธฅเธทเนเธเนเธซเธฅ
        if len(g) > 3:  # cubic เธ•เนเธญเธเธเธฒเธฃเธญเธขเนเธฒเธเธเนเธญเธข 4 เธเธธเธ”
            try:
                g[['x', 'y']] = g[['x', 'y']].interpolate(method='cubic')
            except ImportError:
                g[['x', 'y']] = g[['x', 'y']].interpolate(method='linear')
        else:
            g[['x', 'y']] = g[['x', 'y']].interpolate(method='linear')
        
        g[['x', 'y']] = g[['x', 'y']].ffill().bfill()
        g['part'] = part
        g['landmark_id'] = landmark_id
        groups.append(g.reset_index().rename(columns={'index': 'frame'}))
    out = pd.concat(groups, ignore_index=True)
    return out[['frame', 'part', 'landmark_id', 'x', 'y']]


def _smooth_sequence(seq_df: pd.DataFrame, window: int) -> pd.DataFrame:
    """เธเธฃเธฑเธเธเธฃเธธเธ smoothing เธ”เนเธงเธข Gaussian window"""
    if window <= 1 or seq_df.empty:
        return seq_df
    seq_df = seq_df.sort_values(['part', 'landmark_id', 'frame'])

    def _apply(group: pd.DataFrame) -> pd.DataFrame:
        g = group.set_index('frame')
        # เนเธเน Gaussian window เธชเธณเธซเธฃเธฑเธ smoothing เธ—เธตเนเธ”เธตเธเธงเนเธฒ
        try:
            g[['x', 'y']] = g[['x', 'y']].rolling(
                window=window,
                center=True,
                min_periods=1,
                win_type='gaussian',
            ).mean(std=window / 4)
        except ImportError:
            g[['x', 'y']] = g[['x', 'y']].rolling(
                window=window,
                center=True,
                min_periods=1,
            ).mean()
        return g.reset_index()

    groups = []
    for (part, landmark_id), group in seq_df.groupby(['part', 'landmark_id'], sort=False):
        smoothed_group = _apply(group)
        smoothed_group['part'] = part
        smoothed_group['landmark_id'] = landmark_id
        groups.append(smoothed_group)
    smoothed = pd.concat(groups, ignore_index=True)
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

FACE_RENDER_FEATURES = {
    'oval': connections_by_part['face'][0],
    'left_brow': connections_by_part['face'][1],
    'right_brow': connections_by_part['face'][2],
    'nose_bridge': connections_by_part['face'][3],
    'left_eye': connections_by_part['face'][4],
    'right_eye': connections_by_part['face'][5],
    'mouth_line': [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291],
    'outer_lips': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146, 61],
    'inner_lips': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95, 78],
}

frames = sorted(df['frame'].unique().tolist())
styles = {
    'face': dict(color='gray', line_width=1),
    'pose': dict(color='black', line_width=2),
    'left_hand': dict(color='red', line_width=3),
    'right_hand': dict(color='blue', line_width=3),
}
hand_point_size = 6
POSE_HAND_LINKS = {
    'left_hand': (15, 0),
    'right_hand': (16, 0),
}

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

def build_pose_hand_bridge_path(
    pose_df: pd.DataFrame,
    hand_df: pd.DataFrame,
    pose_wrist_id: int,
    hand_anchor_id: int = 0,
):
    if pose_df.empty or hand_df.empty:
        return None
    pose_pt = pose_df[pose_df['landmark_id'] == pose_wrist_id][['x', 'y']].to_numpy()
    if pose_pt.size == 0:
        return None
    hand_pt = hand_df[hand_df['landmark_id'] == hand_anchor_id][['x', 'y']].to_numpy()
    if hand_pt.size == 0:
        fallback_ids = [5, 17]
        hand_pt = hand_df[hand_df['landmark_id'].isin(fallback_ids)][['x', 'y']].to_numpy()
        if hand_pt.size == 0:
            return None
        hand_pt = np.array([hand_pt.mean(axis=0)])
    return np.vstack([pose_pt[0], hand_pt[0]])

def render_index(i: int):
    i = int(i)
    if not frames:
        return hv.Curve([])
    i = max(0, min(i, len(frames) - 1))
    frame_val = frames[i]
    fdf = df[df['frame'] == frame_val]
    elements = []
    for part, conns in connections_by_part.items():
        part_df = fdf[fdf['part'] == part]
        if part_df.empty:
            continue
        paths = build_paths_for_part(part_df, conns)
        if paths:
            opts = styles.get(part, {})
            elements.append(hv.Path(paths, label=f'{part}').opts(**opts))
        if part in ('left_hand', 'right_hand') and not part_df.empty:
            color = styles[part]['color']
            elements.append(hv.Points(part_df, kdims=['x', 'y']).opts(size=hand_point_size, color=color))
    pose_df = fdf[fdf['part'] == 'pose']
    for part, (pose_wrist_id, hand_anchor_id) in POSE_HAND_LINKS.items():
        hand_df = fdf[fdf['part'] == part]
        bridge = build_pose_hand_bridge_path(pose_df, hand_df, pose_wrist_id, hand_anchor_id)
        if bridge is None:
            continue
        color = styles.get(part, {}).get('color', 'black')
        elements.append(hv.Path([bridge], label=f'{part}_bridge').opts(color=color, line_width=3))
    overlay = hv.Overlay(elements) if elements else hv.Curve([])
    return overlay.opts(xlim=(0, 1), ylim=(1, 0))


def _frame_df_by_index(i: int) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLS)
    i = max(0, min(int(i), len(frames) - 1))
    return df[df['frame'] == frames[i]]


def _safe_point(pts: Dict[int, Tuple[float, float]], key: int, fallback: Tuple[float, float]) -> Tuple[float, float]:
    return pts.get(key, fallback)


def render_css_avatar(i: int, width: int = 900, height: int = 620) -> str:
    fdf = _frame_df_by_index(i)
    pose = fdf[fdf['part'] == 'pose']
    pts: Dict[int, Tuple[float, float]] = {
        int(r.landmark_id): (float(r.x), float(r.y)) for _, r in pose.iterrows()
    }
    ls = _safe_point(pts, 11, (0.42, 0.38))
    rs = _safe_point(pts, 12, (0.58, 0.38))
    lh = _safe_point(pts, 23, (0.45, 0.58))
    rh = _safe_point(pts, 24, (0.55, 0.58))
    le = _safe_point(pts, 13, (0.36, 0.48))
    re = _safe_point(pts, 14, (0.64, 0.48))
    lw = _safe_point(pts, 15, (0.31, 0.58))
    rw = _safe_point(pts, 16, (0.69, 0.58))
    nose = _safe_point(pts, 0, ((ls[0] + rs[0]) / 2.0, 0.28))

    def px_x(x: float) -> float:
        return x * width

    def px_y(y: float) -> float:
        return y * height

    shoulder_c = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
    hip_c = ((lh[0] + rh[0]) / 2.0, (lh[1] + rh[1]) / 2.0)
    body_w = max(80.0, abs(px_x(rs[0]) - px_x(ls[0])) * 1.8)
    body_h = max(150.0, abs(px_y(hip_c[1]) - px_y(shoulder_c[1])) * 2.3)
    head = max(68.0, body_w * 0.48)
    left_arm_len = max(70.0, math.dist((px_x(ls[0]), px_y(ls[1])), (px_x(lw[0]), px_y(lw[1]))))
    right_arm_len = max(70.0, math.dist((px_x(rs[0]), px_y(rs[1])), (px_x(rw[0]), px_y(rw[1]))))
    left_ang = math.degrees(math.atan2(px_y(lw[1]) - px_y(ls[1]), px_x(lw[0]) - px_x(ls[0])))
    right_ang = math.degrees(math.atan2(px_y(rw[1]) - px_y(rs[1]), px_x(rw[0]) - px_x(rs[0])))

    html = f"""
<div style="width:{width}px;height:{height}px;position:relative;overflow:hidden;border-radius:18px;background:linear-gradient(160deg,#f3f8ff,#eefbf6);">
  <div style="position:absolute;left:{px_x(shoulder_c[0]) - body_w/2:.1f}px;top:{px_y(shoulder_c[1]) - 12:.1f}px;width:{body_w:.1f}px;height:{body_h:.1f}px;border-radius:44% 44% 32% 32%;background:#3b82f6;"></div>
  <div style="position:absolute;left:{px_x(nose[0]) - head/2:.1f}px;top:{px_y(nose[1]) - head*0.54:.1f}px;width:{head:.1f}px;height:{head*1.05:.1f}px;border-radius:50%;background:#ffd9b6;"></div>
  <div style="position:absolute;left:{px_x(nose[0]) - head*0.52:.1f}px;top:{px_y(nose[1]) - head*0.72:.1f}px;width:{head*1.04:.1f}px;height:{head*0.45:.1f}px;border-radius:80% 80% 30% 30%;background:#111827;"></div>
  <div style="position:absolute;left:{px_x(ls[0]) - 6:.1f}px;top:{px_y(ls[1]) - 6:.1f}px;width:{left_arm_len:.1f}px;height:12px;background:#1d4ed8;border-radius:999px;transform-origin:6px 6px;transform:rotate({left_ang:.1f}deg);"></div>
  <div style="position:absolute;left:{px_x(rs[0]) - 6:.1f}px;top:{px_y(rs[1]) - 6:.1f}px;width:{right_arm_len:.1f}px;height:12px;background:#1d4ed8;border-radius:999px;transform-origin:6px 6px;transform:rotate({right_ang:.1f}deg);"></div>
  <div style="position:absolute;left:{px_x(lw[0]) - 13:.1f}px;top:{px_y(lw[1]) - 13:.1f}px;width:26px;height:26px;border-radius:50%;background:#ffd9b6;"></div>
  <div style="position:absolute;left:{px_x(rw[0]) - 13:.1f}px;top:{px_y(rw[1]) - 13:.1f}px;width:26px;height:26px;border-radius:50%;background:#ffd9b6;"></div>
  <div style="position:absolute;left:18px;bottom:14px;color:#334155;font:600 14px/1.2 ui-sans-serif;">CSS Avatar Mode</div>
</div>
"""
    return html

def build_three_payload(seq_df: pd.DataFrame, fps: int) -> Dict:
    if seq_df.empty:
        return {"meta": {"fps": fps, "frameCount": 0}, "frames": []}
    grouped = seq_df.groupby("frame", sort=True)
    frames_out = []
    for frame_idx, frame_df in grouped:
        def _points_for(part_name: str) -> Dict[str, List[float]]:
            part_df = frame_df[frame_df["part"] == part_name]
            pts: Dict[str, List[float]] = {}
            for row in part_df[["landmark_id", "x", "y"]].itertuples(index=False):
                lid = int(row.landmark_id)
                pts[str(lid)] = [float(row.x), float(row.y)]
            return pts

        pose_points = _points_for("pose")
        face_points = _points_for("face")
        left_hand_points = _points_for("left_hand")
        right_hand_points = _points_for("right_hand")
        frames_out.append(
            {
                "frame": int(frame_idx),
                "pose": {"points": pose_points},
                "face": {"points": face_points},
                "left_hand": {"points": left_hand_points},
                "right_hand": {"points": right_hand_points},
            }
        )
    return {
        "meta": {"fps": int(fps), "frameCount": int(len(frames_out))},
        "frames": frames_out,
    }

def render_three_viewer_html(payload: Dict) -> str:
    data_json = json.dumps(payload, ensure_ascii=False)
    face_features_json = json.dumps(FACE_RENDER_FEATURES, ensure_ascii=False)
    template = """
<!doctype html>
<html>
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <style>
      html, body { margin: 0; padding: 0; height: 100%; overflow: hidden; background: #f8fafc; }
      #viewer-stage { position: relative; width: 100%; height: 100%; }
      #viewer-container { width: 100%; height: 100%; }
      #face-panel {
        display: none;
        position: absolute;
        right: 18px;
        top: 18px;
        width: 246px;
        height: 176px;
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.96);
        box-shadow: none;
        border: 1px solid rgba(148, 163, 184, 0.18);
        overflow: hidden;
        backdrop-filter: blur(3px);
      }
      #face-panel-label {
        position: absolute;
        left: 12px;
        top: 8px;
        z-index: 2;
        color: #64748b;
        font: 600 11px/1.2 sans-serif;
        letter-spacing: 0.02em;
        pointer-events: none;
      }
      #face-canvas {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
      }
      canvas { display: block; width: 100%; height: 100%; }
    </style>
  </head>
  <body>
    <div id="viewer-stage">
      <div id="viewer-container"><canvas id="viewer-canvas"></canvas></div>
      <div id="face-panel">
        <div id="face-panel-label">Face</div>
        <canvas id="face-canvas"></canvas>
      </div>
    </div>
    <script>
const payload = __PAYLOAD__;
const faceFeatures = __FACE_FEATURES__;
const frames = payload.frames || [];
const fps = Math.max(1, Number(payload.meta && payload.meta.fps ? payload.meta.fps : 30));
const parentDoc = window.parent && window.parent.document ? window.parent.document : null;
const syncEl = parentDoc ? parentDoc.getElementById('frame-sync') : null;
const useSync = !!syncEl;
let lastSync = -1;
const container = document.getElementById('viewer-container');
const canvas = document.getElementById('viewer-canvas');
const facePanel = document.getElementById('face-panel');
const faceCanvas = document.getElementById('face-canvas');
if (!container || !canvas || !facePanel) throw new Error('Skeleton canvas container not found');
const ctx = canvas.getContext('2d');
const faceCtx = faceCanvas.getContext('2d');
if (!ctx) throw new Error('2D canvas context not available');
if (!faceCtx) throw new Error('2D face canvas context not available');

const connections = [
  [11, 12], [11, 13], [13, 15], [12, 14], [14, 16],
  [11, 23], [12, 24], [23, 24], [23, 25], [25, 27],
  [24, 26], [26, 28]
];
const torsoConnections = [
  [11, 12], [11, 23], [12, 24], [23, 24]
];
const poseFacePointIds = new Set(['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10']);
const handConnections = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17]
];

function resizeCanvas() {
  const width = container.clientWidth || 800;
  const height = container.clientHeight || 520;
  canvas.width = width;
  canvas.height = height;
  const faceRect = facePanel.getBoundingClientRect();
  const faceWidth = Math.max(220, Math.round(faceRect.width || facePanel.clientWidth || 246));
  const faceHeight = Math.max(160, Math.round(faceRect.height || facePanel.clientHeight || 176));
  faceCanvas.width = faceWidth;
  faceCanvas.height = faceHeight;
}

function mapToCanvas(x, y) {
  const padX = canvas.width * 0.12;
  const padY = canvas.height * 0.06;
  return [
    padX + x * (canvas.width - padX * 2),
    padY + y * (canvas.height - padY * 2),
  ];
}

function clearCanvas() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = '#f8fafc';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
}

function clearFaceCanvas() {
  faceCtx.clearRect(0, 0, faceCanvas.width, faceCanvas.height);
  faceCtx.fillStyle = 'rgba(248, 250, 252, 0.98)';
  faceCtx.fillRect(0, 0, faceCanvas.width, faceCanvas.height);
}

function drawConnections(points, source, color, lineWidth) {
  if (!source || !Object.keys(source).length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  points.forEach(([a, b]) => {
    const pa = source[String(a)];
    const pb = source[String(b)];
    if (!pa || !pb) return;
    const [ax, ay] = mapToCanvas(pa[0], pa[1]);
    const [bx, by] = mapToCanvas(pb[0], pb[1]);
    ctx.beginPath();
    ctx.moveTo(ax, ay);
    ctx.lineTo(bx, by);
    ctx.stroke();
  });
}

function resolveHandAnchor(source) {
  if (!source || !Object.keys(source).length) return null;
  const wrist = source['0'];
  if (wrist) return wrist;
  const palmBase = ['5', '17']
    .map((key) => source[key])
    .filter((point) => Array.isArray(point) && point.length >= 2);
  if (!palmBase.length) return null;
  const total = palmBase.reduce((acc, [x, y]) => [acc[0] + x, acc[1] + y], [0, 0]);
  return [total[0] / palmBase.length, total[1] / palmBase.length];
}

function drawPoseHandBridge(poseSource, handSource, poseWristId, color, lineWidth) {
  if (!poseSource || !handSource) return;
  const poseWrist = poseSource[String(poseWristId)];
  const handAnchor = resolveHandAnchor(handSource);
  if (!poseWrist || !handAnchor) return;
  const [ax, ay] = mapToCanvas(poseWrist[0], poseWrist[1]);
  const [bx, by] = mapToCanvas(handAnchor[0], handAnchor[1]);
  ctx.save();
  ctx.strokeStyle = color;
  ctx.lineWidth = lineWidth;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.beginPath();
  ctx.moveTo(ax, ay);
  ctx.lineTo(bx, by);
  ctx.stroke();
  ctx.restore();
}

function drawPoints(source, color, radius, excludedKeys = null) {
  if (!source || !Object.keys(source).length) return;
  ctx.fillStyle = color;
  Object.keys(source).forEach((key) => {
    if (excludedKeys && excludedKeys.has(String(key))) return;
    const point = source[key];
    if (!point) return;
    const [x, y] = mapToCanvas(point[0], point[1]);
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawMappedPoints(drawCtx, source, keys, color, radius, mapper) {
  if (!source || !keys || !keys.length) return;
  drawCtx.save();
  drawCtx.fillStyle = color;
  keys.forEach((key) => {
    const pt = source[String(key)];
    if (!pt) return;
    const [x, y] = mapper(pt[0], pt[1]);
    drawCtx.beginPath();
    drawCtx.arc(x, y, radius, 0, Math.PI * 2);
    drawCtx.fill();
  });
  drawCtx.restore();
}

function drawMappedPath(drawCtx, source, keys, color, lineWidth, mapper, closePath = false) {
  if (!source || !keys || keys.length < 2) return;
  drawCtx.save();
  drawCtx.strokeStyle = color;
  drawCtx.lineWidth = lineWidth;
  drawCtx.lineCap = 'round';
  drawCtx.lineJoin = 'round';
  let started = false;
  drawCtx.beginPath();
  keys.forEach((key) => {
    const pt = source[String(key)];
    if (!pt) return;
    const [x, y] = mapper(pt[0], pt[1]);
    if (!started) {
      drawCtx.moveTo(x, y);
      started = true;
    } else {
      drawCtx.lineTo(x, y);
    }
  });
  if (!started) {
    drawCtx.restore();
    return;
  }
  if (closePath) {
    drawCtx.closePath();
  }
  drawCtx.stroke();
  drawCtx.restore();
}

function drawMappedPointsWithHalo(drawCtx, source, keys, haloColor, color, radius, mapper, haloRadius = null) {
  if (!source || !keys || !keys.length) return;
  drawMappedPoints(drawCtx, source, keys, haloColor, haloRadius || (radius + 1.8), mapper);
  drawMappedPoints(drawCtx, source, keys, color, radius, mapper);
}

const mouthKeyIds = Array.from(new Set([
  ...(faceFeatures.outer_lips || []),
  ...(faceFeatures.inner_lips || []),
].map((idx) => String(idx))));

function buildFeatureMapper(source, keys, marginX = 0.22, marginYTop = 0.28, marginYBottom = 0.30) {
  if (!source) return null;
  const points = keys
    .map((idx) => source[String(idx)])
    .filter((pt) => Array.isArray(pt) && pt.length >= 2);
  if (!points.length) return null;
  let minX = Infinity;
  let maxX = -Infinity;
  let minY = Infinity;
  let maxY = -Infinity;
  points.forEach(([x, y]) => {
    minX = Math.min(minX, x);
    maxX = Math.max(maxX, x);
    minY = Math.min(minY, y);
    maxY = Math.max(maxY, y);
  });
  const width = Math.max(0.001, maxX - minX);
  const height = Math.max(0.001, maxY - minY);
  minX -= width * marginX;
  maxX += width * marginX;
  minY -= height * marginYTop;
  maxY += height * marginYBottom;
  const contentW = Math.max(0.001, maxX - minX);
  const contentH = Math.max(0.001, maxY - minY);
  const drawW = Math.max(1, faceCanvas.width - 32);
  const drawH = Math.max(1, faceCanvas.height - 34);
  const scale = Math.min(drawW / contentW, drawH / contentH);
  const offsetX = (faceCanvas.width - contentW * scale) * 0.5;
  const offsetY = (faceCanvas.height - contentH * scale) * 0.5 + 8;
  return (x, y) => [
    offsetX + (x - minX) * scale,
    offsetY + (y - minY) * scale,
  ];
}

function buildFaceMapper(source) {
  if (!source) return null;
  return buildFeatureMapper(source, Object.keys(source), 0.22, 0.28, 0.30);
}

function buildMouthMapper(source) {
  if (!source || !mouthKeyIds.length) return null;
  return buildFeatureMapper(source, mouthKeyIds, 0.32, 0.48, 0.52);
}

function drawFaceMain(face) {
  if (!face || !Object.keys(face).length) return;
  drawMappedPath(ctx, face, faceFeatures.oval, 'rgba(100, 116, 139, 0.34)', 1.15, mapToCanvas, true);
  drawMappedPath(ctx, face, faceFeatures.left_brow, 'rgba(15, 23, 42, 0.55)', 1.2, mapToCanvas, false);
  drawMappedPath(ctx, face, faceFeatures.right_brow, 'rgba(15, 23, 42, 0.55)', 1.2, mapToCanvas, false);
  drawMappedPath(ctx, face, faceFeatures.left_eye, 'rgba(51, 65, 85, 0.40)', 0.95, mapToCanvas, true);
  drawMappedPath(ctx, face, faceFeatures.right_eye, 'rgba(51, 65, 85, 0.40)', 0.95, mapToCanvas, true);
  drawMappedPath(ctx, face, faceFeatures.mouth_line, 'rgba(15, 23, 42, 0.52)', 1.1, mapToCanvas, false);
}

function drawFaceInset(face) {
  clearFaceCanvas();
  if (!face || !Object.keys(face).length || !mouthKeyIds.length) {
    faceCtx.fillStyle = '#64748b';
    faceCtx.font = '600 16px sans-serif';
    faceCtx.textAlign = 'center';
    faceCtx.fillText('No face data', faceCanvas.width / 2, faceCanvas.height / 2 + 8);
    return;
  }
  const mapper = buildFaceMapper(face);
  if (!mapper) {
    faceCtx.fillStyle = '#64748b';
    faceCtx.font = '600 16px sans-serif';
    faceCtx.textAlign = 'center';
    faceCtx.fillText('No face data', faceCanvas.width / 2, faceCanvas.height / 2 + 8);
    return;
  }
  drawMappedPath(faceCtx, face, faceFeatures.oval, 'rgba(100, 116, 139, 0.42)', 2.0, mapper, true);
  drawMappedPath(faceCtx, face, faceFeatures.left_brow, 'rgba(15, 23, 42, 0.60)', 2.1, mapper, false);
  drawMappedPath(faceCtx, face, faceFeatures.right_brow, 'rgba(15, 23, 42, 0.60)', 2.1, mapper, false);
  drawMappedPath(faceCtx, face, faceFeatures.left_eye, 'rgba(51, 65, 85, 0.46)', 1.8, mapper, true);
  drawMappedPath(faceCtx, face, faceFeatures.right_eye, 'rgba(51, 65, 85, 0.46)', 1.8, mapper, true);
  drawMappedPath(faceCtx, face, faceFeatures.mouth_line, 'rgba(15, 23, 42, 0.56)', 2.0, mapper, false);
}

function drawEmptyState() {
  clearCanvas();
  ctx.fillStyle = '#64748b';
  ctx.font = '600 18px sans-serif';
  ctx.textAlign = 'center';
  ctx.fillText('No skeleton loaded', canvas.width / 2, canvas.height / 2);
}

function renderFrame(i) {
  if (!frames.length) {
    drawEmptyState();
    return;
  }
  const safeIndex = Math.max(0, Math.min(i, frames.length - 1));
  const frame = frames[safeIndex];
  if (!frame || !frame.pose || !frame.pose.points) {
    drawEmptyState();
    drawFaceInset(frame && frame.face ? frame.face.points : null);
    return;
  }
  const pts = frame.pose.points;
  if (!pts || Object.keys(pts).length === 0) {
    drawEmptyState();
    drawFaceInset(frame && frame.face ? frame.face.points : null);
    return;
  }
  const face = frame.face ? frame.face.points : null;
  const lhand = frame.left_hand ? frame.left_hand.points : null;
  const rhand = frame.right_hand ? frame.right_hand.points : null;
  clearCanvas();
  drawConnections(connections, pts, '#1f2937', 4);
  drawConnections(torsoConnections, pts, '#ef4444', 5);
  drawPoints(pts, '#2563eb', 4.2, poseFacePointIds);
  drawFaceMain(face);
  drawPoseHandBridge(pts, lhand, 15, '#10b981', 4.2);
  drawPoseHandBridge(pts, rhand, 16, '#3b82f6', 4.2);
  drawConnections(handConnections, lhand, '#10b981', 3.6);
  drawConnections(handConnections, rhand, '#3b82f6', 3.6);
  drawPoints(lhand, '#059669', 3.2);
  drawPoints(rhand, '#2563eb', 3.2);
  drawFaceInset(face);
}

let current = 0;
let lastTick = 0;
function animate(ts) {
  requestAnimationFrame(animate);
  if (!frames.length) {
    drawEmptyState();
    return;
  }
  if (useSync && syncEl) {
    const raw = syncEl.getAttribute('data-frame');
    const idx = raw ? Number(raw) : 0;
    if (!Number.isNaN(idx) && idx !== lastSync) {
      lastSync = idx;
      renderFrame(idx % frames.length);
    }
    return;
  }
  if (!lastTick) lastTick = ts;
  const frameDuration = 1000 / fps;
  const elapsed = ts - lastTick;
  if (elapsed >= frameDuration) {
    const steps = Math.max(1, Math.floor(elapsed / frameDuration));
    lastTick = ts;
    current = (current + steps) % frames.length;
    renderFrame(current);
  }
}

window.addEventListener('resize', () => {
  resizeCanvas();
  renderFrame(lastSync >= 0 ? lastSync : current);
});

resizeCanvas();
renderFrame(0);
animate(0);
    </script>
  </body>
</html>
"""
    inner = template.replace("__PAYLOAD__", data_json).replace("__FACE_FEATURES__", face_features_json)
    srcdoc = _html_escape(inner, quote=True)
    return (
        "<iframe "
        f"srcdoc=\"{srcdoc}\" "
        "style='width:100%;height:58vh;min-height:420px;max-height:620px;border:0;'></iframe>"
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
    app_css = """
    html, body {
      margin: 0;
      background: linear-gradient(180deg, #f7fafc 0%, #eef4f8 100%);
      font-family: "Segoe UI", "Noto Sans Thai", sans-serif;
      color: #0f172a;
    }
    .sl-shell {
      padding: 12px;
      min-height: 100vh;
      box-sizing: border-box;
    }
    .sl-header-card,
    .sl-card {
      background: rgba(255, 255, 255, 0.96);
      border: 1px solid rgba(148, 163, 184, 0.18);
      border-radius: 24px;
      box-shadow: 0 18px 44px rgba(15, 23, 42, 0.08);
    }
    .sl-header-card {
      padding: 16px 18px;
      margin-bottom: 10px;
      background: linear-gradient(135deg, #ffffff 0%, #f5fbff 46%, #effaf5 100%);
    }
    .sl-kicker {
      display: inline-flex;
      align-items: center;
      padding: 7px 12px;
      border-radius: 999px;
      background: #ecfeff;
      color: #0f766e;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
    }
    .sl-header-card h1 {
      margin: 10px 0 6px;
      font-size: 28px;
      line-height: 1.08;
      letter-spacing: -0.03em;
      color: #0f172a;
    }
    .sl-header-card p {
      margin: 0;
      max-width: 760px;
      color: #64748b;
      font-size: 14px;
      line-height: 1.55;
    }
    .sl-chip-row {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 10px;
    }
    .sl-chip {
      display: inline-flex;
      align-items: center;
      padding: 8px 12px;
      border-radius: 999px;
      background: #f8fafc;
      color: #475569;
      font-size: 12px;
      font-weight: 600;
    }
    .sl-content {
      gap: 14px;
      align-items: flex-start;
    }
    .sl-side-column {
      gap: 12px;
      min-width: 250px;
      max-width: 250px;
    }
    .sl-viewer-main {
      min-width: 0;
      flex: 1 1 auto;
    }
    .sl-playbar {
      align-items: center;
      gap: 12px;
    }
    .sl-card {
      padding: 14px;
    }
    .sl-card-title {
      margin: 0;
      font-size: 18px;
      color: #0f172a;
    }
    .sl-card-copy {
      margin: 6px 0 0;
      color: #64748b;
      font-size: 14px;
      line-height: 1.6;
    }
    .sl-status-box {
      padding: 14px 16px;
      border-radius: 16px;
      border: 1px solid rgba(148, 163, 184, 0.16);
      font-size: 14px;
      line-height: 1.6;
      color: #334155;
      background: #f8fafc;
    }
    .sl-status-box.info { background: #f8fafc; color: #475569; }
    .sl-status-box.listening { background: #ecfeff; color: #0f766e; }
    .sl-status-box.working { background: #fff7ed; color: #c2410c; }
    .sl-status-box.success { background: #ecfdf5; color: #047857; }
    .sl-status-box.error { background: #fef2f2; color: #b91c1c; }
    .sl-meta {
      color: #94a3b8;
      font-size: 11px;
      font-weight: 700;
      letter-spacing: 0.14em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }
    .sl-input input {
      border-radius: 16px !important;
      border: 1px solid rgba(148, 163, 184, 0.20) !important;
      background: #fbfdff !important;
      font-size: 15px !important;
    }
    .sl-voice-btn .bk-btn,
    .sl-play-btn .bk-btn,
    .sl-quit-btn .bk-btn {
      border-radius: 16px !important;
      font-weight: 700;
      box-shadow: none !important;
    }
    .sl-voice-btn .bk-btn {
      width: 100% !important;
      min-height: 48px !important;
    }
    .sl-viewer-card {
      padding: 0;
      overflow: hidden;
    }
    .sl-viewer-head {
      padding: 12px 14px 0;
    }
    .sl-viewer-frame iframe {
      width: 100%;
      display: block;
      border: 0;
      background: #ffffff;
    }
    .sl-playback {
      padding: 0 14px 12px;
    }
    @media (max-width: 900px) {
      .sl-shell {
        padding: 12px;
      }
      .sl-header-card {
        padding: 20px;
      }
      .sl-header-card h1 {
        font-size: 28px;
      }
    }
    """
    pn.extension(sizing_mode='stretch_both', raw_css=[app_css])

    hv.opts.defaults(
        hv.opts.Overlay(responsive=True, min_height=400, show_legend=False),
        hv.opts.Path(responsive=True),
        hv.opts.Points(responsive=True),
    )

    def _escape(value: str) -> str:
        return _html_escape(value or '', quote=False)

    def _status_html(message: str, tone: str = 'info') -> str:
        return f"<div class='sl-status-box {tone}'>{_escape(message)}</div>"

    text = pn.widgets.TextInput(
        name='ข้อความหรือคำศัพท์',
        placeholder='พิมพ์คำหรือวลี เช่น hello thank you',
        sizing_mode='stretch_width',
        css_classes=['sl-input'],
    )
    status = pn.pane.HTML(_status_html('พร้อมรับข้อความจากการพิมพ์หรือไมโครโฟน', 'info'), sizing_mode='stretch_width')
    voice_btn = pn.widgets.Button(name='กดเพื่อพูด', button_type='primary', sizing_mode='stretch_width', css_classes=['sl-voice-btn'])

    def set_status(message: str, tone: str = 'info'):
        status.object = _status_html(message, tone)

    def on_voice_process():
        if SPEECH_STATE['is_listening']:
            voice_btn.name = f"{SPEECH_STATE['status']} (กดเพื่อหยุด)"
            voice_btn.button_type = 'warning'
            set_status('กำลังฟังเสียงจากไมโครโฟน โปรดพูดชัดเจน', 'listening')
        else:
            voice_btn.name = 'กดเพื่อพูด'
            voice_btn.button_type = 'primary'
            if SPEECH_STATE['text']:
                text.value = SPEECH_STATE['text']
                SPEECH_STATE['text'] = ''

    def _init_periodic():
        try:
            pn.state.add_periodic_callback(on_voice_process, period=500)
        except Exception as e:
            print(f"Warning: could not add periodic callback: {e}")

    def _on_voice_click(event):
        if not SPEECH_STATE['is_listening']:
            SPEECH_STATE['is_listening'] = True
            SPEECH_STATE['text'] = ''
            SPEECH_STATE['stop_event'].clear()
            SPEECH_STATE['status'] = 'Starting...'
            set_status('กำลังเริ่มไมโครโฟน...', 'working')
            t = threading.Thread(target=threaded_listen)
            t.daemon = True
            t.start()
        else:
            SPEECH_STATE['stop_event'].set()
            SPEECH_STATE['is_listening'] = False
            voice_btn.name = 'Stopping...'
            set_status('กำลังหยุดการฟังเสียง...', 'working')

    voice_btn.on_click(_on_voice_click)

    initial_fps = max(1, int(DEFAULT_TARGET_FPS))
    MIN_INTERVAL_MS = 20
    initial_step = int(math.ceil((initial_fps / 1000.0) * MIN_INTERVAL_MS))
    initial_interval = MIN_INTERVAL_MS

    frame_slider = pn.widgets.IntSlider(name='Frame', start=0, end=max(len(frames) - 1, 1), value=0, visible=True, sizing_mode='stretch_width')
    play_btn = pn.widgets.Button(name='Play', button_type='primary', width=120, align='center', css_classes=['sl-play-btn'])
    playback_info = pn.pane.HTML('', sizing_mode='stretch_width')

    playback = {
        'interval': initial_interval,
        'step': initial_step,
        'running': False,
    }

    def sync_playback_info(*_):
        total_frames = len(frames)
        current_frame = 0 if total_frames == 0 else min(frame_slider.value, total_frames - 1) + 1
        playback_info.object = (
            "<div class='sl-chip-row'>"
            f"<span class='sl-chip'>{'Playing' if playback['running'] else 'Ready'}</span>"
            f"<span class='sl-chip'>Frame {current_frame}/{max(total_frames, 1)}</span>"
            f"<span class='sl-chip'>{DEFAULT_TARGET_FPS} FPS</span>"
            "</div>"
        )

    def advance_frame():
        if not playback['running']:
            return
        current = frame_slider.value
        end = frame_slider.end
        if current >= end:
            stop_playing()
            return
        step = playback['step']
        new_val = current + step
        if new_val > end:
            new_val = end
        frame_slider.value = new_val

    anim_cb = pn.state.add_periodic_callback(advance_frame, period=initial_interval, start=False)

    def start_playing():
        if frame_slider.value >= frame_slider.end:
            frame_slider.value = 0
        playback['running'] = True
        play_btn.name = 'Pause'
        play_btn.button_type = 'warning'
        anim_cb.period = playback['interval']
        if not anim_cb.running:
            anim_cb.start()
        sync_playback_info()

    def stop_playing():
        playback['running'] = False
        play_btn.name = 'Play'
        play_btn.button_type = 'primary'
        if anim_cb.running:
            anim_cb.stop()
        sync_playback_info()

    def toggle_play(event):
        if playback['running']:
            stop_playing()
        else:
            start_playing()

    play_btn.on_click(toggle_play)
    frame_slider.param.watch(sync_playback_info, 'value')

    three_pane = pn.pane.HTML('', sizing_mode='stretch_width', min_height=400, css_classes=['sl-viewer-frame'])
    frame_sync_pane = pn.pane.HTML(
        pn.bind(lambda i: f"<div id='frame-sync' data-frame='{int(i)}' style='display:none'></div>", frame_slider),
        sizing_mode='fixed',
        height=0,
        width=0,
    )
    main_panel = pn.Column(three_pane, frame_sync_pane, sizing_mode='stretch_width')

    state = {'raw': CURRENT_RAW_DF.copy()}

    def _apply_playback_rate(target_fps):
        fps = max(1, int(target_fps))
        min_interval = 20
        step = int(math.ceil((fps / 1000.0) * min_interval))
        interval_ms = min_interval
        playback['interval'] = interval_ms
        playback['step'] = step
        if playback['running']:
            anim_cb.period = interval_ms

    def apply_sequence():
        global df, frames
        raw_df = state['raw']
        if raw_df.empty:
            df = pd.DataFrame(columns=EXPECTED_COLS)
            frames = []
            frame_slider.end = 1
            frame_slider.value = 0
            stop_playing()
            three_pane.object = render_three_viewer_html(build_three_payload(df, DEFAULT_TARGET_FPS))
            sync_playback_info()
            return
        prepared = prepare_sequence(raw_df, UPSAMPLE_FACTOR, SMOOTH_WINDOW)
        df = prepared
        frames = sorted(df['frame'].unique().tolist())
        frame_slider.end = max(len(frames) - 1, 0)
        frame_slider.value = 0
        _apply_playback_rate(DEFAULT_TARGET_FPS)
        three_pane.object = render_three_viewer_html(build_three_payload(df, DEFAULT_TARGET_FPS))
        start_playing()
        sync_playback_info()

    def set_sequence(raw_df: pd.DataFrame):
        global CURRENT_RAW_DF
        CURRENT_RAW_DF = raw_df.copy()
        state['raw'] = raw_df.copy()
        apply_sequence()

    _apply_playback_rate(DEFAULT_TARGET_FPS)

    def _on_input_change(event):
        raw = (event.new or '').strip().lower()
        if not raw:
            set_status('พร้อมรับข้อความจากการพิมพ์หรือไมโครโฟน', 'info')
            return

        words = re.split(r"\s+", raw)
        filtered_words = [w for w in words if w and w not in ['is', 'am', 'are']]
        search_term = ' '.join(filtered_words)

        keys = []
        missing_terms = []
        exact_matches = [k for (k, norm) in SEARCH_INDEX if norm == search_term]
        if exact_matches:
            keys.append(exact_matches[0])
        else:
            for w in filtered_words:
                word_matches = [k for (k, norm) in SEARCH_INDEX if norm == w]
                if word_matches:
                    keys.append(word_matches[0])
                else:
                    missing_terms.append(w)

        if not keys:
            missing_text = ' '.join(missing_terms) if missing_terms else '(ว่าง)'
            set_status(f'ไม่พบไฟล์ที่ตรงกับคำ: {missing_text} | ไฟล์ทั้งหมด: {len(CLIPS)}', 'error')
            return

        set_status('กำลังสร้างแอนิเมชัน...', 'working')
        new_df = build_sequence_from_tokens(keys)
        set_sequence(new_df)

        if missing_terms:
            set_status(f"เล่นแอนิเมชันแล้ว แต่ยังไม่พบคำ: {' '.join(missing_terms)}", 'working')
        else:
            set_status('เล่นแอนิเมชันแล้ว', 'success')

    quit_btn = pn.widgets.Button(name='Quit', button_type='danger', sizing_mode='stretch_width', css_classes=['sl-quit-btn'])
    quit_btn.on_click(lambda event: os._exit(0))

    text.param.watch(_on_input_change, 'value')

    apply_sequence()
    sync_playback_info()
    pn.state.onload(_init_periodic)

    header = pn.pane.HTML(
        f"""
        <div class='sl-header-card'>
          <div class='sl-kicker'>Speech to Sign</div>
          <p>พิมพ์หรือพูดข้อความทางซ้าย ระบบจะจับคู่คำและเล่น animation ทางขวาทันที พร้อมแสดงสถานะโดยไม่ต้องสลับมุมมอง</p>
          <div class='sl-chip-row'>
            <span class='sl-chip'>{len(CLIPS)} clips</span>
            <span class='sl-chip'>{DEFAULT_TARGET_FPS} FPS target</span>
            <span class='sl-chip'>Auto translate while typing</span>
          </div>
        </div>
        """,
        sizing_mode='stretch_width',
    )

    control_card = pn.Column(
        pn.pane.HTML(
            """
            <div>
              <div class='sl-meta'>Input</div>
              <h2 class='sl-card-title'>ป้อนข้อความหรือใช้ไมโครโฟน</h2>
              <p class='sl-card-copy'>พิมพ์คำหรือวลีสั้น ๆ ระบบจะเริ่มจับคู่คำให้อัตโนมัติ และสามารถกดปุ่มพูดเพื่อรับข้อความจากไมโครโฟนได้</p>
            </div>
            """,
            sizing_mode='stretch_width',
        ),
        text,
        voice_btn,
        status,
        quit_btn,
        sizing_mode='stretch_width',
        css_classes=['sl-card'],
        width=250,
    )

    left_panel = pn.Column(
        control_card,
        sizing_mode='fixed',
        width=250,
        css_classes=['sl-side-column'],
    )

    viewer_card = pn.Column(
        pn.pane.HTML(
            """
            <div class='sl-viewer-head'>
              <div class='sl-meta'>Preview</div>
              <h2 class='sl-card-title'>พื้นที่แสดงแอนิเมชันภาษามือ</h2>
              <p class='sl-card-copy'>ดูผลการแปลและควบคุมการเล่นจากแผงนี้ได้ทันที</p>
            </div>
            """,
            sizing_mode='stretch_width',
        ),
        main_panel,
        pn.Column(
            playback_info,
            frame_slider,
            sizing_mode='stretch_width',
            css_classes=['sl-playback'],
        ),
        sizing_mode='stretch_width',
        min_width=0,
        css_classes=['sl-card', 'sl-viewer-card', 'sl-viewer-main'],
    )

    content = pn.Row(
        left_panel,
        viewer_card,
        sizing_mode='stretch_width',
        css_classes=['sl-content'],
    )

    return pn.Column(content, sizing_mode='stretch_both', css_classes=['sl-shell'])
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
        port_env = os.environ.get("PANEL_PORT", "0")
        try:
            port = int(port_env)
        except Exception:
            port = 0
        show_env = os.environ.get("PANEL_SHOW", "1").strip().lower()
        show = show_env not in ("0", "false", "no")
        websocket_origin = build_websocket_origins(port)
        pn.serve(
            build_panel_app,
            show=show,
            port=port,
            start=True,
            threaded=False,
            static_dirs=PANEL_STATIC_DIRS,
            websocket_origin=websocket_origin,
        )
    except Exception as exc:
        import traceback
        traceback.print_exc()
        print(f'Panel server failed to start: {exc}')


if __name__ == '__main__':
    main()











