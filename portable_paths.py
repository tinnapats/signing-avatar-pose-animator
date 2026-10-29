"""Resolve historical absolute paths without editing archived JSON evidence."""
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def relocate_path(value):
    if not isinstance(value, str):
        return value
    normalized = value.replace('\\', '/')
    if not re.match(r'^[A-Za-z]:/', normalized):
        return value
    mappings = [
        ('/019fa278-2887-7562-a831-6544ada9c461/s310/', ROOT/'local_runtime/sdpose_env'),
        ('/model-results/', ROOT/'evaluation/model-results'),
        ('/dependencies/node/node_modules/', ROOT/'frontend-tooling/node_modules'),
        ('/dependencies/node/bin/', ROOT/'local_runtime/node'),
        ('/Programs/Python/Python312/', ROOT/'local_runtime/python312'),
        ('/Google/Chrome/Application/', ROOT/'local_runtime/chrome'),
    ]
    for marker, destination in mappings:
        if marker in normalized:
            return (destination/normalized.split(marker, 1)[1]).as_posix()
    for name in ('signing-avatar-github-tinnapat', 'signing-avatar-pose-animator',
                 'signing-avatar-github-newton-upload'):
        marker = '/'+name
        if normalized.endswith(marker):
            return ROOT.as_posix()
        if marker+'/' in normalized:
            return (ROOT/normalized.split(marker+'/', 1)[1]).as_posix()
    return value


def localize(value):
    if isinstance(value, dict):
        return {relocate_path(k): localize(v) for k, v in value.items()}
    if isinstance(value, list):
        return [localize(v) for v in value]
    return relocate_path(value)


def loads(*args, **kwargs):
    return localize(json.loads(*args, **kwargs))
