"""Rebase local copied environments and maintained scripts; archive original bytes.

Historical result JSON is NOT rewritten. Python experiment readers relocate
historical absolute paths in memory using portable_paths.loads.
"""
import os
from pathlib import Path
from portable_paths import ROOT, relocate_path


def replace_file(path, data):
    old = path.read_bytes()
    if old == data:
        return False
    backup = ROOT/'local_archive/before-path-rebase'/path.relative_to(ROOT)
    backup.parent.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_bytes(old)
    temp = path.with_name(path.name+'.rebase-tmp')
    temp.write_bytes(data)
    os.replace(temp, path)  # detach any existing hard link, preserving originals
    return True


def main():
    # Match both Python/JS normal strings and serialized Windows path literals.
    roots = [
        (str(ROOT.parent/'signing-avatar-github-tinnapat'), str(ROOT)),
        (str(ROOT.parent/'signing-avatar-pose-animator'), str(ROOT)),
        (str(Path.home()/'.codex/visualizations/2026/07/27/019fa278-2887-7562-a831-6544ada9c461/model-results'), str(ROOT/'evaluation/model-results')),
        (str(Path.home()/'.codex/visualizations/2026/07/27/019fa278-2887-7562-a831-6544ada9c461/s310'), str(ROOT/'local_runtime/sdpose_env')),
        (str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules'), str(ROOT/'frontend-tooling/node_modules')),
        (str(Path.home()/'.cache/codex-runtimes/codex-primary-runtime/dependencies/node/bin'), str(ROOT/'local_runtime/node')),
        (str(Path.home()/'AppData/Local/Programs/Python/Python312'), str(ROOT/'local_runtime/python312')),
        ('C:\\Program Files\\Google\\Chrome\\Application', str(ROOT/'local_runtime/chrome')),
    ]
    candidates = list((ROOT/'recognition_audit').glob('*.py')) + list((ROOT/'recognition_audit').glob('*.mjs'))
    base = ROOT/'evaluation/model-results'
    candidates += list(base.glob('*.py')) + list(base.glob('*.mjs'))
    for folder in ('sdpose-trial-20261009', 'hand-reference60-20261014'):
        candidates += list((base/folder).glob('*.py')) + list((base/folder).glob('*.mjs'))
    envs = [ROOT/'.venv', ROOT/'local_runtime/sdpose_env', ROOT/'recognition_audit/rtmpose_env', ROOT/'recognition_audit/paired_video_env',
            base/'sdpose-trial-20261009/env310', base/'sdpose-trial-20261009/env310-official']
    for env in envs:
        candidates += [env/'pyvenv.cfg']
        candidates += [p for p in (env/'Scripts').glob('*') if p.suffix in ('.bat','.ps1','.py') or p.name=='activate']
        candidates += list((env/'Lib/site-packages').glob('*.pth'))
    count = 0
    for file in candidates:
        if not file.is_file():
            continue
        text = file.read_text(encoding='utf-8-sig')
        for old, new in sorted(roots, key=lambda pair: -len(pair[0])):
            for a, b in ((old.replace('\\','/'), new.replace('\\','/')),
                         (old.replace('\\','\\\\'), new.replace('\\','\\\\')), (old,new)):
                text = text.replace(a,b)
        if file.suffix == '.py':
            # Only rewrite real calls, not JSON snippets or source strings used by subprocesses.
            text = text.replace('_local_json_loads(', 'json.loads(')
            import ast
            tree = ast.parse(text)
            raw = text.encode('utf-8')
            offsets = [0]
            for line_bytes in raw.splitlines(keepends=True):
                offsets.append(offsets[-1]+len(line_bytes))
            edits = []
            for node in ast.walk(tree):
                if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute) and isinstance(node.func.value,ast.Name) and node.func.value.id=='json' and node.func.attr=='loads':
                    edits.append((offsets[node.func.lineno-1]+node.func.col_offset,
                                  offsets[node.func.end_lineno-1]+node.func.end_col_offset))
            for start,end in sorted(edits,reverse=True):
                raw = raw[:start]+b'_local_json_loads'+raw[end:]
            text = raw.decode('utf-8')
        if file.suffix == '.py' and '_local_json_loads(' in text and 'from portable_paths import loads as _local_json_loads' not in text:
            # Import after future statements/docstring without changing their semantics.
            import ast
            tree = ast.parse(text)
            line = 0
            for node in tree.body:
                if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value,str)) or (isinstance(node,ast.ImportFrom) and node.module=='__future__'):
                    line = node.end_lineno
                else:
                    break
            prefix = ('\n# Relocate historical paths in memory; keep original JSON evidence unchanged.\n'
                      'import sys as _path_sys\nfrom pathlib import Path as _Path\n'
                      '_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())\n'
                      '_path_sys.path.insert(0, str(_project))\n'
                      'from portable_paths import loads as _local_json_loads\n')
            lines = text.splitlines(keepends=True)
            text = ''.join(lines[:line])+prefix+''.join(lines[line:])
        count += replace_file(file, text.encode('utf-8'))
    print(f'Rebased {count} files; historical JSON untouched; original bytes in local_archive/before-path-rebase')


if __name__=='__main__':
    main()
