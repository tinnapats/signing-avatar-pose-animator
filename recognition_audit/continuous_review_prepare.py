
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import copy,json
from pathlib import Path
HERE=Path(__file__).resolve().parent;OUT=HERE/'continuous_review_20261005'
assert not OUT.exists();OUT.mkdir();(OUT/'cases').mkdir()
index=[];manifest=[]
for label in ['parents','chicken','class']:
    descriptors=_local_json_loads((HERE/'bounded_canvas_20261005/case_index.json').read_text(encoding='utf-8'))
    for mode in ['BASE','SIZE_ONLY','BOUNDED_SIZE']:
        source=next(x for x in descriptors if x['label']==label and x['phrase'].endswith('__'+('SIZE_ONLY' if mode=='BASE' else mode)))
        c=_local_json_loads(Path(source['file']).read_text(encoding='utf-8'));c.update(method=mode,phrase=label+'__'+mode)
        path=OUT/'cases'/f'{len(index):02d}.json';path.write_text(json.dumps(c),encoding='utf-8');index.append(dict(file=str(path),label=label,phrase=c['phrase']))
    manifest.append(dict(label=label,frames=len(c['frameIndices'])))
(OUT/'case_index.json').write_text(json.dumps(index),encoding='utf-8')
(OUT/'manifest.json').write_text(json.dumps(dict(cases=manifest,fps=30,selection='Three existing diagnostics: hand near face, two-hand relation, known recognition regression. Full clips sampled at30FPS, not independent validation.')),encoding='utf-8')
print(manifest)
