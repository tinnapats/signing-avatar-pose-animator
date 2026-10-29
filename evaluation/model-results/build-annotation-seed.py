
# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
from pathlib import Path
p=Path(__file__).parent/'body-reference-v1-20261009';data=_local_json_loads((p/'annotations.json').read_text(encoding='utf8'));assert data['status']=='DRAFT_NOT_VERIFIED';(p/'seed.js').write_text('window.REFERENCE_SEED = '+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+';\n',encoding='utf8')
print('Packaged',len(data['items']),'images; original annotations unchanged.')
