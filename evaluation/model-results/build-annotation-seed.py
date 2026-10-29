import json
from pathlib import Path
p=Path(__file__).parent/'body-reference-v1-20261009';data=json.loads((p/'annotations.json').read_text(encoding='utf8'));assert data['status']=='DRAFT_NOT_VERIFIED';(p/'seed.js').write_text('window.REFERENCE_SEED = '+json.dumps(data,ensure_ascii=False).replace('<','\\u003c')+';\n',encoding='utf8')
print('Packaged',len(data['items']),'images; original annotations unchanged.')
