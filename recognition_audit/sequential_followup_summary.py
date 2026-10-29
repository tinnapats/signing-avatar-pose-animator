"""Summarize predetermined screening and the frozen new-word pilot."""

# Relocate historical paths in memory; keep original JSON evidence unchanged.
import sys as _path_sys
from pathlib import Path as _Path
_project = next(p for p in _Path(__file__).resolve().parents if (p / "portable_paths.py").is_file())
_path_sys.path.insert(0, str(_project))
from portable_paths import loads as _local_json_loads
import json
import math
from pathlib import Path
HERE=Path(__file__).resolve().parent
def read(p):return _local_json_loads((HERE/p).read_text())
def paired(rows,a,b):
    look={(r['label'],r['variant']):r for r in rows};labels=list(dict.fromkeys(r['label'] for r in rows))
    gains=[l for l in labels if look[l,a]['prediction']!=l and look[l,b]['prediction']==l]
    losses=[l for l in labels if look[l,a]['prediction']==l and look[l,b]['prediction']!=l]
    n=len(gains)+len(losses);p=min(1,2*sum(math.comb(n,k) for k in range(min(len(gains),len(losses))+1))/2**n) if n else 1
    return dict(gained=gains,lost=losses,p=p)
mild=read('mild_smoothing200/results.json');render=read('mild_smoothing200/renderer_review.json');g=read('mild_smoothing200/analysis.json')['geometry']
protocol=read('mild_smoothing200/protocol.json');screen=protocol['screening'];m=render['summaries']['no_hold_mild_smoothing__S1A1'];ref=render['summaries']['no_hold_no_smoothing__S1A1']
reduction=1-m['mean_word_common_second_difference']/ref['mean_word_common_second_difference']
checks=dict(accuracy=mild['summaries']['no_hold_mild_smoothing']['correct']>=screen['min_correct'],
    roughness=reduction>=screen['min_common_roughness_reduction'],switches=sum(m['transitions'].values())<=screen['max_switches'],
    native_frames=m['states']['native']<=screen['max_native_hand_frames'],bone_span=g['no_hold_mild_smoothing']['bone_span']<=screen['max_mean_bone_span'])
pre=read('novel_words30_20261001/pre_results.json');post=read('novel_words30_20261001/renderer_results.json')
native={}
for v in ['baseline','no_hold_no_smoothing']:
    rows=[r for r in post['predictions'] if r['variant']==v]
    native[v]={k:sum(r[k] for r in rows) for k in ['native','detailed','switches']}
summary=dict(mild_screen=dict(passed=all(checks.values()),checks=checks,roughness_reduction=reduction),
    novel_pre=pre['summaries'],novel_post=post['summaries'],novel_native=native,
    paired_pre=paired(pre['predictions'],'baseline','no_hold_no_smoothing'),paired_post=paired(post['predictions'],'baseline','no_hold_no_smoothing'))
(HERE/'novel_words30_20261001/followup_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
print(json.dumps(summary,indent=2))
