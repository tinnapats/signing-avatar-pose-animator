"""Reuse identical metrics/media generation with explicit v2 comparison pairs."""
import fallback_trial_evaluate as audit
import json
audit.OUT=audit.HERE/'fallback_v2'
audit.VARIANTS=['scale_only','radial','chain','chain_slew']
audit.PAIRS=[(v,'scale_only') for v in audit.VARIANTS[1:]]
audit.PRIOR_MAPPING={'S50A1':'scale_only'}
audit.CAP_VARIANTS=['radial','chain']
def verify_reference():
    rows=json.loads((audit.OUT/'rendered.json').read_text())
    lookup={(r['variant'],r['phrase']):r['samples'] for r in rows}
    old=json.loads((audit.HERE/'fallback_trial/rendered.json').read_text())
    for r in old:
        if r['variant']!='combined':continue
        new=lookup['radial',r['phrase']]
        for a,b,c in zip(r['samples'],new,lookup['chain',r['phrase']]):
            for side in ['left','right']:
                previous,radial,chain=a['fallback'][side],b['fallback'][side],c['fallback'][side]
                assert abs(previous['factor']-radial['factor'])<1e-9
                for field in ['before','after','limit']:
                    if previous[field] is None:assert radial[field] is None
                    else:assert abs(previous[field]-radial[field])<1e-6
                if radial['limit'] is not None:assert chain['limit']+1e-9>=radial['limit']
                assert chain['factor']+1e-9>=radial['factor']
    print('PASS: radial reproduces v1; chain never tightens its bound or shrinks more than radial.')

if __name__=='__main__':
    verify_reference()
    audit.main()
