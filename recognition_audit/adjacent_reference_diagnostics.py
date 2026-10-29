import adjacent_reference as exp
import numpy as np

r=exp.read(exp.OUT/'results.json')
caps={(c['phrase'],c['variant']):c['samples'] for c in exp.read(exp.OUT/'renderer/rendered.json')}
cases=exp.read(exp.OUT/'case_index.json')
bylabel={d['label']:d['phrase'] for d in cases}
switch=[x for x in r['rows'] if x['kind']=='switch']
prior_failures=[]
for x in switch:
    if x['OLD']['size']<=x['OFF']['size']+1e-9:continue
    ss=caps[bylabel[x['label']],'NEW']; i=x['index']; side=x['side']
    prior_failures.append(dict(label=x['label'],side=side,index=i,OFF=x['OFF'],OLD=x['OLD'],NEW=x['NEW'],ops=[ss[j]['ops'].get(side) for j in [i-1,i]]))
review=exp.read(exp.OUT/'visual_review/rendered.json'); descriptors=exp.read(exp.OUT/'review_index.json'); exact=0
for c in review:
    d=next(d for d in descriptors if exp.read(__import__('pathlib').Path(d['file']))['phrase']==c['phrase'])
    full=caps[d['phrase'],c['variant']]
    for sample in c['samples']:
        assert sample==full[sample['index']]; exact+=1
words=[]
for label in bylabel:
    rr=[x for x in switch if x['label']==label]
    if not rr:continue
    words.append(dict(label=label,n=len(rr),**{m:float(np.mean([x['NEW'][m]-x['OLD'][m] for x in rr])) for m in ['size','center']}))
result=dict(previous_eight_failures=prior_failures,review_samples_exact=exact,word_mean_differences_new_minus_old=words,word_counts={m:dict(better=sum(w[m]<-1e-9 for w in words),worse=sum(w[m]>1e-9 for w in words),unchanged=sum(abs(w[m])<=1e-9 for w in words)) for m in ['size','center']})
exp.write(exp.OUT/'diagnostics.json',result)
print(__import__('json').dumps(result,indent=2))
