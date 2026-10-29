"""Resume as a separately frozen end-to-end test, retaining failed strict-body trial."""
from pathlib import Path
source_path=Path(__file__).with_name('short_gap200.py')
source=source_path.read_text(encoding='utf-8')
source=source.replace("OUT=HERE/'short_gap200_20261005'","OUT=HERE/'short_gap200_e2e_20261005'")
source=source.replace("HERE/'wlasl_stgcn.zip',Path(__file__)","HERE/'wlasl_stgcn.zip',source_path,Path(__file__)")
source=source.replace("Same released", "Same released")
source=source.replace("No changes to eligibility, confidence, long gaps, Gaussian/body, articulation, stabilization, FPS, scale/anchor.","No changes to eligibility, confidence, long gaps, Gaussian/body algorithms, articulation, stabilization, FPS or renderer scale/anchor settings. Global fit-to-canvas can change as a downstream effect; verify a single shared similarity transform explains all pose/face differences. Preserve earlier strict-body failed trial; do not treat it as scored result.")
needle="        assert len(payload['frames'])==len(linear['frames']);changed=0"
assert source.count(needle)==1
source=source.replace(needle,needle+'''
        pa=np.array([[k['position'][axis] for axis in ['x','y']] for f in linear['frames'] for k in f['pose']['keypoints']])
        pb=np.array([[k['position'][axis] for axis in ['x','y']] for f in payload['frames'] for k in f['pose']['keypoints']])
        ca=pa.mean(axis=0);cb=pb.mean(axis=0)
        fit_scale=float(np.sum((pa-ca)*(pb-cb))/np.sum((pa-ca)**2));shift=cb-fit_scale*ca
        residual=float(np.max(np.abs(pb-(fit_scale*pa+shift))))
        assert residual<1e-8 and fit_scale>0, 'Non-affine body change '+label
''')
source=source.replace("            assert a['pose']==b['pose'] and a.get('face')==b.get('face')", """            for ka,kb in zip(a['pose']['keypoints'],b['pose']['keypoints']):
                assert ka['part']==kb['part'] and ka['score']==kb['score']
            fa=np.array(a['face']['positions']).reshape(-1,2);fb=np.array(b['face']['positions']).reshape(-1,2)
            assert np.allclose(fb,fit_scale*fa+shift,atol=1e-8,rtol=0)
            assert {k:v for k,v in a['face'].items() if k!='positions'}=={k:v for k,v in b['face'].items() if k!='positions'}""")
source=source.replace("linear_exact=True,changed_frames=changed", "linear_exact=True,changed_frames=changed,fit_scale=fit_scale,fit_shift=shift.tolist(),fit_residual=residual")
exec(compile(source,str(source_path)+' [e2e framing audit]', 'exec'))
