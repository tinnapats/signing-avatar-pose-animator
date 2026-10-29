import sys,importlib.util
from pathlib import Path
sys.dont_write_bytecode=True
spec=importlib.util.spec_from_file_location('ev',Path(__file__).with_name('rtm-eval.py'));e=importlib.util.module_from_spec(spec);spec.loader.exec_module(e)
old_data=e.data
def data(c):
 a=old_data(c);mp=a['mediapipe'];rt=a['rtmw_det'];h=mp.copy()
 for off in [33,54]:
  missing=~e.np.any(mp[:,off:off+21]!=0,axis=(1,2));h[missing,off:off+21]=rt[missing,off:off+21]
 h2=h.copy();h2[:,:33]=rt[:,:33]
 return dict(mediapipe=mp,rtmw_det=rt,mp_body_mp_priority=h,rtmw_body_mp_priority=h2)
e.save(e.O/'hybrid_protocol.json',dict(rules='Exploratory follow-up AFTER observing phase1 geometry: retain full MediaPipe hand whenever present; otherwise RTMW-det hand. No per-word exceptions, no source/renderer coordinates, no temporal fill. Two body choices. No tuning on scores. Not independent confirmation.',source_results_sha256=e.sha(e.O/'results.json')))
e.data=data;e.VARIANTS=['mediapipe','rtmw_det','mp_body_mp_priority','rtmw_body_mp_priority'];e.RESULT_NAME='hybrid_results.json';e.evaluate()
