// Experimental, geometry-only guard. Never synthesizes a recognized sign.
export function capFactor(radius, limit) {
  if (!Number.isFinite(radius) || !Number.isFinite(limit) || radius <= 0 || limit <= 0) return 1;
  return Math.min(1, limit / radius);
}

// Serialized into the isolated browser build; deliberately self-contained.
export function installFallbackAudit() {
  const history={left:new Map(),right:new Map()};
  window.auditFallback = (illustration, frame, scales, options, frameIndex) => {
    const bodySnapshot = () => JSON.stringify(illustration.scope.project.activeLayer.children
      .filter(p => !p.data.auditNativeSide).map(p => ({opacity:p.opacity,
        segments:p.segments?.map(s => [s.point.x,s.point.y,s.handleIn.x,s.handleIn.y,s.handleOut.x,s.handleOut.y])})));
    const beforeBody=bodySnapshot();
    window.auditBodyReference ||= new Map();
    // Normalization drops the original frame index; identify the full input.
    const key=JSON.stringify([window.auditPayload.meta.sourceClips,frame,options.scale]);
    if (window.auditBodyReference.has(key) && window.auditBodyReference.get(key)!==beforeBody)
      throw Error('Non-hand geometry changed across repeated variants');
    window.auditBodyReference.set(key,beforeBody);
    const records = {};
    for (const side of ['left', 'right']) {
      const hand = frame.hands?.[side];
      const body = frame.pose.keypoints.find(k => k.part === side + 'Wrist');
      const paths = illustration.scope.project.activeLayer.children.filter(p =>
        p.data.auditNativeSide === side && p.visible && p.opacity > 0);
      const radii = [];
      for (const f of window.auditPayload.frames) {
        const h = f.hands?.[side];
        if (!h?.observed || h.score < .2 || h.keypoints?.length !== 21 || h.keypoints.some(k => !k.position || k.score <= 0)) continue;
        const w = h.keypoints[0].position;
        if(options.reference==='chain') {
          // 2D arc length reduces curl dependence, but still has foreshortening.
          const chains=[[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20]];
          radii.push(Math.max(...chains.map(ids=>ids.slice(1).reduce((sum,id,j)=>{
            const a=h.keypoints[ids[j]].position,b=h.keypoints[id].position;
            return sum+Math.hypot(a.x-b.x,a.y-b.y);
          },0))));
        } else radii.push(Math.max(...h.keypoints.map(k => Math.hypot(k.position.x-w.x,k.position.y-w.y))));
      }
      radii.sort((a,b)=>a-b);
      const scale = 1 + options.scale * ((scales[side] || 1)-1);
      // A fixed per-clip bound, shared by all words; 15% allows stroke thickness.
      const limit = radii.length >= 3 ? radii[Math.floor(radii.length/2)] * scale * 1.15 : null;
      const wrist = body?.score >= .2 ? body.position : null;
      const radius = () => wrist && paths.length ? Math.max(...paths.flatMap(p => {
        const b=p.bounds;
        return [[b.left,b.top],[b.right,b.top],[b.left,b.bottom],[b.right,b.bottom]]
          .map(([x,y])=>Math.hypot(x-wrist.x,y-wrist.y));
      })) : null;
      const before = radius();
      const targetFactor = options.cap && before > 0 && limit > 0 ? Math.min(1,limit/before) : 1;
      let factor=targetFactor;
      if(options.slewRate && before>0 && limit>0) {
        if(history[side].has(frameIndex)) factor=history[side].get(frameIndex);
        else {
          const earlier=[...history[side].keys()].filter(i=>i<frameIndex);
          const previousIndex=earlier.length?Math.max(...earlier):null;
          const previous=previousIndex===null?1:history[side].get(previousIndex);
          const dt=previousIndex===null?1/30:(frameIndex-previousIndex)/window.auditPayload.meta.fps;
          const step=options.slewRate*dt;
          factor=Math.max(previous-step,Math.min(previous+step,targetFactor));
        }
      }
      history[side].set(frameIndex,factor);
      if (factor < 1) for (const p of paths) p.scale(factor,new illustration.scope.Point(wrist.x,wrist.y));
      records[side] = {observed:!!hand?.observed, score:hand?.score || 0,
        nativePaths:paths.length, limit, before, after:radius(), factor,targetFactor,
        referenceFrames:radii.length};
    }
    window.auditFallbackRecord = records;
    if(bodySnapshot()!==beforeBody)throw Error('Fallback cap modified non-hand paths');
  };
}
