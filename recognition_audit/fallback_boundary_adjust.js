// Injected into an isolated audit build, immediately before camera framing.
window.auditFallbackOps = {};
if (window.auditOptions?.fallbackAdjust) {
  for (const side of ['left','right']) {
    const paths=canvasScope.project.activeLayer.children.filter(p=>p.data.auditNativeSide===side && p.visible && p.opacity>0);
    if (!paths.length) continue;
    let reference=null, distance=0;
    for(let d=1;d<=6 && !reference;d++) for(const j of [frameIndex-d,frameIndex+d]) {
      const h=sequence.frames[j]?.hands?.[side];
      if(h?.observed && getDetailedHandAlpha(h)>0 && h.keypoints?.length===21){reference=h;distance=d;break;}
    }
    if(!reference)continue;
    const wrist=getPoseKeypoint(frame.pose,side+'Wrist');
    if(!wrist || wrist.score<.2)continue;
    const p=wrist.position, pivot=new canvasScope.Point(p.x,p.y);
    const boxes=paths.map(q=>q.strokeBounds);
    const left=Math.min(...boxes.map(b=>b.left)),right=Math.max(...boxes.map(b=>b.right)),top=Math.min(...boxes.map(b=>b.top)),bottom=Math.max(...boxes.map(b=>b.bottom));
    const nx=(left+right)/2-p.x,ny=(top+bottom)/2-p.y;
    const raw=reference.keypoints.map(k=>k.position),w=raw[0],s=Number(handOverlayScales[side]||1);
    const vx=[5,9,13,17].reduce((t,i)=>t+raw[i].x-w.x,0)/4,vy=[5,9,13,17].reduce((t,i)=>t+raw[i].y-w.y,0)/4;
    if(Math.hypot(nx,ny)<1e-6 || Math.hypot(vx,vy)<1e-6)continue;
    let angle=(Math.atan2(vy,vx)-Math.atan2(ny,nx))*180/Math.PI;
    angle=((angle+540)%360)-180;
    const diagonal=Math.hypot(right-left,bottom-top);
    const target=Math.hypot(Math.max(...raw.map(q=>q.x))-Math.min(...raw.map(q=>q.x)),Math.max(...raw.map(q=>q.y))-Math.min(...raw.map(q=>q.y)))*s;
    if(diagonal<1e-6 || target<1e-6)continue;
    const t=1-distance/7,weight=t*t*(3-2*t);
    const rotation=Math.max(-45,Math.min(45,angle))*weight;
    const scale=1+(Math.max(.85,Math.min(1.15,target/diagonal))-1)*weight;
    for(const path of paths){path.rotate(rotation,pivot);path.scale(scale,pivot);}
    window.auditFallbackOps[side]={distance,rotation,scale};
  }
}
