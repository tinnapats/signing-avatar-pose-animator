import path from 'node:path';
import {fileURLToPath} from 'node:url';
const here=path.dirname(fileURLToPath(import.meta.url));
process.env.FALLBACK_OUTPUT=path.join(here,'fallback_v2');
process.env.FALLBACK_VARIANTS=JSON.stringify([
 {id:'scale_only',scale:.5,anchor:1,cap:false},
 {id:'radial',scale:.5,anchor:1,cap:true},
 {id:'chain',scale:.5,anchor:1,cap:true,reference:'chain'},
 {id:'chain_slew',scale:.5,anchor:1,cap:true,reference:'chain',slewRate:3}
]);
await import('./fallback_trial_render.mjs');
