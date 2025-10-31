
/**
 * realtime_tcn.js
 * Real-Time Drowsiness · Eye+Mouth CNN + TCN (15 FPS, optional per-session μ/σ)
 * 
 * Usage (in HTML):
 * <script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0"></script>
 * <script src="https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/face_mesh.js"></script>
 * <script src="realtime_tcn.js"></script>
 * 
 * Assumes the following DOM elements exist (IDs must match):
 * - video: #video
 * - canvases: #procCanvas (640x480), #downCanvas (320x240)
 * - toggleRes, toggleCrops checkboxes
 * - stats spans: #yaw, #pitch, #roll, #dyaw, #dpitch, #droll, #fps, #domEyeTxt, #procRes, #tcnText
 * - state spans: #eyeState, #blinkState, #mouthState, #nodState
 * - cards: #pecCard, #yawnCard, #nodCard, #blinkCard
 * - baseline UI: #baselineImg, #baselineTime, #downloadBaselineBtn, #normSummary
 * - control buttons: #reqCamBtn, #startBtn, #stopBtn
 * 
 * Optional: a checkbox with id="toggleNorm" to enable/disable μ/σ capture at runtime.
 *   - Or call: window.TCN.setNormalizationEnabled(true|false)
 */

(function(){
  // Guard against double-initialization
  if (window.__REALTIME_TCN_LOADED__) return;
  window.__REALTIME_TCN_LOADED__ = true;

  /* ===================== Config (paths) ===================== */
  const MODEL_PATHS = {
    eye:  "web_model_eye/model.json",
    yawn: "web_model_yawn/model.json",
    tcn:  "tfjs_model/model.json"
  };

  /* ===================== Feature schema ===================== */
  const FEAT_ORDER = [
    "yaw_adj", "pitch_adj", "roll_adj",
    "dyaw_adj_per_s", "dpitch_adj_per_s", "droll_adj_per_s",
    "eye_open_unified", "ema_eye_open_1s", "ema_eye_open_5s", "eye_open_trend_3s",
    "eye_close_dur_s", "eye_run_len_frames", "perclos_30s", "blink_rate_30s",
    "max_close_run_10s", "time_since_last_blink_s",
    "yawn_prob_ema_1s", "mouth_open_run_s", "mouth_open_rate_30s", "time_since_last_yawn_s"
  ];
  const FEAT_NAMES = FEAT_ORDER.slice();
  const F = FEAT_ORDER.length;

  /* ===================== Timing & constants ===================== */
  const TARGET_FPS = 15;
  const DT = 1 / TARGET_FPS;
  const FRAME_INTERVAL_MS = 1000 / TARGET_FPS;
  const TCN_WINDOW = 90;

  // Eye thresholds + durations (extractor parity)
  const EYE_CLOSE_T = 0.40;
  const FRM_2P5S = Math.round(0.8 * TARGET_FPS);
  const BLINK_MIN_F = 2;
  const BLINK_MAX_F = 6;
  const EYE_DEBOUNCE_ON_F = 2;
  const EYE_DEBOUNCE_OFF_F = 2;
  const BLINK_LOCK_FRM = Math.round(2.0 * TARGET_FPS);

  // Rolling windows
  const PER_30S = 30 * TARGET_FPS;

  // Mouth thresholds + hysteresis
  const MOUTH_ON_T = 0.55;
  const MOUTH_OFF_T = 0.45;
  const MOUTH_PROLONG_S = 1.3;
  const MOUTH_SHORT_MIN_S = 0.10;
  const MOUTH_SHORT_MAX_S = 0.50;

  // Caps
  const TS_MAX = 30; // seconds

  // TCN hysteresis for display
  const ON_THRESH  = 0.65;
  const OFF_THRESH = 0.55;

  // μ/σ defaults
  let normalizationEnabled = false; // default per user request
  const NORM_SECONDS = 10.0;
  const MIN_ACCEPTED = 60; // ~4s @15fps minimum

  /* ===================== Simple ring buffer with running sum ===================== */
  class Ring {
    constructor(n) { this.n = n; this.buf = new Float32Array(n); this.sum = 0; this.count = 0; this.i = 0; }
    push(v) {
      if (this.count < this.n) { this.buf[this.i] = v; this.sum += v; this.i = (this.i + 1) % this.n; this.count++; }
      else { this.sum -= this.buf[this.i]; this.buf[this.i] = v; this.sum += v; this.i = (this.i + 1) % this.n; }
    }
    mean() { return this.count ? this.sum / this.count : 0; }
    reset() { this.sum = 0; this.count = 0; this.i = 0; this.buf.fill(0); }
  }

  /* ===================== DOM refs ===================== */
  const $ = id => document.getElementById(id);
  const video = $("video") || $("videoEl") || $("videoInput") || $("video"); // fallbacks
  const videoEl = $("video") || video;

  const procCanvas = $("procCanvas"), pctx = procCanvas.getContext("2d");
  const downCanvas = $("downCanvas"), dctx = downCanvas.getContext("2d");
  const eyeCanvas = $("eyeCanvas");
  const mouthCanvas = $("mouthCanvas");
  const cropsRow = $("cropsRow");
  const toggleRes = $("toggleRes");
  const toggleCrops = $("toggleCrops");
  const baselineImg = $("baselineImg");
  const baselineTime = $("baselineTime");
  const downloadBaselineBtn = $("downloadBaselineBtn");
  const reqCamBtn = $("reqCamBtn");
  const startBtn = $("startBtn");
  const stopBtn = $("stopBtn");
  const toggleNorm = $("toggleNorm"); // optional external control

  const yawSpan = $("yaw"), pitchSpan = $("pitch"), rollSpan = $("roll");
  const dyawSpan = $("dyaw"), dpitchSpan = $("dpitch"), drollSpan = $("droll");
  const eyeStateSpan = $("eyeState"), mouthStateSpan = $("mouthState");
  const nodStateSpan = $("nodState"), fpsSpan = $("fps"), domEyeTxt = $("domEyeTxt");
  const procResSpan = $("procRes"), tcnText = $("tcnText");
  const pecCard = $("pecCard"), yawnCard = $("yawnCard");
  const nodCard = $("nodCard"), blinkCard = $("blinkCard");
  const normSummary = $("normSummary");

  /* ===================== State ===================== */
  let eyeModel, yawnModel, tcnModel;
  let stream = null, rafId = null, running = false, baselineCaptured = false;
  let dominantEye = "both";
  let fpsCounter = 0, lastFpsT = performance.now();
  let lastProcessT = 0;

  // Pose delta smoothing
  const FPS_FIXED = 15, DT_FIXED = 1 / FPS_FIXED, MAXN = 5;
  const angleBufs = { yaw: [], pitch: [], roll: [] };
  const deltaEMA = { yaw: 0, pitch: 0, roll: 0 };

  // Baseline pose calibrator
  function vsub(a,b){return {x:a.x-b.x,y:a.y-b.y,z:(a.z??0)-(b.z??0)};}
  function vcross(a,b){return {x:a.y*(b.z??0)-(a.z??0)*b.y,y:(a.z??0)*b.x-a.x*(b.z??0),z:a.x*b.y-a.y*b.x};}
  function vnorm(v){return Math.hypot(v.x,v.y,v.z??0)||1e-8;}
  function vunit(v){const n=vnorm(v);return {x:v.x/n,y:v.y/n,z:(v.z??0)/n};}
  function eulerFromAxes(X,Y,Z){return {yaw:Math.atan2(Z.x,Z.z),pitch:Math.atan2(Z.y,Z.z),roll:Math.atan2(X.y,X.x)};}
  const IDX = { rightEyeOuter: 33, leftEyeOuter: 263, chin: 152, forehead: 10 };

  function faceAxesFromLandmarks(lm){
    const R=lm[IDX.rightEyeOuter], L=lm[IDX.leftEyeOuter], C=lm[IDX.chin], F=lm[IDX.forehead];
    let X=vunit(vsub(L,R)), Y=vunit(vsub(C,F)), Z=vunit(vcross(X,Y)); Y=vunit(vcross(Z,X));
    return {X,Y,Z};
  }
  function makeBaselineCalibrator(){
    let R0=null;
    const cross=(a,b)=>({x:a.y*(b.z??0)-(a.z??0)*b.y,y:(a.z??0)*b.x-a.x*(b.z??0),z:a.x*b.y-a.y*b.x});
    const mul=(M,v)=>({x:M[0][0]*v.x+M[0][1]*v.y+M[0][2]*(v.z??0),y:M[1][0]*v.x+M[1][1]*v.y+M[1][2]*(v.z??0),z:M[2][0]*v.x+M[2][1]*v.y+M[2][2]*(v.z??0)});
    return{
      reset(){R0=null;},
      has(){return !!R0;},
      setManualAxes(X,Y,Z){const Xn=vunit(X), Zn=vunit(cross(Xn,Y)), Yn=vunit(cross(Zn,Xn)); R0=[[Xn.x,Yn.x,Zn.x],[Xn.y,Yn.y,Zn.y],[Xn.z,Yn.z,Zn.z]];},
      applyIfReady(X,Y,Z){
        if(!R0) return [X,Y,Z];
        const Rt=[[R0[0][0],R0[1][0],R0[2][0]],[R0[0][1],R0[1][1],R0[2][1]],[R0[0][2],R0[1][2],R0[2][2]]];
        const Xo=vunit(mul(Rt,X)), Yo=vunit(mul(Rt,Y)), Zo=vunit(mul(Rt,Z));
        return [Xo,Yo,Zo];
      }
    };
  }
  const baseline = makeBaselineCalibrator();

  // μ/σ accumulation
  let normMode=false, normT0=0, baseCnt=0, baseSum=new Float64Array(FEAT_NAMES.length), baseSqSum=new Float64Array(FEAT_NAMES.length);
  const baselineStats={mu:null,sigma:null};

  function resetNorm(){
    normMode = normalizationEnabled;
    if(!normMode){
      normSummary.textContent = "μ/σ — disabled";
      return;
    }
    normT0 = performance.now()/1000; baseCnt=0; baseSum.fill(0); baseSqSum.fill(0);
    normSummary.textContent = "μ/σ — collecting…"; baselineTime.textContent = `collecting… 0.0 / ${NORM_SECONDS.toFixed(1)}s (0)`;
  }
  function addNormSample(vec){ for(let i=0;i<FEAT_NAMES.length;i++){ baseSum[i]+=vec[i]; baseSqSum[i]+=vec[i]*vec[i]; } baseCnt++; }
  function finalizeNorm(){
    if(!normMode) return true;
    if(baseCnt < MIN_ACCEPTED){ return false; }
    const mu=new Float32Array(FEAT_NAMES.length), sigma=new Float32Array(FEAT_NAMES.length);
    for(let i=0;i<FEAT_NAMES.length;i++){ const m=baseSum[i]/baseCnt; const v=Math.max(1e-6, baseSqSum[i]/baseCnt - m*m); mu[i]=m; sigma[i]=Math.sqrt(v); }
    baselineStats.mu=mu; baselineStats.sigma=sigma;
    normSummary.textContent = `μ/σ ready (N=${baseCnt}) — ` + FEAT_NAMES.map((n,i)=>`${n}: ${mu[i].toFixed(2)}/${sigma[i].toFixed(2)}`).join(" | ");
    return true;
  }
  function rawToVector(raw){
    const v = new Float32Array(FEAT_ORDER.length);
    for(let i=0;i<FEAT_ORDER.length;i++) v[i]=Number(raw[FEAT_ORDER[i]] ?? 0);
    return v;
  }

  // Angle deltas smoothing
  function getSmoothDelta(name, newVal){
    const buf = angleBufs[name]; buf.push(newVal); if(buf.length>MAXN) buf.shift();
    let d=0;
    if(buf.length>=3) d=(buf[buf.length-1]-buf[buf.length-3])/(2*DT_FIXED);
    else if(buf.length>=2) d=(buf[buf.length-1]-buf[buf.length-2])/DT_FIXED;
    const k=0.3; deltaEMA[name] = deltaEMA[name]*(1-k)+d*k; return deltaEMA[name];
  }

  // Feature buffer for TCN
  const featBuf = [];

  function buildRawFeatures(obj){
    // Enforce defaults
    return {
      yaw_adj: obj.yawDeg,
      pitch_adj: obj.pitchDeg,
      roll_adj: obj.rollDeg,
      dyaw_adj_per_s: obj.dYaw,
      dpitch_adj_per_s: obj.dPitch,
      droll_adj_per_s: obj.dRoll,
      eye_open_unified: finiteOr(obj.eye_open_unified, 0.5),
      ema_eye_open_1s: finiteOr(obj.ema_eye_open_1s, 0.5),
      ema_eye_open_5s: finiteOr(obj.ema_eye_open_5s, 0.5),
      eye_open_trend_3s: finiteOr(obj.eye_open_trend_3s, 0.0),
      eye_close_dur_s: finiteOr(obj.eye_close_dur_s, 0.0),
      eye_run_len_frames: obj.eye_run_len_frames ?? 0,
      perclos_30s: finiteOr(obj.perclos_30s, 0.0),
      blink_rate_30s: finiteOr(obj.blink_rate_30s, 0.0),
      max_close_run_10s: finiteOr(obj.max_close_run_10s, 0.0),
      time_since_last_blink_s: finiteOr(obj.time_since_last_blink_s, 0.0),
      yawn_prob_ema_1s: finiteOr(obj.yawn_prob_ema_1s, 0.0),
      mouth_open_run_s: finiteOr(obj.mouth_open_run_s, 0.0),
      mouth_open_rate_30s: finiteOr(obj.mouth_open_rate_30s, 0.0),
      time_since_last_yawn_s: finiteOr(obj.time_since_last_yawn_s, 0.0)
    };
  }
  function finiteOr(v, def){ return Number.isFinite(v) ? v : def; }

  function pushFrameFeatures(raw){
    const row = new Float32Array(F);
    for(let i=0;i<F;i++){
      const key = FEAT_ORDER[i];
      const v = raw[key];
      const mu = baselineStats.mu?.[i] ?? 0;
      const s = Math.max(1e-3, baselineStats.sigma?.[i] ?? 1);
      row[i] = (v - mu) / s;
    }
    featBuf.push(row);
    if(featBuf.length>TCN_WINDOW) featBuf.shift();
  }

  let tcnIsDrowsy=false;

  function tcnPredictIfReady(){
    if(!tcnModel){ tcnText.textContent = "TCN not loaded"; return; }
    if(featBuf.length < TCN_WINDOW){
      tcnText.textContent = baselineStats.mu ? `Warming ${featBuf.length}/${TCN_WINDOW}…` : (normalizationEnabled ? "Awaiting μ/σ…" : `Warming ${featBuf.length}/${TCN_WINDOW}…`);
      return;
    }
    let x=null, y=null;
    try{
      const flat=new Float32Array(TCN_WINDOW * F);
      for(let t=0;t<TCN_WINDOW;t++) flat.set(featBuf[t], t*F);
      x=tf.tensor3d(flat,[1,TCN_WINDOW,F]);
      y=tcnModel.predict(x);
      const prob=y.dataSync()[0];
      if(!tcnIsDrowsy && prob>=ON_THRESH) tcnIsDrowsy=true;
      else if(tcnIsDrowsy && prob<=OFF_THRESH) tcnIsDrowsy=false;
      tcnText.textContent = `${tcnIsDrowsy ? "Drowsy" : "Awake"} (TCN ${prob.toFixed(2)})`;
    } finally { x?.dispose(); y?.dispose(); }
  }

  /* ===================== FaceMesh ===================== */
  const faceMesh = new FaceMesh({ locateFile: f => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${f}` });
  faceMesh.setOptions({ maxNumFaces: 1, refineLandmarks: false, minDetectionConfidence: 0.5, minTrackingConfidence: 0.5 });

  toggleCrops?.addEventListener("change", ()=>{ if(cropsRow) cropsRow.style.display = toggleCrops.checked ? "flex" : "none"; });

  function drawToProcCanvasCover(){
    const vw = videoEl.videoWidth || 1, vh = videoEl.videoHeight || 1;
    const tw = procCanvas.width, th = procCanvas.height, tarAR = tw/th, srcAR = vw/vh;
    let sx,sy,sw,sh;
    if(srcAR>tarAR){ sh=vh; sw=Math.floor(vh*tarAR); sx=Math.floor((vw-sw)/2); sy=0; }
    else{ sw=vw; sh=Math.floor(vw/tarAR); sx=0; sy=Math.floor((vh-sh)/2); }
    pctx.drawImage(videoEl, sx,sy,sw,sh, 0,0, tw,th);
  }
  function setProcResLabel(){
    if(!procResSpan) return;
    if(toggleRes?.checked) procResSpan.textContent = `${downCanvas.width}×${downCanvas.height} (from ${procCanvas.width}×${procCanvas.height})`;
    else procResSpan.textContent = `${procCanvas.width}×${procCanvas.height}`;
  }

  /* ===================== Runtime vars for extractor-parity logic ===================== */
  // Eye
  let eyeClosedDebounced=false, eyeDebOn=0, eyeDebOff=0;
  let eyeRunLenFrames=0;
  let prolongedEyeActive=false, lastProlongStartFrame=-1e9;
  let blinkPulse=0;
  const eyeRuns=[]; // {tEnd,lenFrames,durS}
  let lastBlinkEndTime=0;
  let hasSeenBlink=0;

  // Eye EMAs
  const alphaEye1 = 1 - Math.exp(-DT / 1.0);
  const alphaEye5 = 1 - Math.exp(-DT / 5.0);
  let emaEye1=NaN, emaEye5=NaN;

  // Eye closed ring
  const rbEyeClosed = new Ring(PER_30S);

  // Mouth
  const alphaYawn1 = alphaEye1;
  let yawnEma1=NaN;
  let mouthOpen=false, mouthRunFrames=0, yawnProlonged=false;
  const mouthRuns=[]; // {tEnd,durS}
  let yawnPulse=0, lastYawnEndTime=0;

  // Nod
  let nodActive=false, stableFrames=0;

  // Misc
  let runningFlag=false;
  let streamFlag=false;

  /* ===================== TF init & models ===================== */
  async function initTf(){
    try{ await tf.setBackend("webgl"); }catch{ await tf.setBackend("wasm"); }
    await tf.ready();
    eyeModel = await tf.loadLayersModel(MODEL_PATHS.eye);
    yawnModel = await tf.loadLayersModel(MODEL_PATHS.yawn);
    tcnModel  = await tf.loadLayersModel(MODEL_PATHS.tcn);

    const inShape = tcnModel.inputs[0].shape; // [null, 90, F]
    const modelF = inShape[2];
    if(modelF !== F){
      alert(`TCN expects F=${modelF} features but FEAT_ORDER has ${F}. Update FEAT_ORDER & buildRawFeatures().`);
    }
  }

  /* ===================== Processing loop ===================== */
  faceMesh.onResults(async (res)=>{
    if(!runningFlag) return;
    const nowSec = performance.now()/1000;
    const lm = res.multiFaceLandmarks?.[0];
    if(!lm) return;

    const useDown = !!(toggleRes?.checked);
    const W = useDown ? downCanvas.width : procCanvas.width;
    const H = useDown ? downCanvas.height : procCanvas.height;

    // Pose
    const preAxes = faceAxesFromLandmarks(lm);
    let [X,Y,Z] = baseline.has() ? baseline.applyIfReady(preAxes.X, preAxes.Y, preAxes.Z) : [preAxes.X, preAxes.Y, preAxes.Z];
    const {yaw, pitch, roll} = eulerFromAxes(X,Y,Z);
    const yawDeg = yaw*180/Math.PI, pitchDeg=pitch*180/Math.PI, rollDeg=roll*180/Math.PI;

    // Initial baseline snapshot
    if(!baselineCaptured){
      const rawYawDeg = Math.atan2(preAxes.Z.x, preAxes.Z.z) * 180/Math.PI;
      dominantEye = rawYawDeg > 10 ? "right" : (rawYawDeg < -10 ? "left" : "both");
      domEyeTxt && (domEyeTxt.textContent = dominantEye);

      baseline.reset(); baseline.setManualAxes(preAxes.X, preAxes.Y, preAxes.Z);
      drawToProcCanvasCover();
      procCanvas.toBlob(blob => {
        const url = URL.createObjectURL(blob);
        if(baselineImg) baselineImg.src = url;
        if(baselineTime) baselineTime.textContent = `t=${nowSec.toFixed(2)}s`;
        if(downloadBaselineBtn){
          downloadBaselineBtn.disabled = false;
          downloadBaselineBtn.onclick = ()=>{ const a=document.createElement('a'); a.href = baselineImg.src; a.download = 'baseline_snapshot.png'; a.click(); };
        }
      });
      baselineCaptured = true;
      resetNorm(); // start μ/σ only if enabled
    }

    const dYaw = getSmoothDelta("yaw", yawDeg);
    const dPitch = getSmoothDelta("pitch", pitchDeg);
    const dRoll = getSmoothDelta("roll", rollDeg);

    // Draw frames
    drawToProcCanvasCover();
    if(useDown) dctx.drawImage(procCanvas,0,0,downCanvas.width,downCanvas.height);
    const frame = tf.browser.fromPixels(useDown?downCanvas:procCanvas);

    // Crops + predictions
    const LIDX=[33,133,159,145], RIDX=[362,263,386,374], MIDX=[61,291,13,14,81,178,308,402];
    function paddedBox(xs,ys,W,H,pad=1.8,maxS=300){
      let cx=(Math.min(...xs)+Math.max(...xs))/2, cy=(Math.min(...ys)+Math.max(...ys))/2;
      let w=Math.max(...xs)-Math.min(...xs), h=Math.max(...ys)-Math.min(...ys);
      let size=Math.min(Math.max(w,h)*pad, maxS);
      let x=Math.max(0, Math.floor(cx-size/2)), y=Math.max(0, Math.floor(cy-size/2));
      if(x+size>W) size=W-x; if(y+size>H) size=H-y;
      let s=Math.floor(size); if(s<=10) return null;
      return {x,y,w:s,h:s};
    }
    function cropTensor(idxs,outW,outH){
      const pts=idxs.map(i=>lm[i]); const xs=pts.map(p=>p.x*W), ys=pts.map(p=>p.y*H);
      const box=paddedBox(xs,ys,W,H,1.8,300); if(!box) return null;
      const c=tf.slice(frame,[Math.floor(box.y),Math.floor(box.x),0],[Math.floor(box.h),Math.floor(box.w),3]);
      const g=tf.image.resizeBilinear(c,[outH,outW]).mean(2).expandDims(0).expandDims(-1).div(255.0);
      c.dispose(); return g;
    }

    let unifiedEyeProb=NaN, yawnProb=NaN;
    let left=null,right=null,mouth=null,eyeBatch=null,eyeOut=null,mouthOut=null;
    try{
      left=cropTensor(LIDX,90,90); right=cropTensor(RIDX,90,90);
      if(left && right){
        eyeBatch=tf.concat([left,right],0);
        eyeOut = eyeModel.predict(eyeBatch);
        const ev = eyeOut.dataSync();
        const eyeLProb = ev[0], eyeRProb = ev[1];
        const yawAbs = Math.abs(yawDeg), maxYaw=20, yawClamped=Math.min(yawAbs,maxYaw);
        const wL = (yawDeg>=0 ? 1.0 : Math.max(0.1, 1 - yawClamped/maxYaw));
        const wR = (yawDeg<=0 ? 1.0 : Math.max(0.1, 1 - yawClamped/maxYaw));
        unifiedEyeProb = (eyeLProb*wL + eyeRProb*wR)/(wL+wR);
        if(toggleCrops?.checked) await tf.browser.toPixels(left.squeeze(), eyeCanvas);
      }
      mouth=cropTensor(MIDX,120,120);
      if(mouth){
        mouthOut = yawnModel.predict(mouth);
        const yv = mouthOut.dataSync(); yawnProb = yv[0];
        if(toggleCrops?.checked) await tf.browser.toPixels(mouth.squeeze(), mouthCanvas);
      }
    } finally {
      left?.dispose(); right?.dispose(); eyeBatch?.dispose(); eyeOut?.dispose(); mouth?.dispose(); mouthOut?.dispose(); frame.dispose();
    }

    // ---------- Eye pipeline ----------
    const eyeClosedRaw = (!Number.isNaN(unifiedEyeProb) && unifiedEyeProb < EYE_CLOSE_T) ? 1 : 0;
    rbEyeClosed.push(eyeClosedRaw);
    const perclos_30s = rbEyeClosed.mean();

    if(!isFinite(emaEye1)){ const v=isFinite(unifiedEyeProb)?unifiedEyeProb:1.0; emaEye1=v; emaEye5=v; }
    else{
      const v=isFinite(unifiedEyeProb)?unifiedEyeProb:emaEye1;
      emaEye1 += (1 - Math.exp(-DT/1.0)) * (v - emaEye1);
      emaEye5 += (1 - Math.exp(-DT/5.0)) * (v - emaEye5);
    }
    const eye_open_trend_3s = emaEye1 - emaEye5;

    if(eyeClosedRaw){
      eyeDebOn++; eyeDebOff=0;
      if(!eyeClosedDebounced && eyeDebOn>=EYE_DEBOUNCE_ON_F){
        eyeClosedDebounced=true; eyeRunLenFrames=0;
      }
    } else {
      eyeDebOff++; eyeDebOn=0;
      if(eyeClosedDebounced && eyeDebOff>=EYE_DEBOUNCE_OFF_F){
        const lenF=eyeRunLenFrames; const durS=lenF*DT;
        eyeRuns.push({tEnd:nowSec,lenFrames:lenF,durS});
        const cut30=nowSec-30.0; while(eyeRuns.length && eyeRuns[0].tEnd<cut30) eyeRuns.shift();
        if(lenF>=BLINK_MIN_F && lenF<=BLINK_MAX_F){
          lastBlinkEndTime=nowSec; hasSeenBlink=1; blinkPulse=1;
        }
        eyeClosedDebounced=false; eyeRunLenFrames=0;
      }
    }
    if(eyeClosedDebounced) eyeRunLenFrames++;

    if(eyeClosedDebounced){
      if(!prolongedEyeActive && eyeRunLenFrames>=FRM_2P5S && (featBuf.length - lastProlongStartFrame)>=BLINK_LOCK_FRM){
        prolongedEyeActive=true; lastProlongStartFrame=featBuf.length;
      }
    } else {
      prolongedEyeActive=false;
    }
    const prolonged_eye_state = prolongedEyeActive ? 1 : 0;

    const blink_state = (blinkPulse>0 && !eyeClosedDebounced) ? 1 : 0;
    if(blinkPulse>0) blinkPulse--;

    const tNowSec = nowSec;
    const cut10=tNowSec-10.0, cut30=tNowSec-30.0;
    let blinks30=0, max_close_run_10s=0;
    for(const r of eyeRuns){
      if(r.tEnd>=cut30 && r.lenFrames>=BLINK_MIN_F && r.lenFrames<=BLINK_MAX_F) blinks30++;
      if(r.tEnd>=cut10 && r.durS>max_close_run_10s) max_close_run_10s = r.durS;
    }
    const blink_rate_30s = blinks30/30.0;
    const time_since_last_blink_s = Math.min(Math.max(0, tNowSec - lastBlinkEndTime), TS_MAX);

    // Eye UI
    const eyeClosed = !!eyeClosedRaw;
    if(eyeStateSpan) eyeStateSpan.textContent = isFinite(unifiedEyeProb) ? (eyeClosed?`Closed (${unifiedEyeProb.toFixed(2)})`:`Open (${unifiedEyeProb.toFixed(2)})`) : "—";
    if($("blinkState")) $("blinkState").textContent = blink_state ? "Frequent 👀" : "-";

    // ---------- Mouth / yawn ----------
    if(!isFinite(yawnEma1)){ yawnEma1 = isFinite(yawnProb)?yawnProb:0; }
    else { const vy=isFinite(yawnProb)?yawnProb:yawnEma1; yawnEma1 += (1 - Math.exp(-DT/1.0)) * (vy - yawnEma1); }

    if(mouthOpen){
      if(yawnEma1<=MOUTH_OFF_T){
        const durS = mouthRunFrames*DT;
        mouthRuns.push({tEnd:tNowSec, durS});
        const cutM30=tNowSec-30.0; while(mouthRuns.length && mouthRuns[0].tEnd<cutM30) mouthRuns.shift();
        if(durS>=MOUTH_PROLONG_S){ yawnPulse=1; lastYawnEndTime=tNowSec; }
        mouthOpen=false; mouthRunFrames=0; yawnProlonged=false;
      } else {
        mouthRunFrames++;
        if(!yawnProlonged && (mouthRunFrames*DT)>=MOUTH_PROLONG_S) yawnProlonged=true;
      }
    } else {
      if(yawnEma1>=MOUTH_ON_T){ mouthOpen=true; mouthRunFrames=1; yawnProlonged=false; }
    }

    const mouth_open_state = mouthOpen ? 1 : 0;
    const mouth_open_run_s = mouthOpen ? mouthRunFrames*DT : 0;
    const yawn_prolonged_state = yawnProlonged ? 1 : 0;
    const yawn_event_state = (!mouthOpen && yawnPulse>0) ? 1 : 0;
    if(yawnPulse>0) yawnPulse--;

    let mouthShort30=0;
    for(const r of mouthRuns){
      if(r.tEnd>=cut30 && r.durS>=MOUTH_SHORT_MIN_S && r.durS<=MOUTH_SHORT_MAX_S) mouthShort30++;
    }
    const mouth_open_rate_30s = mouthShort30/30.0;
    const time_since_last_yawn_s = Math.min(Math.max(0, tNowSec - lastYawnEndTime), TS_MAX);

    // Mouth UI
    if(mouthStateSpan) mouthStateSpan.textContent = isFinite(yawnProb) ? (yawnEma1>=MOUTH_ON_T ? `Open (${(yawnProb).toFixed(2)})` : `Closed (${(yawnProb).toFixed(2)})`) : "—";

    // ---------- Nod (rule) ----------
    if(prolonged_eye_state && pitchDeg<=-4 && Math.abs(rollDeg)<=20 && Math.abs(yawDeg)<=10) nodActive=true;
    if(!eyeClosed || Math.abs(yawDeg)>10 || (pitchDeg>=-2)) { nodActive=false; stableFrames=0; }
    if(nodStateSpan) nodStateSpan.textContent = nodActive ? "Nodding Off 😴" : "Awake";

    // Cards
    if(pecCard){ if(eyeClosed && (eyeRunLenFrames*DT)>=0.3){ pecCard.classList.add("active"); $("pecStatus").textContent="Active"; } else { pecCard.classList.remove("active"); $("pecStatus").textContent="Inactive"; } }
    if(yawnCard){ if(yawn_prolonged_state){ yawnCard.classList.add("active"); $("yawnStatus").textContent="Active"; } else { yawnCard.classList.remove("active"); $("yawnStatus").textContent="Inactive"; } }
    if(nodCard){ if(nodActive){ nodCard.classList.add("active"); $("nodStatus").textContent="Active"; } else { nodCard.classList.remove("active"); $("nodStatus").textContent="Inactive"; } }

    // ---------- Build TCN features ----------
    const rawFeat = buildRawFeatures({
      yawDeg, pitchDeg: pitchDeg, rollDeg,
      dYaw, dPitch, dRoll,
      eye_open_unified: unifiedEyeProb,
      ema_eye_open_1s: emaEye1,
      ema_eye_open_5s: emaEye5,
      eye_open_trend_3s,
      eye_close_dur_s: eyeClosedDebounced ? eyeRunLenFrames*DT : 0,
      eye_run_len_frames: eyeRunLenFrames,
      perclos_30s,
      blink_rate_30s,
      max_close_run_10s: Number.isFinite(max_close_run_10s)?max_close_run_10s:0,
      time_since_last_blink_s,
      yawn_prob_ema_1s: yawnEma1,
      mouth_open_run_s,
      mouth_open_rate_30s,
      time_since_last_yawn_s
    });

    // μ/σ collection
    if(normMode){
      const elapsed=(performance.now()/1000)-normT0;
      if(baselineTime) baselineTime.textContent = `collecting… ${elapsed.toFixed(1)} / ${NORM_SECONDS.toFixed(1)}s (${baseCnt})`;
      addNormSample(rawToVector(rawFeat));
      if(elapsed>=NORM_SECONDS){
        normMode=false;
        if(!finalizeNorm()) normSummary && (normSummary.textContent="μ/σ — unstable; try again");
      }
    }

    // Push to TCN & predict
    pushFrameFeatures(rawFeat);
    tcnPredictIfReady();

    // Stats
    yawSpan && (yawSpan.textContent = yawDeg.toFixed(1));
    pitchSpan && (pitchSpan.textContent = pitchDeg.toFixed(1));
    rollSpan && (rollSpan.textContent = rollDeg.toFixed(1));
    dyawSpan && (dyawSpan.textContent = dYaw.toFixed(1));
    dpitchSpan && (dpitchSpan.textContent = dPitch.toFixed(1));
    drollSpan && (drollSpan.textContent = dRoll.toFixed(1));

    // FPS
    fpsCounter++; const tNow=performance.now();
    if(tNow - lastFpsT >= 1000){ fpsSpan && (fpsSpan.textContent = fpsCounter); fpsCounter=0; lastFpsT=tNow; }
    setProcResLabel();
  });

  /* ===================== Camera & throttled loop ===================== */
  async function requestCamera(){
    if(streamFlag) return;
    try{
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width:640, height:480, facingMode:"user", frameRate:{ ideal:15, max:15 } },
        audio: false
      });
      const track = stream.getVideoTracks()[0];
      if(track?.applyConstraints){ await track.applyConstraints({ frameRate:{ max:15 } }); }
      videoEl.srcObject = stream;
      await videoEl.play();
      startBtn && (startBtn.disabled = false);
      reqCamBtn && (reqCamBtn.disabled = true);
      streamFlag = true;
    }catch(e){
      alert("Camera access was denied or failed: " + e.message);
    }
  }

  async function startProcessing(){
    if(!streamFlag && !videoEl.srcObject){ alert("Grant camera access first."); return; }
    runningFlag = true; baselineCaptured=false;
    startBtn && (startBtn.disabled = true);
    stopBtn  && (stopBtn.disabled  = false);

    // Reset states
    eyeClosedDebounced=false; eyeDebOn=0; eyeDebOff=0; eyeRunLenFrames=0;
    prolongedEyeActive=false; lastProlongStartFrame=-1e9; blinkPulse=0;
    eyeRuns.length=0; lastBlinkEndTime=0; hasSeenBlink=0;
    emaEye1=NaN; emaEye5=NaN; rbEyeClosed.reset();

    yawnEma1=NaN; mouthOpen=false; mouthRunFrames=0; yawnProlonged=false; mouthRuns.length=0; yawnPulse=0; lastYawnEndTime=0;

    nodActive=false; stableFrames=0; dominantEye="both"; domEyeTxt && (domEyeTxt.textContent="-");
    baseline.reset(); baselineImg && baselineImg.removeAttribute("src"); baselineTime && (baselineTime.textContent="—"); downloadBaselineBtn && (downloadBaselineBtn.disabled=true);
    featBuf.length=0; tcnIsDrowsy=false; tcnText && (tcnText.textContent="—");

    loop();
  }

  function stopProcessing(){
    runningFlag = false;
    startBtn && (startBtn.disabled=false);
    stopBtn  && (stopBtn.disabled=true);
    if(rafId) cancelAnimationFrame(rafId);
  }

  async function loop(){
    if(!runningFlag) return;
    const now=performance.now();
    if(now - lastProcessT >= FRAME_INTERVAL_MS){
      lastProcessT = now;
      drawToProcCanvasCover();
      const input = (toggleRes?.checked) ? (dctx.drawImage(procCanvas,0,0,downCanvas.width,downCanvas.height), downCanvas) : procCanvas;
      await faceMesh.send({ image: input });
    }
    rafId = requestAnimationFrame(loop);
  }

  /* ===================== Public API ===================== */
  window.TCN = window.TCN || {};
  window.TCN.setNormalizationEnabled = function(enabled){
    normalizationEnabled = !!enabled;
    normSummary && (normSummary.textContent = normalizationEnabled ? "μ/σ — will collect on Start" : "μ/σ — disabled");
  };

  // Hook optional toggle checkbox
  toggleNorm && toggleNorm.addEventListener("change", (e)=>{
    window.TCN.setNormalizationEnabled(e.target.checked);
  });

  // Wire UI
  reqCamBtn && (reqCamBtn.onclick = requestCamera);
  startBtn && (startBtn.onclick = startProcessing);
  stopBtn  && (stopBtn.onclick  = stopProcessing);
  toggleRes && toggleRes.addEventListener("change", setProcResLabel);

  // Boot
  (async ()=>{
    try{
      await initTf();
      const warm=document.createElement('canvas'); warm.width=4; warm.height=4; warm.getContext('2d').fillRect(0,0,4,4);
      await faceMesh.send({ image: warm });
      setProcResLabel();
      // Initialize normalization state text
      window.TCN.setNormalizationEnabled(false); // default off per user
    }catch(e){
      console.error(e);
      alert("Models failed to load. Check paths: web_model_eye/, web_model_yawn/, tfjs_model/.");
    }
  })();

})(); // IIFE
