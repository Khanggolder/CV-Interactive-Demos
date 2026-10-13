// Real local Edge/CDP audit of the running Chapter 2 Streamlit app at :8516.
import {spawn} from 'node:child_process';
import {mkdir,mkdtemp,rm,writeFile} from 'node:fs/promises';
import path from 'node:path';

const chapter=path.join(process.cwd(),'chapter2_webapp');
const out=path.join(chapter,'docs','screenshots','orb');
const profile=await mkdtemp(path.join(chapter,'.orb_audit_profile-'));
const edge='C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
await mkdir(out,{recursive:true});
const child=spawn(edge,['--remote-debugging-port=9232',`--user-data-dir=${profile}`,
  '--headless=new','--no-first-run','--disable-gpu','--disable-extensions',
  '--window-size=1920,1080','http://127.0.0.1:8516'],
  {stdio:['ignore','ignore','pipe'],windowsHide:true});
let stderr='';child.stderr.on('data',p=>stderr+=p.toString());
let socket,serial=1;const pending=new Map(),errors=[];
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function call(method,params={}){
  const id=serial++;
  return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
}
async function connect(){
  let pages;
  for(let n=0;n<80;n++){
    try{pages=await(await fetch('http://127.0.0.1:9232/json/list')).json();if(pages.some(p=>p.type==='page'))break;}
    catch{} await delay(200);
  }
  const page=pages?.find(p=>p.type==='page');
  if(!page)throw new Error(`Edge failed: ${child.exitCode} ${stderr.slice(0,1000)}`);
  socket=new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve,reject)=>{socket.onopen=resolve;socket.onerror=reject;});
  socket.onmessage=e=>{const m=JSON.parse(e.data);
    if(m.method==='Runtime.exceptionThrown')errors.push(m.params.exceptionDetails.text);
    if(m.method==='Log.entryAdded'&&m.params.entry.level==='error')errors.push(m.params.entry.text);
    if(m.id&&pending.has(m.id)){const {resolve,reject}=pending.get(m.id);pending.delete(m.id);
      m.error?reject(new Error(JSON.stringify(m.error))):resolve(m.result);}
  };
  await call('Runtime.enable');await call('Page.enable');await call('Log.enable');
}
async function evalJs(expression){
  const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
}
async function waitFor(value){
  for(let n=0;n<120;n++){
    if(await evalJs(`document.body?.innerText.includes(${JSON.stringify(value)}) &&
       ![...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Stop')`)){
      await delay(500);return;
    }await delay(150);
  }
  throw new Error(`Missing ${value}: `+await evalJs('document.body?.innerText.slice(0,2000)'));
}
async function viewport(width,height){
  await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width<800});
  await delay(450);
  await evalJs(`(()=>{window.scrollTo(0,0);const e=document.querySelector('[data-testid="stAppViewContainer"]');if(e)e.scrollTop=0;})()`);
}
async function shot(name){
  for(let n=0;n<100;n++){
    const busy=await evalJs(`[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Stop')`);
    if(!busy)break;await delay(150);
  }
  await delay(450);
  const state=await evalJs(`({width:innerWidth,root:document.documentElement.scrollWidth,
    body:document.body.scrollWidth,exception:!!document.querySelector('[data-testid="stException"]'),
    plots:document.querySelectorAll('[data-testid="stPlotlyChart"]').length})`);
  if(state.root>state.width+2||state.body>state.width+2||state.exception)
    throw new Error(`${name}: ${JSON.stringify(state)} `+await evalJs("document.querySelector('[data-testid=stException]')?.innerText||document.body.innerText.slice(0,1800)"));
  const {data}=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false,fromSurface:true});
  await writeFile(path.join(out,name),Buffer.from(data,'base64'));
  console.log(name,JSON.stringify(state));
}
async function clickButton(label){
  for(let n=0;n<50;n++){
    const clicked=await evalJs(`(()=>{const b=[...document.querySelectorAll('button')]
      .find(e=>e.innerText.trim()===${JSON.stringify(label)}&&!e.disabled);if(!b)return false;b.click();return true;})()`);
    if(clicked){await delay(850);return;}await delay(150);
  }
  throw new Error(`Enabled button missing: ${label}`);
}
async function clickLabel(label){
  for(let n=0;n<60;n++){
    const ok=await evalJs(`(()=>{const x=[...document.querySelectorAll('label')]
      .find(e=>e.innerText.trim()===${JSON.stringify(label)});if(!x)return false;x.click();return true;})()`);
    if(ok){await delay(750);return;}await delay(150);
  }
  throw new Error(`Label missing: ${label}`);
}
async function select(value,ordinal=0){
  const opened=await evalJs(`(()=>{const c=document.querySelectorAll('[data-testid="stSelectbox"] [role="combobox"]')[${ordinal}];
    if(!c)return false;c.click();return true;})()`);
  if(!opened)throw new Error('Selectbox missing');
  await delay(250);
  const picked=await evalJs(`(()=>{const x=[...document.querySelectorAll('[role="option"]')]
    .find(e=>e.innerText.trim()===${JSON.stringify(value)});if(!x)return false;x.click();return true;})()`);
  if(!picked)throw new Error(`Option missing: ${value}`);
  await delay(850);
}
async function selectByLabel(label,value){
  const opened=await evalJs(`(()=>{const box=[...document.querySelectorAll('[data-testid="stSelectbox"]')]
    .find(e=>e.innerText.includes(${JSON.stringify(label)}));const c=box?.querySelector('[role="combobox"]');
    if(!c)return false;c.click();return true;})()`);
  if(!opened)throw new Error(`Selectbox label missing: ${label}`);
  await delay(250);
  const picked=await evalJs(`(()=>{const x=[...document.querySelectorAll('[role="option"]')]
    .find(e=>e.innerText.trim()===${JSON.stringify(value)});if(!x)return false;x.click();return true;})()`);
  if(!picked)throw new Error(`Option missing: ${value}`);
  await delay(850);
}
async function tab(value){
  const ok=await evalJs(`(()=>{const x=[...document.querySelectorAll('[role="tab"]')]
    .find(e=>e.innerText.trim()===${JSON.stringify(value)});if(!x)return false;x.click();return true;})()`);
  if(!ok)throw new Error(`Tab missing: ${value}`);
  await delay(400);
}
async function scrollToText(value){
  await evalJs(`(()=>{const x=[...document.querySelectorAll('h2,h3,h4,p')]
    .find(e=>e.innerText.includes(${JSON.stringify(value)}));x?.scrollIntoView({block:'start'});})()`);
  await delay(300);
}
async function scrollToVisiblePlot(last=false){
  await evalJs(`(()=>{const a=[...document.querySelectorAll('[data-testid="stPlotlyChart"]')]
    .filter(e=>e.getClientRects().length);(a[${last?"a.length-1":"0"}])?.scrollIntoView({block:'start'});})()`);
  await delay(350);
}

try{
  await connect();await waitFor('Harris & Shi-Tomasi Corner Detection');
  await select('FAST, BRIEF & ORB');await waitFor('BƯỚC 1/18');
  if(process.argv.includes('--mobile-only')){
    await viewport(390,844);
    await clickLabel('Experiment Lab');await waitFor('Lab A · FAST Corner Explorer');
    await clickLabel('Presentation Mode');await waitFor('FAST · vòng 16 pixel');
    await scrollToVisiblePlot();await shot('presentation-mobile-chart.png');
  }else{
  for(const [w,h,label] of [[1920,1080,'desktop'],[1366,768,'laptop'],[768,1024,'tablet'],[390,844,'mobile']]){
    await viewport(w,h);await shot(`guided-initial-${label}.png`);
  }
  await viewport(1366,768);
  const predictions={4:'CORNER',5:'Không',6:'Có',10:'Có',12:'Có',16:'Có',17:'Không',18:'Không'};
  const names={2:'architecture.png',3:'fast-circle.png',4:'fast-prediction.png',
    5:'fast-pretest.png',6:'fast-wraparound.png',7:'fast-score-nms.png',
    8:'orb-pyramid.png',9:'orientation-need.png',10:'centroid.png',
    11:'rotation.png',12:'brief-pair.png',13:'descriptor-bits.png',
    14:'oriented-brief.png',15:'pattern-learning.png',16:'hamming.png',
    17:'matching.png',18:'recap.png'};
  for(let step=1;step<=18;step++){
    if(names[step])await shot(names[step]);
    if(step===3){await scrollToVisiblePlot();await shot('fast-circle-chart.png');}
    if(step===10){await scrollToVisiblePlot();await shot('centroid-chart.png');}
    if(predictions[step]){
      await clickLabel(predictions[step]);await clickButton('Reveal kết quả');await waitFor('Đúng.');
      if([4,5,6,16,17].includes(step))await shot(`revealed-${step}.png`);
    }
    if(step<18){await clickButton('Next →');await waitFor(`BƯỚC ${step+1}/18`);}
  }
  await clickLabel('Experiment Lab');await waitFor('Lab A · FAST Corner Explorer');
  await shot('lab-a.png');await clickLabel('CORNER');await clickButton('Run pretest + full test / Reveal');
  await shot('lab-a-reveal.png');
  await select('Preset 129×129',1);await waitFor('Góc mạnh');await shot('lab-a-preset.png');
  await select('Microscope 15×15',1);
  await tab('B · Orientation/BRIEF');await waitFor('Lab B · Orientation & Binary Descriptor');
  await shot('lab-b.png');await clickButton('Tính lại ảnh quay');await waitFor('Vị trí affine dự kiến');
  await clickButton('Compute 256-bit descriptor');await waitFor('Byte 0');
  await scrollToText('BRIEF · hai pixel');await shot('lab-b-descriptor.png');
  await clickLabel('rBRIEF steering');await waitFor('Thay input/hướng');await shot('lab-b-invalidated.png');
  await clickButton('Compute 256-bit descriptor');await waitFor('Byte 0');
  await tab('C · Matching');await waitFor('Lab C · Matching & SIFT Comparison');
  await clickButton('Calculate A/B descriptors');await waitFor('accepted:');await shot('lab-c.png');
  await selectByLabel('Matching rule','cross-check');await waitFor('cross-check');await shot('lab-c-cross-check.png');
  await clickLabel('Presentation Mode');await waitFor('FAST · BRIEF · ORB-style');await shot('presentation-laptop.png');
  await scrollToVisiblePlot();await shot('presentation-laptop-chart.png');
  await viewport(390,844);await shot('presentation-mobile.png');
  await scrollToVisiblePlot();await shot('presentation-mobile-chart.png');
  await clickLabel('Presentation Mode');await waitFor('FAST · BRIEF · ORB-style');await shot('lab-mobile.png');
  await clickLabel('Guided Tutorial');await waitFor('18 · ORB-style so với SIFT');await shot('guided-end-mobile.png');
  await viewport(1366,768);
  await select('Harris & Shi-Tomasi Corner Detection');await waitFor('Vì sao Harris tìm được góc?');await shot('harris-regression.png');
  await select('Gaussian Scale Space, LoG & DoG');await waitFor('BƯỚC 1/15');await shot('scale-space-regression.png');
  await select('SIFT Interactive Tutorial');await waitFor('BƯỚC 1/18');await shot('sift-regression.png');
  }
  console.log('Browser errors',JSON.stringify(errors));
  if(errors.length)throw new Error('Browser console/JavaScript errors');
}finally{
  try{await call('Browser.close');}catch{child.kill();}
  socket?.close();
  if(path.dirname(path.resolve(profile))!==path.resolve(chapter)||
     !path.basename(profile).startsWith('.orb_audit_profile-'))throw new Error(`Unsafe profile: ${profile}`);
  await rm(profile,{recursive:true,force:true,maxRetries:8,retryDelay:100});
}
