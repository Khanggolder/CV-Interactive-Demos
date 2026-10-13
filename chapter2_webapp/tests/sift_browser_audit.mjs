// Actual Edge/CDP browser audit. Start Chapter 2 Streamlit on 127.0.0.1:8515.
import {spawn} from 'node:child_process';
import {mkdir,mkdtemp,rm,writeFile} from 'node:fs/promises';
import path from 'node:path';

const chapter=path.join(process.cwd(),'chapter2_webapp');
const out=path.join(chapter,'docs','screenshots','sift');
const profile=await mkdtemp(path.join(chapter,'.sift_audit_profile-'));
const edge='C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
await mkdir(out,{recursive:true});
const child=spawn(edge,['--remote-debugging-port=9231',`--user-data-dir=${profile}`,
  '--headless=new','--no-first-run','--disable-gpu','--disable-extensions',
  '--window-size=1920,1080','http://127.0.0.1:8515'],
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
  for(let n=0;n<70;n++){
    try{pages=await(await fetch('http://127.0.0.1:9231/json/list')).json();if(pages.some(p=>p.type==='page'))break;}
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
async function waitFor(text){
  for(let n=0;n<100;n++){
    if(await evalJs(`document.body?.innerText.includes(${JSON.stringify(text)}) &&
      ![...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Stop')`)){
      await delay(700);return;
    }
    await delay(150);
  }
  throw new Error(`Missing ${text}: `+await evalJs('document.body?.innerText.slice(0,1500)'));
}
async function viewport(width,height){
  await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width<800});
  await delay(400);
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
  if(state.root>state.width+2||state.body>state.width+2||state.exception){
    throw new Error(`${name}: ${JSON.stringify(state)} `+await evalJs("document.querySelector('[data-testid=stException]')?.innerText||document.body.innerText.slice(0,1800)"));
  }
  const {data}=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false,fromSurface:true});
  await writeFile(path.join(out,name),Buffer.from(data,'base64'));
  console.log(name,JSON.stringify(state));
}
async function clickButton(label){
  for(let n=0;n<40;n++){
    const clicked=await evalJs(`(()=>{const b=[...document.querySelectorAll('button')]
      .find(e=>e.innerText.trim()===${JSON.stringify(label)}&&!e.disabled);if(!b)return false;b.click();return true;})()`);
    if(clicked){await delay(1000);return;}await delay(150);
  }
  throw new Error(`Enabled button missing: ${label}`);
}
async function clickLabel(label){
  for(let n=0;n<50;n++){
    const ok=await evalJs(`(()=>{const x=[...document.querySelectorAll('label')]
      .find(e=>e.innerText.trim()===${JSON.stringify(label)});if(!x)return false;x.click();return true;})()`);
    if(ok){await delay(1000);return;}await delay(150);
  }
  throw new Error(`Label missing: ${label}`);
}
async function option(value){
  const opened=await evalJs(`(()=>{const c=document.querySelector('[data-testid="stSelectbox"] [role="combobox"]');
    if(!c)return false;c.click();return true;})()`);
  if(!opened)throw new Error('Module selector missing');
  await delay(150);
  const picked=await evalJs(`(()=>{const x=[...document.querySelectorAll('[role="option"]')]
    .find(e=>e.innerText.trim()===${JSON.stringify(value)});if(!x)return false;x.click();return true;})()`);
  if(!picked)throw new Error(`Option missing: ${value}`);
  await delay(650);
}
async function tab(value){
  const ok=await evalJs(`(()=>{const x=[...document.querySelectorAll('[role="tab"]')]
    .find(e=>e.innerText.trim()===${JSON.stringify(value)});if(!x)return false;x.click();return true;})()`);
  if(!ok)throw new Error(`Tab missing: ${value}`);
  await delay(500);
}
async function scrollToText(value){
  await evalJs(`(()=>{const x=[...document.querySelectorAll('h2,h3,h4,p')]
    .find(e=>e.innerText.includes(${JSON.stringify(value)}));x?.scrollIntoView({block:'start'});})()`);
  await delay(300);
}

try{
  await connect();await waitFor('Harris & Shi-Tomasi Corner Detection');
  await option('SIFT Interactive Tutorial');await waitFor('BƯỚC 1/18');await delay(900);
  for(const [w,h,label] of [[1920,1080,'desktop'],[1366,768,'laptop'],[768,1024,'tablet'],[390,844,'mobile']]){
    await viewport(w,h);await shot(`guided-initial-${label}.png`);
  }
  await viewport(1366,768);
  const predictions={3:'Không',5:'Không',6:'Không',7:'Không',10:'Có',14:'Có',16:'Không',17:'Không',18:'Không'};
  const names={2:'dog-candidates.png',3:'subpixel-grid.png',4:'taylor-hessian.png',5:'refinement.png',
    6:'contrast-rejection.png',7:'edge-rejection.png',8:'filter-recap.png',9:'rotation-motivation.png',
    10:'orientation-histogram.png',11:'secondary-orientations.png',12:'aligned-patch.png',
    13:'descriptor-grid.png',14:'descriptor-128d.png',15:'normalization.png',
    16:'nearest-two.png',17:'lowe-ratio.png',18:'full-recap.png'};
  for(let step=1;step<=18;step++){
    if(names[step])await shot(names[step]);
    if(predictions[step]){await clickLabel(predictions[step]);await clickButton('Reveal kết quả');await waitFor('Đúng.');}
    if(step===7){await waitFor('Tr²/Det=');await shot('edge-revealed.png');}
    if(step===17){await waitFor('Ratio test:');await shot('ratio-revealed.png');}
    if(step<18){await clickButton('Next →');await waitFor(`BƯỚC ${step+1}/18`);}
  }
  await clickLabel('Experiment Lab');await waitFor('Lab A · DoG, Taylor và lọc');
  await shot('lab-a.png');await clickLabel('KEEP');await clickButton('Reveal Taylor và quyết định');
  await scrollToText('Hessian DoG 2×2');await shot('lab-a-revealed.png');
  await tab('B · Orientation/Descriptor');await waitFor('Lab B · Hướng và descriptor');
  await shot('lab-b.png');await scrollToText('Chuẩn hóa');await shot('lab-b-vector.png');
  await clickButton('Tính lại ảnh quay');await waitFor('Vị trí affine dự kiến');
  await shot('lab-b-rotation.png');
  await tab('C · Matching');await waitFor('Lab C · Feature matching');
  await clickButton('Tính keypoints và descriptors A/B');await waitFor('Ratio test:');
  await shot('lab-c.png');
  await clickLabel('Presentation Mode');await waitFor('SIFT · Learning Lab');await shot('presentation-laptop.png');
  await viewport(390,844);await shot('presentation-mobile.png');
  await clickLabel('Presentation Mode');await waitFor('SIFT · từ DoG đến descriptor 128D');await shot('mobile-lab.png');
  await clickLabel('Guided Tutorial');await waitFor('BƯỚC 18/18');
  await shot('mobile-guided-end.png');
  await option('Harris & Shi-Tomasi Corner Detection');await waitFor('Vì sao Harris tìm được góc?');
  await shot('harris-regression.png');
  await option('Gaussian Scale Space, LoG & DoG');await waitFor('BƯỚC 1/15');
  await shot('scale-space-regression.png');
  console.log('Browser errors',JSON.stringify(errors));
  if(errors.length)throw new Error('Browser console/JavaScript errors');
}finally{
  try{await call('Browser.close');}catch{child.kill();}
  socket?.close();
  if(path.dirname(path.resolve(profile))!==path.resolve(chapter)||
     !path.basename(profile).startsWith('.sift_audit_profile-'))throw new Error(`Unsafe profile: ${profile}`);
  await rm(profile,{recursive:true,force:true,maxRetries:8,retryDelay:100});
}
