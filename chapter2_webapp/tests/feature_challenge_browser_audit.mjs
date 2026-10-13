// Real local Edge/CDP audit of Chapter 2 :8517. No external site or dependency.
import {spawn} from 'node:child_process';
import {mkdir,mkdtemp,rm,writeFile} from 'node:fs/promises';
import path from 'node:path';

const chapter=path.join(process.cwd(),'chapter2_webapp');
const out=path.join(chapter,'docs','screenshots','feature_challenge');
const profile=await mkdtemp(path.join(chapter,'.feature_challenge_profile-'));
const edge='C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
await mkdir(out,{recursive:true});
const child=spawn(edge,['--remote-debugging-port=9237',`--user-data-dir=${profile}`,
  '--headless=new','--no-first-run','--disable-gpu','--disable-extensions',
  '--window-size=1920,1080','http://127.0.0.1:8517'],
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
    try{pages=await(await fetch('http://127.0.0.1:9237/json/list')).json();if(pages.some(p=>p.type==='page'))break;}
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
  for(let n=0;n<150;n++){
    if(await evalJs(`document.body?.innerText.includes(${JSON.stringify(value)}) &&
       ![...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Stop')`)){
      await delay(250);return;
    }await delay(200);
  }
  throw new Error(`Missing ${value}: `+await evalJs('document.body?.innerText.slice(0,2000)'));
}
async function viewport(width,height){
  await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width<800});
  await delay(300);
  await evalJs(`(()=>{window.scrollTo(0,0);const e=document.querySelector('[data-testid="stAppViewContainer"]');if(e)e.scrollTop=0;})()`);
}
async function shot(name){
  await delay(300);
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
  for(let n=0;n<80;n++){
    const clicked=await evalJs(`(()=>{const b=[...document.querySelectorAll('button')]
      .find(e=>e.innerText.trim()===${JSON.stringify(label)}&&!e.disabled);if(!b)return false;b.click();return true;})()`);
    if(clicked){await delay(450);return;}await delay(200);
  }
  throw new Error(`Enabled button missing: ${label}`);
}
async function clickLabel(label,ordinal=0){
  for(let n=0;n<80;n++){
    const ok=await evalJs(`(()=>{const x=[...document.querySelectorAll('label')]
      .filter(e=>e.innerText.trim()===${JSON.stringify(label)})[${ordinal}];if(!x)return false;x.click();return true;})()`);
    if(ok){await delay(500);return;}await delay(200);
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
  await delay(550);
}
async function selectByLabel(label,value){
  const opened=await evalJs(`(()=>{const box=[...document.querySelectorAll('[data-testid="stSelectbox"]')]
    .find(e=>e.getClientRects().length&&e.innerText.includes(${JSON.stringify(label)}));const c=box?.querySelector('[role="combobox"]');
    if(!c)return false;c.click();return true;})()`);
  if(!opened)throw new Error(`Selectbox label missing: ${label}`);
  await delay(250);
  const picked=await evalJs(`(()=>{const x=[...document.querySelectorAll('[role="option"]')]
    .find(e=>e.innerText.trim()===${JSON.stringify(value)});if(!x)return false;x.click();return true;})()`);
  if(!picked)throw new Error(`Option missing: ${value}`);
  await delay(550);
  for(let n=0;n<40;n++){
    if(await evalJs(`(()=>{const box=[...document.querySelectorAll('[data-testid="stSelectbox"]')]
      .find(e=>e.getClientRects().length&&e.innerText.includes(${JSON.stringify(label)}));return box?.querySelector('[role="combobox"]')?.getAttribute('aria-label')?.includes(${JSON.stringify(value)})})()`))return;
    await delay(150);
  }
  throw new Error(`Selection did not settle: ${label} = ${value}; `+JSON.stringify(await evalJs(
    `[...document.querySelectorAll('[data-testid="stSelectbox"]')].filter(e=>e.getClientRects().length).map(e=>({text:e.innerText.slice(0,150),combo:e.querySelector('[role="combobox"]')?.outerHTML.slice(0,300)}))`)));
}
async function tab(value){
  const ok=await evalJs(`(()=>{const x=[...document.querySelectorAll('[role="tab"]')]
    .find(e=>e.innerText.trim()===${JSON.stringify(value)});if(!x)return false;x.click();return true;})()`);
  if(!ok)throw new Error(`Tab missing: ${value}`);
  await delay(250);
}
async function scrollToPlot(){
  await evalJs(`document.querySelector('[data-testid="stPlotlyChart"]')?.scrollIntoView({block:'start'})`);
  await delay(250);
}

try{
  await connect();await waitFor('Harris & Shi-Tomasi Corner Detection');
  await select('Feature Matching Challenge');await waitFor('Feature Matching Challenge');
  if(process.argv.includes('--lab-b-only')){
    await clickLabel('Experiment Lab');await tab('B · Microscope');
    await waitFor('Lab B · Matching Microscope');
    const opened=await evalJs(`(()=>{const x=[...document.querySelectorAll('summary')]
      .find(e=>e.innerText.includes('Descriptor truy vấn và hai láng giềng'));if(!x)return false;x.click();return true;})()`);
    if(!opened)throw new Error('Neighbor expander missing');
    await waitFor('Láng giềng #2');
    await evalJs(`[...document.querySelectorAll('summary')].find(e=>e.innerText.includes('Descriptor truy vấn và hai láng giềng'))?.scrollIntoView({block:'start'})`);
    await shot('lab-b-neighbors.png');
  }else if(process.argv.includes('--filter-only')){
    await clickLabel('Experiment Lab');await tab('C · Compare');
    await clickButton('Run comparison');await clickButton('Reveal geometry metrics');
    await selectByLabel('Lọc đường match','Đúng/sai hình học');
    await shot('lab-c-classified.png');
  }else{
  for(const [w,h,label] of [[1920,1080,'desktop'],[1366,768,'laptop'],[768,1024,'tablet'],[390,844,'mobile']]){
    await viewport(w,h);await shot(`guided-initial-${label}.png`);
  }
  await viewport(1366,768);
  const names={2:'pipeline.png',3:'prediction.png',4:'transform.png',5:'features.png',
    6:'descriptors.png',7:'distance.png',8:'nearest.png',9:'decision-before.png',
    10:'geometry-before.png',11:'metrics.png',12:'comparison-before.png',
    13:'failure.png',14:'recommendation-before.png'};
  const answers={3:'SIFT',9:'Có',10:'Có',12:'SIFT',13:'Không chắc'};
  for(let step=1;step<=14;step++){
    await waitFor(`${step}/14`);
    if(names[step])await shot(names[step]);
    if(answers[step]){
      await clickLabel(answers[step]);await clickButton('Reveal kết quả');
      await shot(`revealed-${step}.png`);
    }
    if(step===14){
      // AppTest covers typing final justification; CDP verifies the gate and view.
      break;
    }
    await clickButton('Next →');
  }
  await clickLabel('Experiment Lab');await waitFor('Lab A · Image Transformation Studio');
  await shot('lab-a.png');await scrollToPlot();await shot('lab-a-chart.png');
  await selectByLabel('Cảnh thực nghiệm','Hoa văn lặp');await shot('lab-a-repetitive.png');
  const slider=await evalJs(`(()=>{const box=[...document.querySelectorAll('[data-testid="stSlider"]')]
    .find(e=>e.innerText.includes('Góc quay'));const input=box?.querySelector('[role="slider"]');
    if(!input)return false;input.focus();return true;})()`);
  if(!slider)throw new Error('Rotation slider missing in Lab A');
  await call('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});
  await call('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowRight',code:'ArrowRight',windowsVirtualKeyCode:39});
  await delay(600);await shot('lab-a-rotated.png');
  await clickButton('Reset Lab A');
  await tab('B · Microscope');await waitFor('Lab B · Matching Microscope');
  await shot('lab-b-before.png');await clickLabel('Có');await clickButton('Reveal quyết định');
  await shot('lab-b-rule.png');await clickLabel('Có',1);await clickButton('Reveal hình học');
  await shot('lab-b-geometry.png');
  await tab('C · Compare');await waitFor('Lab C · Algorithm Comparison Arena');
  await clickButton('Run comparison');await waitFor('Soi pipeline');await shot('lab-c-before.png');
  if(await evalJs(`[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Save snapshot')`))
    throw new Error('Snapshot leaked before geometry reveal');
  await clickButton('Reveal geometry metrics');await shot('lab-c-revealed.png');
  await selectByLabel('Lọc đường match','Đúng/sai hình học');await shot('lab-c-classified.png');
  await clickButton('Save snapshot');await waitFor('Đã lưu 1 snapshot');await shot('snapshot.png');
  await selectByLabel('Cảnh thực nghiệm','Hoa văn lặp');await waitFor('Cấu hình đã đổi');
  await shot('lab-c-stale-hidden.png');
  await clickLabel('Classroom Challenge');await waitFor('Classroom Challenge · dự đoán trước khi đo');
  await shot('classroom-before.png');await clickLabel('SIFT');await clickButton('Lock prediction');
  await clickButton('Run scenario');await clickButton('Reveal measured quality');
  await shot('classroom-after.png');
  await clickLabel('Presentation Mode');await shot('presentation-laptop.png');
  await scrollToPlot();await shot('presentation-laptop-chart.png');
  await viewport(390,844);await shot('presentation-mobile.png');await scrollToPlot();
  await shot('presentation-mobile-chart.png');
  await clickLabel('Guided Tutorial');await waitFor('14/14');await shot('guided-end-mobile.png');
  await viewport(1366,768);
  for(const [module,name] of [['Harris & Shi-Tomasi Corner Detection','harris-regression.png'],
    ['Gaussian Scale Space, LoG & DoG','scale-regression.png'],
    ['SIFT Interactive Tutorial','sift-regression.png'],
    ['FAST, BRIEF & ORB','orb-regression.png']]){
    await select(module);await shot(name);
  }
  }
  console.log('Browser errors',JSON.stringify(errors));
  if(errors.length)throw new Error('Browser console/JavaScript errors');
}finally{
  try{await call('Browser.close');}catch{child.kill();}
  socket?.close();
  if(path.dirname(path.resolve(profile))!==path.resolve(chapter)||
     !path.basename(profile).startsWith('.feature_challenge_profile-'))throw new Error(`Unsafe profile: ${profile}`);
  await rm(profile,{recursive:true,force:true,maxRetries:8,retryDelay:100});
}
