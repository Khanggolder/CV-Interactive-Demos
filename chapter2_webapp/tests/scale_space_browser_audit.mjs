// Real Edge/CDP smoke audit. Run with Chapter 2 server on 127.0.0.1:8514.
import { spawn } from 'node:child_process';
import { writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const chapter = path.join(root, 'chapter2_webapp');
const out = process.env.CHAPTER2_AUDIT_OUT || path.join(chapter, '.audit_output', 'scale_space');
const profile = await mkdtemp(path.join(chapter, '.scale_space_audit_profile-'));
const edge = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
await mkdir(out, { recursive: true });
const child = spawn(edge, ['--remote-debugging-port=9228', `--user-data-dir=${profile}`,
  '--headless=new', '--no-first-run', '--disable-gpu', '--disable-extensions',
  '--window-size=1920,1080', 'http://127.0.0.1:8514'],
  { stdio: ['ignore','ignore','pipe'], windowsHide: true });
let edgeStderr = '';
child.stderr.on('data', part => {edgeStderr += part.toString();});
let socket, serial = 1;
const pending = new Map(), errors = [];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));
async function call(method, params={}) {
  const id = serial++;
  return new Promise((resolve,reject) => {
    pending.set(id,{resolve,reject});
    socket.send(JSON.stringify({id,method,params}));
  });
}
async function connect() {
  let pages;
  for (let n=0;n<70;n++) {
    try { pages = await (await fetch('http://127.0.0.1:9228/json/list')).json();
      if (pages.some(p => p.type==='page')) break;
    } catch { /* starting */ }
    await delay(200);
  }
  const page = pages?.find(p => p.type==='page');
  if (!page) throw new Error(`Edge did not expose a page; exit=${child.exitCode}; ${edgeStderr.slice(0,2500)}`);
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve,reject) => {socket.onopen=resolve; socket.onerror=reject;});
  socket.onmessage = event => {
    const m = JSON.parse(event.data);
    if (m.method==='Runtime.exceptionThrown') errors.push(m.params.exceptionDetails.text);
    if (m.id && pending.has(m.id)) {
      const {resolve,reject}=pending.get(m.id); pending.delete(m.id);
      m.error ? reject(new Error(JSON.stringify(m.error))) : resolve(m.result);
    }
  };
  await call('Runtime.enable'); await call('Page.enable');
}
async function evaluate(expression) {
  const result = await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if (result.exceptionDetails) throw new Error(JSON.stringify(result.exceptionDetails));
  return result.result.value;
}
async function waitFor(text) {
  for (let n=0;n<90;n++) {
    if (await evaluate(`document.body?.innerText.includes(${JSON.stringify(text)})`)) return;
    await delay(200);
  }
  const excerpt=await evaluate(`document.body?.innerText.slice(0,1800)`);
  throw new Error(`Not visible: ${text}; body=${excerpt}`);
}
async function viewport(width,height) {
  await call('Emulation.setDeviceMetricsOverride',
             {width,height,deviceScaleFactor:1,mobile:width<800});
  await delay(500);
  await evaluate(`(() => {document.activeElement?.blur();window.scrollTo(0,0);
    for(const e of [document.scrollingElement,...document.querySelectorAll(
      '[data-testid="stAppViewContainer"], [data-testid="stMain"], section.main')])
      if(e)e.scrollTop=0;})()`);
}
async function shot(name) {
  for(let n=0;n<70;n++) {
    const busy=await evaluate(`[...document.querySelectorAll('button')]
      .some(b=>b.innerText.trim()==='Stop')`);
    if(!busy) break;
    await delay(200);
  }
  await delay(450);
  const state=await evaluate(`({width:innerWidth,scroll:document.documentElement.scrollWidth,
    body:document.body.scrollWidth,exception:!!document.querySelector('[data-testid="stException"]')})`);
  if (state.scroll>state.width+2 || state.body>state.width+2 || state.exception) {
    const detail=await evaluate(`document.querySelector('[data-testid="stException"]')?.innerText || document.body.innerText.slice(0,2000)`);
    throw new Error(`${name}: ${JSON.stringify(state)}; ${detail}`);
  }
  const {data}=await call('Page.captureScreenshot',
                          {format:'png',captureBeyondViewport:false,fromSurface:true});
  await writeFile(path.join(out,name),Buffer.from(data,'base64'));
  console.log(name,JSON.stringify(state));
}
async function clickButton(label) {
  for (let n=0;n<25;n++) {
    const ok=await evaluate(`(() => {let b=[...document.querySelectorAll('button')]
      .find(e=>e.innerText.trim()===${JSON.stringify(label)} && !e.disabled);
      if(!b)return false; b.click(); return true;})()`);
    if (ok) {await delay(550);return;}
    await delay(200);
  }
  throw new Error(`Enabled button not found: ${label}`);
}
async function clickLabel(label) {
  const ok=await evaluate(`(() => {let e=[...document.querySelectorAll('label')]
    .find(e=>e.innerText.trim()===${JSON.stringify(label)});if(!e)return false;
    e.click();return true;})()`);
  if(!ok) throw new Error(`Label not found: ${label}`);
  await delay(500);
}
async function selectOption(label,value) {
  const opened=await evaluate(`(() => {let e=[...document.querySelectorAll('[data-testid="stSelectbox"]')]
    .find(e=>e.getClientRects().length && (${JSON.stringify(label)}==='Bài học Chapter 2' || e.innerText.includes(${JSON.stringify(label)})));
    let c=e?.querySelector('[role="combobox"]');
    if(!c)return false;c.click();return true;})()`);
  if(!opened) throw new Error(`Selectbox not found: ${label}`);
  await delay(200);
  const picked=await evaluate(`(() => {let e=[...document.querySelectorAll('[role="option"]')]
    .find(e=>e.innerText.trim()===${JSON.stringify(value)});if(!e)return false;e.click();return true;})()`);
  if(!picked) throw new Error(`Option not found: ${value}`);
  await delay(650);
}
async function clickTab(label) {
  const ok=await evaluate(`(() => {let e=[...document.querySelectorAll('[role="tab"]')]
    .find(e=>e.innerText.trim()===${JSON.stringify(label)});if(!e)return false;e.click();return true;})()`);
  if(!ok) throw new Error(`Tab not found: ${label}`);
  await delay(600);
}
async function clickSummary(label) {
  const ok=await evaluate(`(() => {let e=[...document.querySelectorAll('summary')]
    .find(e=>e.innerText.includes(${JSON.stringify(label)}));if(!e)return false;e.click();return true;})()`);
  if(!ok) throw new Error(`Expander not found: ${label}`);
  await delay(450);
}
async function moveSlider(label) {
  const ok=await evaluate(`(() => {let e=[...document.querySelectorAll('[data-testid="stSlider"]')]
    .find(e=>e.innerText.includes(${JSON.stringify(label)}));let s=e?.querySelector('[role="slider"]');
    if(!s)return false;s.focus();return true;})()`);
  if(!ok) throw new Error(`Slider not found: ${label}`);
  await call('Input.dispatchKeyEvent',{type:'keyDown',key:'ArrowRight',code:'ArrowRight',
                                      windowsVirtualKeyCode:39});
  await call('Input.dispatchKeyEvent',{type:'keyUp',key:'ArrowRight',code:'ArrowRight',
                                      windowsVirtualKeyCode:39});
  await delay(550);
}
async function scrollShot(name,amount=650) {
  await evaluate(`(() => {window.scrollTo(0,${amount});let e=document.querySelector('[data-testid="stAppViewContainer"]');
    if(e)e.scrollTop=${amount};})()`);
  await delay(350); await shot(name);
  await evaluate(`(() => {window.scrollTo(0,0);let e=document.querySelector('[data-testid="stAppViewContainer"]');
    if(e)e.scrollTop=0;})()`);
}

try {
  await connect();
  await waitFor('Harris & Shi-Tomasi Corner Detection');
  await selectOption('Bài học Chapter 2','Gaussian Scale Space, LoG & DoG');
  await waitFor('BƯỚC 1/15');
  if (process.env.SCALE_MOBILE_ONLY === '1') {
    await viewport(390,844); await shot('guided-initial-mobile.png');
  } else {
  if (process.env.SCALE_SPACE_QUICK === '1') await viewport(1366,768);
  if (process.env.SCALE_SPACE_QUICK !== '1') {
  for (const [w,h,label] of [[1920,1080,'desktop'],[1366,768,'laptop'],
                              [768,1024,'tablet'],[390,844,'mobile']]) {
    await viewport(w,h); await shot(`guided-initial-${label}.png`);
  }
  await viewport(1366,768);
  const predictions={1:'Không',3:'Nhỏ',7:'Blob nhỏ',8:'Âm',9:'Không',11:'Không'};
  const captures={2:'gaussian-scale.png',3:'sigma-explorer.png',4:'semigroup.png',
                  5:'gaussian-pyramid.png',6:'octave-sigma.png',7:'blob-scale.png',
                  8:'log-signed.png',9:'log-normalized.png',10:'dog-subtraction.png',
                  11:'log-dog.png',12:'dog-pyramid.png',13:'extrema-gate.png',
                  14:'scale-robustness.png'};
  for(let current=1;current<15;current++) {
    if(predictions[current]) {await clickLabel(predictions[current]);await clickButton('Reveal kết quả');}
    if(current===13) {
      await clickLabel('NOT EXTREMUM'); await clickButton('Bắt đầu');
      await clickButton('So sánh →'); await scrollShot('extrema-comparison.png');
      await clickButton('Chạy còn lại'); await clickButton('Reveal quyết định');
      await scrollShot('extrema-revealed.png');
      await viewport(390,844); await scrollShot('extrema-mobile.png',700);
      await viewport(1366,768);
    }
    await clickButton('Next →'); await waitFor(`BƯỚC ${current+1}/15`);
    if(captures[current+1]) await shot(captures[current+1]);
    if(current+1===3) await moveSlider('σ · pixel');
  }
  await clickLabel('Không'); await clickButton('Reveal kết quả');
  await shot('recap.png');
  }
  await clickLabel('Experiment Lab'); await waitFor('Lab A · Gaussian Scale Space');
  await shot('lab-a.png');
  if (process.env.SCALE_PRESENTATION_ONLY !== '1') {
  await clickTab('Lab B · LoG/DoG'); await waitFor('Lab B · LoG, DoG & Extrema');
  await waitFor('Raw strict 26-neighbor:');
  await shot('lab-b.png');
  await clickSummary('Đối chiếu ảnh gốc và resize');
  await evaluate(`([...document.querySelectorAll('summary')]
    .find(e=>e.innerText.includes('Đối chiếu ảnh gốc và resize')))?.scrollIntoView({block:'start'})`);
  await shot('lab-scale-compare.png');
  await clickSummary('Đối chiếu ảnh gốc và resize');
  await clickSummary('Điều khiển · preset, octave, s, contrast');
  await selectOption('Preset','Blob tối');
  await moveSlider('s · khoảng/octave');
  await shot('lab-changed.png');
  }
  await clickLabel('Presentation Mode');
  await waitFor('Điều khiển thí nghiệm');
  await shot('presentation-laptop.png');
  await viewport(390,844); await shot('presentation-mobile.png');
  await clickLabel('Presentation Mode'); await shot('lab-mobile.png');
  await selectOption('Bài học Chapter 2','Harris & Shi-Tomasi Corner Detection');
  await waitFor('Vì sao Harris tìm được góc?'); await shot('harris-regression.png');
  }
  console.log('Browser errors:',JSON.stringify(errors));
  if(errors.length) throw new Error('Browser JavaScript exceptions');
} finally {
  try {await call('Browser.close');} catch {child.kill();}
  socket?.close();
  if(path.dirname(path.resolve(profile))!==path.resolve(chapter) ||
     !path.basename(profile).startsWith('.scale_space_audit_profile-'))
    throw new Error(`Unexpected profile: ${profile}`);
  await rm(profile,{recursive:true,force:true,maxRetries:8,retryDelay:100});
}
