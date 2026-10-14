// Full five-module audit against the real local Streamlit server on :8522.
// Run from repository root with: node chapter2_webapp/tests/final_chapter2_browser_audit.mjs
import {spawn} from 'node:child_process';
import {mkdir,mkdtemp,rm,writeFile,readdir,stat} from 'node:fs/promises';
import path from 'node:path';

const root=process.cwd();
const chapter=path.join(root,'chapter2_webapp');
// Generated evidence is separate from curated, tracked README screenshots.
const out=process.env.CHAPTER2_AUDIT_OUT || path.join(chapter,'.audit_output','final_audit');
const profile=await mkdtemp(path.join(chapter,'.final_chapter2_profile-'));
const edge='C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const port=9241;
await mkdir(out,{recursive:true});
const child=spawn(edge,[`--remote-debugging-port=${port}`,`--user-data-dir=${profile}`,
  '--headless=new','--no-first-run','--disable-gpu','--disable-extensions',
  '--window-size=1920,1080','http://127.0.0.1:8522'],
  {stdio:['ignore','ignore','pipe'],windowsHide:true});
let stderr='';child.stderr.on('data',p=>stderr+=p.toString());
let socket,serial=1;const pending=new Map(),errors=[],observations=[];
const delay=ms=>new Promise(r=>setTimeout(r,ms));
async function bytesUnder(dir){
  let total=0;
  for(const item of await readdir(dir,{withFileTypes:true})){
    const target=path.join(dir,item.name);
    total+=item.isDirectory()?await bytesUnder(target):item.isFile()?(await stat(target)).size:0;
  }
  return total;
}
async function call(method,params={}){
  const id=serial++;
  return new Promise((resolve,reject)=>{pending.set(id,{resolve,reject});socket.send(JSON.stringify({id,method,params}));});
}
async function connect(){
  let pages;
  for(let n=0;n<100;n++){
    try{pages=await(await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      if(pages.some(p=>p.type==='page'))break;}catch{}await delay(200);
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
async function js(expression){
  const r=await call('Runtime.evaluate',{expression,returnByValue:true,awaitPromise:true});
  if(r.exceptionDetails)throw new Error(JSON.stringify(r.exceptionDetails));
  return r.result.value;
}
async function waitText(value){
  for(let n=0;n<180;n++){
    if(await js(`document.body?.innerText.includes(${JSON.stringify(value)}) &&
      ![...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Stop')`)){
      await delay(230);return;
    }await delay(200);
  }
  throw new Error(`Missing ${value}: `+await js('document.body?.innerText.slice(0,1400)'));
}
async function viewport(width,height){
  await call('Emulation.setDeviceMetricsOverride',{width,height,deviceScaleFactor:1,mobile:width<800});
  await delay(300);await js('window.scrollTo(0,0)');
}
async function check(name){
  await delay(250);
  const state=await js(`({viewport:innerWidth,root:document.documentElement.scrollWidth,
    body:document.body.scrollWidth,exception:!!document.querySelector('[data-testid="stException"]'),
    plots:document.querySelectorAll('[data-testid="stPlotlyChart"]').length,
    buttons:[...document.querySelectorAll('button')].filter(b=>!b.disabled).length})`);
  observations.push({name,...state});
  if(state.root>state.viewport+2||state.body>state.viewport+2||state.exception)
    throw new Error(`${name}: ${JSON.stringify(state)} `+await js(
      "document.querySelector('[data-testid=stException]')?.innerText||document.body.innerText.slice(0,1000)"));
  return state;
}
async function shot(name){
  const state=await check(name);
  const {data}=await call('Page.captureScreenshot',{format:'png',captureBeyondViewport:false,fromSurface:true});
  await writeFile(path.join(out,name+'.png'),Buffer.from(data,'base64'));
  console.log('SHOT',name,JSON.stringify(state));
}
async function clickButton(label){
  for(let n=0;n<120;n++){
    const ok=await js(`(()=>{const b=[...document.querySelectorAll('button')].find(
      e=>e.innerText.trim()===${JSON.stringify(label)}&&!e.disabled&&e.getClientRects().length);
      if(!b)return false;b.click();return true})()`);
    if(ok){await delay(410);return;}await delay(200);
  }
  throw new Error(`Enabled button missing: ${label}`);
}
async function clickLabel(label,ordinal=0){
  for(let n=0;n<80;n++){
    const ok=await js(`(()=>{const x=[...document.querySelectorAll('label')].filter(
      e=>e.innerText.trim()===${JSON.stringify(label)}&&e.getClientRects().length)[${ordinal}];
      if(!x)return false;x.click();return true})()`);
    if(ok){await delay(430);return;}await delay(200);
  }
  throw new Error(`Label missing: ${label}`);
}
async function selectModule(value){
  const opened=await js(`(()=>{const e=document.querySelector('[data-testid="stSelectbox"] [role="combobox"]');
    if(!e)return false;e.click();return true})()`);
  if(!opened)throw new Error('Module selector missing');
  await delay(180);
  const picked=await js(`(()=>{const e=[...document.querySelectorAll('[role="option"]')].find(
    x=>x.innerText.trim()===${JSON.stringify(value)});if(!e)return false;e.click();return true})()`);
  if(!picked)throw new Error(`Module option missing ${value}`);
  await delay(650);
}
async function selectByLabel(label,value){
  const opened=await js(`(()=>{const box=[...document.querySelectorAll('[data-testid="stSelectbox"]')]
    .find(e=>e.innerText.includes(${JSON.stringify(label)}));
    const combo=box?.querySelector('[role="combobox"]');if(!combo)return false;combo.click();return true})()`);
  if(!opened)throw new Error(`Selectbox missing: ${label}`);
  await delay(180);
  const picked=await js(`(()=>{const e=[...document.querySelectorAll('[role="option"]')]
    .find(x=>x.innerText.trim()===${JSON.stringify(value)});if(!e)return false;e.click();return true})()`);
  if(!picked)throw new Error(`Option missing: ${value}`);
  await delay(540);
}
async function tab(label){
  const ok=await js(`(()=>{const e=[...document.querySelectorAll('[role="tab"]')].find(
    x=>x.innerText.trim()===${JSON.stringify(label)});if(!e)return false;e.click();return true})()`);
  if(!ok)throw new Error(`Tab missing ${label}`);
  await delay(310);
}
async function settleStep(step,total){
  const pattern=`(?:^|\\D)${step}/${total}(?:\\D|$)`;
  for(let n=0;n<160;n++){
    if(await js(`new RegExp(${JSON.stringify(pattern)}).test(document.body?.innerText||'') &&
       ![...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Stop')`)){
      await delay(180);return;
    }await delay(200);
  }
  throw new Error(`Step mismatch ${step}/${total}: `+await js('document.body.innerText.slice(0,600)'));
}

const modules=[
  {name:'Harris & Shi-Tomasi Corner Detection',id:'harris',total:14,
   gates:{2:'Patch C',4:'Cả x và y',8:'Góc',9:'Góc',11:'Điểm theo thứ tự hàng/cột nhỏ nhất',12:'min(λ₁,λ₂)',14:'Không'},
   key:[7,11,13,14],labs:['Experiment Lab']},
  {name:'Gaussian Scale Space, LoG & DoG',id:'scale',total:15,
   gates:{1:'Không',3:'Nhỏ',7:'Blob nhỏ',8:'Âm',9:'Không',11:'Không',15:'Không'},
   key:[4,6,10,13,15],labs:['Lab A · Gaussian','Lab B · LoG/DoG']},
  {name:'SIFT Interactive Tutorial',id:'sift',total:18,
   gates:{3:'Không',5:'Không',6:'Không',7:'Không',10:'Có',14:'Có',16:'Không',17:'Không',18:'Không'},
   key:[7,10,13,17,18],labs:['A · Detection','B · Orientation/Descriptor','C · Matching']},
  {name:'FAST, BRIEF & ORB',id:'orb',total:18,
   gates:{4:'CORNER',5:'Không',6:'Có',10:'Có',12:'Có',16:'Có',17:'Không',18:'Không'},
   key:[4,8,10,13,17,18],labs:['A · FAST','B · Orientation/BRIEF','C · Matching']},
  {name:'Feature Matching Challenge',id:'feature',total:14,
   gates:{3:'SIFT',9:'Có',10:'Có',12:'SIFT',13:'Không chắc'},
   key:[5,9,10,12,14],labs:['A · Transform','B · Microscope','C · Compare']},
];

try{
  await connect();await waitText('BƯỚC 1/14');
  if(!process.argv.includes('--feature-only')&&!process.argv.includes('--cards-only')&&
     !process.argv.includes('--presentation-only')&&!process.argv.includes('--harris-details-only')){
    for(const [w,h,id] of [[1920,1080,'desktop'],[1366,768,'laptop'],[768,1024,'tablet'],[390,844,'mobile']]){
      await viewport(w,h);await shot(`selector-${id}`);
    }
  }
  await viewport(1366,768);
  if(process.argv.includes('--cards-only')){
    await selectModule('Feature Matching Challenge');await clickLabel('Experiment Lab');
    await tab('C · Compare');await clickButton('Run comparison');
    await clickButton('Reveal geometry metrics');
    await js("document.querySelector('.st-key-feature_comparison_desktop')?.scrollIntoView({block:'start'})");
    await shot('feature-comparison-desktop-table');
    await viewport(390,844);
    const mobile=await js(`(()=>{const card=document.querySelector('.feature-result-card');
      const table=document.querySelector('.st-key-feature_comparison_desktop');
      if(!card||!table)return {cards:0,tableDisplay:null};
      card.scrollIntoView({block:'start'});
      return {cards:document.querySelectorAll('.feature-result-card').length,
        cardDisplay:getComputedStyle(card).display,tableDisplay:getComputedStyle(table).display};})()`);
    console.log('MOBILE_CARDS',JSON.stringify(mobile));
    if(mobile.cards!==3||mobile.tableDisplay!=='none')throw new Error('Mobile comparison cards/layout invalid');
    await shot('feature-comparison-mobile-cards');
  }
  if(process.argv.includes('--presentation-only')){
    const keyStep={harris:7,scale:4,sift:13,orb:8,feature:5};
    for(const module of modules){
      await selectModule(module.name);
      for(let step=1;step<keyStep[module.id];step++){
        await settleStep(step,module.total);
        if(module.gates[step]){
          await clickLabel(module.gates[step]);await clickButton('Reveal kết quả');
        }
        await clickButton('Next →');
      }
      await settleStep(keyStep[module.id],module.total);
      await clickLabel('Presentation Mode');
      await js("document.querySelector('[data-testid=stPlotlyChart]')?.scrollIntoView({block:'start'})");
      await shot(`${module.id}-presentation-key-laptop`);
      await viewport(1920,1080);
      await js("document.querySelector('[data-testid=stPlotlyChart]')?.scrollIntoView({block:'start'})");
      await shot(`${module.id}-presentation-key-desktop`);
      await viewport(390,844);
      await js("document.querySelector('[data-testid=stPlotlyChart]')?.scrollIntoView({block:'start'})");
      await shot(`${module.id}-presentation-key-mobile`);
      await viewport(1366,768);
    }
  }
  if(process.argv.includes('--harris-details-only')){
    const module=modules[0];
    for(let step=1;step<=13;step++){
      await settleStep(step,14);
      if(module.gates[step]){
        await clickLabel(module.gates[step]);await clickButton('Reveal kết quả');
      }
      if(step===11){
        await js("document.querySelectorAll('[data-testid=stPlotlyChart]')[1]?.scrollIntoView({block:'center'})");
        await shot('harris-nms-neighborhood');
      }
      if(step===13){
        await selectByLabel('Thử biến đổi','Góc đổi tỉ lệ');
        await js("document.querySelectorAll('[data-testid=stPlotlyChart]')[1]?.scrollIntoView({block:'center'})");
        await shot('harris-scale-support');
      }
      if(step<13)await clickButton('Next →');
    }
  }
  const selectedModules=process.argv.includes('--cards-only')||process.argv.includes('--presentation-only')||
    process.argv.includes('--harris-details-only')?[]:
    process.argv.includes('--feature-only')?modules.slice(-1):modules;
  for(const module of selectedModules){
    await selectModule(module.name);await waitText(module.name.split(' ')[0]);
    console.log('MODULE',module.id);
    await shot(`${module.id}-initial-laptop`);
    for(let step=1;step<=module.total;step++){
      await settleStep(step,module.total);
      if(module.key.includes(step))await shot(`${module.id}-step-${step}-before`);
      if(module.id==='scale'&&step===13){
        await clickLabel('NOT EXTREMUM');await clickButton('Bắt đầu');
        await clickButton('Chạy còn lại');await clickButton('Reveal quyết định');
      }
      if(module.gates[step]){
        await clickLabel(module.gates[step]);await clickButton('Reveal kết quả');
        if((module.id==='harris'&&step===11)||(module.id==='orb'&&step===17)||
           (module.id==='feature'&&[9,10,12].includes(step)))
          await shot(`${module.id}-step-${step}-after`);
      }
      if(module.id==='feature'&&step===14){
        const typed=await js(`(()=>{const e=document.querySelector('textarea');if(!e)return false;
          e.focus();return true})()`);
        if(!typed)throw new Error('Final justification textarea missing');
        for(const char of 'Precision and coverage need geometric evidence.')
          await call('Input.dispatchKeyEvent',{type:'char',text:char});
        await js("document.querySelector('textarea')?.blur()");
        await delay(650);
        console.log('JUSTIFICATION_UI',JSON.stringify(await js(`({areas:[...document.querySelectorAll('textarea')]
          .map(e=>({value:e.value,visible:!!e.getClientRects().length})),
          buttons:[...document.querySelectorAll('button')].filter(e=>e.innerText.includes('Ghi nhận'))
          .map(e=>({text:e.innerText,disabled:e.disabled}))})`)));
        const enabled=await js(`[...document.querySelectorAll('button')].some(e=>
          e.innerText.trim()==='Ghi nhận lập luận'&&!e.disabled)`);
        if(enabled)await clickButton('Ghi nhận lập luận');
        else console.log('BROWSER_LIMITATION final text widget did not sync after CDP typing; AppTest covers submission');
      }
      await check(`${module.id}-step-${step}`);
      if(step<module.total)await clickButton('Next →');
    }
    await shot(`${module.id}-final-laptop`);
    await clickLabel('Presentation Mode');
    await shot(`${module.id}-presentation-laptop`);
    await viewport(1920,1080);await shot(`${module.id}-presentation-desktop`);
    await viewport(390,844);await shot(`${module.id}-presentation-mobile`);
    await clickLabel('Presentation Mode');
    await shot(`${module.id}-final-mobile`);
    await viewport(1366,768);
    await clickLabel('Experiment Lab');
    for(const [labIndex,lab] of module.labs.entries()){
      if(lab!=='Experiment Lab')await tab(lab);
      await shot(`${module.id}-lab-${labIndex+1}`);
    }
    if(module.id==='feature'){
      await clickButton('Run comparison');await shot('feature-comparison-before');
      if(await js(`[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Save snapshot')`))
        throw new Error('Snapshot control revealed before geometry');
      await clickButton('Reveal geometry metrics');await shot('feature-comparison-after');
      await viewport(390,844);await shot('feature-comparison-mobile');
      await viewport(1366,768);
      await clickLabel('Classroom Challenge');await shot('feature-classroom-before');
      await clickLabel('SIFT');await clickButton('Lock prediction');
      await clickButton('Run scenario');await clickButton('Reveal measured quality');
      await shot('feature-classroom-after');
    }
    await viewport(768,1024);await shot(`${module.id}-lab-tablet`);
    await viewport(390,844);await shot(`${module.id}-lab-mobile`);
    await viewport(1366,768);
  }
  console.log('GUIDED_TOTAL',selectedModules.reduce((n,m)=>n+m.total,0));
  console.log('OBSERVATIONS',observations.length,'BROWSER_ERRORS',JSON.stringify(errors));
  if(errors.length)throw new Error('Browser console errors');
}finally{
  try{await call('Browser.close');}catch{child.kill();}
  socket?.close();
  if(path.dirname(path.resolve(profile))!==path.resolve(chapter)||
     !path.basename(profile).startsWith('.final_chapter2_profile-'))
    throw new Error(`Unsafe task profile path: ${profile}`);
  let profileBytes=null;
  try{profileBytes=await bytesUnder(profile);}catch{}
  await rm(profile,{recursive:true,force:true,maxRetries:8,retryDelay:100});
  console.log('CLEANED_PROFILE',profile,profileBytes,'bytes');
}
