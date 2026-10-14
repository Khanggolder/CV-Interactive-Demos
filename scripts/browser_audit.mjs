// Local, dependency-free Edge/CDP audit. Run from repository root:
// node scripts/browser_audit.mjs chapter1|chapter3
import { spawn } from 'node:child_process';
import { mkdir, mkdtemp, rm, writeFile } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import path from 'node:path';

const chapter = process.argv[2];
if (!['chapter1', 'chapter3'].includes(chapter)) {
  throw new Error('Expected chapter1 or chapter3');
}
const root = process.cwd();
const port = chapter === 'chapter1' ? 8531 : 8533;
const cdpPort = chapter === 'chapter1' ? 9251 : 9253;
const output = process.env.CV_AUDIT_OUT || path.join(root, '.audit_output', chapter);
const profile = await mkdtemp(path.join(tmpdir(), `cv-${chapter}-edge-`));
const edge = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
let browser;
try {
  browser = spawn(edge, [
    `--remote-debugging-port=${cdpPort}`, `--user-data-dir=${profile}`,
    '--headless=new', '--no-first-run', '--disable-gpu', '--disable-extensions',
    '--window-size=1920,1080', `http://127.0.0.1:${port}`,
  ], { stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true });
} catch (error) {
  await rm(profile, { recursive: true, force: true });
  throw error;
}
let browserErrors = '';
browser.stderr.on('data', chunk => { browserErrors += chunk.toString(); });
browser.on('error', error => { browserErrors += error.message; });
let socket;
let nextId = 1;
const pending = new Map();
const errors = [];
const observations = [];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

function call(method, params = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function connect() {
  let pages;
  for (let attempt = 0; attempt < 100; attempt++) {
    try {
      pages = await (await fetch(`http://127.0.0.1:${cdpPort}/json/list`)).json();
      if (pages.some(page => page.type === 'page')) break;
    } catch { /* Edge is still starting. */ }
    await delay(200);
  }
  const page = pages?.find(item => item.type === 'page');
  if (!page) throw new Error(`Edge failed to start: ${browserErrors.slice(0, 1000)}`);
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => { socket.onopen = resolve; socket.onerror = reject; });
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.exceptionThrown') errors.push(message.params.exceptionDetails.text);
    if (message.method === 'Log.entryAdded' && message.params.entry.level === 'error') {
      errors.push(message.params.entry.text);
    }
    if (pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
    }
  };
  await call('Runtime.enable');
  await call('Page.enable');
  await call('Log.enable');
}

async function js(expression) {
  const response = await call('Runtime.evaluate', {
    expression, returnByValue: true, awaitPromise: true,
  });
  if (response.exceptionDetails) throw new Error(response.exceptionDetails.text);
  return response.result.value;
}

async function ready() {
  let stable = 0;
  for (let attempt = 0; attempt < 150; attempt++) {
    const state = await js(`({ready:document.querySelector('[data-testid="stApp"]') !== null &&
      document.querySelector('h1,h2,h3') !== null,
      busy:[...document.querySelectorAll('button')].some(b=>b.innerText.trim()==='Stop')})`);
    stable = state.ready && !state.busy ? stable + 1 : 0;
    if (stable >= 3) return;
    await delay(200);
  }
  throw new Error('Streamlit did not become ready');
}

async function initialReady() {
  await ready();
  for (let attempt = 0; attempt < 100; attempt++) {
    const state = await js(`({sidebar:[...document.querySelectorAll(
      '[data-testid="stSidebar"] label')].length,
      images:document.querySelectorAll('img').length,
      plots:document.querySelectorAll('[data-testid="stPlotlyChart"]').length})`);
    if (state.sidebar >= 8 && (chapter === 'chapter1' ? state.images >= 2 : state.plots >= 1)) {
      return;
    }
    await delay(200);
  }
  throw new Error('Initial teaching visual did not load');
}

async function viewport(width, height) {
  await call('Emulation.setDeviceMetricsOverride', {
    width, height, deviceScaleFactor: 1, mobile: width < 800,
  });
  await delay(300);
}

async function inspect(name, screenshot = false) {
  await ready();
  const state = await js(`({width:innerWidth,scrollWidth:document.documentElement.scrollWidth,
    exception:document.querySelector('[data-testid="stException"]')?.innerText||'',
    headings:[...document.querySelectorAll('h1,h2,h3')].map(x=>x.innerText).slice(0,8),
    plots:document.querySelectorAll('[data-testid="stPlotlyChart"]').length,
    images:document.querySelectorAll('img').length,
    buttons:[...document.querySelectorAll('button')].filter(x=>x.getClientRects().length)
      .map(x=>({label:x.innerText.trim(),disabled:x.disabled})).filter(x=>x.label).slice(0,25),
    sidebarLabels:[...document.querySelectorAll('[data-testid="stSidebar"] label')]
      .map(x=>x.innerText.trim()).filter(Boolean).slice(0,30),
    body:document.body.innerText.slice(0,500)})`);
  observations.push({ name, ...state });
  if (state.exception || state.scrollWidth > state.width + 2) {
    throw new Error(`${name}: exception/overflow ${JSON.stringify(state)}`);
  }
  if (screenshot) {
    await mkdir(output, { recursive: true });
    const { data } = await call('Page.captureScreenshot', {
      format: 'png', captureBeyondViewport: false, fromSurface: true,
    });
    await writeFile(path.join(output, `${name}.png`), Buffer.from(data, 'base64'));
  }
  console.log('OBS', JSON.stringify({ name, width: state.width,
    exception: state.exception, plots: state.plots, images: state.images,
    headings: state.headings.slice(-4), buttons: state.buttons.slice(-12) }));
}

async function clickLabel(label) {
  const clicked = await js(`(()=>{const item=[...document.querySelectorAll('label')]
    .find(x=>x.innerText.trim()===${JSON.stringify(label)}&&x.getClientRects().length);
    if(!item)return false;item.click();return true})()`);
  if (!clicked) throw new Error(`Visible label not found: ${label}`);
  await delay(350);
  await ready();
  await js(`window.scrollTo(0,0);document.querySelector('[data-testid="stMain"]')?.scrollTo(0,0)`);
}

async function clickButton(label) {
  const clicked = await js(`(()=>{const item=[...document.querySelectorAll('button')]
    .find(x=>x.innerText.trim()===${JSON.stringify(label)}&&!x.disabled&&x.getClientRects().length);
    if(!item)return false;item.click();return true})()`);
  if (!clicked) throw new Error(`Enabled button not found: ${label}`);
  await delay(350);
  await ready();
}

async function chapter1Pages() {
  const pages = [
    ['Pixel Explorer', '1 · Pixel Explorer'],
    ['Brightness & Contrast', '2 · Brightness'],
    ['Gamma', '3 · Gamma'],
    ['Histogram', '4 · Histogram'],
    ['Noise & Filtering', '5 · Noise'],
    ['Geometric Transform', '6 · Geometric'],
    ['Fourier', '7 · Fourier'],
    ['Convolution Kernels', '8 · Convolution'],
  ];
  for (const [label, expected] of pages) {
    await clickLabel(label);
    const title = await js('document.body.innerText');
    if (!title.includes(expected)) throw new Error(`${label} did not render: ${title}`);
    await inspect(`chapter1-${label.toLowerCase().replaceAll(/[^a-z0-9]+/g, '-')}`,
      ['Gamma', 'Fourier', 'Geometric Transform'].includes(label));
    if (label === 'Pixel Explorer') {
      await clickLabel('Grayscale');
      await inspect('chapter1-pixel-grayscale');
    } else if (label === 'Brightness & Contrast') {
      await clickButton('↺ Reset Parameters');
      await inspect('chapter1-brightness-reset');
    } else if (label === 'Gamma') {
      await clickButton('γ = 2.0');
      await clickLabel('RGB từng kênh');
      const note = await js('document.body.innerText');
      if (!note.includes('độc lập lên từng kênh')) throw new Error('Gamma RGB note missing');
      await inspect('chapter1-gamma-rgb', true);
    } else if (label === 'Histogram') {
      await clickLabel('CLAHE');
      await inspect('chapter1-histogram-clahe');
    } else if (label === 'Noise & Filtering') {
      await clickLabel('Salt-and-pepper noise');
      await clickLabel('Đã dự đoán — hiển thị kết quả lọc');
      if (!(await js('document.body.innerText')).includes('FILTERED OUTPUT')) {
        throw new Error('Noise reveal failed');
      }
      await inspect('chapter1-noise-revealed', true);
      await clickButton('Generate new noise');
      if ((await js('document.body.innerText')).includes('FILTERED OUTPUT')) {
        throw new Error('Noise reveal did not reset after new seed');
      }
    } else if (label === 'Geometric Transform') {
      await clickLabel('Bilinear');
      await clickLabel('Đã dự đoán — hiển thị kết quả transform');
      await inspect('chapter1-geometry-revealed', true);
      await clickLabel('Scale');
      if (!(await js('document.body.innerText')).includes('Ảnh output đang được ẩn')) {
        throw new Error('Geometry reveal did not reset after operation change');
      }
    } else if (label === 'Fourier') {
      await clickLabel('Gaussian Low-pass');
      await clickLabel('Đã dự đoán — hiển thị mask và kết quả');
      await inspect('chapter1-fourier-revealed', true);
      await clickLabel('Ideal Low-pass');
      if (!(await js('document.body.innerText')).includes('Kết quả đang được ẩn')) {
        throw new Error('Fourier reveal did not reset after filter change');
      }
    } else if (label === 'Convolution Kernels') {
      const loaded = await js(`([...document.querySelectorAll('img')]
        .some(x=>x.complete&&x.naturalWidth>0))`);
      if (!loaded) throw new Error('SlidingWindow GIF did not load');
    }
  }
  await chapter1SourceTests();
}

async function selectSidebarSample(name) {
  const opened = await js(`(()=>{const box=document.querySelector(
    '[data-testid="stSidebar"] [data-testid="stSelectbox"]');
    if(!box)return false;(box.querySelector('[role="combobox"]')||
      box.querySelector('input')||box).click();return true})()`);
  if (!opened) throw new Error('Sample selectbox missing');
  await delay(150);
  const chosen = await js(`(()=>{const option=[...document.querySelectorAll('[role="option"]')]
    .find(x=>x.innerText.trim()===${JSON.stringify(name)}&&x.getClientRects().length);
    if(!option)return false;option.click();return true})()`);
  if (!chosen) throw new Error(`Sample option missing: ${name}`);
  await ready();
}

async function uploadSample(filename) {
  const documentNode = await call('DOM.getDocument');
  const { nodeId } = await call('DOM.querySelector', {
    nodeId: documentNode.root.nodeId, selector: 'input[type="file"]',
  });
  if (!nodeId) throw new Error('Upload input missing');
  await call('DOM.setFileInputFiles', {
    files: [path.join(root, 'chapter1_webapp', 'assets', 'samples', filename)], nodeId,
  });
  await ready();
  const body = await js('document.body.innerText');
  if (!body.includes(filename)) throw new Error(`Upload not visible: ${filename}`);
}

async function chapter1SourceTests() {
  await clickLabel('Fourier');
  await clickLabel('Đã dự đoán — hiển thị mask và kết quả');
  await selectSidebarSample('Ringing test pattern');
  if (!(await js('document.body.innerText')).includes('Kết quả đang được ẩn')) {
    throw new Error('Fourier reveal did not reset after sample changed');
  }
  await inspect('chapter1-fourier-new-sample');
  await clickLabel('Upload ảnh');
  await uploadSample('clear_edges.png');
  await clickLabel('Đã dự đoán — hiển thị mask và kết quả');
  await uploadSample('high_contrast.png');
  if (!(await js('document.body.innerText')).includes('Kết quả đang được ẩn')) {
    throw new Error('Fourier reveal did not reset after another upload');
  }
  await inspect('chapter1-fourier-new-upload');
}

async function chapter1TargetedViewports() {
  await viewport(768, 1024);
  await clickLabel('Geometric Transform');
  await inspect('chapter1-geometry-tablet', true);
  await viewport(390, 844);
  await clickLabel('Gamma');
  await inspect('chapter1-gamma-mobile', true);
  await clickLabel('Fourier');
  await inspect('chapter1-fourier-mobile', true);
}

async function chapter3Modules() {
  const modules = [
    ['RANSAC', 10], ['Hough Transform', 12], ['Sobel Convolution', 12],
    ['Canny Edge Detector', 14], ['Hough Circle Transform', 14],
    ['LSD Line Segment Detector', 14],
  ];
  for (const [label, steps] of modules) {
    if (process.env.CV_AUDIT_MODULE && process.env.CV_AUDIT_MODULE !== label) continue;
    await clickLabel(label);
    await inspect(`chapter3-${label.toLowerCase().replaceAll(/[^a-z0-9]+/g, '-')}-guided`,
      label !== 'RANSAC');
    if (!process.env.CV_AUDIT_SKIP_GUIDED) await guidedTraversal(label, steps);
    await clickLabel('B · Experiment Lab');
    await inspect(`chapter3-${label.toLowerCase().replaceAll(/[^a-z0-9]+/g, '-')}-lab`);
    await exerciseLab(label);
    await clickLabel('Presentation Mode');
    await inspect(`chapter3-${label.toLowerCase().replaceAll(/[^a-z0-9]+/g, '-')}-presentation`, true);
    await clickLabel('Presentation Mode');
    await clickLabel('A · Guided Tutorial');
  }
}

async function chapter3TargetedViewports() {
  await viewport(768, 1024);
  await clickLabel('Canny Edge Detector');
  await inspect('chapter3-canny-tablet', true);
  await viewport(390, 844);
  await clickLabel('Hough Circle Transform');
  await inspect('chapter3-circle-mobile', true);
  await clickLabel('LSD Line Segment Detector');
  await inspect('chapter3-lsd-mobile', true);
  await clickLabel('B · Experiment Lab');
  await clickLabel('Presentation Mode');
  await inspect('chapter3-lsd-presentation-mobile', true);
}

async function exerciseLab(label) {
  if (label === 'RANSAC') {
    await clickLabel('Thấp');
    await clickButton('Chạy 1 iteration');
    const phases = [];
    for (let index = 0; index < 5; index++) {
      const phase = await js(`document.querySelector('.lab-stage-strip .active')?.innerText||''`);
      phases.push(phase);
      if (index < 4) await clickButton('Pha tiếp →');
    }
    if (new Set(phases).size !== 5) throw new Error(`RANSAC phases did not advance: ${phases}`);
    console.log('LAB_PHASES', JSON.stringify(phases));
  } else if (label === 'Hough Transform') {
    await clickLabel('B · Accumulator Voting');
    await clickButton('Next Vote');
  } else if (label === 'Sobel Convolution') {
    await clickButton('Next Product →');
  } else if (label === 'Canny Edge Detector') {
    await clickLabel('B · NMS & Hysteresis Explorer');
    await clickLabel('Hysteresis BFS');
    await clickButton('Next BFS Step →');
  } else if (label === 'Hough Circle Transform') {
    await clickLabel('B · Circle Voting');
    await clickButton('Next Vote');
  } else if (label === 'LSD Line Segment Detector') {
    await clickButton('Start strongest-seed selection / region');
  }
  await inspect(`chapter3-${label.toLowerCase().replaceAll(/[^a-z0-9]+/g, '-')}-lab-interacted`);
}

async function nextDisabled() {
  return await js(`(()=>{const b=[...document.querySelectorAll('button')]
    .find(x=>x.innerText.trim()==='Next →'&&x.getClientRects().length);
    return b ? b.disabled : null})()`);
}

async function gateChoices() {
  return await js(`({buttons:[...document.querySelectorAll('button')]
      .filter(x=>x.getClientRects().length&&x.innerText.trim())
      .map(x=>({label:x.innerText.trim(),disabled:x.disabled})),
    radios:[...document.querySelectorAll('[data-testid="stMain"] [data-testid="stRadio"]')]
      .map(g=>[...g.querySelectorAll('label')].map(x=>x.innerText.trim())),
    selectboxes:[...document.querySelectorAll('[data-testid="stMain"] [data-testid="stSelectbox"]')]
      .map(x=>x.innerText.slice(0,100))})`);
}

async function unlockGate(label, step) {
  for (let attempt = 0; attempt < 8 && await nextDisabled(); attempt++) {
    const choices = await gateChoices();
    const reveal = choices.buttons.filter(x => x.label.startsWith('Reveal') && !x.disabled).at(-1);
    if (reveal) { await clickButton(reveal.label); continue; }
    if (label === 'LSD Line Segment Detector' && step === 6) {
      await clickButton('Next Neighbor'); continue;
    }
    if (label === 'LSD Line Segment Detector' && step === 7) {
      await clickButton('Complete Region'); continue;
    }
    const radio = await js(`(()=>{const groups=[...document.querySelectorAll(
      '[data-testid="stMain"] [data-testid="stRadio"]')];
      const group=groups.at(-1);if(!group)return false;
      const option=[...group.querySelectorAll('label')]
        .find(x=>x.querySelector('input[type="radio"]:not(:checked)'));
      if(!option)return false;option.click();return true})()`);
    if (radio) { await ready(); continue; }
    const select = await js(`(()=>{const items=[...document.querySelectorAll(
      '[data-testid="stMain"] [data-testid="stSelectbox"]')];
      const box=items.at(-1);if(!box)return false;
      const item=box.querySelector('[role="combobox"]')||box.querySelector('input')||box;
      item.click();return true})()`);
    if (select) {
      await delay(150);
      const option = await js(`(()=>{const item=[...document.querySelectorAll('[role="option"]')]
        .find(x=>x.getClientRects().length);if(!item)return false;item.click();return true})()`);
      if (option) { await ready(); continue; }
    }
    const selectMarkup = await js(`document.querySelector(
      '[data-testid="stMain"] [data-testid="stSelectbox"]')?.outerHTML.slice(0,1200)||''`);
    throw new Error(`${label} step ${step} gate unresolved: ${JSON.stringify(choices)} ${selectMarkup}`);
  }
  if (await nextDisabled()) {
    throw new Error(`${label} step ${step} gate still closed: ${JSON.stringify(await gateChoices())}`);
  }
}

async function guidedTraversal(label, steps) {
  const slug = label.toLowerCase().replaceAll(/[^a-z0-9]+/g, '-');
  for (let step = 1; step <= steps; step++) {
    const heading = await js(`([...document.querySelectorAll('h1,h2,h3')]
      .map(x=>x.innerText).find(x=>x.includes('Bước '))||'')`);
    if (!heading.includes(`Bước ${step}/${steps}`)) {
      throw new Error(`${label}: expected step ${step}/${steps}, got ${heading}`);
    }
    const shot = (label === 'RANSAC' && [5, 8].includes(step)) || step === steps;
    await inspect(`chapter3-${slug}-step-${step}`, shot);
    if (step === steps) break;
    if (await nextDisabled()) await unlockGate(label, step);
    await clickButton('Next →');
  }
  console.log('GUIDED_COMPLETE', label, steps);
}

try {
  await connect();
  await viewport(1366, 768);
  await initialReady();
  await inspect(`${chapter}-initial-laptop`, true);
  for (const [width, height] of [[1920, 1080], [768, 1024], [390, 844]]) {
    await viewport(width, height);
    await inspect(`${chapter}-initial-${width}`, true);
  }
  await viewport(1366, 768);
  if (chapter === 'chapter1') {
    if (process.env.CV_AUDIT_MOBILE_ONLY) await chapter1TargetedViewports();
    else await chapter1Pages();
  } else if (process.env.CV_AUDIT_MOBILE_ONLY) await chapter3TargetedViewports();
  else await chapter3Modules();
  console.log('SUMMARY', JSON.stringify({ chapter, observations: observations.length, errors }));
  if (errors.length) process.exitCode = 1;
} finally {
  socket?.close();
  if (browser.pid && browser.exitCode === null) {
    await new Promise(resolve => {
      const killer = spawn('taskkill.exe', ['/PID', String(browser.pid), '/T', '/F'], {
        stdio: 'ignore', windowsHide: true,
      });
      killer.once('error', resolve);
      killer.once('exit', resolve);
    });
  }
  if (browser.exitCode === null) browser.kill();
  await Promise.race([
    new Promise(resolve => browser.once('exit', resolve)),
    delay(5000),
  ]);
  if (browser.exitCode !== null) {
    try { await rm(profile, { recursive: true, force: true }); }
    catch (error) { console.error(`Browser profile retained (${error.message}): ${profile}`); }
  } else console.error(`Browser profile retained (process still alive): ${profile}`);
}
