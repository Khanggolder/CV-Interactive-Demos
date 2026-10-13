// Dependency-free CDP audit against the real local Streamlit server.
// Run from repository root: node chapter2_webapp/tests/browser_audit.mjs
import { spawn } from 'node:child_process';
import { writeFile, mkdir, mkdtemp, rm } from 'node:fs/promises';
import path from 'node:path';

const root = process.cwd();
const out = path.join(root, 'chapter2_webapp', 'docs', 'screenshots');
const chapter = path.join(root, 'chapter2_webapp');
const profile = await mkdtemp(path.join(chapter, '.browser_audit_profile-'));
const edge = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const port = 9227;
await mkdir(out, { recursive: true });
const child = spawn(edge, [`--remote-debugging-port=${port}`, `--user-data-dir=${profile}`,
  '--headless=new', '--no-sandbox', '--no-first-run', '--disable-gpu', '--disable-extensions',
  '--window-size=1920,1080', 'http://localhost:8512'],
  { stdio: ['ignore', 'ignore', 'pipe'], windowsHide: true });
let edgeErrors = '';
child.stderr.on('data', chunk => { edgeErrors += chunk.toString(); });

let socket;
let nextId = 1;
const pending = new Map();
let exceptions = [];
const delay = ms => new Promise(resolve => setTimeout(resolve, ms));

async function connect() {
  let pages;
  for (let i = 0; i < 50; i++) {
    try {
      pages = await (await fetch(`http://127.0.0.1:${port}/json/list`)).json();
      if (pages.some(p => p.type === 'page')) break;
    } catch { /* Edge is starting. */ }
    await delay(200);
  }
  const page = pages?.find(p => p.type === 'page');
  if (!page) throw new Error(`No Edge page target. Edge exit=${child.exitCode}; ${edgeErrors.slice(0, 2000)}`);
  socket = new WebSocket(page.webSocketDebuggerUrl);
  await new Promise((resolve, reject) => {
    socket.onopen = resolve;
    socket.onerror = reject;
  });
  socket.onmessage = event => {
    const message = JSON.parse(event.data);
    if (message.method === 'Runtime.exceptionThrown') exceptions.push(message.params.exceptionDetails.text);
    if (message.id && pending.has(message.id)) {
      const { resolve, reject } = pending.get(message.id);
      pending.delete(message.id);
      message.error ? reject(new Error(JSON.stringify(message.error))) : resolve(message.result);
    }
  };
  await call('Runtime.enable');
  await call('Page.enable');
}

function call(method, params = {}) {
  const id = nextId++;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    socket.send(JSON.stringify({ id, method, params }));
  });
}

async function evaluate(expression) {
  const result = await call('Runtime.evaluate', { expression, returnByValue: true, awaitPromise: true });
  if (result.exceptionDetails) throw new Error(result.exceptionDetails.text);
  return result.result.value;
}

async function waitFor(text) {
  for (let i = 0; i < 50; i++) {
    if (await evaluate(`document.body?.innerText.includes(${JSON.stringify(text)})`)) return;
    await delay(200);
  }
  throw new Error(`Text not visible: ${text}`);
}

async function viewport(width, height) {
  await call('Emulation.setDeviceMetricsOverride', {
    width, height, deviceScaleFactor: 1, mobile: width < 800,
  });
  await delay(350);
}

async function screenshot(name) {
  // Streamlit may render the heading while charts are still being computed.
  for (let i = 0; i < 40; i++) {
    const busy = await evaluate(`[...document.querySelectorAll('button')].some(b => b.innerText.trim() === 'Stop')`);
    if (!busy) break;
    await delay(200);
  }
  await delay(700);
  const { data } = await call('Page.captureScreenshot', {
    format: 'png', captureBeyondViewport: false, fromSurface: true,
  });
  await writeFile(path.join(out, name), Buffer.from(data, 'base64'));
  const state = await evaluate(`({title:document.title, width:innerWidth,
    scrollWidth:document.documentElement.scrollWidth,
    bodyWidth:document.body.scrollWidth,
    streamlitError:document.body.innerText.includes('Traceback') ||
      document.querySelector('[data-testid="stException"]') !== null})`);
  console.log(name, JSON.stringify(state));
  if (state.scrollWidth > state.width + 2 || state.bodyWidth > state.width + 2 || state.streamlitError) {
    throw new Error(`Layout/application error in ${name}`);
  }
}

async function scrolledScreenshot(name, y) {
  const scroll = amount => evaluate(`(() => {
    window.scrollTo(0, ${amount});
    for (const el of [document.scrollingElement,
      ...document.querySelectorAll('[data-testid="stAppViewContainer"], [data-testid="stMain"], section.main')]) {
      if (el && el.scrollHeight > el.clientHeight + 10) el.scrollTop = ${amount};
    }
  })()`);
  await scroll(y);
  await screenshot(name);
  await scroll(0);
}

async function clickButton(label) {
  let clicked = false;
  for (let i = 0; i < 15 && !clicked; i++) {
    clicked = await evaluate(`(() => {
      const button = [...document.querySelectorAll('button')].find(b => b.innerText.trim() === ${JSON.stringify(label)} && !b.disabled);
      if (!button) return false;
      button.click(); return true;
    })()`);
    if (!clicked) await delay(200);
  }
  if (!clicked) {
    const buttons = await evaluate(`[...document.querySelectorAll('button')].map(b=>[b.innerText.trim(),b.disabled])`);
    throw new Error(`Enabled button not found: ${label}; buttons=${JSON.stringify(buttons)}`);
  }
  await delay(400);
}

async function clickText(text) {
  const clicked = await evaluate(`(() => {
    const el = [...document.querySelectorAll('label')].find(e => e.innerText.trim() === ${JSON.stringify(text)});
    if (!el) return false;
    el.click(); return true;
  })()`);
  if (!clicked) throw new Error(`Label not found: ${text}`);
  await delay(350);
}

async function clickSummary(text) {
  const clicked = await evaluate(`(() => {
    const el = [...document.querySelectorAll('summary')].find(e => e.innerText.includes(${JSON.stringify(text)}));
    if (!el) return false;
    el.click(); return true;
  })()`);
  if (!clicked) throw new Error(`Expander not found: ${text}`);
  await delay(350);
}

async function selectOption(label, value) {
  const opened = await evaluate(`(() => {
    const box = [...document.querySelectorAll('[data-testid="stSelectbox"]')]
      .find(e => e.innerText.includes(${JSON.stringify(label)}));
    const control = box?.querySelector('[role="combobox"]');
    if (!control) return false;
    control.click(); return true;
  })()`);
  if (!opened) throw new Error(`Selectbox not found: ${label}`);
  await delay(200);
  const selected = await evaluate(`(() => {
    const option = [...document.querySelectorAll('[role="option"]')]
      .find(e => e.innerText.trim() === ${JSON.stringify(value)});
    if (!option) return false;
    option.click(); return true;
  })()`);
  if (!selected) throw new Error(`Option not found: ${value}`);
  await delay(350);
}

try {
  await connect();
  await waitFor('Vì sao Harris tìm được góc?');
  for (const [w, h, label] of [[1920,1080,'desktop'], [1366,768,'laptop'],
                               [768,1024,'tablet'], [390,844,'mobile']]) {
    await viewport(w, h);
    await screenshot(`guided-initial-${label}.png`);
  }
  await viewport(1366, 768);
  await clickButton('Next →');
  await waitFor('Ba cấu trúc');
  await screenshot('guided-prediction-gate.png');
  await clickText('Patch C');
  await clickButton('Reveal kết quả');
  await screenshot('guided-prediction-revealed.png');
  // Advance to numeric sliding-window view.
  await clickButton('Next →');
  await waitFor('Dịch thật một cửa sổ');
  await screenshot('guided-sliding-window.png');
  const gates = {4:'Cả x và y', 8:'Góc', 9:'Góc',
                 11:'Điểm theo thứ tự hàng/cột nhỏ nhất', 12:'min(λ₁,λ₂)'};
  const captures = {7:'guided-tensor.png', 8:'guided-eigen-gate.png',
                    10:'guided-harris-map.png', 11:'guided-nms-gate.png',
                    12:'guided-shi-gate.png', 13:'guided-robustness.png'};
  for (let current = 3; current < 14; current++) {
    if (gates[current]) {
      await clickText(gates[current]);
      await clickButton('Reveal kết quả');
      if (current === 11) {
        await screenshot('guided-nms-revealed.png');
        await scrolledScreenshot('guided-nms-decision.png', 660);
      }
      if (current === 12) {
        await screenshot('guided-shi-revealed.png');
        await scrolledScreenshot('guided-shi-comparison.png', 650);
      }
    }
    await clickButton('Next →');
    await waitFor(`BƯỚC ${current+1}/14`);
    if (captures[current+1]) await screenshot(captures[current+1]);
    if (current+1 === 7) await scrolledScreenshot('guided-tensor-detail.png', 540);
    if (current+1 === 10) await scrolledScreenshot('guided-harris-map-detail.png', 480);
    if (current+1 === 13) {
      await selectOption('Thử biến đổi', 'Góc đổi tỉ lệ');
      await screenshot('guided-rescaled-corner.png');
    }
  }
  await clickText('Không');
  await clickButton('Reveal kết quả');
  await screenshot('guided-recap.png');
  await clickText('Experiment Lab');
  await waitFor('Reset Lab');
  await screenshot('experiment-lab-laptop.png');
  await scrolledScreenshot('experiment-lab-charts.png', 420);
  await clickSummary('Tensor, response và NMS');
  await clickText('Shi-Tomasi');
  await screenshot('experiment-shi-controls.png');
  await clickSummary('Tensor, response và NMS');
  await clickText('Presentation Mode');
  await waitFor('Harris & Shi-Tomasi');
  await screenshot('presentation-laptop.png');
  await clickText('Presentation Mode');
  await viewport(390, 844);
  await screenshot('experiment-lab-mobile.png');
  await scrolledScreenshot('experiment-lab-mobile-chart.png', 540);
  console.log('Browser exceptions:', JSON.stringify(exceptions));
  if (exceptions.length) throw new Error('Browser JavaScript exceptions');
} finally {
  try { await call('Browser.close'); } catch { child.kill(); }
  socket?.close();
  if (path.dirname(path.resolve(profile)) !== path.resolve(chapter) ||
      !path.basename(profile).startsWith('.browser_audit_profile-')) {
    throw new Error(`Refusing to remove unexpected browser profile: ${profile}`);
  }
  await rm(profile, { recursive: true, force: true, maxRetries: 8, retryDelay: 100 });
}
