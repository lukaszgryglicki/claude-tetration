'use strict';

// The two numbers of each function, in the order of the command line.
const numbers = { tet: ['base', 'height'], ate: ['base', 'value'], ssrt: ['value', 'height'] };
const symbols = { base: 'b', height: 'h', value: 'z' };
const examples = {
  tet: { base: '2', height: '0.5' },
  ate: { base: '2', value: '65536' },
  ssrt: { value: '16', height: '3' },
};

// What was typed, for each function and each form, so that switching keeps it.
const typed = {};

const byId = (id) => document.getElementById(id);
const checked = (name) => document.querySelector(`input[name="${name}"]:checked`).value;

function names(func, form) {
  const [first, second] = numbers[func];
  return form === '2' ? [first, second] : [`${first}_re`, `${first}_im`, `${second}_re`, `${second}_im`];
}

function initial(func, name) {
  if (name.endsWith('_im')) return '0';
  return examples[func][name.replace('_re', '')];
}

function remember() {
  for (const input of byId('arguments').querySelectorAll('input')) {
    typed[input.dataset.key] = input.value;
  }
}

function showArguments() {
  const func = checked('func');
  const form = checked('form');
  const box = byId('arguments');
  box.replaceChildren();
  for (const name of names(func, form)) {
    const key = `${func}.${name}`;
    const row = document.createElement('div');
    row.className = 'row';
    const label = document.createElement('label');
    label.htmlFor = name;
    const number = name.replace(/_(re|im)$/, '');
    const part = name.endsWith('_re') ? ', real part' : name.endsWith('_im') ? ', imaginary part' : '';
    label.textContent = `${number} ${symbols[number]}${part}`;
    const input = document.createElement('input');
    input.type = 'text';
    input.id = name;
    input.dataset.key = key;
    input.value = key in typed ? typed[key] : initial(func, name);
    input.spellcheck = false;
    input.autocomplete = 'off';
    input.autocapitalize = 'off';
    row.append(label, input);
    box.append(row);
  }
}

// A value in a query: everything that could be misread is escaped, and
// what a person would type (2+i, pi/e, 2^^0.5) stays readable.
function readable(value) {
  return encodeURIComponent(value)
    .replace(/%2B/g, '+')
    .replace(/%2F/g, '/')
    .replace(/%2C/g, ',')
    .replace(/%5E/g, '^');
}

function query(encode) {
  const func = checked('func');
  const parts = [`prec=${encode(byId('prec').value.trim())}`, `func=${func}`];
  for (const name of names(func, checked('form'))) {
    parts.push(`${name}=${encode(byId(name).value)}`);
  }
  return `?${parts.join('&')}`;
}

function show(data, status) {
  const failed = Boolean(data.error);
  byId('result').hidden = false;
  byId('result-title').textContent = failed ? 'No result' : `Result: the ${data.result}`;
  byId('re').textContent = failed ? '' : data.re;
  byId('im').textContent = failed ? '' : data.im;
  byId('result').querySelector('dl').hidden = failed;
  byId('error').hidden = !failed;
  byId('error').textContent = failed ? data.error + (status ? ` (HTTP ${status})` : '') : '';
  const seconds = typeof data.seconds === 'number' ? `computed in ${data.seconds} s` : '';
  byId('timing').textContent = data.cached ? `${seconds}, returned from the cache` : seconds;
  byId('command').textContent = data.command || '';
}

async function calculate() {
  const button = byId('calculate');
  const status = byId('status');
  const precision = byId('prec').value.trim();
  if (!/^[0-9]+$/.test(precision) || Number(precision) < 1) {
    show({ error: 'the precision is a whole number of digits, at least 1' }, 0);
    return;
  }
  remember();
  const link = byId('api');
  link.href = query(encodeURIComponent);
  link.textContent = new URL(query(readable), window.location.href).href;
  const started = Date.now();
  const tick = () => {
    status.textContent = `calculating… ${((Date.now() - started) / 1000).toFixed(1)} s`;
  };
  button.disabled = true;
  tick();
  const timer = setInterval(tick, 100);
  try {
    const response = await fetch(query(encodeURIComponent), { headers: { Accept: 'application/json' } });
    let data;
    try {
      data = await response.json();
    } catch (error) {
      data = { error: `the server answered with status ${response.status} and no result` };
    }
    show(data, response.status);
  } catch (error) {
    show({ error: `the server could not be reached: ${error.message}` }, 0);
  } finally {
    clearInterval(timer);
    status.textContent = '';
    button.disabled = false;
  }
}

async function showLimits() {
  try {
    const stats = await (await fetch('stats')).json();
    byId('limits').textContent =
      `${stats.limits.timeout_seconds} s for one calculation, up to ${stats.limits.max_prec} digits`;
    byId('prec').max = stats.limits.max_prec;
    if (stats.revision) byId('revision').textContent = `, revision ${stats.revision}`;
  } catch (error) {
    // The page works without them.
  }
}

for (const radio of document.querySelectorAll('input[name="func"], input[name="form"]')) {
  radio.addEventListener('change', () => {
    remember();
    showArguments();
  });
}
byId('calculate').addEventListener('click', calculate);
document.addEventListener('keydown', (event) => {
  if (event.key === 'Enter' && event.target.tagName === 'INPUT' && !byId('calculate').disabled) {
    calculate();
  }
});
showArguments();
showLimits();
