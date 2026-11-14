'use strict';

// The page is a form that works without this script: sending it asks the
// server for the page again, which then has the result in it. With the
// script the page stays where it is: the fields follow the choices at once,
// the result is taken out of the page that the server answers with, and a
// calculation that takes longer is asked for again until it has ended.

// The two numbers of each function, in the order of the command line.
const numbers = { tet: ['base', 'height'], ate: ['base', 'value'], ssrt: ['value', 'height'] };
const symbols = { base: 'b', height: 'h', value: 'z' };
const examples = {
  tet: { base: '2', height: '0.5' },
  ate: { base: '2', value: '65536' },
  ssrt: { value: '16', height: '3' },
};

// What was typed, for each function and each kind of arguments, so that
// switching keeps it.
const typed = {};

// The number of the calculation that the page shows. One that takes longer
// is asked for again until its result is there, or until another is started.
let latest = 0;

const byId = (id) => document.getElementById(id);
const checked = (name) => document.querySelector(`input[name="${name}"]:checked`).value;
const fields = () => Array.from(byId('arguments').querySelectorAll('input'));

function names(func, count) {
  const [first, second] = numbers[func];
  return count === '2' ? [first, second] : [`${first}_re`, `${first}_im`, `${second}_re`, `${second}_im`];
}

function initial(func, name) {
  if (name.endsWith('_im')) return '0';
  return examples[func][name.replace('_re', '')];
}

function remember() {
  for (const input of fields()) typed[input.dataset.key] = input.value;
}

// The fields of the chosen function and kind of arguments, as the server
// writes them into the page.
function showArguments() {
  const func = checked('func');
  const box = byId('arguments');
  box.replaceChildren();
  for (const name of names(func, checked('numbers'))) {
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
    input.name = name;
    input.dataset.key = key;
    input.value = key in typed ? typed[key] : initial(func, name);
    input.spellcheck = false;
    input.autocomplete = 'off';
    input.autocapitalize = 'off';
    row.append(label, input);
    box.append(row);
  }
}

// The address of the page for what the form says: what the browser asks
// for when it sends the form, with every value escaped in full.
function address() {
  const parts = [
    'format=html',
    `prec=${encodeURIComponent(byId('prec').value.trim())}`,
    `func=${checked('func')}`,
    `numbers=${checked('numbers')}`,
  ];
  for (const input of fields()) parts.push(`${input.name}=${encodeURIComponent(input.value)}`);
  return `/?${parts.join('&')}`;
}

// Says why there is no page to take a result from.
function fail(message) {
  const title = document.createElement('h2');
  title.id = 'result-title';
  title.textContent = 'No result';
  const error = document.createElement('p');
  error.id = 'error';
  error.className = 'error';
  error.textContent = message;
  const result = byId('result');
  delete result.dataset.running;
  result.replaceChildren(title, error);
}

async function calculate() {
  const button = byId('calculate');
  const status = byId('status');
  latest += 1;
  const mine = latest;
  remember();
  const target = address();
  // The address says what the page shows, as it does without the script.
  window.history.replaceState(null, '', target);
  const started = Date.now();
  const tick = () => {
    status.textContent = `calculating… ${((Date.now() - started) / 1000).toFixed(1)} s`;
  };
  button.disabled = true;
  tick();
  const timer = setInterval(tick, 100);
  const free = () => {
    clearInterval(timer);
    status.textContent = '';
    button.disabled = false;
  };
  try {
    for (;;) {
      const response = await fetch(target, { headers: { Accept: 'text/html' } });
      const page = new DOMParser().parseFromString(await response.text(), 'text/html');
      if (mine !== latest) return;
      const result = page.getElementById('result');
      if (!result) {
        fail(`the server answered with status ${response.status} and no result`);
        break;
      }
      byId('result').replaceWith(document.adoptNode(result));
      const again = Number(result.dataset.running);
      if (!again) break;
      // The calculation continues on the server. The page is free for
      // another one, and asks for this one again until that is started.
      free();
      await new Promise((resolve) => { setTimeout(resolve, again * 1000); });
      if (mine !== latest) return;
    }
  } catch (error) {
    if (mine === latest) fail(`the server could not be reached: ${error.message}`);
  } finally {
    clearInterval(timer);
    if (mine === latest) free();
  }
}

for (const radio of document.querySelectorAll('input[name="func"], input[name="numbers"]')) {
  radio.addEventListener('change', () => {
    remember();
    showArguments();
  });
}
byId('form').addEventListener('submit', (event) => {
  event.preventDefault();
  calculate();
});
// A browser may keep the choices of a page that it loads again, and not
// the fields that the server wrote for them.
if (fields().map((input) => input.name).join() !== names(checked('func'), checked('numbers')).join()) {
  showArguments();
}
remember();
// The server made this page for a calculation that still runs: from here
// on the script asks for it.
if (byId('result').dataset.running) calculate();
