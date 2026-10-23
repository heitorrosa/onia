/* ============================================================================
   NOIC FRAME KIT — checagem sem browser
   ----------------------------------------------------------------------------
   Verifica o comportamento dos frames sem abrir o Chrome: carrega o JS real
   do kit em um DOM mínimo e confere as regras que o kit promete.

   Uso:  node noic/frames/kit/check.js
   Sai com código 1 se alguma regra falhar.
   ========================================================================== */
const fs = require('fs');
const path = require('path');

const kitDir = __dirname;
const kitJS = fs.readFileSync(path.join(kitDir, 'kit.js'), 'utf8');
const dist = path.join(kitDir, 'dist');
const src = path.join(kitDir, 'src');
const estatico = path.join(kitDir, 'estatico');
const frames = fs.existsSync(dist)
  ? fs.readdirSync(dist).filter(f => f.endsWith('.html'))
  : [];

let fails = 0;
function ok(cond, label, detail) {
  if (cond) console.log(`  ok   ${label}`);
  else { fails++; console.log(`  FALHA ${label}${detail ? ' — ' + detail : ''}`); }
}

/* --- 1. o build existe e é autocontido ------------------------------------ */
console.log('1. frames em dist/');
ok(frames.length > 0, `${frames.length} frame(s) construído(s)`);
frames.forEach(f => {
  const html = fs.readFileSync(path.join(dist, f), 'utf8');
  ok(!/src="[^"]*\.css"/.test(html), `${f}: sem CSS externo`);
  ok(!/src="[^"]*\.js"/.test(html) || /mathjax/.test(html), `${f}: sem JS externo (fora o MathJax)`);
  ok(html.includes('<style>') && html.includes('NOIC'), `${f}: CSS e JS inline`);
});

/* --- 2. o frame não declara cor nenhuma ---------------------------------- */
console.log('2. frame sem cor declarada');
/* Só o CSS conta: o <style> embutido + os atributos style="" do corpo.
   Texto corrido não é examinado (uma palavra como "black" numa frase não
   é cor). */
const NAMED = ['white', 'black', 'red', 'blue', 'green', 'gray', 'grey', 'silver',
  'orange', 'yellow', 'purple', 'pink', 'brown', 'navy', 'teal', 'olive', 'maroon',
  'lime', 'aqua', 'fuchsia', 'cyan', 'magenta', 'coral', 'salmon', 'khaki', 'indigo',
  'violet', 'crimson', 'gold', 'beige', 'ivory', 'tan', 'plum', 'orchid', 'tomato',
  'wheat', 'azure', 'lavender', 'turquoise', 'chocolate', 'firebrick', 'beige'];
const ALLOWED = ['currentcolor', 'inherit', 'transparent', 'none'];

function cssOf(html) {
  const blocks = [...html.matchAll(/<style[^>]*>([\s\S]*?)<\/style>/gi)].map(m => m[1]);
  const inline = [...html.matchAll(/style="([^"]*)"/gi)].map(m => m[1]);
  /* comentários não são CSS — "zero rgb()" escrito numa nota não é uma cor */
  return (blocks.join('\n') + '\n' + inline.join('\n')).replace(/\/\*[\s\S]*?\*\//g, '');
}

frames.forEach(f => {
  const html = fs.readFileSync(path.join(dist, f), 'utf8');
  const css = cssOf(html);

  const hex = (css.match(/#[0-9a-fA-F]{3,8}\b/g) || []).filter(h => !/^#(ah|a[0-9]|n[0-9])$/i.test(h));
  ok(hex.length === 0, `${f}: 0 cor em hexadecimal no CSS`, hex.join(', '));

  const fn = css.match(/\b(rgba?|hsla?|hwb|lab|lch|oklab|oklch|color-mix|color)\(/gi) || [];
  ok(fn.length === 0, `${f}: 0 função de cor no CSS`, fn.join(', '));

  const named = [];
  css.replace(/[a-z-]+\s*:\s*([^;{}]+)/gi, (m, val) => {
    val.split(/[\s,]+/).forEach(tok => {
      const t = tok.replace(/[()]/g, '').toLowerCase();
      if (NAMED.includes(t) && !ALLOWED.includes(t)) named.push(tok);
    });
    return m;
  });
  ok(named.length === 0, `${f}: 0 cor nomeada no CSS`, named.join(', '));

  const st = css.match(/(?:^|[;{\s])(color|background|background-color|border[a-z-]*color|fill|stroke|accent-color|outline-color|caret-color|text-decoration-color)\s*:\s*([^;}]+)/gi) || [];
  const bad = st.filter(d => {
    const val = d.split(':').slice(1).join(':').toLowerCase();
    return !ALLOWED.some(a => val.trim() === a);
  });
  ok(bad.length === 0, `${f}: toda propriedade de cor usa currentColor/inherit/transparent/none`, bad.join(' | '));

  /* cor em atributo de HTML/SVG (fill="#fff", stroke="red", color="…") */
  const attrs = html.match(/\b(?:fill|stroke|color|bgcolor|border-color)\s*=\s*"([^"]*)"/gi) || [];
  const badAttr = attrs.filter(a => {
    const val = a.split('=').slice(1).join('=').replace(/"/g, '').trim().toLowerCase();
    return !ALLOWED.includes(val);
  });
  ok(badAttr.length === 0, `${f}: 0 cor em atributo de HTML/SVG`, badAttr.join(' | '));
});

/* --- 2b. o dist/ está em dia com o src/ ----------------------------------- */
console.log('2b. dist/ em dia com src/');
{
  const { buildFrame } = require('./build.js');
  const template = fs.readFileSync(path.join(kitDir, 'template.html'), 'utf8');
  const css = fs.readFileSync(path.join(kitDir, 'theme.css'), 'utf8');
  const js = fs.readFileSync(path.join(kitDir, 'kit.js'), 'utf8');
  const sources = fs.existsSync(src) ? fs.readdirSync(src).filter(f => f.endsWith('.html')).sort() : [];
  sources.forEach(file => {
    const slug = file.replace(/\.html$/, '');
    const expected = buildFrame(fs.readFileSync(path.join(src, file), 'utf8'), template, css, js);
    const onDisk = fs.existsSync(path.join(dist, `${slug}.html`))
      ? fs.readFileSync(path.join(dist, `${slug}.html`), 'utf8') : null;
    ok(onDisk === expected, `${slug}.html: dist/ é o build atual do src/ (rebuild não muda nada)`);
  });
  /* todo frame tem gêmeo estático — a interação nunca é o único caminho */
  sources.forEach(file => {
    const slug = file.replace(/\.html$/, '');
    ok(fs.existsSync(path.join(estatico, `${slug}-estatico.html`)),
      `${slug}: tem gêmeo estático em estatico/`);
  });
}

/* --- 3. o JS do kit: highlight + degradação do MathJax -------------------- */
console.log('3. kit.js');
function fakeDom() {
  const mk = (id, depth) => {
    const e = { id, classList: { s: new Set(), add(c) { this.s.add(c); }, remove(c) { this.s.delete(c); } },
                getAttribute() { return null; }, setAttribute(k, v) { this[k] = v; }, dataset: {} };
    if (depth > 0) { e.child = mk(id + '-node', depth - 1); e.querySelector = () => e.child; }
    else e.querySelector = () => ({ classList: { s: new Set(), add() {}, remove() {} } });
    return e;
  };
  const nodes = {};
  ['n0', 'n1', 'n2', 'n3', 'a1', 'a2', 'a3', 'a4'].forEach(id => { nodes[id] = mk(id, 1); });
  return { nodes, doc: { getElementById: id => nodes[id] || null, querySelectorAll: () => [] } };
}

const { nodes, doc } = fakeDom();
global.matchMedia = () => ({ matches: false });
global.document = doc;
global.window = global;
eval(kitJS);

const steps = [['n0', 'a1'], ['n1', 'a2'], ['n2', 'a3'], ['n3', 'a4']];
NOIC.highlight(steps, 0);
ok(nodes.n0.child.classList.s.has('on'), 'highlight: o primeiro nó acende');
ok(nodes.a1.classList.s.has('on'), 'highlight: a seta do passo 1 acende');
NOIC.highlight(steps, 2);
ok(nodes.n2.child.classList.s.has('on'), 'highlight: o passo 3 acende');
ok(!nodes.n0.child.classList.s.has('on'), 'highlight: o passo 1 apaga');
ok(!nodes.a1.classList.s.has('on'), 'highlight: a seta do passo 1 apaga');
ok(nodes.a3.classList.s.has('on'), 'highlight: a seta do passo 3 acende');
ok(NOIC.reducedMotion === false, 'reducedMotion: respeitado (matchMedia)');

/* --- 4. degradação do MathJax: mjx vira texto ---------------------------- */
console.log('4. degradação do MathJax');
const tex = '\\frac{1}{k}\\sum_{i=1}^{k}\\mathbb{1}[c_i(p)]';
const made = [];
global.document.querySelectorAll = () => [{
  querySelector: () => ({ getAttribute: () => tex }),
  textContent: tex,
  replaceWith: (el) => made.push(el)
}];
global.document.createElement = () => ({ set textContent(v) { this._t = v; }, get textContent() { return this._t; }, style: {} });
NOIC.degradeMath();
ok(made.length === 1, 'degradação: o container mjx é substituído');
ok(made[0] && made[0].textContent === tex, 'degradação: o texto LaTeX é preservado');
ok(made[0] && made[0].style.fontFamily.indexOf('monospace') > -1, 'degradação: vira texto monoespaçado');

/* --- 5. prefers-reduced-motion: o ciclo para ----------------------------- */
console.log('5. prefers-reduced-motion');
{
  const realSetInterval = global.setInterval;
  const timers = [];
  global.setInterval = (fn, ms) => { timers.push(ms); return timers.length; };
  global.matchMedia = () => ({ matches: true });
  eval(kitJS);
  ok(NOIC.reducedMotion === true, 'reducedMotion: detectado quando o navegador pede menos movimento');
  let calls = 0;
  NOIC.cycle(function () { calls++; }, 1600);
  ok(calls === 1, 'reducedMotion: o ciclo roda UMA vez (frame fica estático)');
  ok(timers.length === 0, 'reducedMotion: nenhum setInterval registrado (nada anima)');
  global.setInterval = realSetInterval;
  global.matchMedia = () => ({ matches: false });
  eval(kitJS);
  ok(NOIC.reducedMotion === false, 'reducedMotion: normal quando o navegador não pede');
}

console.log(fails === 0 ? '\nTODAS AS CHEGAGENS PASSARAM' : `\n${fails} CHEGAGEM(S) FALHARAM`);
process.exit(fails === 0 ? 0 : 1);
