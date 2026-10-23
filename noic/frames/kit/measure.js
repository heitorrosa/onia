/* ============================================================================
   NOIC FRAME KIT — medidor de altura
   ----------------------------------------------------------------------------
   O WordPress do NOIC não ajusta a altura do iframe depois do carregamento
   (o preview do editor não executa postMessage), então a altura é FIXA e
   precisa ser medida — não chutada.

   Este script mede cada frame em dist/ na largura de conteúdo do site
   (620px, medida no CSS do tema) e devolve:
     · a altura em px
     · o bloco <figure class="noic-frame"> pronto para colar no WordPress

   Uso:  node noic/frames/kit/measure.js [larguraEmPx]
   Requer: Edge/Chrome headless no PATH padrão do Windows.

   Também é módulo:  const { measureFrames } = require('./measure.js')
   → measureFrames(620) devolve [{ frame, height }]
   ========================================================================== */
const fs = require('fs');
const path = require('path');
const { execFile } = require('child_process');
const http = require('http');

const kitDir = __dirname;
const dist = path.join(kitDir, 'dist');
const WIDTH = Number(process.argv[2]) || 620;   /* content-size do tema NOIC */
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';

/* O wrapper é escrito dentro de dist/ para servir de mesma origem (o
   contentDocument do iframe só é legível se a origem casar).
   O iframe do wrapper é BAIXO de propósito: o scrollHeight do documento
   interno nunca é menor que a altura da janela interna, então um wrapper
   alto actua como piso (com 400px, todo frame abaixo de 400px media 400). */
const wrapper = `<!DOCTYPE html><html><head><meta charset="utf-8"><title>medir</title></head>
<body style="margin:0">
<iframe id="f" src="FRAME" style="width:${WIDTH}px;height:60px;border:0"></iframe>
<script>
document.getElementById('f').onload = function () {
  var d = this.contentDocument;
  document.body.setAttribute('data-h', Math.max(d.documentElement.scrollHeight, d.body.scrollHeight));
};
</script></body></html>`;

const tmp = path.join(dist, '_medir.html');

/* dump-dom do Edge, com timeout */
function dumpDom(url) {
  return new Promise((resolve) => {
    const p = execFile(EDGE, [
      '--headless=new', '--disable-gpu', '--no-sandbox',
      '--virtual-time-budget=9000', '--dump-dom', url
    ], { encoding: 'utf8', timeout: 60000, maxBuffer: 16 * 1024 * 1024 });
    let out = '';
    p.stdout.on('data', d => { out += d; });
    p.on('close', () => resolve(out));
    p.on('error', () => resolve(''));
  });
}

/* servidor que serve dist/ (mesma origem do wrapper) */
function makeServer() {
  return http.createServer((req, res) => {
    const name = path.basename(decodeURIComponent(req.url.split('?')[0]));
    const file = path.join(dist, name);
    if (!fs.existsSync(file)) { res.writeHead(404); res.end('404'); return; }
    res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
    res.end(fs.readFileSync(file, 'utf8'));
  });
}

/* Mede todo frame de dist/ na largura pedida.
   Devolve [{ frame, height }] — height = 0 quando não deu para medir. */
async function measureFrames(width = WIDTH) {
  if (!fs.existsSync(EDGE)) {
    console.error(`Edge não encontrado em ${EDGE} — meça na mão e ajuste a altura.`);
    return [];
  }
  if (!fs.existsSync(dist)) {
    console.error('dist/ não existe — rode build.js antes.');
    return [];
  }
  const frames = fs.readdirSync(dist).filter(f => f.endsWith('.html')).sort();
  if (!frames.length) {
    console.error('dist/ está vazio — rode build.js antes.');
    return [];
  }

  const server = makeServer();
  await new Promise(res => server.listen(0, '127.0.0.1', res));
  const port = server.address().port;
  const results = [];
  try {
    for (const f of frames) {
      fs.writeFileSync(tmp, wrapper.replace('FRAME', f), 'utf8');
      const dom = await dumpDom(`http://127.0.0.1:${port}/_medir.html`);
      const m = dom.match(/data-h="(\d+)"/);
      results.push({ frame: f, height: m ? Number(m[1]) : 0 });
      if (!m) console.error(`  ! não consegui medir ${f}`);
    }
  } finally {
    try { fs.unlinkSync(tmp); } catch (e) { /* o wrapper é temporário */ }
    server.close();
  }
  return results;
}

module.exports = { measureFrames, WIDTH, EDGE };

if (require.main === module) {
  (async () => {
    const results = await measureFrames(WIDTH);
    if (!results.length) process.exit(1);

    console.log(`\nAltura dos frames em ${WIDTH}px (content-size do tema NOIC)\n`);
    console.log('| frame | altura | bloco para o WordPress |');
    console.log('|---|---|---|');
    results.forEach(r => {
      console.log(`| ${r.frame} | ${r.height || '?'}px | \`style="height:${r.height || 520}px"\` |`);
    });

    console.log('\n--- bloco completo para colar (o mesmo para todos) ---');
    console.log(`
<figure class="noic-frame">
  <iframe src="https://<host>/noic/classes/FRAMESLUG.html"
          title="TITULO"
          loading="lazy" referrerpolicy="no-referrer"
          style="width:100%;height:ALTURAPX;border:0;border-radius:8px;"
          sandbox="allow-scripts"></iframe>
  <figcaption><strong>Fig. N</strong> — o que a interação mostra.</figcaption>
</figure>
`);

    if (results.some(r => !r.height)) process.exit(1);
  })();
}
