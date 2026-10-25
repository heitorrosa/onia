/* probe temporário: dirige o frame e lê o readout nos quatro estados */
const fs = require('fs');
const path = require('path');
const http = require('http');
const { execFile } = require('child_process');

const dist = path.join(__dirname, 'dist');
const EDGE = 'C:\\Program Files (x86)\\Microsoft\\Edge\\Application\\msedge.exe';
const FRAME = 'eixo7-pilha-e-cache.html';

const wrapper = `<!DOCTYPE html><html><head><meta charset="utf-8"></head><body style="margin:0">
<iframe id="f" src="${FRAME}" style="width:620px;height:700px;border:0"></iframe>
<script>
document.getElementById('f').onload = function () {
  var d = this.contentDocument, w = this.contentWindow;
  function snap(tag) {
    return tag + ' | cache=' + d.getElementById('rC').textContent +
           ' novos=' + d.getElementById('rN').textContent +
           ' rel=' + d.getElementById('rX').textContent +
           ' | ' + d.getElementById('msg').textContent;
  }
  var out = [];
  var click = function (id) { d.getElementById(id).click(); };
  out.push(snap('inicial'));
  click('bTurno');      out.push(snap('1 turno'));
  click('bSistema');    out.push(snap('editado'));
  click('bComprimir');  out.push(snap('comprimido'));
  click('bReset');      out.push(snap('reset'));
  document.body.setAttribute('data-out', out.join('\\n'));
};
</script></body></html>`;

const server = http.createServer((req, res) => {
  const name = path.basename(decodeURIComponent(req.url.split('?')[0]));
  if (name === '_probe.html') { res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' }); res.end(wrapper); return; }
  const file = path.join(dist, name);
  if (!fs.existsSync(file)) { res.writeHead(404); res.end('404'); return; }
  res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
  res.end(fs.readFileSync(file, 'utf8'));
});

server.listen(0, '127.0.0.1', () => {
  const port = server.address().port;
  execFile(EDGE, ['--headless=new', '--disable-gpu', '--no-sandbox', '--virtual-time-budget=6000',
    '--dump-dom', `http://127.0.0.1:${port}/_probe.html`],
    { encoding: 'utf8', timeout: 60000, maxBuffer: 8 * 1024 * 1024 }, (err, stdout) => {
      server.close();
      const m = (stdout || '').match(/data-out="([\s\S]*?)"/);
      console.log(m ? m[1].replace(/&quot;/g, '"') : 'SEM RESULTADO\n' + (stdout || err));
    });
});
