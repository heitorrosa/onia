/* ============================================================================
   NOIC FRAME KIT — build determinístico
   ----------------------------------------------------------------------------
   Padrão do repositório (question-bank/template.html + build.js → exam.html),
   aplicado aos frames: fonte editável em src/ → HTML autocontido em dist/.

   Uso:  node noic/frames/kit/build.js
   Saída: noic/frames/kit/dist/<slug>.html  (um arquivo, zero build
          dependency em runtime, CSS e JS inline).

   Contrato da fonte (src/<slug>.html):
     • primeira linha: <!--title: ...-->  (vira <title>)
     • restante: o corpo do frame (fragmento HTML)
   ========================================================================== */
const fs = require('fs');
const path = require('path');

const kitDir = __dirname;
const srcDir = path.join(kitDir, 'src');
const distDir = path.join(kitDir, 'dist');

/* Monta um frame a partir do shell + CSS + JS + corpo.
   Função pura: o mesmo input dá sempre o mesmo output (sem timestamp, sem
   random) — é o que torna o build determinístico. `check.js` usa esta
   função para conferir se o dist/ está em dia, sem reescrever nada. */
function buildFrame(raw, template, css, js) {
  const m = raw.match(/^<!--title:\s*(.+?)\s*-->\r?\n?/);
  if (!m) throw new Error('frame sem a primeira linha <!--title: ...-->');
  return template
    .replace('{{TITLE}}', m[1])
    .replace('{{CSS}}', css.trim())
    .replace('{{JS}}', js.trim())
    .replace('{{BODY}}', raw.slice(m[0].length).trim());
}

function main() {
  const template = fs.readFileSync(path.join(kitDir, 'template.html'), 'utf8');
  const css = fs.readFileSync(path.join(kitDir, 'theme.css'), 'utf8');
  const js = fs.readFileSync(path.join(kitDir, 'kit.js'), 'utf8');

  if (!fs.existsSync(srcDir)) {
    console.error('src/ não existe — nada a construir.');
    process.exit(1);
  }
  fs.mkdirSync(distDir, { recursive: true });

  const files = fs.readdirSync(srcDir).filter(f => f.endsWith('.html')).sort();
  if (!files.length) {
    console.error('src/ está vazio — nada a construir.');
    process.exit(1);
  }

  let n = 0;
  for (const file of files) {
    const slug = file.replace(/\.html$/, '');
    const raw = fs.readFileSync(path.join(srcDir, file), 'utf8');
    let html;
    try {
      html = buildFrame(raw, template, css, js);
    } catch (e) {
      console.error(`ERRO: ${file} ${e.message}`);
      process.exit(1);
    }
    fs.writeFileSync(path.join(distDir, `${slug}.html`), html, 'utf8');
    console.log(`built ${slug}.html  ${html.length} bytes`);
    n++;
  }
  console.log(`\n${n} frame(s) construído(s) em dist/`);
}

if (require.main === module) main();
module.exports = { buildFrame };
