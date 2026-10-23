/* ============================================================================
   NOIC FRAME KIT — checagem do checklist §7 do TEMPLATE-classe-noic.md
   ----------------------------------------------------------------------------
   O check.js audita o FRAME por dentro (o que ele declara, o que ele anima).
   Este audita a CLASSE por fora: o markdown da classe e os blocos <iframe>
   que vao para o WordPress.

   Regras cobertas (TEMPLATE-classe-noic.md §7):
     Conteudo   · titulo sem preambulo ("Introducao a...")
                · forma da classe: topicos e subtopicos, sem secao de fechamento
                · nenhum exercicio, questao ou gabarito
                · cada equacao de exibicao tem uma frase de leitura
                · formulas em \\( \\) e \\[ \\], nunca em HTML
     Visual     · 1 cor de acento por classe, sem verde
     Interativo · cada iframe: loading=lazy, sandbox, title, altura fixa
                · o src aponta para um frame que existe em dist/
                · a altura escrita e a altura MEDIDA (mede.js)
                · o kit tem o gemeo estatico do frame (a classe nao usa)

   Uso:  node noic/frames/kit/check-classe.js [classe.md ...]
   Sem argumento: audita todas as aulas em noic/classes/<slug>/aula.md.
   Sai com codigo 1 se alguma regra falhar.
   ========================================================================== */
const fs = require('fs');
const path = require('path');

const kitDir = __dirname;
const repoRoot = path.resolve(kitDir, '..', '..', '..');
const classesDir = path.join(repoRoot, 'noic', 'classes');
const distDir = path.join(kitDir, 'dist');
const estaticoDir = path.join(kitDir, 'estatico');
const templateClasse = path.join(repoRoot, 'noic', '_exploracao', 'TEMPLATE-classe-noic.md');

let fails = 0;
let skips = 0;
function ok(cond, label, detail) {
  if (cond) { console.log(`  ok   ${label}`); return true; }
  fails++;
  console.log(`  FALHA ${label}${detail ? ' — ' + detail : ''}`);
  return false;
}
function skip(label) { skips++; console.log(` skip ${label}`); }

/* ------------------------------------------------------------------ */
/* utilidades                                                          */
/* ------------------------------------------------------------------ */
function norm(s) {
  return s.normalize('NFD').replace(/[\u0300-\u036f]/g, '').toLowerCase();
}
function headings(md) {
  return md.split('\n')
    .filter(l => /^#{1,6}\s+/.test(l))
    .map(l => l.replace(/^#{1,6}\s+/, '').trim());
}

/* ------------------------------------------------------------------ */
/* medicao (a altura do iframe tem de ser a medida, nao a estimativa)   */
/* ------------------------------------------------------------------ */
let measured = null;
async function getMeasured() {
  if (measured) return measured;
  try {
    const { measureFrames } = require('./measure.js');
    measured = await measureFrames(620);
  } catch (e) {
    measured = [];
  }
  return measured;
}

/* ------------------------------------------------------------------ */
/* auditoria de uma classe                                             */
/* ------------------------------------------------------------------ */
async function auditClass(file) {
  const md = fs.readFileSync(file, 'utf8');
  const name = path.basename(file);
  const n = norm(md);
  console.log(`\n== ${name} (${md.length} B) ==`);

  /* --- conteudo: titulo ---
     O titulo pode ser "verbo + objeto" (verbo do syllabus) OU uma frase de
     substantivo, como os titulos de secao do Poliedro ("Carga eletrica",
     "Geradores"). O que nao passa e o preambulo ("Introducao a...", "Sobre",
     "Panorama", "Um olhar sobre..."). Os VERBOS do syllabus sao obrigatorios
     nos OBJETIVOS, que sao checados logo abaixo. */
  const h1 = headings(md).find(h => !/^classe/i.test(h)) || '';
  const firstWord = h1.split(/\s+/)[0] || '';
  ok(!/^(introdu[cç][ãa]o|sobre|panorama|panor[âa]mica|no [a-z]+ de|um olhar)\b/i.test(h1),
    'titulo nao e "Introducao a..."', h1);
  ok(/[A-Za-zÀ-ÿ]/.test(h1), 'titulo existe e nao e vazio', h1);

  /* --- conteudo: forma da classe (decisao do autor) ---
     A classe nao abre com bloco de contrato (2026-10-21) e nao tem secao de
     fechamento: o que era "onde o modelo quebra", "conexoes", "mapa",
     "fechamento" e "apendice" foi dissolvido no proprio texto. */
  const headLines = md.split('\n').filter(l => /^#{1,6}\s+/.test(l));
  const h2s = headLines.filter(l => /^##\s/.test(l)).map(l => l.replace(/^##\s+/, '').trim());
  const h3s = headLines.filter(l => /^###\s/.test(l)).map(l => l.replace(/^###\s+/, '').trim());
  ok(h2s.length >= 2, `a classe tem pelo menos 2 topicos (${h2s.length})`);
  ok(h3s.length >= 3, `a classe tem pelo menos 3 subtopicos (${h3s.length})`);
  const proibidas = ['onde o modelo quebra', 'mapa', 'conexoes', 'fechamento', 'apendice'];
  const achadas = h2s.concat(h3s).filter(h => proibidas.some(p => norm(h) === norm(p)));
  ok(achadas.length === 0,
    'nenhuma secao de fechamento (quebras, conexoes, mapa, fechamento, apendice)', achadas.join(' | '));

  /* --- conteudo: sem exercicios --- */
  const hs = headings(md);
  const exHs = hs.filter(h => /exerc|quest|gabarit|fixa[çc]|resolva|pr[áa]tica|tarefa de casa/i.test(h));
  ok(exHs.length === 0, 'nenhum titulo de exercicio/questao/gabarito', exHs.join(' | '));

  /* --- conteudo: cada equacao de exibicao tem frase de leitura --- */
  const caixas = (md.match(/<div class="caixa">[\s\S]*?<\/div>/g) || []).join('\n');
  const eqs = (caixas.match(/\\\[/g) || []).length;
  const leituras = (caixas.match(/class="leitura"/g) || []).length;
  ok(eqs > 0, `a classe tem equacoes de exibicao (${eqs})`);
  ok(eqs === leituras, `cada equacao de exibicao tem uma frase de leitura (${eqs}/${leituras})`);

  /* --- conteudo: matematica --- */
  const o = (md.match(/\\\(/g) || []).length, c = (md.match(/\\\)/g) || []).length;
  const ob = (md.match(/\\\[/g) || []).length, cb = (md.match(/\\\]/g) || []).length;
  ok(o === c, `delimitadores \\( \\) balanceados (${o}/${c})`);
  ok(ob === cb, `delimitadores \\[ \\] balanceados (${ob}/${cb})`);
  ok(!/\$\$/.test(md), 'nenhum $ (use \\[ \\])');

  // o marked corta o bloco HTML numa linha em branco: o <pre> continua, mas o
  // resto vira markdown (e uma linha comecada por # vira titulo).
  const pres = md.match(/<pre>[\s\S]*?<\/pre>/g) || [];
  ok(pres.every(b => !/\n[ \t]*\n/.test(b)), 'nenhuma linha em branco dentro de <pre> (' + pres.length + ' bloco(s))');
  ok(!/<(math|equation|mml)\b/i.test(md), 'nenhuma equacao em HTML');
  ok(ob > 0, 'a classe tem pelo menos uma formula de exibicao');

  /* --- visual: 1 acento por classe, sem verde --- */
  const hexes = [...new Set((md.match(/#[0-9a-fA-F]{3,8}\b/g) || [])
    .filter(h => !/^#(ah|a[0-9]|n[0-9]|d[0-9]|s[0-9]|mc|mv[0-9]|ml[0-9])$/i.test(h)))];
  const accents = [...md.matchAll(/--accent\s*:\s*(#[0-9a-fA-F]{3,8})/g)].map(m => m[1]);
  ok(accents.length === 1, `a classe declara 1 cor de acento (${accents.length})`, accents.join(', '));
  const greens = hexes.filter(h => {
    const s = h.slice(1);
    const r = parseInt(s.length === 3 ? s[0] + s[0] : s.slice(0, 2), 16);
    const g = parseInt(s.length === 3 ? s[1] + s[1] : s.slice(2, 4), 16);
    const b = parseInt(s.length === 3 ? s[2] + s[2] : s.slice(4, 6), 16);
    return g > r + 18 && g > b + 18;
  });
  ok(greens.length === 0, 'nenhuma cor verde na classe', greens.join(', '));

  /* --- interativo: os iframes --- */
  const iframes = [...md.matchAll(/<iframe\b[\s\S]*?<\/iframe>/gi)].map(m => m[0]);
  ok(iframes.length > 0, `a classe tem ${iframes.length} iframe(s)`);
  const meas = await getMeasured();
  const heights = new Map(meas.map(r => [r.frame, r.height]));

  for (const tag of iframes) {
    const src = (tag.match(/src="([^"]*)"/) || [, ''])[1];
    const slug = path.basename(src);
    const ttl = (tag.match(/title="([^"]*)"/) || [, ''])[1];
    const sty = (tag.match(/style="([^"]*)"/) || [, ''])[1];
    const h = (sty.match(/height:(\d+)px/) || [])[1];

    ok(/loading="lazy"/.test(tag), `${slug}: loading="lazy"`);
    ok(/sandbox="allow-scripts"/.test(tag), `${slug}: sandbox="allow-scripts"`);
    ok(!!ttl && !/[<>]/.test(ttl), `${slug}: title descritivo`, ttl);
    ok(/width:100%/.test(sty), `${slug}: width:100%`);
    ok(!!h, `${slug}: altura fixa (height:Npx)`, sty);
    ok(!/pxpx/.test(sty), `${slug}: a altura nao tem px duplo`, sty);

    const built = fs.existsSync(path.join(distDir, slug));
    ok(built, `${slug}: o frame existe em dist/`);
    if (built && h) {
      if (heights.has(slug) && heights.get(slug)) {
        ok(Number(h) === heights.get(slug),
          `${slug}: a altura escrita (${h}px) e a medida (${heights.get(slug)}px)`);
      } else {
        skip(`${slug}: altura nao comparada (Edge indisponivel para medir)`);
      }
    }

    /* o frame tem gêmeo estatico no kit (a classe nao usa: decisao do autor) */
    const twinName = slug.replace(/\.html$/, '-estatico.html');
    ok(fs.existsSync(path.join(estaticoDir, twinName)),
      `${slug}: o kit tem o gemeo estatico (${twinName})`);
  }
}

/* ------------------------------------------------------------------ */
(async () => {
  const args = process.argv.slice(2);
  const files = args.length
    ? args.map(a => path.resolve(a))
    : (fs.existsSync(classesDir)
      ? fs.readdirSync(classesDir, { withFileTypes: true })
        .filter(d => d.isDirectory())
        .map(d => path.join(classesDir, d.name, 'aula.md'))
        .filter(f => fs.existsSync(f))
      : []);

  if (!files.length) {
    console.log('nenhuma classe para auditar em noic/classes/');
    process.exit(0);
  }

  for (const f of files) {
    if (!fs.existsSync(f)) { console.log(`\n== ${f} ==\n  FALHA arquivo nao existe`); fails++; continue; }
    await auditClass(f);
  }

  console.log(fails === 0
    ? `\nTUDO OK (${skips} item(ns) pulado(s))`
    : `\n${fails} REGRA(S) DO CHECKLIST §7 FALHARAM`);
  process.exit(fails === 0 ? 0 : 1);
})();
