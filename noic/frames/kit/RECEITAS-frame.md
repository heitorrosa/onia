# RECEITAS DE FRAME — NOIC / Guia ONIA

> **O que este arquivo é:** as **4 formas** de frame do kit, prontas para copiar.
> Cada receita é o esqueleto exato de um `src/<slug>.html`: o HTML, o JS, e o que
> o gêmeo estático precisa repetir. Copie a receita, troque o conteúdo, escreva o
> gêmeo — pronto.
>
> **Contrato de todo frame** (vale para as 4):
> - 1ª linha: `<!--title: ...-->` (vira o `<title>` do arquivo gerado).
> - o resto é o **corpo** do frame (fragmento HTML) — o shell, o CSS e o JS vêm do kit.
> - **nenhuma cor declarada**: só `currentColor` e opacidades. Estado = forma + glifo.
> - **1 interação por seção**, ≤ 4 por classe, e sempre com um bloco (b)–(e) do texto que a sustenta.
> - **gêmeo estático obrigatório** em `estatico/<slug>-estatico.html`: a mesma ideia em HTML puro. O kit exige o arquivo; a classe não o cola na página (só o iframe vai).
> - o JS usa só a API `window.NOIC` (`reducedMotion`, `degradeMath()`, `cycle()`, `highlight()`).
>
> **Rodar:** `node kit\build.js` → `node kit\check.js` → `node kit\measure.js` (a altura do iframe) → abrir `kit\preview.html`.

---

## A · Simulador de conceito
**1 controle + 1 visual que responde.** Use quando a intuição é uma **relação** que o texto não mostra.
*Exemplo do Eixo 3:* mexer na fração do grupo no treino e ver a acurácia geral subir enquanto a do grupo fica para trás.

**Componentes:** `.sim` (grade) · `.ctl` (controle) · `.gauge` (par de barras) · `.bar` / `.bar.alt` (contínua × tracejada) · `.readout` (número) · `.scale` (marcas do controle).

```html
<!--title: A fração do grupo no treino-->
<div class="wrap">
  <p class="kicker">Fig. 2 — arquétipo A: o simulador</p>
  <h1>O que acontece quando o grupo some do treino</h1>
  <p class="sub">Mova o controle e veja as duas acurácias responderem.</p>

  <div class="sim">
    <div class="ctl">
      <span class="name">fração do grupo no treino</span>
      <span class="val" id="qval">20%</span>
      <input type="range" id="q" min="0" max="100" value="20">
      <div class="scale"><span>0%</span><span>50%</span><span>100%</span></div>
    </div>
    <div class="gauge">
      <div class="row">
        <span class="name">geral</span>
        <span class="bar"><i id="g1" style="width:88%"></i></span>
        <span class="val" id="v1">88%</span>
      </div>
      <div class="row">
        <span class="name">grupo</span>
        <span class="bar alt"><i id="g2" style="width:40%"></i></span>
        <span class="val" id="v2">40%</span>
      </div>
    </div>
  </div>

  <p class="foot">O estado é forma: a barra do grupo é <b>tracejada</b> e mais clara; a geral é contínua e cheia.</p>
</div>

<script>
(function () {
  var q = document.getElementById('q');
  function render() {
    var p = Number(q.value) / 100;
    var geral = 80 + 15 * p;
    var grupo = 10 + 60 * Math.pow(p, 0.7);
    document.getElementById('qval').textContent = q.value + '%';
    document.getElementById('g1').style.width = geral + '%';
    document.getElementById('g2').style.width = grupo + '%';
    document.getElementById('v1').textContent = Math.round(geral) + '%';
    document.getElementById('v2').textContent = Math.round(grupo) + '%';
  }
  q.addEventListener('input', render);
  render();
})();
</script>
```

**Gêmeo estático:** as mesmas duas barras viram **tabela** (valor de `q` → acurácia geral → acurácia do grupo) + a fórmula que liga as três colunas. Sem slider: a tabela é o conteúdo.

---

## B · Workflow animado
**Etapas que se destacam uma a uma.** Use quando a ideia é um **processo**.
*Exemplo do Eixo 7:* Especificar → Gerar → Criticar → Refinar.

**Componentes:** `.flow` (SVG) · `.node` / `.node.on` · `.nlabel` / `.nsub` · `.arrow` / `.arrow.on` · `.mk` (a ponta da seta) · `.loop` (a volta tracejada).

```html
<!--title: O ciclo de trabalho do prompt-->
<div class="wrap">
  <p class="kicker">Fig. 1 — arquétipo B: o ciclo de trabalho</p>
  <h1>Do prompt vago ao prompt verificável</h1>

  <svg class="flow" viewBox="0 0 380 210" role="img" aria-label="Ciclo: Especificar, Gerar, Criticar, Refinar">
    <defs>
      <marker id="ah" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path class="mk" d="M0 0 L10 5 L0 10 z"/>
      </marker>
    </defs>
    <path class="arrow" id="a1" d="M108 42 H 272" marker-end="url(#ah)"/>
    <path class="arrow" id="a2" d="M300 68 V 130" marker-end="url(#ah)"/>
    <path class="arrow" id="a3" d="M272 160 H 108" marker-end="url(#ah)"/>
    <path class="arrow loop" id="a4" d="M20 159 H 10 V 41 H 17" marker-end="url(#ah)"/>
    <g id="n0"><rect class="node" x="20" y="20" width="88" height="42" rx="9"/><text class="nlabel" x="64" y="38">Especificar</text><text class="nsub" x="64" y="52">critérios</text></g>
    <g id="n1"><rect class="node" x="272" y="20" width="88" height="42" rx="9"/><text class="nlabel" x="316" y="38">Gerar</text><text class="nsub" x="316" y="52">saída</text></g>
    <g id="n2"><rect class="node" x="272" y="138" width="88" height="42" rx="9"/><text class="nlabel" x="316" y="156">Criticar</text><text class="nsub" x="316" y="170">contra o critério</text></g>
    <g id="n3"><rect class="node" x="20" y="138" width="88" height="42" rx="9"/><text class="nlabel" x="64" y="156">Refinar</text><text class="nsub" x="64" y="170">o prompt</text></g>
  </svg>
</div>

<script>
(function () {
  var steps = [['n0', 'a1'], ['n1', 'a2'], ['n2', 'a3'], ['n3', 'a4']];
  var idx = 0;
  /* NOIC.cycle para sozinho quando o navegador pede menos movimento:
     a fn roda uma vez e o frame fica estático. */
  NOIC.cycle(function () {
    NOIC.highlight(steps, idx);
    idx = (idx + 1) % steps.length;
  }, 1600);
})();
</script>
```

**Gêmeo estático:** o mesmo ciclo vira uma **`<ol>`** ("1. Especificar — … 2. Gerar — …") com a mesma ordem e os mesmos rótulos.

> **Regra do SVG:** o `id` da seta é opcional — `highlight()` aceita `[idNo, null]`. O `marker-end` precisa de um `id` **único por frame** (aqui `#ah`), porque o id vive dentro do iframe.

---

## C · Verificação de entendimento
**O aluno marca, recebe feedback imediato.** Use no fim de um bloco, para checar a leitura.
*Exemplo do Eixo 7:* "qual destes 4 critérios torna a saída auditável?"

**Componentes:** `.choice` (botão) · `.mark` (o glifo) · `.verdict` (a frase de retorno) · `data-verdict="right|wrong"`.

```html
<!--title: Qual critério torna a saída auditável-->
<div class="wrap">
  <p class="kicker">Fig. 3 — arquétipo C: a checagem</p>
  <h1>Qual destes critérios torna a saída auditável?</h1>

  <button class="choice" data-v="a"><span class="mark">·</span>"Resposta boa" — o professor decide</button>
  <button class="choice" data-v="b"><span class="mark">·</span>Toda afirmação factual é rastreável a um trecho da fonte</button>
  <button class="choice" data-v="c"><span class="mark">·</span>Não invente números que não estejam na fonte</button>
  <button class="choice" data-v="d"><span class="mark">·</span>Use vocabulário acessível</button>

  <p class="verdict" id="vd">Escolha uma opção.</p>
</div>

<script>
(function () {
  var OK = { b: 'Certo — é o único critério que um terceiro consegue conferir sem perguntar ao autor.' };
  var WHY = {
    a: 'Não: sem critério explícito, "boa" é opinião, e opinião não se revisa.',
    c: 'Quase: limita o que inventar, mas não diz como saber o que está certo.',
    d: 'Não: é um critério de estilo. Ajuda a ler, não a auditar.'
  };
  document.querySelectorAll('.choice').forEach(function (b) {
    b.addEventListener('click', function () {
      var v = b.getAttribute('data-v');
      document.querySelectorAll('.choice').forEach(function (o) {
        o.removeAttribute('data-verdict');
        o.querySelector('.mark').textContent = '·';
      });
      b.setAttribute('data-verdict', OK[v] ? 'right' : 'wrong');
      b.querySelector('.mark').textContent = OK[v] ? '✓' : '✗';
      document.getElementById('vd').innerHTML = OK[v] ? '<b>Certo.</b> ' + OK[v] : WHY[v];
    });
  });
})();
</script>
```

**Gêmeo estático:** as 4 opções viram uma **lista com a resposta marcada** ("a) … ✗ · b) … ✓ — porque …"). O glifo e a justificativa bastam; o clique é o que se perde.

> **Nota de estilo:** o `data-verdict` é o que o CSS escuta (`.choice[data-verdict="right"]` fica em negrito; `[data-verdict="wrong"]` fica com opacidade menor). O **glifo** é quem carrega o acerto — nunca a cor.

---

## D · Mapa mental navegável
**O "NO É ASSIM" do Poliedro em SVG clicável.** Use como revisão: liga cada vértice ao trecho da classe.

**Componentes:** `.mind` · `.link` / `.link.on` · `.vtx` / `.vtx.on` · `.vlabel` / `.vsub`.

```html
<!--title: O mapa da aula-->
<div class="wrap">
  <p class="kicker">Fig. 4 — arquétipo D: o mapa</p>
  <h1>O que a aula ensina</h1>

  <svg class="flow mind" viewBox="0 0 380 210" role="img" aria-label="Mapa mental: 4 vértices ligados a um centro">
    <path class="link" id="ml0" d="M150 80 L 130 56"/>
    <path class="link" id="ml1" d="M230 80 L 250 56"/>
    <path class="link" id="ml2" d="M150 124 L 130 156"/>
    <path class="link" id="ml3" d="M230 124 L 250 156"/>
    <g id="mc"><rect class="vtx" x="150" y="80" width="80" height="44" rx="9"/><text class="vlabel" x="190" y="98">O que a aula</text><text class="vlabel" x="190" y="113">ensina</text></g>
    <g id="mv0" tabindex="0" role="button" aria-label="Vértice 1"><rect class="vtx" x="20" y="16" width="110" height="40" rx="9"/><text class="vlabel" x="75" y="33">Especificar</text><text class="vsub" x="75" y="47">os critérios</text></g>
    <g id="mv1" tabindex="0" role="button" aria-label="Vértice 2"><rect class="vtx" x="250" y="16" width="110" height="40" rx="9"/><text class="vlabel" x="305" y="33">Gerar</text><text class="vsub" x="305" y="47">a saída</text></g>
    <g id="mv2" tabindex="0" role="button" aria-label="Vértice 3"><rect class="vtx" x="20" y="156" width="110" height="40" rx="9"/><text class="vlabel" x="75" y="173">Criticar</text><text class="vsub" x="75" y="187">contra o critério</text></g>
    <g id="mv3" tabindex="0" role="button" aria-label="Värtice 4"><rect class="vtx" x="250" y="156" width="110" height="40" rx="9"/><text class="vlabel" x="305" y="173">Refinar</text><text class="vsub" x="305" y="187">o prompt</text></g>
  </svg>
  <p class="verdict" id="mindmsg">Clique num vértice.</p>
</div>

<script>
(function () {
  var pairs = [['mv0', 'ml0'], ['mv1', 'ml1'], ['mv2', 'ml2'], ['mv3', 'ml3']];
  function pick(i) {
    pairs.forEach(function (p, j) {
      document.getElementById(p[0]).querySelector('.vtx').classList.toggle('on', j === i);
      document.getElementById(p[1]).classList.toggle('on', j === i);
    });
    document.getElementById('mindmsg').textContent = 'Vértice ' + (i + 1) + ' da aula.';
  }
  pairs.forEach(function (p, i) {
    var g = document.getElementById(p[0]);
    g.addEventListener('click', function () { pick(i); });
    g.addEventListener('keydown', function (e) {
      if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); pick(i); }
    });
  });
})();
</script>
```

**Gêmeo estático:** o mapa vira uma **tabela** (vértice | o que é | onde aparece na classe). A tabela é mais consultável que o mapa — o mapa é o visual, a tabela é a leitura.

---

## AChoice que resolve o resto

| o que muda entre frames | onde mexer |
|---|---|
| **a forma** (espaçamento, raio, peso, tamanho do tipo) | `kit/theme.css` — e só lá. O build recoloca o CSS em cada frame. |
| **a cor** | **não existe no frame.** A cor do eixo vive no corpo do artigo e no gêmeo estático. |
| **a altura do iframe** | `node kit\measure.js` — **mede**, não estima (620px é a largura real do conteúdo do NOIC). |
| **o conteúdo** | `kit/src/<slug>.html` + `kit/estatico/<slug>-estatico.html` |

**Ordem de trabalho de um frame novo:** copie a receita → escreva o `src/` → escreva o gêmeo em `estatico/` → `node kit\build.js` → `node kit\check.js` → `node kit\measure.js` → olhe em `kit\preview.html` (620px, com o gêmeo abaixo) → cole o bloco `<figure class="noic-frame">` do `TEMPLATE-classe-iframe.md` §3 no WordPress, com a altura medida.
