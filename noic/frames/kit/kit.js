/* ============================================================================
   NOIC FRAME KIT — JS compartilhado
   ----------------------------------------------------------------------------
   Só o que os frames realmente precisam:
     1. NOIC.degradeMath() — degradação do MathJax (se o CDN não responder, a
                      fórmula continua legível em texto);
     2. NOIC.reducedMotion — respeitar prefers-reduced-motion;
     3. NOIC.cycle(fn, ms) e NOIC.highlight(steps, i) — o ciclo de etapas
                      do diagrama, com a seta de estado aceso.
   Sem dependência, sem build, sem rede além do MathJax.
   ========================================================================== */
(function (global) {
  'use strict';

  var reduced = global.matchMedia && global.matchMedia('(prefers-reduced-motion: reduce)').matches;

  /* Degradação do MathJax ------------------------------------------------- */
  /* Se o CDN não respondeu em 2.5s, cada container mjx vira texto plano com
     a expressão original entre \( \) — legível, sem depender de rede. */
  function degradeMath() {
    if (global.MathJax && global.MathJax.typesetPromise) return;   /* MathJax carregou */
    var nodes = document.querySelectorAll('mjx-container');
    for (var i = 0; i < nodes.length; i++) {
      var src = nodes[i].querySelector('[data-latex]') || nodes[i];
      var tex = src.getAttribute && src.getAttribute('data-latex');
      if (!tex) tex = nodes[i].textContent;
      var span = document.createElement('code');
      span.textContent = tex;
      span.style.fontFamily = 'ui-monospace, Consolas, monospace';
      span.style.fontSize = '13px';
      nodes[i].replaceWith(span);
    }
  }
  global.setTimeout(degradeMath, 2500);

  /* Preferência de movimento ------------------------------------------------ */
  function cycle(fn, ms) {
    if (reduced) { fn(0); return null; }
    return global.setInterval(function () { fn(Date.now()); }, ms);
  }

  global.NOIC = {
    reducedMotion: reduced,
    degradeMath: degradeMath,
    cycle: cycle,
    /* marca o nó ativo de um diagrama de etapas: [ [idNo, idSeta], ... ]
       Como o frame não declara cor, "ativo" é Forma: o nó ganha traço mais
       grosso e um preenchimento de 12% da tinta; a seta, idem. */
    highlight: function (steps, index) {
      steps.forEach(function (s) {
        var n = document.getElementById(s[0]);
        var a = document.getElementById(s[1]);
        if (n) { n.querySelector('.node').classList.remove('on'); }
        if (a) { a.classList.remove('on'); }
      });
      var cur = steps[index % steps.length];
      var cn = document.getElementById(cur[0]);
      var ca = document.getElementById(cur[1]);
      if (cn) { cn.querySelector('.node').classList.add('on'); }
      if (ca) { ca.classList.add('on'); }
    }
  };
})(window);
