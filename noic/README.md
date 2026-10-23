# NOIC — Guia ONIA 2026-2027

Material do Departamento de Inteligência Artificial para o *Guia ONIA* (WordPress, `noic.com.br`).

```
noic/
  classes/          uma pasta por aula
    template.html   shell da página de aula (só revisão local)
    build.js        gera o index.html de cada aula
    <slug>/
      aula.md       o texto da aula — a fonte (é o que vai para o WordPress)
      index.html    a página de revisão local (gerada; nunca editar à mão)
  frames/           os frames interativos e seus gêmeos estáticos
    kit/            o kit: theme.css, template.html, kit.js, build.js, check.js,
                    measure.js, check-classe.js, RECEITAS-frame.md, preview.html
    src/            fonte de cada frame (1ª linha: <!--title: ...-->)
    estatico/       o gêmeo estático de cada frame (fica no kit; a classe não cola)
    dist/           HTML autocontido gerado (o que se sobe ao servidor)
  skills/           as metodologias do projeto (as mesmas que o harness carrega do
                     ~/.config/opencode/skills; aqui versionadas no repo)
  _exploracao/      análise, relatório, fórmulas e material-fonte (fora do git por enquanto)
```

## As aulas

| aula | eixo | acento | frames | arquivo |
|---|---|---|---|---|
| Dataset e labels | Eixo 3 — Dados: o Combustível da IA | violeta `#9b51e0` | 3 | `classes/eixo3-dataset-e-rotulos/aula.md` |
| Ciclo de vida dos dados | Eixo 3 — Dados: o Combustível da IA | terracota `#d8613c` | 3 | `classes/eixo3-ciclo-e-qualidade/aula.md` |

> O mapa do Eixo 3 tem 4 aulas (dataset e rótulos · ciclo de vida e qualidade · divisão e
> vazamento · viés e equidade) e o Eixo 7 tem 3. Cada aula é escrita e revisada **uma por vez**.

## Revisar local

```powershell
node noic\classes\build.js      # gera a página de cada aula
python -m http.server 8793 --bind 127.0.0.1 --directory noic
# http://127.0.0.1:8793/classes/eixo3-dataset-e-rotulos/index.html
```

O `aula.md` já está com o `src` do iframe no endereço do WordPress; o build só troca o `src`
na página gerada, para o frame renderizar localmente. No fim, o que entra no site é o
markdown convertido, sem mexer em mais nada.

## O kit (uma vez, para todos os frames)

O frame **não declara cor nenhuma** — só `currentColor` e opacidades. Estado é marcado por
**forma** (traço, contorno, peso, preenchido vs. vazado) e por **glifo** (✓/✗). A cor do eixo
vive no corpo do artigo, nunca dentro do iframe.

```powershell
node noic\frames\kit\build.js         # src/ -> dist/ (determinístico)
node noic\frames\kit\check.js         # confere o kit sem abrir o browser
node noic\frames\kit\measure.js       # mede a altura de cada frame em 620px
node noic\frames\kit\check-classe.js  # portão de publicação: audita cada aula contra o §7
python -m http.server 8792 --bind 127.0.0.1 --directory noic\frames\kit
# http://127.0.0.1:8792/preview.html  - a conferência na largura real do site
```

A **fórmula da classe** (estrutura, linguagem visual, forma do cálculo, checklist de publicação)
está em `_exploracao/TEMPLATE-classe-noic.md`; o **molde de classe com iframe** e as
**4 receitas de frame** estão em `_exploracao/TEMPLATE-classe-iframe.md` e
`frames/kit/RECEITAS-frame.md`.

## As skills (as metodologias, versionadas aqui)

| skill | o que ela é |
|---|---|
| `noic-class-formula` | a forma canônica de uma aula: estrutura, os 5 tempos, a matemática na prosa, o código como caixa, a terminologia, a rubrica de revisão |
| `noic-frame-kit` | a forma de um frame: zero cor declarada, os 4 arquétipos, o gêmeo estático, a altura medida, o contrato do `<iframe>` |
| `heitor-writing-notes-style` | a voz: a ordem da derivação, a cor como sistema, o contraste lado a lado, o cálculo na forma de uma linha |
| `poliedro-textbook-arrangement` | como o livro didático se organiza: espinha de 5 níveis, grade de 2 colunas, os 14 movimentos retóricos, a gramática de imagem |

A cópia do repo é a fonte versionada; a cópia em `~/.config/opencode/skills/` é a que o harness
carrega. Ao mexer em uma, copie para a outra (mesmo nome, mesmo `SKILL.md`).
