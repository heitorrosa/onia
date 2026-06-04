const fs = require('fs');
const path = require('path');

const dir = path.dirname(__filename || '.');
const template = fs.readFileSync(path.join(dir, 'template.html'), 'utf8');
const examMd = fs.readFileSync(path.join(dir, 'exam.md'), 'utf8');

// Escape for JS template literal
const safe = examMd
  .replace(/\\/g, '\\\\')
  .replace(/`/g, '\\`')
  .replace(/\$/g, '\\$');

const html = template.replace('EMBEDDED_MD_PLACEHOLDER', safe);
fs.writeFileSync(path.join(dir, 'exam.html'), html, 'utf8');
console.log(`Built exam.html: ${html.length} bytes`);
