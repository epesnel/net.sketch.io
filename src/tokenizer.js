export class Token {
  constructor(type, value, line, col) {
    this.type = type;
    this.value = value;
    this.line = line;
    this.col = col;
  }
}

export class Tokenizer {
  constructor(source) {
    this.source = source.replace(/\r\n/g, '\n').replace(/\r/g, '\n').replace(/\\\n/g, ' ');
    this.pos = 0;
    this.line = 1;
    this.col = 0;
    this.tokens = [];
    this.indentStack = [0];
    this.parenDepth = 0;
    this.atLineStart = true;
  }

  get ch() { return this.pos < this.source.length ? this.source[this.pos] : '\0'; }
  get nextCh() { return this.pos + 1 < this.source.length ? this.source[this.pos + 1] : '\0'; }

  advance() {
    const ch = this.source[this.pos++];
    if (ch === '\n') { this.line++; this.col = 0; } else { this.col++; }
    return ch;
  }

  emit(type, value) {
    this.tokens.push(new Token(type, value, this.line, this.col));
  }

  tokenize() {
    while (this.pos < this.source.length) {
      if (this.atLineStart && this.parenDepth === 0) {
        this.processIndentation();
        if (this.atLineStart) continue;
      }
      if (this.pos < this.source.length) this.readNext();
    }
    const last = this.tokens[this.tokens.length - 1];
    if (last && last.type !== 'NEWLINE' && last.type !== 'INDENT' && last.type !== 'DEDENT') {
      this.emit('NEWLINE', '\n');
    }
    while (this.indentStack.length > 1) {
      this.indentStack.pop();
      this.emit('DEDENT', '');
    }
    this.emit('EOF', '');
    return this.tokens;
  }

  processIndentation() {
    let indent = 0;
    while (this.pos < this.source.length) {
      if (this.ch === ' ') { indent++; this.pos++; this.col++; }
      else if (this.ch === '\t') { indent = (Math.floor(indent / 4) + 1) * 4; this.pos++; this.col++; }
      else break;
    }
    if (this.pos >= this.source.length || this.ch === '\n' || this.ch === '#') {
      this.atLineStart = true;
      if (this.ch === '\n') this.advance();
      else if (this.ch === '#') {
        while (this.pos < this.source.length && this.ch !== '\n') this.advance();
        if (this.pos < this.source.length) this.advance();
      }
      return;
    }
    this.atLineStart = false;
    const top = this.indentStack[this.indentStack.length - 1];
    if (indent > top) {
      this.indentStack.push(indent);
      this.emit('INDENT', '');
    } else if (indent < top) {
      while (this.indentStack.length > 1 && this.indentStack[this.indentStack.length - 1] > indent) {
        this.indentStack.pop();
        this.emit('DEDENT', '');
      }
    }
  }

  readNext() {
    while (this.pos < this.source.length && (this.ch === ' ' || this.ch === '\t')) {
      this.pos++; this.col++;
    }
    if (this.pos >= this.source.length) return;
    const ch = this.ch;
    if (ch === '#') { while (this.pos < this.source.length && this.ch !== '\n') this.advance(); return; }
    if (ch === '\n') {
      this.advance();
      if (this.parenDepth === 0) {
        const last = this.tokens[this.tokens.length - 1];
        if (last && last.type !== 'NEWLINE' && last.type !== 'INDENT' && last.type !== 'DEDENT') {
          this.emit('NEWLINE', '\n');
        }
        this.atLineStart = true;
      }
      return;
    }
    if (ch === '"' || ch === "'") { this.readString(); return; }
    if (ch >= '0' && ch <= '9') { this.readNumber(); return; }
    if (this.isNameStart(ch)) { this.readName(); return; }
    this.readOperator();
  }

  isNameStart(ch) { return (ch >= 'a' && ch <= 'z') || (ch >= 'A' && ch <= 'Z') || ch === '_'; }
  isNamePart(ch) { return this.isNameStart(ch) || (ch >= '0' && ch <= '9'); }

  readString() {
    const startLine = this.line, startCol = this.col;
    const quote = this.ch;
    this.advance();
    let triple = false;
    if (this.ch === quote && this.nextCh === quote) { triple = true; this.advance(); this.advance(); }
    let value = '';
    while (this.pos < this.source.length) {
      if (this.ch === '\\') {
        this.advance();
        if (this.pos < this.source.length) {
          const esc = this.ch; this.advance();
          switch (esc) {
            case 'n': value += '\n'; break; case 't': value += '\t'; break;
            case 'r': value += '\r'; break; case '\\': value += '\\'; break;
            case "'": value += "'"; break; case '"': value += '"'; break;
            default: value += esc;
          }
        }
      } else if (triple && this.ch === quote && this.nextCh === quote && this.source[this.pos + 2] === quote) {
        this.advance(); this.advance(); this.advance(); break;
      } else if (!triple && this.ch === quote) {
        this.advance(); break;
      } else if (!triple && this.ch === '\n') {
        break;
      } else {
        value += this.ch; this.advance();
      }
    }
    this.tokens.push(new Token('STRING', value, startLine, startCol));
  }

  readNumber() {
    const startLine = this.line, startCol = this.col;
    let value = '';
    while (this.pos < this.source.length && this.ch >= '0' && this.ch <= '9') { value += this.ch; this.advance(); }
    if (this.ch === '.' && this.nextCh !== '.') { value += '.'; this.advance(); while (this.pos < this.source.length && this.ch >= '0' && this.ch <= '9') { value += this.ch; this.advance(); } }
    if (this.ch === 'e' || this.ch === 'E') {
      value += this.ch; this.advance();
      if (this.ch === '+' || this.ch === '-') { value += this.ch; this.advance(); }
      while (this.pos < this.source.length && this.ch >= '0' && this.ch <= '9') { value += this.ch; this.advance(); }
    }
    this.tokens.push(new Token('NUMBER', value, startLine, startCol));
  }

  readName() {
    const startLine = this.line, startCol = this.col;
    let value = '';
    while (this.pos < this.source.length && this.isNamePart(this.ch)) { value += this.ch; this.advance(); }
    if (/^[fFbBrRuU]{1,2}$/.test(value) && (this.ch === '"' || this.ch === "'")) { this.readString(); return; }
    this.tokens.push(new Token('NAME', value, startLine, startCol));
  }

  readOperator() {
    const startLine = this.line, startCol = this.col;
    const ch = this.ch, next = this.nextCh;
    let type;
    switch (ch) {
      case '(': case ')': case '[': case ']': case '{': case '}':
        type = ch; this.advance();
        if ('([{'.includes(ch)) this.parenDepth++;
        if (')]}'.includes(ch)) this.parenDepth = Math.max(0, this.parenDepth - 1);
        break;
      case ':': case ',': case ';': case '~': case '@':
        type = ch; this.advance(); break;
      case '+': type = next === '=' ? (this.advance(), this.advance(), '+=') : (this.advance(), '+'); break;
      case '-':
        if (next === '=') { this.advance(); this.advance(); type = '-='; }
        else if (next === '>') { this.advance(); this.advance(); type = '->'; }
        else { this.advance(); type = '-'; }
        break;
      case '*':
        if (next === '*') { this.advance(); this.advance(); type = '**'; }
        else if (next === '=') { this.advance(); this.advance(); type = '*='; }
        else { this.advance(); type = '*'; }
        break;
      case '/':
        if (next === '/') { this.advance(); this.advance(); type = '//'; }
        else if (next === '=') { this.advance(); this.advance(); type = '/='; }
        else { this.advance(); type = '/'; }
        break;
      case '%':
        type = next === '=' ? (this.advance(), this.advance(), '%=') : (this.advance(), '%'); break;
      case '=':
        type = next === '=' ? (this.advance(), this.advance(), '==') : (this.advance(), '='); break;
      case '!':
        type = next === '=' ? (this.advance(), this.advance(), '!=') : (this.advance(), '!'); break;
      case '<':
        if (next === '=') { this.advance(); this.advance(); type = '<='; }
        else if (next === '<') { this.advance(); this.advance(); type = '<<'; }
        else { this.advance(); type = '<'; }
        break;
      case '>':
        if (next === '=') { this.advance(); this.advance(); type = '>='; }
        else if (next === '>') { this.advance(); this.advance(); type = '>>'; }
        else { this.advance(); type = '>'; }
        break;
      case '.':
        if (next === '.' && this.source[this.pos + 2] === '.') { this.advance(); this.advance(); this.advance(); type = '...'; }
        else { this.advance(); type = '.'; }
        break;
      case '|': type = '|'; this.advance(); break;
      case '&': type = '&'; this.advance(); break;
      case '^': type = '^'; this.advance(); break;
      default: this.advance(); return;
    }
    this.tokens.push(new Token(type, type, startLine, startCol));
  }
}
