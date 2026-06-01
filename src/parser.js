export class Parser {
  constructor(tokens) {
    this.tokens = tokens;
    this.pos = 0;
  }

  cur() { return this.tokens[this.pos] || { type: 'EOF', value: '', line: 0, col: 0 }; }
  peek(n = 1) { return this.tokens[this.pos + n] || { type: 'EOF', value: '', line: 0, col: 0 }; }
  advance() { return this.tokens[this.pos++]; }
  check(v) { const t = this.cur(); return t.type === v || t.value === v; }
  match(v) { if (this.check(v)) return this.advance(); return null; }

  expect(v) {
    if (this.check(v)) return this.advance();
    const t = this.cur();
    throw new ParseError(`Expected '${v}', got '${t.value}' (${t.type})`, t.line);
  }

  expectNewline() {
    if (this.check('NEWLINE') || this.check('EOF')) { if (this.check('NEWLINE')) this.advance(); return; }
    if (this.check(';')) { this.advance(); return; }
  }

  skipNL() { while (this.check('NEWLINE')) this.advance(); }

  parseModule() {
    this.skipNL();
    const body = [];
    while (!this.check('EOF')) {
      if (this.check('DEDENT')) { this.advance(); continue; }
      const prevPos = this.pos;
      try { body.push(this.parseStmt()); } catch (e) {
        if (this.pos === prevPos) this.advance();
        while (!this.check('NEWLINE') && !this.check('EOF')) this.advance();
      }
      this.skipNL();
    }
    return { type: 'Module', body };
  }

  parseStmt() {
    const t = this.cur();
    if (t.type === '@') return this.parseDecorated();
    switch (t.value) {
      case 'class': return this.parseClassDef();
      case 'def': return this.parseFuncDef();
      case 'if': return this.parseIf();
      case 'for': return this.parseFor();
      case 'while': return this.parseWhile();
      case 'return': return this.parseReturn();
      case 'pass': this.advance(); this.expectNewline(); return { type: 'Pass' };
      case 'break': this.advance(); this.expectNewline(); return { type: 'Break' };
      case 'continue': this.advance(); this.expectNewline(); return { type: 'Continue' };
      case 'import': return this.parseImport();
      case 'from': return this.parseFromImport();
      case 'del': return this.parseDel();
      case 'raise': return this.parseRaise();
      case 'assert': return this.parseAssert();
      case 'with': return this.parseWith();
      case 'try': return this.parseTry();
    }
    return this.parseAssignOrExpr();
  }

  parseBlock() {
    if (!this.check('NEWLINE')) {
      const body = [];
      body.push(this.parseStmt());
      return body;
    }
    this.expect('NEWLINE');
    this.skipNL();
    this.expect('INDENT');
    const body = [];
    this.skipNL();
    while (!this.check('DEDENT') && !this.check('EOF')) {
      const prevPos = this.pos;
      try { body.push(this.parseStmt()); } catch (e) {
        if (this.pos === prevPos) this.advance();
        while (!this.check('NEWLINE') && !this.check('EOF') && !this.check('DEDENT')) this.advance();
      }
      this.skipNL();
      if (this.pos === prevPos) this.advance();
    }
    if (this.check('DEDENT')) this.advance();
    return body;
  }

  parseDecorated() {
    const decorators = [];
    while (this.match('@')) {
      decorators.push(this.parseExpr());
      this.expectNewline();
      this.skipNL();
    }
    const stmt = this.parseStmt();
    stmt.decorators = decorators;
    return stmt;
  }

  parseClassDef() {
    const line = this.cur().line;
    this.expect('class');
    const name = this.expect('NAME').value;
    let bases = [];
    if (this.match('(')) {
      if (!this.check(')')) bases = this.parseCommaSep(() => this.parseExpr(), ')');
      this.expect(')');
    }
    this.expect(':');
    const body = this.parseBlock();
    return { type: 'ClassDef', name, bases, body, decorators: [], line };
  }

  parseFuncDef() {
    const line = this.cur().line;
    this.expect('def');
    const name = this.expect('NAME').value;
    this.expect('(');
    const params = this.parseParams();
    this.expect(')');
    let returnType = null;
    if (this.match('->')) returnType = this.parseExpr();
    this.expect(':');
    const body = this.parseBlock();
    return { type: 'FunctionDef', name, params, body, returnType, decorators: [], line };
  }

  parseParams() {
    const params = [];
    while (!this.check(')') && !this.check('EOF')) {
      if (params.length > 0) this.expect(',');
      if (this.check(')')) break;
      let kind = 'positional';
      if (this.match('**')) kind = 'kwargs';
      else if (this.match('*')) {
        if (this.check(',') || this.check(')')) { params.push({ kind: 'separator' }); continue; }
        kind = 'varargs';
      }
      const name = this.expect('NAME').value;
      let annotation = null, dflt = null;
      if (this.match(':')) annotation = this.parseExpr();
      if (this.match('=')) dflt = this.parseExpr();
      params.push({ kind, name, annotation, default: dflt });
    }
    return params;
  }

  parseIf() {
    if (this.check('elif')) this.advance(); else this.expect('if');
    const test = this.parseExpr();
    this.expect(':');
    const body = this.parseBlock();
    this.skipNL();
    let orelse = [];
    if (this.check('elif')) orelse = [this.parseIf()];
    else if (this.match('else')) { this.expect(':'); orelse = this.parseBlock(); }
    return { type: 'If', test, body, orelse };
  }

  parseFor() {
    const line = this.cur().line;
    this.expect('for');
    const target = this.parseForTarget();
    this.expect('in');
    const iter = this.parseExpr();
    this.expect(':');
    const body = this.parseBlock();
    return { type: 'For', target, iter, body, line };
  }

  parseForTarget() {
    const first = this.parseExpr(5);
    if (this.check(',') && !this.check('in')) {
      const elts = [first];
      while (this.match(',')) {
        if (this.check('in')) break;
        elts.push(this.parseExpr(5));
      }
      return { type: 'Tuple', elts };
    }
    return first;
  }

  parseWhile() {
    this.expect('while');
    const test = this.parseExpr();
    this.expect(':');
    const body = this.parseBlock();
    return { type: 'While', test, body };
  }

  parseReturn() {
    const line = this.cur().line;
    this.expect('return');
    let value = null;
    if (!this.check('NEWLINE') && !this.check('EOF')) value = this.parseExpr();
    this.expectNewline();
    return { type: 'Return', value, line };
  }

  parseImport() {
    this.expect('import');
    const names = [];
    do {
      let name = this.expect('NAME').value;
      while (this.match('.')) name += '.' + this.expect('NAME').value;
      let alias = null;
      if (this.match('as')) alias = this.expect('NAME').value;
      names.push({ name, alias });
    } while (this.match(','));
    this.expectNewline();
    return { type: 'Import', names };
  }

  parseFromImport() {
    this.expect('from');
    let module = '';
    while (this.match('.')) module += '.';
    if (this.check('NAME') && this.cur().value !== 'import') {
      module += this.expect('NAME').value;
      while (this.match('.')) module += '.' + this.expect('NAME').value;
    }
    this.expect('import');
    if (this.match('*')) { this.expectNewline(); return { type: 'ImportFrom', module, names: [{ name: '*', alias: null }] }; }
    const paren = !!this.match('(');
    const names = [];
    do {
      if (this.check(')')) break;
      const name = this.expect('NAME').value;
      let alias = null;
      if (this.match('as')) alias = this.expect('NAME').value;
      names.push({ name, alias });
    } while (this.match(','));
    if (paren) this.expect(')');
    this.expectNewline();
    return { type: 'ImportFrom', module, names };
  }

  parseDel() { this.expect('del'); const t = [this.parseExpr()]; while (this.match(',')) t.push(this.parseExpr()); this.expectNewline(); return { type: 'Delete', targets: t }; }
  parseRaise() { this.expect('raise'); let v = null; if (!this.check('NEWLINE') && !this.check('EOF')) v = this.parseExpr(); this.expectNewline(); return { type: 'Raise', value: v }; }
  parseAssert() { this.expect('assert'); const t = this.parseExpr(); let m = null; if (this.match(',')) m = this.parseExpr(); this.expectNewline(); return { type: 'Assert', test: t, msg: m }; }

  parseWith() {
    this.expect('with');
    const items = [];
    do {
      const ctx = this.parseExpr(); let v = null;
      if (this.match('as')) v = this.parseExpr();
      items.push({ context: ctx, variable: v });
    } while (this.match(','));
    this.expect(':');
    return { type: 'With', items, body: this.parseBlock() };
  }

  parseTry() {
    this.expect('try'); this.expect(':');
    const body = this.parseBlock(); this.skipNL();
    const handlers = [];
    while (this.check('except')) {
      this.advance(); let tp = null, nm = null;
      if (!this.check(':')) { tp = this.parseExpr(); if (this.match('as')) nm = this.expect('NAME').value; }
      this.expect(':');
      handlers.push({ type: tp, name: nm, body: this.parseBlock() });
      this.skipNL();
    }
    let finalbody = null;
    if (this.match('finally')) { this.expect(':'); finalbody = this.parseBlock(); }
    return { type: 'Try', body, handlers, finalbody };
  }

  parseAssignOrExpr() {
    const line = this.cur().line;
    const first = this.parseExpr();

    if (this.check(',') && !this.check('NEWLINE') && !this.check('EOF')) {
      const elts = [first];
      while (this.match(',')) {
        if (this.check('=') || this.check('NEWLINE') || this.check('EOF') || this.check(';')) break;
        elts.push(this.parseExpr());
      }
      const tuple = elts.length === 1 ? elts[0] : { type: 'Tuple', elts };
      if (this.check('=') && !this.check('==')) {
        this.advance();
        const value = this.parseAssignValue();
        this.expectNewline();
        return { type: 'Assign', target: tuple, value, line };
      }
      this.expectNewline();
      return { type: 'ExprStmt', value: tuple, line };
    }

    if (this.check('=') && !this.check('==')) {
      this.advance();
      const value = this.parseAssignValue();
      this.expectNewline();
      return { type: 'Assign', target: first, value, line };
    }
    const augOps = ['+=', '-=', '*=', '/=', '//=', '%=', '**='];
    for (const op of augOps) {
      if (this.check(op)) { this.advance(); const v = this.parseExpr(); this.expectNewline(); return { type: 'AugAssign', target: first, op, value: v, line }; }
    }
    if (this.check(':') && !this.check('NEWLINE')) {
      this.advance(); this.parseExpr();
      if (this.check('=')) { this.advance(); const v = this.parseExpr(); this.expectNewline(); return { type: 'Assign', target: first, value: v, line }; }
    }
    this.expectNewline();
    return { type: 'ExprStmt', value: first, line };
  }

  parseAssignValue() {
    const first = this.parseExpr();
    if (this.check(',') && !this.check('NEWLINE') && !this.check('EOF') && !this.check(')') && !this.check(']')) {
      const elts = [first];
      while (this.match(',')) {
        if (this.check('NEWLINE') || this.check('EOF')) break;
        elts.push(this.parseExpr());
      }
      return { type: 'Tuple', elts };
    }
    return first;
  }

  // ─── Pratt expression parser ──────────────────────────

  parseExpr(minBP = 0) {
    let left = this.parseUnary();
    while (true) {
      const bp = this.infixBP();
      if (bp <= minBP) break;
      const t = this.cur();

      if (t.value === 'if' && bp > minBP) {
        this.advance();
        const test = this.parseExpr();
        this.expect('else');
        left = { type: 'IfExpr', body: left, test, orelse: this.parseExpr(1) };
        continue;
      }
      if (t.value === 'or') { this.advance(); left = { type: 'BoolOp', op: 'or', values: [left, this.parseExpr(2)] }; continue; }
      if (t.value === 'and') { this.advance(); left = { type: 'BoolOp', op: 'and', values: [left, this.parseExpr(3)] }; continue; }
      if (this.isCompareOp()) { left = this.parseCompareChain(left); continue; }

      const op = this.advance().type;
      left = { type: 'BinOp', op, left, right: this.parseExpr(bp) };
    }
    return left;
  }

  infixBP() {
    const t = this.cur();
    if (t.value === 'if') return 1;
    if (t.value === 'or') return 2;
    if (t.value === 'and') return 3;
    if (this.isCompareOp()) return 4;
    if (t.type === '|') return 5;
    if (t.type === '^') return 6;
    if (t.type === '&') return 7;
    if (t.type === '<<' || t.type === '>>') return 8;
    if (t.type === '+' || t.type === '-') return 9;
    if (t.type === '*' || t.type === '/' || t.type === '//' || t.type === '%') return 10;
    return 0;
  }

  isCompareOp() {
    const t = this.cur();
    return ['==', '!=', '<', '>', '<=', '>='].includes(t.type) || t.value === 'in' || t.value === 'is'
      || (t.value === 'not' && this.peek().value === 'in');
  }

  parseCompareChain(left) {
    const ops = [], comparators = [];
    while (this.isCompareOp()) {
      const t = this.cur();
      let op;
      if (t.value === 'not') { this.advance(); this.expect('in'); op = 'not in'; }
      else if (t.value === 'is') { this.advance(); op = this.match('not') ? 'is not' : 'is'; }
      else if (t.value === 'in') { this.advance(); op = 'in'; }
      else { op = this.advance().type; }
      ops.push(op);
      comparators.push(this.parseExpr(4));
    }
    return { type: 'Compare', left, ops, comparators };
  }

  parseUnary() {
    const t = this.cur();
    if (t.value === 'not') { this.advance(); return { type: 'UnaryOp', op: 'not', operand: this.parseExpr(3) }; }
    if (t.type === '-' || t.type === '+' || t.type === '~') { this.advance(); return { type: 'UnaryOp', op: t.type, operand: this.parseUnary() }; }
    if (t.value === 'lambda') return this.parseLambda();
    return this.parsePower();
  }

  parsePower() {
    let base = this.parsePrimary();
    if (this.match('**')) return { type: 'BinOp', op: '**', left: base, right: this.parseUnary() };
    return base;
  }

  parsePrimary() {
    let expr = this.parseAtom();
    while (true) {
      if (this.check('(')) expr = this.parseCallExpr(expr);
      else if (this.check('[')) expr = this.parseSubscriptExpr(expr);
      else if (this.check('.')) { this.advance(); expr = { type: 'Attribute', value: expr, attr: this.expect('NAME').value }; }
      else break;
    }
    return expr;
  }

  parseAtom() {
    const t = this.cur();
    if (t.type === 'NUMBER') { this.advance(); return { type: 'Num', value: t.value.includes('.') || /e/i.test(t.value) ? parseFloat(t.value) : parseInt(t.value, 10) }; }
    if (t.type === 'STRING') {
      this.advance(); let v = t.value;
      while (this.check('STRING')) v += this.advance().value;
      return { type: 'Str', value: v };
    }
    if (t.type === 'NAME') {
      if (t.value === 'True') { this.advance(); return { type: 'Bool', value: true }; }
      if (t.value === 'False') { this.advance(); return { type: 'Bool', value: false }; }
      if (t.value === 'None') { this.advance(); return { type: 'NoneValue' }; }
      if (t.value === 'lambda') return this.parseLambda();
      this.advance(); return { type: 'Name', id: t.value };
    }
    if (t.type === '(') {
      this.advance();
      if (this.check(')')) { this.advance(); return { type: 'Tuple', elts: [] }; }
      const first = this.parseExpr();
      if (this.cur().value === 'for') { const r = this.parseComprehension(first); this.expect(')'); return r; }
      if (this.match(',')) {
        const elts = [first];
        while (!this.check(')') && !this.check('EOF')) { elts.push(this.parseExpr()); if (!this.match(',')) break; }
        this.expect(')'); return { type: 'Tuple', elts };
      }
      this.expect(')'); return first;
    }
    if (t.type === '[') {
      this.advance();
      if (this.check(']')) { this.advance(); return { type: 'List', elts: [] }; }
      const first = this.parseExpr();
      if (this.cur().value === 'for') { const r = this.parseComprehension(first); this.expect(']'); return r; }
      const elts = [first];
      while (this.match(',')) { if (this.check(']')) break; elts.push(this.parseExpr()); }
      this.expect(']'); return { type: 'List', elts };
    }
    if (t.type === '{') {
      this.advance();
      if (this.check('}')) { this.advance(); return { type: 'Dict', keys: [], values: [] }; }
      const first = this.parseExpr();
      if (this.match(':')) {
        const keys = [first], values = [this.parseExpr()];
        while (this.match(',')) { if (this.check('}')) break; keys.push(this.parseExpr()); this.expect(':'); values.push(this.parseExpr()); }
        this.expect('}'); return { type: 'Dict', keys, values };
      }
      const elts = [first];
      while (this.match(',')) { if (this.check('}')) break; elts.push(this.parseExpr()); }
      this.expect('}'); return { type: 'Set', elts };
    }
    if (t.type === '...') { this.advance(); return { type: 'Ellipsis' }; }
    throw new ParseError(`Unexpected token: '${t.value}' (${t.type})`, t.line);
  }

  parseCallExpr(func) {
    this.expect('(');
    const args = [], kwargs = [];
    while (!this.check(')') && !this.check('EOF')) {
      if (args.length + kwargs.length > 0) { this.expect(','); if (this.check(')')) break; }
      if (this.check('**')) { this.advance(); kwargs.push({ key: null, value: this.parseExpr() }); }
      else if (this.check('*') && this.peek().type !== '*') { this.advance(); args.push({ type: 'Starred', value: this.parseExpr() }); }
      else {
        const expr = this.parseExpr();
        if (this.check('=') && !this.check('==') && expr.type === 'Name') { this.advance(); kwargs.push({ key: expr.id, value: this.parseExpr() }); }
        else args.push(expr);
      }
    }
    this.expect(')');
    return { type: 'Call', func, args, kwargs };
  }

  parseSubscriptExpr(value) {
    this.expect('[');
    const first = this.parseSubscriptElement();
    if (this.check(',')) {
      const dims = [first];
      while (this.match(',')) { if (this.check(']')) break; dims.push(this.parseSubscriptElement()); }
      this.expect(']');
      return { type: 'Subscript', value, slice: { type: 'Tuple', elts: dims } };
    }
    this.expect(']');
    return { type: 'Subscript', value, slice: first };
  }

  parseSubscriptElement() {
    if (this.check(':')) return this.parseSlice(null);
    if (this.check('...')) { this.advance(); return { type: 'Ellipsis' }; }
    const expr = this.parseExpr();
    if (this.check(':')) return this.parseSlice(expr);
    return expr;
  }

  parseSlice(lower) {
    this.expect(':');
    let upper = null, step = null;
    if (!this.check(':') && !this.check(']') && !this.check(',') && !this.check(')')) upper = this.parseExpr();
    if (this.match(':')) { if (!this.check(']') && !this.check(',')) step = this.parseExpr(); }
    return { type: 'Slice', lower, upper, step };
  }

  parseLambda() {
    this.expect('lambda');
    const params = [];
    while (!this.check(':') && !this.check('EOF')) {
      if (params.length > 0) this.expect(',');
      const name = this.expect('NAME').value;
      let dflt = null;
      if (this.match('=')) dflt = this.parseExpr();
      params.push({ kind: 'positional', name, default: dflt });
    }
    this.expect(':');
    return { type: 'Lambda', params, body: this.parseExpr() };
  }

  parseComprehension(elt) {
    const generators = [];
    while (this.cur().value === 'for') {
      this.advance();
      const target = this.parseForTarget();
      this.expect('in');
      const iter = this.parseExpr();
      const ifs = [];
      while (this.cur().value === 'if') { this.advance(); ifs.push(this.parseExpr()); }
      generators.push({ target, iter, ifs });
    }
    return { type: 'ListComp', elt, generators };
  }

  parseCommaSep(parseFn, closer) {
    const items = [];
    while (!this.check(closer) && !this.check('EOF')) {
      if (items.length > 0) this.expect(',');
      if (this.check(closer)) break;
      items.push(parseFn());
    }
    return items;
  }
}

class ParseError extends Error {
  constructor(msg, line) { super(`Line ${line}: ${msg}`); this.line = line; }
}
