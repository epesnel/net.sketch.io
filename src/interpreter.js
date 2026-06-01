const KNOWN_LAYERS = new Set([
  'Conv2D', 'Conv1D', 'Conv3D', 'Conv2DTranspose', 'Conv1DTranspose',
  'DepthwiseConv2D', 'SeparableConv2D',
  'Dense', 'MaxPooling2D', 'MaxPooling1D', 'AveragePooling2D', 'AveragePooling1D',
  'GlobalAveragePooling2D', 'GlobalMaxPooling2D', 'GlobalAveragePooling1D',
  'UpSampling2D', 'UpSampling1D',
  'BatchNormalization', 'LayerNormalization', 'GroupNormalization',
  'Dropout', 'SpatialDropout2D',
  'Flatten', 'Reshape', 'Permute', 'RepeatVector',
  'Activation', 'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'Softmax',
  'Add', 'Concatenate', 'Multiply', 'Average', 'Subtract',
  'Embedding', 'LSTM', 'GRU', 'SimpleRNN', 'Bidirectional', 'TimeDistributed',
  'MultiHeadAttention', 'Input',
  'ZeroPadding2D', 'Cropping2D', 'ZeroPadding1D',
  'Lambda',
]);

const IDENTITY_LAYERS = new Set([
  'BatchNormalization', 'LayerNormalization', 'GroupNormalization',
  'Dropout', 'SpatialDropout2D', 'Activation', 'ReLU', 'LeakyReLU', 'PReLU', 'ELU', 'Softmax',
]);

export class Interpreter {
  constructor() {
    this.graph = { nodes: [], edges: [] };
    this.nodeId = 0;
    this.warnings = [];
  }

  analyze(ast, inputShape = [1, 256, 256, 3]) {
    this.graph = { nodes: [], edges: [] };
    this.nodeId = 0;
    this.warnings = [];

    const globalEnv = this.createEnv(null);
    this.registerBuiltins(globalEnv);

    for (const stmt of ast.body) {
      if (stmt.type === 'Import' || stmt.type === 'ImportFrom') {
        this.processImport(stmt, globalEnv);
      }
    }

    const allClasses = [];
    for (const stmt of ast.body) {
      if (stmt.type === 'ClassDef') {
        globalEnv.bindings.set(stmt.name, { kind: 'classDef', def: stmt });
        allClasses.push(stmt);
      }
    }

    let modelClass = null;
    let ctorArgs = {};

    for (const stmt of ast.body) {
      if (stmt.type === 'Assign' && stmt.value?.type === 'Call') {
        const fn = this.flattenName(stmt.value.func);
        const match = allClasses.find(c => c.name === fn);
        if (match) {
          modelClass = match;
          ctorArgs = this.extractCallKwargs(stmt.value, globalEnv);
        }
      }
    }

    if (!modelClass) {
      for (const c of allClasses) {
        for (const base of c.bases) {
          const bn = this.flattenName(base);
          if (bn && ['Model', 'tf.keras.Model', 'keras.Model', 'Layer', 'tf.keras.layers.Layer'].includes(bn)) {
            modelClass = c;
          }
        }
      }
    }
    if (!modelClass && allClasses.length > 0) modelClass = allClasses[allClasses.length - 1];
    if (!modelClass) { this.warnings.push('No model class found'); return this.graph; }

    const initFn = modelClass.body.find(s => s.type === 'FunctionDef' && s.name === '__init__');
    if (initFn && Object.keys(ctorArgs).length === 0) {
      for (const p of initFn.params) {
        if (p.name === 'self' || p.name === 'name' || p.kind === 'kwargs' || p.kind === 'varargs') continue;
        if (!p.default) {
          const n = p.name.toLowerCase();
          if (n.includes('ratio') || n.includes('scale')) ctorArgs[p.name] = { kind: 'scalar', value: 0.5 };
          else if (n.includes('filter') || n.includes('channel') || n.includes('dim') || n === 'bf') ctorArgs[p.name] = { kind: 'scalar', value: 16 };
          else if (n.includes('stage') || n.includes('depth') || n.includes('block')) ctorArgs[p.name] = { kind: 'scalar', value: 3 };
          else ctorArgs[p.name] = { kind: 'scalar', value: 1 };
        }
      }
    }

    const self = { kind: 'self', attrs: new Map(), classDef: modelClass };
    if (initFn) {
      const env = this.createEnv(globalEnv);
      env.bindings.set('self', self);
      this.bindParams(initFn.params, ctorArgs, env);
      this.execBlock(initFn.body, env);
    }

    const callFn = modelClass.body.find(s => s.type === 'FunctionDef' && (s.name === 'call' || s.name === 'forward'));
    if (!callFn) { this.warnings.push('No call/forward method found'); return this.graph; }

    const inputNode = this.addNode('Input', 'Input', {}, [], inputShape);
    const inputTensor = { kind: 'tensor', shape: [...inputShape], nodeId: inputNode.id };
    const callEnv = this.createEnv(globalEnv);
    callEnv.bindings.set('self', self);
    const xParam = callFn.params.find(p => p.name !== 'self');
    if (xParam) callEnv.bindings.set(xParam.name, inputTensor);
    for (const p of callFn.params) {
      if (p.name === 'self' || p.name === xParam?.name) continue;
      if (p.default) callEnv.bindings.set(p.name, this.evalExpr(p.default, callEnv));
    }

    const result = this.execBlock(callFn.body, callEnv);
    let outTensor = null;
    if (result?.kind === 'tensor') outTensor = result;
    else if (result?.kind === 'tuple' && result.items) {
      outTensor = result.items.find(i => i?.kind === 'tensor');
    } else if (result?.kind === 'dict') {
      outTensor = result.values?.find(v => v?.kind === 'tensor');
    }
    if (outTensor && outTensor.nodeId) {
      const outNode = this.addNode('Output', 'Output', {}, [outTensor.shape], outTensor.shape);
      this.graph.edges.push({ from: outTensor.nodeId, to: outNode.id });
    }

    return this.graph;
  }

  // ─── Environment ──────────────────────────────────────

  createEnv(parent) { return { parent, bindings: new Map() }; }

  envGet(env, name) {
    let e = env;
    while (e) { if (e.bindings.has(name)) return e.bindings.get(name); e = e.parent; }
    return undefined;
  }

  envSet(env, name, value) { env.bindings.set(name, value); }

  // ─── Imports ──────────────────────────────────────────

  processImport(stmt, env) {
    if (stmt.type === 'Import') {
      for (const { name, alias } of stmt.names) {
        env.bindings.set(alias || name.split('.')[0], { kind: 'namespace', path: name });
      }
    } else {
      for (const { name, alias } of stmt.names) {
        const local = alias || name;
        const full = stmt.module ? `${stmt.module}.${name}` : name;
        env.bindings.set(local, this.resolveImportPath(full, name));
      }
    }
  }

  resolveImportPath(full, short) {
    if (KNOWN_LAYERS.has(short)) return { kind: 'layerFactory', name: short };
    if (['Model', 'Sequential', 'Layer'].includes(short)) return { kind: 'classRef', name: short };
    return { kind: 'namespace', path: full };
  }

  // ─── Builtins ─────────────────────────────────────────

  registerBuiltins(env) {
    env.bindings.set('range', { kind: 'builtin', name: 'range' });
    env.bindings.set('len', { kind: 'builtin', name: 'len' });
    env.bindings.set('int', { kind: 'builtin', name: 'int' });
    env.bindings.set('float', { kind: 'builtin', name: 'float' });
    env.bindings.set('print', { kind: 'builtin', name: 'print' });
    env.bindings.set('enumerate', { kind: 'builtin', name: 'enumerate' });
    env.bindings.set('zip', { kind: 'builtin', name: 'zip' });
    env.bindings.set('list', { kind: 'builtin', name: 'list' });
    env.bindings.set('tuple', { kind: 'builtin', name: 'tuple' });
    env.bindings.set('min', { kind: 'builtin', name: 'min' });
    env.bindings.set('max', { kind: 'builtin', name: 'max' });
    env.bindings.set('abs', { kind: 'builtin', name: 'abs' });
    env.bindings.set('sum', { kind: 'builtin', name: 'sum' });
    env.bindings.set('isinstance', { kind: 'builtin', name: 'isinstance' });
    env.bindings.set('reversed', { kind: 'builtin', name: 'reversed' });
    env.bindings.set('super', { kind: 'builtin', name: 'super' });
    env.bindings.set('__name__', { kind: 'scalar', value: '__not_main__' });
    env.bindings.set('__file__', { kind: 'scalar', value: '' });
    env.bindings.set('tf', { kind: 'namespace', path: 'tensorflow' });
    env.bindings.set('np', { kind: 'namespace', path: 'numpy' });
    env.bindings.set('layers', { kind: 'namespace', path: 'tensorflow.keras.layers' });
    env.bindings.set('Layer', { kind: 'classRef', name: 'Layer' });
    env.bindings.set('Model', { kind: 'classRef', name: 'Model' });
  }

  callBuiltin(name, args) {
    switch (name) {
      case 'range': {
        let start = 0, stop, step = 1;
        const a = args.map(v => this.toScalar(v));
        if (a.length === 1) stop = a[0];
        else if (a.length === 2) { start = a[0]; stop = a[1]; }
        else { start = a[0]; stop = a[1]; step = a[2]; }
        if (typeof stop !== 'number') return { kind: 'unknown' };
        const result = [];
        for (let i = start; step > 0 ? i < stop : i > stop; i += step) result.push({ kind: 'scalar', value: i });
        return { kind: 'list', items: result };
      }
      case 'len': return args[0]?.kind === 'list' ? { kind: 'scalar', value: args[0].items.length } : { kind: 'unknown' };
      case 'int': return { kind: 'scalar', value: Math.floor(this.toScalar(args[0]) || 0) };
      case 'float': return { kind: 'scalar', value: parseFloat(this.toScalar(args[0]) || 0) };
      case 'min': { const vals = args.length === 1 && args[0]?.kind === 'list' ? args[0].items.map(v => this.toScalar(v)) : args.map(v => this.toScalar(v)); return { kind: 'scalar', value: Math.min(...vals) }; }
      case 'max': { const vals = args.length === 1 && args[0]?.kind === 'list' ? args[0].items.map(v => this.toScalar(v)) : args.map(v => this.toScalar(v)); return { kind: 'scalar', value: Math.max(...vals) }; }
      case 'abs': return { kind: 'scalar', value: Math.abs(this.toScalar(args[0]) || 0) };
      case 'sum': { const items = args[0]?.kind === 'list' ? args[0].items : args; return { kind: 'scalar', value: items.reduce((s, v) => s + (this.toScalar(v) || 0), 0) }; }
      case 'print': return { kind: 'none' };
      case 'super': return { kind: 'super' };
      case 'isinstance': return { kind: 'scalar', value: true };
      case 'enumerate': {
        if (args[0]?.kind !== 'list') return { kind: 'unknown' };
        return { kind: 'list', items: args[0].items.map((v, i) => ({ kind: 'tuple', items: [{ kind: 'scalar', value: i }, v] })) };
      }
      case 'zip': {
        const lists = args.filter(a => a?.kind === 'list');
        if (lists.length === 0) return { kind: 'unknown' };
        const len = Math.min(...lists.map(l => l.items.length));
        return { kind: 'list', items: Array.from({ length: len }, (_, i) => ({ kind: 'tuple', items: lists.map(l => l.items[i]) })) };
      }
      case 'list': return args[0]?.kind === 'list' ? args[0] : { kind: 'list', items: [] };
      case 'tuple': return args[0]?.kind === 'list' ? { kind: 'tuple', items: args[0].items } : { kind: 'tuple', items: [] };
      case 'reversed': {
        if (args[0]?.kind === 'list') return { kind: 'list', items: [...args[0].items].reverse() };
        return args[0] ?? { kind: 'unknown' };
      }
      default: return { kind: 'unknown' };
    }
  }

  // ─── Exec statements ─────────────────────────────────

  execBlock(stmts, env) {
    let returnValue = null;
    for (const stmt of stmts) {
      try {
        const result = this.execStmt(stmt, env);
        if (result?._return) return result.value;
        if (result?._break || result?._continue) return result;
      } catch (e) {
        this.warnings.push(`Line ${stmt.line || '?'}: ${e.message}`);
      }
    }
    return returnValue;
  }

  execStmt(stmt, env) {
    if (stmt.line) this._currentLine = stmt.line;
    switch (stmt.type) {
      case 'Assign': return this.execAssign(stmt, env);
      case 'AugAssign': return this.execAugAssign(stmt, env);
      case 'ExprStmt': this.evalExpr(stmt.value, env); return null;
      case 'Return': {
        const val = stmt.value ? this.evalExpr(stmt.value, env) : { kind: 'none' };
        return { _return: true, value: val };
      }
      case 'For': return this.execFor(stmt, env);
      case 'If': return this.execIf(stmt, env);
      case 'While': return this.execWhile(stmt, env);
      case 'Pass': case 'Import': case 'ImportFrom': case 'Delete':
      case 'Raise': case 'Assert': case 'Try': case 'With':
        return null;
      default: return null;
    }
  }

  execAssign(stmt, env) {
    const value = this.evalExpr(stmt.value, env);
    if (value?.kind === 'tensor' && value.nodeId) {
      const node = this.graph.nodes.find(n => n.id === value.nodeId);
      if (node && (node.type === 'Slice' || node.type === 'SpaceToDepth' || node.type === 'DepthToSpace')) {
        const name = stmt.target?.type === 'Name' ? stmt.target.id : (stmt.target?.type === 'Attribute' ? stmt.target.attr : null);
        if (name) node.label = name;
      }
    }
    this.assignTarget(stmt.target, value, env);
  }

  execAugAssign(stmt, env) {
    const current = this.evalExpr(stmt.target, env);
    const rhs = this.evalExpr(stmt.value, env);
    const result = this.applyBinOp(stmt.op.replace('=', ''), current, rhs, env);
    this.assignTarget(stmt.target, result, env);
  }

  assignTarget(target, value, env) {
    if (target.type === 'Name') {
      env.bindings.set(target.id, value);
    } else if (target.type === 'Attribute') {
      const obj = this.evalExpr(target.value, env);
      if (obj?.kind === 'self') obj.attrs.set(target.attr, value);
    } else if (target.type === 'Subscript') {
      const obj = this.evalExpr(target.value, env);
      const idx = this.toScalar(this.evalExpr(target.slice, env));
      if (obj?.kind === 'list' && typeof idx === 'number') obj.items[idx] = value;
    } else if (target.type === 'Tuple') {
      if (value?.kind === 'tuple' || value?.kind === 'list') {
        for (let i = 0; i < target.elts.length; i++) {
          this.assignTarget(target.elts[i], value.items?.[i] || { kind: 'unknown' }, env);
        }
      }
    }
  }

  execFor(stmt, env) {
    const iter = this.evalExpr(stmt.iter, env);
    if (iter?.kind !== 'list') { this.warnings.push(`Cannot iterate over ${iter?.kind}`); return null; }
    for (const item of iter.items) {
      this.assignTarget(stmt.target, item, env);
      const result = this.execBlock(stmt.body, env);
      if (result?._return) return result;
      if (result?._break) break;
    }
    return null;
  }

  execIf(stmt, env) {
    const test = this.evalExpr(stmt.test, env);
    const truthy = this.isTruthy(test);
    if (truthy) return this.execBlock(stmt.body, env);
    if (stmt.orelse.length > 0) return this.execBlock(stmt.orelse, env);
    return null;
  }

  execWhile(stmt, env) {
    let guard = 0;
    while (guard++ < 1000) {
      const test = this.evalExpr(stmt.test, env);
      if (!this.isTruthy(test)) break;
      const result = this.execBlock(stmt.body, env);
      if (result?._return) return result;
      if (result?._break) break;
    }
    return null;
  }

  // ─── Eval expressions ─────────────────────────────────

  evalExpr(expr, env) {
    if (!expr) return { kind: 'none' };
    switch (expr.type) {
      case 'Num': return { kind: 'scalar', value: expr.value };
      case 'Str': return { kind: 'scalar', value: expr.value };
      case 'Bool': return { kind: 'scalar', value: expr.value };
      case 'NoneValue': return { kind: 'none' };
      case 'Name': return this.envGet(env, expr.id) ?? { kind: 'unknown', name: expr.id };
      case 'Attribute': return this.evalAttribute(expr, env);
      case 'Subscript': return this.evalSubscript(expr, env);
      case 'List': return { kind: 'list', items: expr.elts.map(e => this.evalExpr(e, env)) };
      case 'Tuple': return { kind: 'tuple', items: expr.elts.map(e => this.evalExpr(e, env)) };
      case 'Dict': return { kind: 'dict', keys: expr.keys.map(k => this.evalExpr(k, env)), values: expr.values.map(v => this.evalExpr(v, env)) };
      case 'Call': return this.evalCall(expr, env);
      case 'BinOp': return this.applyBinOp(expr.op, this.evalExpr(expr.left, env), this.evalExpr(expr.right, env), env);
      case 'UnaryOp': return this.applyUnaryOp(expr.op, this.evalExpr(expr.operand, env));
      case 'BoolOp': return this.evalBoolOp(expr, env);
      case 'Compare': return this.evalCompare(expr, env);
      case 'IfExpr': return this.isTruthy(this.evalExpr(expr.test, env)) ? this.evalExpr(expr.body, env) : this.evalExpr(expr.orelse, env);
      case 'Lambda': return { kind: 'lambda', params: expr.params, body: expr.body, env };
      case 'ListComp': return this.evalListComp(expr, env);
      case 'Starred': return this.evalExpr(expr.value, env);
      default: return { kind: 'unknown' };
    }
  }

  evalAttribute(expr, env) {
    const obj = this.evalExpr(expr.value, env);
    if (obj?.kind === 'self') {
      const val = obj.attrs.get(expr.attr);
      if (val) return val;
      if (obj.classDef) {
        const fn = obj.classDef.body.find(s => s.type === 'FunctionDef' && s.name === expr.attr);
        if (fn) return { kind: 'boundMethod', fn, self: obj };
        const classAttr = obj.classDef.body.find(s => s.type === 'Assign' && s.target?.type === 'Name' && s.target.id === expr.attr);
        if (classAttr) {
          const v = this.evalExpr(classAttr.value, this.createEnv(null));
          return v;
        }
      }
      return { kind: 'unknown', name: `self.${expr.attr}` };
    }
    if (obj?.kind === 'namespace') return this.resolveNamespaceAttr(obj.path, expr.attr);
    if (obj?.kind === 'list') {
      if (expr.attr === 'append') return { kind: 'method', obj, method: 'append' };
      if (expr.attr === 'insert') return { kind: 'method', obj, method: 'insert' };
      if (expr.attr === 'pop') return { kind: 'method', obj, method: 'pop' };
      if (expr.attr === 'extend') return { kind: 'method', obj, method: 'extend' };
    }
    if (obj?.kind === 'dict') {
      if (['update', 'get', 'keys', 'values', 'items'].includes(expr.attr)) return { kind: 'method', obj, method: expr.attr };
    }
    if (obj?.kind === 'tensor') {
      if (expr.attr === 'shape') return { kind: 'list', items: (obj.shape || []).map(d => ({ kind: 'scalar', value: d })) };
    }
    if (obj?.kind === 'super') return { kind: 'superMethod', method: expr.attr };
    if (obj?.kind === 'classDef') {
      const fn = obj.def.body.find(s => s.type === 'FunctionDef' && s.name === expr.attr);
      if (fn) return { kind: 'classFn', fn, cls: obj.def };
    }
    return { kind: 'unknown', name: expr.attr };
  }

  resolveNamespaceAttr(path, attr) {
    const full = `${path}.${attr}`;
    if (KNOWN_LAYERS.has(attr)) return { kind: 'layerFactory', name: attr };
    if (attr === 'Sequential') return { kind: 'sequentialFactory' };
    const nsPatterns = ['tensorflow.keras.layers', 'tensorflow.keras', 'tensorflow.image',
      'tensorflow.nn', 'tensorflow.math', 'tensorflow.raw_ops', 'keras.layers', 'keras'];
    if (nsPatterns.some(p => full.endsWith(p) || full === p)) return { kind: 'namespace', path: full };
    const tfOps = {
      'tensorflow.concat': 'concat', 'tensorflow.split': 'split', 'tensorflow.stack': 'stack',
      'tensorflow.pad': 'pad', 'tensorflow.reshape': 'reshape', 'tensorflow.transpose': 'transpose',
      'tensorflow.squeeze': 'squeeze', 'tensorflow.expand_dims': 'expand_dims',
      'tensorflow.reduce_mean': 'reduce_mean', 'tensorflow.reduce_sum': 'reduce_sum',
      'tensorflow.reduce_max': 'reduce_max',
      'tensorflow.cast': 'cast', 'tensorflow.shape': 'tf_shape',
      'tensorflow.zeros': 'zeros', 'tensorflow.ones': 'ones', 'tensorflow.zeros_like': 'zeros_like',
      'tensorflow.clip_by_value': 'clip_by_value',
      'tensorflow.image.resize': 'image_resize', 'tensorflow.image.resize_with_pad': 'image_resize',
      'tensorflow.nn.space_to_depth': 'space_to_depth', 'tensorflow.nn.depth_to_space': 'depth_to_space',
      'tensorflow.nn.relu': 'nn_act', 'tensorflow.nn.sigmoid': 'nn_act', 'tensorflow.nn.softmax': 'nn_act',
      'tensorflow.nn.leaky_relu': 'nn_act', 'tensorflow.nn.elu': 'nn_act', 'tensorflow.nn.tanh': 'nn_act',
      'tensorflow.math.log': 'math_fn', 'tensorflow.math.exp': 'math_fn', 'tensorflow.math.abs': 'math_fn',
      'tensorflow.math.round': 'math_fn', 'tensorflow.math.ceil': 'math_fn', 'tensorflow.math.floor': 'math_fn',
      'tensorflow.raw_ops.ResizeBilinear': 'image_resize', 'tensorflow.raw_ops.ResizeBicubic': 'image_resize',
    };
    if (tfOps[full]) return { kind: 'tfOp', op: tfOps[full], name: attr, fullPath: full };
    if (['Model', 'Sequential', 'Layer'].includes(attr)) return { kind: 'classRef', name: attr };
    return { kind: 'namespace', path: full };
  }

  evalSubscript(expr, env) {
    const obj = this.evalExpr(expr.value, env);
    if (obj?.kind === 'tensor') {
      const shape = obj.shape ? [...obj.shape] : [null, null, null, null];
      const sl = expr.slice;
      if (sl.type === 'Tuple' && sl.elts) {
        const dims = sl.elts;
        const lastDim = dims[dims.length - 1];
        if (lastDim?.type === 'Slice' && lastDim.lower && lastDim.upper) {
          const lo = this.toScalar(this.evalExpr(lastDim.lower, env));
          const hi = this.toScalar(this.evalExpr(lastDim.upper, env));
          if (typeof lo === 'number' && typeof hi === 'number') shape[shape.length - 1] = hi - lo;
        } else if (lastDim?.type === 'Slice' && lastDim.upper && !lastDim.lower) {
          const hi = this.toScalar(this.evalExpr(lastDim.upper, env));
          if (typeof hi === 'number') shape[shape.length - 1] = hi;
        }
        for (let d = 1; d < dims.length - 1 && d < shape.length; d++) {
          const dim = dims[d];
          if (dim?.type === 'Slice' && dim.upper) {
            const hi = this.toScalar(this.evalExpr(dim.upper, env));
            if (typeof hi === 'number') shape[d] = hi;
          }
        }
      } else if (sl.type === 'Slice') {
        if (sl.lower != null && sl.upper != null) {
          const lo = this.toScalar(this.evalExpr(sl.lower, env));
          const hi = this.toScalar(this.evalExpr(sl.upper, env));
          if (typeof lo === 'number' && typeof hi === 'number') shape[shape.length - 1] = hi - lo;
        }
      }
      const node = this.addNode('Slice', 'slice', {}, [obj.shape], shape);
      if (obj.nodeId) this.graph.edges.push({ from: obj.nodeId, to: node.id });
      return { kind: 'tensor', shape, nodeId: node.id };
    }
    const sl = expr.slice;
    if (sl.type === 'Slice' && (obj?.kind === 'list' || obj?.kind === 'tuple')) {
      const items = obj.items || [];
      const lo = sl.lower ? this.toScalar(this.evalExpr(sl.lower, env)) : 0;
      const hi = sl.upper ? this.toScalar(this.evalExpr(sl.upper, env)) : items.length;
      const start = typeof lo === 'number' ? (lo < 0 ? Math.max(0, items.length + lo) : lo) : 0;
      const end = typeof hi === 'number' ? (hi < 0 ? Math.max(0, items.length + hi) : hi) : items.length;
      return { kind: obj.kind, items: items.slice(start, end) };
    }
    if (sl.type === 'Slice' || sl.type === 'Tuple') return { kind: 'unknown' };
    const idx = this.evalExpr(sl, env);
    const i = this.toScalar(idx);
    if ((obj?.kind === 'list' || obj?.kind === 'tuple') && typeof i === 'number') {
      const actual = i < 0 ? obj.items.length + i : i;
      return obj.items[actual] ?? { kind: 'unknown' };
    }
    return { kind: 'unknown' };
  }

  evalCall(expr, env) {
    const func = this.evalExpr(expr.func, env);
    const args = expr.args.map(a => this.evalExpr(a, env));
    const kwargs = {};
    for (const kw of expr.kwargs) {
      if (kw.key) kwargs[kw.key] = this.evalExpr(kw.value, env);
    }

    if (func?.kind === 'builtin') return this.callBuiltin(func.name, args);
    if (func?.kind === 'superMethod' || func?.kind === 'super') return { kind: 'none' };
    if (func?.kind === 'boundMethod') return this.callBoundMethod(func, args, kwargs, env);
    if (func?.kind === 'method') return this.callMethod(func, args);
    if (func?.kind === 'layerFactory') return this.createLayer(func.name, args, kwargs);
    if (func?.kind === 'layer') return this.applyLayer(func, args, kwargs, env);
    if (func?.kind === 'sequential') return this.applySequential(func, args, env);
    if (func?.kind === 'sequentialFactory') return this.createSequential(args);
    if (func?.kind === 'tfOp') return this.applyTfOp(func, args, kwargs, env);
    if (func?.kind === 'lambda') return this.callLambda(func, args);
    if (func?.kind === 'classDef') return this.instantiateClass(func.def, args, kwargs, env);
    if (func?.kind === 'self' && func.classDef) return this.callModelInstance(func, args, kwargs, env);
    if (func?.kind === 'classFn') return { kind: 'none' };
    if (func?.kind === 'classRef') return { kind: 'none' };
    const tensorArg = args.find(a => a?.kind === 'tensor');
    if (tensorArg) return { kind: 'tensor', shape: [...(tensorArg.shape || [])], nodeId: tensorArg.nodeId };
    return { kind: 'unknown' };
  }

  callMethod(func, args) {
    const { obj, method } = func;
    if (obj.kind === 'list') {
      if (method === 'append' && args.length > 0) { obj.items.push(args[0]); return { kind: 'none' }; }
      if (method === 'insert' && args.length > 1) { const i = this.toScalar(args[0]); obj.items.splice(i, 0, args[1]); return { kind: 'none' }; }
      if (method === 'pop') { return obj.items.pop() || { kind: 'none' }; }
      if (method === 'extend' && args[0]?.kind === 'list') { obj.items.push(...args[0].items); return { kind: 'none' }; }
    }
    if (obj.kind === 'dict') {
      if (method === 'update' && args[0]?.kind === 'dict') {
        for (let i = 0; i < args[0].keys.length; i++) obj.keys.push(args[0].keys[i]);
        for (let i = 0; i < args[0].values.length; i++) obj.values.push(args[0].values[i]);
        return { kind: 'none' };
      }
      if (method === 'get') return args[1] || { kind: 'none' };
    }
    return { kind: 'none' };
  }

  createSequential(args) {
    const layerList = args[0];
    if (layerList?.kind === 'list') {
      return { kind: 'sequential', layers: layerList.items };
    }
    return { kind: 'unknown' };
  }

  applySequential(seq, args, env) {
    let x = args[0];
    if (!x || x.kind !== 'tensor') return { kind: 'unknown' };
    for (const layer of seq.layers) {
      if (layer?.kind === 'layer') {
        x = this.applyLayer(layer, [x], {}, env);
      } else if (layer?.kind === 'self' && layer.classDef) {
        x = this.callModelInstance(layer, [x], {}, env);
      } else if (x?.kind === 'tensor') {
        x = { kind: 'tensor', shape: [...x.shape], nodeId: x.nodeId };
      }
      if (!x || x.kind !== 'tensor') break;
    }
    return x || { kind: 'unknown' };
  }

  callLambda(func, args) {
    const lambdaEnv = this.createEnv(func.env);
    for (let i = 0; i < func.params.length; i++) {
      lambdaEnv.bindings.set(func.params[i].name, args[i] || (func.params[i].default ? this.evalExpr(func.params[i].default, func.env) : { kind: 'none' }));
    }
    return this.evalExpr(func.body, lambdaEnv);
  }

  callBoundMethod(func, args, kwargs, env) {
    const fn = func.fn;
    const methodEnv = this.createEnv(env);
    methodEnv.bindings.set('self', func.self);
    const params = fn.params.filter(p => p.name !== 'self');
    for (let i = 0; i < params.length; i++) {
      if (i < args.length) methodEnv.bindings.set(params[i].name, args[i]);
      else if (kwargs[params[i].name] !== undefined) methodEnv.bindings.set(params[i].name, kwargs[params[i].name]);
      else if (params[i].default) methodEnv.bindings.set(params[i].name, this.evalExpr(params[i].default, methodEnv));
    }
    return this.execBlock(fn.body, methodEnv) || { kind: 'unknown' };
  }

  instantiateClass(def, args, kwargs, env) {
    const self = { kind: 'self', attrs: new Map(), classDef: def };
    const initFn = def.body.find(s => s.type === 'FunctionDef' && s.name === '__init__');
    if (initFn) {
      const initEnv = this.createEnv(env);
      initEnv.bindings.set('self', self);
      const merged = {};
      const positionalParams = initFn.params.filter(p => p.name !== 'self' && p.kind === 'positional');
      for (let i = 0; i < args.length && i < positionalParams.length; i++) {
        merged[positionalParams[i].name] = args[i];
      }
      Object.assign(merged, kwargs);
      this.bindParams(initFn.params, merged, initEnv);
      this.execBlock(initFn.body, initEnv);
    }
    return self;
  }

  callModelInstance(instance, args, kwargs, env) {
    const callFn = instance.classDef.body.find(s => s.type === 'FunctionDef' && (s.name === 'call' || s.name === 'forward'));
    if (!callFn) return { kind: 'unknown' };
    const callEnv = this.createEnv(env);
    callEnv.bindings.set('self', instance);
    const params = callFn.params.filter(p => p.name !== 'self');
    for (let i = 0; i < params.length; i++) {
      if (i < args.length) callEnv.bindings.set(params[i].name, args[i]);
      else if (kwargs[params[i].name] !== undefined) callEnv.bindings.set(params[i].name, kwargs[params[i].name]);
      else if (params[i].default) callEnv.bindings.set(params[i].name, this.evalExpr(params[i].default, callEnv));
    }
    return this.execBlock(callFn.body, callEnv) || { kind: 'unknown' };
  }

  // ─── Layer creation and application ───────────────────

  createLayer(type, args, kwargs) {
    const config = { ...kwargs };
    for (const [k, v] of Object.entries(config)) {
      const s = this.toScalar(v);
      if (s !== undefined) config[k] = s;
      else if (v?.kind === 'list' || v?.kind === 'tuple') config[k] = v.items.map(i => this.toScalar(i));
    }
    const positionalMap = this.getLayerPositionals(type);
    for (let i = 0; i < args.length && i < positionalMap.length; i++) {
      const s = this.toScalar(args[i]);
      if (s !== undefined) config[positionalMap[i]] = s;
      else if (args[i]?.kind === 'list' || args[i]?.kind === 'tuple') config[positionalMap[i]] = args[i].items.map(item => this.toScalar(item));
      else config[positionalMap[i]] = args[i];
    }
    return { kind: 'layer', type, config, defLine: this._currentLine || 0 };
  }

  getLayerPositionals(type) {
    switch (type) {
      case 'Conv2D': case 'Conv1D': case 'Conv3D': case 'Conv2DTranspose': case 'Conv1DTranspose':
        return ['filters', 'kernel_size', 'strides', 'padding'];
      case 'DepthwiseConv2D': return ['kernel_size', 'strides', 'padding'];
      case 'SeparableConv2D': return ['filters', 'kernel_size', 'strides', 'padding'];
      case 'Dense': return ['units', 'activation'];
      case 'MaxPooling2D': case 'AveragePooling2D': return ['pool_size', 'strides', 'padding'];
      case 'MaxPooling1D': case 'AveragePooling1D': return ['pool_size', 'strides', 'padding'];
      case 'UpSampling2D': return ['size'];
      case 'UpSampling1D': return ['size'];
      case 'Dropout': case 'SpatialDropout2D': return ['rate'];
      case 'Activation': return ['activation'];
      case 'Reshape': return ['target_shape'];
      case 'Embedding': return ['input_dim', 'output_dim'];
      case 'LSTM': case 'GRU': case 'SimpleRNN': return ['units'];
      case 'ZeroPadding2D': return ['padding'];
      case 'Cropping2D': return ['cropping'];
      case 'LeakyReLU': return ['alpha'];
      case 'PReLU': return ['shared_axes'];
      default: return [];
    }
  }

  applyLayer(layer, args, kwargs, env) {
    let inputs = args.filter(a => a?.kind === 'tensor');
    if (inputs.length === 0 && args.length === 1 && (args[0]?.kind === 'list' || args[0]?.kind === 'tuple')) {
      inputs = args[0].items.filter(a => a?.kind === 'tensor');
    }
    if (inputs.length === 0) return { kind: 'unknown' };
    const inputTensor = inputs[0];
    const inputShape = inputTensor.shape;
    const outputShape = this.inferLayerShape(layer.type, layer.config, inputShape, inputs);

    let label = layer.label || layer.type;
    const node = this.addNode(layer.type, label, layer.config, [inputShape], outputShape, layer.defLine);
    this.graph.edges.push({ from: inputTensor.nodeId, to: node.id });

    if (inputs.length > 1) {
      for (let i = 1; i < inputs.length; i++) {
        this.graph.edges.push({ from: inputs[i].nodeId, to: node.id });
      }
    }

    return { kind: 'tensor', shape: outputShape, nodeId: node.id };
  }

  applyTfOp(func, args, kwargs, env) {
    switch (func.op) {
      case 'concat': {
        const tensors = args[0]?.kind === 'list' ? args[0].items.filter(t => t?.kind === 'tensor') : args.filter(a => a?.kind === 'tensor');
        if (tensors.length === 0) return { kind: 'unknown' };
        const axis = this.toScalar(kwargs.axis ?? args[1]) ?? -1;
        const shape = [...tensors[0].shape];
        const ax = axis < 0 ? shape.length + axis : axis;
        let concatDim = shape[ax] || 0;
        for (let i = 1; i < tensors.length; i++) {
          const d = tensors[i].shape?.[ax];
          if (typeof d === 'number' && typeof concatDim === 'number') concatDim += d;
          else concatDim = null;
        }
        shape[ax] = concatDim;
        const node = this.addNode('Concatenate', 'concat', { axis }, tensors.map(t => t.shape), shape);
        for (const t of tensors) this.graph.edges.push({ from: t.nodeId, to: node.id });
        return { kind: 'tensor', shape, nodeId: node.id };
      }
      case 'split': {
        const tensor = args.find(a => a?.kind === 'tensor');
        if (!tensor) return { kind: 'unknown' };
        const splitArg = args[1] ?? kwargs.num_or_size_splits;
        const axis = this.toScalar(args[2] ?? kwargs.axis) ?? -1;
        const baseShape = [...tensor.shape];
        const ax = axis < 0 ? baseShape.length + axis : axis;
        if (splitArg?.kind === 'list' || splitArg?.kind === 'tuple') {
          const sizes = splitArg.items.map(s => this.toScalar(s));
          return { kind: 'list', items: sizes.map(sz => {
            const s = [...baseShape]; if (typeof sz === 'number') s[ax] = sz;
            return { kind: 'tensor', shape: s, nodeId: tensor.nodeId };
          })};
        }
        const n = this.toScalar(splitArg) || 2;
        const s = [...baseShape];
        if (typeof s[ax] === 'number') s[ax] = Math.floor(s[ax] / n);
        return { kind: 'list', items: Array.from({ length: n }, () => ({ kind: 'tensor', shape: [...s], nodeId: tensor.nodeId })) };
      }
      case 'image_resize': {
        const tensor = args.find(a => a?.kind === 'tensor') ?? (kwargs.images?.kind === 'tensor' ? kwargs.images : null);
        if (!tensor) return { kind: 'unknown' };
        const size = args[1] ?? kwargs.size ?? kwargs.size_;
        let h = null, w = null;
        if (size?.kind === 'list' || size?.kind === 'tuple') {
          h = this.toScalar(size.items[0]); w = this.toScalar(size.items[1]);
        }
        const shape = [...tensor.shape];
        if (h !== null) shape[1] = h;
        if (w !== null) shape[2] = w;
        const node = this.addNode('Resize', 'resize', { height: h, width: w }, [tensor.shape], shape);
        this.graph.edges.push({ from: tensor.nodeId, to: node.id });
        return { kind: 'tensor', shape, nodeId: node.id };
      }
      case 'space_to_depth': {
        const tensor = args.find(a => a?.kind === 'tensor');
        if (!tensor) return { kind: 'unknown' };
        const bs = this.toScalar(args[1] ?? kwargs.block_size) || 2;
        const shape = [...tensor.shape];
        if (typeof shape[1] === 'number') shape[1] = Math.floor(shape[1] / bs);
        if (typeof shape[2] === 'number') shape[2] = Math.floor(shape[2] / bs);
        if (typeof shape[3] === 'number') shape[3] = shape[3] * bs * bs;
        const node = this.addNode('SpaceToDepth', 'space_to_depth', { block_size: bs }, [tensor.shape], shape);
        this.graph.edges.push({ from: tensor.nodeId, to: node.id });
        return { kind: 'tensor', shape, nodeId: node.id };
      }
      case 'depth_to_space': {
        const tensor = args.find(a => a?.kind === 'tensor');
        if (!tensor) return { kind: 'unknown' };
        const bs = this.toScalar(args[1] ?? kwargs.block_size) || 2;
        const shape = [...tensor.shape];
        if (typeof shape[1] === 'number') shape[1] = shape[1] * bs;
        if (typeof shape[2] === 'number') shape[2] = shape[2] * bs;
        if (typeof shape[3] === 'number') shape[3] = Math.floor(shape[3] / (bs * bs));
        const node = this.addNode('DepthToSpace', 'depth_to_space', { block_size: bs }, [tensor.shape], shape);
        this.graph.edges.push({ from: tensor.nodeId, to: node.id });
        return { kind: 'tensor', shape, nodeId: node.id };
      }
      case 'pad': {
        const tensor = args.find(a => a?.kind === 'tensor');
        if (!tensor) return { kind: 'unknown' };
        const shape = [...tensor.shape];
        const paddings = args[1] ?? kwargs.paddings;
        if (paddings?.kind === 'list' && paddings.items) {
          for (let d = 0; d < paddings.items.length && d < shape.length; d++) {
            const pair = paddings.items[d];
            if (pair?.kind === 'list' && pair.items?.length === 2) {
              const lo = this.toScalar(pair.items[0]) || 0;
              const hi = this.toScalar(pair.items[1]) || 0;
              if (typeof shape[d] === 'number') shape[d] += lo + hi;
            }
          }
        }
        const node = this.addNode('Pad', 'pad', {}, [tensor.shape], shape);
        this.graph.edges.push({ from: tensor.nodeId, to: node.id });
        return { kind: 'tensor', shape, nodeId: node.id };
      }
      case 'nn_act': case 'math_fn': case 'cast': case 'clip_by_value': case 'zeros_like': {
        const tensor = args.find(a => a?.kind === 'tensor');
        if (!tensor) return args[0] ?? { kind: 'unknown' };
        return { kind: 'tensor', shape: [...tensor.shape], nodeId: tensor.nodeId };
      }
      case 'reduce_mean': case 'reduce_sum': case 'reduce_max': {
        const tensor = args.find(a => a?.kind === 'tensor');
        if (!tensor) return { kind: 'unknown' };
        const axisRaw = args[1] ?? kwargs.axis;
        const keepdims = this.toScalar(kwargs.keepdims) || false;
        const shape = [...tensor.shape];
        let axes = [];
        if (axisRaw?.kind === 'list' || axisRaw?.kind === 'tuple') {
          axes = axisRaw.items.map(a => this.toScalar(a)).filter(a => typeof a === 'number');
        } else {
          const a = this.toScalar(axisRaw);
          if (typeof a === 'number') axes = [a];
        }
        for (const ax of axes.sort((a, b) => b - a)) {
          const idx = ax < 0 ? shape.length + ax : ax;
          if (keepdims) shape[idx] = 1; else shape.splice(idx, 1);
        }
        const node = this.addNode('Reduce', func.name, { axis: axes }, [tensor.shape], shape);
        this.graph.edges.push({ from: tensor.nodeId, to: node.id });
        return { kind: 'tensor', shape, nodeId: node.id };
      }
      case 'reshape': {
        const tensor = args.find(a => a?.kind === 'tensor');
        if (!tensor) return { kind: 'unknown' };
        const shapeArg = args[1];
        let newShape = tensor.shape;
        if (shapeArg?.kind === 'list' || shapeArg?.kind === 'tuple') {
          newShape = shapeArg.items.map(i => this.toScalar(i));
        }
        return { kind: 'tensor', shape: newShape, nodeId: tensor.nodeId };
      }
      case 'squeeze': case 'expand_dims': case 'transpose': {
        const tensor = args.find(a => a?.kind === 'tensor');
        if (!tensor) return { kind: 'unknown' };
        return { kind: 'tensor', shape: [...tensor.shape], nodeId: tensor.nodeId };
      }
      case 'tf_shape': {
        const tensor = args.find(a => a?.kind === 'tensor');
        if (!tensor) return { kind: 'unknown' };
        return { kind: 'list', items: tensor.shape.map(d => ({ kind: 'scalar', value: d })) };
      }
      case 'zeros': case 'ones': {
        const shapeArg = args[0];
        let shape = [null];
        if (shapeArg?.kind === 'list' || shapeArg?.kind === 'tuple') shape = shapeArg.items.map(i => this.toScalar(i));
        return { kind: 'tensor', shape, nodeId: null };
      }
    }
    return { kind: 'unknown' };
  }

  // ─── Shape inference ──────────────────────────────────

  inferLayerShape(type, config, inputShape, inputs) {
    if (!inputShape || inputShape.length === 0) return inputShape;
    const shape = [...inputShape];

    if (IDENTITY_LAYERS.has(type)) return shape;

    switch (type) {
      case 'Conv2D': case 'SeparableConv2D': {
        const strides = this.normalizeInt(config.strides, 1);
        const padding = (config.padding || 'valid').toLowerCase();
        if (padding === 'same') { shape[1] = this.ceilDiv(shape[1], strides); shape[2] = this.ceilDiv(shape[2], strides); }
        else { const k = this.normalizeInt(config.kernel_size, 3); shape[1] = this.ceilDiv(shape[1] - k + 1, strides); shape[2] = this.ceilDiv(shape[2] - k + 1, strides); }
        shape[3] = config.filters || shape[3];
        return shape;
      }
      case 'Conv2DTranspose': {
        const strides = this.normalizeInt(config.strides, 1);
        shape[1] = typeof shape[1] === 'number' ? shape[1] * strides : null;
        shape[2] = typeof shape[2] === 'number' ? shape[2] * strides : null;
        shape[3] = config.filters || shape[3];
        return shape;
      }
      case 'DepthwiseConv2D': {
        const strides = this.normalizeInt(config.strides, 1);
        const padding = (config.padding || 'valid').toLowerCase();
        if (padding === 'same') { shape[1] = this.ceilDiv(shape[1], strides); shape[2] = this.ceilDiv(shape[2], strides); }
        else { const k = this.normalizeInt(config.kernel_size, 3); shape[1] = this.ceilDiv(shape[1] - k + 1, strides); shape[2] = this.ceilDiv(shape[2] - k + 1, strides); }
        return shape;
      }
      case 'Conv1D': {
        const strides = this.normalizeInt(config.strides, 1);
        const padding = (config.padding || 'valid').toLowerCase();
        if (padding === 'same') shape[1] = this.ceilDiv(shape[1], strides);
        else { const k = this.normalizeInt(config.kernel_size, 3); shape[1] = this.ceilDiv(shape[1] - k + 1, strides); }
        shape[2] = config.filters || shape[2];
        return shape;
      }
      case 'Dense': {
        shape[shape.length - 1] = config.units || shape[shape.length - 1];
        return shape;
      }
      case 'MaxPooling2D': case 'AveragePooling2D': {
        const pool = this.normalizeInt(config.pool_size, 2);
        const strides = this.normalizeInt(config.strides, pool);
        const padding = (config.padding || 'valid').toLowerCase();
        if (padding === 'same') { shape[1] = this.ceilDiv(shape[1], strides); shape[2] = this.ceilDiv(shape[2], strides); }
        else { shape[1] = this.ceilDiv(shape[1] - pool + 1, strides); shape[2] = this.ceilDiv(shape[2] - pool + 1, strides); }
        return shape;
      }
      case 'UpSampling2D': {
        const size = this.normalizeInt(config.size, 2);
        shape[1] = typeof shape[1] === 'number' ? shape[1] * size : null;
        shape[2] = typeof shape[2] === 'number' ? shape[2] * size : null;
        return shape;
      }
      case 'GlobalAveragePooling2D': case 'GlobalMaxPooling2D':
        return [shape[0], shape[3]];
      case 'Flatten':
        return [shape[0], shape.slice(1).reduce((a, b) => (typeof a === 'number' && typeof b === 'number') ? a * b : null, 1)];
      case 'Reshape': {
        const ts = config.target_shape;
        if (Array.isArray(ts)) return [shape[0], ...ts];
        return shape;
      }
      case 'Concatenate': case 'Add': case 'Multiply': case 'Average': case 'Subtract': {
        if (type === 'Concatenate') {
          const axis = config.axis ?? -1;
          const ax = axis < 0 ? shape.length + axis : axis;
          let concatDim = 0;
          for (const inp of inputs) {
            const d = inp?.shape?.[ax];
            if (typeof d === 'number') concatDim += d; else { concatDim = null; break; }
          }
          shape[ax] = concatDim;
        }
        return shape;
      }
      case 'Embedding':
        return [...shape, config.output_dim || 64];
      case 'LSTM': case 'GRU': case 'SimpleRNN': {
        const units = config.units || 64;
        if (config.return_sequences) return [shape[0], shape[1], units];
        return [shape[0], units];
      }
      case 'ZeroPadding2D': {
        const p = config.padding;
        if (Array.isArray(p) && p.length >= 2) {
          if (Array.isArray(p[0])) { shape[1] = typeof shape[1] === 'number' ? shape[1] + p[0][0] + p[0][1] : null; shape[2] = typeof shape[2] === 'number' ? shape[2] + p[1][0] + p[1][1] : null; }
          else { shape[1] = typeof shape[1] === 'number' ? shape[1] + 2 * p[0] : null; shape[2] = typeof shape[2] === 'number' ? shape[2] + 2 * p[1] : null; }
        }
        return shape;
      }
      default: return shape;
    }
  }

  normalizeInt(val, dflt) {
    if (val === undefined || val === null) return dflt;
    if (typeof val === 'number') return val;
    if (Array.isArray(val)) return typeof val[0] === 'number' ? val[0] : dflt;
    if (val?.kind === 'list' || val?.kind === 'tuple') {
      const first = val.items?.[0];
      const s = this.toScalar(first);
      return typeof s === 'number' ? s : dflt;
    }
    const s = this.toScalar(val);
    return typeof s === 'number' ? s : dflt;
  }

  ceilDiv(a, b) {
    if (typeof a !== 'number' || typeof b !== 'number') return null;
    return Math.ceil(a / b);
  }

  // ─── BinOp / UnaryOp ─────────────────────────────────

  applyBinOp(op, left, right, env) {
    if (left?.kind === 'tensor' && right?.kind === 'tensor') {
      const shape = [...left.shape];
      let nodeType, label;
      if (op === '+') { nodeType = 'Add'; label = 'add'; }
      else if (op === '-') { nodeType = 'Subtract'; label = 'subtract'; }
      else if (op === '*') { nodeType = 'Multiply'; label = 'multiply'; }
      else { nodeType = 'BinOp'; label = op; }
      const node = this.addNode(nodeType, label, {}, [left.shape, right.shape], shape);
      this.graph.edges.push({ from: left.nodeId, to: node.id });
      this.graph.edges.push({ from: right.nodeId, to: node.id });
      return { kind: 'tensor', shape, nodeId: node.id };
    }
    if (left?.kind === 'tensor' && right?.kind !== 'tensor') {
      if (right?.kind === 'unknown' && ['+', '-', '*'].includes(op)) {
        return { kind: 'tensor', shape: [...left.shape], nodeId: left.nodeId };
      }
      return { kind: 'tensor', shape: [...left.shape], nodeId: left.nodeId };
    }
    if (right?.kind === 'tensor' && left?.kind !== 'tensor') {
      return { kind: 'tensor', shape: [...right.shape], nodeId: right.nodeId };
    }

    const a = this.toScalar(left), b = this.toScalar(right);
    if (typeof a === 'number' && typeof b === 'number') {
      let result;
      switch (op) {
        case '+': result = a + b; break; case '-': result = a - b; break;
        case '*': result = a * b; break; case '/': result = a / b; break;
        case '//': result = Math.floor(a / b); break; case '%': result = a % b; break;
        case '**': result = Math.pow(a, b); break;
        case '<<': result = a << b; break; case '>>': result = a >> b; break;
        case '|': result = a | b; break; case '&': result = a & b; break;
        case '^': result = a ^ b; break;
        default: return { kind: 'unknown' };
      }
      return { kind: 'scalar', value: result };
    }
    if (typeof a === 'string' && typeof b === 'string' && op === '+') return { kind: 'scalar', value: a + b };
    if (typeof a === 'string' && typeof b === 'number' && op === '*') return { kind: 'scalar', value: a.repeat(b) };
    if (left?.kind === 'list' && right?.kind === 'list' && op === '+') return { kind: 'list', items: [...left.items, ...right.items] };
    return { kind: 'unknown' };
  }

  applyUnaryOp(op, operand) {
    if (operand?.kind === 'tensor') return operand;
    const v = this.toScalar(operand);
    if (op === '-' && typeof v === 'number') return { kind: 'scalar', value: -v };
    if (op === '+' && typeof v === 'number') return { kind: 'scalar', value: +v };
    if (op === '~' && typeof v === 'number') return { kind: 'scalar', value: ~v };
    if (op === 'not') return { kind: 'scalar', value: !this.isTruthy(operand) };
    return { kind: 'unknown' };
  }

  evalBoolOp(expr, env) {
    if (expr.op === 'or') {
      for (const val of expr.values) { const v = this.evalExpr(val, env); if (this.isTruthy(v)) return v; }
      return { kind: 'scalar', value: false };
    }
    let last;
    for (const val of expr.values) { last = this.evalExpr(val, env); if (!this.isTruthy(last)) return last; }
    return last;
  }

  evalCompare(expr, env) {
    let left = this.evalExpr(expr.left, env);
    for (let i = 0; i < expr.ops.length; i++) {
      const right = this.evalExpr(expr.comparators[i], env);
      let result;
      const op = expr.ops[i];
      if (op === 'in' || op === 'not in') {
        const items = (right?.kind === 'list' || right?.kind === 'tuple') ? right.items : [];
        const lv = this.toScalar(left);
        const found = items.some(it => this.toScalar(it) === lv);
        result = op === 'in' ? found : !found;
      } else if ((left?.kind === 'tuple' || left?.kind === 'list') && (right?.kind === 'tuple' || right?.kind === 'list')) {
        const eq = left.items.length === right.items.length && left.items.every((v, j) => this.toScalar(v) === this.toScalar(right.items[j]));
        result = op === '==' ? eq : op === '!=' ? !eq : true;
      } else {
        const a = this.toScalar(left), b = this.toScalar(right);
        switch (op) {
          case '==': result = a === b; break; case '!=': result = a !== b; break;
          case '<': result = a < b; break; case '>': result = a > b; break;
          case '<=': result = a <= b; break; case '>=': result = a >= b; break;
          case 'is': result = left === right; break; case 'is not': result = left !== right; break;
          default: result = true;
        }
      }
      if (!result) return { kind: 'scalar', value: false };
      left = right;
    }
    return { kind: 'scalar', value: true };
  }

  evalListComp(expr, env) {
    const results = [];
    const runGen = (idx) => {
      if (idx >= expr.generators.length) { results.push(this.evalExpr(expr.elt, env)); return; }
      const gen = expr.generators[idx];
      const iter = this.evalExpr(gen.iter, env);
      if (iter?.kind !== 'list') return;
      for (const item of iter.items) {
        this.assignTarget(gen.target, item, env);
        let pass = true;
        for (const ifExpr of gen.ifs) { if (!this.isTruthy(this.evalExpr(ifExpr, env))) { pass = false; break; } }
        if (pass) runGen(idx + 1);
      }
    };
    runGen(0);
    return { kind: 'list', items: results };
  }

  // ─── Helpers ──────────────────────────────────────────

  addNode(type, label, config, inputShapes, outputShape, defLine) {
    const id = `n${this.nodeId++}`;
    const node = { id, type, label, config: { ...config }, inputShapes, outputShape, params: this.estimateParams(type, config, inputShapes, outputShape), srcLine: this._currentLine || 0, defLine: defLine || 0 };
    this.graph.nodes.push(node);
    return node;
  }

  toScalar(v) {
    if (v === undefined || v === null) return undefined;
    if (typeof v === 'number' || typeof v === 'string' || typeof v === 'boolean') return v;
    if (v?.kind === 'scalar') return v.value;
    if (v?.kind === 'none') return null;
    return undefined;
  }

  isTruthy(v) {
    if (!v) return false;
    if (v.kind === 'scalar') return !!v.value;
    if (v.kind === 'none') return false;
    if (v.kind === 'list') return v.items.length > 0;
    if (v.kind === 'unknown') return true;
    return true;
  }

  flattenName(expr) {
    if (!expr) return null;
    if (expr.type === 'Name') return expr.id;
    if (expr.type === 'Attribute') { const p = this.flattenName(expr.value); return p ? `${p}.${expr.attr}` : expr.attr; }
    return null;
  }

  bindParams(params, kwargs, env) {
    for (const p of params) {
      if (p.name === 'self' || p.kind === 'separator' || p.kind === 'varargs' || p.kind === 'kwargs') continue;
      const val = kwargs[p.name];
      if (val !== undefined) {
        env.bindings.set(p.name, typeof val === 'object' && val?.kind ? val : { kind: 'scalar', value: val });
      } else if (p.default) {
        env.bindings.set(p.name, this.evalExpr(p.default, env));
      }
    }
  }

  extractCallKwargs(callExpr, env) {
    const result = {};
    const func = this.flattenName(callExpr.func);
    for (const kw of callExpr.kwargs) {
      if (kw.key) {
        try { result[kw.key] = this.evalExpr(kw.value, env); } catch { result[kw.key] = { kind: 'unknown' }; }
      }
    }
    return result;
  }

  estimateParams(type, config, inputShapes, outputShape) {
    const inC = inputShapes?.[0]?.[inputShapes[0]?.length - 1];
    switch (type) {
      case 'Conv2D': case 'Conv1D': case 'Conv3D': {
        const k = this.normalizeInt(config.kernel_size, 3);
        const f = config.filters || 0;
        const bias = config.use_bias === false ? 0 : f;
        const dim = type === 'Conv1D' ? 1 : type === 'Conv3D' ? 3 : 2;
        return typeof inC === 'number' ? Math.pow(k, dim) * inC * f + bias : 0;
      }
      case 'Conv2DTranspose': { const k = this.normalizeInt(config.kernel_size, 3); const f = config.filters || 0; const bias = config.use_bias === false ? 0 : f; return typeof inC === 'number' ? k * k * inC * f + bias : 0; }
      case 'SeparableConv2D': { const k = this.normalizeInt(config.kernel_size, 3); const f = config.filters || 0; const bias = config.use_bias === false ? 0 : f; return typeof inC === 'number' ? k * k * inC + inC * f + bias : 0; }
      case 'DepthwiseConv2D': { const k = this.normalizeInt(config.kernel_size, 3); const bias = config.use_bias === false ? 0 : inC; return typeof inC === 'number' ? k * k * inC + bias : 0; }
      case 'Dense': { const u = config.units || 0; const bias = config.use_bias === false ? 0 : u; return typeof inC === 'number' ? inC * u + bias : 0; }
      case 'BatchNormalization': return typeof inC === 'number' ? inC * 4 : 0;
      case 'LayerNormalization': return typeof inC === 'number' ? inC * 2 : 0;
      case 'PReLU': return typeof inC === 'number' ? inC : 0;
      case 'Embedding': return (config.input_dim || 0) * (config.output_dim || 0);
      case 'LSTM': { const u = config.units || 0; return typeof inC === 'number' ? 4 * (inC * u + u * u + u) : 0; }
      case 'GRU': { const u = config.units || 0; return typeof inC === 'number' ? 3 * (inC * u + u * u + u) : 0; }
      default: return 0;
    }
  }
}
