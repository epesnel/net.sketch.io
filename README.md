# net.sketch.io

Paste TF/Keras Python code, get a publication-ready architecture diagram.

**[Try it live](https://epesnel.github.io/net.sketch.io/)**

## Features

- **Static analysis** — no Python runtime needed, runs entirely in the browser
- **Accurate** — 73/73 exact param match against TF ground truth across 24 architectures
- **Publication-ready SVG export** — pastel color palette, clean edges, white background
- **Interactive** — click a node to see its config, highlight the source line, edit parameters
- **MACs / Params heatmap** — instantly see computational bottlenecks
- **Code editor** — syntax highlighting, line numbers, copy button

## Supported layers

Conv2D, Conv1D, Conv2DTranspose, DepthwiseConv2D, SeparableConv2D, Dense, MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D, UpSampling2D, BatchNormalization, LayerNormalization, Dropout, PReLU, LeakyReLU, ReLU, Flatten, Reshape, Embedding, LSTM, GRU, and TF ops (concat, split, resize, space_to_depth, depth_to_space, pad, reduce_mean).

## How it works

```
Python code → Tokenizer → Parser (AST) → Interpreter (graph + shapes) → Renderer (SVG)
```

All four stages run in JavaScript — no backend, no build step, no dependencies beyond dagre.js for graph layout.

## Development

```bash
# Edit src/*.js or style.css, then rebuild:
python3 build.py

# Run tests (requires TF 2.x + Python 3.11):
python3.11 tests/test_vs_tf.py   # 73/73 param accuracy
python3.11 tests/test_gui.py     # 48/48 topology checks
```

## Project structure

```
src/
  tokenizer.js    — Python tokenizer with INDENT/DEDENT
  parser.js       — Recursive descent + Pratt parser → AST
  interpreter.js  — Partial evaluator, shape inference, graph builder
  renderer.js     — SVG rendering, heatmaps, zoom, interactivity
  dagre.min.js    — Graph layout (Sugiyama algorithm)
style.css         — UI styling
build.py          — Bundles src/ + CSS into self-contained app.html
tests/
  test_vs_tf.py   — Non-regression vs TF ground truth (24 models)
  test_gui.py     — Graph topology coherence checks
  models/         — Test architectures (LoResNet, HeavyTeacher, etc.)
```

## License

MIT
