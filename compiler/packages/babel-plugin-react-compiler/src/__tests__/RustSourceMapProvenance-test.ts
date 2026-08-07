/**
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/**
 * Source location provenance for the Rust compiler backend.
 *
 * The backend runs as a Babel plugin: Babel parses, the plugin hands back a
 * compiled Babel AST, and Babel's own generator emits the code *and* the
 * source map. The generator derives that map purely from `node.loc`, so a
 * materialized node only appears in the source map if codegen gave it the
 * location of the source construct it represents.
 *
 * Snapshot fixtures cannot catch regressions here: `node.loc` is not part of
 * the printed source, so a fixture's `.expect.md` is byte-for-byte identical
 * whether locations are correct or entirely absent.
 *
 * The contract under test is that semantic nodes inherit precise source
 * provenance while synthetic compiler scaffolding stays unmapped. Every
 * assertion below is absolute — an exact expected location — rather than a
 * comparison against the TypeScript backend. The TS backend is not a
 * specification: it smears several locations (the whole component's span onto
 * outlined helpers, an element's span onto its tag name), and pinning Rust to
 * it would both encode those bugs and break this suite whenever TS changed.
 *
 * Three complementary checks are needed, because no single one sees the whole
 * contract:
 *
 *   1. Throw-site mapping, per construct. The strongest and most readable
 *      check, and the one that matches what a user actually experiences: run
 *      the compiled code, capture the frame that threw, map it back, and
 *      assert it lands exactly on the originating `new Error(...)`.
 *   2. Whole-map position parity. Catches provenance that codegen *drops* on
 *      nodes no stack frame happens to point at — for example the binding of a
 *      `for...of` loop.
 *   3. Direct `node.loc` assertions. Source maps are blind to a location being
 *      applied too widely: smearing an instruction's location over a generated
 *      subtree only ever duplicates a position that is already mapped, so it
 *      never changes the mapped position set. Reading `node.loc` is the only
 *      way to see over-assignment, and the only way to assert that scaffolding
 *      stayed unmapped.
 */

import * as BabelCore from '@babel/core';
import {
  TraceMap,
  eachMapping,
  originalPositionFor,
} from '@jridgewell/trace-mapping';
import * as vm from 'vm';

// --------------------------------------------------------------------------
// Backend
// --------------------------------------------------------------------------

/**
 * The Rust backend is a native addon built by cargo. It is not committed, so a
 * checkout that has never run a Rust build has nothing to test.
 *
 * Skipping in that case is deliberate: someone working only on the TypeScript
 * backend should not need a Rust toolchain to run this package's tests.
 *
 * Skipping *silently* in CI is not deliberate — it would let every assertion
 * below rot while the build stayed green, which is the exact failure mode this
 * suite exists to prevent. `.github/workflows/compiler_rust.yml` builds the
 * addon and sets `REQUIRE_RUST_BACKEND=1`, which turns a failed load into a
 * hard error instead of a skip.
 */
let rustPlugin: BabelCore.PluginItem | null = null;
let rustLoadError: unknown = null;
try {
  rustPlugin = require('babel-plugin-react-compiler-rust').default;
} catch (distError) {
  try {
    rustPlugin =
      require('../../../babel-plugin-react-compiler-rust/src').default;
  } catch {
    rustLoadError = distError;
  }
}

if (rustPlugin == null && process.env['REQUIRE_RUST_BACKEND'] != null) {
  throw new Error(
    'REQUIRE_RUST_BACKEND is set, but the Rust compiler backend could not be ' +
      'loaded. Without it this suite would report success while running none ' +
      'of its assertions.\n' +
      'Build the backend with:\n' +
      '  cargo build -p react_compiler_napi\n' +
      '  cp target/debug/libreact_compiler_napi.so \\\n' +
      '    packages/babel-plugin-react-compiler-rust/native/index.node\n' +
      '  yarn workspace babel-plugin-react-compiler-rust run build\n' +
      `Underlying require error: ${String(rustLoadError)}`,
  );
}

const describeIfRust = rustPlugin != null ? describe : describe.skip;

/** Only dereferenced inside `describeIfRust`, where the backend is present. */
const RUST_PLUGIN = rustPlugin as BabelCore.PluginItem;

const FILENAME = 'RustSourceMapFixture.jsx';

// --------------------------------------------------------------------------
// Compilation
// --------------------------------------------------------------------------

interface CompileOptions {
  /**
   * Lower JSX to `h(...)` calls. Required to execute the output, but it
   * destroys `JSXIdentifier` nodes, so tests that assert on them opt out.
   */
  transformJsx: boolean;
}

function pluginList({
  transformJsx,
}: CompileOptions): Array<BabelCore.PluginItem> {
  const plugins: Array<BabelCore.PluginItem> = [
    [
      RUST_PLUGIN,
      {
        compilationMode: 'all',
        panicThreshold: 'all_errors',
        environment: {
          // Jest defines `__DEV__: true`, which makes a JS backend treat the
          // build as a dev build and prepend an extra cache-invalidation
          // block. The Rust backend is a native build with no `__DEV__`, so
          // leaving this on would make the compiled output depend on which
          // runner invoked it.
          enableResetCacheOnSourceFileChanges: false,
        },
      },
    ],
  ];
  if (transformJsx) {
    // Classic runtime keeps the JSX call shape simple enough to invoke
    // directly in the sandbox below.
    plugins.push([
      '@babel/plugin-transform-react-jsx',
      {runtime: 'classic', pragma: 'h'},
    ]);
  }
  return plugins;
}

interface Compiled {
  code: string;
  map: BabelCore.BabelFileResult['map'];
}

function compile(source: string): Compiled {
  const result = BabelCore.transformSync(source, {
    filename: FILENAME,
    sourceMaps: true,
    // Keep the source map anchored to the fixture's own name rather than an
    // absolute path, so the expected positions below are machine-independent.
    sourceFileName: FILENAME,
    configFile: false,
    babelrc: false,
    plugins: [
      ...pluginList({transformJsx: true}),
      // The compiler emits `import {c as _c} from "react/compiler-runtime"`,
      // which `vm.Script` cannot parse. Lower it to `require` so the output is
      // executable. This runs after the compiler, so it only rewrites module
      // syntax and does not affect the locations under test.
      '@babel/plugin-transform-modules-commonjs',
    ],
  });

  if (result == null || result.code == null || result.map == null) {
    throw new Error('Expected Babel to produce code and a source map');
  }
  return {code: result.code, map: result.map};
}

/** The compiled Babel AST, before it is printed. */
function compileToAst(
  source: string,
  options: CompileOptions = {transformJsx: true},
): unknown {
  let program: unknown = null;
  const capture = (): BabelCore.PluginObj => ({
    visitor: {
      Program: {
        exit(path) {
          program = JSON.parse(JSON.stringify(path.node));
        },
      },
    },
  });

  BabelCore.transformSync(source, {
    filename: FILENAME,
    configFile: false,
    babelrc: false,
    // Without the JSX transform in the plugin list, the parser still has to be
    // told to accept JSX syntax.
    parserOpts: {plugins: ['jsx']},
    plugins: [...pluginList(options), capture],
  });

  if (program == null) {
    throw new Error('Expected to capture a compiled Program');
  }
  return program;
}

// --------------------------------------------------------------------------
// AST inspection
// --------------------------------------------------------------------------

const IGNORED_FIELDS = new Set([
  'loc',
  'start',
  'end',
  'range',
  'extra',
  'leadingComments',
  'trailingComments',
  'innerComments',
]);

function formatLoc(node: {
  loc?: BabelCore.types.SourceLocation | null;
}): string {
  const loc = node.loc;
  if (loc == null) {
    return '<none>';
  }
  return `${loc.start.line}:${loc.start.column}-${loc.end.line}:${loc.end.column}`;
}

function walkNodes(root: unknown, visit: (node: any) => void): void {
  (function walk(node: any): void {
    if (node == null || typeof node !== 'object') {
      return;
    }
    if (Array.isArray(node)) {
      node.forEach(walk);
      return;
    }
    if (typeof node.type === 'string') {
      visit(node);
    }
    for (const field of Object.keys(node)) {
      if (IGNORED_FIELDS.has(field) || field.startsWith('_')) continue;
      walk(node[field]);
    }
  })(root);
}

/** Every node of `type` in the compiled AST, in traversal order. */
function nodesOfType(ast: unknown, type: string): Array<any> {
  const found: Array<any> = [];
  walkNodes(ast, node => {
    if (node.type === type) {
      found.push(node);
    }
  });
  return found;
}

/**
 * Every `JSXIdentifier` in the compiled AST, as `name @ loc`, in traversal
 * order. Only meaningful when the JSX transform has been left out, since it
 * lowers these nodes away.
 */
function jsxIdentifierLocations(ast: unknown): Array<string> {
  return nodesOfType(ast, 'JSXIdentifier').map(
    node => `${node.name} @ ${formatLoc(node)}`,
  );
}

// --------------------------------------------------------------------------
// Execution + stack frame capture
// --------------------------------------------------------------------------

interface Frame {
  line: number;
  /** 1-based, as reported by V8. */
  column: number;
}

const MEMO_CACHE_SENTINEL = Symbol.for('react.memo_cache_sentinel');

/**
 * How to make the compiled component reach its `throw`: either during render,
 * or by invoking a callback the component passed through JSX props.
 */
type Trigger = 'render' | 'callback';

/**
 * Run the compiled component and return the generated position of the frame
 * that threw.
 *
 * The capture runs *inside* the sandbox: the error is constructed by the vm
 * realm's `Error`, so only that realm's `prepareStackTrace` hook is consulted.
 * Installing the hook on the host's `Error` would have no effect. Using the
 * hook gives structured CallSite objects, which avoids parsing stack strings
 * and yields exact line/column.
 */
function captureThrowFrame(
  code: string,
  generatedFilename: string,
  trigger: Trigger,
): Frame {
  const sandbox: Record<string, unknown> = {
    // Minimal `react/compiler-runtime` + React stubs.
    require(id: string): unknown {
      if (id === 'react/compiler-runtime' || id === 'react') {
        return {c: (size: number) => new Array(size).fill(MEMO_CACHE_SENTINEL)};
      }
      throw new Error(`Unexpected require(${id})`);
    },
    // Classic JSX pragma: build a plain element object.
    h: (type: unknown, props: unknown, ...children: unknown[]) => ({
      type,
      props: props ?? {},
      children,
    }),
    exports: {},
    __filename__: generatedFilename,
    __trigger__: trigger,
    __frame__: undefined as unknown,
  };

  const script = new vm.Script(
    `${code}
__frame__ = (function () {
  var previous = Error.prepareStackTrace;
  Error.prepareStackTrace = function (_error, callSites) { return callSites; };
  try {
    var element = Component({
      items: [{name: 'a'}],
      map: {k: 1},
      kind: 'a',
      risky: function () { throw new Error('from props.risky'); },
    });
    if (__trigger__ === 'callback') {
      element.props.onClick();
    }
    return {error: 'Expected the compiled component to throw'};
  } catch (error) {
    var callSites = error.stack;
    if (!Array.isArray(callSites)) {
      return {error: 'Expected structured stack frames'};
    }
    for (var i = 0; i < callSites.length; i++) {
      if (callSites[i].getFileName() === __filename__) {
        return {
          line: callSites[i].getLineNumber(),
          column: callSites[i].getColumnNumber(),
        };
      }
    }
    return {error: 'No stack frame from ' + __filename__};
  } finally {
    Error.prepareStackTrace = previous;
  }
})();`,
    {filename: generatedFilename},
  );
  vm.createContext(sandbox);
  script.runInContext(sandbox as vm.Context);

  const captured = sandbox['__frame__'] as
    | {line: number; column: number; error?: undefined}
    | {error: string};
  if (captured.error != null) {
    throw new Error(captured.error);
  }
  return {line: captured.line, column: captured.column};
}

// --------------------------------------------------------------------------
// Source map helpers
// --------------------------------------------------------------------------

interface OriginalPosition {
  source: string | null;
  line: number | null;
  column: number | null;
}

function mapFrameToOriginal(
  map: BabelCore.BabelFileResult['map'],
  frame: Frame,
): OriginalPosition {
  const tracer = new TraceMap(map as never);
  // V8 columns are 1-based; source map columns are 0-based.
  const position = originalPositionFor(tracer, {
    line: frame.line,
    column: frame.column - 1,
  });
  return {
    source: position.source ?? null,
    line: position.line ?? null,
    column: position.column ?? null,
  };
}

/**
 * Every original position the map attributes a mapping to, as one entry per
 * source line: `line: col,col,col`.
 *
 * Grouped by line purely so the expectation stays reviewable, but compared at
 * column granularity: a materialized node that inherits its enclosing
 * statement's location instead of its own still lands on the right line, so a
 * line-level check would miss it.
 */
function originalPositions(
  map: BabelCore.BabelFileResult['map'],
): Array<string> {
  const tracer = new TraceMap(map as never);
  const byLine = new Map<number, Set<number>>();
  eachMapping(tracer, mapping => {
    if (mapping.originalLine == null || mapping.originalColumn == null) {
      return;
    }
    let columns = byLine.get(mapping.originalLine);
    if (columns == null) {
      columns = new Set();
      byLine.set(mapping.originalLine, columns);
    }
    columns.add(mapping.originalColumn);
  });
  return [...byLine.entries()]
    .sort(([a], [b]) => a - b)
    .map(
      ([line, columns]) =>
        `${line}: ${[...columns].sort((a, b) => a - b).join(',')}`,
    );
}

/** Every distinct `source` referenced by the map's mappings. */
function mappedSources(map: BabelCore.BabelFileResult['map']): Set<string> {
  const tracer = new TraceMap(map as never);
  const sources = new Set<string>();
  eachMapping(tracer, mapping => {
    if (mapping.source != null) {
      sources.add(mapping.source);
    }
  });
  return sources;
}

// --------------------------------------------------------------------------
// Throw-site fixtures
// --------------------------------------------------------------------------

interface ThrowCase {
  name: string;
  source: string;
  trigger: Trigger;
}

/**
 * One fixture per construct that materializes new AST nodes during codegen. In
 * every case the `throw` sits inside the construct under test, so the mapped
 * frame is unambiguous evidence that the construct kept its provenance.
 */
const THROW_CASES: Array<ThrowCase> = [
  {
    name: 'a for...of body',
    trigger: 'render',
    source: `function Component(props) {
  const out = [];
  for (const item of props.items) {
    throw new Error('boom');
  }
  return <div>{out}</div>;
}
`,
  },
  {
    name: 'a for...in body',
    trigger: 'render',
    source: `function Component(props) {
  const out = [];
  for (const key in props.map) {
    throw new Error('boom');
  }
  return <div>{out}</div>;
}
`,
  },
  {
    name: 'a switch case',
    trigger: 'render',
    source: `function Component(props) {
  switch (props.kind) {
    case 'a': {
      throw new Error('boom');
    }
    default: {
      break;
    }
  }
  return <div>{props.kind}</div>;
}
`,
  },
  {
    name: 'a catch clause',
    trigger: 'render',
    source: `function Component(props) {
  try {
    props.risky();
  } catch (e) {
    throw new Error('boom');
  }
  return <div>{props.kind}</div>;
}
`,
  },
  {
    name: 'an object method',
    trigger: 'render',
    source: `function Component(props) {
  const obj = {
    run() {
      throw new Error('boom');
    },
  };
  return <div>{obj.run()}</div>;
}
`,
  },
  {
    name: 'a nested arrow invoked from JSX props',
    trigger: 'callback',
    source: `function Component(props) {
  const onClick = () => {
    throw new Error('boom');
  };
  const label = props.items.map(item => item.name).join(', ');
  return <div onClick={onClick}>{label}</div>;
}
`,
  },
];

/**
 * Position of `new Error(` in a fixture. V8 reports the throwing frame at the
 * `new Error(...)` expression being evaluated, not at the `throw` keyword, so
 * that is the position the map must resolve back to.
 */
function throwSite(source: string): {line: number; column: number} {
  const lines = source.split('\n');
  for (let index = 0; index < lines.length; index++) {
    const column = lines[index].indexOf('new Error(');
    if (column !== -1) {
      return {line: index + 1, column};
    }
  }
  throw new Error('Fixture has no `new Error(` throw site');
}

// --------------------------------------------------------------------------
// Whole-program fixture
// --------------------------------------------------------------------------

/**
 * A single component exercising every construct whose provenance codegen is
 * most likely to drop. Constructs are grouped into one component rather than
 * split across fixtures because the whole-map checks below compare entire
 * maps, and because the memoization scaffolding they also assert on is only
 * emitted once per component.
 *
 * Line/column numbers appear in the expectations below, so edits to this
 * fixture will require re-deriving them.
 */
const FIXTURE = `function Component(props) {
  const onClick = () => {
    throw new Error('boom');
  };
  const offset = -42;
  const seen = [];
  for (const key in props.map) {
    seen.push(key);
  }
  for (const item of props.items) {
    seen.push(item.name);
  }
  switch (props.kind) {
    case 'a': {
      seen.push('a');
      break;
    }
    default: {
      seen.push('z');
    }
  }
  try {
    seen.push(props.risky());
  } catch (e) {
    seen.push(String(e));
  }
  const obj = {
    method() {
      return seen.length + offset;
    },
  };
  const label = props.items.map(item => item.name).join(', ');
  return <div onClick={onClick}>{label}{obj.method()}</div>;
}
`;

// --------------------------------------------------------------------------
// Tests
// --------------------------------------------------------------------------

describeIfRust('Rust backend source location provenance', () => {
  describe('maps a throw back to its exact original position in', () => {
    for (const testCase of THROW_CASES) {
      it(testCase.name, () => {
        const {code, map} = compile(testCase.source);
        const frame = captureThrowFrame(code, 'output.js', testCase.trigger);
        expect(mapFrameToOriginal(map, frame)).toEqual({
          source: FILENAME,
          ...throwSite(testCase.source),
        });
      });
    }
  });

  /**
   * Asserted on the AST *before* the JSX transform runs, because that
   * transform lowers JSX identifiers into string literals and object keys, at
   * which point the tag name's location is no longer observable.
   */
  it('gives JSX tag and attribute names their own source location', () => {
    const source = `function Component(props) {
  const onClick = () => {};
  return <div onClick={onClick} className="x">{props.label}</div>;
}
`;
    // Columns on line 3: `div` at 10, `onClick` at 14, `className` at 32. Each
    // name spans only itself — not the enclosing element, and not the
    // attribute's value.
    expect(
      jsxIdentifierLocations(compileToAst(source, {transformJsx: false})),
    ).toEqual([
      'div @ 3:10-3:13',
      'onClick @ 3:14-3:21',
      'className @ 3:32-3:41',
      'div @ 3:10-3:13',
    ]);
  });

  describe('whole-program provenance', () => {
    let compiled: Compiled;
    let ast: unknown;

    beforeAll(() => {
      compiled = compile(FIXTURE);
      ast = compileToAst(FIXTURE);
    });

    it('attributes every mapping to the original filename', () => {
      expect(mappedSources(compiled.map)).toEqual(new Set([FILENAME]));
    });

    it('maps exactly the expected set of original positions', () => {
      // Both directions matter. A position disappearing means codegen dropped
      // provenance; a position appearing means it invented some, which is how
      // scaffolding ends up blaming user code. Lines absent from this list
      // (9, 17, 18, 20, 28, 30, 34) hold only closing braces, which produce no
      // mapping of their own.
      expect(originalPositions(compiled.map)).toEqual([
        '1: 0,19,24',
        '2: 2,8,15,18',
        '3: 4,10,14,19,20,26,27,28',
        '4: 3,4',
        '5: 17',
        '6: 8,12,15,17',
        '7: 2,7,13,16,20,25',
        '8: 4,8,13,14,17,18',
        '10: 2,7,13,17,21,26,32',
        '11: 4,8,13,14,18,23,24',
        '12: 3',
        '13: 2,10,15,20',
        '14: 9,12',
        '15: 6,10,15,16,19,20',
        '16: 6,12',
        '19: 6,10,15,16,19,20',
        '21: 2,3',
        '22: 2',
        '23: 4,8,13,14,19,25,26,27,28',
        '24: 11,12',
        '25: 4,8,13,14,20,21,22,23,24',
        '26: 3',
        '27: 2,8,11,14',
        '29: 6,13,17,24,33,34',
        '31: 2,3,4',
        '32: 2,8,13,16,21,27,31,32,36,40,44,49,50,55,56,60,61',
        '33: 2,9,14,21,23,30,33,38,40,43,50,51,58,59,60',
      ]);
    });

    it('gives each outlined helper the span of the function it was extracted from', () => {
      // The arrow on line 2 and the `item => item.name` callback on line 32
      // are outlined into top-level declarations. Each must carry its own
      // span: assigning them the enclosing component's span (1:0-34:1) would
      // make every frame inside them blame the whole component.
      expect(nodesOfType(ast, 'FunctionDeclaration').map(formatLoc)).toEqual([
        '1:0-34:1', // Component itself
        '32:32-32:49', // `item => item.name`
        '2:18-4:3', // `onClick`
      ]);
    });

    it('does not smear a location over the interior of a materialized node', () => {
      // `const offset = -42;` materializes a UnaryExpression wrapping a
      // NumericLiteral. Only the root of that pair represents the source
      // construct; copying the location onto the operand as well is invisible
      // to the source map (it just repeats an already-mapped position) but is
      // exactly the recursive smearing that makes a location meaningless.
      const negations = nodesOfType(ast, 'UnaryExpression');
      expect(negations).toHaveLength(1);
      expect(formatLoc(negations[0])).toBe('5:17-5:20');
      expect(formatLoc(negations[0].argument)).toBe('<none>');
    });

    it('leaves memoization scaffolding unmapped', () => {
      // Rule: compiler-only nodes stay unmapped. Giving the memo cache or its
      // guards a nearby user location makes stack traces blame user code for
      // machinery the user never wrote.
      const cacheDeclarations = nodesOfType(ast, 'VariableDeclaration').filter(
        node =>
          node.declarations?.some(
            (declarator: any) =>
              declarator.id?.type === 'Identifier' &&
              declarator.id.name === '$',
          ),
      );
      expect(cacheDeclarations).toHaveLength(1);
      expect(formatLoc(cacheDeclarations[0])).toBe('<none>');

      // The fixture contains no `if` of its own, so every `IfStatement` in the
      // output is a memo cache guard.
      const guards = nodesOfType(ast, 'IfStatement');
      expect(guards.length).toBeGreaterThan(0);
      expect(guards.map(formatLoc)).toEqual(guards.map(() => '<none>'));
    });
  });
});
