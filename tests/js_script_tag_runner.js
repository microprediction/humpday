/**
 * Run a documentation page's own script sequence the way a browser would, and then run every
 * optimizer the page ends up with.
 *
 * Node's CommonJS branch is the reason the parity tests never saw #342: `base-optimizer.js`
 * takes `require('./prng.js')` when `module` exists, and reads window globals when it does not.
 * Every page omitted `prng.js`, so in a browser the base module captured nulls for portableLog
 * and portableExp and three optimizers threw on their first call. This context therefore has a
 * `window` and deliberately has no `module`, which is the only configuration that can see it.
 *
 * Usage: node js_script_tag_runner.js <nTrials> <nDim> <page.html> [page.html ...]
 * Prints JSON: [{ page, scripts, ran, failures }, ...], one entry per page.
 */

const fs = require('fs');
const path = require('path');
const vm = require('vm');

const nTrials = Number(process.argv[2]);
const nDim = Number(process.argv[3]);
const pages = process.argv.slice(4);

function runPage(pagePath) {
const html = fs.readFileSync(pagePath, 'utf8');
const pageDir = path.dirname(pagePath);

// The library modules this page loads, in the order the page loads them. Page-level scripts are
// left out: they drive the DOM, which is not what is under test here.
const scripts = [];
const tag = /<script\s+src="([^"]+)"><\/script>/g;
let match;
while ((match = tag.exec(html)) !== null) {
    if (match[1].includes('js/modules/')) {
        scripts.push(match[1]);
    }
}

const context = vm.createContext({ console, Math, Date, JSON });
context.window = context;
context.globalThis = context;

const failures = {};
const declared = [];
for (const src of scripts) {
    const file = path.resolve(pageDir, src);
    const source = fs.readFileSync(file, 'utf8');
    // Every optimizer this page loads, named by its own source. A page that does not load
    // index.js has no HumpDayOptimizers roster, and a class declaration is lexically scoped
    // rather than a property of the global object, so neither is discoverable by scanning
    // `context` -- but both are reachable by name from inside it.
    for (const m of source.matchAll(/^class\s+(\w+)\s+extends\s+Optimizer\b/gm)) {
        declared.push(m[1]);
    }
    try {
        vm.runInContext(source, context, { filename: file });
    } catch (err) {
        failures[src] = `${err.constructor.name}: ${err.message}`;
    }
}

const roster = {};
for (const name of declared) {
    try {
        const cls = vm.runInContext(`typeof ${name} !== 'undefined' ? ${name} : null`, context);
        roster[name] = typeof cls === 'function' ? cls : null;
    } catch (err) {
        roster[name] = null;
    }
}
const ran = [];
const objective = (x) => x.reduce((s, v) => s + v * v, 0);

for (const name of Object.keys(roster)) {
    if (!roster[name]) {
        failures[name] = 'not defined by the scripts this page loads';
        continue;
    }
    try {
        const result = vm.runInContext(
            'globalThis.__run(globalThis.__name, globalThis.__objective, ' +
            'globalThis.__nTrials, globalThis.__nDim)',
            Object.assign(context, {
                __name: name,
                __objective: objective,
                __nTrials: nTrials,
                __nDim: nDim,
                __run: (n, f, t, d) => new roster[n](f, t, d).optimize(),
            })
        );
        const value = result && typeof result === 'object' ? result.bestValue : result;
        if (!Number.isFinite(Number(value))) {
            failures[name] = `returned ${JSON.stringify(result)}`;
        } else {
            ran.push(name);
        }
    } catch (err) {
        failures[name] = `${err.constructor.name}: ${err.message}`;
    }
}

return { page: pagePath, scripts, ran: ran.sort(), failures };
}

console.log(JSON.stringify(pages.map(runPage), null, 2));
