// Runs every roster optimizer at tiny budgets and small dimensions, in both the
// direct optimize() mode and the scalar ask/tell mode, and reports the number of
// objective calls each made. tests/test_budget_cap.py asserts none exceeds its
// budget (#345).
const path = require("path");

const modules = require(path.resolve(__dirname, "../docs/js/modules/index.js"));
const { Optimizer } = require(path.resolve(__dirname, "../docs/js/modules/base-optimizer.js"));

const BUDGETS = [0, 1, 2, 5, 10];
const DIMS = [1, 2, 10];
const out = [];

const roster = Object.keys(modules).filter(
    (k) => typeof modules[k] === "function" && modules[k].prototype instanceof Optimizer
);

for (const name of roster) {
    for (const budget of BUDGETS) {
        for (const dim of DIMS) {
            let calls = 0;
            const objective = (x) => {
                calls += 1;
                return x.reduce((s, v) => s + v * v, 0);
            };
            let opt = new modules[name](objective, budget, dim);
            opt.optimize();
            out.push({ name, mode: "optimize", budget, dim, calls, evaluations: opt.evaluations });

            calls = 0;
            opt = new modules[name](objective, budget, dim);
            if (typeof opt._run !== "function") continue; // legacy loop-owning port: no ask/tell yet
            for (;;) {
                const x = opt.suggestNext();
                if (x === null) break;
                opt.receiveUpdate(objective(x));
            }
            out.push({ name, mode: "asktell", budget, dim, calls, evaluations: opt.evaluations });
        }
    }
}
process.stdout.write(JSON.stringify(out));
