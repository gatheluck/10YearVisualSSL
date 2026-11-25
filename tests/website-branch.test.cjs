const assert = require("node:assert/strict");
const { execFileSync } = require("node:child_process");
const fs = require("node:fs");
const test = require("node:test");

test("the website checkout contains only site assets and their maintenance files", () => {
  const files = execFileSync("git", ["ls-files", "--stage", "-z"], { encoding: "utf8" })
    .split("\0").filter(Boolean).map(record => record.split("\t"));
  const allowed = /^(?:paper-site\/[^/]+\.(?:js|html|css|png)|tests\/(?:paper-[\w-]+\.cjs|website-branch\.test\.cjs)|scripts\/(?:test-site|push-check)\.mjs|\.github\/workflows\/paper-pages\.yml|\.githooks\/pre-(?:commit|push)|package(?:-lock)?\.json|\.gitignore|README\.md|AGENTS\.md|LICENSE)$/;
  const unexpected = files.filter(([mode, file]) => !/^100(?:644|755) /.test(mode) || !allowed.test(file));
  assert.deepEqual(unexpected.map(([, file]) => file).slice(0, 8), []);
  for (const file of ["paper-site/index.html", "paper-site/leaderboard.html", "paper-site/history-data.js",
    "paper-site/control-data.js", "paper-site/extended-data.js", "paper-site/results.js", "LICENSE"]) {
    assert.ok(files.some(([, path]) => path === file), file);
  }
});

test("the public artifact contains only the verified website", () => {
  const workflow = ".github/workflows/paper-pages.yml";
  assert.ok(fs.existsSync(workflow), "Missing Pages release workflow");
  const text = fs.readFileSync(workflow, "utf8");
  assert.match(text, /pull_request:\s+branches: \[codex\/paper-website, gh-pages\]/);
  assert.match(text, /push:\s+branches: \[gh-pages\]/);
  assert.match(text, /run: npm test/);
  assert.match(text, /path: paper-site\s/);
  assert.match(text, /needs: verify/);
  assert.match(text, /github\.event_name == 'push' && github\.ref == 'refs\/heads\/gh-pages'/);
});
