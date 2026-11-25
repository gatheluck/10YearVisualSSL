import { execFileSync, spawnSync } from "node:child_process";
import { readFileSync, writeFileSync, rmSync } from "node:fs";
import { resolve } from "node:path";

const git = (...args) => execFileSync("git", args, { encoding: "utf8" }).trim();
const receipt = resolve(git("rev-parse", "--git-path", "paper-website-tested-head"));
const head = git("rev-parse", "HEAD");
const mode = process.argv[2];

function requireCleanCheckout() {
  if (git("status", "--porcelain", "--untracked-files=all")) {
    throw new Error("Commit website changes before validating the push.");
  }
}

if (mode === "prepare") {
  rmSync(receipt, { force: true });
  requireCleanCheckout();
  const result = spawnSync(process.execPath, ["scripts/test-site.mjs"], { stdio: "inherit" });
  if (result.status !== 0) process.exit(result.status ?? 1);
  requireCleanCheckout();
  if (git("rev-parse", "HEAD") !== head) throw new Error("HEAD changed during validation.");
  writeFileSync(receipt, head, { mode: 0o600 });
} else if (mode === "check") {
  const updates = readFileSync(0, "utf8").trim();
  if (updates) {
    requireCleanCheckout();
    const branch = git("branch", "--show-current");
    if (!branch.startsWith("codex/") || branch === "codex/paper-website" ||
        updates.split("\n").some(line => {
          const [, commit, target] = line.split(" ");
          return commit !== head || target !== "refs/heads/" + branch;
        })) throw new Error("Push only the checked-out feature branch. Use PRs for release branches.");
    if (readFileSync(receipt, "utf8") !== head) throw new Error("Run npm run verify:push for this commit.");
  }
} else {
  throw new Error("Expected prepare or check.");
}
