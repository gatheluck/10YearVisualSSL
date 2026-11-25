import { readdirSync } from "node:fs";
import { spawnSync } from "node:child_process";

const files = readdirSync("tests").filter(name => /^paper-.*\.cjs$/.test(name)).sort();
if (!files.length) throw new Error("Missing browser tests");
const env = { ...process.env };
for (const key of ["BOARD_HOME", "CONTROL_HOME", "HERO_URL", "HISTORY_HOME", "OVERVIEW_HOME", "OVERVIEW_URL"]) {
  delete env[key];
}
for (const args of [["--test", "tests/website-branch.test.cjs"], ...files.map(name => ["tests/" + name])]) {
  console.log(process.execPath, ...args);
  const result = spawnSync(process.execPath, args, { stdio: "inherit", env });
  if (result.error) throw result.error;
  if (result.status !== 0) process.exit(result.status ?? 1);
}
