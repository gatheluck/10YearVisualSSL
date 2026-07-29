#!/usr/bin/env bash
# 全テストを実行する。標準ライブラリのみ。追加依存なし。
# **必ず終了コードで判定する。** grep で成功文字列を探すと失敗を見逃す。
set -Eeuo pipefail
cd "$(dirname "$0")/.."
echo "== 構文チェック =="
for f in bin/*.py; do
  [ -e "$f" ] || continue
  python3 -c "import ast,sys;ast.parse(open(sys.argv[1]).read())" "$f"; echo "  ok $f"
done
for f in bin/*.sh .githooks/*; do
  [ -e "$f" ] || continue
  bash -n "$f"; echo "  ok $f"
done
echo
echo "== ユニットテスト =="
python3 -m unittest discover -s tests -v
echo
echo "全テスト通過"
