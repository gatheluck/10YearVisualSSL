# 10 Year Visual SSL

過去10年の視覚ドメイン自己教師あり学習（SSL）手法を、**ABCI 以外の一般的な
環境で動く形**に移植した公開用パッケージ。ABCI 対応は疎結合なモジュールとして
分離し、コアが ABCI を前提としない。

**現在 private。監査後に公開する。**

## 状態

| 部品 | 状態 |
|---|---|
| `bin/contract-test.py` | **実装・テスト済み**。移植完了を機械で判定する |
| `platforms/` | **実装・テスト済み**。実行基盤の分離。`local` だけで完結する |
| アダプタ | 未着手。パイロットは `1_context_prediction` と `VideoGen`(LTX-2) |
| ランチャ | 未着手 |
| `LICENSE` | **MIT**（Copyright (c) 2026 LIMIT.Lab）|

## ライセンス

**このリポジトリのコードは MIT**（`LICENSE`、Copyright (c) 2026 LIMIT.Lab）。

ただし MIT が及ぶのは**我々が書いたコードだけ**である。

| 対象 | 扱い |
|---|---|
| 我々のコード | MIT |
| 著者の公開コード | **複製しない。** pinned submodule で参照する。各リポジトリのライセンスがそのまま適用される |
| 二次的著作物と判定したもの | **このリポジトリに含めない**（Capture 側 `official-manifest.txt` で `derivative` と判定した4件）|

著者リポジトリ31件のライセンスと扱いは Capture 側 `docs/INVENTORY.md` にある。
**うち12件が非商用ライセンス。** submodule 参照なので再配布は発生しないが、
**利用の可否は別途の判断が要る。**

## 設計文書の在り処

**設計の正は Capture 側リポジトリ**（`gatheluck/10YearVisualSSLCapturePrivate`、
永久 private）にある。実装がこちらへ移っても出所を1つに保つため。

| 文書 | 内容 |
|---|---|
| `docs/DESIGN.md` | 設計の哲学と根拠 |
| `docs/CONTRACT.md` | **アダプタ契約** |
| `docs/INVENTORY.md` | 著者リポジトリ31件の棚卸しと扱いの推奨 |

## 実行基盤

**特定の計算機環境での実行はオプショナル。** 既定は手元（`platforms/local`）で、
それだけで完結する。基盤対応は疎結合なモジュールで、コアはどれも前提としない。

分離は**機構で守っている** — `tests/test_platform_isolation.py`。
詳細は [docs/PLATFORMS.md](docs/PLATFORMS.md)。

## 開発の進め方

厳格な TDD。**必ず終了コードで判定する。**

```bash
./tests/run-tests.sh; echo "EXIT=$?"
git config core.hooksPath .githooks   # clone ごとに1回
```

規則は [CLAUDE.md](CLAUDE.md)。標準ライブラリのみで動く。
