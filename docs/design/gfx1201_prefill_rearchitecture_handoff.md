# Luna handoff: gfx1201 prefill rearchitecture

作業ブランチ: `feat/gfx1201-prefill-rearchitecture`

起点: `9938409e924ba0418b65dfae8e56304a070726a6`

計画本体: [gfx1201_prefill_rearchitecture.md](gfx1201_prefill_rearchitecture.md)

## 最初の作業

旧P2.2/P3/v2/P4とB1/B2の不合格は確定のまま残す。今回の最初の作業は
**新計画のR0からR1まで**。R1の判定とartifactをコミットして停止・報告する。
R2のMXFP4融合、production登録、既定値変更へ自動で進まない。

作業ツリーにユーザーの変更がある場合は、別worktreeを使う。既存の
`.gitignore`、`.tools/`、実験artifactにreset/cleanを行わない。

1. R0でprovenance、SDK、実際のGEMM shapeと呼出回数、基準入力、数値基準を固定する。
   原点は過去のA0の倍率ではなく、同じ環境のBF16 GEMMとproduction full emulation。
2. R1はK=64単位の転送、wide load、LDS配置、register accumulator、直接出力を
   一体のmappingとして実装する。候補はH1/H2に限定し、旧B1/B2の微修正を続けない。
3. correctness、実用的なGEMMとの差、VRAM、生成objectを記録する。
   M=256/N>=512の加重結果とM=64を分け、N=96はfallbackのまま評価する。
4. 採用済みdecode/MTP2/K8V4/graph/prefix reuseは変更しない。
   全shapeでA0比5倍やattention由来の絶対誤差を新ゲートへ流用しない。

既に公開Radianceソースを比較したため、この計画はsource-informedである。
source-blindなclean-roomと呼ばない。コードや表の転載には由来と再利用条件を
確認する。未解決ならコピーせず独立仕様・公式AMD資料を使い、閲覧履歴を記録する。

重みの並べ替えとscale foldingは別の変更。後者は丸めを伴うため別の数値modeと
品質評価が必要。元のdecode用重みを上書きせず、追加packed copyの全VRAMを計上する。
32GBで全モデルの2重保持が可能だと仮定しない。

## 準備コミットに含まれるもの

- 計画書とこの引継ぎ。
- 未登録のPython contract/prepare/quantize/launch placeholder。
- ビルドに接続されていないCU実装位置。
- `--describe`だけ動くbenchmark entry point。未実装stageは例外で停止する。
- vLLM/GPU初期化を避けたstandalone CPUテスト。

対応shapeでも`can_use_gfx1201_prefill_v3`は必ずFalse。
既存のimport/dispatch/env/CMakeへの接続はゼロである。

## 準備側の確認とLuna側の確認

準備環境では以下を確認した。GPU性能やモデル精度の試験ではない。

```bash
python tests/kernels/quantization/test_gfx1201_prefill_v3_contract.py
python benchmarks/kernels/benchmark_gfx1201_prefill_v3.py --describe
python -m compileall -q \
  vllm/model_executor/kernels/linear/mxfp4/gfx1201_prefill_v3.py \
  benchmarks/kernels/benchmark_gfx1201_prefill_v3.py \
  tests/kernels/quantization/test_gfx1201_prefill_v3_contract.py
```

standaloneは8 test methods、profile不適合は14 subcases。
未実装R0/R1の呼出しも非ゼロ終了し、結果ファイルを作らないことを確認する。

準備環境にはruff/pre-commit/ROCm toolchainが揃っていないため、全repository hooks、
mypy、GPU試験を通過したとは扱わない。Lunaは最初に既存環境のhooksを新規ファイルへ
実行し、format等を必要に応じて修正する。既存productionを巻き込む整形はしない。

最終目標はRadiance級のcold prefillとllama.cpp級のdecodeを同一構成で得ること。
R1終了はその最終目標の達成ではなく、次の実装判断に足る計測が完了した状態を指す。
