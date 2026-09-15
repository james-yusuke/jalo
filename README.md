# JALO

Python／PyTorchによる車載動画の物体検出・車両マスクの研究用実装です。
検出クエリ、Attention、分類・矩形・マスクヘッド、損失を実装し、外部の事前学習済み重みは特徴抽出のResNet-18（ImageNet）のみに使用します。

ソースはこのリポジトリ、学習済みモデルは **[タグ付きReleases](https://github.com/james-yusuke/jalo/releases)** で配布します。
`v0.1.0` の配布モデルは **COCOのみで学習した単一フレームの試作版** です。車載動画への追加学習は含みません。

## セットアップ

Python 3.11以上と、libx264を含むFFmpegが必要です。

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install .
# macOSでFFmpegを用意する場合
brew install ffmpeg
python -m jalo doctor
```

`--device auto` はCUDA → MPS → CPUの順で選択します。標準精度はFP32です。
Mac／MPSで確認済み、CUDAは実機未検証です。

## 学習済みモデルを使う

[v0.1.0](https://github.com/james-yusuke/jalo/releases/tag/v0.1.0) から `jalo-coco-vehicles.pt` を取得し、`checkpoints/` に置きます。
GitHub CLIで取得する場合は次を実行します。非公開リポジトリの場合はアクセス権のあるアカウントで認証してください。

```bash
gh release download v0.1.0 --repo james-yusuke/jalo \
  --pattern jalo-coco-vehicles.pt --dir checkpoints

python -m jalo demo --input demo.webm \
  --checkpoint checkpoints/jalo-coco-vehicles.pt \
  --render masks --foreground-only --duration 20 \
  --output outputs/demo_masks.mp4 --device auto --no-preview
```

車体の予測マスクに45%の半透明色を重ねます。矩形で代用せず、ByteTrackの追跡IDごとに色を付けます。
MP4はH.264／yuv420p／faststartで保存し、元音声があればAACへ変換して保持します。
フレーム時刻・クラス・信頼度・追跡ID・マスクRLEは同名のJSONLへ保存します。
`--no-preview` を外すとプレビューを表示し、Spaceで一時停止、Q／Escで終了できます。

配布ファイルは推論用の重み・構成です。optimizerや乱数状態を含むローカルの学習チェックポイントは別に保持します。
信頼できる提供元のファイルを使用し、Releaseの `SHA256SUMS` で照合してください。

## 無料で利用できる運転動画

入力例として、Wikimedia Commonsの次の動画を掲載します。

- **[Driving eastbound on I-495 from the I-270 Spur to Cedar Lane (1 June 2026)](https://commons.wikimedia.org/wiki/File:Driving_eastbound_on_I-495_from_the_I-270_Spur_to_Cedar_Lane_(1_June_2026).webm)**
- 作者：**Illegitimate Barrister**。投稿ページでは作者自身の撮影作品として掲載されています。
- ライセンス：**[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**。
- 約5分、1280×720、音声なし。動画ページの「Original file」から取得できます。

無料で利用・改変できますが、著作権の放棄を意味しません。作者・出典・ライセンスを表示し、変更点を明記してください。
加工した動画を配布する場合は、同じCC BY-SA 4.0の条件に従ってください。この動画は配布モデルの学習・モデル選択には使用していません。
動画ファイル自体はリポジトリに含めず、提供元のページを案内しています。

## 学習と評価

配布モデルの学習データはCOCO 2017公式trainから固定選択した2,000枚（車両あり1,800枚／なし200枚）です。
対象はcar・truck・bus、参照はCOCO公式インスタンスマスクです。別の公式val 300枚でモデルを選択しました。

```bash
python -m jalo prepare --dataset coco-instances --root data/coco_vehicle
python -m jalo train --config configs/vehicle_masks.yaml \
  --variant single --run-dir runs/coco_new --device auto

python -m jalo evaluate --checkpoint checkpoints/jalo-coco-vehicles.pt \
  --config configs/vehicle_masks.yaml --split val --device auto

# 自分で学習したチェックポイントから配布用ファイルを作成
python -m jalo export --checkpoint runs/coco_new/best.pt \
  --output checkpoints/my-model.pt
```

配布モデルは1,500更新時点です。公式valの固定300枚に対し、矩形AP **1.03%**、マスクAP **4.66%**。
しきい値・追跡処理前のCOCO形式APで、画像サイズは384×640です。
見逃しや背景への誤着色が多く、実用的な認識品質には達していません。運転判断に使うためのモデルではありません。
詳細な構成、学習条件、計測値、元チェックポイントの識別情報はReleaseの `MODEL_CARD.md` と `evaluation.json` に収録します。
Releaseの `training_config.yaml` が元の学習設定です。リポジトリの設定は、新規実験用に上限を10,000更新／3時間にしています。

時間Attentionの比較用に `single`／`temporal`／`gated` と `configs/mac_small.yaml`／`configs/full.yaml` も含みます。
BDD100Kでの実データ比較は未実施です。独自実装と学術的な新規性は別であり、YOLOに対する優位性を主張しません。

## 配布範囲とライセンス

ソースコードはMIT Licenseです。データ・動画・外部の事前学習済み重みは、それぞれの提供元の条件を参照してください。
COCO画像を著作権フリーとして再配布するものではありません。[COCOの利用条件](https://cocodataset.org/#termsofuse) も確認してください。

Gitには実行用の `jalo/`、基本設定、依存関係、READMEとLICENSEを登録します。
学習データ・動画・重み・ローカルの `scripts/`・`docs/` はGit履歴に含めません。モデルはタグ付きReleaseから取得してください。
