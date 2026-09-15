# JALO

運転動画に映る車両を見つけ、車体の予測マスクに半透明の色を重ねるPythonの研究プロジェクトです。
対象は乗用車・トラック・バス。色は追跡IDごとにそろえ、検出枠は表示しません。

**現在のモデルは試作段階です。見逃しや背景への誤着色があり、実用的な認識品質には達していません。**
以下に、無料で利用できる実際の運転動画での結果を掲載しています。

## まず結果を見る

[評価動画とモデルをダウンロード](https://github.com/james-yusuke/jalo/releases/tag/v0.1.1)

同じ配布モデルを使い、学習に使用していない運転動画の3区間を処理しました。
**評価結果：車両を安定して認識・分離できていません。** 近くの車両でも未着色になり、着色された場面では車間の道路まで塗る例がありました。

![無料の運転動画での比較。左は元映像、右はJALOの予測。未着色と、複数車両・背景をまとめて塗る失敗例。](assets/i495_comparison.jpg)

左が元映像、右が予測です。事前に固定した9時刻を目視確認し、0:10・2:10の未着色例と4:18の誤着色例を掲載しています。
4:18では、ひとつのマスクが複数の車両と車間の背景をまとめて塗っています。予測マスクを手で修正する処理は行っていません。

| 元動画の区間 | 結果動画（各20秒） | 着色が発生したフレーム |
|---|---|---:|
| 0:00–0:20 | [MP4を見る・保存する](https://github.com/james-yusuke/jalo/releases/download/v0.1.1/i495_masks_000_020.mp4) | 3 / 600 |
| 2:00–2:20 | [MP4を見る・保存する](https://github.com/james-yusuke/jalo/releases/download/v0.1.1/i495_masks_120_140.mp4) | 0 / 600 |
| 4:00–4:20 | [MP4を見る・保存する](https://github.com/james-yusuke/jalo/releases/download/v0.1.1/i495_masks_240_260.mp4) | 213 / 600 |

「着色が発生した」は、正しく認識できたという意味ではありません。ダウンロードが始まった場合は、保存したMP4を動画プレーヤーで開いてください。

合計60秒・1,800フレーム、出力1280×720／30fps／H.264、音声なしです。
Mac／MPS、入力384×640、クラスしきい値0.3、マスクしきい値0.5、背景候補の除外あり、色の濃さ45%で評価しました。
元動画のフレーム間隔が一定ではないため、映像の時刻を保ちながら30fpsへそろえて処理しています。
全出力フレームの再読込、再生時間、記録された時刻との対応を確認しました。
詳細な記録・9時刻の比較画像・フレームごとの予測は[Releaseの添付ファイル](https://github.com/james-yusuke/jalo/releases/tag/v0.1.1)から取得できます。


この動画に正解の車両マスクはありません。上の結果は動作と着色の目視評価であり、認識精度のAPやRecallを示すものではありません。
動画を学習・モデル選択・しきい値調整には使っていません。

## 使用した無料の運転動画

[Driving eastbound on I-495 from the I-270 Spur to Cedar Lane (1 June 2026)](https://commons.wikimedia.org/wiki/File:Driving_eastbound_on_I-495_from_the_I-270_Spur_to_Cedar_Lane_(1_June_2026).webm)

作者：**Illegitimate Barrister**。ライセンス：**[CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/)**。
元動画は約5分、1280×720、音声なしです。上のリンクの「Original file」から無料で取得できます。

掲載した結果動画と比較画像は、この映像の一部を抽出し、JALOの予測色・比較用の配置や説明を加えたものです。
これらの映像・画像も**CC BY-SA 4.0**で提供します。再利用する場合は、作者・元動画・ライセンスを表示し、変更点を明記してください。
無料利用にはこの条件があり、著作権が放棄された動画ではありません。

## 改善実験の準備状況（開発版）

[開発版のソース・公開モデル・確認記録](https://github.com/james-yusuke/jalo/releases/tag/v0.2.0-alpha.1)を用意しました。
**認識品質の改善はまだ完了していません。配布している重みは引き続き `v0.1.1` と同じCOCO学習済みモデルです。**

この開発版では、現在の重みを車両ごとの局所マスク構成へ引き継ぐ処理、道路への誤着色の評価、動画単位のデータ分離、学習の再開処理を追加しました。
Mac／MPSで重みの移植とマスク層への勾配、既存モデルでのMP4保存を確認しました。この動作確認は、認識精度の改善を示す実験ではありません。

追加学習用として次の4動画を取得し、指定した時刻の340枚を抽出しました。作者名とライセンスは元の配布ページで確認できます。

| 今後の用途 | 動画（元の配布ページ） | 作者・ライセンス | 抽出済み画像 |
|---|---|---|---:|
| 学習 | [I-495](https://commons.wikimedia.org/wiki/File:Driving_eastbound_on_I-495_from_the_I-270_Spur_to_Cedar_Lane_(1_June_2026).webm) | Illegitimate Barrister・CC BY-SA 4.0 | 95枚 |
| 学習 | [Broad Creek → Jennifer Road](https://commons.wikimedia.org/wiki/File:Driving_from_Broad_Creek_to_Jennifer_Road_in_Annapolis,_Maryland_(1_June_2026).webm) | Illegitimate Barrister・CC BY-SA 4.0 | 95枚 |
| 検証・しきい値選択 | [Leaman Farm Road → Game Preserve Road](https://commons.wikimedia.org/wiki/File:Driving_from_Leaman_Farm_Road_to_Game_Preserve_Road_in_Gaithersburg,_Maryland_(1_June_2026).webm) | Illegitimate Barrister・CC BY-SA 4.0 | 95枚 |
| 未学習の最終評価 | [原州市の運転動画](https://commons.wikimedia.org/wiki/File:2020-04-16_원주시_도로주행.webm) | Choi Kwang-mo・[CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/) | 55枚 |

米国の3動画は15秒から300秒未満、原州市は15秒から180秒未満を3秒間隔で抽出しています。
各ページの「Original file」から無料で取得できます。[CC BY-SA 4.0の利用条件](https://creativecommons.org/licenses/by-sa/4.0/)に従い、再配布時には作者・出典・変更内容を表示してください。
既に結果を見たI-495は次の学習用に選び、今後の未見評価には使用しません。

**学習開始前の条件で停止しています。** 340枚すべての確認済み参照注釈が必要ですが、現時点ではI-495の4枚についてAI作成の輪郭と重畳表示を確認した段階です。
輪郭の補助抽出には、車体とタイヤ下の影・道路を分離できない例がありました。未確認の注釈を学習用・評価用の正解として使用していません。
新しい動画での追加学習、しきい値選択、最終評価、改善モデルの全編動画は未実施です。原州市の動画には予測を実行していません。
新しい精度値や「実用的になった」という評価は掲載していません。

<details>
<summary>開発版で追加した研究用の実行手順</summary>

この手順には、全画像の参照注釈と重畳表示の確認を完了したデータが必要です。添付の注釈例は4枚の途中成果であり、学習用データセットではありません。

```bash
python -m jalo prepare --dataset video-instances --root data/free_driving_adapt --sources data/free_driving_adapt/sources.json --annotations data/free_driving_adapt/annotations.json
python -m jalo train --config configs/free_driving_adapt.yaml --initialize checkpoints/jalo-coco-vehicles.pt --device mps
```

初期化では公開モデルのバックボーン、ピクセル特徴、Attention、正規化、FFN、分類層を移します。
矩形の補正出力はゼロ初期化し、移植した層を初期学習中に固定します。パラメータ14,252,556個／全15,156,129個を継承することを照合しました。
過学習確認は学習画像8枚で行い、そのマスクと勾配の確認記録がそろうまで次の段階へ進みません。
`--resume`では同じ設定・注釈・分割・予算記録を使い、モデル、optimizer、scheduler、乱数とデータ順を復元します。
学習・評価を共有の8時間／50,000更新以内で管理し、最終検証と最終評価の時間も確保します。

```bash
python -m jalo evaluate --checkpoint runs/free_driving_adapt/model/best.pt --baseline checkpoints/jalo-coco-vehicles.pt --config configs/free_driving_adapt.yaml --split val --device mps
python -m jalo evaluate --checkpoint runs/free_driving_adapt/model/validated.pt --baseline checkpoints/jalo-coco-vehicles.pt --config configs/free_driving_adapt.yaml --split test --device mps
python -m jalo export --checkpoint runs/free_driving_adapt/model/validated.pt --output checkpoints/jalo-local-vehicles.pt
```

検証で描画設定を固定してから、旧モデルと改善モデルの最終比較を一度だけ実行します。
着色画素Precision 90%、長辺32px以上の車両Recall 80%、背景・車内の誤着色率0.5%以下を目標とします。
車内が映らない場合は該当指標を欠測扱いにし、最終評価の車両が100件未満なら判定を保留します。
検出器とByteTrackを含む描画結果の両方で基準を満たした場合だけ、新構成の全編出力を許可します。
推論用ファイルには設定と評価記録を保持し、元の注釈フォルダへの依存をなくします。
**この手順全体による実データ学習・評価は、まだ完了していません。**

</details>

## 自分の動画で試す

### 1. ファイルをダウンロードする

- [ソースコードをZIPでダウンロード](https://github.com/james-yusuke/jalo/archive/refs/tags/v0.1.1.zip)して展開します。
- [学習済みモデルをダウンロード（約60 MB）](https://github.com/james-yusuke/jalo/releases/download/v0.1.1/jalo-coco-vehicles.pt)します。展開したフォルダに`checkpoints`フォルダを作り、その中に置きます。
- 処理したい動画を同じフォルダに置き、以下の例では`demo.webm`という名前にします。自分のMP4などの名前に読み替えても構いません。

### 2. 実行環境を用意する

[Python 3.11以上](https://www.python.org/downloads/)と[FFmpeg](https://ffmpeg.org/download.html)が必要です。FFmpegは`libx264`を含むものを使い、`ffmpeg`と`ffprobe`を実行できるようにしてください。
ソースを展開したフォルダでターミナルを開き、次を実行します。

```bash
python -m venv .venv
```

macOS／Linuxでは`source .venv/bin/activate`、Windows PowerShellでは`.venv\Scripts\Activate.ps1`で仮想環境を有効にします。
macOSなどで`python`が見つからない場合は、`python3`に読み替えてください。

```bash
python -m pip install .
python -m jalo doctor
```

### 3. 動画を処理する

まず20秒間を処理する例です。

```bash
python -m jalo demo --input demo.webm --checkpoint checkpoints/jalo-coco-vehicles.pt --render masks --foreground-only --duration 20 --output outputs/demo_masks.mp4 --device auto --no-preview
```

`outputs/demo_masks.mp4`を開くと結果を再生できます。全編を処理する場合は`--duration 20`を外します。
映像はH.264のMP4で保存し、元の音声があればAACで残します。色の濃さは45%です。
`--no-preview`を外すと処理中に表示できます。Spaceで一時停止、Q／Escで終了します。
同名のJSONLにはフレーム時刻、クラス、信頼度、追跡ID、予測マスクも保存します。

`--device auto`は利用できるGPUを選びます（CUDA → MacのMPS → CPU）。Mac／MPSで動作確認済み、CUDAは実機未検証です。

## モデルと学習データ

ImageNetで学習済みのResNet-18を特徴抽出に使用し、Attention、分類、矩形、マスクの各ヘッドをPyTorchで実装しています。
既存の検出・セグメンテーションモデルの重みは使用していません。

配布モデルはCOCO 2017公式trainから選んだ2,000枚（車両あり1,800枚／なし200枚）で学習した単一フレーム版です。
公式valの別の300枚でモデルを選択しました。運転動画での追加学習は含みません。
`v0.1.1`の重みは`v0.1.0`と同じで、今回の更新は動画の時刻処理の修正、運転動画での評価、案内の改善です。

| COCO検証画像300枚での指標 | 結果 |
|---|---:|
| 矩形AP | 1.03% |
| マスクAP | 4.66% |

これは1,500更新時点・入力384×640のCOCO形式APです。上の運転動画に対する精度ではありません。
学習設定、画像ID、評価値、ファイルの照合用ハッシュは[Release](https://github.com/james-yusuke/jalo/releases/tag/v0.1.1)に添付しています。
配布モデルは推論用で、学習状態の完全再開に必要なoptimizer等は含みません。

<details>
<summary>学習・研究用のコマンド</summary>

```bash
python -m jalo prepare --dataset coco-instances --root data/coco_vehicle
python -m jalo train --config configs/vehicle_masks.yaml --variant single --run-dir runs/coco_new --device auto
python -m jalo evaluate --checkpoint checkpoints/jalo-coco-vehicles.pt --config configs/vehicle_masks.yaml --split val --device auto
python -m jalo export --checkpoint runs/coco_new/best.pt --output checkpoints/my-model.pt
```

Releaseの`training_config.yaml`が元の学習設定です。リポジトリの設定は新規実験用に上限を10,000更新／3時間にしています。
時間Attentionの比較用に`single`／`temporal`／`gated`と`configs/mac_small.yaml`／`configs/full.yaml`も含みます。
BDD100Kでの実データ比較は未実施です。独自実装と学術的な新規性は別であり、YOLOに対する優位性を主張しません。

</details>

## ライセンス

ソースコードは[MIT License](LICENSE)です。掲載した運転動画の加工結果・比較画像は前述のCC BY-SA 4.0です。
データと外部の事前学習済み重みは各提供元の条件を参照してください。[COCOの利用条件](https://cocodataset.org/#termsofuse)も確認してください。
学習データ・元動画・モデルの重み・`scripts/`・`docs/`はGitに含めず、配布モデルと結果動画はReleaseに置いています。
