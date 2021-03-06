# tic

チック症など苦手な音について、スペクトログラムから判定し、**アンガーマネジメントとしてカウント**と**逆位相の音を再生**してくれるHSPに優しいツールです。

![tic](https://github.com/yasutakatou/tic/blob/pic/tic.gif)

## 動作ロジック 

 1. 予め録音した苦手な音をパラメータに与え、ツールを起動します。と同時に苦手な音がスペクトログラムに変換されます
 1. 苦手な音の半分の長さでマイクから今の環境音を録音します
 1. 録音された環境音はスペクトログラムに変換されます
 1. OpenCVにより環境音と苦手な音の画像の類似度を判定します
 1. 起動時に与えた閾値パラメータより**下**なら苦手な音の位相の音を生成し、再生。また怒りカウンターを加算して表示します

> OpenCVのアルゴリズム上、類似度が高ければ数値が低くなるのでご注意

## 使い方

 - まず、超がんばって嫌な音をサンプリングします。record.pyをバックグラウンドで動かして離籍中にこっそり・・とかいいかもしれません。

```
python record.py test.wav 8
```

こちらのように保存するファイル名と録音する長さを指定してください。

 - pythonのインストールと追加モジュール(OpenCVとか)をインストールして環境整備してください。

> たしかPyInstallerはOpenCV使うときは1バイナリ化できなかったので今回やってません

 - 嫌な音ファイルをなんとか入手したらtic.pyに読み込ませてガード開始。python tic.py (嫌な音ファイル) (閾値)として動かします。

```
python tic.py test.wav 70
```

咳払いとか突発的な音ではあまりうまいとこ動いてはくれません。連続的な環境音では70辺りを閾値に期待通りの作動をしました。
ま、MVPでありオモチャですね。
