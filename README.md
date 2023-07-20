# YouTube Live Chat Analytics
YouTubeのライブチャットを分析するためのコマンドラインツールとWeb UI。

## 特徴

- 指定したYouTubeのライブチャットを分析し、チャットの数、チャットの速さ、絵文字の使用回数のグラフを作成する。
- チャットに勢いがある時のタイムスタンプとリンクを表示する。

<p>
    <img src="https://github.com/shiguruikai/screenshots/raw/main/ylca_webui_002.png" width="400">
    <img src="https://github.com/shiguruikai/screenshots/raw/main/ylca_webui_003.png" width="400">
</p>

## バージョン

Python 3.10 で動作します。<br>
それ以外のバージョンでも動作する可能性はありますが未検証です。

## インストール

### 1. GTKランタイムのインストール

svgファイルを読み込む処理に[CairoSVG](https://github.com/Kozea/CairoSVG)を使用しているため、GTKランタイムをインストールする必要があります。<br>
ランタイムのdllが正しく読み込めない場合、実行時にエラーが発生します。<br>

（参考リンク）
- https://github.com/Kozea/CairoSVG/issues/329
- https://github.com/Kozea/CairoSVG/issues/388

Windows用のGTKランタイムのインストーラーは、以下からダウンロードできます。<br>
https://github.com/tschoonj/GTK-for-Windows-Runtime-Environment-Installer/releases

デフォルト設定でインストールして問題ありません。

### 2. Pythonのインストール

Pythonのインストール方法は、自分で調べてください。

### 3. インストール

Python仮想環境は、他のを使ってもいいです。

```shell
git clone https://github.com/shiguruikai/youtube-live-chat-analysis
cd youtube-live-chat-analysis
python -m venv venv
./venv/Scripts/activate
pip install -r requirements.txt
```

## 使い方

### コマンドラインの場合

```shell
cd youtube-live-chat-analysis
./venv/Scripts/activate
```

```
python youtube_live_chat_analysis.py ＜引数＞
```

ヘルプメッセージ：
```
usage: youtube_live_chat_analysis.py [-h] [--url URL] [--id ID] [--out DIR]
                                     [--size W H] [--font_size N]
                                     [--emoji_size N] [--min_emoji_count N]
                                     [--max_emoji_plot N]
                                     [--count_repeated_emoji] [--semilogy]
                                     [--gui] [--refetch]
                                     [--remove_matplotlib_cache] [--title]
                                     [-v]

YouTube動画のライブコメントを分析してグラフの画像を生成するツール

options:
  -h, --help            show this help message and exit
  --url URL             YouTube動画のURL
  --id ID               YouTubeの動画ID
  --out DIR             画像の保存先ディレクトリ（指定しない場合は保存しない）
  --size W H            画像全体のサイズ (default: (768, 768))
  --font_size N         描画するテキストのサイズ (default: 10)
  --emoji_size N        描画する絵文字のサイズ (default: 20)
  --min_emoji_count N   一定間隔でプロットする絵文字の最小使用回数 (default: 1)
  --max_emoji_plot N    一定間隔でプロットする絵文字の最大数 (default: 9)
  --count_repeated_emoji
                        1回のチャットで繰り返し使用された絵文字の数もカウントする。
  --semilogy            Y軸を対数スケールで表示する。
  --gui                 matplotlibのGUIを表示する。
  --refetch             動画とライブチャットの情報をキャッシュから読み込まずに再取得する。
  --remove_matplotlib_cache
                        matplotlibのキャッシュを削除してから実行する。
  --title               動画のタイトルを追加する。
  -v, --verbose         詳細な処理情報を出力する。
```

### Web UIの場合

以下のコマンドを実行し、表示されたURLをブラウザで開いてください。<br>
一応、サーバーの起動が完了すると、自動的にブラウザで開かれるはずです。

```shell
cd youtube-live-chat-analysis
./venv/Scripts/activate
```

```
streamlit run app.py
```

または

コマンドの代わりに`webui.bat`を実行する。（実はライブラリのインストールも自動でやれる）

## License

[MIT](/LICENSE)
