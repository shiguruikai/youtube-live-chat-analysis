import argparse
import asyncio
import collections
import io
import json
import math
import os
import re
import shutil
from functools import reduce
from typing import TypedDict

import aiohttp
import cairosvg
import emoji
import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import yt_dlp
from matplotlib.axes import Axes
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from PIL import Image


# TODO: docstring の書き方を調べてドキュメントを書く。


class ChatItem(TypedDict):
    text: str
    emoji_text_list: list[str]
    emoji_text_to_url: dict[str, str]
    author_name: str
    timestamp_usec: int
    offset_time_msec: int


class VideoInfo(TypedDict):
    url: str
    id: str
    title: str


def load_live_chat_file(file) -> list[ChatItem]:
    chat_items: list[ChatItem] = []

    with open(file, encoding="utf-8", mode="r") as f:
        try:
            for line_number, line in enumerate(f, 1):
                chat: dict = json.loads(line)
                chat_item_action: dict = chat["replayChatItemAction"]
                actions: list[dict] = chat_item_action["actions"]

                for action in actions:
                    if "addChatItemAction" not in action.keys():
                        continue

                    item: dict = action["addChatItemAction"]["item"]

                    message_renderer: dict = item.get(
                        "liveChatTextMessageRenderer"
                    ) or item.get("liveChatPaidMessageRenderer")

                    if (
                        message_renderer is None
                        or "message" not in message_renderer.keys()
                    ):
                        continue

                    text: str = ""
                    emoji_text_list = []
                    emoji_text_to_url: dict[str, str] = {}
                    author_name: str = ""

                    messages: list[dict] = message_renderer["message"]["runs"]
                    for message in messages:
                        if "text" in message.keys():
                            text += message["text"]
                        elif "emoji" in message.keys():
                            message_emoji: dict = message["emoji"]

                            emoji_text = ""
                            if emoji.is_emoji(message_emoji["emojiId"]):
                                emoji_text = message_emoji["emojiId"]
                            elif "shortcuts" in message_emoji.keys():
                                emoji_text = message_emoji["shortcuts"][0]

                            emoji_text_list.append(emoji_text)

                            text += emoji_text

                            emoji_thumb_list: list[dict] = message_emoji["image"][
                                "thumbnails"
                            ]

                            if emoji_thumb_list:
                                # サイズ情報があれば最もサイズが大きいサムネイルを探す。
                                emoji_thumb = max(
                                    emoji_thumb_list,
                                    key=lambda it: (
                                        it.get("width", 0) * it.get("height", 0)
                                    ),
                                )
                                emoji_text_to_url[emoji_text] = emoji_thumb["url"]

                            author_name = message_renderer.get("authorName", {}).get(
                                "simpleText", ""
                            )

                    chat_items.append(
                        {
                            "text": text,
                            "emoji_text_list": emoji_text_list,
                            "emoji_text_to_url": emoji_text_to_url,
                            "author_name": author_name,
                            "timestamp_usec": int(message_renderer["timestampUsec"]),
                            "offset_time_msec": int(
                                chat_item_action["videoOffsetTimeMsec"]
                            ),
                        }
                    )
        except Exception:
            print(f"{line_number}行目で失敗しました。")
            raise

    return chat_items


def fetch_chat_items(
    url_or_video_id: str,
    refetch: bool = False,
    verbose: bool = False,
) -> tuple[VideoInfo, list[ChatItem]]:
    if url_or_video_id.startswith("http"):
        video_id = get_youtube_id(url_or_video_id)
        if video_id is None:
            raise RuntimeError(f"invalid url {url_or_video_id}")
        url = f"https://www.youtube.com/watch?v={video_id}"
    else:
        video_id = url_or_video_id
        url = f"https://www.youtube.com/watch?v={url_or_video_id}"

    cache_dir_path = os.path.join(os.path.dirname(__file__), ".cache")

    dl_params = {
        "skip_download": True,
        "verbose": verbose,
        "outtmpl": os.path.join(cache_dir_path, "%(title)s [%(id)s]"),
        # ライブチャットのファイルをダウンロードするための設定
        "writesubtitles": True,
        "subtitleslangs": ["live_chat"],
    }

    ydl = yt_dlp.YoutubeDL(dl_params)

    chat_file_path = find_live_chat_file_path(cache_dir_path, video_id)

    if refetch or chat_file_path is None:
        ret_code = ydl.download(url)
        chat_file_path = find_live_chat_file_path(cache_dir_path, video_id)
        if ret_code != 0 or chat_file_path is None:
            raise RuntimeError("ライブチャットの取得に失敗しました。")

    title = re.sub(
        rf" \[{video_id}\]\.live_chat\.json$", "", os.path.basename(chat_file_path)
    )

    chat_items = load_live_chat_file(chat_file_path)

    info: VideoInfo = {
        "url": url,
        "id": video_id,
        "title": title,
    }

    return info, chat_items


def find_live_chat_file_path(directory: str, video_id: str) -> str | None:
    if not os.path.isdir(directory):
        return None

    json_files = [
        f for f in os.listdir(directory) if f.endswith(f"[{video_id}].live_chat.json")
    ]
    file_paths = [os.path.join(directory, file) for file in json_files]
    latest_file_path = max(file_paths, key=os.path.getmtime, default=None)
    return latest_file_path


def get_youtube_id(url: str) -> str | None:
    m = re.search(r"(?<=\?v=)[^&]+", url)
    if m is None:
        return None
    return m.group()


class FetchedEmoji(TypedDict):
    emoji_text: str
    url: str
    image: Image.Image


async def async_fetch_emoji(
    session: aiohttp.ClientSession,
    sem: asyncio.Semaphore,
    emoji_text: str,
    url: str,
) -> FetchedEmoji | None:
    async with sem:
        try:
            async with session.get(url) as res:
                content_bytes = await res.read()
        except Exception as e:
            print(f"[warn] 画像の取得に失敗しました。 url: {url}")
            print(e)
            return None

    try:
        # svgファイルは、Pillowがサポートしていないため、pngに変換してから読み込む。
        if url.endswith(".svg"):
            image = Image.open(
                io.BytesIO(
                    cairosvg.svg2png(
                        content_bytes,
                        output_width=64,
                        output_height=None,
                    )
                )
            )
        else:
            image = Image.open(io.BytesIO(content_bytes))
    except Exception as e:
        print(f"[warn] 画像の読み込みに失敗しました。 emoji_text: {emoji_text} url: {url}")
        print(e)
        return None

    return {"emoji_text": emoji_text, "url": url, "image": image}


def plot_image(x, y, image: OffsetImage, ax: Axes | None = None):
    if ax is None:
        ax = plt.gca()

    ab = AnnotationBbox(
        image,
        (x, y),
        xycoords="data",
        # フレームを非表示
        frameon=False,
    )
    ax.add_artist(ab)
    ax.plot(x, y, alpha=0)


def set_semilogy(ax: Axes | None = None):
    if ax is None:
        ax = plt.gca()
    ax.semilogy(base=2)
    ax.autoscale()
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f"))


def setup_matplotlib(font_size: int):
    plt.rcParams["font.size"] = font_size

    # 日本語フォント指定しないと文字化けする。
    # 新たに追加したフォントを使用する場合は、matplotlibのキャッシュを削除しないと適用されない場合がある。
    plt.rcParams["font.sans-serif"] = [
        "Meiryo",
        "Hiragino Sans",
        "IPAexGothic",
        "IPAPGothic",
    ]
    plt.rcParams["font.family"] = "sans-serif"


def ema(df: pd.DataFrame, span: int):
    """指数平滑移動平均（初期値は単純移動平均と同じ）"""
    init_value = df.iloc[:span].rolling(span).mean()
    return pd.concat([init_value, df.iloc[span:]]).ewm(span=span, adjust=False)


_GRID_PARAMS = {
    "which": "major",
    "axis": "both",
    "color": "gray",
    "linestyle": "--",
    "linewidth": 0.6,
    "alpha": 0.6,
}

MAX_CONCURRENT_REQUESTS = 4


class AnalyzedTimestamp(TypedDict):
    """
    Attributes
    ----------
    timestamp :
        タイムスタンプ
    emoji_list :
        絵文字のテキストと使用回数のタプルのリスト
    """

    timestamp: pd.Timestamp
    emoji_list: list[tuple[str, float]]


class AnalyzeResult(TypedDict):
    url: str
    video_id: str
    video_title: str
    emoji_text_to_url: dict[str, str]
    analyzed_list: list[AnalyzedTimestamp]


async def analyze_live_chat(
    url_or_video_id: str,
    out: str | None = None,
    min_emoji_count: int = 1,
    max_emoji_plot: int = 9,
    count_repeated_emoji: bool = False,
    font_size: int = 12,
    emoji_size: int = 20,
    semilogy: bool = False,
    gui: bool = False,
    size: tuple[int, int] = (768, 768),
    refetch: bool = False,
    need_title: bool = False,
    verbose: bool = False,
) -> AnalyzeResult:
    setup_matplotlib(font_size=font_size)

    video_info, chat_items = fetch_chat_items(
        url_or_video_id, refetch=refetch, verbose=verbose
    )

    # 配信開始前のチャットを除外する。
    chat_items = [it for it in chat_items if it["offset_time_msec"] > 0]

    emoji_text_to_url: dict[str, str] = reduce(
        lambda acc, it: {**acc, **it},
        (it["emoji_text_to_url"] for it in chat_items),
        {},
    )

    request_sem = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)

    async with aiohttp.ClientSession() as session:
        # 全ての絵文字の画像を非同期で取得する。
        fetched_emoji_futures = [
            async_fetch_emoji(session, request_sem, k, v)
            for k, v in emoji_text_to_url.items()
        ]

        fig = plt.figure()

        fig.set_size_inches(w=size[0] / fig.dpi, h=size[1] / fig.dpi)

        if need_title:
            fig.suptitle(f"{video_info['title']} [{video_info['id']}]", wrap=True)

        timestamp_df = pd.DataFrame(chat_items, columns=["offset_time_msec"])
        timestamp_df = timestamp_df.apply(lambda it: pd.to_datetime(it, unit="ms"))
        timestamp_df.rename(columns={"offset_time_msec": "timestamp"}, inplace=True)
        timestamp_df.set_index("timestamp", inplace=True)
        timestamp_df.sort_index(ascending=True, inplace=True)

        timestamp_period: pd.Timedelta = timestamp_df.index[-1] - timestamp_df.index[0]

        timestamp_period_total_sec = timestamp_period.total_seconds()

        ################################################################################
        # (1) チャットの数
        ################################################################################

        ax1 = fig.add_subplot(7, 1, (1, 2))
        ax1.grid(**_GRID_PARAMS)
        ax1.set_xlabel("経過時間")
        ax1.set_ylabel("チャットの数")
        ax1.set_title("チャットの数")
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))

        # ヒストグラムの棒の数は、とりあえず平方根にしておく。
        ax1.hist(timestamp_df.index, bins=int(math.sqrt(len(timestamp_df))))

        if semilogy:
            set_semilogy(ax1)

        ################################################################################
        # (2) チャットの速さ
        ################################################################################

        ax2 = fig.add_subplot(7, 1, (3, 4))
        ax2.grid(**_GRID_PARAMS)
        ax2.set_xlabel("経過時間")
        ax2.set_ylabel("チャットの数／秒")
        ax2.set_title("チャットの速さ")

        ma_window = max(1, int(timestamp_period_total_sec * 0.02))

        chat_speed_df = timestamp_df.copy()

        # 1秒毎の数をカウントし、移動平均をプロットする。
        chat_speed_df["count"] = 0
        chat_speed_df = chat_speed_df.resample("1s").count()

        # NOTE: 参考までに残す。
        # 単純移動平均
        # chat_speed_sam_df = chat_speed_df.rolling(ma_window).mean()
        # ax2.plot(chat_speed_sam_df, label=f"SMA({ma_window}秒)")

        # 指数平滑移動平均
        chat_speed_ema_df = ema(chat_speed_df, ma_window).mean()
        ax2.plot(chat_speed_ema_df, label=f"EMA({ma_window}秒)")

        # NOTE: 参考までに残す。
        # 線形加重移動平均
        # weights = np.linspace(1, ma_window, ma_window)
        # sum_weights = weights.sum()
        # chat_speed_lwma_df = chat_speed_df.rolling(ma_window).apply(
        #     lambda x: np.sum(x * weights) / sum_weights, raw=True
        # )
        # ax2.plot(chat_speed_lwma_df, label=f"LWMA({ma_window}秒)")

        chat_speed_ma_df = chat_speed_ema_df

        # 移動平均が計算できない最初の期間の値がNaNになり、最初の期間がグラフにプロットされなくなるため、
        # 一番最初の非NaNの値を期初と期末にプロットして、X軸が他のグラフの範囲と同じになるようにする。
        valid_value = chat_speed_ma_df.loc[
            chat_speed_ma_df.first_valid_index(), "count"
        ]
        ax2.plot(timestamp_df.index[0], valid_value, alpha=0)
        ax2.plot(timestamp_df.index[-1], valid_value, alpha=0)

        # 上位10%のタイムスタンプを抽出
        top_timestamp_df = chat_speed_ma_df[
            chat_speed_ma_df["count"].rank(pct=True) > 0.9
        ]

        # 上位10%を赤い太線で上書きする。
        ax2.plot(
            top_timestamp_df["count"].asfreq("1s"),
            color="red",
            linewidth=1.5,
            label="上位10%",
        )

        ##################
        # X軸の目盛り設定
        ##################

        # タイムスタンプとの差の最小間隔
        xticks_interval_threshold = pd.Timedelta(
            seconds=max(1, int(timestamp_period_total_sec * 0.025))
        )
        # 上位10%のうち1つ前のタイムスタンプとの差
        time_diff = top_timestamp_df.index.to_series().diff()
        # タイムスタンプの差が、しきい値以上またはNaT(1番目の差)を抽出
        chat_speed_xticks_ser = time_diff[
            (time_diff >= xticks_interval_threshold) | time_diff.isnull()
        ]
        # datetimeをFixedLocatorに使うには、数値に変換する必要がある。
        x_major_locator = ticker.FixedLocator(
            mdates.date2num(chat_speed_xticks_ser.index)
        )
        ax2.xaxis.set_major_locator(x_major_locator)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax2.xaxis.set_tick_params(rotation=45)

        # 凡例の表示
        ax2.legend(
            loc="upper left", bbox_to_anchor=(0, 1), ncol=1, fontsize=font_size * 0.8
        )

        if semilogy:
            set_semilogy(ax2)

        ################################################################################
        # (3) 絵文字の数
        ################################################################################
        ax3 = fig.add_subplot(7, 1, (5, 7), sharex=ax1)
        ax3.grid(**_GRID_PARAMS)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
        ax3.set_xlabel("経過時間")
        ax3.set_ylabel("絵文字の数")

        emoji_df_resample_sec = max(1, int(timestamp_period_total_sec * 0.035))

        ax3.set_title(f"絵文字の数（{emoji_df_resample_sec}秒毎）")

        emoji_df = timestamp_df.copy()

        # 念のためデータサイズをチェック
        assert len(emoji_df.index) == len(chat_items)

        # 絵文字の使用回数列を追加
        for i, emoji_text_list in zip(
            emoji_df.index, (it["emoji_text_list"] for it in chat_items)
        ):
            if count_repeated_emoji:
                for emoji_text, count in collections.Counter(emoji_text_list).items():
                    emoji_df.loc[i, emoji_text] = count
            else:
                for emoji_text in set(emoji_text_list):
                    emoji_df.loc[i, emoji_text] = 1

        # 一定時間毎に集約
        resampled_emoji_df = emoji_df.resample(f"{emoji_df_resample_sec}s").sum()

        fetched_emoji_list: list[FetchedEmoji | None] = await asyncio.gather(
            *fetched_emoji_futures
        )

        emoji_text_to_image = {
            it["emoji_text"]: it["image"] for it in fetched_emoji_list if it is not None
        }

        for index, row in resampled_emoji_df.iterrows():
            # 使用回数がしきい値以上の絵文字を使用回数の低い順にループ
            for emoji_text, count in (
                row.sort_values(ascending=False)[:max_emoji_plot][::-1]
            ).items():
                # 使用回数が最小値に満たない絵文字はプロットしない。
                if count < min_emoji_count:
                    continue

                img = emoji_text_to_image.get(emoji_text)
                if img is None:
                    continue

                zoom = emoji_size / img.width
                offset_img = OffsetImage(img, zoom=zoom)

                plot_image(x=index, y=count, image=offset_img, ax=ax3)

        # 軸の自動調整
        ax3.autoscale()

        if semilogy:
            set_semilogy(ax3)
        else:
            # 絵文字の画像がはみ出すため、Y軸を少し拡大しておく。
            _, y_top = ax3.get_ylim()
            f = emoji_size * 0.003
            ax3.set_ylim(bottom=y_top * -f, top=y_top * (1 + f))

        # 余白部分の自動調整
        fig.tight_layout()

        if out:
            out_dir_path = os.path.join(out)
            os.makedirs(out_dir_path, exist_ok=True)
            out_file_path = os.path.join(
                out_dir_path, f"{video_info['title']} [{video_info['id']}].png"
            )
            plt.savefig(out_file_path)

        if gui:
            plt.show()

        analyzed_list: list[AnalyzedTimestamp] = []
        time_range = pd.Timedelta(seconds=ma_window // 2)
        for time in chat_speed_xticks_ser.index:
            sum_within_range = emoji_df.loc[time - time_range : time + time_range].sum()
            sum_within_range = sum_within_range[sum_within_range >= 1]
            sum_within_range.sort_values(ascending=False, inplace=True)
            analyzed_list.append(
                {
                    "timestamp": time,
                    "emoji_list": [(k, v) for k, v in sum_within_range.items()],
                }
            )

        return {
            "url": video_info["url"],
            "video_id": video_info["id"],
            "video_title": video_info["title"],
            "analyzed_list": analyzed_list,
            "emoji_text_to_url": emoji_text_to_url,
        }


async def main():
    parser = argparse.ArgumentParser(description="YouTube動画のライブコメントを分析してグラフの画像を生成するツール")
    parser.add_argument(
        "--url",
        metavar="URL",
        type=str,
        help="YouTube動画のURL",
    )
    parser.add_argument(
        "--id",
        metavar="ID",
        type=str,
        help="YouTubeの動画ID",
    )
    parser.add_argument(
        "--out", metavar="DIR", type=str, help="画像の保存先ディレクトリ（指定しない場合は保存しない）"
    )
    parser.add_argument(
        "--size",
        type=int,
        nargs=2,
        metavar=("W", "H"),
        default=(768, 768),
        help="画像全体のサイズ (default: %(default)s)",
    )
    parser.add_argument(
        "--font_size",
        metavar="N",
        type=int,
        default=10,
        help="描画するテキストのサイズ (default: %(default)s)",
    )
    parser.add_argument(
        "--emoji_size",
        metavar="N",
        type=int,
        default=20,
        help="描画する絵文字のサイズ (default: %(default)s)",
    )
    parser.add_argument(
        "--min_emoji_count",
        metavar="N",
        type=int,
        default=1,
        help="一定間隔でプロットする絵文字の最小使用回数 (default: %(default)s)",
    )
    parser.add_argument(
        "--max_emoji_plot",
        metavar="N",
        type=int,
        default=9,
        help="一定間隔でプロットする絵文字の最大数 (default: %(default)s)",
    )
    parser.add_argument(
        "--count_repeated_emoji",
        action="store_true",
        help="1回のチャットで繰り返し使用された絵文字の数もカウントする。",
    )
    parser.add_argument("--semilogy", action="store_true", help="Y軸を対数スケールで表示する。")
    parser.add_argument("--gui", action="store_true", help="matplotlibのGUIを表示する。")
    parser.add_argument(
        "--refetch",
        action="store_true",
        help="動画とライブチャットの情報をキャッシュから読み込まずに再取得する。",
    )
    parser.add_argument(
        "--remove_matplotlib_cache",
        action="store_true",
        help="matplotlibのキャッシュを削除してから実行する。",
    )
    parser.add_argument(
        "--title",
        action="store_true",
        help="動画のタイトルを追加する。",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="詳細な処理情報を出力する。")

    args = parser.parse_args()

    if not args.url and not args.id:
        raise RuntimeError("--url または --id を指定してください。")

    if args.remove_matplotlib_cache:
        shutil.rmtree(matplotlib.get_cachedir(), ignore_errors=True)

    await analyze_live_chat(
        url_or_video_id=args.url or args.id,
        out=args.out,
        min_emoji_count=args.min_emoji_count,
        max_emoji_plot=args.max_emoji_plot,
        font_size=args.font_size,
        emoji_size=args.emoji_size,
        count_repeated_emoji=args.count_repeated_emoji,
        semilogy=args.semilogy,
        gui=args.gui,
        size=args.size,
        refetch=args.refetch,
        need_title=args.title,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    asyncio.run(main())
