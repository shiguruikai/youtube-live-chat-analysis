import datetime
import os
import time

import numpy as np
import streamlit as st
from matplotlib import pyplot as plt

import youtube_live_chat_analysis as ya


def to_hms_str(seconds: float) -> str:
    return str(datetime.timedelta(seconds=int(seconds))).zfill(8)


page_title = "YouTube Live Chat Analytics"

st.set_page_config(
    page_title=page_title,
    page_icon=":bar_chart:",
    layout="centered",
    menu_items={
        "Get Help": "https://github.com/shiguruikai/youtube-live-chat-analysis"
    },
)

st.write(
    """
<style>
    .main .block-container {
        max-width: 54rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

root_container = st.container()

try:
    with root_container:
        st.title(page_title)

        with st.form("main_form"):
            url = st.text_input(
                "動画のURL", placeholder="https://www.youtube.com/watch?v="
            )

            with st.expander(":gear: 詳細設定", expanded=True):
                col1, col2, col3, col4 = st.columns(4, gap="medium")
                with col1:
                    width = st.slider(
                        "画像の横幅", value=768, min_value=512, max_value=1600, step=64
                    )
                with col2:
                    height = st.slider(
                        "画像の高さ", value=768, min_value=512, max_value=1600, step=64
                    )
                with col3:
                    font_size = st.slider(
                        "フォントサイズ", value=10, min_value=5, max_value=20, step=1
                    )
                with col4:
                    emoji_size = st.slider(
                        "絵文字サイズ", value=20, min_value=8, max_value=64, step=4
                    )
                col5, col6 = st.columns(2, gap="medium")
                with col5:
                    min_emoji_count = st.number_input(
                        "一定間隔でプロットする絵文字の最小使用回数", value=1, min_value=1, step=1
                    )
                with col6:
                    max_emoji_plot = st.number_input(
                        "一定間隔でプロットする絵文字の最大数", value=9, min_value=1, step=1
                    )
                count_repeated_emoji = st.checkbox(
                    "1回のチャットで繰り返し使用された絵文字の数もカウントする。", value=False
                )
                semilogy = st.checkbox("Y軸を対数スケールで表示する。", value=False)
                need_title = st.checkbox("グラフの画像に動画のタイトルを追加する。", value=False)
                refetch = st.checkbox("動画とライブチャットの情報をキャッシュから読み込まずに再取得する。", value=False)

            submitted = st.form_submit_button(
                "分析実行",
                type="primary",
                use_container_width=True,
            )

        def on_click_submit_button():
            if not url:
                st.error("動画のURLを入力してください。")
                return

            start = time.perf_counter()
            with st.spinner("しばらくお待ちください。ライブチャットの情報の取得には数分かかる場合があります。"):
                analyze_result = ya.analyze_live_chat(
                    url_or_video_id=url,
                    out=os.path.join(os.path.dirname(__file__), "out"),
                    min_emoji_count=min_emoji_count,
                    max_emoji_plot=max_emoji_plot,
                    count_repeated_emoji=count_repeated_emoji,
                    font_size=font_size,
                    emoji_size=emoji_size,
                    semilogy=semilogy,
                    size=(width, height),
                    refetch=refetch,
                    need_title=need_title,
                )
            elapsed = time.perf_counter() - start
            st.success(f"分析が完了しました。 処理時間：{to_hms_str(elapsed)}秒")

            st.divider()

            st.markdown(f"#### {analyze_result['video_title']}")

            st.pyplot(plt.gcf(), clear_figure=True)

            md = """
##### チャットの速さ上位10%
||経過時間|URL|
|--:|:--:|:--|
"""
            n = 0
            for s in (
                int(ns / 1e9) for ns in analyze_result["hot_timestamp"].astype(np.int64)
            ):
                n += 1
                hms = to_hms_str(s)
                link = f"{analyze_result['url']}&t={s}s"
                md += f"|{n}|{hms}|{link}|\n"
            st.markdown(md)

        if submitted:
            on_click_submit_button()

except Exception as e:
    st.exception(e)
