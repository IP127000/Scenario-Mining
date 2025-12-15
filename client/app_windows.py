import os
import csv
import shutil
import tempfile
import re
import time
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import List, Dict, Tuple, Any, Optional
import cv2
import requests
import gradio as gr
import json
import argparse

try:
    import tkinter as tk
    from tkinter import filedialog
except Exception:
    tk = None
    filedialog = None

UNMATCHED_LABEL = "未匹配"
SUCCESS_COLUMN = "是否解析成功"
DEFAULT_MAX_NEW_TOKENS = 512
DEFAULT_TEMPERATURE = 0.0
DEFAULT_MAX_PIXELS = 1920 * 1080
DEFAULT_FPS = 1
DEFAULT_COMPRESSED_FPS = 10
SERVER_URL = "http://10.0.135.47:31538"
ENDPOINT = "/generate_from_video_upload"
API_URL = SERVER_URL.rstrip("/") + ENDPOINT

DEFAULT_ATTRIBUTE_DATA = [
    ["行驶场景", "城市市区、城区快速路/高速路段、隧道、桥梁、高速匝道、停车场、收费站、山路、内部道路、其他场景"],
    ["天气状况", "晴天、小雨、中雨、大雨、雾天、阴天、下雪"],
    ["路面材质", "水泥、沥青、水泥沥青交替、非铺装路面"],
    ["交通流量", "空旷、稀疏、中等、密集、拥堵"],
    ["是否有道路施工", "是、否"],
    ["是否有交通事故", "是、否"],
    ["是否出现转弯", "是、否"],
    ["是否出现调头", "是、否"],
    ["是否出现红绿灯", "是、否"],
    ["是否出现动物", "是、否"],
    ["是否有特殊车辆", "否、急救车、大型货车、校车、工程车、摩托、电动自行车、三轮车"],
    ["是否出现路口", "是、否"],
]

def load_config(config_path: str) -> dict:
    if not config_path:
        return {}
    if not os.path.isfile(config_path):
        print(f"[WARN] 配置文件未找到: {config_path} 使用默认配置。")
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            if not isinstance(cfg, dict):
                print(f"[WARN] 配置文件格式错误（应为 JSON 对象）: {config_path}  使用默认配置。")
                return {}
            return cfg
    except Exception as e:
        print(f"读取配置文件失败: {e} 使用默认配置。")
        return {}

def merge_attribute_data(config_data: List[List[Any]]) -> List[List[Any]]:
    merged_dict = {}
    if config_data:
        for row in config_data:
            if isinstance(row, (list, tuple)) and len(row) >= 2:
                attr_name = str(row[0]).strip()
                attr_vals = str(row[1]).strip()
                merged_dict[attr_name] = attr_vals
    merged_list: List[List[Any]] = []
    for attr, vals in merged_dict.items():
        merged_list.append([attr, vals])
    return merged_list

def apply_config(cfg: dict):
    global SERVER_URL, ENDPOINT, API_URL, DEFAULT_ATTRIBUTE_DATA
    global DEFAULT_MAX_NEW_TOKENS, DEFAULT_TEMPERATURE, DEFAULT_MAX_PIXELS
    global DEFAULT_FPS, DEFAULT_COMPRESSED_FPS
    if "server_url" in cfg:
        SERVER_URL = cfg["server_url"]
    else:
        address = cfg.get("address", "http://10.0.135.47")
        port = cfg.get("port", 30698)
        if address.startswith("http://") or address.startswith("https://"):
            SERVER_URL = f"{address}:{port}"
        else:
            SERVER_URL = f"http://{address}:{port}"
    ENDPOINT = cfg.get("endpoint", "/generate_from_video_upload")
    API_URL = SERVER_URL.rstrip("/") + ENDPOINT
    config_attr_data = cfg.get("attribute_data")
    if config_attr_data:
        DEFAULT_ATTRIBUTE_DATA = merge_attribute_data(config_attr_data)
    DEFAULT_MAX_NEW_TOKENS = cfg.get("default_max_new_tokens", 512)
    DEFAULT_TEMPERATURE = cfg.get("default_temperature", 0.7)
    DEFAULT_MAX_PIXELS = cfg.get("default_max_pixels", 1920 * 1080)
    DEFAULT_FPS = cfg.get("default_fps", 1)
    DEFAULT_COMPRESSED_FPS = cfg.get("default_compressed_fps", 10)

def open_folder_dialog(initial_dir: str = None) -> str:
    if tk is None or filedialog is None:
        return ""
    root = tk.Tk()
    root.withdraw()
    root.update()
    try:
        root.attributes("-topmost", True)
    except Exception:
        pass
    folder_path = filedialog.askdirectory(
        title="请选择文件夹",
        initialdir=initial_dir or os.getcwd(),
    )
    root.destroy()
    return folder_path

def create_prompt(attribute_dict: Dict[str, str]) -> str:
    prompt = "<video>\n"
    prompt += "根据视频内容，输出以下内容：\n\n"
    prompt += "【标签和可选值列表】（请仅从以下可选值中挑选，可以多选）：\n\n"
    for attr, values in attribute_dict.items():
        prompt += f"{attr}：{values}\n"
    prompt += "\n【注意】输出时不要加入额外的文字说明，只返回上述格式和内容：\n{标签}:{值}\n"
    return prompt

def parse_result(result_text: Any, attributes: List[str]) -> Dict[str, str]:
    if not isinstance(result_text, str):
        result_text = str(result_text)
    attr_dict = {attr: "" for attr in attributes}
    for line in result_text.splitlines():
        line = line.strip()
        if not line:
            continue
        match = re.match(r"([^:：]+)[:：]\s*(.*)", line)
        if match:
            key = match.group(1).strip()
            value = match.group(2).strip()
            if key in attr_dict:
                attr_dict[key] = value
    return attr_dict

def process_single_video(
    video_path: str,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    fps: int = DEFAULT_FPS,
) -> str:
    if not os.path.isfile(video_path):
        return f"视频文件不存在: {video_path}"
    with open(video_path, "rb") as f:
        files = {"video": (os.path.basename(video_path), f, "application/octet-stream")}
        data = {
            "text": prompt,
            "max_new_tokens": str(max_new_tokens),
            "temperature": str(temperature),
            "max_pixels": str(max_pixels),
        }
        if fps is not None:
            data["fps"] = str(fps)
        try:
            resp = requests.post(API_URL, data=data, files=files, timeout=300)
            resp.raise_for_status()
            result = resp.json().get("result", "")
            if isinstance(result, list):
                result = " ".join(map(str, result))
            elif not isinstance(result, str):
                result = str(result)
            return result
        except Exception as e:
            return f"API调用失败 ({video_path}): {str(e)}"

def parse_attribute_df(attribute_df: List[List[Any]]) -> Dict[str, List[str]]:
    attr_possible_vals: Dict[str, List[str]] = {}
    for row in attribute_df:
        if not row or len(row) < 2:
            continue
        attr = str(row[0]).strip()
        values_str = str(row[1]).strip()
        candidates = [
            v.strip()
            for v in re.split(r"[、,，;；|]+", values_str)
            if v.strip()
        ]
        attr_possible_vals[attr] = candidates
    return attr_possible_vals

def compress_video(
    input_path: str,
    output_path: str = None,
    *,
    compress: bool = True,
    target_fps: int = DEFAULT_COMPRESSED_FPS,
    scale_factor: float = 0.5,
    frame_interval: int = 5,
) -> str:
    if not compress:
        return input_path
    if output_path is None:
        tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        output_path = tmp.name
        tmp.close()
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        print(f"[compress_video] 无法打开视频文件: {input_path}")
        return input_path
    fps_original = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    new_width = max(1, int(width * scale_factor))
    new_height = max(1, int(height * scale_factor))
    out_fps = target_fps if target_fps is not None else fps_original
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, out_fps, (new_width, new_height))
    if not out.isOpened():
        print(f"[compress_video] VideoWriter 打开失败: {output_path}")
        cap.release()
        return input_path
    frame_idx = 0
    saved_cnt = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_interval <= 1 or frame_idx % frame_interval == 0:
            if scale_factor != 1.0:
                frame = cv2.resize(frame, (new_width, new_height))
            out.write(frame)
            saved_cnt += 1
        frame_idx += 1
    cap.release()
    out.release()
    if saved_cnt == 0:
        print("[compress_video] 未写入任何帧，返回原文件")
        return input_path
    return output_path

def read_previous_results(csv_path: Optional[str]) -> Dict[str, Dict[str, str]]:
    result: Dict[str, Dict[str, str]] = {}
    if not csv_path:
        return result
    if not os.path.isfile(csv_path):
        return result
    try:
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                video_file = row.get("视频文件") or row.get("视频文件".strip())
                if video_file:
                    result[video_file] = row
    except Exception as e:
        print(f"[read_previous_results] 读取 CSV 失败: {e}")
    return result

def compute_success_flag(attr_map: Dict[str, str]) -> str:
    for v in attr_map.values():
        if v == UNMATCHED_LABEL:
            return "否"
    return "是"


def process_videos(
    video_folder: str,
    output_folder: str,
    attribute_df: List[List[Any]],
    compress_flag: bool = True,
    scale_factor: float = 0.5,
    frame_interval: int = 5,
    compress_fps: int = DEFAULT_COMPRESSED_FPS,
    previous_csv_file: Optional[str] = None,          
    progress=gr.Progress(),
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    max_pixels: int = DEFAULT_MAX_PIXELS,
    fps: int = DEFAULT_FPS,
) -> Tuple[Optional[str], List[List[Any]]]:
    if not video_folder or not Path(video_folder).is_dir():
        return None, []
    if not output_folder:
        return None, []
    Path(output_folder).mkdir(parents=True, exist_ok=True)
    clean_df = [row for row in attribute_df if len(row) >= 2 and str(row[0]).strip()]
    if not clean_df:
        return None, []
    attr_possible_vals = parse_attribute_df(clean_df)
    ATTRIBUTES = list(attr_possible_vals.keys())
    previous_results = read_previous_results(previous_csv_file)
    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg"]
    video_files: List[Path] = []
    for ext in video_exts:
        video_files.extend(Path(video_folder).rglob(f"*{ext}"))
    total_videos = len(video_files)
    if total_videos == 0:
        return None, []
    attr_counters: Dict[str, Counter] = {attr: Counter() for attr in ATTRIBUTES}
    attribute_ranges = {attr: "、".join(vals) for attr, vals in attr_possible_vals.items()}
    prompt = create_prompt(attribute_ranges)
    per_video_results: Dict[str, Dict[str, str]] = {}
    for video_path in progress.tqdm(video_files, desc="处理视频"):
        video_path_str = str(video_path)
        if video_path_str in previous_results and previous_results[video_path_str].get(SUCCESS_COLUMN, "").strip() == "是":
            historic_row = previous_results[video_path_str]
            attr_map: Dict[str, str] = {}
            for attr in ATTRIBUTES:
                attr_map[attr] = historic_row.get(attr, UNMATCHED_LABEL).strip() or UNMATCHED_LABEL
                raw_val = attr_map[attr]
                if raw_val == UNMATCHED_LABEL:
                    attr_counters[attr][UNMATCHED_LABEL] += 1
                else:
                    for mv in raw_val.split("；"):
                        attr_counters[attr][mv] += 1
            attr_map[SUCCESS_COLUMN] = "是"
            per_video_results[video_path_str] = attr_map
            continue
        compressed_path = compress_video(
            video_path_str,
            compress=compress_flag,
            target_fps=compress_fps,
            scale_factor=scale_factor,
            frame_interval=frame_interval,
        )
        result_text = process_single_video(
            compressed_path,
            prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            max_pixels=max_pixels,
            fps=fps,
        )
        attr_dict = parse_result(result_text, ATTRIBUTES)
        per_video_attr_map: Dict[str, str] = {}
        for attr in ATTRIBUTES:
            raw_val = attr_dict.get(attr, "").strip()
            if not raw_val:
                attr_counters[attr][UNMATCHED_LABEL] += 1
                per_video_attr_map[attr] = UNMATCHED_LABEL
                continue
            matched_vals = [
                cand for cand in attr_possible_vals.get(attr, [])
                if cand and cand in raw_val
            ]
            if matched_vals:
                for mv in matched_vals:
                    attr_counters[attr][mv] += 1
                per_video_attr_map[attr] = "；".join(matched_vals)
            else:
                attr_counters[attr][UNMATCHED_LABEL] += 1
                per_video_attr_map[attr] = UNMATCHED_LABEL
        per_video_attr_map[SUCCESS_COLUMN] = compute_success_flag(per_video_attr_map)
        per_video_results[video_path_str] = per_video_attr_map
        if compressed_path != video_path_str:
            try:
                os.remove(compressed_path)
            except OSError as e:
                print(f"删除临时压缩文件失败: {compressed_path}, error: {e}")
        time.sleep(0.1)

    stats_rows: List[List[Any]] = []
    for attr in ATTRIBUTES:
        for val, cnt in sorted(attr_counters[attr].items(), key=lambda x: x[1], reverse=True):
            stats_rows.append([attr, val, cnt])

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    agg_csv_name = f"video_attribute_stats_{timestamp}.csv"
    agg_csv_path = Path(output_folder) / agg_csv_name
    with agg_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["场景", "取值", "计数"])
        for row in stats_rows:
            writer.writerow(row)

    per_video_csv_name = f"video_attribute_details_{timestamp}.csv"
    per_video_csv_path = Path(output_folder) / per_video_csv_name
    with per_video_csv_path.open("w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        header = ["视频文件"] + ATTRIBUTES + [SUCCESS_COLUMN]
        writer.writerow(header)
        for video_path_str, attr_vals in per_video_results.items():
            row = [video_path_str] + [
                attr_vals.get(attr, UNMATCHED_LABEL) for attr in ATTRIBUTES
            ] + [attr_vals.get(SUCCESS_COLUMN, "否")]
            writer.writerow(row)

    temp_per_video_csv = tempfile.NamedTemporaryFile(suffix=".csv", delete=False)
    temp_per_video_csv.close()
    shutil.copyfile(per_video_csv_path, temp_per_video_csv.name)
    return temp_per_video_csv.name, stats_rows

def get_video_count(video_folder: str) -> str:
    if not video_folder or not Path(video_folder).is_dir():
        return "请输入有效的文件夹路径"
    video_exts = [".mp4", ".avi", ".mov", ".mkv", ".webm", ".flv", ".wmv", ".mpeg"]
    count = 0
    for ext in video_exts:
        count += len(list(Path(video_folder).rglob(f"*{ext}")))
    return f"已检测到 {count} 段视频"

CUSTOM_CSS = """
.gradio-container {
    background-color: #f7fafc;
}
.subpanel {
    background-color: #ffffff;
    border-radius: 8px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
    padding: 24px;
    margin-bottom: 20px;
}
.subpanel .gr-component {
    margin-bottom: 12px;
}
.subpanel .gr-textbox,
.subpanel .gr-button,
.subpanel .gr-slider,
.subpanel .gr-checkbox,
.subpanel .gr-dataframe,
.subpanel .gr-file,
.subpanel .gr-progress {
    width: 100% !important;
}
.subpanel .gr-button {
    height: 48px;
}
.gr-button-primary {
    background-color: #4a90e2;
    border-color: #4a90e2;
}
.gr-button-primary:hover {
    background-color: #357ab8;
}
.folder-btn {
    background-color: #e5e7eb;  
    border-color: #c1c4c9;      
    color: #374151;              
}
.folder-btn:hover {
    background-color: #d1d5db;
    border-color: #9ca3af;
}
"""

def _parse_config_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("-c", "--config", type=str, help="configuration file")
    parser.add_argument("--host", default="0.0.0.0", help="服务器监听地址")
    parser.add_argument("--port", type=int, default=80, help="服务器端口")
    parser.add_argument("--share", action="store_true", help="公开分享链接")
    args, _ = parser.parse_known_args()
    return args

args = _parse_config_arg()
_config_path = args.config
_config = load_config(_config_path) if _config_path else {}
apply_config(_config)

with gr.Blocks(
    title="VLM场景挖掘",
    theme=gr.themes.Default(primary_hue="indigo", secondary_hue="gray"),
    css=CUSTOM_CSS,
) as demo:
    gr.Markdown("# VLM 场景挖掘工具")
    gr.Markdown("使用 VLM 对视频场景进行自定义挖掘")

    with gr.Row():
        with gr.Column(scale=1):
            with gr.Column(elem_classes=["subpanel"]):
                gr.Markdown("### 文件夹设置")
                with gr.Row():
                    with gr.Column():
                        video_folder = gr.Textbox(
                            label="视频文件夹路径",
                            placeholder="请选择视频文件夹",
                            interactive=False,
                        )
                        select_video_folder_btn = gr.Button(
                            "选择视频文件夹📂",
                            variant="secondary",
                            size="lg",
                            elem_classes=["folder-btn"],
                        )
                    with gr.Column():
                        output_folder = gr.Textbox(
                            label="结果保存文件夹",
                            placeholder="请选择保存文件夹",
                            interactive=False,
                        )
                        select_output_folder_btn = gr.Button(
                            "选择保存文件夹📂",
                            variant="secondary",
                            size="lg",
                            elem_classes=["folder-btn"],
                        )
                video_count = gr.Textbox(
                    label="视频数量",
                    value="请输入文件夹路径",
                    interactive=False,
                )
            with gr.Column(elem_classes=["subpanel"]):
                gr.Markdown("### 导入已有解析结果")
                previous_csv = gr.File(
                    label="上传历史结果 CSV（可选）",
                    file_types=[".csv"],
                    type="filepath",            
                )
            with gr.Column(elem_classes=["subpanel"]):
                gr.Markdown("### 视频压缩选项")
                with gr.Accordion("压缩设置", open=False):
                    compress_flag = gr.Checkbox(label="是否压缩视频", value=True)
                    scale_factor = gr.Slider(
                        label="缩放因子",
                        minimum=0.1,
                        maximum=1.0,
                        step=0.05,
                        value=0.5,
                    )
                    frame_interval = gr.Slider(
                        label="每隔几帧取一帧",
                        minimum=1,
                        maximum=10,
                        step=1,
                        value=1,
                    )
                    compress_fps = gr.Slider(
                        label="压缩后视频帧率",
                        minimum=1,
                        maximum=30,
                        step=1,
                        value=DEFAULT_COMPRESSED_FPS,
                    )
            with gr.Column(elem_classes=["subpanel"]):
                gr.Markdown("### 待挖掘场景范围")
                with gr.Accordion("场景编辑", open=False):
                    attribute_df = gr.Dataframe(
                        headers=["场景", "取值范围"],
                        datatype=["str", "str"],
                        value=DEFAULT_ATTRIBUTE_DATA,
                        interactive=True,
                        type="array",
                    )
        with gr.Column(scale=1):
            with gr.Column(elem_classes=["subpanel"]):
                gr.Markdown("### 处理进度")
                progress_info = gr.Textbox(
                    label="处理状态",
                    value="等待开始处理...",
                    interactive=False,
                    lines=1,
                )
                progress_bar = gr.Progress()
            with gr.Column(elem_classes=["subpanel"]):
                gr.Markdown("### 挖掘结果统计")
                stats_df = gr.Dataframe(
                    headers=["场景", "取值", "计数"],
                    datatype=["str", "str", "int"],
                    interactive=False,
                )
            with gr.Column(elem_classes=["subpanel"]):
                gr.Markdown("### 下载详细结果")
                per_video_file = gr.File(
                    label="下载详细挖掘结果",
                    interactive=False,
                )
                process_btn = gr.Button("开始挖掘", variant="primary", size="lg")
    select_video_folder_btn.click(
        fn=open_folder_dialog,
        inputs=[],
        outputs=video_folder,
    ).then(
        fn=get_video_count,
        inputs=video_folder,
        outputs=video_count,
    )
    select_output_folder_btn.click(
        fn=open_folder_dialog,
        inputs=[],
        outputs=output_folder,
    )
    process_btn.click(
        fn=lambda: "处理中，请稍候…",
        outputs=progress_info,
    ).then(
        fn=process_videos,
        inputs=[
            video_folder,
            output_folder,
            attribute_df,
            compress_flag,
            scale_factor,
            frame_interval,
            compress_fps,
            previous_csv,              
        ],
        outputs=[per_video_file, stats_df],
    ).then(
        fn=lambda: "处理完成！",
        outputs=progress_info,
    )

def main():
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        debug=False,
    )
    return 0

if __name__ == "__main__":
    sys.exit(main())
