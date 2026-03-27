import os
import sys
import json
import argparse
import tempfile
import shutil
import logging
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
os.chdir(str(ROOT))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mobilenetv3 import predict
import yolov5

def download_or_copy(src, dst):
    if src.lower().startswith(("http://", "https://")):
        torch.hub.download_url_to_file(src, dst)
    elif os.path.exists(src):
        shutil.copy(src, dst)
    else:
        raise FileNotFoundError(src)

def run_reports(images, report_id="debug-001", use_lsu=False):
    return_info = []
    with tempfile.TemporaryDirectory() as target_dir:
        for row in images:
            _id = row.get("id")
            _url = row.get("url")
            if not _id or not _url:
                return {"code": 2, "message": "url和id不能为空!"}
            image_path = os.path.join(target_dir, f"{_id}.jpg")
            download_or_copy(_url, image_path)
            if use_lsu:
                import lightness_saturation_using as lsu
                lsu.lightness_saturation(target_dir, target_dir)
            with torch.no_grad():
                try:
                    res = predict.mobilenetv3(img_path=image_path)
                    if res[0] == 1:
                        info = yolov5.detect(source=image_path)
                    else:
                        info = res
                except Exception:
                    image_dict = {"id": _id, "results": "3", "diseases": []}
                else:
                    image_dict = {"id": _id, "results": info[0], "diseases": info[1:]}
            return_info.append(image_dict)
    return {"code": 0, "message": "解析完成", "data": {"reportId": report_id, "images": return_info}}

def auto_pick_test_image():
    test_dir = ROOT / "test"
    for pattern in ("*.jpg", "*.jpeg", "*.png", "*.JPG", "*.JPEG", "*.PNG"):
        files = list(test_dir.glob(pattern))
        if files:
            return str(files[0])
    return None

def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", help="本地图片路径")
    ap.add_argument("--url", help="远程图片URL")
    ap.add_argument("--report-id", default="debug-001")
    ap.add_argument("--use-lsu", action="store_true")
    args = ap.parse_args()

    src = args.url or args.path
    if not src:
        pick = auto_pick_test_image()
        if not pick:
            print("未找到测试图片，请通过 --path 或 --url 指定")
            sys.exit(1)
        src = pick

    images = [{"id": "img1", "url": src}]
    result = run_reports(images=images, report_id=args.report_id, use_lsu=args.use_lsu)
    print(json.dumps(result, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()