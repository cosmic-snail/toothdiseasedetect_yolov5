
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mobilenetv3 import predict
import yolov5

img = "test/test1.jpg"

res = predict.mobilenetv3(img_path=img)
print("mobilenet结果:", res)

if res[0] == 1:
    info = yolov5.detect(source=img)
    print("最终检测结果:", info)
else:
    print("最终接口会返回 results=3")