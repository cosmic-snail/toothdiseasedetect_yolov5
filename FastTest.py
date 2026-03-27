import os, tempfile, shutil
from mobilenetv3 import predict
import yolov5
import lightness_saturation_using as lsu

src = "/home/rootroot/hystooth_flask/test/test.jpg"

with tempfile.TemporaryDirectory() as d:
    img = os.path.join(d, "test.jpg")
    shutil.copy(src, img)

    # lsu.lightness_saturation(d, d)

    res = predict.mobilenetv3(img_path=img)
    print("接口同款 mobilenet结果:", res)

    if res[0] == 1:
        info = yolov5.detect(source=img)
        print("接口同款 最终检测结果:", info)
    else:
        print("接口同款 最终会返回 results=3")