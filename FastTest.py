import os
import tempfile
import shutil

from mobilenetv3 import predict
import yolov5
import lightness_saturation_using as lsu

src = "/home/rootroot/hystooth_flask/test/test3.jpg"
lightness_values = [-40, -20, 0, 20, 40, 60, 80, 100]
saturation = 0

with tempfile.TemporaryDirectory() as d:
    original = os.path.join(d, "original.jpg")
    shutil.copy(src, original)

    for l in lightness_values:
        img = os.path.join(d, f"L{l}_S{saturation}.jpg")
        lsu.update(original, img, lightness=l, saturation=saturation)

        res = predict.mobilenetv3(img_path=img)
        status = res[0]
        diseases = []
        
        info = yolov5.detect(source=img)
        status = info[0]
        diseases = info[1:]

        print({"lightness": l, "saturation": saturation, "results": status, "diseases": diseases})