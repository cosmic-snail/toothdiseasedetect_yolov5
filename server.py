from flask import Flask, request, jsonify, Response
import torch
import yolov5
import timedelta
import json, os, time
import tempfile
import requests
import logging
from mobilenetv3 import predict
import lightness_saturation_using as lsu
# from logging.handlers import TimedRotatingFileHandler

# 日志设置(celery异步任务有单独的日志记录， uwsgi也有主程序运行日志， 所以不需要单独记录日志到文件)

# filename = time.strftime("%Y_%m", time.localtime()) + ".log"
# file_log_handler = TimedRotatingFileHandler(os.path.join("static/logs/", filename), when='D', interval=10, backupCount=3, encoding='utf8')
# formatter = logging.Formatter("%(asctime)s %(filename)s [line:%(lineno)d] %(funcName)s [%(levelname)s] %(message)s")
# file_log_handler.setFormatter(formatter)
# logging.getLogger().addHandler(file_log_handler)

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s [line:%(lineno)d] %(funcName)s [%(levelname)s] %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S')





ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
# app.send_file_max_age_default = timedelta(seconds=1)
g_lightness_values = [-40, -20, 0, 20, 40, 60, 80, 100]
g_saturation = 100

# 如果未检测出病种，则调整亮度重试，直到检测出病种或重试结束
def detect_tooth_with_retry(image_dir, target_dir):
    final_res = predict.mobilenetv3(img_path=image_dir)

    try:
        info = yolov5.detect(source=image_dir)
        if isinstance(info, list) and len(info) > 1 and info[0] == 4:
            return info
    except Exception as e:
        logging.warning("YOLO检测失败:%s", e)

    base_name = os.path.splitext(os.path.basename(image_dir))[0]
    for l in g_lightness_values:
        try:
            variant = os.path.join(target_dir, f"{base_name}_L{l}.jpg")
            lsu.update(image_dir, variant, lightness=l, saturation=g_saturation)

            final_res = predict.mobilenetv3(img_path=variant)
            info = yolov5.detect(source=variant)
            if isinstance(info, list) and len(info) > 1 and info[0] == 4:
                return info
        except Exception as e:
            logging.warning("亮度重试失败:%s", e)
            continue

    return final_res

# 用户选择照片的报告接口
@app.route('/api/reports', methods=['POST'])
def reports():
    body = request.json
    if body:
        with tempfile.TemporaryDirectory() as target_dir:
            print(target_dir)

            return_info = []
            for row in body["images"]:
                id = row.get("id", None)
                url = row.get("url", None)
                if id == None or url == None:
                    logging.info("输入参数不正确"+"图片id:" +str(id)+"图片地址:"+str(url))
                    return Response(json.dumps({"code":2, "message":"url和id不能为空!"}), mimetype='application/json')

                image_dir = os.path.join(target_dir,'{}.jpg'.format(id))
                torch.hub.download_url_to_file(url, image_dir)
                with torch.no_grad():
                    try:
                        info = detect_tooth_with_retry(image_dir, target_dir)
                    except Exception as e:
                        image_dict = {"id":id, "results":"3","diseases":[]}
                        logging.error("图片id:"+str(id)+"图片地址:"+str(url)+"图片解析失败:" + str(e))
                    else:
                        image_dict = {"id":id, "results":info[0],"diseases":info[1:]}
                                           
                    return_info.append(image_dict)
            
            report_id = body.get("reportId", None)
            return_dict = {"code":0, "message":"解析完成", "data":{"reportId":report_id, "images":return_info}}
            logging.info("解析成功:"+"report_id:" +str(report_id))
            return Response(json.dumps(return_dict), mimetype='application/json')
    else:
        logging.info("输入参数不正确"+"<输入报文:>" + str(body))
        return Response(json.dumps({"code":1, "message":"请求参数未传!"}), mimetype='application/json')




# 有无牙齿检测接口
# @app.route('/api/check', methods=['POST'])
# def upload_check():
#     url = request.json.get("url", None)
#     id = request.json.get("id", None)
#     if url and id:
#         with tempfile.TemporaryDirectory() as target_check_dir:
#             image_dir = os.path.join(target_check_dir,'{}.jpg'.format(id))
#             torch.hub.download_url_to_file(url, image_dir)
#             with torch.no_grad():
#                 info = yolov5.detect(source=image_dir)
#                 print(info)
#                 for row in info:
#                     if row["detail"]["label"] == "tooth" and row["detail"]["confidences"] > '0.7':
#                         return jsonify({"code":200, "msg":"success"})
#                 return jsonify({"code":400, "msg":"未识别到牙齿，请重新拍摄"})
#     else:
#         return jsonify({"code":400, "msg":"url和id不能为空"})



# 周报告（专业版报告）接口
from celery import Celery
app.config['CELERY_BROKER_URL'] = 'redis://@127.0.0.1:6379/0'
app.config['CELERY_RESULT_BACKEND'] = 'redis://@127.0.0.1:6379/1'
celery = Celery(app.name, broker=app.config['CELERY_BROKER_URL'], include=['server'], backend=app.config["CELERY_RESULT_BACKEND"])
celery.conf.update(app.config)


@app.route('/api/asyncReports', methods=['POST'])
def async_reports():
    body = request.json
    if body:
        for row in body["images"]:
            id = row.get("id", None)
            url = row.get("url", None)
            if id == None or url == None:
                logging.info("输入参数不正确"+"图片id:" +str(id)+"图片地址:"+str(url))
                return Response(json.dumps({"code":2, "message":"url和id不能为空!"}), mimetype='application/json')
        # 调用异步任务
        try:
            task = async_task.apply_async(args=[body])
        except Exception as e:
            logging.error("异步任务调用失败！"+ str(e))
            return Response(json.dumps({"code":3, "message":"接口服务出错，接收失败！"}), mimetype='application/json')
        else:
            return Response(json.dumps({"code":0, "message":"接收成功！"}), mimetype='application/json')

    else:
        logging.info("输入参数不正确"+"<输入报文:>" + str(body))
        return Response(json.dumps({"code":1, "message":"请求参数未传!"}), mimetype='application/json')


@celery.task(name="async_task")
def async_task(request_body):
    # 下载解析图片
    with tempfile.TemporaryDirectory() as target_dir:
        # print(target_dir)
        return_info = []
        for row in request_body["images"]:
            id = row["id"]
            url = row["url"]
            image_dir = os.path.join(target_dir,'{}.jpg'.format(id))
            torch.hub.download_url_to_file(url, image_dir)
            with torch.no_grad():
                try:
                    info = detect_tooth_with_retry(image_dir, target_dir)
                except Exception as e:
                    image_dict = {"id":id, "results":"3","diseases":[]}
                    logging.error("图片id:"+str(id)+"图片地址:"+str(url)+"图片解析失败:" + str(e))
                else:
                    image_dict = {"id":id, "results":info[0],"diseases":info[1:]}
                                        
                return_info.append(image_dict)

        report_id = request_body.get("reportId", None)    
        return_dict = {"reportId":report_id, "images":return_info}

    # 回调接口，上传解析结果
    try:
        res = requests.post(url="https://app.yueyayun.com/apptoothbrush/ai/callback", data=json.dumps(return_dict), headers={"content-type":"application/json"})
    except Exception as e:
        logging.error("<输入报文:>"+ str(return_dict)+"回调接口调取失败:" + str(e))
    else:
        logging.info("<输入报文:>"+ str(return_dict)+"回调接口调取成功:" + str(res.text))


    







if __name__ == '__main__':
    app.debug = False  # 设置调试模式，生产模式的时候要关掉debug
    app.run()
