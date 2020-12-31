import os
import shutil
import torch
import json
import cv2
import base64
from flask import Flask, jsonify, request, render_template
from flask_cors import CORS 
from datetime import timedelta

from scripts.tools import zip_file, unzip_file
from scripts.classify import get_predict_one, get_predict
from scripts.detect import get_detect_one, get_detect

app = Flask(__name__)
app.send_file_max_age_default = timedelta(seconds=1)
CORS(app)

'''
# -------------------------------------------------------- #
#                    书籍的封面/封底分类                    #
# -------------------------------------------------------- #
'''
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

# 处理单张图像
@app.route("/predict_1", methods=["POST"])
@torch.no_grad()
def predict_1():
    f = request.files["file"]
    if not allowed_file(f.filename):
        return jsonify({"code": 1, "result": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
    img_bytes = f.read()
    print(type(img_bytes))
    txt_labels = get_predict_one(img_bytes)

    return jsonify(txt_labels)

# 处理单张图像
@app.route("/predict_2", methods=["POST"])
@torch.no_grad()
def predict_2():
    f = request.files["file"]
    if not allowed_file(f.filename):
        return jsonify({"code": 1, "result": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
    img_bytes = f.read()
    txt_labels = get_predict_one(img_bytes)

    return jsonify(txt_labels)

# 处理批量图像
@app.route("/predict_3", methods=["POST"])
def predict_3():
    '''
    处理批量上传的文件
    file_path:  # 从Web获得的文件保存到的路径
    target_path: # 解压后的文件保存到的路径
    out_dir: # 处理后需传到Web的文件保存的路径
    '''
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    os.makedirs(BASE_DIR, exist_ok=True)

    obj = request.files["file"]
    file_path = os.path.join(BASE_DIR,obj.filename) 
    obj.save(file_path)

    target_path = os.path.join(BASE_DIR, os.path.splitext(obj.filename)[0])  
    os.makedirs(target_path, exist_ok=True)
    ret = unzip_file(file_path, target_path)
    

    out_dir = get_predict(img_f = target_path, out_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)),"static/assets",f"save_{os.path.splitext(obj.filename)[0]}"))
    zip_f = zip_file(out_dir)

    os.remove(file_path)  # 删除文件
    shutil.rmtree(target_path)
    shutil.rmtree(out_dir) 
    print(zip_f)
    return zip_f


'''
# -------------------------------------------------------- #
#             书籍的书页侧/书脊侧、书本数量检测              #
# -------------------------------------------------------- #
'''
# 处理单张图像
@app.route("/predict_4", methods=["POST"])
def predict_4():
    f = request.files["file"]
    if not allowed_file(f.filename):
        return jsonify({"code": 1, "result": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})
    img_bytes = f.read()
    print('img_bytes:', type(img_bytes) )
    img = get_detect_one(img_bytes)
    
    array_bytes = img.tobytes()
    
    success,encoded_image = cv2.imencode(".jpg",img)

    dst_bytes = encoded_image.tostring()
    img_dst = base64.b64encode(dst_bytes)
    return img_dst


# 处理批量图像
@app.route("/predict_5", methods=["POST"])
def predict_5():
    BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tmp")
    os.makedirs(BASE_DIR, exist_ok=True)

    obj = request.files["file"]
    file_path = os.path.join(BASE_DIR,obj.filename) 
    obj.save(file_path)

    target_path = os.path.join(BASE_DIR, os.path.splitext(obj.filename)[0])  
    os.makedirs(target_path, exist_ok=True)
    ret = unzip_file(file_path, target_path)

    save_dir = get_detect(imgf = target_path, save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),"static/assets",f"save_{os.path.splitext(obj.filename)[0]}"))

    zip_f = zip_file(save_dir)

    os.remove(file_path)  # 删除文件
    shutil.rmtree(target_path)
    shutil.rmtree(save_dir) 
    print(zip_f)
    return zip_f



@app.route("/", methods=["GET", "POST"])
def root():
    return render_template("up.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=5000)
