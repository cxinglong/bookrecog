import os
import json
from glob import glob
import cv2
import torch
import numpy as np

from skimage.io import imsave
from deepcv.base import make_class_dic, to_rgb, import_model, th_val, denorm_sample_img
from deepcv.infer import make_parallel_model, parallel_infer, make_sample
from scripts.tools import custom_dict, putText



cls_conf = custom_dict(
    {
    "model_f" : "./samples/model_classifier.py:mobilenet_v3",
    "ckpt_f" : "./weights/mobilenet_v3_epoch_50.pth",
    "ds" : "book",
    "gpu_ids" : [0],
    "score_thr" : 0.6,
    "transform" : True,
    })


def load_parm():
    # load config
    cfg = import_model(cls_conf.model_f, cls_conf.ds)
    model = cfg.make_model(classes=cfg.classes)

    # load weight
    if torch.cuda.is_available():
        weights = torch.load(cls_conf.ckpt_f, map_location=f'cuda:{cls_conf.gpu_ids[0]}')
    else:
        weights = torch.load(cls_conf.ckpt_f, map_location=torch.device('cpu'))

    for key in ['state_dict', 'model', 'net']:
        if key in weights:
            weights = weights[key]
            break
    
    model.load_state_dict(weights)
    print ("loaded weight")

    # load model
    pmodel = make_parallel_model(model, cls_conf.gpu_ids, rescale=None)
    print (f"start infering [{cls_conf.ds} {cls_conf.model_f}, {cls_conf.ckpt_f}]")

    # load classes
    classes = cfg.classes
    cls_id2name = {}
    for tsk_name, v in classes.items():
        cls_id2name[tsk_name] = {
            cls_id: cls_name for cls_id, cls_name in enumerate(v)}
    print(cls_id2name)

    img_trans = cfg.make_img_transform(cfg.img_size)

    return pmodel, cls_id2name, img_trans


def get_predict_one(img_bytes):
    img_string = np.array(img_bytes).tostring()
    img_string = np.asarray(bytearray(img_string), dtype="uint8")
    image = cv2.imdecode(img_string, cv2.IMREAD_COLOR)
    # print(type(image))

    img = to_rgb(image)
    img_meta = {'filename': img}
    results = parallel_infer(pmodel, [img], img_trans, batch_per_gpu=1,img_metas=[img_meta], to_dev_id='np')
    # print(results)
    ret = results[0]
    # print(ret)
    txt_labels = []
    for tsk, pred in ret.items():
        lbl = pred.argmax()
        score = pred.max()
        txt_lbl = cls_id2name[tsk][lbl]
        txt_labels.append({f"{tsk}": f"{txt_lbl}","score":f"{score:.3f}"})

    return json.dumps(str({"code": 0, "result" : txt_labels}))



def get_predict(img_f, out_dir = './files/out', font_path = "./simsun.ttc", batch_size = 1):

    all_img_fs = sorted(glob(img_f+'/*.jpg'))
    
    img_metas = [{'filename': i} for i in all_img_fs]

    batch_per_gpu = max(batch_size // len(cls_conf.gpu_ids), 1)
    batch_nr = len(all_img_fs) // batch_size

    group_img_fs = [all_img_fs[i*batch_size:(i+1)*batch_size] for i in range(batch_nr)]
    print(group_img_fs)
    
    for batch_img_fs in group_img_fs:

        imgs = [to_rgb(cv2.imread(i)) for i in batch_img_fs]
        results = parallel_infer(pmodel, imgs, img_trans, batch_per_gpu=batch_per_gpu, img_metas=img_metas, to_dev_id='np')

        if len(results) == len(batch_img_fs):
           for img_f, img, insts in zip(batch_img_fs, imgs, results):
                canvas = img.copy()
                print("img_f, img, insts: ", img_f, insts)
                put_list = []
                for tsk, pred in insts.items():
                    lbl = pred.argmax()
                    score = pred.max()
                    txt_lbl = cls_id2name[tsk][lbl]
                    print(txt_lbl , score)
                    cls_insts = {"normal":"朝上","inverse":"朝下","front":"封面","back":"封底"}
                    put_str = cls_insts[txt_lbl]
                    put_list.append([put_str, np.around(score, decimals=2)])
                w,h, _ = img.shape
                cv2.rectangle(canvas, (0, 0), (1000, 400), (96, 96, 96), thickness=-1)

                #(255,0,0) 红 (0, 255, 0) 绿 (255, 255, 0) 黄
                if put_list[0][1] >= 0.95:
                    canvas = putText(canvas, str(put_list[0]), (40, 100), font_path, (0, 255, 0), 100)
                elif 0.85 <= put_list[0][1] < 0.95:
                    canvas = putText(canvas, str(put_list[0]), (40, 100), font_path, (255, 255, 0), 100)
                else:
                    canvas = putText(canvas, str(put_list[0]), (40, 100), font_path, (255, 0, 0), 100)

                if put_list[1][1] >= 0.95:
                    canvas = putText(canvas, str(put_list[1]), (40, 200), font_path, (0, 255, 0), 100)
                elif 0.85 <= put_list[1][1] < 0.95:
                    canvas = putText(canvas, str(put_list[1]), (40, 200), font_path, (255, 255, 0), 100)
                else:
                    canvas = putText(canvas, str(put_list[1]), (40, 200), font_path, (255, 0, 0), 100)
                
                os.makedirs(out_dir, exist_ok=True)
                dst_f = os.path.join(out_dir, os.path.basename(img_f))
                imsave(dst_f, canvas)
    return out_dir



# 全局
pmodel, cls_id2name, img_trans = load_parm()
