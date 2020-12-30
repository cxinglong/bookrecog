# 文件结构
|-bookrecog_server
                |-run_server.py                         # 主程序，服务启动入口
                |-samples           |- ...              # mobilenet_v3 模型、工具
                                    |-simsun.ttc        # 中文宋体字库
                                    |-tasks.yaml        # 分类的"classes"
                |-models            |- ...              # yolov5 模型
                |-utils             |- ...              # yolov5 工具
                |-scripts           |- ...              # 书籍分类、检测程序及工具
                |-static            |- ...              # Web 静态文件、
                                    |-assets            # 需要上传到web的zip文件存储地址
                |-templates         |- ...              # Web 模板
                |-tmp                                   # 临时（图像）文件存储

                |-weights           |- ...              # 权重文件


# 启动服务
python run_server.py

# 安装依赖
进入虚拟环境后，运行
conda install --yes --file requirements.txt 
