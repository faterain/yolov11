import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# 解决中文显示问题（可选）
plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 1. 读取CSV文件
csv_dir=r"E:\vscode\data\python\ddl\yolov11\ultralytics\ultralytics\runs\detect\train"
csvs=os.listdir(csv_dir)
plt.figure(figsize=(10, 6))
styles = {
    "yolov11n+hpa": {"color": "#1f77b4", "linestyle": "-", "marker": "o", "markersize": 3},  # 蓝+实线+圆圈
    "yolov11n+hpa+transformer+ATFL": {"color": "#d62728", "linestyle": "--", "marker": "s", "markersize": 3},  # 红+虚线+方块
    "yolov11n+transformer": {"color": "#2ca02c", "linestyle": ":", "marker": "^", "markersize": 3},  # 绿+点线+三角
    "yolov11n": {"color": "#ff7f0e", "linestyle": "-.", "marker": "x", "markersize": 4},  # 橙+点划线+叉号
    "yolov11n+ATFL": {"color": "#9467bd", "linestyle": "-", "marker": "D", "markersize": 3}  # 紫+实线+菱形
}
str="train/box_loss,train/cls_loss,train/dfl_loss,metrics/precision(B),metrics/recall(B),metrics/mAP50(B),metrics/mAP50-95(B),val/box_loss,val/cls_loss,val/dfl_loss,lr/pg0,lr/pg1,lr/pg2"
classes=str.strip().split(",")
for cls in classes:
    for i in range(len(csvs)):
        csv_path=os.path.join(csv_dir,csvs[i],"results.csv")
        df = pd.read_csv(csv_path)  # 替换为你的CSV路径
        style = styles[csvs[i]]
        plt.plot(df["epoch"], df[cls],
                 label=csvs[i], color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=style["markersize"],
                linewidth=1)
    # 2. 设置画布大小


    # 3. 绘制多折线（按模型列逐一绘制）
    # 可自定义颜色、线条样式、标记（和示例图匹配）
    
    # 4. 添加图表元素（标题、坐标轴、图例、网格）
    plt.title(f"不同YOLO模型{cls}曲线", fontsize=14)
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel(cls, fontsize=12)
    plt.legend(loc="upper right", fontsize=10)  # 图例位置
    plt.grid(True, alpha=0.3)  # 浅色网格线

    # 5. 保存高清图片（可选，dpi=300适合论文）
    # cls=cls.replace("/","_")

    # plt.savefig(f"E:\\vscode\\data\\python\\ddl\\yolov11\\ultralytics\\ultralytics\\results_pic\\{cls}.png", dpi=300, bbox_inches="tight")

    # 6. 显示图表
    plt.show()