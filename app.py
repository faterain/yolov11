import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# 页面配置
st.set_page_config(page_title="土豆病害检测系统", layout="wide")
st.title("🍅 改进YOLOv11土豆病害检测系统")

# 加载模型（支持ONNX/PT格式）
@st.cache_resource
def load_model():
    model = YOLO("best.onnx")  # 替换为你的模型路径
    return model

model = load_model()

# 上传图片
uploaded_file = st.file_uploader("上传土豆叶片图片", type=["jpg", "png", "jpeg"])
if uploaded_file is not None:
    # 读取图片
    image = Image.open(uploaded_file)
    img_np = np.array(image)

    # 检测按钮
    if st.button("开始检测"):
        with st.spinner("检测中..."):
            # 模型推理
            results = model(img_np, imgsz=640, conf=0.5)  # conf=0.5过滤低置信度结果
            # 绘制检测框
            result_img = results[0].plot()  # 自动绘制类别、置信度、检测框
            # 转换颜色通道（OpenCV BGR→RGB）
            result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

            # 显示结果
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("原始图片")
                st.image(image, use_column_width=True)
            with col2:
                st.subheader("检测结果")
                st.image(result_img, use_column_width=True)

            # 输出病害分析
            st.subheader("📊 病害分析结果")
            detections = results[0].boxes
            if len(detections) == 0:
                st.success("未检测到病害，叶片健康！")
            else:
                for box in detections:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls]  # 获取类别名称（健康/早疫病/晚疫病）
                    st.warning(f"检测到：{cls_name}，置信度：{conf:.2f}")
                    # 给出防治建议（农业场景适配）
                    if cls_name == "早疫病":
                        st.info("防治建议：喷施代森锰锌、百菌清等杀菌剂，7天一次，连续2~3次")
                    elif cls_name == "晚疫病":
                        st.info("防治建议：及时清除病株，喷施甲霜灵锰锌、霜脲氰等杀菌剂，加强田间通风")