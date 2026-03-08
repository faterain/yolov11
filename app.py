import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np

# 页面配置
st.set_page_config(page_title="土豆病害检测系统", layout="wide")
st.title("🍅 改进YOLOv11土豆病害检测系统")

# ---------------------- 模型选择配置 ----------------------
# 定义可选模型（键：显示名称，值：模型路径）
AVAILABLE_MODELS = {
    
    "YOLOv11n": "ultralytics\\ultralytics\\runs\\detect\\train\\yolov11n\\weights\\best.pt",
    "YOLOv11n+hpa": "ultralytics\\ultralytics\\runs\\detect\\train\\yolov11n+hpa\\weights\\best.pt",
    "YOLOv11n+ATFL": "ultralytics\\ultralytics\\runs\\detect\\train\\yolov11n+ATFL\\weights\\best.pt",
    "YOLOv11n+transformer": "ultralytics\\ultralytics\\runs\\detect\\train\\yolov11n+transformer\\weights\\best.pt",
    "YOLOv11n+hpa+transformer+ATFL": "ultralytics\\ultralytics\\runs\\detect\\train\\yolov11n+hpa+transformer+ATFL\\weights\\best.pt",
    # 可添加更多模型路径，示例：
    # "YOLOv11s 土豆病害检测": "path/to/yolov11s/best.pt",
    # "YOLOv11m 土豆病害检测": "path/to/yolov11m/best.pt",
    # "ONNX格式模型": "path/to/model.onnx"
}

# 侧边栏模型选择
st.sidebar.header("⚙️ 模型配置")
selected_model_name = st.sidebar.selectbox(
    "选择检测模型",
    options=list(AVAILABLE_MODELS.keys()),
    index=0  # 默认选中第一个模型
)

# 加载选中的模型（带缓存）
@st.cache_resource
def load_selected_model(model_path):
    """加载指定路径的模型，缓存避免重复加载"""
    try:
        model = YOLO(model_path)
        st.sidebar.success(f"✅ 模型加载成功：{selected_model_name}")
        return model
    except Exception as e:
        st.sidebar.error(f"❌ 模型加载失败：{str(e)}")
        st.stop()

# 获取选中模型的路径并加载
selected_model_path = AVAILABLE_MODELS[selected_model_name]
model = load_selected_model(selected_model_path)

# ---------------------- 核心检测功能 ----------------------
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