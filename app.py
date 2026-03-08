import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import json
import hashlib
import os

# ---------------------- 新增：用户认证相关 ----------------------
# 本地存储用户信息的文件路径
USER_DATA_FILE = "user_data.json"

# 初始化用户数据文件
def init_user_file():
    if not os.path.exists(USER_DATA_FILE):
        with open(USER_DATA_FILE, "w") as f:
            json.dump({}, f)

# 密码加密（简单MD5，仅示例）
def encrypt_pwd(pwd):
    return hashlib.md5(pwd.encode()).hexdigest()

# 注册功能
def register(username, pwd):
    init_user_file()
    with open(USER_DATA_FILE, "r") as f:
        users = json.load(f)
    if username in users:
        return False, "用户名已存在"
    users[username] = encrypt_pwd(pwd)
    with open(USER_DATA_FILE, "w") as f:
        json.dump(users, f)
    return True, "注册成功"

# 登录功能
def login(username, pwd):
    init_user_file()
    with open(USER_DATA_FILE, "r") as f:
        users = json.load(f)
    if username not in users:
        return False, "用户名不存在"
    if users[username] != encrypt_pwd(pwd):
        return False, "密码错误"
    return True, "登录成功"

# ---------------------- 会话状态管理 ----------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False
if "username" not in st.session_state:
    st.session_state.username = ""

# ---------------------- 登录/注册界面 ----------------------
st.set_page_config(page_title="土豆病害检测系统", layout="wide")
if not st.session_state.logged_in:
    st.title("🍅 土豆病害检测系统 - 登录/注册")
    # 切换登录/注册标签
    tab1, tab2 = st.tabs(["登录", "注册"])
    
    with tab1:
        username = st.text_input("用户名")
        pwd = st.text_input("密码", type="password")
        if st.button("登录"):
            success, msg = login(username, pwd)
            if success:
                st.session_state.logged_in = True
                st.session_state.username = username
                st.success(msg)
                # 刷新页面（Streamlit 特性）
                st.rerun()
            else:
                st.error(msg)
    
    with tab2:
        new_username = st.text_input("新用户名")
        new_pwd = st.text_input("新密码", type="password")
        confirm_pwd = st.text_input("确认密码", type="password")
        if st.button("注册"):
            if new_pwd != confirm_pwd:
                st.error("两次密码不一致")
            else:
                success, msg = register(new_username, new_pwd)
                if success:
                    st.success(msg)
                else:
                    st.error(msg)
else:
    # ---------------------- 原有功能（仅登录后可见） ----------------------
    st.title(f"🍅 改进YOLOv11土豆病害检测系统（当前用户：{st.session_state.username}）")
    st.sidebar.button("退出登录", on_click=lambda: st.session_state.update(logged_in=False, username=""))
    
    # 原有模型配置、检测功能（复制原代码）
    AVAILABLE_MODELS = {
        "YOLOv11n": "ultralytics\\ultralytics\\runs\\detect\\train\\yolov11n\\weights\\best.pt",
        "YOLOv11n+hpa": "ultralytics\\ultralytics\\runs\\detect\\train\\yolov11n+hpa\\weights\\best.pt",
        "YOLOv11n+ATFL": "ultralytics\\ultralytics\\runs\\detect\\train\\yolov11n+ATFL\\weights\\best.pt",
        "YOLOv11n+transformer": "ultralytics\\ultralytics\\runs\\detect\\train\\yolov11n+transformer\\weights\\best.pt",
        "YOLOv11n+hpa+transformer+ATFL": "ultralytics\\ultralytics\\runs\\detect\\train\\yolov11n+hpa+transformer+ATFL\\weights\\best.pt",
    }

    st.sidebar.header("⚙️ 模型配置")
    selected_model_name = st.sidebar.selectbox(
        "选择检测模型",
        options=list(AVAILABLE_MODELS.keys()),
        index=0
    )

    @st.cache_resource
    def load_selected_model(model_path):
        try:
            model = YOLO(model_path)
            st.sidebar.success(f"✅ 模型加载成功：{selected_model_name}")
            return model
        except Exception as e:
            st.sidebar.error(f"❌ 模型加载失败：{str(e)}")
            st.stop()

    selected_model_path = AVAILABLE_MODELS[selected_model_name]
    model = load_selected_model(selected_model_path)

    # 上传图片 + 检测逻辑（复制原代码）
    uploaded_file = st.file_uploader("上传土豆叶片图片", type=["jpg", "png", "jpeg"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        img_np = np.array(image)

        if st.button("开始检测"):
            with st.spinner("检测中..."):
                results = model(img_np, imgsz=640, conf=0.5)
                result_img = results[0].plot()
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)

                col1, col2 = st.columns(2)
                with col1:
                    st.subheader("原始图片")
                    st.image(image, use_column_width=True)
                with col2:
                    st.subheader("检测结果")
                    st.image(result_img, use_column_width=True)

                st.subheader("📊 病害分析结果")
                detections = results[0].boxes
                if len(detections) == 0:
                    st.success("未检测到病害，叶片健康！")
                else:
                    for box in detections:
                        cls = int(box.cls[0])
                        conf = float(box.conf[0])
                        cls_name = model.names[cls]
                        st.warning(f"检测到：{cls_name}，置信度：{conf:.2f}")
                        if cls_name == "早疫病":
                            st.info("防治建议：喷施代森锰锌、百菌清等杀菌剂，7天一次，连续2~3次")
                        elif cls_name == "晚疫病":
                            st.info("防治建议：及时清除病株，喷施甲霜灵锰锌、霜脲氰等杀菌剂，加强田间通风")