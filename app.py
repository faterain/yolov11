import streamlit as st
from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import warnings
import json
import os
import hashlib
from datetime import datetime
from pathlib import Path

#运行streamlit run app.py --server.headless true
# ===================== 基础配置 =====================
# 关闭所有无关警告（适配新版Streamlit）
warnings.filterwarnings("ignore")

# 页面配置（必须放在最顶部）
st.set_page_config(
    page_title="土豆病害检测系统",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# 数据存储路径（自动创建）
USER_DATA_PATH = "user_data.json"  # 用户信息
HISTORY_PATH = "detection_history.json"  # 检测历史
UPLOAD_DIR = "uploaded_images"  # 上传图片存储目录

# 创建必要目录
for dir_path in [UPLOAD_DIR]:
    Path(dir_path).mkdir(exist_ok=True)

# 初始化数据文件
for file_path in [USER_DATA_PATH, HISTORY_PATH]:
    if not os.path.exists(file_path):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump({}, f, ensure_ascii=False, indent=2)

# ===================== 工具函数 =====================
def hash_password(password):
    """密码加密"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_json(file_path):
    """加载JSON文件"""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(file_path, data):
    """保存JSON文件"""
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def save_uploaded_image(uploaded_file, username):
    """保存上传的图片，返回存储路径"""
    # 生成唯一文件名（时间戳+原文件名）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_ext = uploaded_file.name.split(".")[-1]
    file_name = f"{username}_{timestamp}.{file_ext}"
    file_path = os.path.join(UPLOAD_DIR, file_name)
    
    # 保存图片
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return file_path

# ===================== 模型加载 =====================
@st.cache_resource(show_spinner="正在加载检测模型...")
def load_detection_model():
    """加载YOLO模型（兼容PT/ONNX格式）"""
    # 替换为你的模型路径（支持.pt或.onnx格式）
    model_path = "ultralytics\\ultralytics\\runs\\detect\\train\\yolov11n+hpa+transformer+ATFL\\weights\\best.pt"  # 优先用.pt格式，兼容性更好
    
    # 加载模型
    model = YOLO(model_path)
    return model

# 加载模型（全局唯一）
model = load_detection_model()

# ===================== 登录/注册功能 =====================
def auth_section():
    """用户认证模块"""
    st.sidebar.title("🔐 用户中心")
    
    # 初始化会话状态
    if "logged_in" not in st.session_state:
        st.session_state.logged_in = False
    if "username" not in st.session_state:
        st.session_state.username = ""
    if "show_history" not in st.session_state:
        st.session_state.show_history = False
    
    # 未登录状态
    if not st.session_state.logged_in:
        tab1, tab2 = st.sidebar.tabs(["登录", "注册"])
        
        # 登录标签
        with tab1:
            username = st.text_input("用户名")
            password = st.text_input("密码", type="password")
            
            if st.button("登录"):
                users = load_json(USER_DATA_PATH)
                # 验证用户
                if username in users and users[username] == hash_password(password):
                    st.session_state.logged_in = True
                    st.session_state.username = username
                    st.sidebar.success(f"欢迎回来，{username}！")
                    st.rerun()  # 刷新页面
                else:
                    st.sidebar.error("用户名或密码错误！")
        
        # 注册标签
        with tab2:
            new_username = st.text_input("新用户名")
            new_password = st.text_input("新密码", type="password")
            confirm_password = st.text_input("确认密码", type="password")
            
            if st.button("注册"):
                users = load_json(USER_DATA_PATH)
                # 验证输入
                if new_username in users:
                    st.sidebar.error("用户名已存在！")
                elif new_password != confirm_password:
                    st.sidebar.error("两次密码不一致！")
                elif len(new_password) < 6:
                    st.sidebar.error("密码长度不能少于6位！")
                else:
                    # 保存用户信息
                    users[new_username] = hash_password(new_password)
                    save_json(USER_DATA_PATH, users)
                    st.sidebar.success("注册成功！请登录")
    
    # 已登录状态
    else:
        st.sidebar.write(f"当前用户：{st.session_state.username}")
        
        # 退出登录
        if st.sidebar.button("退出登录"):
            st.session_state.logged_in = False
            st.session_state.username = ""
            st.session_state.show_history = False
            st.rerun()
        
        # 历史记录入口
        if st.sidebar.button("📜 查看检测历史"):
            st.session_state.show_history = True
        else:
            st.session_state.show_history = False

# ===================== 检测功能 =====================
def detection_section():
    """病害检测核心功能"""
    st.title("🍅 改进YOLOv11土豆病害检测系统")
    
    # 未登录提示
    if not st.session_state.logged_in:
        st.warning("请先在左侧边栏登录/注册，才能使用检测功能！")
        return
    
    # 历史记录页面
    if st.session_state.get("show_history", False):
        show_history_section()
        return
    
    # 检测功能页面
    st.subheader("上传土豆叶片图片进行检测")
    
    # 上传图片
    uploaded_file = st.file_uploader(
        "选择图片（支持JPG/PNG格式）",
        type=["jpg", "png", "jpeg"],
        accept_multiple_files=False
    )
    
    if uploaded_file is not None:
        # 显示原始图片
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("原始图片")
            image = Image.open(uploaded_file)
            st.image(image, use_column_width=True)
        
        # 检测按钮
        if st.button("🚀 开始检测", type="primary"):
            with st.spinner("正在检测，请稍候..."):
                # 1. 保存上传的图片
                img_path = save_uploaded_image(uploaded_file, st.session_state.username)
                
                # 2. 模型推理
                img_np = np.array(image)
                results = model(
                    img_np, 
                    imgsz=640,  # 与训练时一致
                    conf=0.5,   # 置信度阈值
                    device="cpu" # 强制用CPU推理（兼容性更好）
                )
                result_img = results[0].plot()  # 绘制检测框
                result_img = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
                
                # 3. 保存检测结果图片
                result_filename = os.path.basename(img_path).replace(".", "_result.")
                result_path = os.path.join(UPLOAD_DIR, result_filename)
                cv2.imwrite(result_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                
                # 4. 解析检测结果
                detections = []
                防治建议 = {
                    "早疫病": "喷施代森锰锌、百菌清等杀菌剂，7天一次，连续2~3次",
                    "晚疫病": "及时清除病株，喷施甲霜灵锰锌、霜脲氰等杀菌剂，加强田间通风",
                    "健康": "无需防治，继续做好田间管理"
                }
                
                for box in results[0].boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    cls_name = model.names[cls]
                    detections.append({
                        "病害类型": cls_name,
                        "置信度": round(conf, 2),
                        "防治建议": 防治建议.get(cls_name, "暂无针对性建议")
                    })
                
                # 5. 显示检测结果
                with col2:
                    st.subheader("检测结果")
                    st.image(result_img, use_column_width=True)
                
                # 6. 显示病害分析
                st.subheader("📊 病害分析结果")
                if len(detections) == 0:
                    st.success("✅ 未检测到病害，叶片健康！")
                    detections.append({"病害类型": "健康", "置信度": 1.0, "防治建议": 防治建议["健康"]})
                else:
                    for det in detections:
                        st.warning(f"🔍 检测到：{det['病害类型']}（置信度：{det['置信度']}）")
                        st.info(f"💡 防治建议：{det['防治建议']}")
                
                # 7. 保存历史记录
                history = load_json(HISTORY_PATH)
                if st.session_state.username not in history:
                    history[st.session_state.username] = []
                
                history[st.session_state.username].append({
                    "检测时间": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "原始图片路径": img_path,
                    "结果图片路径": result_path,
                    "检测结果": detections
                })
                save_json(HISTORY_PATH, history)
                st.success("✅ 检测完成，记录已保存！")

# ===================== 历史记录功能 =====================
def show_history_section():
    """显示检测历史记录"""
    st.title("📜 检测历史记录")
    
    # 加载当前用户的历史记录
    history = load_json(HISTORY_PATH)
    user_history = history.get(st.session_state.username, [])
    
    if not user_history:
        st.info("暂无检测记录，快去上传图片检测吧！")
        return
    
    # 按时间倒序排列
    user_history.reverse()
    
    # 遍历显示每条记录
    for idx, record in enumerate(user_history):
        with st.expander(f"📅 检测记录 {idx+1} | {record['检测时间']}", expanded=False):
            # 显示图片
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("原始图片")
                st.image(record["原始图片路径"], use_column_width=True)
            with col2:
                st.subheader("检测结果")
                st.image(record["结果图片路径"], use_column_width=True)
            
            # 显示检测结果
            st.subheader("检测详情")
            for det in record["检测结果"]:
                st.write(f"• 病害类型：{det['病害类型']}（置信度：{det['置信度']}）")
                st.write(f"• 防治建议：{det['防治建议']}")
            
            # 删除按钮
            if st.button(f"删除这条记录", key=f"delete_{idx}"):
                # 移除记录
                user_history.pop(idx)
                history[st.session_state.username] = user_history
                save_json(HISTORY_PATH, history)
                st.success("记录已删除！")
                st.rerun()
    
    # 清空所有记录
    if st.button("🗑️ 清空所有历史记录", type="secondary"):
        history[st.session_state.username] = []
        save_json(HISTORY_PATH, history)
        st.success("所有记录已清空！")
        st.rerun()

# ===================== 主程序 =====================
if __name__ == "__main__":
    # 显示认证模块
    auth_section()
    # 显示检测模块
    detection_section()