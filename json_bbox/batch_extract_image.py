import fitz  # PyMuPDF
import json
import os
import re
import cv2
import numpy as np
import math

# ==========================================
# 核心功能函数
# ==========================================

def get_precise_skew_angle(page):
    """
    高精度倾斜角度检测
    渲染较高分辨率(200dpi)的页面图像，利用霍夫变换检测文本基线。
    """
    try:
        # 提高 DPI 到 72 或更高以获得线条检测
        pix = page.get_pixmap(dpi=72)
        img_arr = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
        
        if pix.n >= 3:
            img = cv2.cvtColor(img_arr, cv2.COLOR_RGB2BGR)
        else:
            img = img_arr

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 使用 Canny 边缘检测
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # 霍夫变换检测直线
        min_line_len = pix.w // 5
        lines = cv2.HoughLinesP(edges, 1, np.pi / 1800, threshold=50, minLineLength=min_line_len, maxLineGap=20)
        
        if lines is None:
            return 0.0

        angles = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue # 忽略垂直线
            
            # 计算角度 (弧度转角度)
            angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
            
            # 只关注微小倾斜 (-3度 到 3度)
            if -3 < angle < 3:
                angles.append(angle)
        
        if not angles:
            return 0.0
        
        # 使用中位数，排除极值干扰
        median_angle = np.median(angles)
        return median_angle

    except Exception as e:
        # print(f"    [警告] 角度检测出错: {e}")
        return 0.0

def rotate_image(cv_image, angle):
    """
    旋转图像，背景填充白色，保持原图尺寸
    """
    if abs(angle) < 0.02:
        return cv_image
    
    (h, w) = cv_image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(cv_image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return rotated

def imwrite_safe(filename, img):
    """支持中文路径保存，并设置 JPG 质量"""
    try:
        ext = os.path.splitext(filename)[1].lower()
        if not ext: ext = ".jpg"
        
        encode_params = []
        if ext in ['.jpg', '.jpeg']:
            encode_params = [int(cv2.IMWRITE_JPEG_QUALITY), 85] # 质量调整为85
            
        is_success, im_buf = cv2.imencode(ext, img, encode_params)
        
        if is_success:
            im_buf.tofile(filename)
            return True
    except Exception as e:
        print(f"保存失败: {e}")
    return False

def extract_images_from_pdf(pdf_path, image_data, output_folder):
    """
    处理单个 PDF 文件的核心逻辑
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    doc = fitz.open(pdf_path)
    page_skew_cache = {} 
    
    pdf_name = os.path.basename(pdf_path)
    print(f"正在处理: {pdf_name} -> 共 {len(image_data)} 张图")

    success_count = 0

    for i, item in enumerate(image_data):
        try:
            img_id = item.get('id', f'img_{i}')
            page_num = item['page'] - 1 
            bbox_raw = item['data-bbox']

            # 解析坐标
            if isinstance(bbox_raw, str):
                bbox_coords = json.loads(bbox_raw)
            else:
                bbox_coords = bbox_raw
                
            top, left, bottom, right = bbox_coords

            # 简单的页码检查
            if page_num < 0 or page_num >= len(doc):
                print(f"  [跳过] 页码越界: P{item['page']}")
                continue

            page = doc.load_page(page_num)
            
            # === 1. 计算/获取页面倾斜角 ===
            if page_num not in page_skew_cache:
                skew = get_precise_skew_angle(page)
                page_skew_cache[page_num] = skew
            
            skew_angle = page_skew_cache[page_num]

            # === 2. 坐标转换 ===
            page_rect = page.rect
            page_w, page_h = page_rect.width, page_rect.height

            x0 = (left / 1000) * page_w
            y0 = (top / 1000) * page_h
            x1 = (right / 1000) * page_w
            y1 = (bottom / 1000) * page_h

            # === 3. 严格裁剪 ===
            clip_rect = fitz.Rect(max(0, x0), max(0, y0), min(page_w, x1), min(page_h, y1))

            # === 4. 高清渲染 (2倍图) ===
            matrix = fitz.Matrix(2, 2) 
            pix = page.get_pixmap(matrix=matrix, clip=clip_rect)

            # === 5. 转 OpenCV 并旋转 ===
            img_array = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, pix.n)
            
            if pix.n >= 3:
                cv_img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            else:
                cv_img = img_array
            
            final_img = rotate_image(cv_img, skew_angle)

            # === 6. 保存 ===
            safe_id = re.sub(r'[\\/:*?"<>|]', '_', img_id).strip()
            output_path = os.path.join(output_folder, f"{safe_id}.jpg")

            if imwrite_safe(output_path, final_img):
                success_count += 1
                # print(f"  已保存: {safe_id}.jpg") # 减少刷屏，需要可以打开

        except Exception as e:
            print(f"  ❌ 提取错误 [{img_id}]: {e}")
            continue

    doc.close()
    print(f"完成: {pdf_name}, 成功提取 {success_count}/{len(image_data)}\n")

# ==========================================
# 批量处理逻辑
# ==========================================

def batch_process_folder(work_dir):
    """
    遍历文件夹，寻找成对的 pdf 和 json 文件进行处理
    """
    if not os.path.exists(work_dir):
        print(f"错误: 文件夹不存在 -> {work_dir}")
        return

    # 获取所有 pdf 文件
    files = os.listdir(work_dir)
    pdf_files = [f for f in files if f.lower().endswith('.pdf')]
    
    if not pdf_files:
        print("未找到 PDF 文件。")
        return

    print(f"=== 开始批量处理，工作目录: {work_dir} ===\n")

    for pdf_file in pdf_files:
        # 获取文件名（不含扩展名），例如 "01_第一章"
        base_name = os.path.splitext(pdf_file)[0]
        
        pdf_path = os.path.join(work_dir, pdf_file)
        json_path = os.path.join(work_dir, base_name + ".json")
        
        # 检查是否存在对应的 JSON 文件
        if not os.path.exists(json_path):
            print(f"⚠️  跳过: {pdf_file} (未找到对应的 {base_name}.json)")
            continue
        
        # 定义输出目录：在当前文件夹下创建一个同名的文件夹
        output_dir = os.path.join(work_dir, base_name)
        
        try:
            # 读取 JSON 数据
            with open(json_path, 'r', encoding='utf-8') as jf:
                data = json.load(jf)
            
            # 执行提取
            extract_images_from_pdf(pdf_path, data, output_dir)
            
        except json.JSONDecodeError:
            print(f"❌ 错误: JSON 文件格式有误 -> {base_name}.json")
        except Exception as e:
            print(f"❌ 未知错误处理 {pdf_file}: {e}")

    print("=== 所有任务处理完毕 ===")

# ==========================================
# 运行配置
# ==========================================

if __name__ == "__main__":
    # -------------------------------------------------
    # 配置：请在这里填写包含 PDF 和 JSON 的文件夹路径
    # 注意：Windows 路径如果包含反斜杠 \ 请使用双反斜杠 \\ 或在字符串前加 r
    # -------------------------------------------------
    
    WORK_DIR = r"C:\Users\xxx\Desktop\01" 
    
    # 也可以直接写当前目录
    # WORK_DIR = "." 

    batch_process_folder(WORK_DIR)