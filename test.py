import os
import urllib.request
from diagrams import Diagram, Cluster, Edge
from diagrams.custom import Custom

# --- 1. CẤU HÌNH USER-AGENT (Để tải icon không bị lỗi) ---
opener = urllib.request.build_opener()
opener.addheaders = [('User-Agent', 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36')]
urllib.request.install_opener(opener)

# --- 2. LINK ICON MINH HỌA ---
# Sử dụng các icon hình học đơn giản để đại diện cho các khối lớp
icons_url = {
    "input": "https://cdn-icons-png.flaticon.com/512/3342/3342137.png", # Ảnh đầu vào
    "conv": "https://cdn-icons-png.flaticon.com/512/8637/8637179.png", # Lớp Conv (Hình bánh răng/xử lý)
    "depthwise": "https://cdn-icons-png.flaticon.com/512/17651/17651543.png", # Lớp Depthwise (Xử lý chi tiết)
    "add": "https://cdn-icons-png.flaticon.com/512/3524/3524388.png", # Phép cộng (Residual)
    "pool": "https://cdn-icons-png.flaticon.com/512/2085/2085217.png", # Pooling
    "softmax": "https://cdn-icons-png.flaticon.com/512/4301/4301546.png" # Output Classification
}

# --- 3. HÀM TẢI ICON ---
def get_icon(name, url):
    filename = f"icon_{name}.png"
    # Xóa file lỗi cũ nếu có
    if os.path.exists(filename) and os.path.getsize(filename) < 1000:
        try: os.remove(filename)
        except: pass
        
    if not os.path.exists(filename):
        try:
            print(f"Đang tải icon: {name}...")
            urllib.request.urlretrieve(url, filename)
        except Exception as e:
            print(f"Lỗi tải {name}: {e}")
            return "icon_conv.png" # Fallback
    return filename

# --- 4. VẼ BIỂU ĐỒ ---
graph_attr = {
    "fontsize": "22",
    "bgcolor": "#FFFFFF",
    "splines": "ortho", # Đường nối vuông góc cho giống sơ đồ kỹ thuật
    "pad": "0.5"
}

print("Đang tạo sơ đồ MobileNetV2...")

# Tải tài nguyên
paths = {}
for key, url in icons_url.items():
    paths[key] = get_icon(key, url)

with Diagram("Kiến trúc MobileNetV2: Inverted Residual Block", 
             show=True, 
             filename="mobilenet_structure", 
             outformat="png", 
             graph_attr=graph_attr, 
             direction="LR"): # Vẽ từ Trái sang Phải (Left to Right)

    # 1. Input
    input_img = Custom("Input Image\n(224x224x3)", paths["input"])

    # 2. Stem (Lớp đầu tiên)
    stem = Custom("Conv2D\n(32 filters)", paths["conv"])

    # 3. THE CORE: INVERTED RESIDUAL BLOCK
    # Đây là phần quan trọng nhất của MobileNetV2
    with Cluster("Inverted Residual Block (Linear Bottleneck)", graph_attr={"bgcolor": "#E3F2FD", "style": "rounded"}):
        
        # Bước 1: Expansion (Mở rộng kênh)
        expand = Custom("1x1 Conv\n(Expansion)", paths["conv"])
        
        # Bước 2: Depthwise Conv (Xử lý nhẹ)
        depthwise = Custom("3x3 Depthwise\n(Lightweight)", paths["depthwise"])
        
        # Bước 3: Projection (Nén lại)
        project = Custom("1x1 Conv\n(Projection)", paths["conv"])
        
        # Luồng đi thẳng
        expand >> Edge(label="Expand (x6)") >> depthwise 
        depthwise >> Edge(label="Relu6") >> project

    # 4. Residual Connection (Đường tắt)
    # Kết nối từ Stem vòng qua Block để cộng vào sau Projection
    # Lưu ý: Trong diagram as code, vẽ đường cong ngược hơi khó, 
    # nên ta minh họa logic luồng dữ liệu đi tiếp.
    
    # 5. Classifier Head
    with Cluster("Classifier Head", graph_attr={"bgcolor": "#FFF3E0"}):
        pool = Custom("Global Avg Pooling", paths["pool"])
        dense = Custom("Dense Layer\n(Softmax)", paths["softmax"])

    # KẾT NỐI TỔNG THỂ
    input_img >> Edge(label="Features") >> stem >> expand
    
    # Mũi tên từ Block ra Classifier
    project >> Edge(label="Linear Output") >> pool >> Edge(label="Predictions") >> dense

print("Đã vẽ xong! Kiểm tra file: mobilenet_structure.png")