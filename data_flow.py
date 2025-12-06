import os

# Thay đổi đường dẫn dưới đây tới nơi bạn cài Graphviz thực tế
# Lưu ý: Dấu r'' ở trước để tránh lỗi ký tự đặc biệt
os.environ["PATH"] += os.pathsep + r'C:\Program Files\Graphviz\bin'

import graphviz # Import sau khi đã set PATH

def create_data_flow_diagram():
    """
    Hàm tạo Sơ đồ luồng dữ liệu (DFD) tối ưu cho báo cáo Word (Khổ A4).
    Xuất ra file: data_flow_diagram.png
    """
    # Khởi tạo đồ thị
    dot = graphviz.Digraph(comment='Quy trình Xử lý Phân loại Rác', format='png')
    
    # --- CẤU HÌNH GIAO DIỆN CHUẨN WORD ---
    dot.attr(rankdir='TB')     # Top to Bottom
    dot.attr(dpi='300')        # Độ phân giải cao (Print quality)
    dot.attr(size='6.5,10')    # Kích thước tối đa: Rộng 6.5 inch (vừa lề Word), Cao tự do
    dot.attr(ratio='fill')     # Tự động cân chỉnh tỷ lệ lấp đầy
    dot.attr(nodesep='0.8')    # Tăng khoảng cách ngang để 2 nhánh rẽ không dính nhau
    dot.attr(ranksep='0.5')    # Giảm khoảng cách dọc để sơ đồ gọn hơn
    
    # Style node
    dot.attr('node', shape='rect', style='rounded,filled', 
             fontname='Arial', fontsize='11', margin='0.15,0.1')

    # --- ĐỊNH NGHĨA CÁC NODE ---
    
    # 1. Input
    dot.node('A', 'Người dùng / Webcam\n(Input Image)', shape='ellipse', fillcolor='#E3F2FD', color='#1565C0', penwidth='1.5')
    
    # 2. Server & Preprocessing
    dot.node('B', 'Server Flask\n(Nhận Request)', fillcolor='#F5F5F5')
    dot.node('C', 'Tiền xử lý ảnh\n(Resize 224x224, Scale 0-1)', fillcolor='#FFF3E0')
    
    # 3. AI Model
    dot.node('D', 'Mô hình AI\n(MobileNetV2)', shape='component', fillcolor='#E8F5E9', color='#2E7D32')
    dot.node('E', 'Dự đoán (Inference)\nVector xác suất', fillcolor='#F5F5F5')
    
    # 4. Logic Xử lý
    dot.node('F', 'Lấy nhãn có\nxác suất cao nhất', fillcolor='#F5F5F5')
    
    # Node Quyết định
    dot.node('G', 'Score >= 70% ?', shape='diamond', 
             style='filled', fillcolor='#FFF9C4', color='#FBC02D', height='1.0', width='1.5')
    
    # Nhánh Kết quả - Sẽ căn chỉnh hàng ngang sau
    dot.node('H', 'CHẤP NHẬN\n(Giữ nguyên nhãn)', fillcolor='#C8E6C9', color='#2E7D32', width='2.0')
    dot.node('I', 'TỪ CHỐI\n(Gán "TRASH")', fillcolor='#FFCDD2', color='#C62828', width='2.0')
    
    # 5. Hậu xử lý
    dot.node('J', 'Mapping nhãn\nTiếng Việt', fillcolor='#F3E5F5')
    dot.node('K', 'Tạo âm thanh\n(Google TTS)', fillcolor='#E1BEE7')
    
    # 6. Output
    dot.node('L', 'Phản hồi (Response)\n{ Ảnh, Tên, Audio }', shape='ellipse', fillcolor='#E3F2FD', color='#1565C0', penwidth='1.5')

    # --- ĐỊNH NGHĨA LIÊN KẾT ---
    dot.edge('A', 'B')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F')
    dot.edge('F', 'G')
    
    # Nhánh điều kiện
    dot.edge('G', 'H', label='Đúng (Yes)', fontcolor='green', color='green')
    dot.edge('G', 'I', label='Sai (No)', fontcolor='red', color='red')
    
    # Gộp luồng
    dot.edge('H', 'J')
    dot.edge('I', 'J')
    
    dot.edge('J', 'K')
    dot.edge('K', 'L')

    # --- CĂN CHỈNH BỐ CỤC (QUAN TRỌNG) ---
    # Ép buộc 2 nhánh H và I phải nằm ngang hàng nhau
    with dot.subgraph() as s:
        s.attr(rank='same')
        s.node('H')
        s.node('I')

    # --- XUẤT FILE ---
    output_filename = 'data_flow_diagram'
    try:
        output_path = dot.render(output_filename, view=False)
        print(f"✅ Đã tạo sơ đồ chuẩn Word: {output_path}")
    except Exception as e:
        print(f"❌ Lỗi: {e}")

if __name__ == "__main__":
    create_data_flow_diagram()