import graphviz
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def draw_data_flow_diagram():
    """
    Hàm vẽ Sơ đồ luồng dữ liệu (Data Flow Diagram) sử dụng Graphviz.
    Lưu kết quả thành file ảnh 'data_flow_diagram'.
    """
    print("Đang tạo Sơ đồ luồng dữ liệu (DFD)...")
    
    # Tạo đối tượng Digraph
    dot = graphviz.Digraph(comment='Quy trình Xử lý Phân loại Rác', format='png')
    dot.attr(rankdir='TB')  # Top to Bottom layout
    dot.attr('node', shape='rect', style='rounded', fontname='Arial')
    
    # Định nghĩa các node (theo Mermaid graph bạn cung cấp)
    dot.node('A', 'Người dùng / Webcam', shape='ellipse')
    dot.node('B', 'Server Flask\n(Nhận file/Base64)')
    dot.node('C', 'Tiền xử lý ảnh\n(Resize 224x224, Rescale 0-1)')
    dot.node('D', 'Mô hình AI MobileNetV2')
    dot.node('E', 'Dự đoán xác suất\n(Probability Vector)')
    dot.node('F', 'Lấy nhãn cao nhất\n(Max Confidence Score)')
    
    # Node quyết định (hình thoi)
    dot.node('G', 'Score >= 70% ?', shape='diamond', style='filled', fillcolor='#ffcc00')
    
    # Các nhánh kết quả
    dot.node('H', 'Giữ nguyên nhãn dự đoán', style='filled', fillcolor='#ccffcc') # Xanh nhạt
    dot.node('I', 'Gán nhãn: "TRASH"\n(Rác không xác định)', style='filled', fillcolor='#ff9999') # Đỏ nhạt
    
    dot.node('J', 'Dịch sang Tiếng Việt\n(Mapping Label)')
    dot.node('K', 'Tạo file âm thanh .mp3\n(Google TTS)')
    dot.node('L', 'Trả về Client:\nẢnh + Tên rác + Audio', shape='ellipse', style='filled', fillcolor='#e6e6e6')

    # Định nghĩa các cạnh (Edges) nối các node
    dot.edge('A', 'B', label='Gửi ảnh')
    dot.edge('B', 'C')
    dot.edge('C', 'D')
    dot.edge('D', 'E')
    dot.edge('E', 'F')
    dot.edge('F', 'G')
    
    # Logic rẽ nhánh
    dot.edge('G', 'H', label='Đúng (Yes)')
    dot.edge('G', 'I', label='Sai (No)')
    
    dot.edge('H', 'J')
    dot.edge('I', 'J')
    
    dot.edge('J', 'K')
    dot.edge('K', 'L')

    # Render ảnh
    output_path = 'data_flow_diagram'
    try:
        dot.render(output_path, view=False)
        print(f"-> Đã lưu sơ đồ tại: {output_path}.png")
    except Exception as e:
        print(f"Lỗi khi render Graphviz (Đảm bảo bạn đã cài đặt Graphviz binary): {e}")

def visualize_threshold_logic():
    """
    Hàm vẽ biểu đồ minh họa logic ngưỡng tin cậy 0.7 sử dụng Matplotlib.
    Mô phỏng 3 trường hợp: Rõ ràng, Mờ nhạt (Rác), và Trung bình.
    """
    print("Đang tạo Biểu đồ minh họa Logic Ngưỡng...")
    
    # Dữ liệu mô phỏng
    labels = ['Vỏ chuối\n(Rõ nét)', 'Chai nhựa\n(Hơi mờ)', 'Vật thể lạ\n(Nhiễu/Mờ)']
    scores = [0.95, 0.72, 0.45] # Điểm số tin cậy
    threshold = 0.7
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Màu sắc cột: Xanh nếu đạt, Đỏ nếu trượt
    colors = ['#2ecc71' if s >= threshold else '#e74c3c' for s in scores]
    
    bars = ax.bar(labels, scores, color=colors, width=0.5)
    
    # Vẽ đường ngưỡng (Threshold Line)
    ax.axhline(y=threshold, color='#f1c40f', linestyle='--', linewidth=3, label=f'Ngưỡng tin cậy ({threshold*100}%)')
    
    # Trang trí biểu đồ
    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Độ tin cậy (Confidence Score)', fontsize=12)
    ax.set_title('Minh họa Logic: Kiểm tra Ngưỡng Tin Cậy (Threshold Check)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Thêm text chú thích lên từng cột
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        status = "CHẤP NHẬN" if score >= threshold else "LOẠI -> RÁC (TRASH)"
        
        # Hiển thị điểm số
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{score*100:.0f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Hiển thị quyết định cuối cùng
        ax.text(bar.get_x() + bar.get_width()/2., height/2,
                status,
                ha='center', va='center', color='white', fontweight='bold', rotation=90 if score < 0.8 else 0)

    # Thêm chú thích text bên ngoài cho dễ hiểu
    explanation = (
        "LOGIC XỬ LÝ:\n"
        "• Nếu Score >= 0.7: Giữ nguyên nhãn AI dự đoán.\n"
        "• Nếu Score < 0.7: Hệ thống tự động gán nhãn 'TRASH'\n"
        "  để tránh nhận diện sai các vật thể lạ."
    )
    plt.gcf().text(0.02, 0.02, explanation, fontsize=10, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'))

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    
    output_path = 'logic_threshold_viz.png'
    plt.savefig(output_path, dpi=300)
    print(f"-> Đã lưu biểu đồ logic tại: {output_path}")

if __name__ == "__main__":
    # 1. Vẽ sơ đồ luồng dữ liệu (Yêu cầu cài đặt Graphviz)
    draw_data_flow_diagram()
    
    # 2. Vẽ biểu đồ logic ngưỡng (Chỉ cần Matplotlib)
    visualize_threshold_logic()