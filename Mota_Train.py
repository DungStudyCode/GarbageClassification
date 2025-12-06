import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch # Thêm thư viện này để vẽ hộp bo tròn
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import get_file
import cv2
import urllib.request # Thêm thư viện để tải ảnh với Header tùy chỉnh
import os # Thêm thư viện os để kiểm tra file nội bộ

# Cấu hình font chữ cho biểu đồ đẹp hơn trong báo cáo
plt.rcParams.update({'font.size': 12, 'font.family': 'sans-serif'})

def visualize_augmentation():
    """
    1. MINH HỌA DATA AUGMENTATION
    Ưu tiên lấy ảnh từ thư mục dự án (Local). Nếu không có sẽ tải từ Internet.
    """
    print("--- Đang tạo hình ảnh minh họa Data Augmentation ---")
    
    # 1. Cấu hình nguồn ảnh
    # Ưu tiên 1: Lấy ảnh từ máy (Local)
    # Hãy đảm bảo bạn có file ảnh này trong thư mục dự án, hoặc đổi tên file bên dưới
    local_image_path = r'D:\Python\GarbageClassification\test\glass\glass37.jpg'  # Đường dẫn ảnh cốc thủy tinh trong máy bạn

    if os.path.exists(local_image_path):
        print(f"-> Sử dụng ảnh có sẵn trong project: {local_image_path}")
        path = local_image_path
    else:
        print(f"-> Không tìm thấy file '{local_image_path}' trong thư mục dự án.")
        print("-> Đang chuyển sang tải ảnh mẫu từ Internet...")
        
        # Logic tải ảnh online (Fallback)
        # Sử dụng kỹ thuật giả lập User-Agent để tránh lỗi 403 Forbidden từ Wikimedia
        url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/1/15/Glass_of_water.jpg/320px-Glass_of_water.jpg'
        local_filename = 'glass_sample.jpg'
        
        try:
            # Tạo request với User-Agent giả lập trình duyệt
            req = urllib.request.Request(
                url, 
                data=None, 
                headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
            )
            
            print(f"Đang tải ảnh từ: {url}")
            with urllib.request.urlopen(req) as response, open(local_filename, 'wb') as out_file:
                out_file.write(response.read())
            path = local_filename
            print("-> Tải ảnh thành công.")
            
        except Exception as e:
            print(f"⚠️ Không thể tải ảnh cái cốc (Lỗi: {e}).")
            print("-> Đang chuyển sang ảnh mẫu dự phòng (Cat sample) của Google.")
            # Fallback về ảnh mẫu của Google nếu ảnh trên bị lỗi
            backup_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/320px-Felis_catus-cat_on_snow.jpg'
            path = get_file('cat_sample.jpg', origin=backup_url)
    
    # Load ảnh từ đường dẫn đã xác định
    try:
        img = tf.keras.preprocessing.image.load_img(path, target_size=(224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array.reshape((1,) + img_array.shape) # Reshape thành (1, 224, 224, 3)

        # 2. Cấu hình Data Augmentation (Y hệt code của bạn)
        datagen = ImageDataGenerator(
            rescale=1./255,             # Lưu ý: Khi hiển thị cần convert ngược lại
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            shear_range=0.2,
            fill_mode='nearest'
        )

        # 3. Sinh ảnh và vẽ
        fig, axes = plt.subplots(1, 5, figsize=(20, 5))
        
        # Hiển thị ảnh gốc
        axes[0].imshow(img_array[0].astype('uint8'))
        axes[0].set_title("Ảnh gốc (Original)")
        axes[0].axis('off')

        # Hiển thị 4 biến thể
        iterator = datagen.flow(img_array, batch_size=1)
        
        for i in range(1, 5):
            batch = next(iterator)
            aug_img = batch[0] # Ảnh này đã rescale 1./255 nên giá trị từ 0-1
            axes[i].imshow(aug_img) 
            axes[i].set_title(f"Biến thể {i}\n(Augmented)")
            axes[i].axis('off')

        plt.suptitle("Minh họa kỹ thuật Tăng cường dữ liệu (Data Augmentation)", fontsize=16, y=1.05)
        plt.tight_layout()
        plt.savefig('augmentation_demo.png', bbox_inches='tight', dpi=300)
        print("-> Đã lưu: augmentation_demo.png")
        plt.show()
    except Exception as e:
        print(f"❌ Lỗi khi xử lý ảnh: {e}")

def visualize_model_architecture():
    """
    2. VẼ SƠ ĐỒ KHỐI KIẾN TRÚC MÔ HÌNH (BLOCK DIAGRAM)
    Vẽ sơ đồ kết nối giữa Base Model và Custom Head bằng Matplotlib
    (Thay thế cho plot_model vì plot_model cài đặt graphviz phức tạp)
    """
    print("\n--- Đang vẽ sơ đồ kiến trúc mô hình ---")
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Định nghĩa các hàm vẽ hộp
    def draw_box(x, y, text, color='#E3F2FD', width=3, height=1.5):
        # SỬA LỖI: Dùng FancyBboxPatch thay cho plt.Rectangle để hỗ trợ 'boxstyle'
        # Rectangle không hỗ trợ boxstyle, gây ra lỗi AttributeError
        rect = FancyBboxPatch((x, y), width, height, 
                              boxstyle="round,pad=0.1", 
                              facecolor=color, 
                              edgecolor='#1565C0', 
                              linewidth=2, 
                              alpha=0.8,
                              mutation_scale=1.0) # mutation_scale giúp scale boxstyle chính xác hơn
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2, text, ha='center', va='center', fontsize=11, fontweight='bold', color='#0D47A1')
        return x + width/2, y, y + height

    # Vẽ các khối
    # Input
    cx_in, y_in_bottom, y_in_top = draw_box(3.5, 8.0, "INPUT IMAGE\n(224 x 224 x 3)", color='#FFF3E0')
    
    # Base Model
    cx_base, y_base_bottom, y_base_top = draw_box(2.5, 5.5, "BASE MODEL\nMobileNetV2\n(Frozen Weights)", color='#E8F5E9', width=5)
    
    # GAP
    cx_gap, y_gap_bottom, y_gap_top = draw_box(3.5, 3.5, "Global Average Pooling\n(Flatten 3D -> 1D)", color='#F3E5F5')
    
    # Dense
    cx_dense, y_dense_bottom, y_dense_top = draw_box(3.5, 1.8, "DENSE LAYER\n(128 Units, ReLU)", color='#F3E5F5')
    
    # Output
    cx_out, y_out_bottom, y_out_top = draw_box(3.5, 0.0, "OUTPUT LAYER\n(Softmax, 6 Classes)", color='#FFEBEE')

    # Vẽ mũi tên kết nối
    def draw_arrow(x, y_start, y_end):
        ax.arrow(x, y_start, 0, y_end - y_start, head_width=0.15, head_length=0.15, fc='black', ec='black', length_includes_head=True)

    draw_arrow(cx_in, y_in_bottom, y_base_top)
    draw_arrow(cx_base, y_base_bottom, y_gap_top)
    draw_arrow(cx_gap, y_gap_bottom, y_dense_top)
    draw_arrow(cx_dense, y_dense_bottom, y_out_top)

    # Chú thích bên cạnh
    ax.text(8.0, 6.25, "Transfer Learning\n(ImageNet Weights)", ha='left', va='center', fontsize=10, style='italic', color='green')
    ax.text(8.0, 2.65, "Fine-tuning / Classifier\n(Trainable Params)", ha='left', va='center', fontsize=10, style='italic', color='purple')

    # Vẽ ngoặc hoặc đường chia
    ax.plot([7.8, 7.8], [5.5, 7.0], color='green', linestyle='--')
    ax.plot([7.8, 7.8], [0.0, 5.0], color='purple', linestyle='--')

    plt.title("Kiến trúc Mạng Nơ-ron (MobileNetV2 + Custom Head)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('model_architecture_diagram.png', dpi=300)
    print("-> Đã lưu: model_architecture_diagram.png")
    plt.show()

def visualize_training_history():
    """
    3. BIỂU ĐỒ HUẤN LUYỆN GIẢ LẬP
    Giả lập dữ liệu history để vẽ biểu đồ Loss và Accuracy đẹp chuẩn báo cáo
    """
    print("\n--- Đang vẽ biểu đồ huấn luyện ---")
    
    epochs = np.arange(1, 11) # 10 epochs
    
    # Giả lập dữ liệu (Mô phỏng quá trình hội tụ thực tế)
    # Accuracy tăng dần
    acc = [0.55, 0.68, 0.75, 0.81, 0.84, 0.87, 0.89, 0.90, 0.91, 0.92]
    val_acc = [0.52, 0.65, 0.73, 0.78, 0.81, 0.83, 0.85, 0.86, 0.86, 0.87]
    
    # Loss giảm dần
    loss = [1.5, 1.1, 0.9, 0.7, 0.55, 0.45, 0.38, 0.32, 0.28, 0.25]
    val_loss = [1.6, 1.2, 1.0, 0.85, 0.7, 0.65, 0.60, 0.58, 0.59, 0.57] # Validation loss thường chững lại

    plt.figure(figsize=(14, 6))

    # Biểu đồ Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label='Training Accuracy', marker='o', linewidth=2)
    plt.plot(epochs, val_acc, label='Validation Accuracy', marker='o', linewidth=2, linestyle='--')
    plt.title('Độ chính xác (Accuracy) qua các Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--', alpha=0.7)

    # Biểu đồ Loss
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label='Training Loss', marker='o', color='red', linewidth=2)
    plt.plot(epochs, val_loss, label='Validation Loss', marker='o', color='orange', linewidth=2, linestyle='--')
    plt.title('Hàm mất mát (Loss) qua các Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.7)

    plt.suptitle("Kết quả Huấn luyện Mô hình (Mô phỏng 10 Epochs)", fontsize=16)
    plt.tight_layout()
    plt.savefig('training_history_chart.png', dpi=300)
    print("-> Đã lưu: training_history_chart.png")
    plt.show()

if __name__ == "__main__":
    visualize_augmentation()
    visualize_model_architecture()
    visualize_training_history()