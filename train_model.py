# train_model.py
import tensorflow as tf #Nhúng toàn bộ thư viện TensorFlow
# pyrefly: ignore[missing-import]
from tensorflow.keras.applications import MobileNetV2 
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model # Dùng để tạo mô hình tùy chỉnh bằng cách kết nối các lớp đầu vào, giữa và đầu ra
# pyrefly: ignore[missing-import]
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
import os #Xử lý đường dẫn, thư mục hình ảnh huấn luyện/test/val, kiểm tra file tồn tại

DATASET_ROOT_DIR = '' 
# Đường dẫn đến các thư mục con train và test trong dataset
# Sử dụng os.path.join để đảm bảo đường dẫn đúng trên mọi hệ điều hành
TRAIN_DIR = os.path.join(DATASET_ROOT_DIR, 'train') 
TEST_DIR = os.path.join(DATASET_ROOT_DIR, 'test')  

IMG_HEIGHT = 224
IMG_WIDTH = 224
BATCH_SIZE = 32

# --- Chuẩn bị Data Augmentation và Data Generators ---
print("Đang chuẩn bị Data Generators...")
train_datagen = ImageDataGenerator(
    rescale=1./255,                 # Chuẩn hóa giá trị pixel về [0, 1]
    rotation_range=20,              # Xoay ảnh ngẫu nhiên tối đa 20 độ
    width_shift_range=0.2,          # Dịch chuyển chiều rộng ngẫu nhiên
    height_shift_range=0.2,         # Dịch chuyển chiều cao ngẫu nhiên
    horizontal_flip=True,           # Lật ngang ảnh ngẫu nhiên
    zoom_range=0.2,                 # Phóng to/thu nhỏ ảnh ngẫu nhiên
    shear_range=0.2,                # Xén ảnh ngẫu nhiên
)

test_datagen = ImageDataGenerator(rescale=1./255) # Chỉ chuẩn hóa cho tập test/validation

# flow_from_directory sẽ tự động tìm các lớp con trong TRAIN_DIR
train_generator = train_datagen.flow_from_directory(
    TRAIN_DIR,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical' # Phân loại đa lớp
)

# flow_from_directory sẽ tự động tìm các lớp con trong TEST_DIR
validation_generator = test_datagen.flow_from_directory(
    TEST_DIR, 
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Lấy số lượng lớp tự động từ generator
NUM_CLASSES = train_generator.num_classes
print(f"Số lượng lớp tìm thấy: {NUM_CLASSES}")

# Lưu ánh xạ nhãn để sử dụng trong Flask
labels = list(train_generator.class_indices.keys())
with open('labels.txt', 'w') as f:
    for label in labels:
        f.write(f"{label}\n")
print("Nhãn đã được lưu thành labels.txt")

# --- Xây dựng Mô hình MobileNetV2 (Transfer Learning) ---
print("Đang xây dựng mô hình MobileNetV2...")
# Tải mô hình MobileNetV2 với trọng số ImageNet, bỏ qua lớp đầu ra cuối cùng
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

# Đóng băng các lớp của base_model để không huấn luyện lại chúng
base_model.trainable = False

# Thêm các lớp tùy chỉnh cho bài toán phân loại rác thải
x = base_model.output
x = GlobalAveragePooling2D()(x) # Giảm chiều dữ liệu từ các feature map
x = Dense(128, activation='relu')(x) # Lớp Dense với 128 units và hàm kích hoạt ReLU
predictions = Dense(NUM_CLASSES, activation='softmax')(x) # Lớp đầu ra với số units bằng số lớp và hàm softmax cho xác suất

model = Model(inputs=base_model.input, outputs=predictions)

# --- Biên dịch và Huấn luyện Mô hình ---
print("Đang biên dịch mô hình...")
model.compile(optimizer='adam', 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# Huấn luyện mô hình
print("Bắt đầu huấn luyện mô hình...")
history = model.fit(
    train_generator,
    epochs=10, #có thể tăng số epoch để đạt độ chính xác cao hơn, hoặc dùng EarlyStopping
    validation_data=validation_generator
)
print("Huấn luyện hoàn tất.")

# --- Lưu Mô hình đã Huấn luyện ---
model_save_path = 'trash_classifier_mobilenetv2.h5'
model.save(model_save_path)
print(f"Mô hình đã được lưu thành {model_save_path}")

# (Tùy chọn) Đánh giá thêm trên tập test cuối cùng nếu cần
# loss, accuracy = model.evaluate(validation_generator)
# print(f"Độ chính xác trên tập kiểm tra: {accuracy*100:.2f}%")
