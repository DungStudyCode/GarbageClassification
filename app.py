import os #Cung cấp các hàm để tương tác với hệ điều hành, ví dụ như tạo/thay đổi thư mục, kiểm tra đường dẫn, hoặc xử lý tệp
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory
#Flask: Tạo ứng dụng web
#render_template: Kết xuất file HTML (template) với dữ liệu
#request: Lấy dữ liệu từ form, file, method GET/POST
#redirect: Chuyển hướng người dùng sang route khác
#url_for: Tạo URL dựa trên tên hàm route
#flash: Hiển thị thông báo (message) tạm thời cho người dùng
#send_from_directory: Gửi file từ thư mục cụ thể (thường dùng để hiển thị ảnh, file tải về, v.v.)

from werkzeug.utils import secure_filename #Làm sạch tên file người dùng upload để tránh lỗi hoặc lỗ hổng bảo mật (loại bỏ ký tự đặc biệt, dấu)
import numpy as np #Thư viện toán học mạnh mẽ cho xử lý ma trận, số học, mảng số hiệu quả
from PIL import Image #Dùng để xử lý hình ảnh (mở, chuyển kích thước, chuyển đổi định dạng)
import tensorflow as tf #hức năng: Dùng để làm việc với các mô hình học sâu (Deep Learning)


# --- Cấu hình Flask App ---
app = Flask(__name__)
# Thư mục để lưu trữ tạm thời ảnh tải lên
app.config['UPLOAD_FOLDER'] = 'uploads/' 
# Giới hạn kích thước tệp tải lên (ví dụ: 16 MB)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024 
# Khóa bí mật cho thông báo flash (cần thiết để hiển thị thông báo)
app.secret_key = 'supersecretkey_for_flash_messages' 

# Các phần mở rộng tệp ảnh được cho phép
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}


# --- Cấu hình mô hình AI ---
MODEL_PATH = 'trash_classifier_mobilenetv2.h5' # Đường dẫn đến tệp mô hình
LABELS_PATH = 'labels.txt'                     # Đường dẫn đến tệp nhãn
IMG_SIZE = (224, 224)                          # Kích thước ảnh đầu vào của mô hình (phải khớp với lúc huấn luyện)

# --- Tải mô hình và nhãn khi ứng dụng bắt đầu ---
# Điều này đảm bảo mô hình chỉ được tải một lần, giúp tối ưu hiệu suất
model = None
class_names = []
try:
    # Kiểm tra xem tf.keras có tồn tại không trước khi tải mô hình
    if not hasattr(tf, 'keras'):
        raise ImportError("Không tìm thấy TensorFlow Keras API (tf.keras). Vui lòng đảm bảo đã cài đặt đúng TensorFlow 2.x.")
        
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r', encoding='utf-8') as f: # Đảm bảo đọc bằng UTF-8 nếu nhãn có ký tự tiếng Việt
        class_names = [line.strip() for line in f]
    print(f"Mô hình '{MODEL_PATH}' và nhãn đã được tải thành công.")
except ImportError as ie:
    print(f"LỖI TẢI THƯ VIỆN: {ie}")
    class_names = ["Lỗi tải nhãn - 0", "Lỗi tải nhãn - 1", "Lỗi tải nhãn - 2", "Lỗi tải nhãn - 3", "Lỗi tải nhãn - 4", "Lỗi tải nhãn - 5"]
except Exception as e:
    print(f"LỖI: Không thể tải mô hình hoặc nhãn. Vui lòng đảm bảo các file '{MODEL_PATH}' và '{LABELS_PATH}' tồn tại và đúng định dạng. Chi tiết lỗi: {e}")
    # Gán nhãn mặc định để tránh lỗi nếu không tải được
    class_names = ["Lỗi tải nhãn - 0", "Lỗi tải nhãn - 1", "Lỗi tải nhãn - 2", "Lỗi tải nhãn - 3", "Lỗi tải nhãn - 4", "Lỗi tải nhãn - 5"]


# --- Hàm kiểm tra định dạng tệp hợp lệ ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# --- Hàm tiền xử lý ảnh cho mô hình ---
def preprocess_image(image_path):
    """Tiền xử lý ảnh để phù hợp với yêu cầu đầu vào của mô hình."""
    img = Image.open(image_path).convert('RGB')   # Mở ảnh và chuyển sang định dạng RGB
    img = img.resize(IMG_SIZE)                    # Đổi kích thước ảnh
    img_array = np.array(img)                     # Chuyển sang mảng NumPy
    img_array = np.expand_dims(img_array, axis=0) # Thêm chiều batch (1, H, W, C)
    img_array = img_array / 255.0                 # Chuẩn hóa về khoảng [0, 1]
    return img_array

# --- Tuyến đường phục vụ ảnh đã tải lên ---
# Cần thiết để trình duyệt hiển thị ảnh từ thư mục 'uploads'
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Tuyến chính của ứng dụng ---
@app.route('/', methods=['GET', 'POST'])
def upload_file():
    uploaded_image_url = None # Khởi tạo biến ảnh tải lên

    if request.method == 'POST':
        # Kiểm tra xem có phần tệp nào trong yêu cầu không
        if 'file' not in request.files:
            flash('Không có phần tệp nào trong yêu cầu.')
            return redirect(request.url)
        
        file = request.files['file']
        
        # Nếu người dùng không chọn tệp và nhấn submit
        if file.filename == '':
            flash('Không có file được chọn. Vui lòng chọn một hình ảnh.')
            return redirect(request.url)
        
        # Nếu tệp tồn tại và định dạng hợp lệ
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename) # Làm sạch tên file để tránh lỗi bảo mật
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath) # Lưu tạm thời vào thư mục uploads/

            # Tạo đường dẫn URL cho ảnh để hiển thị lại trên trang
            uploaded_image_url = url_for('uploaded_file', filename=filename)

            # Kiểm tra xem mô hình đã được tải chưa
            if model is None:
                flash("Lỗi hệ thống: Mô hình AI chưa được tải. Vui lòng thông báo quản trị viên.")
                # os.remove(filepath) # KHÔNG XÓA Ở ĐÂY nếu muốn hiển thị ảnh
                return render_template('index.html', prediction=None, uploaded_image_url=uploaded_image_url)

            try:
                # Tiền xử lý ảnh và thực hiện dự đoán
                processed_image = preprocess_image(filepath)
                predictions = model.predict(processed_image)
                
                # Lấy chỉ số và tên lớp có độ tin cậy cao nhất
                predicted_class_index = np.argmax(predictions[0])
                predicted_class_name = class_names[predicted_class_index]
                confidence = predictions[0][predicted_class_index] * 100

                # Trả về trang HTML kèm kết quả và ảnh
                return render_template(
                    'index.html', 
                    prediction=predicted_class_name, 
                    confidence=f"{confidence:.2f}%",
                    uploaded_image_url=uploaded_image_url # TRUYỀN BIẾN ẢNH VÀO
                )
            except Exception as e:
                flash(f"Lỗi trong quá trình xử lý ảnh hoặc dự đoán: {e}")
                # Cân nhắc việc xóa ảnh khi có lỗi hoặc giữ lại để debug
                # if os.path.exists(filepath):
                #     os.remove(filepath)
                return render_template('index.html', prediction=None, uploaded_image_url=uploaded_image_url)
        else:
            flash('Định dạng tệp không được phép. Vui lòng tải lên ảnh PNG, JPG, JPEG hoặc GIF.')
            return render_template('index.html', prediction=None, uploaded_image_url=uploaded_image_url)
            
    # Hiển thị trang ban đầu khi truy cập qua phương thức GET
    return render_template('index.html', prediction=None, uploaded_image_url=None) # TRUYỀN BIẾN ẢNH CHO YÊU CẦU GET

# --- Khởi tạo thư mục uploads nếu chưa tồn tại ---
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])
    print(f"Đã tạo thư mục '{app.config['UPLOAD_FOLDER']}'.")

# --- Chạy ứng dụng Flask ---
if __name__ == '__main__':
    # debug=True sẽ tự động tải lại server khi có thay đổi và hiển thị lỗi chi tiết
    app.run(debug=True, host='0.0.0.0', port=5000) 
    # host='0.0.0.0' cho phép truy cập từ các thiết bị khác trong cùng mạng
