import os
from flask import Flask, render_template, request, redirect, url_for, flash, send_from_directory, jsonify
from werkzeug.utils import secure_filename
import numpy as np
from PIL import Image
import tensorflow as tf
import base64
import io
import uuid
from gtts import gTTS

# --- Cấu hình Flask App ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads/'
app.config['AUDIO_FOLDER'] = 'static/audio/'  # Thư mục lưu file MP3
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'supersecretkey_for_flash_messages'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- Cấu hình mô hình AI ---
MODEL_PATH = 'trash_classifier_mobilenetv2.h5'
LABELS_PATH = 'labels.txt'
IMG_SIZE = (224, 224)

# --- Tải mô hình và nhãn ---
model = None
class_names = []
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        class_names = [line.strip() for line in f]
    print(f"Mô hình và nhãn đã được tải.")
except Exception as e:
    print(f"LỖI: Không thể tải mô hình hoặc nhãn. Lỗi: {e}")
    class_names = ["Lỗi", "Lỗi", "Lỗi", "Lỗi", "Lỗi", "Lỗi"]

# --- BỘ TỪ ĐIỂN DỊCH (Python) ---
TRANSLATION_MAP = {
    "metal": "kim loại",
    "paper": "giấy",
    "glass": "thủy tinh",
    "plastic": "nhựa",
    "cardboard": "bìa cứng",
    "trash": "rác hữu cơ"
}

# --- Các hàm trợ giúp (không đổi) ---
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0
    return img_array

def create_speech_audio(text_to_speak):
    try:
        tts = gTTS(text=text_to_speak, lang='vi')
        filename = f"speech_{uuid.uuid4()}.mp3"
        save_path = os.path.join(app.config['AUDIO_FOLDER'], filename)
        tts.save(save_path)
        audio_url = url_for('static', filename=f'audio/{filename}')
        return audio_url
    except Exception as e:
        print(f"Lỗi khi tạo file gTTS: {e}")
        return None

# --- Route phục vụ ảnh (không đổi) ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Tuyến chính (UPLOAD) ---
@app.route('/', methods=['GET', 'POST'])
def upload_file_route():
    uploaded_image_url = None
    audio_url = None

    if request.method == 'POST':
        if 'file' not in request.files:
            flash('Không có phần tệp nào trong yêu cầu.')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            flash('Không có file được chọn.')
            return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_image_url = url_for('uploaded_file', filename=filename)

            if model is None:
                flash("Lỗi hệ thống: Mô hình AI chưa được tải.")
                return render_template('index.html', prediction=None, uploaded_image_url=uploaded_image_url, audio_url=None)

            try:
                processed_image = preprocess_image(filepath)
                predictions = model.predict(processed_image)
                
                predicted_class_index = np.argmax(predictions[0])
                predicted_class_name = class_names[predicted_class_index]
                confidence = predictions[0][predicted_class_index] * 100
                confidence_str = f"{confidence:.2f}%"

                # === SỬA ĐỔI (CHO UPLOAD) ===
                # Dịch nhãn
                vietnamese_label = TRANSLATION_MAP.get(predicted_class_name.lower(), predicted_class_name)
                # Tạo chuỗi "99 phẩy 86"
                spoken_confidence = f"{confidence:.2f}".replace('.', ' phẩy ')
                # Tạo văn bản nói chính xác
                text_to_speak = f"Loại rác được nhận diện: {vietnamese_label}. Độ tin cậy: {spoken_confidence} phần trăm."
                audio_url = create_speech_audio(text_to_speak)
                # ============================

                return render_template(
                    'index.html', 
                    prediction=predicted_class_name, 
                    confidence=confidence_str,
                    uploaded_image_url=uploaded_image_url,
                    audio_url=audio_url
                )
            except Exception as e:
                flash(f"Lỗi trong quá trình xử lý ảnh hoặc dự đoán: {e}")
                return render_template('index.html', prediction=None, uploaded_image_url=uploaded_image_url, audio_url=None)
        else:
            flash('Định dạng tệp không được phép.')
            return redirect(request.url)
            
    return render_template('index.html', prediction=None, uploaded_image_url=None, audio_url=None)

# --- Tuyến (CAMERA) ---
@app.route('/predict_cam', methods=['POST'])
def predict_cam_route():
    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'Không tìm thấy dữ liệu ảnh'}), 400

    try:
        image_data_base64 = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        temp_filename = "webcam_capture.jpg"
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        image.save(temp_filepath)

        if model is None:
            return jsonify({'error': 'Mô hình AI chưa được tải'}), 500

        processed_image = preprocess_image(temp_filepath)
        predictions = model.predict(processed_image)
        
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = class_names[predicted_class_index]
        confidence = predictions[0][predicted_class_index] * 100
        confidence_str = f"{confidence:.2f}%"

        # === SỬA ĐỔI (CHO CAMERA) ===
        # Dịch nhãn
        vietnamese_label = TRANSLATION_MAP.get(predicted_class_name.lower(), predicted_class_name)
        # Tạo chuỗi "99 phẩy 86"
        spoken_confidence = f"{confidence:.2f}".replace('.', ' phẩy ')
        # Tạo văn bản nói chính xác
        text_to_speak = f"Loại rác được nhận diện: {vietnamese_label}. Độ tin cậy: {spoken_confidence} phần trăm."
        audio_url = create_speech_audio(text_to_speak)
        # ============================

        return jsonify({
            'prediction': predicted_class_name,
            'confidence': confidence_str,
            'audio_url': audio_url
        })

    except Exception as e:
        print(f"Lỗi khi xử lý ảnh từ camera: {e}")
        return jsonify({'error': str(e)}), 500

# --- Chạy ứng dụng ---
if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['AUDIO_FOLDER']):
        os.makedirs(app.config['AUDIO_FOLDER'])
    
    app.run(debug=True, host='0.0.0.0', port=5000)