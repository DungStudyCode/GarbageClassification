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
app.config['AUDIO_FOLDER'] = 'static/audio/'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.secret_key = 'supersecretkey_for_flash_messages'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# --- CẤU HÌNH NGƯỠNG TIN CẬY CAO HƠN ---
# Đặt 70% để phân loại rõ ràng. 
# Nếu không chắc chắn chắn 70%, thà nhận là Trash còn hơn nhận nhầm.
CONFIDENCE_THRESHOLD = 70.0 

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
    class_names = ["Lỗi"] * 6

# --- BỘ TỪ ĐIỂN DỊCH (PHÂN RÕ RÀNG TỪNG LOẠI) ---
TRANSLATION_MAP = {
    "metal": "kim loại",
    "paper": "giấy",         # Giấy là Giấy
    "glass": "thủy tinh",
    "plastic": "nhựa",
    "cardboard": "bìa các-tông", # Bìa là Bìa
    "trash": "rác thải khác"
}

# --- Các hàm trợ giúp ---
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
        return url_for('static', filename=f'audio/{filename}')
    except Exception as e:
        print(f"Lỗi gTTS: {e}")
        return None

# --- XỬ LÝ LOGIC NGHIÊM NGẶT (STRICT LOGIC) ---
def apply_strict_logic(predictions, class_names):
    """
    Hàm này phân loại rạch ròi dựa trên ngưỡng tin cậy cao.
    """
    predicted_index = np.argmax(predictions[0])
    raw_label = class_names[predicted_index]
    confidence_score = predictions[0][predicted_index] * 100
    
    # Nếu độ tin cậy < 70%, coi như không nhận diện được (Trash)
    # Điều này giúp tránh việc Giấy (40-50%) bị nhận nhầm thành Bìa
    if confidence_score < CONFIDENCE_THRESHOLD:
        final_label = "trash"
    else:
        final_label = raw_label # Giữ nguyên nhãn gốc (Giấy là Giấy, Bìa là Bìa)
        
    return final_label, confidence_score

# --- Route phục vụ ảnh ---
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

# --- Tuyến chính (UPLOAD) ---
@app.route('/', methods=['GET', 'POST'])
def upload_file_route():
    uploaded_image_url = None
    audio_url = None

    if request.method == 'POST':
        if 'file' not in request.files: return redirect(request.url)
        file = request.files['file']
        if file.filename == '': return redirect(request.url)
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            uploaded_image_url = url_for('uploaded_file', filename=filename)

            if model is None:
                return render_template('index.html', prediction=None, uploaded_image_url=uploaded_image_url)

            try:
                processed_image = preprocess_image(filepath)
                predictions = model.predict(processed_image)
                
                # --- ÁP DỤNG LOGIC NGHIÊM NGẶT ---
                predicted_class_name, confidence = apply_strict_logic(predictions, class_names)
                confidence_str = f"{confidence:.2f}%"

                # Tạo âm thanh
                vietnamese_label = TRANSLATION_MAP.get(predicted_class_name.lower(), predicted_class_name)
                spoken_confidence = f"{confidence:.2f}".replace('.', ' phẩy ')
                
                if predicted_class_name == "trash" and confidence < CONFIDENCE_THRESHOLD:
                    text_to_speak = f"Hình ảnh chưa đủ rõ ràng để phân biệt. Tôi đoán đây là {vietnamese_label}."
                else:
                    text_to_speak = f"Loại rác được nhận diện: {vietnamese_label}. Độ tin cậy: {spoken_confidence} phần trăm."
                
                audio_url = create_speech_audio(text_to_speak)

                return render_template('index.html', prediction=predicted_class_name, confidence=confidence_str, uploaded_image_url=uploaded_image_url, audio_url=audio_url)
            except Exception as e:
                flash(f"Lỗi: {e}")
                return render_template('index.html', prediction=None, uploaded_image_url=uploaded_image_url)
            
    return render_template('index.html', prediction=None, uploaded_image_url=None)

# --- Tuyến (CAMERA) ---
@app.route('/predict_cam', methods=['POST'])
def predict_cam_route():
    data = request.get_json()
    try:
        image_data = base64.b64decode(data['image'].split(',')[1])
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        
        temp_filepath = os.path.join(app.config['UPLOAD_FOLDER'], "webcam_capture.jpg")
        image.save(temp_filepath)

        if model is None: return jsonify({'error': 'Model error'}), 500

        processed_image = preprocess_image(temp_filepath)
        predictions = model.predict(processed_image)
        
        # --- ÁP DỤNG LOGIC NGHIÊM NGẶT ---
        predicted_class_name, confidence = apply_strict_logic(predictions, class_names)
        confidence_str = f"{confidence:.2f}%"

        # Tạo âm thanh
        vietnamese_label = TRANSLATION_MAP.get(predicted_class_name.lower(), predicted_class_name)
        spoken_confidence = f"{confidence:.2f}".replace('.', ' phẩy ')

        if predicted_class_name == "trash" and confidence < CONFIDENCE_THRESHOLD:
             text_to_speak = f"Hình ảnh chưa đủ rõ ràng để phân biệt. Tôi đoán đây là {vietnamese_label}."
        else:
             text_to_speak = f"Loại rác được nhận diện: {vietnamese_label}. Độ tin cậy: {spoken_confidence} phần trăm."

        audio_url = create_speech_audio(text_to_speak)

        return jsonify({'prediction': predicted_class_name, 'confidence': confidence_str, 'audio_url': audio_url})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']): os.makedirs(app.config['UPLOAD_FOLDER'])
    if not os.path.exists(app.config['AUDIO_FOLDER']): os.makedirs(app.config['AUDIO_FOLDER'])
    app.run(debug=True, host='0.0.0.0', port=5000)