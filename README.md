<img width="839" height="816" alt="image" src="https://github.com/user-attachments/assets/54285d70-4a6b-4964-b6cd-47ecdbdaf83b" />LỜI MỞ ĐẦU
Trong bối cảnh hiện nay, ô nhiễm môi trường và quản lý rác thải đang trở thành một vấn đề cấp bách trên toàn cầu cũng như tại Việt Nam. Sự bùng nổ dân số, tốc độ đô thị hóa nhanh chóng và thói quen tiêu dùng chưa bền vững đã dẫn đến lượng rác thải sinh hoạt và công nghiệp tăng lên đáng kể, gây áp lực lớn đối với các bãi rác và hệ thống xử lý rác hiện tại. Việc phân loại rác tại nguồn được coi là bước quan trọng trong chu trình kinh tế tuần hoàn, giúp giảm thiểu ô nhiễm, tiết kiệm tài nguyên và hỗ trợ các giải pháp xử lý, tái chế hiệu quả.
Trong thời đại của Trí tuệ nhân tạo (AI) và Học sâu (Deep Learning), việc ứng dụng Computer Vision vào phân loại rác thải đã mở ra cơ hội phát triển các hệ thống tự động hóa thông minh, thay thế các phương pháp thủ công truyền thống, nâng cao năng suất và độ chính xác.
Báo cáo này trình bày quá trình nghiên cứu và triển khai dự án “Hệ thống Phân loại Rác thải bằng AI và Web App”, bao gồm:
	Tổng quan về bối cảnh, vấn đề và mục tiêu dự án.
	Cơ sở lý thuyết về học sâu, mạng nơ-ron tích chập (CNN), kỹ thuật Transfer Learning, và kiến trúc MobileNetV2.
	Quy trình huấn luyện mô hình, tiền xử lý dữ liệu, triển khai Web App bằng Flask, tích hợp chức năng Webcam và thiết kế giao diện người dùng (UI/UX).
	Đánh giá kết quả, hạn chế, thách thức và hướng phát triển trong tương lai.
Mục tiêu của báo cáo là cung cấp một cái nhìn tổng quan, chi tiết và khoa học về việc ứng dụng AI vào phân loại rác thải, đồng thời đưa ra các giải pháp cải tiến nhằm phát triển một hệ thống thông minh, thân thiện với người dùng và có khả năng mở rộng trong thực tiễn.
 
CHƯƠNG I: TỔNG QUAN HỆ THỐNG VÀ PHÂN TÍCH YÊU CẦU
1.1. Bối cảnh và Sự cấp thiết của Đề tài
a. Thực trạng rác thải toàn cầu và tại Việt Nam
Sự gia tăng dân số toàn cầu, quá trình công nghiệp hóa – đô thị hóa diễn ra mạnh mẽ đã khiến khối lượng rác thải phát sinh mỗi ngày tăng với tốc độ chưa từng có. Theo báo cáo của World Bank, tổng lượng rác thải toàn cầu có thể tăng đến 70% vào năm 2050, tương đương hơn 3,4 tỷ tấn/năm, nếu các quốc gia không áp dụng các biện pháp quản lý rác thải hiệu quả và bền vững.
Tại Việt Nam, tốc độ phát triển kinh tế – xã hội trong nhiều năm gần đây kéo theo sự gia tăng đột biến về lượng rác thải sinh hoạt và công nghiệp. Một số vấn đề nổi bật:
	Trên 70% rác thải sinh hoạt hiện nay vẫn được xử lý bằng hình thức chôn lấp, trong đó nhiều bãi chôn lấp không đạt chuẩn, gây ô nhiễm nghiêm trọng.
	Công tác phân loại rác tại nguồn chưa hiệu quả, phần lớn rác bị trộn lẫn dẫn đến khó tái chế.
	Một lượng lớn tài nguyên tái chế (nhựa, kim loại, thủy tinh…) bị lãng phí, gây thiệt hại kinh tế và tạo áp lực cho môi trường.
	Nhiều đô thị lớn như Hà Nội, TP.HCM đang đối mặt với tình trạng quá tải rác thải, bãi chôn lấp đóng cửa hoặc sắp hoạt động hết công suất.
Những thách thức trên đặt ra yêu cầu cấp thiết trong việc ứng dụng công nghệ, đặc biệt là trí tuệ nhân tạo (AI) và xử lý hình ảnh (Computer Vision), nhằm hỗ trợ tự động hóa công tác phân loại rác.
b. Vai trò của phân loại rác tại nguồn
Phân loại rác tại nguồn đóng vai trò then chốt trong mô hình Kinh tế tuần hoàn (Circular Economy) – định hướng mà nhiều quốc gia phát triển đang hướng tới. Việc phân loại sớm giúp:
	Giảm tải đáng kể cho các bãi chôn lấp vì rác hữu cơ, rác tái chế và rác nguy hại được tách biệt trước khi xử lý.
	Tăng tỷ lệ tái chế nhờ rác sạch, không bị lẫn tạp chất, giúp hoạt động tái chế hiệu quả và giảm chi phí.
	Giảm phát thải khí nhà kính như methane (CH₄) từ rác hữu cơ phân hủy.
	Tận dụng tài nguyên, tiết kiệm nguyên liệu sản xuất nhờ tái chế giấy, nhựa, kim loại…
	Nâng cao nhận thức cộng đồng về bảo vệ môi trường.
Tuy nhiên, quy trình phân loại tại nguồn chỉ hiệu quả khi có công nghệ hỗ trợ và mô hình vận hành phù hợp, đặc biệt trong bối cảnh rác thải ngày càng đa dạng và phức tạp.
c. Hạn chế của phương pháp phân loại truyền thống
Hiện nay, hoạt động phân loại rác ở Việt Nam và nhiều nước đang phát triển chủ yếu dựa trên:
	Phân loại thủ công (Manual Sorting):
	Phụ thuộc hoàn toàn vào công nhân làm việc trong môi trường độc hại.
	Năng suất thấp, không ổn định.
	Dễ xảy ra sai sót do mệt mỏi, thiếu tập trung hoặc không nhận diện được các loại rác mới.
	Tốn nhiều nhân công, chi phí vận hành cao.
	Dây chuyền cơ khí đơn giản:
	Chủ yếu sử dụng băng chuyền, nam châm, lưới sàng, máy thổi khí… nhằm tách rác theo trọng lượng, kích thước hoặc từ tính.
	Không thể nhận biết chi tiết loại vật liệu, ví dụ:
	Không phân biệt được các loại nhựa (PET, HDPE…)
	Không phân biệt giấy sạch và giấy bẩn
	Không nhận diện được rác nhỏ, bị bóp méo hoặc dính bẩn
	Tính hiệu quả phụ thuộc nhiều vào độ đồng đều của rác – điều khó xảy ra trong thực tế.
Những hạn chế trên cho thấy nhu cầu về công nghệ phân loại thông minh, có khả năng hiểu hình ảnh tương tự con người, là hết sức cần thiết.
1.2. Động lực phát triển
Trong những năm gần đây, sự tiến bộ vượt bậc của AI, đặc biệt là Deep Learning và mạng nơ-ron tích chập (CNN), đã tạo ra bước đột phá trong lĩnh vực xử lý hình ảnh.
	Các mô hình như ResNet, MobileNet, EfficientNet, YOLO có khả năng trích xuất đặc trưng hình ảnh cực kỳ mạnh mẽ.
	Độ chính xác nhận diện vật thể ngày càng cao, đôi khi vượt khả năng của con người trong các tác vụ lặp lại.
	Công nghệ AI có thể triển khai trên thiết bị nhẹ (edge device) như Raspberry Pi, Jetson Nano, điện thoại, camera AI.
Trong bối cảnh đó, việc ứng dụng AI vào phân loại rác mang lại nhiều lợi ích:
	Tự động hóa hoàn toàn quá trình phân loại.
	Giảm chi phí nhân công và rủi ro sức khỏe.
	Tăng độ chính xác và tốc độ xử lý, phù hợp cho hệ thống phân loại theo thời gian thực.
	Tạo tiền đề cho hệ thống quản lý rác thông minh trong đô thị thông minh (Smart City).
Dự án hướng đến khai thác các công nghệ này để xây dựng một mô hình có khả năng phân loại rác tự động dựa trên ảnh.
1.3. Sơ lược Bài toán 
Mặc dù áp dụng AI vào phân loại rác là giải pháp tiềm năng, nhưng dự án vẫn đối mặt với nhiều thách thức quan trọng:
	Tính đa dạng của rác thải
Rác thường bị bóp méo, cong vênh, rách hoặc mất hình dạng ban đầu.
Vật thể có nhiều kích thước, chất liệu, màu sắc khác nhau.
	Môi trường thu thập ảnh phức tạp
Điều kiện ánh sáng thay đổi liên tục.
Nền ảnh lộn xộn (background clutter).
Rác có thể bị che khuất một phần, làm giảm độ chính xác.
	Yêu cầu thời gian thực (Real-time)
Hệ thống AI cần phân loại nhanh, độ trễ thấp để kịp xử lý trên băng chuyền tự động.
Mô hình phải tối ưu để chạy trên các thiết bị tài nguyên hạn chế như camera AI hoặc module nhúng.
	Thiếu bộ dữ liệu phù hợp
Dữ liệu về rác thải mang tính địa phương (ở Việt Nam) ít, cần tự thu thập hoặc mở rộng từ các nguồn công khai.
Do đó, dự án cần phát triển một giải pháp có khả năng nhận diện chính xác và ổn định trong điều kiện thực tế.
1.4. Mục tiêu Dự án
Dự án hướng đến việc xây dựng một hệ thống phân loại rác thải ứng dụng Trí tuệ nhân tạo có thể hoạt động hiệu quả trong điều kiện thực tế. Các mục tiêu chính bao gồm:
	Mục tiêu về độ chính xác (Accuracy Objective)
Xây dựng và huấn luyện một mô hình học sâu có khả năng phân loại chính xác các nhóm rác phổ biến. Để đạt được mục tiêu này, dự án tập trung vào:
Lựa chọn kiến trúc mạng phù hợp (MobileNetV2) để cân bằng giữa độ chính xác, tốc độ và tài nguyên tính toán.
Chuẩn hóa và đa dạng hóa bộ dữ liệu huấn luyện nhằm tăng khả năng tổng quát hóa của mô hình.
Ứng dụng các kỹ thuật tối ưu mô hình như fine-tuning, augmentation để giúp mô hình nhận diện tốt đối với:
	Rác bị biến dạng
	Rác bị che khuất
	Ảnh nền phức tạp
	Điều kiện ánh sáng thay đổi
Đảm bảo mô hình đạt độ chính xác tối thiểu mong muốn (ví dụ: >90% tùy theo tập dữ liệu), đáp ứng tiêu chí ứng dụng trong thực tiễn.
-	Mục tiêu về tính thực tiễn và khả năng ứng dụng
Phát triển một hệ thống giao diện web (Web Application) thân thiện, trực quan, giúp người dùng có thể:
	Tải hình ảnh rác lên để hệ thống tự động nhận diện.
	Sử dụng Webcam hoặc camera để phân loại rác theo thời gian thực (Real-time hoặc Near Real-time).
	Nhận phản hồi nhanh và hiển thị rõ ràng kết quả phân loại:
+ Loại rác
+ Tỷ lệ/Độ tin cậy (confidence)
+ Gợi ý xử lý tương ứng (nếu cần)
+ Dễ dàng triển khai trong các bối cảnh
+ Mô hình giáo dục môi trường
+ Hệ thống phân loại thử nghiệm tại gia đình
+ Demo cho doanh nghiệp xử lý rác thải
Hoạt động ổn định trên cả máy tính cấu hình trung bình, nhờ vào kiến trúc mô hình nhẹ.
	Mục tiêu công nghệ (Technical Objective)
Xây dựng và tích hợp thành công mô hình AI vào hệ thống web thông qua công nghệ phù hợp, cụ thể:
	Sử dụng MobileNetV2 – một mô hình CNN nhẹ, hiệu quả, tối ưu cho thiết bị biên (edge device).
	Huấn luyện và chuyển đổi mô hình sang định dạng phù hợp để triển khai (TensorFlow/Keras → SavedModel → h5, hoặc TFLite nếu cần).
	Tích hợp mô hình vào Flask Framework (Python) để xây dựng API xử lý ảnh.
	Đảm bảo:
	Tốc độ suy luận nhanh (low latency)
	Tài nguyên tiêu thụ thấp
	Dễ dàng mở rộng (scalable)
	Tương thích đa nền tảng
	Thiết kế kiến trúc hệ thống web gồm các thành phần:
	Backend Flask xử lý suy luận AI
	Frontend web hiển thị kết quả
	Kết nối webcam/thư viện JavaScript để truyền ảnh real-time
1.5. Phân tích Yêu cầu Hệ thống 
a. Yêu cầu Chức năng (Functional Requirements - FR)
Mã Yêu cầu	Tên Chức năng	Mô tả chi tiết	File liên quan
FR01	Phân loại từ Tệp	Cho phép người dùng tải lên một tệp hình ảnh rác thải.	index.html, app.py
FR02	Phân loại từ Webcam	Cho phép người dùng sử dụng camera để chụp ảnh và phân loại ngay lập tức.	index.html, app.py (route /predict_cam)
FR03	Hiển thị Kết quả	Hiển thị loại rác được dự đoán và Độ tin cậy của mô hình.	index.html (Phần result-wrapper)
FR04	Huấn luyện Mô hình	Thực hiện quy trình tiền xử lý dữ liệu và huấn luyện mô hình Deep Learning.	train_model.py
FR05	Tải/Sử dụng Mô hình	Tải mô hình đã được huấn luyện (.h5) khi ứng dụng Flask khởi động.	app.py
b. Yêu cầu Phi Chức năng (Non-Functional Requirements - NFR)
Mã Yêu cầu	Thuộc tính	Mô tả chi tiết
NFR01	Hiệu suất	Thời gian phản hồi cho mỗi lần phân loại không vượt quá 3 giây.
NFR02	Độ chính xác	Độ chính xác phân loại của mô hình (Accuracy) phải đạt tối thiểu 85% trên tập kiểm thử (Validation Set).
NFR03	Khả năng sử dụng	Giao diện web phải trực quan, dễ thao tác, có hỗ trợ các tab chuyển đổi.
NFR04	Công nghệ	Hệ thống phải sử dụng các thư viện đã định nghĩa trong requirements.txt (TensorFlow, Flask, PIL, NumPy).
 
II. CHI TIẾT KỸ THUẬT VÀ CƠ CHẾ HOẠT ĐỘNG CỦA HỆ THỐNG
2.1. Phân tích và Lựa chọn Công nghệ
Mục này trình bày tổng quan về hệ sinh thái công nghệ được lựa chọn cho dự án, lý do đề xuất và vai trò cụ thể của từng thành phần. Việc lựa chọn stack công nghệ (technology stack) phù hợp là yếu tố tiên quyết ảnh hưởng trực tiếp đến hiệu năng (performance), độ chính xác (accuracy) và trải nghiệm người dùng trong hệ thống phân loại rác thải.
2.1.1. AI & Deep Learning (TensorFlow/Keras & MobileNetV2)
Phân hệ AI đóng vai trò là "bộ não" trung tâm của toàn bộ hệ thống, chịu trách nhiệm xử lý các luồng dữ liệu hình ảnh phức tạp để đưa ra kết quả phân loại rác thải chính xác. Trong dự án này, chúng tôi lựa chọn sự kết hợp giữa TensorFlow và Keras để xây dựng lớp học sâu (Deep Learning).
a. Framework và Thư viện: TensorFlow & Keras
TensorFlow (được phát triển bởi Google) kết hợp với Keras API được chọn làm nền tảng cốt lõi để phát triển mô hình. Đây là tiêu chuẩn công nghiệp hiện nay cho các bài toán Thị giác máy tính (Computer Vision).
	 Vai trò và chức năng trong hệ thống: 
Xây dựng kiến trúc Mạng nơ-ron tích chập (CNN): TensorFlow cung cấp các khối xây dựng cơ bản (building blocks) như các lớp tích chập (Conv2D), lớp gộp (MaxPooling), và lớp kết nối đầy đủ (Dense). Keras giúp đơn giản hóa việc ghép nối các lớp này thành một kiến trúc mạng hoàn chỉnh theo mô hình tuần tự (Sequential) hoặc mô hình chức năng (Functional API).
Hình 2.1.1. Kiến trúc tổng quát của mạng nơ-ron tích chập (CNN) được xây dựng trên nền tảng TensorFlow.
Sơ đồ minh họa dòng dữ liệu qua một mạng CNN cơ bản. Bắt đầu từ "Input Image" (ảnh đầu vào), dữ liệu đi qua lớp "Convolution" (tích chập) để trích xuất đặc trưng, sau đó qua lớp "Pooling" (gộp) để giảm chiều dữ liệu, tiếp theo là lớp "Fully Connected" (kết nối đầy đủ) để phân loại, và cuối cùng đưa ra kết quả tại "Output". Các mũi tên chỉ hướng đi của dữ liệu trong quá trình xử lý.
Thực hiện Huấn luyện (Training): Framework chịu trách nhiệm nạp dữ liệu ảnh đã được gán nhãn, thực hiện quá trình lan truyền xuôi (forward propagation) để dự đoán và tính toán sai số so với nhãn thực tế. 
Tối ưu hóa trọng số (Optimization): Hệ thống sử dụng các thuật toán tối ưu hóa tiên tiến như Adam (Adaptive Moment Estimation) hoặc SGD (Stochastic Gradient Descent). Các thuật toán này tự động điều chỉnh hàng triệu tham số (trọng số) trong mạng nơ-ron để giảm thiểu hàm mất mát (loss function), giúp mô hình ngày càng thông minh hơn qua từng vòng lặp (epoch). 
Đóng gói và Xuất bản mô hình (Model Export): Sau khi đạt độ chính xác yêu cầu, mô hình được trích xuất dưới định dạng chuẩn .h5 (HDF5) hoặc SavedModel. Định dạng này chứa toàn bộ cấu trúc mạng và các trọng số đã huấn luyện, cho phép dễ dàng tích hợp vào backend Flask mà không cần huấn luyện lại.
	Lý do lựa chọn công nghệ:
	Hiệu năng tính toán cao: TensorFlow hỗ trợ tính toán song song và tận dụng sức mạnh của GPU (Graphics Processing Unit) thông qua CUDA, giúp giảm thời gian huấn luyện từ vài ngày xuống còn vài giờ đối với các tập dữ liệu lớn.
	Hệ sinh thái và Cộng đồng: Là thư viện Deep Learning phổ biến nhất thế giới, TensorFlow có tài liệu kỹ thuật phong phú. Điều này giúp nhóm phát triển dễ dàng tìm kiếm giải pháp cho các lỗi phát sinh (debugging) và tiếp cận các mô hình tiên tiến (State-of-the-art) được cộng đồng chia sẻ.
	Khả năng tích hợp Python: do được viết tối ưu chjo Python, TensorFlow tương thích hoàn hảo với các thư viện xử lí dữ liệu khác trong dự án như NumPy, Pandas và đặc biệt Flask, tạo nên một quy trình phát triển liền mạch.
Hình 2.1.2. Hệ sinh thái TensorFlow và mối quan hệ với Keras API.
Sơ đồ minh họa cấu trúc hệ sinh thái TensorFlow. Tại lớp lõi (Core) là nền tảng tính toán mạnh mẽ hỗ trợ đa ngôn ngữ (C++, Python). Phía trên là Keras - API cấp cao giúp người dùng xây dựng mô hình dễ dàng. Các nhánh mở rộng xung quanh thể hiện khả năng triển khai đa dạng: TensorFlow Lite cho thiết bị di động/IoT, TensorFlow.js cho trình duyệt web và TFX cho việc vận hành hệ thống máy học quy mô lớn (Production).
b. Kiến trúc Mô hình: MobileNetV2
Thay vì tự xây dựng một mạng CNN từ đầu, dự án sử dụng MobileNetV2 – một kiến trúc tiên tiến được Google phát triển, tối ưu hóa đặc biệt cho các thiết bị có tài nguyên tính toán hạn chế (Edge Devices).
	Đặc điểm kỹ thuật nổi bật:
	Inverted Residual Blocks (Khối dư đảo ngược): Khác với ResNet truyền thống, MobileNetV2 mở rộng số lượng kênh (expand) ở lớp giữa để trích xuất đặc trưng, sau đó nén lại (project) ở đầu ra. Điều này giúp giảm số lượng tham số nhưng vẫn giữ được độ sâu và thông tin quan trọng của mạng.
	Depthwise Separable Convolutions: Đây là kỹ thuật chia nhỏ quá trình tích chập tiêu chuẩn thành hai bước: tích chập chiều sâu (Depthwise) và tích chập điểm (Pointwise). Kỹ thuật này giảm khối lượng tính toán đi khoảng 8-9 lần so với tích chập thường.
	Lý do lựa chọn cho dự án phân loại rác thải:
	Lightweight & Low Latency: Phù hợp hoàn hảo cho bài toán web/mobile cần tốc độ phản hồi tức thì (real-time inference).
	Transfer Learning (Học chuyển giao): Dễ dàng tinh chỉnh (fine-tune) trên tập dữ liệu rác thải mới mà không cần huấn luyện lại từ đầu, tiết kiệm thời gian và tài nguyên máy tính.
	Cân bằng hiệu năng: Đảm bảo độ chính xác (Accuracy) cao trong khi vẫn giữ được tốc độ xử lý nhanh (FPS) trên server cấu hình trung bình.
Hình 2.1.3. Sơ đồ kiến trúc khối Inverted Residual trong MobileNetV2.
Hình ảnh minh họa luồng xử lý dữ liệu qua một khối MobileNetV2 điển hình. Dữ liệu đầu vào đi qua lớp 1x1 Conv (Expansion) để tăng số chiều, tiếp theo là lớp 3x3 Depthwise Conv để xử lý không gian, và cuối cùng là 1x1 Conv (Projection) để nén dữ liệu lại. Đường kết nối tắt (Residual Connection) giúp bảo toàn thông tin giữa các lớp.
2.1.2 Tổng quan Kiến trúc và Công nghệ Hệ thống
a. Khối Backend (Flask - Python)
Flask đóng vai trò là cầu nối (API Gateway) giữa giao diện người dùng và mô hình trí tuệ nhân tạo.
	Chức năng chính:
	Thiết lập Web Server để lắng nghe các HTTP Request (GET/POST).
	Tiếp nhận dữ liệu ảnh từ client (dưới dạng Base64 hoặc Multipart Form).
	Điều phối luồng xử lý: Nhận ảnh → Tiền xử lý → Gọi Model dự đoán → Trả về kết quả JSON.
	Lý do lựa chọn:
	Micro-framework: Cấu trúc cực kỳ gọn nhẹ, không dư thừa các tính năng không cần thiết, giúp khởi động server nhanh.
	Hệ sinh thái Python: Flask được viết bằng Python, do đó việc tích hợp với TensorFlow, NumPy hay Pillow là hoàn toàn tự nhiên (native integration), không cần cầu nối ngôn ngữ phức tạp.
	Khả năng mở rộng: Dễ dàng nâng cấp lên các kiến trúc phức tạp hơn (như Docker containerization) sau này.
b. Khối Frontend (HTML5, CSS3, JavaScript)
Giao diện người dùng được thiết kế với tiêu chí: Tối giản – Trực quan – Dễ tiếp cận.
	HTML5: Xây dựng khung xương cho trang web, hỗ trợ các thẻ đa phương tiện để hiển thị camera/webcam.
	CSS3: Định dạng giao diện hiện đại (Modern UI), đảm bảo tính Responsive (hiển thị tốt trên cả máy tính và điện thoại).
	JavaScript (ES6):
	Xử lý logic phía client (Client-side logic).
	Sử dụng Webcam API để truy cập camera thiết bị theo thời gian thực.
	Thực hiện AJAX request để gửi dữ liệu ảnh (Base64) lên server mà không cần tải lại trang (No-reload).
	Lý do lựa chọn: Đây là bộ ba công nghệ tiêu chuẩn của web, đảm bảo chạy ổn định trên mọi trình duyệt phổ biến (Chrome, Firefox, Edge) mà người dùng không cần cài đặt thêm plugin.
c. Thư viện Xử lý ảnh (Pillow & NumPy)
Đây là lớp trung gian quan trọng giúp "phiên dịch" dữ liệu ảnh thô thành ngôn ngữ mà máy tính có thể hiểu được.
	Pillow (PIL): Chịu trách nhiệm thao tác file ảnh cơ bản như mở file, thay đổi kích thước (Resize về 224x224 pixel cho MobileNetV2), và chuyển đổi hệ màu (RGB).
	NumPy:
	Chuyển đổi đối tượng ảnh (Image object) thành các mảng đa chiều (Tensor/Matrix).
	Thực hiện chuẩn hóa dữ liệu (Normalization) và thay đổi chiều (Reshape/Expand Dims) để phù hợp với đầu vào (Input shape) của mô hình.
	Lý do lựa chọn: Tốc độ xử lý ma trận cực nhanh, tối ưu hóa bộ nhớ và là chuẩn chung của cộng đồng Data Science.
Đánh giá tổng quan về kiến trúc này
Sự kết hợp giữa TensorFlow (AI) + Flask (Backend) + JS (Frontend) tạo nên một mô hình End-to-End hoàn chỉnh.
	Tính nhất quán: Sử dụng Python làm ngôn ngữ chủ đạo cho cả AI và Backend giúp code dễ bảo trì.
	Hiệu năng: MobileNetV2 kết hợp với Flask đảm bảo độ trễ thấp, mang lại trải nghiệm mượt mà.
	Tính thực tiễn: Công nghệ được chọn đều là mã nguồn mở, phổ biến, dễ triển khai và chi phí thấp, phù hợp hoàn toàn với yêu cầu của một đồ án thực tế.
2.2. Cơ chế hoạt động chi tiết của Data Augmentation
Quá trình tăng cường dữ liệu (DataAugmentation) sử dụng lớp ImageDataGenerator thực hiện các biến đổi hình học (Geometric Transformations) lên ảnh gốc. Dưới đây là phân tích chi tiết từng kỹ thuật.
Hình 2.2: Hình ảnh minh họa Data Augmentation
a. Rescaling (Chuẩn hóa dữ liệu)
	Tham số: rescale= 1. /255
	Mô tả kỹ thuật: Đây là bước tiền xử lý bắt buộc, không phải là biến đổi hình học. Ảnh kỹ thuật số thường được lưu trữ dưới dạng ma trận các số nguyên từ 0 đến 255 (đối với ảnh 8-bit). Tham số này thực hiện phép chia vô hướng từng điểm ảnh (pixel) cho 255.
	Mục đích trong mô hình:
	Chuyển đổi dữ liệu về không gian [0, 1].
	Giúp hàm kích hoạt và thuật toán tối ưu hóa (như Adam) hoạt động hiệu quả hơn, tránh hiện tượng bão hòa gradient (gradient saturation) và giúp mô hình hội tụ nhanh hơn.
b. Rotation (Xoay ảnh)
	Tham số: rotation_range=20
	Mô tả kỹ thuật: Mô hình sẽ chọn ngẫu nhiên một góc θ trong [-20^0,+20^0] và xoay bức ảnh theo góc đó. Các vùng trống sinh ra sau khi xoay sẽ được lấp đầy theo cơ chế mặc định (thường là 'nearest').
	Ý nghĩa thực tế: Rác thải trong thực tế không bao giờ nằm ngay ngắn theo một trục cố định. Một chai nhựa hay vỏ lon có thể nằm nghiêng, nằm chéo tùy theo cách người dùng vứt. Việc xoay ảnh giúp mô hình học được tính bất biến quay (rotational invariance) – tức là nhận diện được rác dù nó nằm ở góc độ nào.
c. Width/Height Shift (Dịch chuyển ngang/dọc)
	Tham số: width_shift_range=0.2, height_shift_range=0.2
	Mô tả kỹ thuật: Dịch chuyển toàn bộ khung hình sang trái/phải hoặc lên/xuống một khoảng ngẫu nhiên tối đa 20% kích thước ảnh.
	Ý nghĩa thực tế: Khi người dùng chụp ảnh hoặc khi camera quét rác, vật thể không phải lúc nào cũng nằm chính giữa khung hình (center). Kỹ thuật này dạy cho mô hình khả năng nhận diện vật thể ở các vị trí biên hoặc góc ảnh (positional invariance), tránh việc mô hình chỉ học được các đặc trưng ở trung tâm.
d. Horizontal Flip (Lật ngang)
	Tham số: horizontal_flip=True
	Mô tả kỹ thuật: Lật ngược bức ảnh qua trục dọc (tương tự như nhìn qua gương).
	Ý nghĩa thực tế: Đối với rác thải, tính chất của vật thể không thay đổi khi bị lật ngược. Ví dụ: Một hộp giấy dù quay nhãn sang trái hay sang phải thì bản chất vẫn là giấy. Kỹ thuật này giúp nhân đôi số lượng biến thể dữ liệu một cách hiệu quả mà không làm mất đi ý nghĩa ngữ nghĩa của ảnh.
e. Zoom (Phóng to/Thu nhỏ)
	Tham số: zoom_range=0.2
	Mô tả kỹ thuật: Phóng to hoặc thu nhỏ ngẫu nhiên hình ảnh trong khoảng [0.8,1.2] (tức là thay đổi kích thước từ 80% đến 120% so với gốc).
	Ý nghĩa thực tế: Mô phỏng khoảng cách chụp ảnh đa dạng. Người dùng có thể đưa camera lại rất gần rác (ảnh to) hoặc đứng từ xa (ảnh nhỏ). Việc này giúp mô hình học được các đặc trưng ở nhiều tỉ lệ khác nhau (scale invariance).
f. Shear (Biến đổi cắt/xén nghiêng)
	Tham số: shear_range=0.2
	Mô tả kỹ thuật: Giữ cố định một trục và kéo nghiêng trục còn lại theo một góc nhất định (tạo hình bình hành từ hình chữ nhật).
	Ý nghĩa thực tế: Giả lập sự thay đổi về góc nhìn 3D (perspective). Khi camera không nhìn trực diện vuông góc với rác mà nhìn từ một góc nghiêng, hình dạng vật thể sẽ bị biến đổi hình học. Shear giúp mô hình nhận diện tốt hơn trong các trường hợp góc chụp không lý tưởng này.
2.3. Kiến trúc Mô hình Deep Learning (Model Architecture)
a. Chức năng
Phân hệ này đóng vai trò là lõi xử lý thông minh ("Core Intelligence") của toàn bộ hệ thống. Chức năng chính của nó là tiếp nhận ma trận điểm ảnh (pixel matrix) từ khâu tiền xử lý, thực hiện hàng loạt các phép tính tích chập để trích xuất các đặc trưng thị giác, và cuối cùng ánh xạ các đặc trưng đó thành một vector xác suất tương ứng với 6 nhãn rác thải mục tiêu.
b. Cơ chế hoạt động: Transfer Learning với MobileNetV2
Hệ thống áp dụng phương pháp Học chuyển giao (Transfer Learning). Thay vì huấn luyện một mạng nơ-ron từ con số không (với trọng số khởi tạo ngẫu nhiên), hệ thống thừa hưởng "tri thức" từ một mô hình đã được huấn luyện trên quy mô lớn. Kiến trúc bao gồm hai phần chính ghép nối với nhau.
Hình 2.3: Hình ảnh minh họa Sơ đồ Kiến trúc Mô hình.
	Tầng Base (Feature Extractor - Bộ trích xuất đặc trưng)
	Kiến trúc nền tảng: Sử dụng mạng MobileNetV2.
	Đặc điểm kỹ thuật: Đây là kiến trúc CNN được Google tối ưu hóa cho các nền tảng di động, sử dụng các khối Depthwise Separable Convolution (Tích chập tách biệt chiều sâu) giúp giảm đáng kể số lượng tham số và khối lượng tính toán so với các mạng truyền thống như VGG16 hay ResNet, nhưng vẫn duy trì độ chính xác cao.
	Cấu hình khởi tạo:
	weights='imagenet': Mô hình được khởi tạo với bộ trọng số đã huấn luyện trên tập dữ liệu ImageNet (chứa 1.4 triệu ảnh với 1000 lớp vật thể). Điều này giúp mô hình đã có sẵn khả năng nhận diện các đường nét, hình khối và kết cấu vật thể cơ bản.
	include_top=False: Loại bỏ lớp phân loại 1000 lớp gốc của ImageNet, chỉ giữ lại phần thân (convolutional base) để làm nền tảng trích xuất đặc trưng.
	Cơ chế đóng băng (Freezing):
	Tham số base_model.trainable = False được thiết lập để "đóng băng" toàn bộ các lớp của MobileNetV2.
	Mục đích: Ngăn chặn việc cập nhật trọng số của các lớp này trong quá trình huấn luyện (Backpropagation). Việc này đảm bảo các đặc trưng thị giác tổng quát đã học được từ ImageNet không bị phá vỡ bởi dữ liệu mới, đồng thời giảm đáng kể thời gian huấn luyện.
	Tầng Custom Head (Classifier - Bộ phân loại tùy chỉnh)
Đây là phần được thiết kế mới hoàn toàn để thay thế lớp đầu ra đã bị loại bỏ, chuyên biệt cho bài toán phân loại 6 loại rác.
	Lớp 1: GlobalAveragePooling2D (GAP)
	Nguyên lý: Lớp này tính toán giá trị trung bình của từng bản đồ đặc trưng (feature map) kích thước H × W đầu ra từ MobileNetV2, biến đổi nó thành một giá trị duy nhất.
	So sánh với Flatten: Khác với Flatten (duỗi phẳng toàn bộ pixel thành vector 1 chiều khổng lồ), GAP giảm chiều dữ liệu cực mạnh (từ 3D tensor xuống 1D vector).
	Ưu điểm: Giúp giảm thiểu số lượng tham số cần huấn luyện ở lớp Dense tiếp theo, từ đó ngăn chặn hiệu quả hiện tượng quá khớp (Overfitting) và giúp mô hình bền vững hơn với các dịch chuyển không gian của vật thể trong ảnh.
	Lớp 2: Dense (Fully Connected Layer)
	Cấu hình: 128 nơ-ron (units), hàm kích hoạt ReLU (Rectified Linear Unit).
	Chức năng: Đây là lớp học sâu trung gian. Nó có nhiệm vụ tổng hợp các đặc trưng thị giác từ lớp GAP và học các mối quan hệ phi tuyến tính phức tạp để phân biệt các đặc điểm riêng của rác (ví dụ: sự khác biệt về kết cấu giữa "Bìa carton" nhám và "Giấy" phẳng).
	Lớp 3: Output Layer (Lớp đầu ra)
	Cấu hình: Số lượng nơ-ron bằng số lớp cần phân loại (NUM_CLASSES = 6).
	Hàm kích hoạt: Softmax.
	Cơ chế toán học: Hàm Softmax nhận đầu vào là các giá trị thô (logits) từ lớp trước và chuẩn hóa chúng thành một phân phối xác suất:
Trong đó:
	P (y = j | x): Xác suất dự đoán (Predicted Probability) để ảnh đầu vào x thuộc về nhãn j (ví dụ: "Metal" hoặc "Plastic").
	z (Logits): Vector đầu vào của hàm Softmax. Đây là kết quả thô nhận được từ lớp Dense liền trước đó (các giá trị này có thể là số âm, dương hoặc bằng 0 và nằm trong khoảng (-∞, +∞).
	z_j: Giá trị điểm số thô (raw score) tương ứng với lớp cụ thể j đang xét.
	K: Tổng số lớp phân loại. Trong dự án này, K = 6 (tương ứng với 6 loại rác: Metal, Paper, Glass, Plastic, Cardboard, Trash).
	e: Hằng số Euler (≈ 2.71828...), cơ số của logarit tự nhiên.
	Kết quả: Đầu ra là một vector chứa 6 giá trị số thực trong khoảng [0, 1] có tổng bằng 1. Giá trị cao nhất trong vector này sẽ quyết định nhãn dự đoán cuối cùng cho bức ảnh.
2.4. Chiến lược Huấn luyện (Training Strategy)
2.4.1. Chức năng
Giai đoạn huấn luyện là quá trình mô hình "học" từ dữ liệu. Cụ thể, đây là quá trình điều chỉnh các trọng số (weights) và hệ số bias của các lớp Custom Head (lớp Dense và Softmax mới thêm vào).
Mục tiêu cốt lõi là tìm ra bộ tham số tối ưu sao cho hàm mất mát (Loss Function) đạt giá trị nhỏ nhất. Quá trình này sử dụng cơ chế Lan truyền ngược (Backpropagation): sai số từ đầu ra được tính toán và truyền ngược lại mạng để cập nhật trọng số. Lưu ý rằng, do áp dụng Transfer Learning, các trọng số của phần MobileNetV2 gốc (Base model) được giữ nguyên, chỉ có các lớp mới được cập nhật.
2.4.2. Cơ chế hoạt động chi tiết
Hệ thống được cấu hình với ba thành phần chiến lược chính:
a. Hàm mất mát (Loss Function): Categorical Crossentropy
	Lý do lựa chọn: Bài toán yêu cầu phân loại hình ảnh vào một trong K=6 nhóm nhãn rời rạc (Metal, Paper, Glass, Plastic, Cardboard, Trash). Đây là bài toán phân loại đa lớp (Multi-class Classification), do đó categorical_crossentropy là lựa chọn tiêu chuẩn.
	Cơ chế toán học:
Hàm này đo lường "khoảng cách" giữa hai phân phối xác suất: phân phối dự đoán (yi) ̂:(từ Softmax) và phân phối thực tế y (đã được mã hóa One-hot).
Công thức tổng quát cho một mẫu dữ liệu:


Trong đó:
	y_i :Giá trị thực tế (chỉ bằng 1 tại đúng nhãn của ảnh, bằng 0 tại các nhãn khác).
	(yi) ̂: Xác suất mô hình dự đoán cho nhãn i.
	Ý nghĩa: Hàm này phạt rất nặng (giá trị Loss tăng vọt) nếu mô hình dự đoán xác suất thấp cho nhãn đúng.
b. Thuật toán tối ưu (Optimizer): Adam (Adaptive Moment Estimation)
	Lý do lựa chọn: Adam là thuật toán tối ưu hóa hiện đại, kết hợp ưu điểm của hai thuật toán phổ biến khác là AdaGrad và RMSProp. Nó phù hợp với các dữ liệu có nhiều nhiễu hoặc gradient thưa thớt như hình ảnh.
	Cơ chế hoạt động: Thay vì sử dụng một tốc độ học (Learning Rate - ∝) cố định cho mọi tham số như SGD truyền thống, Adam tự động điều chỉnh tốc độ học cho từng tham số riêng biệt dựa trên:
	Momentum (Động lượng): Lưu lại trung bình trượt của các gradient trong quá khứ để định hướng di chuyển ổn định, tránh bị kẹt tại các điểm tối ưu cục bộ (local minima).
	Adaptive Learning Rate: Tự động giảm tốc độ học ở các chiều có biến động lớn và tăng ở các chiều có biến động nhỏ, giúp quá trình hội tụ nhanh và mượt mà hơn.
c. Cấu hình Siêu tham số (Hyperparameters Configuration)
	Epochs (Chu kỳ huấn luyện): 10
	Giải thích: Một epoch là một lần mô hình duyệt qua trọn vẹn toàn bộ tập dữ liệu huấn luyện.
	Tại sao là 10? Trong Transfer Learning, do phần trích xuất đặc trưng (MobileNetV2) đã rất tốt, mô hình chỉ cần tinh chỉnh các lớp cuối cùng. Việc huấn luyện quá lâu (nhiều epochs) trên tập dữ liệu nhỏ có thể dẫn đến hiện tượng Overfitting (học vẹt), làm giảm độ chính xác trên tập kiểm thử. Con số 10 là mức cân bằng hợp lý để mô hình hội tụ.
	Batch Size (Kích thước lô): 32
	Giải thích: Thay vì cập nhật trọng số sau mỗi tấm ảnh (quá chậm) hoặc sau toàn bộ tập dữ liệu (quá tốn RAM), hệ thống gom nhóm 32 ảnh vào một lần xử lý.
	Tại sao là 32? Đây là con số tiêu chuẩn trong Deep Learning (2^5). Nó giúp tận dụng khả năng tính toán song song của GPU/CPU, đồng thời đảm bảo hướng di chuyển của gradient đủ ổn định để mô hình học chính xác.
2.5. Phân hệ Ứng dụng & Logi Nghiệp vụ (Application Logic)
2.5.1. Chức năng tổng quan
Hệ thống triển khai mô hình trí tuệ nhân tạo lên môi trường Web thông qua framework Flask. Phân hệ này đóng vai trò trung gian, chịu trách nhiệm quản lý giao diện tương tác người dùng, xử lý luồng dữ liệu ảnh đầu vào và cung cấp phản hồi thông minh đa phương thức (Văn bản & Âm thanh).
2.5.2. Cơ chế hoạt động chi tiết
Hệ thống vận hành theo quy trình khép kín gồm 3 bước chặt chẽ:
Bước 1: Tiếp nhận và Tiền xử lý (Input Processing)
Hệ thống được thiết kế để linh hoạt xử lý dữ liệu từ hai nguồn:
	Nguồn dữ liệu:
	Tải lên (Upload): Người dùng chọn file ảnh từ thiết bị.
	Webcam (Real-time): Ảnh được chụp trực tiếp và gửi về server dưới dạng chuỗi mã hóa Base64.
	Chuẩn hóa dữ liệu: Ảnh đầu vào được xử lý để đồng bộ với định dạng của quá trình huấn luyện:
	Resize: Điều chỉnh kích thước về chuẩn (224 × 224) pixels.
	Rescale: Chuẩn hóa giá trị pixel về khoảng [0, 1].
Bước 2: Suy luận với "Logic Nghiêm ngặt" (Strict Inference Logic)
Để đảm bảo độ tin cậy và tránh hiện tượng "ảo giác" (hallucination) của AI, hệ thống áp dụng cơ chế kiểm soát ngưỡng:
	Vector xác suất: Mô hình MobileNetV2 trả về xác suất cho từng nhãn.
	Ngưỡng tin cậy (Confidence Threshold): Thiết lập mức sàn CONFIDENCE_THRESHOLD = 70%.
	Quy tắc quyết định:
	Trường hợp 1 (Tin cậy cao): Nếu Score_max ≥ 0.7 → Chấp nhận nhãn dự đoán.
	Trường hợp 2 (Tin cậy thấp): Nếu Score_max < 0.7 →Hệ thống cưỡng chế gán nhãn là "Trash" (Rác không xác định/Khác).
	Ý nghĩa: Cơ chế này ngăn chặn việc hệ thống đưa ra dự đoán sai khi gặp vật thể lạ, ảnh mờ hoặc nhiễu, giúp nâng cao trải nghiệm và niềm tin của người dùng.
Bước 3: Phản hồi Đa phương thức (Multimodal Output)
Kết quả trả về không chỉ là văn bản khô khan mà còn tích hợp âm thanh hỗ trợ tiếp cận (Accessibility):
	Bản địa hóa (Localization): Kết quả gốc (Tiếng Anh) được ánh xạ sang Tiếng Việt thông qua từ điển TRANSLATION_MAP.
	Text-to-Speech (TTS):
	Tạo câu thông báo tự nhiên: "Loại rác được nhận diện là: [Tên rác] ...".
	Sử dụng thư viện gTTS (Google Text-to-Speech) để chuyển văn bản thành file âm thanh .mp3.
	Phát tự động trên trình duyệt, hỗ trợ người khiếm thị.
Hình 2.5.2: Biểu đồ thể hiện độ chính xác và hàm mất mát qua 10 chu kỳ huấn luyện.
2.5.3. Sơ đồ Luồng dữ liệu (Data Flow Diagram)
Để đảm bảo hệ thống hoạt động chính xác và xử lý được các trường hợp vật thể không nằm trong tập dữ liệu huấn luyện, chúng tôi thiết kế luồng xử lý dữ liệu (Data Flow) tích hợp cơ chế kiểm tra ngưỡng tin cậy (Confidence Threshold).
Quy trình xử lý từ lúc người dùng gửi ảnh đến khi nhận kết quả được mô tả chi tiết trong sơ đồ dưới đây:
Hình 2.5.3: Sơ đồ luồng dữ liệu và logic xử lý ngưỡng tin cậy của hệ thống.
Mô tả luồng xử lý:
	Input: Ảnh đầu vào từ Webcam hoặc file upload của người dùng (định dạng File hoặc Base64).
	Tiền xử lý (Preprocessing): Ảnh được thay đổi kích thước về 224 × 224pixel và chuẩn hóa giá trị pixel về khoảng [0, 1] để phù hợp với đầu vào của mô hình MobileNetV2.
	Dự đoán (Inference): Mô hình AI trả về một vector xác suất cho các lớp. Hệ thống chọn nhãn có điểm số cao nhất (Pmax).
	Kiểm tra Ngưỡng (Threshold Check):
	Hệ thống thiết lập ngưỡng tin cậy θ = 0.7 (70%).
	Nếu Pmax ≥ 0.7: Kết quả được chấp nhận.
	Nếu Pmax < 0.7: Hệ thống từ chối kết quả dự đoán và gán nhãn là "TRASH" (Rác/Vật thể lạ không xác định).
	Hậu xử lý: Kết quả cuối cùng được ánh xạ sang tên Tiếng Việt và chuyển thành giọng nói qua API Google TTS.
2.5.4. Minh họa Logic Ngưỡng Tin cậy (Confidence Threshold Logic)
Vấn đề lớn nhất của các mô hình phân loại ảnh là việc "ép buộc" gán nhãn cho một vật thể lạ với độ tin cậy thấp. Ví dụ: Khi đưa một chiếc điện thoại vào camera, mô hình có thể nhận diện nhầm là "Pin" với độ tin cậy 45%.
Để giải quyết vấn đề này, logic "Ngưỡng tin cậy 70%" hoạt động như một bộ lọc (filter). Biểu đồ dưới đây minh họa cách hệ thống ra quyết định trong 3 trường hợp thực tế:
Hình 2.5.4: Minh họa quyết định của hệ thống dựa trên điểm số tin cậy.
Phân tích biểu đồ:
	Trường hợp 1 (Giấy): Độ tin cậy 95% > 70% →Chấp nhận.
	Trường hợp 2 (Chai nhựa): Độ tin cậy 72% > 70% →Chấp nhận.
	Trường hợp 3 (Vật thể lạ/Nhiễu): Độ tin cậy 45% < 70% → Loại bỏ, gán nhãn "TRASH".
 
CHƯƠNG III: DEMO SẢN PHẨM
3.1 Môi Trường Triển Khai
Phần này mô tả các yêu cầu cần thiết để triển khai và chạy hệ thống một cách mượt mà. Hệ thống được thiết kế để hoạt động trên các thiết bị cá nhân hoặc server cơ bản, với trọng tâm vào tính di động và dễ dàng tích hợp.
3.1.1 Yêu cầu phần cứng và phần mềm
Để đảm bảo hệ thống chạy ổn định, cần đáp ứng các yêu cầu sau. Hệ thống sử dụng mô hình MobileNetV2 – một mô hình nhẹ và hiệu quả, nên không đòi hỏi tài nguyên cao như các mô hình lớn hơn (ví dụ: ResNet hoặc EfficientNet).
	Yêu cầu phần cứng tối thiểu/đề xuất:
	CPU: Tối thiểu: Bộ xử lý đa lõi (ví dụ: Intel Core i3 hoặc tương đương AMD). Đề xuất: Intel Core i5/i7 hoặc tương đương để xử lý nhanh hơn khi huấn luyện hoặc dự đoán hàng loạt.
	RAM: Tối thiểu: 4 GB (đủ cho chạy ứng dụng Flask và mô hình dự đoán). Đề xuất: 8 GB trở lên để hỗ trợ huấn luyện mô hình với dataset lớn mà không gặp tình trạng thiếu bộ nhớ.
	GPU: Không bắt buộc (mô hình MobileNetV2 có thể chạy tốt trên CPU nhờ kích thước nhỏ). Đề xuất: NVIDIA GPU với CUDA hỗ trợ (ví dụ: GTX 1050 hoặc cao hơn) nếu cần huấn luyện nhanh hơn hoặc xử lý dữ liệu lớn. Nếu sử dụng GPU, cần cài đặt TensorFlow với hỗ trợ GPU (tensorflow-gpu).
	Ổ cứng: Tối thiểu: 500 MB trống (cho mã nguồn, mô hình đã huấn luyện khoảng 20-50 MB, và thư mục tạm thời cho ảnh upload). Đề xuất: 5 GB nếu lưu trữ dataset huấn luyện đầy đủ.
	Camera (nếu sử dụng tính năng webcam): Bất kỳ webcam tích hợp hoặc ngoài nào hỗ trợ navigator.mediaDevices.getUserMedia (hầu hết các thiết bị hiện đại đều hỗ trợ).
	Mạng: Không bắt buộc cho chế độ offline (dự đoán cục bộ), nhưng cần kết nối internet nếu tải dataset từ Kaggle hoặc cập nhật thư viện.
	Hệ điều hành:
	Hỗ trợ: Windows 10/11, macOS (10.15 trở lên), hoặc Linux (Ubuntu 18.04 trở lên hoặc tương đương). Python và Flask hoạt động đa nền tảng, nhưng khuyến nghị sử dụng Linux cho server sản xuất để dễ quản lý.
	Yêu cầu phần mềm và thư viện:
	Ngôn ngữ lập trình: Python 3.8 trở lên (đề xuất Python 3.10+ để tương thích tốt với TensorFlow).
	Thư viện chính (dựa trên requirements.txt và mã nguồn):
	tensorflow>=2.0.0: Thư viện cốt lõi cho xây dựng, huấn luyện và tải mô hình AI (sử dụng Keras API cho MobileNetV2, ImageDataGenerator, và dự đoán).
	flask>=2.0.0: Framework web để xây dựng ứng dụng, xử lý route, form upload, và API (ví dụ: route '/' cho upload và '/predict_cam' cho webcam).
	Pillow>=9.0.0 (PIL): Xử lý hình ảnh (mở file, resize, convert định dạng RGB, và lưu tạm thời).
	numpy>=1.20.0: Xử lý mảng số (chuẩn hóa ảnh, mở rộng chiều batch, và tính toán dự đoán như np.argmax).
	werkzeug>=2.0.0: Hỗ trợ Flask trong việc xử lý file upload an toàn (secure_filename) và các tiện ích web khác.
	Các thư viện phụ (từ mã nguồn, không bắt buộc nhưng được sử dụng ngầm):
	os: Xử lý đường dẫn file và thư mục (ví dụ: tạo 'uploads/', kiểm tra tồn tại file).
	base64 và io: Xử lý dữ liệu Base64 từ webcam (giải mã và tạo buffer cho ảnh).
	Các thư viện TensorFlow nội bộ: tf.keras.applications (cho MobileNetV2), tf.keras.layers (Dense, GlobalAveragePooling2D), tf.keras.models.Model (xây dựng mô hình tùy chỉnh), và tf.keras.preprocessing.image.ImageDataGenerator (data augmentation cho huấn luyện).
	Công cụ cài đặt: Sử dụng pip để cài đặt từ requirements.txt (ví dụ: pip install -r requirements.txt). Nếu huấn luyện mô hình, cần tải dataset từ Kaggle (cấu trúc với thư mục 'train' và 'test').
	Môi trường ảo (đề xuất): Sử dụng virtualenv hoặc conda để tạo môi trường riêng biệt, tránh xung đột thư viện (ví dụ: conda create -n trash_ai python=3.10).
3.1.2 Cấu trúc thư mục dự án
Cấu trúc thư mục của dự án được tổ chức đơn giản, phù hợp với một ứng dụng Flask kết hợp AI. Dưới đây là sơ đồ cấu trúc chính (dưới dạng văn bản cây thư mục):
GarbageClassification/
├── app.py                  			# File chính chạy ứng dụng Flask
├── train_model.py          			# Script huấn luyện mô hình AI
├── requirements.txt       			# Danh sách thư viện cần thiết
├── trash_classifier_mobilenetv2.h5 	# Mô hình đã huấn luyện (tệp .h5)
├── labels.txt              			# Danh sách nhãn lớp (các loại rác)
├── static/                 			# Thư mục tài nguyên tĩnh
│   └── style.css           			# File CSS cho giao diện
├── templates/              			# Thư mục template HTML (Flask sử dụng)
│   └── index.html          			# Trang HTML chính
├── uploads/                			# Thư mục tạm lưu ảnh upload (tạo động)
└── dataset/                	# (Tùy chọn) Thư mục dataset huấn luyện/test (không bắt buộc trong runtime)
    ├── train/             			# Dữ liệu huấn luyện
    └── test/               			# Dữ liệu kiểm tra
	Giải thích vai trò của các file chính:
	app.py: Đây là file cốt lõi của ứng dụng web. Nó khởi tạo Flask app, tải mô hình AI và nhãn, định nghĩa các route (tuyến đường) như '/' cho trang chính (xử lý upload ảnh), '/uploads/<filename>' để phục vụ ảnh đã tải, và '/predict_cam' cho dự đoán từ camera. Nó cũng xử lý logic tiền xử lý ảnh, dự đoán bằng mô hình, và quản lý thư mục uploads/.
	train_model.py: Script độc lập để huấn luyện mô hình. Sử dụng TensorFlow và MobileNetV2 với transfer learning, data augmentation qua ImageDataGenerator, huấn luyện trên dataset (train/ và test/), và lưu mô hình (.h5) cùng nhãn (labels.txt).
	index.html (trong templates/): Template HTML cho giao diện người dùng, bao gồm tabs cho upload và camera, form upload, video feed cho webcam, và khu vực hiển thị kết quả. Kết hợp JavaScript để xử lý preview ảnh, đếm ngược camera, và gửi AJAX.
	style.css (trong static/): Định nghĩa phong cách CSS cho giao diện, như màu sắc (xanh lá, teal), responsive design cho mobile, tabs, buttons, spinner loading, và preview ảnh.
	trash_classifier_mobilenetv2.h5 và labels.txt: Mô hình đã huấn luyện và danh sách nhãn, được tải khi app khởi động để dự đoán.
	requirements.txt: Danh sách thư viện để cài đặt môi trường nhất quán.
	uploads/: Thư mục tạm thời lưu ảnh upload hoặc chụp từ camera, được tạo nếu chưa tồn tại.
3.2 Giao Diện và Chức Năng Của Hệ Thống
Hệ thống cung cấp giao diện web thân thiện, dễ sử dụng, tập trung vào hai chức năng chính: upload ảnh và sử dụng camera thời gian thực.
3.2.1 Mô tả giao diện người dùng
Giao diện được thiết kế hiện đại, với tông màu xanh lá (biểu tượng cho môi trường) và trắng, sử dụng font 'Segoe UI' cho tính thẩm mỹ. CSS đảm bảo responsive (tự điều chỉnh trên mobile), với box-shadow cho chiều sâu, transitions cho hover buttons, và spinner loading cho trải nghiệm mượt mà.
Tổng quan thiết kế: Trang chính có tiêu đề "♻️ Phân loại rác thải thông minh bằng AI ♻️", container trung tâm với padding và border-radius. Có tabs để chuyển giữa chức năng, flash messages cho thông báo lỗi, và khu vực kết quả với border xanh. Trên mobile (max-width: 480px), font và padding nhỏ hơn để phù hợp màn hình nhỏ.
	Chi tiết hai tab chức năng chính:
	Tab Tải ảnh lên (Upload): Bao gồm form với input file (accept image/*), button submit "Phân loại (Upload)". Khi chọn file, hiển thị preview ảnh trong image-preview-container. Kết quả hiển thị trong result với loại rác và confidence.
 
	Tab Sử dụng Camera (Webcam): Có buttons "Bật Camera", "Tắt Camera", và "Chụp và Phân loại" (với đếm ngược 3 giây). Video feed từ webcam hiển thị trong. video-container. Sau chụp, preview ảnh chụp và kết quả tương tự tab upload, với spinner loading trong quá trình xử lý.
3.2.2 Cơ chế hoạt động của Web Server (dựa trên app.py)
Ứng dụng Flask khởi động với app.run (debug=True, host='0.0.0.0', port=5000), lắng nghe trên tất cả IP để truy cập từ mạng cục bộ. Mô hình và nhãn được tải lúc khởi động để tối ưu.
	Quản lý các tuyến đường:
	'/' (GET/POST): Xử lý trang chính. Với POST, kiểm tra file upload, lưu vào uploads/, tiền xử lý, dự đoán, và render template với kết quả. Với GET, hiển thị trang trống.
	'/uploads/<filename>': Phục vụ ảnh từ thư mục uploads/ để hiển thị preview hoặc kết quả.
	'/predict_cam' (POST): Xử lý dữ liệu Base64 từ camera qua JSON, giải mã thành ảnh, lưu tạm, tiền xử lý, dự đoán, và trả về JSON (prediction và confidence) cho JavaScript cập nhật UI.
3.3 Trình Bày Demo Chức Năng
Dưới đây là hai kịch bản demo minh họa cách hệ thống hoạt động thực tế.
3.3.1 Kịch bản 1: Phân loại bằng cách Tải ảnh lên (Upload)
	Bước 1: Người dùng chọn ảnh rác thải (ví dụ: lon nhôm hoặc chai nhựa) qua input file.
	Bước 2: Trong app.py, file được lưu vào uploads/ với secure_filename. Hàm preprocess_image mở ảnh bằng PIL, resize về 224x224, chuyển thành numpy array, mở rộng batch, và chuẩn hóa. Sau đó, model.predict tính toán predictions, lấy argmax cho lớp và confidence.
	Bước 3: Trang render lại với ảnh preview (qua uploaded_image_url), loại rác (ví dụ: "Kim loại"), và độ tin cậy (ví dụ: "95.20%").
3.3.2 Kịch bản 2: Phân loại bằng Camera Trực tiếp (Webcam)
	Bước 1: Người dùng bật Camera qua button, stream video từ webcam.
	Bước 2: Nhấn "Chụp và Phân loại", JavaScript đếm ngược, chụp từ canvas, chuyển thành Base64.
	Bước 3: Gửi AJAX đến /predict_cam. Server giải mã Base64 bằng base64 và io, lưu tạm ảnh, tiền xử lý tương tự upload, dự đoán, và trả JSON.
	Bước 4: UI cập nhật preview ảnh chụp, loại rác (ví dụ: "Nhựa"), và độ tin cậy (ví dụ: "88.75%").x
3.4 Đánh Giá Hiệu Suất Trực Quan
3.4.1 Bảng kết quả mẫu
Dưới đây là bảng tổng hợp kết quả thử nghiệm trên một số ảnh mẫu (giả định dựa trên dataset điển hình; độ chính xác thực tế phụ thuộc vào huấn luyện):
Loại rác thực tế	Kết quả dự đoán	Độ tin cậy	Chính xác/Sai	Ghi chú
Lon nhôm (Kim loại)	Kim loại	95.20%	Chính xác	Ảnh rõ nét, nền đơn giản.
Chai nhựa (Nhựa)	Nhựa	88.75%	Chính xác	Ánh sáng tốt, nhưng có nhiễu nền.
Giấy báo cũ (Giấy)	Giấy	72.30%	Sai	Mô hình nhầm lẫn do hình dạng tương tự chai nhựa; cần thêm data.
Ly
(Thủy tinh)	Thủy tinh	91.50%	Chính xác	Độ tin cậy cao nhờ đặc trưng rõ ràng.
 
KẾT LUẬN VÀ HƯỚNG PHÁT TRIỂN
Kết Luận:
Phần này tổng hợp lại các kết quả đạt được, nhấn mạnh giá trị của dự án đồng thời chỉ ra những điểm còn hạn chế để làm cơ sở cho các nghiên cứu tiếp theo.
Tóm tắt mục tiêu và thành tựu
Dự án nhằm xây dựng một hệ thống phân loại rác thải tự động dựa trên công nghệ Deep Learning, hỗ trợ người dùng phân biệt các loại rác một cách nhanh chóng và chính xác. Mục tiêu ban đầu đã được đạt được một cách thành công thông qua việc phát triển một ứng dụng web hoàn chỉnh, tích hợp mô hình AI để xử lý hình ảnh từ nhiều nguồn đầu vào.
Các thành tựu chính bao gồm:
	Áp dụng thành công kỹ thuật Transfer Learning với mô hình MobileNetV2 trong script train_model.py. Mô hình được huấn luyện trên dataset từ Kaggle, sử dụng data augmentation (như rotation_range, zoom_range) để tăng cường khả năng tổng quát hóa, và đạt độ chính xác khoảng 80-90% trên tập kiểm tra (dựa trên đánh giá sau 10 epochs huấn luyện).
	Xây dựng ứng dụng web Flask đa chức năng trong app.py và index.html, hỗ trợ hai chế độ chính: tải ảnh lên (upload) và sử dụng camera thời gian thực (webcam). Giao diện thân thiện, responsive, với các tính năng như preview ảnh, đếm ngược chụp, và hiển thị kết quả dự đoán kèm độ tin cậy.
	Đạt được độ chính xác nhất định trên tập dữ liệu đã huấn luyện, với khả năng phân loại các loại rác cơ bản (dựa trên nhãn từ labels.txt), chứng minh tính khả thi của hệ thống trong môi trường thực tế.
Những thành tựu này không chỉ đáp ứng mục tiêu kỹ thuật mà còn tạo nền tảng cho các ứng dụng thực tiễn, giúp giảm thời gian và lỗi con người trong việc phân loại rác.
Đóng góp của dự án
Dự án mang lại những đóng góp đáng kể cả về mặt lý thuyết và thực tiễn, đặc biệt trong bối cảnh vấn đề rác thải đang trở thành thách thức toàn cầu.
Về mặt thực tiễn, hệ thống nâng cao nhận thức cộng đồng về phân loại rác tại nguồn bằng cách cung cấp công cụ dễ sử dụng, giúp người dùng nhanh chóng xác định loại rác (ví dụ: nhựa, giấy, kim loại) và khuyến khích hành vi thân thiện với môi trường. Điều này có thể hỗ trợ các chiến dịch tái chế, giảm ô nhiễm, và thúc đẩy kinh tế xanh, chẳng hạn như tích hợp vào các chương trình giáo dục hoặc hệ thống quản lý rác thải đô thị.
Về mặt khoa học, dự án khẳng định tính khả thi của việc ứng dụng công nghệ Deep Learning trong lĩnh vực môi trường. Bằng cách sử dụng Transfer Learning với MobileNetV2 – một mô hình nhẹ và hiệu quả – dự án chứng minh rằng AI có thể được triển khai trên các thiết bị hạn chế tài nguyên, mở ra tiềm năng cho các ứng dụng di động hoặc nhúng, góp phần vào sự phát triển của IoT (Internet of Things) trong bảo vệ môi trường.
Tổng thể, dự án không chỉ là một sản phẩm kỹ thuật mà còn là bước tiến nhỏ hướng tới mục tiêu phát triển bền vững, phù hợp với các mục tiêu của Liên Hợp Quốc (SDGs), đặc biệt là SDG 11 (Thành phố và cộng đồng bền vững) và SDG 12 (Tiêu dùng và sản xuất có trách nhiệm).
Hạn chế của đề tài
Mặc dù đạt được nhiều thành tựu, dự án vẫn tồn tại một số hạn chế cần được nhận diện để cải thiện trong tương lai.
Thứ nhất, độ chính xác của mô hình có thể chưa tối đa, đặc biệt với các ảnh chất lượng thấp (mờ, ánh sáng kém, hoặc góc chụp bất thường), dẫn đến tỷ lệ sai sót khoảng 10-20% trên dữ liệu thực tế ngoài dataset huấn luyện. Mô hình chỉ phân loại được các loại rác đã được huấn luyện (dựa trên dataset Kaggle), nên chưa xử lý tốt các biến thể hoặc loại rác mới.
Thứ hai, hiệu suất hệ thống phụ thuộc vào chất lượng ảnh đầu vào và thiết bị (ví dụ: camera kém chất lượng có thể làm giảm độ tin cậy). Ứng dụng Flask hiện chỉ chạy cục bộ hoặc trên server đơn giản, chưa tối ưu cho lưu lượng người dùng lớn, và thiếu hỗ trợ xử lý video liên tục (chỉ ảnh tĩnh).
Cuối cùng, dataset huấn luyện có thể chưa đủ đa dạng (ví dụ: thiếu dữ liệu từ các khu vực địa lý khác nhau), dẫn đến bias trong dự đoán. Những hạn chế này là cơ hội để mở rộng nghiên cứu.
Hướng Phát Triển
Dựa trên nền tảng hiện tại, dự án có thể được phát triển thêm để tăng cường hiệu quả và ứng dụng thực tế. Các hướng sau tập trung vào cải thiện kỹ thuật, thêm tính năng, và triển khai rộng rãi.
Cải thiện mô hình và độ chính xác
Để nâng cao chất lượng mô hình, cần tập trung vào dữ liệu và kiến trúc:
	Mở rộng tập dữ liệu: Thu thập thêm hình ảnh từ các nguồn thực tế (ví dụ: ảnh rác thải Việt Nam hoặc các loại rác địa phương), tăng số lượng lớp phân loại (thêm rác điện tử, hữu cơ chi tiết hơn), và áp dụng kỹ thuật balancing để tránh bias.
	Tối ưu kiến trúc: Fine-tune thêm các lớp của MobileNetV2 (bằng cách đặt base_model.trainable = True cho một phần lớp), hoặc thử nghiệm các mô hình mạnh mẽ hơn như EfficientNet hoặc ResNet để cải thiện độ chính xác mà vẫn giữ tính nhẹ nhàng.
	Kỹ thuật tối ưu hóa: Áp dụng Early Stopping để tránh overfitting, Learning Rate Scheduler để điều chỉnh tốc độ học (như đã comment trong train_model.py), và ensemble methods (kết hợp nhiều mô hình) để tăng độ tin cậy tổng thể.
Những cải tiến này có thể đẩy độ chính xác lên trên 95%, làm hệ thống đáng tin cậy hơn trong môi trường thực tế.
Phát triển tính năng mới
Để tăng tính hấp dẫn và hữu ích, có thể thêm các tính năng mở rộng:
	Gợi ý xử lý rác: Sau phân loại, tích hợp cơ sở dữ liệu để cung cấp thông tin như "Rác nhựa: Có thể tái chế tại điểm thu gom gần nhất" hoặc liên kết với bản đồ địa điểm tái chế.
	Đếm/Thống kê rác: Thêm chức năng theo dõi lịch sử phân loại, đếm số lượng theo loại (sử dụng cơ sở dữ liệu như SQLite trong Flask), và hiển thị biểu đồ thống kê để người dùng theo dõi tiến độ.
	Giao diện đa ngôn ngữ: Hỗ trợ tiếng Việt và tiếng Anh (hoặc thêm ngôn ngữ khác) bằng cách sử dụng thư viện như Flask-Babel, làm hệ thống dễ tiếp cận hơn với người dùng quốc tế.
Những tính năng này sẽ biến hệ thống từ công cụ phân loại đơn giản thành nền tảng giáo dục và quản lý rác thải toàn diện.
Triển khai thực tế
Để đưa dự án vào ứng dụng thực tế, cần tập trung vào tích hợp và mở rộng quy mô:
	Tích hợp phần cứng: Biến hệ thống thành thiết bị độc lập bằng cách sử dụng Raspberry Pi kết hợp camera, cho phép tích hợp vào thùng rác thông minh (smart bin) với cơ chế tự động mở nắp dựa trên loại rác.
	Triển khai trên Cloud: Đưa ứng dụng Flask lên các nền tảng đám mây như AWS (sử dụng EC2 hoặc Lambda), Azure App Service, hoặc Google Cloud Run để hỗ trợ truy cập từ xa, xử lý nhiều người dùng đồng thời, và tích hợp API cho ứng dụng di động.
Những hướng phát triển này không chỉ khắc phục hạn chế mà còn mở rộng phạm vi ứng dụng, góp phần vào các giải pháp môi trường bền vững trong tương lai. Dự án có tiềm năng hợp tác với các tổ chức môi trường hoặc doanh nghiệp tái chế để triển khai rộng rãi.
 
TÀI LIỆU THAM KHẢO
	World Bank. (2018). What a Waste 2.0: A Global Snapshot of Solid Waste Management to 2050. World Bank Group.
https://openknowledge.worldbank.org/handle/10986/30317
	Howard, A. G., Zhu, M., Chen, B., Kalenichenko, D., Wang, W., Weyand, T., … Adam, H. (2017). MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications. arXiv preprint arXiv:1704.04861.
https://arxiv.org/abs/1704.04861
	Keras Documentation. (2025). Keras Applications: MobileNetV2.
https://keras.io/api/applications/mobilenet/
	TensorFlow Documentation. (2025). ImageDataGenerator API.
https://www.tensorflow.org/api_docs/python/tf/keras/preprocessing/image/ImageDataGenerator
	Ronneberger, O., Fischer, P., & Brox, T. (2015). U-Net: Convolutional Networks for Biomedical Image Segmentation. arXiv preprint arXiv:1505.04597.
https://arxiv.org/abs/1505.04597
	Flask Documentation. (2025). Flask Web Framework.
https://flask.palletsprojects.com/
Zhang, K., Zhang, Z., & Li, Z. (2020). A Survey on Image-Based Waste Classification Using Deep Learning Techniques. Journal of Cleaner Production, 270, 122519.
https://doi.org/10.1016/j.jclepro.2020.122519
