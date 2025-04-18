import os
import torch
import mlflow.pytorch
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms

# ===== Cấu hình =====
device = "cuda" if torch.cuda.is_available() else "cpu"
test_folder = "test_images"
class_names = [
    'T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
    'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot'
]

# ===== Tiền xử lý ảnh =====
transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((28, 28)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten 28x28 → 784
])

# ===== Load model từ MLflow =====
def load_model_from_mlflow(run_id=None, model_name="fmnist_model"):
    if run_id:
        model_uri = f"runs:/{run_id}/{model_name}"
    else:
        print("❗️Bạn cần cung cấp `run_id` từ MLflow.")
        exit(1)

    print(f"🔄 Đang tải model từ MLflow URI: {model_uri}")
    model = mlflow.pytorch.load_model(model_uri).to(device)
    model.eval()
    return model

# ===== Dự đoán ảnh trong thư mục =====
def predict_folder(model):
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print(f"📁 Đã tạo thư mục '{test_folder}'. Hãy thêm ảnh PNG/JPG và chạy lại.")
        return

    image_files = [f for f in os.listdir(test_folder) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not image_files:
        print(f"⚠️ Không tìm thấy ảnh hợp lệ trong thư mục '{test_folder}'.")
        return

    for img_name in image_files:
        img_path = os.path.join(test_folder, img_name)
        try:
            img = Image.open(img_path).convert('L')  # Đảm bảo grayscale
            input_tensor = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                output = model(input_tensor)
                _, pred_class = output.max(1)

            label = class_names[pred_class.item()]
            print(f"🖼️ {img_name} → Predict: {label}")

            plt.imshow(img, cmap='gray')
            plt.title(f'Predicted: {label}')
            plt.axis('off')
            plt.show()

        except Exception as e:
            print(f"❌ Lỗi xử lý {img_name}: {e}")

# ===== Main =====
if __name__ == "__main__":
    # 🔧 NHẬP run_id MLflow ở đây
    run_id = "1e33901108a5469b87326bc5b1bfc011"  # Thay thế dòng này bằng Run ID thật của bạn

    model = load_model_from_mlflow(run_id)
    predict_folder(model)
