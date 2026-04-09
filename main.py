import os
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
from PIL import Image, ImageOps, ImageFilter, ImageTk
import tensorflow as tf
from tensorflow.keras import layers, models


MODEL_FILE = "mnist_model.keras"



# 1. побудова та навчання моделі
def build_model():
    data_augmentation = tf.keras.Sequential([
        layers.RandomRotation(0.08),
        layers.RandomZoom(0.10),
        layers.RandomTranslation(0.10, 0.10)
    ])

    model = models.Sequential([
        layers.Input(shape=(28, 28, 1)),
        data_augmentation,
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(10, activation="softmax")
    ])

    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


def train_model():
    print("Завантаження набору даних MNIST...")
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    x_train = np.expand_dims(x_train, -1)
    x_test = np.expand_dims(x_test, -1)

    y_train = tf.keras.utils.to_categorical(y_train, 10)
    y_test = tf.keras.utils.to_categorical(y_test, 10)

    model = build_model()

    print("Початок навчання моделі...")
    model.fit(
        x_train,
        y_train,
        epochs=12,
        batch_size=128,
        validation_split=0.1,
        verbose=1
    )

    test_loss, test_acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Точність на тестовій вибірці: {test_acc * 100:.2f}%")

    model.save(MODEL_FILE)
    print(f"Модель збережено у файл: {MODEL_FILE}")

    return model


def load_or_train_model():
    if os.path.exists(MODEL_FILE):
        print("Завантаження збереженої моделі...")
        return tf.keras.models.load_model(MODEL_FILE)

    print("Файл моделі не знайдено. Починається навчання...")
    return train_model()



# задання шуму для зображення

def add_gaussian_noise(pil_img, mean=0, sigma=35):
    img = pil_img.convert("L")
    arr = np.array(img).astype(np.float32)

    noise = np.random.normal(mean, sigma, arr.shape)
    noisy = arr + noise
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)

    return Image.fromarray(noisy)

# ROF-очищення
def denoise_rof(gray_array, tv_weight=35, tau=0.125, tolerance=0.01, max_iter=80):
    m, n = gray_array.shape
    U = gray_array.copy()
    Px = gray_array.copy()
    Py = gray_array.copy()
    error = 1.0
    i = 0

    while error > tolerance and i < max_iter:
        U_old = U.copy()

        grad_x = np.roll(U, -1, axis=1) - U
        grad_y = np.roll(U, -1, axis=0) - U

        Px_new = Px + (tau / tv_weight) * grad_x
        Py_new = Py + (tau / tv_weight) * grad_y

        norm_new = np.maximum(1.0, np.sqrt(Px_new ** 2 + Py_new ** 2))
        Px = Px_new / norm_new
        Py = Py_new / norm_new

        rx_px = np.roll(Px, 1, axis=1)
        ry_py = np.roll(Py, 1, axis=0)
        div_p = (Px - rx_px) + (Py - ry_py)

        U = gray_array + tv_weight * div_p

        error = np.linalg.norm(U - U_old) / np.sqrt(n * m)
        i += 1

    return U

# підготовка до розпізнання MNIST
def center_digit_on_canvas(img, size=28, inner_size=20):
    img = img.convert("L")
    img = ImageOps.invert(img)
    img = img.point(lambda p: 255 if p > 70 else 0)

    bbox = img.getbbox()
    if bbox is None:
        blank = Image.new("L", (size, size), 0)
        arr = np.array(blank).astype("float32") / 255.0
        return np.expand_dims(arr, axis=(0, -1))

    img = img.crop(bbox)

    w, h = img.size
    if w > h:
        new_w = inner_size
        new_h = max(1, int(h * inner_size / w))
    else:
        new_h = inner_size
        new_w = max(1, int(w * inner_size / h))

    try:
        resample_method = Image.Resampling.LANCZOS
    except AttributeError:
        resample_method = Image.LANCZOS

    img = img.resize((new_w, new_h), resample_method)

    new_img = Image.new("L", (size, size), 0)
    paste_x = (size - new_w) // 2
    paste_y = (size - new_h) // 2
    new_img.paste(img, (paste_x, paste_y))

    img_array = np.array(new_img).astype("float32") / 255.0
    return np.expand_dims(img_array, axis=(0, -1))


def preprocess_with_rof(img):
    img = img.convert("L")
    img = img.filter(ImageFilter.MedianFilter(size=3))

    arr = np.array(img).astype("float32") / 255.0
    denoised = denoise_rof(arr, tv_weight=35, tau=0.125, tolerance=0.01, max_iter=80)
    denoised = np.clip(denoised, 0, 1)

    denoised_img = Image.fromarray((denoised * 255).astype("uint8"))
    denoised_img = denoised_img.point(lambda p: 255 if p > 120 else 0)

    model_input = center_digit_on_canvas(denoised_img)

    # щоб показати, що саме бачить модель
    preview_28 = (model_input[0, :, :, 0] * 255).astype("uint8")
    preview_img = Image.fromarray(preview_28)

    return model_input, preview_img


# розпізнавання
def predict_with_rotations_and_noise(model, pil_img):
    angles = [-40, -30, -20, -10, 10, 20, 30, 40]

    noise_variants = [
        ("гаусів шум", add_gaussian_noise(pil_img, sigma=35)),
    ]

    best_digit = None
    best_prob = -1
    best_angle = None
    best_variant_name = None
    best_noisy_rotated_img = None
    best_cleaned_img = None

    for variant_name, noisy_img in noise_variants:
        for angle in angles:
            rotated = noisy_img.rotate(angle, fillcolor="white")

            img_array, cleaned_img = preprocess_with_rof(rotated)
            prediction = model.predict(img_array, verbose=0)[0]

            digit = int(np.argmax(prediction))
            prob = float(np.max(prediction))

            if prob > best_prob:
                best_prob = prob
                best_digit = digit
                best_angle = angle
                best_variant_name = variant_name
                best_noisy_rotated_img = rotated
                best_cleaned_img = cleaned_img

    return (
        best_digit,
        best_prob,
        best_angle,
        best_variant_name,
        best_noisy_rotated_img,
        best_cleaned_img
    )

# функція для відображення цифри
def prepare_preview_image(img, max_size=(180, 180)):
    preview = img.copy()
    preview.thumbnail(max_size)
    return ImageTk.PhotoImage(preview)



# інтерфейс
class DigitRecognizerApp:
    def __init__(self, root, model):
        self.root = root
        self.model = model

        self.root.title("Лабораторна №2")
        self.root.geometry("760x620")
        self.root.resizable(False, False)

        self.original_photo_tk = None
        self.noisy_photo_tk = None
        self.rotated_photo_tk = None

        title_label = tk.Label(
            root,
            text="Завантаж фото цифри",
            font=("Arial", 20, "bold")
        )
        title_label.pack(pady=15)

        self.upload_button = tk.Button(
            root,
            text="Завантажити фото",
            command=self.load_image_and_predict,
            width=20,
            height=2,
            font=("Arial", 12)
        )
        self.upload_button.pack(pady=10)

        self.images_frame = tk.Frame(root)
        self.images_frame.pack(pady=15)

        self.original_title = tk.Label(self.images_frame, text="Оригінал", font=("Arial", 12, "bold"))
        self.original_title.grid(row=0, column=0, padx=10, pady=5)

        self.noisy_title = tk.Label(self.images_frame, text="Шум + поворот", font=("Arial", 12, "bold"))
        self.noisy_title.grid(row=0, column=1, padx=10, pady=5)

        self.rotated_title = tk.Label(self.images_frame, text="Після очищення", font=("Arial", 12, "bold"))
        self.rotated_title.grid(row=0, column=2, padx=10, pady=5)

        self.original_image_label = tk.Label(self.images_frame, width=180, height=180, bd=1, relief="solid")
        self.original_image_label.grid(row=1, column=0, padx=10, pady=5)

        self.noisy_image_label = tk.Label(self.images_frame, width=180, height=180, bd=1, relief="solid")
        self.noisy_image_label.grid(row=1, column=1, padx=10, pady=5)

        self.rotated_image_label = tk.Label(self.images_frame, width=180, height=180, bd=1, relief="solid")
        self.rotated_image_label.grid(row=1, column=2, padx=10, pady=5)

        self.result_label = tk.Label(
            root,
            text="Результат: -",
            font=("Arial", 24, "bold")
        )
        self.result_label.pack(pady=20)

        self.info_label = tk.Label(
            root,
            text="",
            font=("Arial", 14),
            justify="center"
        )
        self.info_label.pack()

    def load_image_and_predict(self):
        file_path = filedialog.askopenfilename(
            title="Виберіть зображення",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp"),
                ("All files", "*.*")
            ]
        )

        if not file_path:
            return

        try:
            original_img = Image.open(file_path).convert("L")

            digit, prob, angle, variant_name, noisy_rotated_img, cleaned_img = predict_with_rotations_and_noise(
                self.model, original_img
            )

            self.original_photo_tk = prepare_preview_image(original_img)
            self.noisy_photo_tk = prepare_preview_image(noisy_rotated_img)
            self.rotated_photo_tk = prepare_preview_image(cleaned_img)

            self.original_image_label.config(image=self.original_photo_tk)
            self.noisy_image_label.config(image=self.noisy_photo_tk)
            self.rotated_image_label.config(image=self.rotated_photo_tk)

            self.result_label.config(text=f"Результат: {digit}")
            self.info_label.config(
                text=f"Шум: {variant_name}\nКут: {angle}°\nЙмовірність: {prob * 100:.2f}%"
            )

        except Exception as e:
            messagebox.showerror("Помилка", f"Не вдалося обробити зображення:\n{e}")


def main():
    model = load_or_train_model()

    root = tk.Tk()
    app = DigitRecognizerApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()