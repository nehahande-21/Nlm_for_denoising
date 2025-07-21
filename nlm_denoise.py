import cv2
import numpy as np
import time

def enhance_underwater_image(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced_img

def denoise_image(image_path, output_path):
    print(f"📂 Loading image: {image_path}")
    img = cv2.imread(image_path)
    if img is None:
        print("❌ Error: Image not found or unreadable.")
        return

    enhanced = enhance_underwater_image(img)
    print("📈 Starting denoising...")

    start_time = time.time()
    denoised = cv2.fastNlMeansDenoisingColored(
        enhanced, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
    )
    end_time = time.time()

    elapsed_time = end_time - start_time
    print(f"✅ Denoising complete. Time taken: {elapsed_time:.3f} seconds")

    cv2.imwrite(output_path, denoised)
    print(f"🖼️ Denoised image saved to: {output_path}")

    # Optional: Show the output image
    cv2.imshow("Denoised Output", denoised)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# 🔄 Replace with your actual noisy image filename
denoise_image("your_underwater_image.jpeg", "denoised_output.jpeg")
