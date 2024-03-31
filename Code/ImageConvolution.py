import cv2
import numpy as np
import matplotlib.pyplot as plt

# load 图像
image_path = 'D:/BF4/Code/ImageDog.png'
image = cv2.imread(image_path, cv2.IMREAD_COLOR)

# 看图像路径
if image is None:
    raise ValueError("The image could not be loaded. Please check the file path.")

# 把图像变成gray
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Define a 3x3 kernel(卷积函数)
kernel = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)

# Apply the kernel to the grayscale image
convolved_image = cv2.filter2D(gray_image, -1, kernel)

# Display the original and convolved image
fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
axes[0].set_title('Original Image')
axes[0].axis('off')

axes[1].imshow(convolved_image, cmap='gray')
axes[1].set_title('Convolved Image')
axes[1].axis('off')

plt.tight_layout()
plt.show()
