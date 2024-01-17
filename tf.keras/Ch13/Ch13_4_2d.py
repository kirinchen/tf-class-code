from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
import matplotlib.pyplot as plt

img = load_img("koala.png") 
x = img_to_array(img) 
x = x.reshape((1,) + x.shape)  # reshape (1, hight, width, 3)
print(x.shape)

datagen = ImageDataGenerator(horizontal_flip=True,
                             vertical_flip=True)

numOfImgs = 6
i = 0
batch_imgs = []
for batch_img in datagen.flow(x, batch_size=1):
    batch_imgs.append(batch_img[0].astype("int"))
    i += 1
    if i >= numOfImgs:
        break 
    
plt.figure(figsize=(8,8))
for i in range(numOfImgs):
    plt.subplot(230+1+i)
    plt.axis("off")
    plt.imshow(batch_imgs[i])
plt.show()
 

