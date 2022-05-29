from PIL import Image
import numpy as np
import pandas as pd
import seaborn as sns

def load_img(filepath):
    img = Image.open(filepath).convert('RGB')

    # y, _, _ = img.split()
    return img

c_1 = np.array(load_img("G:/DarkEnhancement/macro/DarkEnhancement/datasets/LOL/test/low/1.png").convert('L'))
c_2 = np.array(load_img("G:/DarkEnhancement/macro/DarkEnhancement/datasets/LOL/test/high/1.png").convert('L'))
c1 = pd.DataFrame(c_1)
c2 = pd.DataFrame(c_2)

sns.displot(c1)
