import cv2
import numpy as np
from glob import glob

import base64
import io
from PIL import Image
from io import BytesIO

try:
    from StringIO import StringIO
except ImportError:
    from io import StringIO
#from StringIO import StringIO

# K-means step1
def k_means_step1(img, Class, imgo):
	#  get shape
	H, W, C = img.shape

	# initiate random seed
	np.random.seed(0)

	# reshape
	img = np.reshape(img, (H * W, -1))

	# select one index randomly
	i = np.random.choice(np.arange(H * W), Class, replace=False)
	Cs = img[i].copy()


	clss = np.zeros((H * W), dtype=int); print(clss.shape)

	# each pixel
	for i in range(H * W):
		# get distance from base pixel
		dis = np.sqrt(np.sum((Cs - img[i]) ** 2, axis=1))
		#print(Cs[i])
		clss[i] = np.argmin(dis) #clss[i] = Cs[i]     # clss[i] = ; print(clss.shape)

	#print("clss:", clss)

	out = np.reshape(clss, (H, W)) * 50; print(out) ; out = out.astype(np.uint8)

	return imgo

"""
def base64toimage(base64_string):
    sbuf = StringIO()
    sbuf.write(base64.b64decode(io.BytesIO(base64_string)))
    pimg = Image.open(sbuf)
    return cv2.cvtColor(np.array(pimg), cv2.COLOR_RGB2BGR)
"""
def base64toimage(img_base64):  # 入力
    """
    img = base64.b64decode(img_base64)  #; jpg=np.frombuffer(img,dtype=np.uint8)#; img = cv2.imdecode(jpg, cv2.IMREAD_COLOR)
    img_binarystream = io.BytesIO(img)
    img_pil = Image.open(img_binarystream)
    img_numpy = np.asarray(img_pil)
    img = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2BGR)
    print(img)
    """
    img = base64.b64decode(img_base64)
    #print("base64", img)
    img_binarystream = io.BytesIO(img)
    #with open("./img.png", 'wb') as f:
    #    f.write(img)
    #PILイメージ <- バイナリーストリーム
    #print(type(img_binarystream))
    img_pil = Image.open(img_binarystream)
    img_numpy = np.asarray(img_pil)

    #numpy配列(BGR) <- numpy配列(RGBA)
    img = cv2.cvtColor(img_numpy, cv2.COLOR_RGBA2BGR)

    return img

def imagetobase64(img):         # 出力
    """
    with open("sample.png", "wb") as f:
        f.write(img)
    with open("sample.png", "rb") as f:
        encode = base64.b64encode(f.read())
    """
    #buffered = BytesIO()
    #img.save(img, format="JPEG")
    #img_str = base64.b64encode(buffered.getvalue())
    img = base64.b64encode(img)
    return img  # base64

def main(img_base64):
    img = base64toimage(img_base64)
    #print(len(img))

    ## 画像処理 ##
    imgo = np.copy(img) # もともとのカラー画像
    img = img.astype(np.float32)

    kernel = np.ones((3,3),np.uint8)
    #kernel2 = np.ones((5,5),np.uint8)
    #kernel3 = np.ones((3,3),np.uint8)
    #kernel4 = np.ones((1,1),np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel2)
    #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel3)
    #img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel4)

    # K-means
    out = k_means_step1(img, 5, imgo)

    return imagetobase64(out)