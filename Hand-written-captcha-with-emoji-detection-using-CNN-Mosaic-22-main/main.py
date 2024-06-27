import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow.keras.models import load_model

def preprocess2(mask):
    p,q = mask.shape
    for i in range(p):
      for j in range(q):
        if mask[i][j]>165.0:
          mask[i][j]=255.0
        # else:
        #   mask[i][j]=255.0 
        mask[i][j]=255-mask[i][j]
    return mask

def add_border(imgg1):
  colour = [0,0,0]
  constant= cv2.copyMakeBorder(imgg1,50,50,50,50,cv2.BORDER_CONSTANT,value=colour)
  return constant

def segment(img):
  kernel = np.ones((3,3))
  gray_image=img
  lst=[]
  for j in range(gray_image.shape[1]):
    sum = 0
    for i in range(gray_image.shape[0]):
      sum = sum+gray_image[i][j]
    #print(sum)
    lst.append(sum)
  # print(lst)
  ptt_x = []
  f = 0
  #print(len(lst))
  for i in range(len(lst)):
    if lst[i]>1200 and f==0:
      x1=i
      #print(i)
      f=1
    elif lst[i] ==0 and f==1:
      x2=i
      if(abs(x1-x2)>30):
        ptt_x.append((min(x1,x2),max(x1,x2)))
      f=0 
  ptt_y=[]
  for x1,x2 in ptt_x:
    lst2=[]
    for i in range(gray_image.shape[0]):
      sum=0
      for j in range(min(x1,x2),max(x1,x2)+1):
        sum = sum+gray_image[i][j]
      lst2.append(sum)

    f=0
    for i in range(len(lst2)):
      if lst2[i] >1200 and f==0:
        y1=i
        f=1
      if lst2[i] ==0 and f==1:
        y2=i
        if(abs(y2-y1)>30):
          ptt_y.append((min(y1,y2),max(y1,y2)))
          break
        f=0 
  return gray_image, ptt_x, ptt_y

def func(test_img):
  map_p={
    1:'A',
    2:'B',
    3:'C',
    4:'D',
    5:'E',
    6:'F',
    7:'G',
    8:'H',
    9:'I',
    10:'J',
    11:'K',
    12:'L',
    13:'M',
    14:'N',
    15:'O',
    16:'P',
    17:'Q',
    18:'R',
    19:'S',
    20:'T',
    21:'U',
    22:'V',
    23:'W',
    24:'X',
    25:'Y',
    26:'Z',
    27:'1',
    28:'2',
    29:'3',
    30:'4',
    31:'5',
    32:'6',
    33:'7'
  }
  model = load_model('model_for_alpha_fin.h5')
  model_em = load_model('model_for_emo_fin.h5')
  test_img = np.array(test_img)
  test_img = cv2.cvtColor(test_img,cv2.COLOR_BGR2GRAY)
  test_img1 = preprocess2(test_img)
  gr_im1,pt_x,pt_y=segment(test_img1)
  str=""
  for i in range(len(pt_x)):
    x1,x2 = pt_x[i]
    y1,y2 = pt_y[i]
    pr_img = test_img1[y1:y2,x1:x2]
    pr_img = add_border(pr_img)
    pr_img = cv2.resize(pr_img,(28,28),interpolation= cv2.INTER_AREA)
    #print(pr_img)
    if i==1:
      plt.imshow(pr_img,cmap='gray')
    pr_img = np.resize(pr_img,(1,28,28))
    #print(pr_img.shape)
    ans1 = model.predict(pr_img)
    ans2= model_em.predict(pr_img)
    max1 = np.max(ans1)
    max2 = np.max(ans2)
    pos1 = np.argmax(ans1,axis=1)
    pos2 = np.argmax(ans2,axis=1)
    #print(pos1)
    ans=-1
    if max1>0.90:
      ans=pos1[0]
    if ans==-1:
      if max1>max2:
        ans = pos1[0]
      else:
        ans = pos2[0]+26
    str=str+map_p[ans+1]
  print(str)


if __name__=="__main__":
    print("Enter Image path")

    while True:
        path = input()
        if not os.path.exists(path):
            print("Enter Correct Path")
            continue
        img = cv2.imread(path)
        func(img)
        print("Enter next path for testing")
  
