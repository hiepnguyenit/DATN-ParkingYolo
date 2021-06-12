import numpy as np
import cv2
import torch

def letterbox_image(img, inp_dim):
    """
    Function:
        Thay đổi kích thước hình ảnh với tỷ lệ khung hình không thay đổi
        bằng cách sử dụng padding cho các vùng còn lại bằng màu (128,128,128)
        
    Arguments:
        img -- Ảnh đầu vào
        inp_dim -- Kích thước để thay đổi kích thước hình ảnh (kích thước đầu vào)
    
    Return:
        canvas -- Ảnh được đổi kích thước   
    """

    img_w, img_h = img.shape[1], img.shape[0]
    w, h = inp_dim
    new_w = int(img_w * min(w/img_w, h/img_h))
    new_h = int(img_h * min(w/img_w, h/img_h))
    resized_image = cv2.resize(img, (new_w,new_h), interpolation = cv2.INTER_CUBIC)
    
    canvas = np.full((inp_dim[1], inp_dim[0], 3), 128)

    canvas[(h-new_h)//2:(h-new_h)//2 + new_h,(w-new_w)//2:(w-new_w)//2 + new_w,  :] = resized_image
    
    return canvas



def prep_image(img, inp_dim):
    """
    Function:
        Chuẩn bị hình ảnh để nhập vào mạng nơ-ron
        
    Arguments:
        img -- Ảnh
        inp_dim -- Kích thước để thay đổi kích thước hình ảnh (kích thước đầu vào)
    
    Return:
        img -- Ảnh sau khi chuẩn bị
    """
    
    img = (letterbox_image(img, (inp_dim, inp_dim))) # Thay đổi kích thước ảnh đầu vào
    img = img[:,:,::-1].transpose((2,0,1)).copy()  # BGR -> RGB | H X W C -> C X H X W 
    img = torch.from_numpy(img).float().div(255.0).unsqueeze(0)  #Convert thành float
    return img
