import time
from collections import Counter
import pickle as pkl
import torch 
from torch.autograd import Variable
import cv2 
from util.parser import load_classes
from util.model import Darknet
from util.image_processor import prep_image
from util.utils import non_max_suppression

confidence = 0.75
nms_thesh = 0.7
start = 0
CUDA = torch.cuda.is_available()
classes = load_classes("data/parking.names")

# print("Đang load cấu hình và trọng số mô hình.....")
model = Darknet("config/yolov3.cfg")
model.load_weights("weights/yolov3.weights")
# print("mô hình load thành công!")

num_classes = model.num_classes
inp_dim = 320
assert inp_dim % 32 == 0 
assert inp_dim > 32

if CUDA:
    model.cuda()

# Đặt mô hình ở chế độ evaluation, 
model.eval()
obj_counter = {}

def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results
    cls = int(x[-1])
    label = "{0}".format(classes[cls]) ## label = "{0}: {1}".format(classes[cls],str(obj_counter[cls]))
    color = colors[cls%100]
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, "Empty: {0}".format(str(obj_counter[0])), (5,30), cv2.FONT_HERSHEY_DUPLEX, 1, [96,0,128], 2)
    cv2.putText(img, "Occupied: {0}".format(str(obj_counter[1])), (5,60), cv2.FONT_HERSHEY_DUPLEX, 1, [96,0,128], 2)
    cv2.putText(img, "Total: {0}".format(str(output.size(0))), (5,90), cv2.FONT_HERSHEY_DUPLEX, 1, [96,0,128], 2)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img


cap = cv2.VideoCapture(0)

# Đặt độ phân giải của frame, nhận các độ phân giải Mặc định
# chuyển đổi từ float sang int
frame_width = int(cap.get(3))
frame_height = int(cap.get(4))

frames = 0  
start = time.time()

while True:
    ret, frame = cap.read()
    
    if ret:   
        img = prep_image(frame, inp_dim)
        im_dim = frame.shape[1], frame.shape[0]
        im_dim = torch.FloatTensor(im_dim).repeat(1,2)   
                     
        if CUDA:
            im_dim = im_dim.cuda()
            img = img.cuda()

        with torch.no_grad():
            output = model(Variable(img))
        output = non_max_suppression(output, confidence, num_classes, nms_conf = nms_thesh)


        if type(output) == int:
            frames += 1
            print("FPS of the video is {:5.4f}".format( frames / (time.time() - start)))
            cv2.imshow("frame", frame)
            key = cv2.waitKey(1)
            continue
        
        
        obj_counter = output[:,-1]
        obj_counter = Counter(obj_counter.cpu().numpy().astype(int).tolist())


        im_dim = im_dim.repeat(output.size(0), 1)
        scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
        
        output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
        output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
        
        output[:,1:5] /= scaling_factor

        for i in range(output.shape[0]):
            output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
            output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
        
        classes = load_classes('data/parking.names')
        colors = pkl.load(open("colors/pallete", "rb"))
        list(map(lambda x: write(x, frame), output))
        
        cv2.imshow("frame", frame)

        key = cv2.waitKey(1)
        if not cap.isOpened() or not cap.isOpened():
            break
        frames += 1
        print(time.time() - start)
        print("FPS of the video is {:5.2f}".format( frames / (time.time() - start)))
    else:
        break     
