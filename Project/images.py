import argparse
import os 
import os.path as osp
import pickle as pkl
import pandas as pd
import time
from collections import Counter

import torch 
from torch.autograd import Variable
import cv2 

from util.parser import load_classes
from util.model import Darknet
from util.image_processor import prep_image
from util.utils import non_max_suppression


def arg_parse():
    """
    Phân tích cú pháp đối số địa chỉ truyền vào
    In ra các vị trí với empty là trống, occupied là đã đỗ, total là tổng cộng các vị trí hiện có

    """
    parser = argparse.ArgumentParser(description='Module nhận diện qua hình ảnh')
    parser.add_argument("--images", dest = 'images', help = "Ảnh/thư mục cần nhận diện", default = "imgs", type = str)    
    return parser.parse_args()


# Khởi tạo các siêu tham số
args = arg_parse()
images = args.images
outputs = "outputs"
batch_size = 1
confidence = 0.8
nms_thesh = 0.7
start = 0
CUDA = torch.cuda.is_available()

classes = load_classes("data/parking.names")

# Cài đặt thông số mạng neurol
print("Đang load cấu hình và trọng số mô hình.....")
model = Darknet("config/yolov3.cfg")
model.load_weights("weights/yolov3.weights")
print("Mô hình load thành công!")

inp_dim = 320
# Kiểm tra xem kích thước ảnh đầu vào có chia hết cho 32 không
assert inp_dim % 32 == 0 
assert inp_dim > 32

num_classes = model.num_classes


# Nếu có GPU, đặt mô hình vào GPU xử lý
if CUDA:
    model.cuda()

# Đặt mô hình ở chế độ đánh giá, kết hợp với no_grad() 
# giảm bớt tính toán gradient trong quá trình training
# từ đó tiết kiệm dung lượng bộ nhớ ram gpu
model.eval()

read_dir = time.time()

try:
    imlist = [osp.join(osp.realpath('.'), images, img) for img in os.listdir(images)]
except NotADirectoryError:
    imlist = []
    imlist.append(osp.join(osp.realpath('.'), images))
except FileNotFoundError:
    print ("Địa chỉ hoặc tên ảnh/thư mục sai!")
    exit()
    
if not os.path.exists(outputs):
    os.makedirs(outputs)

load_batch = time.time()
loaded_ims = [cv2.imread(x) for x in imlist]

# Biến pytorch cho danh sách hình ảnh
im_batches = list(map(prep_image, loaded_ims, [inp_dim for x in range(len(imlist))]))

# Danh sách chứa kích thước của hình ảnh gốc
im_dim_list = [(x.shape[1], x.shape[0]) for x in loaded_ims]
im_dim_list = torch.FloatTensor(im_dim_list).repeat(1,2)

# tạo các Batchs
leftover = 0
if (len(im_dim_list) % batch_size):
    leftover = 1

if batch_size != 1:
    num_batches = len(imlist) // batch_size + leftover            
    im_batches = [torch.cat((im_batches[i*batch_size : min((i +  1)*batch_size,
                        len(im_batches))]))  for i in range(num_batches)]  

write = 0


if CUDA:
    im_dim_list = im_dim_list.cuda()
    
start_outputs_loop = time.time()
for i, batch in enumerate(im_batches):
# Tải ảnh
    start = time.time()
    if CUDA:
        batch = batch.cuda()
    # áp dụng các offsets để dự đoán kết quả
    # Biến đổi các dự đoán theo mô tả trong paper gốc
    # B x ( tọa độ bbox x số anchors) x grid_w x grid_h --> B x bbox x (tất cả các hộp)
    # Đặt mọi ô được đề xuất thành một hàng

    with torch.no_grad():
        prediction = model(Variable(batch))
    
    # lấy các hộp với độ tin cậy của đối tượng object confidence > threshold
    # Chuyển đổi những tọa độ thành tọa độ tuyệt đối
    # thực hiện NMS trên các hộp này và lưu kết quả
    prediction = non_max_suppression(prediction, confidence, num_classes, nms_conf = nms_thesh)
    end = time.time()

    # Nếu đầu ra của hàm non_max_suppression là int(0), thì có nghĩa là không có phát hiện nào
    # thoát khỏi vòng lặp và bỏ qua các bước còn lại
    if type(prediction) == int:
        for im_num, image in enumerate(imlist[i*batch_size: min((i +  1)*batch_size, len(imlist))]):
            im_id = i*batch_size + im_num
            print("----------------------------------------------------------------------")
            print("{0:1s} được nhận diện trong{1:6.3f} giây".format(image.split("/")[-1], (end - start)/batch_size))
            print("----------------------------------------------------------------------")
        continue

    prediction[:,0] += i*batch_size  # chuyển đổi thuộc tính index trong batch thành index trong imlist

    if not write:  # Nếu chưa khởi tạo đầu ra
        output = prediction  
        write = 1
    else:
        output = torch.cat((output,prediction))

    for im_num, image in enumerate(imlist[i*batch_size: min((i + 1)*batch_size, len(imlist))]):
        im_id = i*batch_size + im_num
        objs = [classes[int(x[-1])] for x in output if int(x[0]) == im_id]
        print("{0:1s} được nhận diện trong{1:6.3f} giây".format(image.split("/")[-1], (end - start)/batch_size))
        freq = {}
        for items in objs:
            freq[items] = objs.count(items)
      
        for key, value in freq.items():
            print ("%s: %d"%(key, value))
        print('Tổng các vị trí:', len(objs))
        print("----------------------------------------------------------------------")

    if CUDA:
        torch.cuda.synchronize()   # đảm bảo rằng kernel CUDA được đồng bộ hóa với CPU

# kiểm tra xem có phát hiện nào hay không, nếu không, thoát khỏi chương trình
try:
    output
except NameError:
    print ("Không có phát hiện nào")
    exit()

# biến đổi tọa độ của các hộp được đo đối với ranh giới của khu vực padding 
# trên hình ảnh có chứa hình ảnh gốc
im_dim_list = torch.index_select(im_dim_list, 0, output[:,0].long())

scaling_factor = torch.min(inp_dim/im_dim_list,1)[0].view(-1,1)

output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim_list[:,0].view(-1,1))/2
output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim_list[:,1].view(-1,1))/2

output[:,1:5] /= scaling_factor # để lấy tọa độ của hộp giới hạn trên hình ảnh gốc

# vẽ các hộp vượt ngoài biên ảnh vào các cạnh
for i in range(output.shape[0]):
    output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim_list[i,0])
    output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim_list[i,1])
    
    
output_recast = time.time()
class_load = time.time()
colors = pkl.load(open("colors/pallete", "rb")) # pickled file

draw = time.time()

clss = {}

for i in output:
    if int(i[0]) not in clss:
        clss[int(i[0])] = []
    clss[int(i[0])].append(int(i[-1]))

for key, value in clss.items():
    clss[key] = Counter(value)


def write(x, results):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = results[int(x[0])] 
    # if int(x[0]) not in clss:
    #     clss[int(x[0])] = []
    cls = int(x[-1])
    color = colors[cls%100]
    label = "{0}".format(classes[cls]) # label = "{0}: {1}".format(classes[cls],str(clss[int(x[0])][cls]))
    cv2.rectangle(img, c1, c2,color, 1)
    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = c1[0] + t_size[0] + 3, c1[1] + t_size[1] + 4
    cv2.rectangle(img, c1, c2,color, -1)
    cv2.putText(img, "Empty: {0}".format(str(clss[int(x[0])][0])), (5,30), cv2.FONT_HERSHEY_DUPLEX, 1, [96,0,128], 2)
    cv2.putText(img, "Occupied: {0}".format(str(clss[int(x[0])][1])), (5,60), cv2.FONT_HERSHEY_DUPLEX, 1, [96,0,128], 2)
    cv2.putText(img, "Total: {0}".format(str(len(objs))), (5,90), cv2.FONT_HERSHEY_DUPLEX, 1, [96,0,128], 2)
    cv2.putText(img, label, (c1[0], c1[1] + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [225,255,255], 1)
    return img


# vẽ bb trên các ảnh
list(map(lambda x: write(x, loaded_ims), output))

#  lưu các phát hiện
outputs_names = pd.Series(imlist).apply(lambda x: "{}/{}".format(outputs,x.split("/")[-1]))

# ghi các hình ảnh nhận diện được vào địa chỉ trong outputs_names
list(map(cv2.imwrite, outputs_names, loaded_ims))

end = time.time()

    
print("Tổng Kết")
print("----------------------------------------------------------------------")
print("{:35s}: {}".format("Công việc", "Thời gian xử lý (tính bằng giây)"))
print()
print("{:35s}: {:2.3f}".format("Đọc địa chỉ", load_batch - read_dir))
print("{:35s}: {:2.3f}".format("Loading batch", start_outputs_loop - load_batch))
print("{:35s}: {:2.3f}".format("Nhận diện (" + str(len(imlist)) +  " ảnh)", output_recast - start_outputs_loop))
print("{:35s}: {:2.3f}".format("Xử lý đầu ra", class_load - output_recast))
print("{:35s}: {:2.3f}".format("Vẽ hộp giới hạn", end - draw))
print("{:35s}: {:2.3f}".format("Thời gian trung bình mỗi ảnh", (end - load_batch)/len(imlist)))
print("----------------------------------------------------------------------")


torch.cuda.empty_cache()
