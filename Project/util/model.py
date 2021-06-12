from collections import defaultdict

import numpy as np
import torch
from torch import nn

from .parser import parse_model_configuration
from .moduler import modules_creator


class Darknet(nn.Module):
    """Mô hình nhận diện của Yolov3"""

    def __init__(self, config_path, img_size=320):
        """
        Function:
            Hàm tạo cho lớp Darknet
    
        Arguments:
            config_path -- Đường dẫn file config
            img_size -- Kích thước của ảnh đầu vào
        """
        super(Darknet, self).__init__()
        self.blocks = parse_model_configuration(config_path)
        self.hyperparams, self.module_list ,self.num_classes = modules_creator(self.blocks)
        self.img_size = img_size
        self.seen = 0
        self.header_info = np.array([0, 0, 0, self.seen, 0])

    
    # Chức năng hàm forward: Tính toán kết quả đầu ra,
    # biến đổi các đầu ra phát hiện của feature maps theo cách có thể được xử lý dễ dàng hơn
    def forward(self, x):
        """
        Function:
            Truyền tải nguồn cấp dữ liệu cho dự đoán
        
        Arguments:
            x -- input tensor 
            
        Returns:
            output -- tensor đầu ra của mô hình
        """
        
        output = []
        self.losses = defaultdict(float)
        layer_outputs = []
        for i, (block, module) in enumerate(zip(self.blocks, self.module_list)): 
            # chạy đầu vào qua từng module để có được đầu ra
            if block["type"] in ["convolutional", "upsample", "maxpool"]:
                x = module(x)
            elif block["type"] == "route":
                layer_i = [int(x) for x in block["layers"].split(",")]
                x = torch.cat([layer_outputs[i] for i in layer_i], 1)
            elif block["type"] == "shortcut":
                layer_i = int(block["from"])
                x = layer_outputs[-1] + layer_outputs[layer_i]
            elif block["type"] == "yolo":
                x = module(x)
                output.append(x)
            layer_outputs.append(x)
        return torch.cat(output, 1)

    def load_weights(self, weights_path):        
        """
        Function:
            Phân tích cú pháp và tải các trọng số
            
        Arguments:
            weights_path -- Đường dẫn file weights
        """
        
        fp = open(weights_path, "rb")
        
        # Lấy 4 giá trị header đầu tiên
        # 1. Major version number - Số phiên bản chính
        # 2. Minor Version Number - Số phiên bản nhỏ
        # 3. Subversion number - số phiên bản được xem lại
        # 4. Số ảnh đã được xem
        header = np.fromfile(fp, dtype=np.int32, count=5)

        self.seen = header[3]
        weights = np.fromfile(fp, dtype=np.float32)  # Phần còn lại là những trọng số
        fp.close()

        ptr = 0 # theo dõi vị trí trong mảng trọng số
        # lặp qua tệp trọng số và tải trọng số vào các mô-đun của mạng
        for i, (block, module) in enumerate(zip(self.blocks, self.module_list)):
            if block["type"] == "convolutional":
                # Nếu module tích chập load trọng số
                # Nếu không thì bỏ qua
                conv_layer = module[0]
                try:
                    block["batch_normalize"]
                except:
                    block["batch_normalize"] = 0
                if block["batch_normalize"]:
                    # Load BN bias - độ lệch, weights - trọng số,
                    # running mean - trung vị and running variance - phương sai
                    # Truyền trọng số đã load vào trọng số của mô hình
                    # cop dữ liệu vào mô hình
                    bn_layer = module[1]
                    num_b = bn_layer.bias.numel()  # số bias
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.bias)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.weight)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                # Nếu batch_normalize flase, chỉ cần tải các bias ​​của lớp chập.
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    # resize lại trọng số đã load theo kích thước của trọng số mô hình
                    conv_b = torch.from_numpy(weights[ptr : ptr + num_b]).view_as(conv_layer.bias)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel() # load trọng số cho các lớp tích chập
                conv_w = torch.from_numpy(weights[ptr : ptr + num_w]).view_as(conv_layer.weight)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
