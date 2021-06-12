import torch 
import torch.nn as nn


class EmptyLayer(nn.Module):
    """Placeholder cho lớp 'route' và 'shortcut'"""

    def __init__(self):            
        """
        Function:
            Hàm tạo cho lớp EmptyLayer
        """
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    """Lớp phát hiện"""

    def __init__(self, anchors, num_classes, img_dim):
        """
        Function:
            Hàm tạo cho lớp DetectionLayer
            
        Arguments:
            anchors -- danh sách các kích thước hộp neo
            num_classes -- số lớp mô hình cần phân loại
            img_dim -- kích thước ảnh đầu vào
        """
        super(DetectionLayer, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.bbox_attrs = 5 + num_classes
        self.image_dim = img_dim
        self.ignore_thres = 0.5
        self.lambda_coord = 1


    def forward(self, x):
        """
        Function:
            Truyền tải nguồn cấp dữ liệu cho dự đoán
        
        Arguments:
            x -- input tensor 
            
        Returns:
            output -- tensor đầu ra của mô hình
        """
        
        nA = self.num_anchors
        nB = x.size(0)
        nG = x.size(2)
        stride = self.image_dim / nG

        # Tensors để hỗ trợ cuda
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor
        LongTensor = torch.cuda.LongTensor if x.is_cuda else torch.LongTensor
        ByteTensor = torch.cuda.ByteTensor if x.is_cuda else torch.ByteTensor

        prediction = x.view(nB, nA, self.bbox_attrs, nG, nG).permute(0, 1, 3, 4, 2).contiguous()

        # Nhận kết quả đầu ra
        x = torch.sigmoid(prediction[..., 0])  # Tọa độ x
        y = torch.sigmoid(prediction[..., 1])  # Tọa độ y
        w = prediction[..., 2]  # Chiều rộng
        h = prediction[..., 3]  # Chiều cao
        pred_conf = torch.sigmoid(prediction[..., 4])  # Độ chắc chắn
        pred_cls = torch.sigmoid(prediction[..., 5:])  # Điểm số lớp

        # Tính toán offset cho mỗi lưới
        grid_x = torch.arange(nG).repeat(nG, 1).view([1, 1, nG, nG]).type(FloatTensor)
        grid_y = torch.arange(nG).repeat(nG, 1).t().view([1, 1, nG, nG]).type(FloatTensor)
        scaled_anchors = FloatTensor([(a_w / stride, a_h / stride) for a_w, a_h in self.anchors])
        anchor_w = scaled_anchors[:, 0:1].view((1, nA, 1, 1))
        anchor_h = scaled_anchors[:, 1:2].view((1, nA, 1, 1))

        # Thêm offset và tỷ lệ bằng neo
        pred_boxes = FloatTensor(prediction[..., :4].shape)
        pred_boxes[..., 0] = x.data + grid_x
        pred_boxes[..., 1] = y.data + grid_y
        pred_boxes[..., 2] = torch.exp(w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(h.data) * anchor_h

        output = torch.cat(
            (
                pred_boxes.view(nB, -1, 4) * stride,
                pred_conf.view(nB, -1, 1),
                pred_cls.view(nB, -1, self.num_classes),
            ),
            -1,
        )
        return output
        

def modules_creator(blocks):
    """
    Function:
        Hàm tạo danh sách module từ trình phân tích cú pháp từ tệp cấu hình
    
    Arguments:
        blocks -- dictionary của mỗi khối chứa thông tin của nó
    
    Returns:
        hyperparams -- dictionary chứa thông tin về mô hình
        modules_list -- danh sách module pytorch
    """
    hyperparams = blocks.pop(0)
    output_filters = [int(hyperparams["channels"])]
    modules_list = nn.ModuleList()
    # lặp lại danh sách các khối và tạo mô-đun PyTorch cho mỗi khối
    for i, block in enumerate(blocks):
        modules = nn.Sequential()

        # 1. kiểm tra loại khối
        # 2. tạo một mô-đun mới cho khối
        # 3. thêm vào module_list

        if block["type"] == "convolutional": # Nhận thông tin về lớp
            try:
                bn = int(block["batch_normalize"])
            except:
                bn = 0
            filters = int(block["filters"])
            kernel_size = int(block["size"])
            pad = (kernel_size - 1) // 2 if int(block["pad"]) else 0

            # Thêm lớp tích chập
            modules.add_module(
                "conv_%d" % i,
                nn.Conv2d(
                    in_channels=output_filters[-1],
                    out_channels=filters,
                    kernel_size=kernel_size,
                    stride=int(block["stride"]),
                    padding=pad,
                    bias=not bn,
                ),
            )

            # Thêm lớp Batch Norm
            if bn:
                modules.add_module("batch_norm_%d" % i, nn.BatchNorm2d(filters))

            # Kiểm tra hàm kích hoạt
            if block["activation"] == "leaky":
                modules.add_module("leaky_%d" % i, nn.LeakyReLU(0.1))

        elif block["type"] == "maxpool":
            kernel_size = int(block["size"])
            stride = int(block["stride"])
            if kernel_size == 2 and stride == 1:
                padding = nn.ZeroPad2d((0, 1, 0, 1))
                modules.add_module("_debug_padding_%d" % i, padding)
            maxpool = nn.MaxPool2d(
                kernel_size=int(block["size"]),
                stride=int(block["stride"]),
                padding=int((kernel_size - 1) // 2),
            )
            modules.add_module("maxpool_%d" % i, maxpool)

        elif block["type"] == "upsample":
            upsample = nn.Upsample(scale_factor=int(block["stride"]), mode="nearest")
            modules.add_module("upsample_%d" % i, upsample)

        elif block["type"] == "route":
            layers = [int(x) for x in block["layers"].split(",")]
            filters = sum([output_filters[layer_i] for layer_i in layers])
            modules.add_module("route_%d" % i, EmptyLayer())

        # emptylayer(), thực hiện hoạt động đơn giản, tránh việc thiết kế từng lớp route/shortcut,
        # xây dựng các đối tượng nn.Module, nhận các đối số truyền vào và lớp forward 
        # nhưng do mã nối khá đơn giản và ngắn (torch.cat), nếu xây dựng các lớp trên
        # sẽ dẫn đến sự trừu tượng không cần thiết mà chỉ làm tăng boiler plate code
        # Thay vào đó, đặt một lớp giả thay cho lớp route được đề xuất, 
        # và sau đó thực hiện nối trực tiếp trong hàm forward của đối tượng nn.Module đại diện cho darknet.

        elif block["type"] == "shortcut":
            filters = output_filters[int(block["from"])]
            modules.add_module("shortcut_%d" % i, EmptyLayer())

        elif block["type"] == "yolo":
            anchor_idxs = [int(x) for x in block["mask"].split(",")]
            # trích xuất các anchor
            anchors = [int(x) for x in block["anchors"].split(",")]
            anchors = [(anchors[i], anchors[i + 1]) for i in range(0, len(anchors), 2)]
            anchors = [anchors[i] for i in anchor_idxs]
            num_classes = int(block["classes"])
            img_height = int(hyperparams["height"])
            # Xác định lớp phát hiện
            yolo_layer = DetectionLayer(anchors, num_classes, img_height)
            modules.add_module("yolo_%d" % i, yolo_layer)
        # ghi danh sách mô-đun và số lượng filter đầu ra
        modules_list.append(modules)
        output_filters.append(filters)

    return hyperparams, modules_list, num_classes

