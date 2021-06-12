import numpy as np
import torch

def bbox_iou(box1, box2, x1y1x2y2=True):
    """
    Function:
        Tính toán intersection over union giữa hai hộp
        
    Arguments:
       box1 -- Hộp thứ nhất
       box2 -- Hộp thứ 2
       x1y1x2y2 -- bool value
    Return:
        iou --  IoU của 2 bounding boxe
    """
    if not x1y1x2y2:
        # Chuyển đổi từ tâm và chiều rộng sang tọa độ chính xác
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2
    else:
        # Lấy tọa độ của các bounding boxe
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # lấy tọa độ của phần giao nhau giữa 2 bounding boxe
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Diện tích phần giao nhau
    if torch.cuda.is_available():
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0).cuda() * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0).cuda()
    else:
        inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
            inter_rect_y2 - inter_rect_y1 + 1, min=0)
    # Diện tích phần hội giữa 2 bounding boxe
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

# Vì có thể có nhiều phát hiện đúng của cùng một lớp, 
# unique để đưa các lớp hiện tại trong bất kỳ hình ảnh nhất định nào
def unique(tensor):
    """
    Function:
        Nhận các lớp khác nhau được phát hiện trong hình ảnh
    
    Arguments:
        tensor -- torch tensor
    
    Return:
        tensor_res -- torch tensor sau khi được chuẩn bị
    """
    tensor_np = tensor.cpu().detach().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def non_max_suppression(prediction, confidence, num_classes, nms_conf):
    """
    Function:
        Loại bỏ các phát hiện có điểm tin cậy đối tượng object confidence score thấp hơn 'conf_thres' 
        và thực hiện  Non-Maximum Suppression để lọc thêm các phát hiện
        
    Arguments:
        prediction -- tensor dự đoán của yolo
        confidence -- giá trị float để xóa tất cả dự đoán có giá trị tin cậy thấp hơn độ tin cậy
        num_classes -- số lớp (nhãn)
        nms_conf -- giá trị float (non max suppression) để xóa bbox, nó lớn hơn nms_conf
    
    Return:
        output -- tuple (x1, y1, x2, y2, object_conf, class_score, class_pred) 
    """
    # Đối với mỗi hộp giới hạn có điểm đối tượng objectness score dưới ngưỡng threshold,
    # đặt các giá trị của mọi thuộc tính của nó thành 0.
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    
    # chúng tôi biến đổi các thuộc tính (tâm x, tâm y, chiều cao, chiều rộng) của các hộp thành
    # (góc trên cùng bên trái x, góc trên cùng bên trái y, góc dưới cùng bên phải x, góc dưới cùng bên phải y)
    box_corner = prediction.new(prediction.shape)
    box_corner[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_corner[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_corner[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_corner[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_corner[:,:,:4]
    
    batch_size = prediction.size(0)

    write = False # cho biết tensor đã được khởi tạo hay chưa
    


    for ind in range(batch_size):
        # chọn hình ảnh từ batch
        image_pred = prediction[ind]          #Tensor ảnh
       # ngưỡng tin cậy - confidence threshholding 
       #NMS
    
        # Nhận lớp có điểm tối đa và index của lớp đó
        # Loại bỏ điểm số softmax của num_classes
        # Thêm index lớp và điểm lớp của lớp confidence class có điểm tối đa
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        # Loại bỏ các mục có thuộc tính bằng không
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        try:
            image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        except:
            continue
        
        if image_pred_.shape[0] == 0:
            continue       
#        
  
        # Nhận các lớp khác nhau được phát hiện trong hình ảnh
        img_classes = unique(image_pred_[:,-1])  # -1 index, giữ class index
        
        
        for cls in img_classes:
            # thực hiện NMS

        
            # nhận được phát hiện với một lớp cụ thể
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)
            
            # sắp xếp các phát hiện sao cho mục có objectness confidence tối đa ở trên cùng
            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)   # Số lượng phát hiện, số hàng trong image_pred_class
            
            for i in range(idx):
                # Nhận IOU của tất cả các hộp đứng sau hộp đang xem xét trong vòng lặp
                try:
                    ious = bbox_iou(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:])
                # cắt image_pred_class [i + 1:] có thể trả về một tensor trống,
                # chỉ định nó kích hoạt ValueError
                except ValueError:
                    break
                # nếu một giá trị bị xóa khỏi image_pred_class, chúng ta không thể có các lần lặp idx. 
                # Do đó, chúng tôi có thể cố gắng lập chỉ mục một giá trị nằm ngoài giới hạn (IndexError)
                except IndexError:
                    break
            
                # Loại bỏ tất cả các thám tử có IoU > ngưỡng
                iou_mask = (ious < nms_conf).float().unsqueeze(1)
                image_pred_class[i+1:] *= iou_mask       
            
                # Xóa các mục non-zero
                non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
                
            # Kết nối batch_id của hình ảnh với các phát hiện, điều này giúp xác định
            # hình ảnh nào mà phát hiện tương ứng.
            # Sử dụng cấu trúc tuyến tính linear straucture để giữ TẤT CẢ các phát hiện 
            # từ batch_dim được làm phẳng batch được xác định bằng cột batch bổ sung
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    try:
        return output
        # tensor shape D x 8
        # D:  index của hình ảnh trong batch, tọa độ 4 góc, điểm đối tượng objectness score, 
        # điểm của lớp có độ tin cậy tối đa và index của lớp đó.
    except:
        return 0 # không có một phát hiện nào trong bất kỳ hình ảnh nào

