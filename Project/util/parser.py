def parse_model_configuration(path):
    """
    Function:
        Lấy tệp cấu hình
    
    
    Arguments:
        path -- Đường dẫn tới file cấu hình
        
    Returns:
        blocks -- một danh sách các khối. Mỗi khối mô tả một khối trong mạng nơ-ron sẽ được xây dựng. 
        Khối được biểu diễn dưới dạng dictionary trong danh sách
    """
    
    # lưu nội dung của tệp cfg trong một danh sách các chuỗi
    with open(path) as cfg:
        lines = cfg.read()
    
    lines = lines.split("\n") # lưu trữ các dòng trong một danh sách
    lines = [line for line in lines if not line.startswith("#") and line] # Loại bỏ comments
    lines = [line.strip() for line in lines] # loại bỏ khoảng trắng ở rìa
    
    blocks = []
    
    # lặp qua danh sách kết quả để nhận các khối
    for line in lines:
        if line.startswith("["): # đánh dấu sự bắt đầu của một khối mới
            blocks.append({}) # thêm nó vào danh sách khối
            blocks[-1]["type"] = line[1:-1].strip() # Xóa khoảng trắng ở đầu và cuối chuỗi
            if blocks[-1]["type"] == "convolutional": 
                blocks[-1]["batch_normalize"] = 0
            
        else:
            key, value = line.split("=")
            key, value = key.strip(), value.strip()
            blocks[-1][key] = value
    
    return blocks

def load_classes(path):
    """
    Function:
        Đọc các nhãn tại 'path'
    
    Arguments:
        path -- đường dẫn tới tệp nhãn
        
    Returns:
        name -- 1 danh sách nhãn
    """
  
    fp = open(path, "r")
    names = fp.read().split("\n")[:-1]
    return names
