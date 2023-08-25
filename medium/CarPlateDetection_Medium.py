import os
import cv2
import glob
import easyocr
import numpy as np

path = "C:\\Users\\KenYuen\\OneDrive - sjtu.edu.cn\\Desktop\\UNI\\SEM 6 Spring Sem '22\\Computer Vision\\Assignment\\Project\\medium\\Resize"
images = glob.glob("C:\\Users\\KenYuen\\OneDrive - sjtu.edu.cn\\Desktop\\UNI\\SEM 6 Spring Sem '22\\Computer Vision\\Assignment\\Project\\medium\\*.jpg")
images_resize = glob.glob("C:\\Users\\KenYuen\\OneDrive - sjtu.edu.cn\\Desktop\\UNI\\SEM 6 Spring Sem '22\\Computer Vision\\Assignment\\Project\\medium\\Resize\\*.jpg")
img_index = 0 # 照片顺序

def cal_bgd_color_area(i): 
    img_hsv = cv2.cvtColor(i, cv2.COLOR_BGR2HSV)

    #蓝色的HSV值
    lower_blue = np.array([100, 43, 46])
    upper_blue = np.array([124, 255, 255])
    mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
    ret_blue, binary_blue = cv2.threshold(mask_blue, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binary_blue', binary_blue)
    contours, hierarchy = cv2.findContours(binary_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_blue = 0
    for c in contours:
        area_blue += cv2.contourArea(c) # 计算蓝色背景的面积
    
    #青色的HSV值
    lower_cyan = np.array([78, 43, 46])
    upper_cyan = np.array([99, 255, 255])
    mask_cyan = cv2.inRange(img_hsv, lower_cyan, upper_cyan)
    ret_cyan, binary_cyan = cv2.threshold(mask_cyan, 127, 255, cv2.THRESH_BINARY)
    # cv2.imshow('binary_cyan', binary_cyan)
    contours, hierarchy = cv2.findContours(binary_cyan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_cyan = 0
    for c in contours:
        area_cyan += cv2.contourArea(c) # 计算青色背景的面积

    return area_blue, area_cyan

for filename in images:
    img = cv2.imread(filename)

    # 调整图片大小
    resize_h = 1000
    height = img.shape[0]
    width = img.shape[1]
    scale = float(resize_h)/height
    img_resize = cv2.resize(img, (int(scale*width), resize_h))
    cv2.imwrite(os.path.join(path, "Img_Resize_" + str(img_index) + ".jpg"), img_resize)
    img_index += 1

img_index = 0 # 照片顺序
for filename in images_resize:
    img = cv2.imread(filename)
    img_contours = img.copy()
    if not os.path.exists(os.path.join(path, str(img_index))): # 创立新文件夹
        os.mkdir(os.path.join(path,str(img_index)))
    
    #图片预处理
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gray, (11, 11), 0, 0, cv2.BORDER_DEFAULT) # 降低噪声
    cv2.imwrite(os.path.join(path, str(img_index), "Img_Gaussian.jpg"), gaussian)
    kernel = np.ones((50, 100), np.uint8)
    imgOpen = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)                           
    # cv2.imshow("imgOpen", imgOpen)
    cv2.imwrite(os.path.join(path, str(img_index), "Img_Open.jpg"), imgOpen)
                                                                                    
    imgOpenWeight = cv2.addWeighted(gray, 1, imgOpen, -1, 0)                          
    # cv2.imshow("imgOpenWeight", imgOpenWeight)
    cv2.imwrite(os.path.join(path, str(img_index), "Img_OpenWeight.jpg"), imgOpenWeight)                                  
                                                                                    
    ret, imgBin = cv2.threshold(imgOpenWeight, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # cv2.imshow("imgBin", imgBin)                                                         
    cv2.imwrite(os.path.join(path, str(img_index), "Img_Bin.jpg"), imgBin)
                                                                                    
    imgEdge = cv2.Canny(imgBin, 100, 300)
    cv2.imwrite(os.path.join(path, str(img_index), "Img_Canny.jpg"), imgEdge)                                             
    # cv2.imshow("imgEdge", imgEdge)                                                       
                                                                                    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (50, 15)) # 核
    kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (65, 1)) # x轴核
    imgEdge = cv2.morphologyEx(imgEdge, cv2.MORPH_CLOSE, kernel)
    imgEdge = cv2.morphologyEx(imgEdge, cv2.MORPH_OPEN, kernel)
    imgEdge = cv2.morphologyEx(imgEdge, cv2.MORPH_CLOSE, kernel_x)                                      
    cv2.imwrite(os.path.join(path, str(img_index), "Img_Structure.jpg"), imgEdge)

    contours, hierarchy = cv2.findContours(imgEdge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE) 
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20000]  #对轮廓进行面积的筛选  

    imgDark = np.zeros(img.shape, dtype = img.dtype)
    # print(img.shape)
    cropped_bbox = []
    dimensions = []

    for index, contour in enumerate(contours):                                                 
        rect = cv2.minAreaRect(contour) # [中心(x,y), (宽,高), 旋转角度]                                
        w, h = rect[1]
        angle = rect[2]
                                                                                
        if w < h:                                                                              
            w, h = h, w                                                                        
        scale = w/h                                                                            
        if scale > 3 and scale < 6 and ((angle >= 85.0 and angle <= 90.0) or (angle >= 0 and angle <= 5)): # 对矩形进行高宽比和角度的筛选                                                                                                                   
            # CarPlateList.append(rect)                                                          
            # cv2.drawContours(imgDark, contours, index, (255, 255, 255), 2, 8) # 轮廓绘制                        
            
            box = cv2.boxPoints(rect)  # Vertices Coordinates                                        
            box = np.int0(box)
            cv2.drawContours(imgDark, [box], 0, (0, 0, 255), 2) # 矩形绘制  
            for x in range(4): # 修正矩形坐标
                for y in range(2):
                    if(box[x][y] < 0):
                        box[x][y] = 0

            c_img = img_contours[box[0][1]:box[2][1],box[0][0]:box[2][0]] # 裁剪矩形范围内的图像
            if (len(c_img) == 0):
                c_img = img_contours[box[1][1]:box[3][1],box[1][0]:box[3][0]]
                height = box[3][1] - box[1][1]
                width = box[3][0] - box[1][0]
            else:
                height = box[2][1] - box[0][1]
                width = box[2][0] - box[0][0]
            dimensions.append([width,height])
            # cv2.imshow("cropped",c_img)
            # cv2.waitKey()
            cropped_bbox.append(c_img)                                              
                                 
    # cv2.imwrite(os.path.join(path, str(i), "Img_Contour.jpg"), imgDark)
    # cv2.waitKey()

    CarPlateList = [] # 记录最终车牌照片
    CarPlateColor = "" # 记录最终车牌的背景颜色
    scale = [] # 记录每个照片的比例
    bgd_color = [] # 记录每个照片的背景颜色
    max_val_scale = 0 # 颜色占整个招牌的最大比例
    location = 0

    for index in range(len(cropped_bbox)):
        blue, cyan = cal_bgd_color_area(cropped_bbox[index])
        width = dimensions[index][0]
        height = dimensions[index][1]
        area = abs(width)*abs(height)
        blue_scale = float(blue)/float(area)
        cyan_scale = float(cyan)/float(area)
        if blue_scale > cyan_scale:
            bgd_color.append("blue")
        else:
            bgd_color.append("cyan")
        scale_max = max(blue_scale, cyan_scale)
        scale.append(scale_max)
    
    for i in range(len(scale)): # 选取蓝色或绿色占比最大的照片
        if(scale[i] > max_val_scale): 
            max_val_scale = scale[i]
            location = i
    
    CarPlateList.append(cropped_bbox[location])
    CarPlateColor = bgd_color[location]
    # cv2.imshow("CarPlate", CarPlateList[0])
    # cv2.waitKey()

    if(CarPlateColor == "blue"):
        img_hsv = cv2.cvtColor(CarPlateList[0], cv2.COLOR_BGR2HSV)
        lower_blue = np.array([100, 43, 46])
        upper_blue = np.array([124, 255, 255])
        mask_blue = cv2.inRange(img_hsv, lower_blue, upper_blue)
        ret_blue, binary_blue = cv2.threshold(mask_blue, 127, 255, cv2.THRESH_BINARY)
        # cv2.imshow('binary_blue', binary_blue)
        contours, hierarchy = cv2.findContours(binary_blue, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    elif(CarPlateColor == "cyan"):
        img_hsv = cv2.cvtColor(CarPlateList[0], cv2.COLOR_BGR2HSV)
        lower_cyan = np.array([78, 43, 46])
        upper_cyan = np.array([99, 255, 255])
        mask_cyan = cv2.inRange(img_hsv, lower_cyan, upper_cyan)
        ret_cyan, binary_cyan = cv2.threshold(mask_cyan, 127, 255, cv2.THRESH_BINARY)
        # cv2.imshow('binary_cyan', binary_cyan)
        contours, hierarchy = cv2.findContours(binary_cyan, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > 20000]

    imgDark = np.zeros(CarPlateList[0].shape, dtype = img.dtype)
    FinalCarPlate = []
    for index, contour in enumerate(contours): 
        rect = cv2.minAreaRect(contour)
        w, h = rect[1]
        angle = rect[2]
        if w < h:                                                                              
            w, h = h, w  
        if w == 0 or h == 0:
            continue                                                     
        scale = w/h
        area = abs(float(w))*abs(float(h))
        if scale > 3 and scale < 6 and ((angle >= 85.0 and angle <= 90.0) or (angle >= 0 and angle <= 5)) and area > 100:
            box = cv2.boxPoints(rect)  # Vertices Coordinates                                        
            box = np.int0(box)
            cv2.drawContours(imgDark, [box], 0, (0, 0, 255), 2) # 矩形绘制
            # cv2.imshow("Final", imgDark)
            for i in range(4): # 修正矩形坐标
                for j in range(2):
                    if(box[i][j] < 0):
                        box[i][j] = 0

            fcp = CarPlateList[0][box[0][1]:box[2][1],box[0][0]:box[2][0]] # 裁剪矩形范围内的图像
            if (len(fcp) == 0):
                fcp = CarPlateList[0][box[1][1]:box[3][1],box[1][0]:box[3][0]]
            # cv2.imshow("FinalCarPlate",fcp)
            cv2.imwrite(os.path.join(path, str(img_index), "Final_Car_Plate.jpg"), fcp)
            # cv2.waitKey()
            FinalCarPlate = fcp
            # print(fcp.shape)
            cv2.waitKey()
    
    # 调整车牌图片大小
    resize_w = 500 # 500
    height = FinalCarPlate.shape[0]
    width = FinalCarPlate.shape[1]
    scale = float(resize_w)/width
    img_resize = cv2.resize(FinalCarPlate, (resize_w, int(scale*height)))
    # cv2.imshow("Resize",img_resize)
    # cv2.waitKey()
    # cv2.imwrite(os.path.join(path, "Img_Resize_" + str(i) + ".jpg"), img_resize)
            
    img_letter_detected = img_resize.copy()
    img_gray = cv2.cvtColor(img_resize, cv2.COLOR_BGR2GRAY)
    kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 1)) # x轴核 (10, 1)
    kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5)) # y轴核 (1, 5)
    # colour = "None" # 记录照片背景颜色
    # cv2.imshow("gray", img_thres_gray)
    
    # 判断车牌背景颜色
    if (CarPlateColor == "blue"):
        # print("Blue CarPlate")
        ret_gray, img_thres_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY) # 二值化
        img_closing = cv2.morphologyEx(img_thres_gray, cv2.MORPH_CLOSE, kernel_x) # x轴闭运算
        # cv2.imshow("close1", img_closing)
        img_closing = cv2.morphologyEx(img_closing, cv2.MORPH_OPEN, kernel_y) # y轴开运算
        # cv2.imshow("close2", img_closing)
        img_closing = cv2.morphologyEx(img_closing, cv2.MORPH_CLOSE, kernel_y) # y轴闭运算
        # cv2.imshow("close3", img_closing)
        # cv2.waitKey()
    elif(CarPlateColor == "cyan"):
        # print("Green CarPlate")
        ret_gray, img_thres_gray = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY) # 二值化
        img_thres_gray = 255 - img_thres_gray # 像素值反转
        img_closing = cv2.morphologyEx(img_thres_gray, cv2.MORPH_CLOSE, kernel_x) # x轴闭运算
        # cv2.imshow("close1", img_closing)
        img_closing = cv2.morphologyEx(img_closing, cv2.MORPH_OPEN, kernel_y) # y轴开运算
        # cv2.imshow("close2", img_closing)
        img_closing = cv2.morphologyEx(img_closing, cv2.MORPH_CLOSE, kernel_y) # y轴闭运算
        # cv2.imshow("close3", img_closing)
        # cv2.waitKey()
        
    # cv2.imshow("closed",img_closing)
    cnts, hier = cv2.findContours(img_closing, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    letters = [] # 所有字符的边框信息
    letters_img = [] # 字符分割的结果图片
    for c in cnts:
        letter = []
        rect = cv2.boundingRect(c) # 用矩形边框绘出轮廓
        letter.append(rect[0]) # 矩形边框左上角的x轴坐标
        letter.append(rect[1]) # 矩形边框左上角的y轴坐标
        letter.append(rect[2]) # 矩形边框的宽度
        letter.append(rect[3]) # 矩形边框的高度
        letters.append(letter) # 矩形边框的所有信息

    letters = sorted(letters, key = lambda s :s[0])
    cropped_imgs = [] # 单个字符的照片
    for letter in letters:
        # print(i, letter)
        if (letter[3] > (letter[2] * 1)) and (letter[3] < (letter[2] * 4)) and (letter[2] > 10): # (letter[3] < (letter[2] * 3))
            splited_image = img_closing[letter[1]:letter[1] + letter[3], letter[0]:letter[0] + letter[2]]
            # cv2.imshow("splited", splited_image)
            # cv2.waitKey()
            if (splited_image.shape[0] > 80) & (splited_image.shape[1] > 15):
                cv2.rectangle(img_letter_detected, (letter[0],letter[1]), (letter[0] + letter[2], letter[1] + letter[3]),(0,0,255), 1) # 绘出字符边框
                # cv2.imshow("detected", img_letter_detected)
                # cv2.waitKey()
                cropped_img = img_letter_detected[letter[1]:letter[1]+letter[3],letter[0]:letter[0]+letter[2]]
                cropped_imgs.append(cropped_img)
                # cv2.imshow("cropped", cropped_img)
                #cv2.waitKey()
    
    thres_cropped_imgs = [] # 单个字符的二值化照片
    for i in range(len(cropped_imgs)):
        img_gray = cv2.cvtColor(cropped_imgs[i], cv2.COLOR_BGR2GRAY)
        if(CarPlateColor == "blue"):
            ret_gray, img_thres_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY) # 二值化
            """
            img_thres_gray = cv2.morphologyEx(img_thres_gray, cv2.MORPH_OPEN, (20,20)) # y轴开运算
            img_thres_gray = cv2.dilate(img_thres_gray,(1,10))
            cv2.imshow("gray", img_thres_gray)
            cv2.waitKey()
            """
            img_thres_gray = 255 - img_thres_gray # 像素值反转
        else:
            ret_gray, img_thres_gray = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY) # 二值化
            # img_thres_gray = 255 - img_thres_gray # 像素值反转
        img_thres_gray = np.pad(img_thres_gray, (40,40), constant_values = 255)
        thres_cropped_imgs.append(img_thres_gray)
        cv2.imwrite(os.path.join(path, str(img_index), "char" + str(i+1) + ".jpg"), thres_cropped_imgs[i])
        # cv2.imshow("gray", thres_cropped_imgs[i])
        # cv2.waitKey()
    
    ocr_result = ""
    correct_result = 0
    true_ocr_result = ["沪EWM957", "豫B20E68", "沪A93S20"]
    
    chineseChar = ["川", "鄂", "赣", "甘", "贵", "桂", "黑", "沪", "冀", "津", "京", "吉", "辽", "鲁", "蒙", "闽", "宁", "青", "琼", "陕", "苏", "晋", "皖", "湘", "新", "豫", "渝", "粤", "云", "藏", "浙"]
    char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    reader_ch_sim = easyocr.Reader(['ch_sim'])
    reader_en = easyocr.Reader(['en'])
    result = reader_ch_sim.readtext(thres_cropped_imgs[0], allowlist = chineseChar)
    ocr_result = ocr_result + str(result[0][1])
    for i in range(len(thres_cropped_imgs) - 1):
        result = reader_en.readtext(thres_cropped_imgs[i+1], allowlist = char)
        ocr_result = ocr_result + str(result[0][1])
    print(ocr_result)
    total = len(ocr_result)

    for l in range(len(ocr_result)):
        if(ocr_result[l] == true_ocr_result[img_index][l]):
            correct_result += 1
    accuracy = float(correct_result)/float(total)
    print(accuracy)
    correct_result = 0
    img_index += 1