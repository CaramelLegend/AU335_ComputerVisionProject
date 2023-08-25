import os
import cv2
import glob
import easyocr
import numpy as np

path = "C:\\Users\\KenYuen\\OneDrive - sjtu.edu.cn\\Desktop\\UNI\\SEM 6 Spring Sem '22\\Computer Vision\\Assignment\\Project\\easy\\Resize"
images = glob.glob("C:\\Users\\KenYuen\\OneDrive - sjtu.edu.cn\\Desktop\\UNI\\SEM 6 Spring Sem '22\\Computer Vision\\Assignment\\Project\\easy\\*.jpg")
images_resize = glob.glob("C:\\Users\\KenYuen\\OneDrive - sjtu.edu.cn\\Desktop\\UNI\\SEM 6 Spring Sem '22\\Computer Vision\\Assignment\\Project\\easy\\Resize\\*.jpg")

# 计算各背景颜色的面积
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

img_index = 0
for filename in images:
    img = cv2.imread(filename)

    # 调整图片大小
    resize_h = 500
    height = img.shape[0]
    width = img.shape[1]
    scale = float(resize_h)/height
    img_resize = cv2.resize(img, (int(scale*width), resize_h))
    cv2.imwrite(os.path.join(path, "Img_Resize_" + str(img_index) + ".jpg"), img_resize)
    img_index += 1

img_index = 0
for filename in images_resize:
    if not os.path.exists(os.path.join(path, str(img_index))): # 创立新文件夹
        os.mkdir(os.path.join(path,str(img_index)))
    img = cv2.imread(filename)
    img_letter_detected = img.copy()
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(os.path.join(path, str(img_index), "Img_Gray.jpg"), img_gray)
    kernel_x = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1)) # x轴核
    kernel_y = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 50)) # y轴核
    colour = "None" # 记录照片背景颜色
    # cv2.imshow("gray", img_thres_gray)

    area_blue, area_cyan = cal_bgd_color_area(img) 
    
    # 判断车牌背景颜色
    if area_blue > area_cyan:
        # print("Blue CarPlate")
        colour = "blue"
        ret_gray, img_thres_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY) # 二值化
        img_closing = cv2.morphologyEx(img_thres_gray, cv2.MORPH_CLOSE, kernel_x) # x轴闭运算
        img_closing = cv2.morphologyEx(img_closing, cv2.MORPH_CLOSE, kernel_y) # y轴闭运算
        cv2.imwrite(os.path.join(path, str(img_index), "Img_Closing.jpg"), img_closing)
    else:
        # print("Green CarPlate")
        colour = "cyan"
        ret_gray, img_thres_gray = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY) # 二值化
        img_thres_gray = 255 - img_thres_gray # 像素值反转
        img_closing = cv2.morphologyEx(img_thres_gray, cv2.MORPH_CLOSE, kernel_x) # x轴闭运算
        img_closing = cv2.morphologyEx(img_closing, cv2.MORPH_CLOSE, kernel_y) # y轴闭运算
        cv2.imwrite(os.path.join(path, str(img_index), "Img_Closing.jpg"), img_closing)
        
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
        if (letter[3] > (letter[2] * 1)) and (letter[3] < (letter[2] * 3)) and (letter[2] > 10):
            splited_image = img_closing[letter[1]:letter[1] + letter[3], letter[0]:letter[0] + letter[2]]
            if (splited_image.shape[0] > 250) & (splited_image.shape[1] > 50):
                cv2.rectangle(img_letter_detected, (letter[0],letter[1]), (letter[0] + letter[2], letter[1] + letter[3]),(0,0,255), 5) # 绘出字符边框
                cropped_img = img[letter[1]:letter[1]+letter[3],letter[0]:letter[0]+letter[2]]
                cropped_imgs.append(cropped_img)
                # cv2.imwrite(os.path.join(path, str(img_index), "Img_Cropped.jpg"), cropped_img)
                # cv2.imshow("cropped", cropped_img)
                # cv2.waitKey()

    """
    for i in range(0,len(letters_img)):
        cv2.imshow("splited letters", letters_img[i])
        cv2.waitKey()
    """

    # cv2.imshow("detected",img_letter_detected)
    # cv2.waitKey()
    
    thres_cropped_imgs = [] # 单个字符的二值化照片
    for i in range(len(cropped_imgs)):
        img_gray = cv2.cvtColor(cropped_imgs[i], cv2.COLOR_BGR2GRAY)
        if(colour == "blue"):
            ret_gray, img_thres_gray = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY) # 二值化
            img_thres_gray = 255 - img_thres_gray # 像素值反转
        else:
            ret_gray, img_thres_gray = cv2.threshold(img_gray, 100, 255, cv2.THRESH_BINARY) # 二值化
            # img_thres_gray = 255 - img_thres_gray # 像素值反转
        img_thres_gray = np.pad(img_thres_gray, (70,70), constant_values = 255)

        # 调整图片大小
        # resize_h = 500
        height = img_thres_gray.shape[0]
        width = img_thres_gray.shape[1]
        # scale = float(resize_h)/height
        img_thres_gray = cv2.resize(img_thres_gray, (int(width*0.4), int(height*0.4)))

        thres_cropped_imgs.append(img_thres_gray)
        # cv2.imshow("gray", thres_cropped_imgs[i])
        cv2.imwrite(os.path.join(path, str(img_index), "char" + str(i+1) + ".jpg"), thres_cropped_imgs[i])
        # cv2.waitKey()
    
    ocr_result = "" # 字符识别结果
    correct_result = 0 # 正确的字符识别数量
    true_ocr_result = ["沪EWM957", "沪AF02976", "鲁NBK268"]

    chineseChar = ["川", "鄂", "赣", "甘", "贵", "桂", "黑", "沪", "冀", "津", "京", "吉", "辽", "鲁", "蒙",
    "闽", "宁", "青", "琼", "陕", "苏", "晋", "皖", "湘", "新", "豫", "渝", "粤", "云", "藏", "浙"]
    char = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
    'J', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    reader_ch_sim = easyocr.Reader(['ch_sim'])
    reader_en = easyocr.Reader(['en'])
    result = reader_ch_sim.readtext(thres_cropped_imgs[0], allowlist = chineseChar)
    ocr_result = ocr_result + str(result[0][1])
    for i in range(len(thres_cropped_imgs)-1):
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
