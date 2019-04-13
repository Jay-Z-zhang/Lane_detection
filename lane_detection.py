# -*- coding:utf-8 -*-


from selenium import webdriver
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 高斯滤波核大小
blur_ksize = 5

# Canny边缘检测高低阈值
canny_lth = 51.5
canny_hth = 150

# 霍夫变换参数
rho = 1  # rho的步长，即直线到图像原点(0,0)点的距离
theta = np.pi / 180   # theta的范围
threshold = 20    # 累加器中的值高于它时才认为是一条直线
min_line_len = 40     # 线的最短长度，比这个短的都被忽略
max_line_gap = 20    # 两条直线之间的最大间隔，小于此值，认为是一条直线

def information(image) :
    print(image.shape)
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    print("width: %s,height: %s,channels: %s"%(width,height,channels))

def process_an_image(img):
    cv2.destroyAllWindows()


    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # RGB转换为HSV
    cv2.imshow("hsv", hsv)
    # 1. 灰度化、滤波和Canny
    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # cv2.imshow("gray", gray)#转化为灰度图像

    # plt.hist(gray.ravel(), 256, (0, 256))
    # plt.show()

    # dst = cv2.equalizeHist(gray)
    # cv2.imshow("dst",dst)#直方图整体均衡化

    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(3, 3))
    # dst = clahe.apply(hsv) # 直方图局部自适应均衡化
    # cv2.imshow("dst", dst)

    # plt.hist(dst.ravel(), 256, (0, 256))
    # plt.show()

    # blur_gray = cv2.GaussianBlur(hsv, (blur_ksize, blur_ksize), 1) # 高斯滤波
    # blur_gray = cv2.blur(hsv,(5,5))#均值模糊
    blur_gray = cv2.medianBlur(hsv,5)#中值滤波

    cv2.imshow("blur_gray", blur_gray)


    """
    kernel1 = np.array([[0, 1, 2], [-1, 0, 1], [-2, -1, 0]])  #自定义算子
    a = cv2.filter2D(blur_gray, cv2.CV_32F, kernel=kernel1)
    kernel2 = np.array([[-2, -1, 0], [-1, 0, 1], [0, 1, 2]])
    b = cv2.filter2D(blur_gray, cv2.CV_32F, kernel=kernel2)  # 图像卷积运算
    mid = cv2.addWeighted(a, 0.5, b, 0.5, 0)
    edges = cv2.convertScaleAbs(mid)  # 变成8位单通道255结果，
    cv2.imshow("custom",edges)
    """
    # lap = cv2.Laplacian(blur_gray, cv2.CV_32F)  # laplace算子
    # edges = cv2.convertScaleAbs(lap)

    # grad_x = cv2.Sobel(blur_gray, cv2.CV_32F, 1, 0)  # sobel算子
    # grad_y = cv2.Sobel(blur_gray, cv2.CV_32F, 0, 1)
    # gradx = cv2.convertScaleAbs(grad_x)
    # grady = cv2.convertScaleAbs(grad_y)
    # edges = cv2.addWeighted(gradx, 0.5, grady, 0.5, 0)


    edges = cv2.Canny(blur_gray, canny_lth, canny_hth)  # canny边缘检测
    cv2.imshow("edges", edges)

    # 2. 标记四个坐标点用于ROI截取
    rows, cols = edges.shape#图像的高与宽
    points = np.array([[(50, 500), (480, 325), (520, 325), (960, 500)]])
    # [[[0 540], [460 325], [520 325], [960 540]]]
    roi_edges = roi_mask(edges, points)
    cv2.imshow("roi_edges",roi_edges)

    # 3. 霍夫直线提取
    drawing, lines = hough_lines(roi_edges, rho, theta,threshold, min_line_len, max_line_gap)

    # 4. 车道拟合计算
    draw_lanes(drawing, lines)


    # 5. 最终将结果合在原图上
    result = cv2.addWeighted(img, 0.9, drawing, 0.2, 0)
    return result


def roi_mask(img, corner_points):

    mask = np.zeros_like(img)# 创建掩膜
    cv2.fillPoly(mask, corner_points, 255)

    masked_img = cv2.bitwise_and(img, mask)
    return masked_img


def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    # 统计概率霍夫直线变换
    lines = cv2.HoughLinesP(img, rho, theta, threshold,
                            minLineLength=min_line_len, maxLineGap=max_line_gap)
    #rho: 线段以像素为单位的距离精度，double类型的，推荐用1.0 ；theta： 线段以弧度为单位的角度精度，推荐用numpy.pi/180；
    # 新建一副空白画布

    drawing = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    #画出直线检测结果
    draw_lines(drawing, lines)

    return drawing, lines


def draw_lines(img, lines, color=[0, 0, 255], thickness=1):#画线
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)


def draw_lanes(img, lines, color=[255, 0, 0], thickness=8):#画车道线
    # a. 划分左右车道列表
    left_lines, right_lines = [], []
    #直线
    if (len(left_lines) >= 0 or len(right_lines) >= 0):
        for line in lines:
            for x1, y1, x2, y2 in line:
                k = (y2 - y1) / (x2 - x1)
                if k < -0.5:#根据斜率划分左右线段
                    left_lines.append(line)
                elif k > 0.5:
                    right_lines.append(line)

    # #左转
    # elif (len(left_lines) <= 0 and len(right_lines) >= 0):
    #     for line in lines:
    #         left_lines.append(line)
    #         right_lines.append(line)
    #
    # #右转
    # elif (len(left_lines) <= 0 and len(right_lines) <= 0):
    #     for line in lines:
    #         left_lines.append(line)
    #         right_lines.append(line)
    else:
        return

    # b. 清理异常数据
    clean_lines(left_lines,0.1)
    clean_lines(right_lines, 0.1)

    # c. 得到左右车道线点的集合，拟合直线
    left_points = [(x1, y1) for line in left_lines for x1, y1, x2, y2 in line]
    # 提取左侧直线族中的所有的第一个点
    left_points = left_points + [(x2, y2)for line in left_lines for x1, y1, x2, y2 in line]
    # 提取左侧直线族中的所有的第二个点
    right_points = [(x1, y1)for line in right_lines for x1, y1, x2, y2 in line]
    # 提取右侧直线族中的所有的第一个点
    right_points = right_points + [(x2, y2) for line in right_lines for x1, y1, x2, y2 in line]
    # 提取右侧侧直线族中的所有的第二个点
    left_results = least_squares_fit(left_points, 325, img.shape[0])
    # 拟合点集，生成直线表达式，并计算左侧直线在图像中的两个端点的坐标
    right_results = least_squares_fit(right_points, 325, img.shape[0])
    # 拟合点集，生成直线表达式，并计算右侧直线在图像中的两个端点的坐标,325是y最小,最大是图像y

    # 注意这里点的顺序
    #vtxs = np.array([[left_results[1], left_results[0], right_results[0], right_results[1]]])

    # d.填充车道区域
    #cv2.fillPoly(img, vtxs, (0, 255, 0))

     #或者只画车道线
    cv2.line(img, left_results[0], left_results[1], [0, 255, 0], 8)
    cv2.line(img, right_results[0], right_results[1], [0, 255, 0], 8)

    kl =(left_results[1][1] - left_results[0][1])/(left_results[1][0] - left_results[0][0])
    kr =(right_results[1][1] - right_results[0][1])/(right_results[1][0] - right_results[0][0])
    c = kr/(kr - kl)
    print(c)

def clean_lines(lines, threshold):
    # 迭代计算斜率均值，排除掉与差值差异较大的数据
    slope = [(y2 - y1) / (x2 - x1)
             for line in lines for x1, y1, x2, y2 in line]
    while len(lines) > 0: # 计算斜率平均值
        mean = np.mean(slope) # 计算斜率平均值,因为后面会将直线和斜率值弹出
        diff = [abs(s - mean) for s in slope]#计算每条直线斜率与平均值的差值
        idx = np.argmax(diff)#找出差值最大的下标
        if diff[idx] > threshold:# 如果下标对应的差值大于threshold就删去该直线
            slope.pop(idx)#弹出斜率
            lines.pop(idx)#弹出直线
        else:
            break

#拟合点集，生成直线表达式，并计算直线在图像中的两个端点的坐标
def least_squares_fit(point_list, ymin, ymax):
    # 最小二乘法拟合
    x = [p[0] for p in point_list]#提取x
    y = [p[1] for p in point_list]#提取y

    fit = np.polyfit(y, x, 1)
    fit_fn = np.poly1d(fit)

    xmin = int(fit_fn(ymin))#计算这条直线在图像中最左侧的横坐标
    xmax = int(fit_fn(ymax))#计算这条直线在图像中最右侧的横坐标

    print(xmin,ymin,xmax,ymax)
    return [(xmin, ymin), (xmax, ymax)]

if __name__ == "__main__":
    img = cv2.imread('way4.jpg')
    img = cv2.resize(img, (960, 540), 0, 0, cv2.INTER_LINEAR)
    information(img)
    result = process_an_image(img)
    cv2.imshow("result", result)
    cv2.waitKey(0)