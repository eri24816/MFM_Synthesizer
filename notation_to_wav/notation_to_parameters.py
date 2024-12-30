import cv2
import numpy as np
import matplotlib.pyplot as plt

plot = False

# note_range = np.array([67, 79])
# note_range = np.array([-0.5,  0.5])
note_range = np.array([-1,  1])
freq_range = 440 * np.power(2, (note_range-69)/12)
max_slice_height = 200
thres = 240

def myfindContour(gray_image):
    # Step 1: 判斷有效區域
    _, binary_image = cv2.threshold(gray_image, thres, 255, cv2.THRESH_BINARY_INV)

    # 平滑處理強化輪廓判定
    # Step 1: 進行模糊處理
    blurred_image = cv2.GaussianBlur(binary_image, (7, 7), 0)

    # Step 2: 影像膨脹，使區域更大
    kernel = np.ones((5, 5), np.uint8)
    dilated_image = cv2.dilate(blurred_image, kernel, iterations=1)

    # Step 3: 閉運算
    closed_image = cv2.morphologyEx(dilated_image, cv2.MORPH_CLOSE, kernel)

    # Step 4: 進行輪廓檢測
    contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def column_bound(column, slice_width, y, height):
    # edges = cv2.Canny(column, 20, 30)
    edges = cv2.Laplacian(column, cv2.CV_64F)
    edges = cv2.convertScaleAbs(edges)
    edge_positions = np.where(edges > 0)
    top_edge_positions = np.zeros(shape=(slice_width,))
    bottom_edge_positions = np.ones(shape=(slice_width,)) * height
    for i in range(len(edge_positions[1])):
        top_edge_positions[edge_positions[1][i]] = \
        max(edge_positions[0][i], top_edge_positions[edge_positions[1][i]])
        bottom_edge_positions[edge_positions[1][i]] = \
        min(edge_positions[0][i], bottom_edge_positions[edge_positions[1][i]])
        
    top_edge_positions[top_edge_positions == 0] = np.max(top_edge_positions)
    bottom_edge_positions[bottom_edge_positions == height] = np.min(bottom_edge_positions)
    return y + top_edge_positions, y + bottom_edge_positions

def notation_to_parameters(file):
    # 加載影像
    image = cv2.imread(file)
    # image = cv2.resize(image, (1600, 200))
    image = cv2.resize(image, (1600, 100))
    max_slice_height = 100
    border_size = round(image.shape[0] / 2)
    
    image = cv2.copyMakeBorder(
    image,
    top=border_size,
    bottom=border_size,
    left=0,
    right=0,
    borderType=cv2.BORDER_CONSTANT,
    value=(255, 255, 255)  # 填充為白色
    )
    
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 轉換為 HSV 色彩空間
    height = image.shape[0]
    width = image.shape[1]
    
    
    contours = myfindContour(gray_image)
    
    # 畫出輪廓
    # output = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image_copy = image.copy()
    cv2.drawContours(image_copy, contours, -1, (0, 255, 0), 2)
    

    step = 10
    sw = step * 2 # slice width

    # 初始化存放結果的列表
    left_to_right_note = []
    left_to_right_intensity = []
    left_to_right_density = []
    left_to_right_hue = []
    left_to_right_saturation = []
    left_to_right_value = []
    # left_to_right_jitter = []
    # left_to_right_warmth = []
    effective_x = []
    

    # Step 2: 對每個筆觸進行處理
    for contour in contours:
        # 計算包含該輪廓的邊界框
        x, y, w, h = cv2.boundingRect(contour)
        
        # 初始化單一筆觸的強度和抖動序列
        intensity_trend = []
        note_trend = []
        density_trend = []
        hue_trend = []
        value_trend = []
        saturation_trend = []
        current_angle = []
        # warmth_trend = []
        # jitter_trend = []
        effective_x = [range(x, x + w - sw, step)] + effective_x
        
        # Step 3: 從左至右分割區域，進行逐列計算
        for i in effective_x[0]:  # 從左到右，逐列分析
            # 提取單列的ROI（感興趣區域）
            slice_width = min(sw, x + w - i)
            column = gray_image[y:y + h, i:i + slice_width]
            
            # initial boundary for slice and pitch
            top_edge_positions, bottom_edge_positions = column_bound(column, slice_width, y, height)
            mid_positions = (top_edge_positions + bottom_edge_positions) / 2
            slice_height = min(max_slice_height, round(np.max(top_edge_positions) - np.min(bottom_edge_positions)))
            
            x_center = (i + slice_width / 2)
            y_center = np.mean(mid_positions)
            angle = np.arctan2(1, np.mean(np.diff(mid_positions)))
            current_angle.append(angle)
            rotation_matrix = cv2.getRotationMatrix2D((x_center, y_center), -np.degrees(angle)+90, 1.0)
            rotated_image = cv2.warpAffine(hsv_image, rotation_matrix, (width, height))
            column_hsv = cv2.getRectSubPix(rotated_image, (slice_width, slice_height), (x_center, y_center))
            column_BRG = cv2.cvtColor(column_hsv, cv2.COLOR_HSV2BGR)
            column_gray = cv2.cvtColor(column_BRG, cv2.COLOR_BGR2GRAY)
            
            # rotated boundary
            top_edge_positions, bottom_edge_positions = column_bound(column_gray, slice_width, 0, slice_height)
                
            # intensity
            intensity = np.mean(top_edge_positions - bottom_edge_positions)
            intensity_trend.append(intensity / max_slice_height)
            
            # pitch
            y_avg = y_center
            note_avg = (1 - y_avg / height) * (note_range[1] - note_range[0]) + note_range[0]
            note_trend.append(np.mean(note_avg))
            
            # extract all pixels
            pixels_gray = []
            pixels_hsv = []
            awared_pos = []
            all_pos = []
            for j in range(slice_width):
                for k in range(int(bottom_edge_positions[j]), int(top_edge_positions[j]) ):
                    if column_gray[k, j] <= 223:
                        pixels_gray.append(column_gray[k, j])
                        pixels_hsv.append(column_hsv[k, j, :])
                        awared_pos.append((k, j))
                    all_pos.append((k, j))
            pixels_gray = np.array(pixels_gray)
            pixels_hsv = np.array(pixels_hsv)
            awared_pos = np.array(awared_pos)
            all_pos = np.array(all_pos)
            # print(pixels_hsv.shape)
            
            # 計算濃密度（深色像素比例
            
            # hsv attribute
            if awared_pos.shape[0] > 0:
                density = awared_pos.shape[0] / all_pos.shape[0]
                mean_hue = np.mean(pixels_hsv[:, 0])
                mean_saturation = np.mean(pixels_hsv[:, 1])
                mean_value = np.mean(pixels_hsv[:, 2])
            else:
                # density = density_trend[-1]
                # mean_hue = hue_trend[-1]
                # mean_saturation = saturation_trend[-1]
                # mean_value = value_trend[-1]
                
                density = 0
                mean_hue = 0
                mean_saturation = 0
                mean_value = 0
            
            density_trend.append(density)
            hue_trend.append(mean_hue)
            saturation_trend.append(mean_saturation / 256)
            value_trend.append(mean_value / 256)
            
            # jitter
            # edge_variance = np.std(top_edge_positions) + np.std(bottom_edge_positions)  # 計算邊緣位置的標準差
            # jitter_trend.append(edge_variance)
            
            # 計算冷暖程度（色相的平均值）
            # value_channel = column_hsv[:, :, 2] / 255.0  # 明度 (0-1)
            # mean_value = np.mean(value_channel)
            # if mean_hue >= 90:
            #     mean_hue -= 180
            # warmth = 1 - np.abs(mean_hue) / 90 - 0.5
            # warmth_adjusted = warmth * (1 - abs(mean_value - 0.5) * 2) + 0.5 * abs(mean_value - 0.5) * 2
            # warmth_adjusted = warmth * mean_value * 2
            # warmth_trend.append(warmth_adjusted)
            
        # plt.plot(current_angle)
        # plt.show()
        
        
        # 將每個筆觸的結果加入到整體結果中
        left_to_right_intensity = [intensity_trend] + left_to_right_intensity
        left_to_right_note = [note_trend] + left_to_right_note
        left_to_right_density = [density_trend] + left_to_right_density
        left_to_right_hue = [hue_trend] + left_to_right_hue
        left_to_right_saturation = [saturation_trend] + left_to_right_saturation
        left_to_right_value = [value_trend] + left_to_right_value
        
        # left_to_right_jitter.append(jitter_trend)
        # left_to_right_warmth.append(warmth_trend)


    # 顯示每個筆觸的強度和抖動趨勢
    if plot:
        plt.imshow(image_copy)
        plt.title("contour detect")
        plt.axis('off')
        plt.show()
        
        plt.figure(figsize=(14, 8))
        for i, (intensity, note, density, hue, saturation, value, x) in enumerate(zip(left_to_right_intensity, left_to_right_note, 
                                                                      left_to_right_density, left_to_right_hue, 
                                                                      left_to_right_saturation, left_to_right_value, effective_x)):
            plt.subplot(3, 2, 1)
            plt.plot(x, note, label=f'{i+1} height')
            plt.title("height(midi note)")
            plt.legend()
            
            plt.subplot(3, 2, 3)
            plt.plot(x, intensity, label=f'{i+1} width')
            plt.title("width(intensity)")
            
            plt.subplot(3, 2, 5)
            plt.plot(x, density, label=f'{i+1} density')
            plt.title("density(noisy)")
            
            plt.subplot(3, 2, 2)
            plt.plot(x, hue, label=f'{i+1} hue')
            plt.title("hue")
            
            plt.subplot(3, 2, 4)
            plt.plot(x, saturation, label=f'{i+1} saturation')
            plt.title("saturation")
            
            plt.subplot(3, 2, 6)
            plt.plot(x, value, label=f'{i+1} value')
            plt.xlabel("position")
            plt.title("value")
            
            # plt.subplot(5, 1, 3)
            # plt.plot(x, jitter, label=f'{i+1} jitter')
            # plt.title("jitter")
            
            # plt.subplot(5, 1, 5)
            # plt.plot(x, warmth, label=f'{i+1} warmth')
            # plt.xlabel("position")
            # plt.title("warmth")

        plt.tight_layout()
        plt.show()
        
    for i in range(len(effective_x)):
        effective_x[i] = (np.array(effective_x[i]) / width)
    
    return [left_to_right_intensity, left_to_right_note, left_to_right_density, 
            left_to_right_hue, left_to_right_saturation, left_to_right_value, effective_x]

# 加載影像
if __name__ == '__main__':
    notation_to_parameters('./notation/note_test25.png') 


    