#%%
from ultralytics import YOLO
import cv2
from collections import Counter  # 用于统计类别数量
#%%
# 加载预训练的 YOLOv11n 模型
model = YOLO('yolo11n.pt')
source = 'image/three_fruit.jpeg' #更改为自己的图片路径
#%%
# 3. 运行推理
results = model.predict(source, save=True, conf=0.5)  # conf为置信度阈值
#%%
# 4. 统计并打印结果
for result in results:
    # 统计每个类别的出现次数
    class_counts = Counter([result.names[int(box.cls)] for box in result.boxes])
    annotated_frame = result.plot()  # 这个方法会返回带标注的图像
    # 打印总检测数量和类别统计
    print(f"\n检测到的目标总数: {len(result.boxes)}")
    print("类别统计:")
    for class_name, count in class_counts.items():
        print(f"- {class_name}: {count}")

    # 打印每个目标的详细信息（可选）
    print("\n详细信息:")
    for box in result.boxes:
        class_name = result.names[int(box.cls)]
        confidence = float(box.conf)
        print(f"  {class_name} (置信度: {confidence:.2f})")
#%%
## 5. 处理并显示结果
### 注意启动后不要点击X号，要点任意一个按键

    # 5.3 显示结果
cv2.imshow('YOLOv8 Detection', annotated_frame)
cv2.waitKey(0)  # 等待任意按键关闭窗口
cv2.destroyAllWindows()