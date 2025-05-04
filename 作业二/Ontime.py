from ultralytics import YOLO
import cv2
from collections import Counter

# 加载预训练的 YOLOv8 模型（如果是YOLOv11n，请替换为你的模型）
model = YOLO('../作业一/yolo11n.pt')  # 或者 'yolo11n.pt'

# 打开摄像头（0表示默认摄像头）
cap = cv2.VideoCapture(0)

# 设置窗口名称
window_name = 'YOLOv11 Real-Time Detection'
cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

try:
    while cap.isOpened():
        # 读取摄像头帧
        ret, frame = cap.read()
        if not ret:
            break

        # 运行YOLO推理
        results = model.predict(frame, conf=0.5, verbose=False)  # verbose=False关闭冗余输出

        # 处理每个检测结果
        for result in results:
            # 绘制检测框
            annotated_frame = result.plot()

            # 统计类别数量
            class_counts = Counter([result.names[int(box.cls)] for box in result.boxes])

            # 在图像上显示统计信息
            stats_text = ", ".join([f"{k}:{v}" for k, v in class_counts.items()])
            cv2.putText(annotated_frame, stats_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # 打印到控制台（每秒打印一次，避免刷屏）
            if cv2.getTickCount() % 20 == 0:  # 每20帧打印一次
                print("\n=== 实时检测统计 ===")
                print(f"总检测数: {len(result.boxes)}")
                for class_name, count in class_counts.items():
                    print(f"{class_name}: {count}")
                print("==================")

            # 显示结果
            cv2.imshow(window_name, annotated_frame)

        # 按'q'退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    # 释放资源
    cap.release()
    cv2.destroyAllWindows()
    print("摄像头检测已停止")