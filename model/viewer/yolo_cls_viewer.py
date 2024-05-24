from utils import prediction_chart
from ultralytics import YOLO
import cv2 as cv
import torch
import os

# device 설정
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# YOLOv8n-cls 모델 로드
model = YOLO('../yolov8n-cls.pt').to(device)

# 입력 동영상 폴더 경로
input_video_folder = "video"

# 출력 동영상 폴더 경로
output_video_folder = os.path.join(input_video_folder, 'output')

# 출력 폴더가 존재하지 않으면 생성
if not os.path.exists(output_video_folder):
    os.makedirs(output_video_folder)

# 이미지 사이즈
image_size = (640, 640)

# 입력 동영상 폴더 내의 모든 동영상 파일 처리
for filename in os.listdir(input_video_folder):
    if filename.endswith('.mp4') or filename.endswith('.avi'):
        input_video_path = os.path.join(input_video_folder, filename)
        output_video_path = os.path.join(output_video_folder, f"{os.path.splitext(filename)[0]}_output.avi")

        # 동영상 파일 열기
        cap = cv.VideoCapture(input_video_path)

        # 동영상 파일이 열려있는지 확인
        if not cap.isOpened():
            print(f"Error: Could not open video {filename}.")
            continue

        # 동영상 속성 확인
        fps = int(cap.get(cv.CAP_PROP_FPS))
        frame_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

        # 동영상 파일 저장
        fourcc = cv.VideoWriter_fourcc(*'XVID')
        out = cv.VideoWriter(output_video_path, fourcc, fps, (image_size[0] * 2, image_size[1]))

        # 동영상 프레임 읽기
        while cap.isOpened():
            ret, frame = cap.read()

            if not ret:
                break

            # 이미지 크기 조정
            frame = cv.resize(frame, image_size)

            # YOLOv8n-cls 모델 추론
            results = model(frame)

            # 상위 5개의 예측 클래스와 해당 확률을 추출
            probs = results[0].probs  # 확률 추출
            top5_indices = probs.top5  # 상위 5개의 인덱스
            top5_probs = probs.top5conf.to('cpu').numpy()  # 상위 5개의 확률
            top5_classes = [model.names[i] for i in top5_indices]  # 상위 5개의 클래스 이름

            # 예측 차트 생성
            chart_image = prediction_chart(top5_classes, top5_probs)

            # 차트 이미지 크기 조정 (프레임과 동일하게)
            chart_image = cv.resize(chart_image, image_size)

            # 이미지 결합
            combined_image = cv.hconcat([frame, chart_image])

            # 동영상 파일에 프레임 추가
            out.write(combined_image)

            # 실시간으로 보여주기
            cv.imshow('Combined Frame', combined_image)

            if cv.waitKey(1) & 0xFF == ord('q'):
                break

        # 동영상 파일 닫기
        cap.release()
        out.release()
        cv.destroyAllWindows()

print('동영상 처리 완료')