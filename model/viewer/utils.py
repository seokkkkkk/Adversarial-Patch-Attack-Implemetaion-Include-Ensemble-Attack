from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


# 표의 라벨 설정
def set_label(text, max_length):
    if len(text) <= max_length:
        return text
    else:
        return '\n'.join([text[i:i + max_length] for i in range(0, len(text), max_length)])


# 예측 차트 생성
def prediction_chart(classes, probs, max_class_length=5):
    classes = [set_label(cls, max_class_length) for cls in classes]

    fig, ax = plt.subplots(figsize=(6.67, 6.67))
    canvas = FigureCanvas(fig)
    y_pos = np.arange(len(classes))

    ax.barh(y_pos, probs, align='center', height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, rotation=0, ha='right', fontsize=10)
    ax.set_xticks([])
    ax.set_xlabel('Probability')
    ax.set_title('Top 5 Predictions')

    ax.set_xlim(0, 1)

    fig.tight_layout(pad=1.0)
    chart_image = convert_fig_to_image(canvas)
    plt.close(fig)
    return chart_image


# 그래프 이미지로 변환
def convert_fig_to_image(canvas):
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
    return img
