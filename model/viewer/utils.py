from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


# 길이를 고려하여 라벨 설정
def set_label(text, prob, max_length=5):
    if "_" in text:
        text = text.replace("_", " ")
    words = text.split()
    lines = []
    current_line = ""
    for word in words:
        if len(current_line) + len(word) + 1 <= max_length:
            current_line += (word + " ")
        else:
            lines.append(current_line.strip())
            current_line = word + " "
    if current_line:
        lines.append(current_line.strip())

    lines.append(f"{prob:.2f}")
    return "\n".join(lines)


# 예측 차트 생성
def prediction_chart(classes, probs, max_length=5):
    # 클래스 정렬
    sorted_indices = np.argsort(probs)[::-1]
    classes = [classes[i] for i in sorted_indices]
    probs = [probs[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(6.67, 6.67))
    canvas = FigureCanvas(fig)
    y_pos = np.arange(len(classes))

    bars = ax.barh(y_pos, probs, align='center', height=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([])  # y축 라벨 제거
    ax.set_xticks([])
    ax.set_xlabel('Probability')
    ax.set_title('Top 5 Predictions')

    ax.set_xlim(0, 1)

    # 라벨 설정
    for bar, cls, prob in zip(bars, classes, probs):
        label = set_label(cls, prob, max_length)
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                label, va='center', ha='left', fontsize=10, color='black')

    fig.tight_layout(pad=1.0)
    chart_image = convert_fig_to_image(canvas)
    plt.close(fig)
    return chart_image


# 이미지로 변환
def convert_fig_to_image(canvas):
    canvas.draw()
    buf = canvas.buffer_rgba()
    img = np.asarray(buf)
    img = cv.cvtColor(img, cv.COLOR_RGBA2BGR)
    return img