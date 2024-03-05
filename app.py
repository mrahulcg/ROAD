import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO
from moviepy.editor import VideoFileClip
from ultralytics import YOLO

# Load a YOLO model for road damage detection
damage_model = YOLO("best.pt")
damage_class_names = damage_model.names

# Function to perform YOLO detection on the uploaded image for road damage
def yolo_detection(image):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (1020, 500))
    h, w, _ = img.shape
    results = damage_model.predict(img)

    for r in results:
        boxes = r.boxes
        masks = r.masks

    if masks is not None:
        masks = masks.data.cpu()
        for seg, box in zip(masks.data.cpu().numpy(), boxes):
            seg = cv2.resize(seg, (w, h))
            contours, _ = cv2.findContours((seg).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                d = int(box.cls)
                c = damage_class_names[d]
                x, y, x1, y1 = cv2.boundingRect(contour)
                cv2.polylines(img, [contour], True, color=(0, 0, 255), thickness=1)
                cv2.putText(img, c, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return img

def grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def canny(img, low_threshold, high_threshold):
    return cv2.Canny(img, low_threshold, high_threshold)

def gaussian_blur(img, kernel_size):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

def region_of_interest(img, vertices):
    mask = np.zeros_like(img)

    if len(img.shape) > 2:
        channel_count = img.shape[2]
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    cv2.fillPoly(mask, vertices, ignore_mask_color)

    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def draw_lines(img, lines, color=[255, 0, 0], thickness=10):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def slope_lines(image, lines):
    img = image.copy()
    poly_vertices = []
    order = [0, 1, 3, 2]

    left_lines = []  # Like /
    right_lines = []  # Like \
    for line in lines:
        for x1, y1, x2, y2 in line:

            if x1 == x2:
                pass  # Vertical Lines
            else:
                m = (y2 - y1) / (x2 - x1)
                c = y1 - m * x1

                if m < 0:
                    left_lines.append((m, c))
                elif m >= 0:
                    right_lines.append((m, c))

    left_line = np.mean(left_lines, axis=0)
    right_line = np.mean(right_lines, axis=0)

    for slope, intercept in [left_line, right_line]:
        rows, cols = image.shape[:2]
        y1 = int(rows)  # image.shape[0]
        y2 = int(rows * 0.6)  # int(0.6*y1)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        poly_vertices.append((x1, y1))
        poly_vertices.append((x2, y2))
        draw_lines(img, np.array([[[x1, y1, x2, y2]]]))

    poly_vertices = [poly_vertices[i] for i in order]
    cv2.fillPoly(img, pts=np.array([poly_vertices], 'int32'), color=(0, 255, 0))
    return cv2.addWeighted(image, 0.7, img, 0.4, 0.)

def hough_lines(img, rho, theta, threshold, min_line_len, max_line_gap):
    lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]), minLineLength=min_line_len, maxLineGap=max_line_gap)
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    line_img = slope_lines(line_img, lines)
    return line_img

def get_vertices(image):
    rows, cols = image.shape[:2]
    bottom_left = [cols * 0.15, rows]
    top_left = [cols * 0.45, rows * 0.6]
    bottom_right = [cols * 0.95, rows]
    top_right = [cols * 0.55, rows * 0.6]

    ver = np.array([[bottom_left, top_left, top_right, bottom_right]], dtype=np.int32)
    return ver

def weighted_img(img, initial_img, α=0.1, β=1., γ=0.):
    lines_edges = cv2.addWeighted(initial_img, α, img, β, γ)
    return lines_edges

# Function to process images for road lane detection
def lane_finding_pipeline(image):
    gray_img = grayscale(image)
    smoothed_img = gaussian_blur(img=gray_img, kernel_size=5)
    canny_img = canny(img=smoothed_img, low_threshold=180, high_threshold=240)
    masked_img = region_of_interest(img=canny_img, vertices=get_vertices(image))
    houghed_lines = hough_lines(img=masked_img, rho=1, theta=np.pi / 180, threshold=20, min_line_len=20, max_line_gap=180)
    output = weighted_img(img=houghed_lines, initial_img=image, α=0.8, β=1., γ=0.)

    return output

# Function to process images combining both road lane and road damage detection
def process_image(image, option):
    if option == "Road Lane Detection":
        return lane_finding_pipeline(image)
    elif option == "Road Damage Detection":
        return yolo_detection(image)

# Function to process videos combining both road lane and road damage detection
def process_video(video_contents, option):
    # Save video contents to a temporary file
    temp_filename = "temp_video.mp4"

    with open(temp_filename, "wb") as temp_file_write:
        temp_file_write.write(video_contents)

    # Read the temporary file using MoviePy
    clip = VideoFileClip(temp_filename)

    if option == "Road Lane Detection":
        # Process the video for lane detection
        processed_clip = clip.fl_image(lane_finding_pipeline)
    elif option == "Road Damage Detection":
        # Process the video for road damage detection
        processed_clip = clip.fl_image(yolo_detection)

    # Create a BytesIO buffer to store the processed video
    processed_video_buffer = BytesIO()

    # Write the processed video to the buffer in MP4 format
    processed_clip.write_videofile(temp_filename, codec="libx264", audio_codec="aac", temp_audiofile="temp_audio.m4a", remove_temp=True)

    # Read the processed video into the buffer
    with open(temp_filename, "rb") as processed_file:
        processed_video_buffer.write(processed_file.read())

    # Cleanup: Close the MoviePy clip and remove the temporary file
    clip.reader.close()
    clip.audio.reader.close_proc()

    return processed_video_buffer.getvalue()

def main():
    st.title("Combined Detection App")

    option = st.sidebar.selectbox("Select Detection Type", ["Road Lane Detection", "Road Damage Detection"])

    if option == "Road Lane Detection":
        uploaded_image = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_image is not None:
            image = Image.open(uploaded_image)
            image = np.array(image)

            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Run Lane Detection on Image", key="lane_image"):
                result = process_image(image, option)
                st.image(result, caption="Processed Image", use_column_width=True)

        uploaded_video = st.file_uploader("Upload Video", type=["mp4"])

        if uploaded_video is not None:
            st.video(uploaded_video)

            if st.button("Run Lane Detection on Video", key="lane_video"):
                with st.spinner("Processing..."):
                    video_contents = uploaded_video.read()
                    video_result = process_video(video_contents, option)
                    st.video(video_result, format="video/mp4")

    elif option == "Road Damage Detection":
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

        if uploaded_file is not None:
            image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
            st.image(image, caption="Uploaded Image", use_column_width=True)

            if st.button("Detect Road Damage", key="damage_detection"):
                detected_image = yolo_detection(image)
                st.image(detected_image, caption="Detected Objects", use_column_width=True)

if __name__ == "__main__":
    main()