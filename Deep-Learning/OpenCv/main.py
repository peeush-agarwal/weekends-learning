from edge_detection import detect_edges, detect_edges_real_time, detect_edges_real_time_advanced
from flip_images import flip_webcam_frames_vertically

if __name__ == "__main__":
    flip_webcam_frames_vertically()


    if False:
        detect_edges_real_time_advanced()
        detect_edges_real_time()


        image_filename = '.\\Data\\1.1.12.tiff'
        edges = detect_edges(image_filename, 100, 200)
