from edge_detection import detect_edges, detect_edges_real_time, detect_edges_real_time_advanced

if __name__ == "__main__":
    detect_edges_real_time_advanced()

    if False:
        detect_edges_real_time()


        image_filename = '.\\Data\\1.1.12.tiff'
        edges = detect_edges(image_filename, 100, 200)
