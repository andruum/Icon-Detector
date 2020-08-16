import cv2
import numpy as np


class OriginDetector:

    def __init__(self):
        self.canny_thresh1 = 50
        self.canny_thresh2 = 150
        self.canny_size = 3

        self.hough_thresh = 200

    def detect_origin(self, img: np.ndarray):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        edges = cv2.Canny(gray, self.canny_thresh1, self.canny_thresh2, apertureSize=self.canny_size)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, self.hough_thresh)

        min_x_vertical = np.inf
        min_vertical_line = None

        min_y_horizontal = np.inf
        min_horizontal_line = None

        if lines is None:
            return None, None

        for line in lines:
            r, theta = line[0]

            if abs(theta) > np.pi / 180. and abs(theta - np.pi / 180. * 90.) > np.pi / 180.:
                continue  ## not vertical line nor horizontal

            a = np.cos(theta)
            b = np.sin(theta)

            x0 = a * r
            y0 = b * r

            x1 = int(x0 + 1000 * (-b))
            y1 = int(y0 + 1000 * (a))
            x2 = int(x0 - 1000 * (-b))
            y2 = int(y0 - 1000 * (a))

            if abs(abs(theta)) < np.pi / 180.:
                if min_x_vertical > x0:
                    min_x_vertical = x0
                    min_vertical_line = (x1, y1, x2, y2, r, theta)

            if abs(abs(theta) - np.pi / 180. * 90.) < np.pi / 180.:
                if min_y_horizontal > y0:
                    min_y_horizontal = y0
                    min_horizontal_line = (x1, y1, x2, y2, r, theta)

        if min_vertical_line is None or min_vertical_line is None:
            return None, None

        x_origin = min_vertical_line[4]
        y_origin = min_horizontal_line[4]

        return int(x_origin), int(y_origin)