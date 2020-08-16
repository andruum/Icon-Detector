import argparse

import cv2
from oneshot_detector import OneShotDetector
from origin_detector import OriginDetector


def parse_args():
    parser = argparse.ArgumentParser(description="Icons detector on the image")
    parser.add_argument(
        "--target_images",
        default=['data/test/icons-huawei/gp.png', 'data/test/icons-huawei/settings.jpg'],
        help="Target image to find",
        type=str,
        nargs='+',
    )
    parser.add_argument(
        "--source_image",
        default='data/test/huawei_real.jpg',
        help="Image for search",
        type=str,
    )
    args = parser.parse_args()

    if args.source_image is None:
        raise Exception("No source image specified")

    if args.target_images is None:
        raise Exception("No target image specified")

    return args.target_images, args.source_image

def main():
    target_paths, source_path = parse_args()

    def load(path):
        return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)

    targets = [load(p) for p in target_paths]
    source = load(source_path)

    detector = OneShotDetector("experiments/config_eval_icons.yml", score_threshold=0.5)
    origin_detector = OriginDetector()

    origin = origin_detector.detect_origin(source)
    bboxes, scores = detector.detect(targets, source)
    print(scores)

    source = cv2.cvtColor(source, cv2.COLOR_RGB2BGR)

    cv2.line(source, (origin[0], 0), (origin[0], source.shape[0]),
             (0, 0, 255), 2)
    cv2.line(source, (0, origin[1]), (source.shape[1], origin[1]),
             (0, 0, 255), 2)
    cv2.circle(source, (origin[0], origin[1]), 8, (0, 255, 0), 8)

    cv2.putText(source, '(0,0)', (origin[0], origin[1] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 1, cv2.LINE_AA)

    for bb in bboxes:
        bbox = [int(bb[0] * source.shape[1]),
                int(bb[1] * source.shape[0]),
                int(bb[2] * source.shape[1]),
                int(bb[3] * source.shape[0])]

        cv2.rectangle(source,
                      (bbox[0], bbox[1]),
                      (bbox[2], bbox[3]),
                      (0, 255, 0), 2)

        bbox_pos_relative = (bbox[0] + (bbox[2] - bbox[0]) // 2 - origin[0], bbox[1] + (bbox[3] - bbox[1]) // 2 - origin[1])
        ##relative position of the bbox

    cv2.imwrite("./result.jpg", source)

if __name__ == '__main__':
    main()
