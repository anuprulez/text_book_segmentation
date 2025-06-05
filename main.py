import os
import layoutparser as lp
import cv2

BASE_PATH = "../data/"

def detect_layout():

    # Load the pre-trained model
    image = cv2.imread(BASE_PATH + "two_sides_0.JPG")
    image = image[..., ::-1]
    model = lp.Detectron2LayoutModel('lp://PubLayNet/faster_rcnn_R_50_FPN_3x/config',
                                 extra_config=["MODEL.ROI_HEADS.SCORE_THRESH_TEST", 0.8],
                                 label_map={0: "Text", 1: "Title", 2: "List", 3:"Table", 4:"Figure"})
    layout = model.detect(image)
    # Detect the layout of the input image

    lp.draw_box(image, layout, box_width=3)
    # Show the detected layout of the input image


if __name__ == "__main__":
    detect_layout()
