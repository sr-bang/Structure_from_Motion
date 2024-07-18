import cv2
import numpy as np


def show_matches(image_1, image_2, pts1, pts2, color, file_name):

    concat = np.concatenate((image_1, image_2), axis=1)

    if pts1 is not None:
        corners_1_x = pts1[:, 0].copy().astype(int)
        corners_1_y = pts1[:, 1].copy().astype(int)
        corners_2_x = pts2[:, 0].copy().astype(int)
        corners_2_y = pts2[:, 1].copy().astype(int)
        corners_2_x += image_1.shape[1]

        for i in range(corners_1_x.shape[0]):
            # cv2.circle(image_1, (pts1[0],pts1[1]), radius = 4, color=(0,0,255), thickness = 2)
            # cv2.circle(image_2, (pts2[0],pts2[1]), radius = 4, color=(0,255,0), thickness = 2)
            cv2.line(concat, (corners_1_x[i], corners_1_y[i]),
                     (corners_2_x[i], corners_2_y[i]), color, 1)
    cv2.imshow(file_name, concat)
    cv2.waitKey()
    if file_name is not None:
        cv2.imwrite(file_name, concat)
    cv2.destroyAllWindows()
    return concat


def makeImageSizeSame(imgs):
    images = imgs.copy()
    sizes = []
    for image in images:
        x, y, ch = image.shape
        sizes.append([x, y, ch])

    sizes = np.array(sizes)
    x_target, y_target, _ = np.max(sizes, axis=0)

    images_resized = []

    for i, image in enumerate(images):
        image_resized = np.zeros((x_target, y_target, sizes[i, 2]), np.uint8)
        image_resized[0:sizes[i, 0], 0:sizes[i, 1], 0:sizes[i, 2]] = image
        images_resized.append(image_resized)
