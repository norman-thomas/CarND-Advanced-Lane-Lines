import cv2


def warp(image, src, dst):
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)

