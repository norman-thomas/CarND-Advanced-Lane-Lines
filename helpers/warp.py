import cv2


def warp(image, src, dst, newdims=None):
    M = cv2.getPerspectiveTransform(src, dst)
    if newdims is None:
        newdims = image.shape[1], image.shape[0]
    return cv2.warpPerspective(image, M, newdims, flags=cv2.INTER_LINEAR)

