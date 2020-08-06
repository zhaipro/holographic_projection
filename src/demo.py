import sys
import time

import cv2
import numpy as np

import draw3d


def show(iframe, pad=0):
    oframe = np.zeros((2 * 600 + pad, 2 * 600 + pad, 3), dtype='uint8')
    draw3d.draw3d(iframe, oframe)
    oframe = cv2.resize(oframe, (512, 512))
    cv2.imshow('a', oframe)
    cv2.waitKey()
    cv2.imwrite('result.jpg', oframe)


def test(iframe, n, pad=100):
    oframe = np.zeros((2 * 600 + pad, 2 * 600 + pad, 3), dtype='uint8')
    start = time.time()
    for _ in range(n):
        draw3d.draw3d(iframe, oframe)
    print(n / (time.time() - start))


if __name__ == '__main__':
    fn = sys.argv[1] if len(sys.argv) > 1 else 'demo.png'
    iframe = cv2.imread(fn)
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
        tests(iframe, n)
    else:
        show(iframe)
