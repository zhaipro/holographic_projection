import sys
import time

import cv2
import numpy as np

import draw3d
import green


def show(iframe, pad=100):
    oframe = np.zeros((2 * 1080 + pad, 2 * 1080 + pad, 3), dtype='uint8')
    draw3d.draw3d(oframe, iframe, iframe, iframe, iframe)
    oframe = cv2.resize(oframe, (512, 512))
    cv2.imshow('a', oframe)
    cv2.waitKey()
    cv2.imwrite('result.jpg', oframe)


def test(iframe, n, pad=100):
    greenrendering_mine = green.GreenRenderingCore([68, 138, 81], [0.042, 0.049])
    oframe = np.zeros((2 * 1080 + pad, 2 * 1080 + pad, 3), dtype='uint8')
    start = time.time()
    for _ in range(n):
        draw3d.draw3d(oframe, iframe, iframe, iframe, iframe)
        result = greenrendering_mine.process(oframe)
        result = cv2.resize(result, (1080, 1080))
    print(n / (time.time() - start))
    cv2.imwrite('result.jpg', result)
    cv2.imwrite('oframe.jpg', oframe)


def demo(pad=0):
    w = cv2.imread('w.jpg')
    s = cv2.imread('s.jpg')
    a = cv2.imread('a.jpg')
    d = cv2.imread('d.jpg')
    oframe = np.zeros((1000 + pad, 1400 + pad, 3), dtype='uint8')
    draw3d.draw3d(oframe, w, s, a, d)
    oframe = cv2.resize(oframe, None, fx=0.5, fy=0.5)
    cv2.imwrite('result.jpg', oframe)


if __name__ == '__main__':
    fn = sys.argv[1] if len(sys.argv) > 1 else 'demo.png'
    iframe = cv2.imread(fn)
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
        test(iframe, n)
    elif len(sys.argv) > 1:
        show(iframe)
    else:
        demo()
