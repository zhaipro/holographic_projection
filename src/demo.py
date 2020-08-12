import sys
import time
import threading

import cv2
import numpy as np

import draw3d
import green


def show(iframe):
    iframe = cv2.resize(iframe, None, fx=0.5, fy=0.5)
    oframe = np.zeros((1080, 1920, 3), dtype='uint8')
    draw3d.draw3d(oframe[:, 420:-420], iframe, iframe, iframe, iframe)
    oframe = cv2.resize(oframe, None, fx=0.5, fy=0.5)
    cv2.imshow('a', oframe)
    cv2.waitKey()
    cv2.imwrite('result.jpg', oframe)


def test(iframe, n, pad=100):
    greenrendering_mine = green.GreenRenderingCore([68, 138, 81], [0.042, 0.049])
    oframe = np.zeros((2 * 1080 + pad, 2 * 1920 + pad, 3), dtype='uint8')
    start = time.time()
    for _ in range(n):
        draw3d.draw3d(oframe, iframe, iframe, iframe, iframe)
        result = greenrendering_mine.process(oframe)
        result = cv2.resize(result, None, fx=0.5, fy=0.5)
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


def fopen(fn):
    cap = cv2.VideoCapture(fn)
    def fread():
        i = 0
        s = time.time()
        while fread.is_run:
            _, frame = cap.read()
            fread.out = frame
            i += 1
        print(fn, i / (time.time() - s))
    _, frame = cap.read()
    fread.out = frame
    fread.is_run = True
    thread = threading.Thread(target=fread)
    thread.start()
    return fread


def xixi():
    pad = 400
    im = cv2.imread('face.jpg')
    greenrendering_mine = green.GreenRenderingCore([68, 138, 81], [0.042, 0.049])
    t1 = fopen('rtsp://admin:Mb123456@172.16.68.193/h264/ch1/main/av_stream')
    t2 = fopen('rtsp://admin:Mb123456@172.16.68.194/h264/ch1/main/av_stream')
    oframe = np.zeros((1080, 1920, 3), dtype='uint8')
    n = 0
    start = time.time()
    while True:
        w = cv2.resize(t1.out, None, fx=0.5, fy=0.5)
        s = cv2.resize(t2.out, None, fx=0.5, fy=0.5)
        ad = cv2.resize(im, None, fx=0.5, fy=0.5)
        draw3d.draw3d(oframe, w, s, ad, ad)
        result = greenrendering_mine.process(oframe)
        result = cv2.resize(result, None, fx=0.50, fy=0.50)
        cv2.imshow('a', result)
        if cv2.waitKey(1) == ord(' '):
            break
        n += 1
    print(t1.out.shape, t2.out.shape)
    print(n / (time.time() - start))
    t1.is_run = False
    t2.is_run = False


if __name__ == '__main__':
    fn = sys.argv[1] if len(sys.argv) > 1 else 'demo.png'
    iframe = cv2.imread(fn)
    if len(sys.argv) > 2:
        n = int(sys.argv[2])
        test(iframe, n)
    elif len(sys.argv) > 1:
        show(iframe)
    else:
        xixi()
        # demo()
