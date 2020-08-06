import cv2
import numpy as np


im = cv2.imread('00000.jpg')
h, w, _ = im.shape
mask_down = np.zeros((h, w, 1), dtype='uint8')
fill=400
pts = np.array([
    [0, h], [w, h], [w, w // 2 - fill],
    [w // 2 + fill, 0], [w // 2 - fill, 0],
    [0, w // 2 - fill]], dtype='int32')
cv2.fillPoly(mask_down, pts=[pts], color=255)
mask_down = mask_down.astype('bool')

new_frame = np.zeros(((h + fill) * 2, (h + fill) * 2, 3), dtype='uint8')


def func(frame, fill=400):
    h, w, _ = frame.shape
    pad = h - w // 2
    new_frame[h + 2 * fill:, pad + fill:-pad - fill] += mask_down * frame
    new_frame[pad + fill:-pad - fill, h + 2 * fill:] += np.rot90(mask_down * frame, k=1)
    new_frame[:h, pad + fill:-pad - fill] += np.rot90(mask_down * frame, k=2)
    new_frame[pad + fill:-pad - fill, :h] += np.rot90(mask_down * frame, k=3)
    return new_frame


for _ in range(100):
    func(im)
exit()
im = func(im)
im = cv2.resize(im, (512, 512))
cv2.imshow('a', im)
cv2.waitKey()
cv2.imwrite('a.jpg', im)
