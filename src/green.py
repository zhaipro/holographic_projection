# encoding : utf-8
import os
import sys

import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
from pycuda import gpuarray
from pycuda.compiler import SourceModule


class GreenRenderingCore(object):

    def __init__(self, chromakey_rgb, simi_blend):
        with open('green_rendering_kernel.cu', 'r') as f:
            source = f.read()
        mod = SourceModule(source)
        self._func_rendering = mod.get_function('rendering_kernel')

        self._chromakey_rgb = np.array(chromakey_rgb, 'uint8')
        self._simi_blend = np.array(simi_blend, 'float32')

        self._in_uint8_gpu = None

    def process(self, img_in):
        h, w, _ = img_in.shape
        if self._in_uint8_gpu is None:
            self._in_uint8_gpu = gpuarray.zeros((h, w, 3), dtype=np.uint8)
            self._out_uint8_gpu = gpuarray.zeros((h, w, 3), dtype=np.uint8)
            self._chromakey_rgb_gpu = gpuarray.to_gpu(self._chromakey_rgb)
            self._simi_blend_gpu = gpuarray.to_gpu(self._simi_blend)
            self._in_diff_gpu = gpuarray.zeros((h, w), dtype=np.float32)

        ## +++++++++++++++++++++++++++
        ## forward
        self._in_uint8_gpu.set(img_in)

        self._func_rendering(
            np.intc(h), np.intc(w),
            self._chromakey_rgb_gpu, self._simi_blend_gpu,
            self._in_uint8_gpu,
            self._in_diff_gpu,
            self._out_uint8_gpu,
            block=(16, 16, 1),
            grid=(8, 8, 1))         # ?????????????

        return self._out_uint8_gpu.get()


if __name__ == '__main__':
    chromakey_rgb_list = [68, 138, 81]
    simi_blend_list = [0.06, 0.07]

    greenrendering_mine = GreenRenderingCore(chromakey_rgb_list, simi_blend_list)

    ## -----------------------
    ## 单张测试
    import cv2
    img = cv2.imread("00000.jpg")
    img_rst = greenrendering_mine.process(img)
    cv2.imwrite('a.png', img_rst)

    # img_gpu.copy()
