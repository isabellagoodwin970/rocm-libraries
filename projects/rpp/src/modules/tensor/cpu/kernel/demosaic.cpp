/*
MIT License

Copyright (c) 2019 - 2025 Advanced Micro Devices, Inc.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
*/

#include "host_tensor_executors.hpp"

// Bilinear demosaic for an RGGB Bayer pattern anchored at image (0, 0):
//   (even row, even col) = R    (even row, odd col) = G
//   (odd row, even col)  = G    (odd row, odd col)  = B
// Each missing channel is the average of its nearest same-color neighbors, assuming color varies
// smoothly between nearby samples.

// Reflects an index back into [lo, hi) without repeating the edge sample (lo - 1 -> lo + 1), so a
// reflected neighbor keeps the same parity and therefore lands on the same Bayer color
inline Rpp32s demosaic_reflect_index(Rpp32s idx, Rpp32s lo, Rpp32s hi) {
    if (hi - lo == 1) return lo;
    if (idx < lo) return 2 * lo - idx;
    if (idx >= hi) return 2 * (hi - 1) - idx;
    return idx;
}

// Loads a pixel as Rpp32f in the range expected by saturate_pixel() (i8 is shifted to 0 - 255)
inline Rpp32f demosaic_load_pixel(Rpp8u pixel) { return static_cast<Rpp32f>(pixel); }
inline Rpp32f demosaic_load_pixel(Rpp8s pixel) { return static_cast<Rpp32f>(pixel) + 128.0f; }
inline Rpp32f demosaic_load_pixel(Rpp32f pixel) { return pixel; }
inline Rpp32f demosaic_load_pixel(Rpp16f pixel) { return static_cast<Rpp32f>(pixel); }

template <typename T>
static RppStatus demosaic_bilinear_host_tensor_impl(T* srcPtr, RpptDescPtr srcDescPtr, T* dstPtr,
                                                    RpptDescPtr dstDescPtr,
                                                    RpptROIPtr roiTensorPtrSrc,
                                                    RpptRoiType roiType, rpp::Handle& handle) {
    RpptROI roiDefault = rpp_make_roi_xywh_full((Rpp32s)srcDescPtr->w, (Rpp32s)srcDescPtr->h);
    omp_set_dynamic(0);
    omp_set_num_threads(handle.GetNumThreads());
#pragma omp parallel for
    for (int batchCount = 0; batchCount < dstDescPtr->n; batchCount++) {
        RpptROI roi;
        RpptROIPtr roiPtrInput = &roiTensorPtrSrc[batchCount];
        compute_roi_validation_host(roiPtrInput, &roi, &roiDefault, roiType);

        T *srcPtrImage, *dstPtrImage;
        srcPtrImage = srcPtr + batchCount * srcDescPtr->strides.nStride;
        dstPtrImage = dstPtr + batchCount * dstDescPtr->strides.nStride;

        // ROI bounds in absolute image coordinates, so the RGGB phase stays anchored at (0, 0)
        Rpp32s xStart = roi.xywhROI.xy.x;
        Rpp32s yStart = roi.xywhROI.xy.y;
        Rpp32s xEnd = xStart + roi.xywhROI.roiWidth;
        Rpp32s yEnd = yStart + roi.xywhROI.roiHeight;

        // Neighbors outside the ROI are mirrored back inside it, so padding is never read
        auto srcPixel = [&](Rpp32s y, Rpp32s x) -> Rpp32f {
            y = demosaic_reflect_index(y, yStart, yEnd);
            x = demosaic_reflect_index(x, xStart, xEnd);
            return demosaic_load_pixel(
                srcPtrImage[y * srcDescPtr->strides.hStride + x * srcDescPtr->strides.wStride]);
        };

        for (Rpp32s y = yStart; y < yEnd; y++) {
            T* dstPtrRow = dstPtrImage + (y - yStart) * dstDescPtr->strides.hStride;
            bool isEvenRow = ((y & 1) == 0);
            for (Rpp32s x = xStart; x < xEnd; x++) {
                bool isEvenCol = ((x & 1) == 0);
                Rpp32f center = srcPixel(y, x);
                Rpp32f red, green, blue;

                if (isEvenRow && isEvenCol) {
                    // Red location - green from the 4 cross neighbors, blue from the 4 diagonals
                    red = center;
                    green = 0.25f * (srcPixel(y - 1, x) + srcPixel(y + 1, x) +
                                     srcPixel(y, x - 1) + srcPixel(y, x + 1));
                    blue = 0.25f * (srcPixel(y - 1, x - 1) + srcPixel(y - 1, x + 1) +
                                    srcPixel(y + 1, x - 1) + srcPixel(y + 1, x + 1));
                } else if (!isEvenRow && !isEvenCol) {
                    // Blue location - green from the 4 cross neighbors, red from the 4 diagonals
                    blue = center;
                    green = 0.25f * (srcPixel(y - 1, x) + srcPixel(y + 1, x) +
                                     srcPixel(y, x - 1) + srcPixel(y, x + 1));
                    red = 0.25f * (srcPixel(y - 1, x - 1) + srcPixel(y - 1, x + 1) +
                                   srcPixel(y + 1, x - 1) + srcPixel(y + 1, x + 1));
                } else {
                    // Green location - red and blue from the 2 horizontal or 2 vertical neighbors
                    Rpp32f horizontal = 0.5f * (srcPixel(y, x - 1) + srcPixel(y, x + 1));
                    Rpp32f vertical = 0.5f * (srcPixel(y - 1, x) + srcPixel(y + 1, x));
                    green = center;
                    red = isEvenRow ? horizontal : vertical;   // red row: R is left/right
                    blue = isEvenRow ? vertical : horizontal;  // blue row: B is left/right
                }

                // wStride/cStride handle both NCHW (planar) and NHWC (packed) destinations
                T* dstPtrTemp = dstPtrRow + (x - xStart) * dstDescPtr->strides.wStride;
                saturate_pixel(red, dstPtrTemp);
                saturate_pixel(green, dstPtrTemp + dstDescPtr->strides.cStride);
                saturate_pixel(blue, dstPtrTemp + 2 * dstDescPtr->strides.cStride);
            }
        }
    }

    return RPP_SUCCESS;
}

RppStatus demosaic_bilinear_u8_u8_host_tensor(Rpp8u* srcPtr, RpptDescPtr srcDescPtr, Rpp8u* dstPtr,
                                              RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType, RppLayoutParams layoutParams,
                                              rpp::Handle& handle) {
    return demosaic_bilinear_host_tensor_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                              roiTensorPtrSrc, roiType, handle);
}

RppStatus demosaic_bilinear_f32_f32_host_tensor(Rpp32f* srcPtr, RpptDescPtr srcDescPtr,
                                                Rpp32f* dstPtr, RpptDescPtr dstDescPtr,
                                                RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                                RppLayoutParams layoutParams, rpp::Handle& handle) {
    return demosaic_bilinear_host_tensor_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                              roiTensorPtrSrc, roiType, handle);
}

RppStatus demosaic_bilinear_f16_f16_host_tensor(Rpp16f* srcPtr, RpptDescPtr srcDescPtr,
                                                Rpp16f* dstPtr, RpptDescPtr dstDescPtr,
                                                RpptROIPtr roiTensorPtrSrc, RpptRoiType roiType,
                                                RppLayoutParams layoutParams, rpp::Handle& handle) {
    return demosaic_bilinear_host_tensor_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                              roiTensorPtrSrc, roiType, handle);
}

RppStatus demosaic_bilinear_i8_i8_host_tensor(Rpp8s* srcPtr, RpptDescPtr srcDescPtr, Rpp8s* dstPtr,
                                              RpptDescPtr dstDescPtr, RpptROIPtr roiTensorPtrSrc,
                                              RpptRoiType roiType, RppLayoutParams layoutParams,
                                              rpp::Handle& handle) {
    return demosaic_bilinear_host_tensor_impl(srcPtr, srcDescPtr, dstPtr, dstDescPtr,
                                              roiTensorPtrSrc, roiType, handle);
}
