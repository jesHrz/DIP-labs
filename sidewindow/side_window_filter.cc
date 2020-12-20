#include "side_window_filter.h"

#include <cstdio>
#include <vector>

static std::vector<std::vector<float>> scales;
static std::vector<std::vector<float>> results;

static std::vector<std::vector<int>> params;

void init(int height, int width, int radius) {
    int pixel_num = height * width;

    params.resize(side_window_directions_num);
    params[0] = {-radius, 0, -radius, radius}; // L
    params[1] = {0, radius, -radius, radius};  // R
    params[2] = {-radius, radius, -radius, 0}; // U
    params[3] = {-radius, radius, 0, radius};  // D
    params[4] = {-radius, 0, -radius, 0};      // NW
    params[5] = {0, radius, -radius, 0};       // NE
    params[6] = {-radius, 0, 0, radius};       // SW
    params[7] = {0, radius, 0, radius};        // SE

    std::vector<float> eye(pixel_num, 1.0f);

    scales.resize(side_window_directions_num);
    results.resize(side_window_directions_num);
    for (int i = 0; i < side_window_directions_num; ++i) {
        scales[i].resize(pixel_num);
        results[i].resize(pixel_num);
        side_window_impl(eye.data(), height, width, i, scales[i].data());
    }
}

void side_window_filter(const float *input, int height, int width, float *output) {
    int pixel_num = height * width;

    for (int i = 0; i < side_window_directions_num; ++i) {
        float *results_ptr = results[i].data();
        float *scales_ptr = scales[i].data();
        
        side_window_impl(input, height, width, i, results_ptr);

        for (int j = 0; j < pixel_num; ++j) {
            results_ptr[j] /= scales_ptr[j];
        }
    }

    for (int i = 0; i < pixel_num; ++i) {
        float min_idx = 0;
        for (int j = 1; j < side_window_directions_num; ++j) {
            float diff1 = results[j][i] - input[i];
            float diff2 = results[min_idx][i] - input[i];
            if (diff1 * diff1 < diff2 * diff2) {
                min_idx = j;
            }
        }
        output[i] = results[min_idx][i];
    }
}

static void side_window_impl(const float *input, int height, int width, int direction, float *output) {
    int pixel_num = height * width;

    const int w_start = params[direction][0];
    const int w_end = params[direction][1];
    const int h_start = params[direction][2];
    const int h_end = params[direction][3];

    float *row_ptr, *col_ptr;

    std::vector<float> tmp_row(pixel_num);
    std::vector<float> tmp_col(pixel_num);

    row_ptr = tmp_row.data();
    for (int h = 0; h < height; ++h) {
        int offset = h * width;

        float sum = 0;
        for (int w = 0; w < w_end; ++w) {
            sum += input[offset + w];
        }
        for (int w = 0; w < width; ++w) {
            if (w + w_end < width) {
                sum += input[offset + w + w_end];
            }
            if (w + w_start - 1 >= 0) {
                sum -= input[offset + w + w_start - 1];
            }
            row_ptr[offset + w] = sum;
        }
    }

    col_ptr = tmp_col.data();
    float *add_ptr = row_ptr + h_end * width;
    float *sub_ptr = row_ptr + (h_start - 1) * width;

    for (int h = 0; h < h_end; ++h) {
        int offset = h * width;
        for (int w = 0; w < width; ++w) {
            col_ptr[w] += row_ptr[offset + w];
        }
    }
    for (int h = 0; h < height; ++h) {
        int offset = h * width;

        for (int w = 0; w < width; ++w) {
            if (h + h_end < height) {
                col_ptr[w] += add_ptr[offset + w];
            }
            if (h >= -h_start + 1) {
                col_ptr[w] -= sub_ptr[offset + w];
            }
            output[offset + w] = col_ptr[w];
        }
    }          
}