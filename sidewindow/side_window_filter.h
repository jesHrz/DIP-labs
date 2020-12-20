#ifndef _SIDE_WINDOW_BOX_FILTER_H_
#define _SIDE_WINDOW_BOX_FILTER_H_


const int side_window_directions_num = 8;

void init(int height, int width, int radius);
void side_window_filter(const float *input, int height, int width, float *output);
static void side_window_impl(const float *input, int height, int width, int direction, float *output);

#endif // _SIDE_WINDOW_BOX_FILTER_H_