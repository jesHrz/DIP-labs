#ifndef _BOX_FILTER_H_
#define _BOX_FILTER_H_

void median_filter(const float * src, int height, int width, int radius, float *dst);
void mean_filter(const float *src, int height, int width, int radius, float *dst);

#endif // _BOX_FILTER_H_