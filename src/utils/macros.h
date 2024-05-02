#pragma once

#define UP_DIV(x, y) (((x) + (y)-1) / (y))
#define UP_ROUND(x, y) ((((x) + (y)-1) / (y)) * (y))
#define DOWN_ROUND(x, y) (((x) / (y)) * (y))