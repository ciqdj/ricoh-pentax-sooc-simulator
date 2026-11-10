#ifndef PHOTO_LAUNCHER_SPLASH_H
#define PHOTO_LAUNCHER_SPLASH_H
#include <windows.h>

enum { SPLASH_PREPARE, SPLASH_COPY, SPLASH_VERIFY, SPLASH_START };
int launcher_splash_english(void);
int launcher_splash_open(HINSTANCE instance, int visible);
void launcher_splash_stage(int stage, int percent);
void launcher_splash_pump(void);
void launcher_splash_close(void);
HWND launcher_splash_window(void);
/* Offscreen diagnostic only; writes a new bitmap and never creates a window. */
int launcher_splash_preview(const wchar_t *filename);
#endif
