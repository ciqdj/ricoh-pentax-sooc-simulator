/* System-font, native GDI startup artwork. No network or external assets. */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <ctype.h>
#include <math.h>
#include <string.h>
#include <wchar.h>
#include "launcher_splash.h"

#define ART_W 920
#define ART_H 340
static HWND window=NULL;
static HDC art_dc=NULL;
static HBITMAP art_bitmap=NULL;
static HGDIOBJ art_previous=NULL;
static int frame_w=ART_W,frame_h=ART_H,phase=SPLASH_PREPARE,progress=0;
static ULONGLONG started=0,last_pump=0;

int launcher_splash_english(void){
    static int english=-1;
    if(english>=0)return english;
    english=0;
    wchar_t path[32768];
    const wchar_t suffix[]=L"\\DCU5DataTransfer\\language.json";
    DWORD count=GetEnvironmentVariableW(L"APPDATA",path,32768);
    if(!count || count+wcslen(suffix)>=32768)return english;
    wcscpy(path+count,suffix);
    HANDLE file=CreateFileW(path,GENERIC_READ,FILE_SHARE_READ|FILE_SHARE_WRITE|FILE_SHARE_DELETE,
        NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(file==INVALID_HANDLE_VALUE)return english;
    char json[128];
    DWORD bytes=0;
    BOOL read=ReadFile(file,json,sizeof(json)-1,&bytes,NULL);
    CloseHandle(file);
    if(!read)return english;
    json[bytes]='\0';
    const char *key=strstr(json,"\"language\"");
    const char *value=key?strchr(key,':'):NULL;
    if(!value)return english;
    do{value++;}while(isspace((unsigned char)*value));
    english=strncmp(value,"\"en\"",4)==0;
    return english;
}

static int px(int value,int width){return MulDiv(value,width,ART_W);}
static COLORREF shade(int r,int g,int b){return RGB(r<0?0:r>255?255:r,g<0?0:g>255?255:g,b<0?0:b>255?255:b);}
static HBITMAP bitmap(HDC dc,int width,int height,void **pixels){
    BITMAPINFO info={0};info.bmiHeader.biSize=sizeof(BITMAPINFOHEADER);
    info.bmiHeader.biWidth=width;info.bmiHeader.biHeight=-height;
    info.bmiHeader.biPlanes=1;info.bmiHeader.biBitCount=32;info.bmiHeader.biCompression=BI_RGB;
    return CreateDIBSection(dc,&info,DIB_RGB_COLORS,pixels,NULL,0);
}
static void text(HDC dc,const wchar_t *value,int x,int y,int w,int h,int size,int weight,COLORREF color,int width){
    HFONT font=CreateFontW(-px(size,width),0,0,0,weight,FALSE,FALSE,FALSE,DEFAULT_CHARSET,
        OUT_DEFAULT_PRECIS,CLIP_DEFAULT_PRECIS,ANTIALIASED_QUALITY,DEFAULT_PITCH|FF_SWISS,L"Segoe UI");
    HGDIOBJ previous=SelectObject(dc,font);SetBkMode(dc,TRANSPARENT);SetTextColor(dc,color);
    RECT rect={px(x,width),px(y,width),px(x+w,width),px(y+h,width)};
    DrawTextW(dc,value,-1,&rect,DT_LEFT|DT_TOP|DT_SINGLELINE|DT_NOPREFIX);
    SelectObject(dc,previous);DeleteObject(font);
}
static void line(HDC dc,int x1,int y1,int x2,int y2,int thickness,COLORREF color,int width){
    HPEN pen=CreatePen(PS_SOLID,px(thickness,width)>0?px(thickness,width):1,color);
    HGDIOBJ previous=SelectObject(dc,pen);MoveToEx(dc,px(x1,width),px(y1,width),NULL);
    LineTo(dc,px(x2,width),px(y2,width));SelectObject(dc,previous);DeleteObject(pen);
}
static void ellipse(HDC dc,int cx,int cy,int radius,int thickness,COLORREF color,int width){
    HPEN pen=CreatePen(PS_SOLID,px(thickness,width)>0?px(thickness,width):1,color);
    HGDIOBJ old_pen=SelectObject(dc,pen),old_brush=SelectObject(dc,GetStockObject(NULL_BRUSH));
    Ellipse(dc,px(cx-radius,width),px(cy-radius,width),px(cx+radius,width),px(cy+radius,width));
    SelectObject(dc,old_brush);SelectObject(dc,old_pen);DeleteObject(pen);
}
static void artwork(HDC dc,DWORD *pixels,int width,int height){
    /* A soft photographic light field, with restrained deterministic grain. */
    for(int y=0;y<height;y++)for(int x=0;x<width;x++){
        double fx=(double)x/width,fy=(double)y/height;
        double a=(fx-.90)*(fx-.90)*6+(fy-.16)*(fy-.16)*.75;
        double b=(fx-.73)*(fx-.73)*7+(fy-.88)*(fy-.88)*1.2;
        double amber=1.0/(1.0+a*45.0),teal=1.0/(1.0+b*32.0);
        int grain=(int)(((unsigned)x*1664525u+(unsigned)y*1013904223u)>>29)-3;
        int r=(int)(17+amber*45+teal*5)+grain/2;
        int g=(int)(22+amber*24+teal*34)+grain/2;
        int bl=(int)(27+amber*7+teal*32)+grain/2;
        pixels[(size_t)y*width+x]=((DWORD)r<<16)|((DWORD)g<<8)|(DWORD)bl;
    }
    /* Large cropped optical rings and an abstract six-blade aperture. */
    ellipse(dc,787,119,151,1,RGB(91,90,75),width);
    ellipse(dc,787,119,137,1,RGB(71,82,77),width);
    ellipse(dc,787,119,109,2,RGB(115,128,113),width);
    ellipse(dc,787,119,100,1,RGB(44,66,63),width);
    ellipse(dc,787,119,80,1,RGB(91,113,103),width);
    const COLORREF colors[6]={RGB(128,111,75),RGB(157,135,91),RGB(92,123,114),RGB(43,101,102),RGB(64,108,110),RGB(128,126,95)};
    for(int i=0;i<6;i++){
        double a=i*3.141592653589793/3-.32;
        POINT points[4];double angles[4]={a,a+.75,a+1.25,a+.25};int radii[4]={78,78,39,39};
        for(int j=0;j<4;j++){points[j].x=px(787+(int)(radii[j]*cos(angles[j])),width);points[j].y=px(119+(int)(radii[j]*sin(angles[j])),width);}
        HBRUSH brush=CreateSolidBrush(colors[i]);HPEN pen=CreatePen(PS_SOLID,1,colors[i]);
        HGDIOBJ ob=SelectObject(dc,brush),op=SelectObject(dc,pen);Polygon(dc,points,4);
        SelectObject(dc,op);SelectObject(dc,ob);DeleteObject(pen);DeleteObject(brush);
    }
    ellipse(dc,787,119,35,1,RGB(191,190,151),width);
    /* Fine registration lines lend the banner a photographic print feel. */
    line(dc,52,49,78,49,2,RGB(209,197,163),width);
    text(dc,L"PHOTO WORKBENCH",91,39,300,25,15,FW_MEDIUM,RGB(178,186,185),width);
    text(dc,L"RICOH & PENTAX",49,94,660,80,57,FW_SEMIBOLD,RGB(244,244,234),width);
    text(dc,L"SOOC Simulator  3.5.12",54,180,560,30,17,FW_NORMAL,RGB(156,176,171),width);
    line(dc,52,311,868,311,1,RGB(49,65,66),width);
}
static void loading(HDC dc,int width,int stage,int percent,ULONGLONG tick){
    int english=launcher_splash_english();
    wchar_t caption[100];
    const wchar_t *status=english?L"Choosing a writable drive":L"正在选择可用的本地空间";
    if(stage==SPLASH_COPY)status=english?L"Preparing application files":L"正在准备应用资源";
    else if(stage==SPLASH_VERIFY)status=english?L"Checking application files":L"正在核对资源完整性";
    else if(stage==SPLASH_START)status=english?L"Starting the workbench":L"正在启动工作台";
    _snwprintf(caption,100,L"%ls",status);
    text(dc,caption,53,263,660,28,15,FW_NORMAL,RGB(203,213,208),width);
    if(stage==SPLASH_COPY){
        _snwprintf(caption,100,L"%d%%",percent);
        text(dc,caption,818,263,65,28,15,FW_MEDIUM,RGB(222,220,202),width);
        line(dc,52,311,52+816*percent/100,311,2,RGB(218,209,178),width);
    }else{
        /* No fabricated percentage for Python extraction or GUI initialization. */
        int start=(int)((tick/6)%976)-160;
        int left=start<0?0:start,right=start+160>816?816:start+160;
        if(right>left)line(dc,52+left,311,52+right,311,2,RGB(218,209,178),width);
        text(dc,english?L"Please wait":L"请稍候",785,263,105,28,14,FW_NORMAL,RGB(149,168,162),width);
    }
}
static LRESULT CALLBACK splash_proc(HWND hwnd,UINT message,WPARAM wp,LPARAM lp){
    if(message==WM_ERASEBKGND)return 1;
    if(message==WM_TIMER){RECT rect={0,px(250,frame_w),frame_w,frame_h};InvalidateRect(hwnd,&rect,FALSE);return 0;}
    if(message==WM_PAINT){
        PAINTSTRUCT paint;HDC dc=BeginPaint(hwnd,&paint);
        if(art_dc)BitBlt(dc,0,0,frame_w,frame_h,art_dc,0,0,SRCCOPY);
        loading(dc,frame_w,phase,progress,GetTickCount64()-started);EndPaint(hwnd,&paint);return 0;
    }
    if(message==WM_MOUSEACTIVATE)return MA_NOACTIVATE;
    /* AUDIT-P28: the banner is closed only by launcher_splash_close() once the
       first frame is ready. A stray WM_CLOSE must not hide it and leave the user
       with no visible sign that the app is still loading. */
    if(message==WM_CLOSE)return 0;
    return DefWindowProcW(hwnd,message,wp,lp);
}
int launcher_splash_open(HINSTANCE instance,int visible){
    if(window)return 1;
    SetProcessDPIAware();RECT area={0,0,GetSystemMetrics(SM_CXSCREEN),GetSystemMetrics(SM_CYSCREEN)};
    SystemParametersInfoW(SPI_GETWORKAREA,0,&area,0);
    HDC screen=GetDC(NULL);int dpi=GetDeviceCaps(screen,LOGPIXELSX);ReleaseDC(NULL,screen);
    frame_w=MulDiv(ART_W,dpi,96);if(frame_w>area.right-area.left-80)frame_w=area.right-area.left-80;
    if(frame_w<460)frame_w=460;frame_h=MulDiv(ART_H,frame_w,ART_W);
    WNDCLASSW klass={0};klass.lpfnWndProc=splash_proc;klass.hInstance=instance;
    klass.lpszClassName=L"PhotoWorkbenchStartupBanner";klass.style=CS_DROPSHADOW;
    RegisterClassW(&klass);
    void *pixels=NULL;art_dc=CreateCompatibleDC(NULL);art_bitmap=bitmap(art_dc,frame_w,frame_h,&pixels);
    if(!art_dc||!art_bitmap){launcher_splash_close();return 0;}
    art_previous=SelectObject(art_dc,art_bitmap);artwork(art_dc,pixels,frame_w,frame_h);
    /* AUDIT-P28: WS_EX_TOOLWINDOW keeps a window out of the taskbar, so a slow
       first launch looked like nothing had started. WS_EX_APPWINDOW puts the
       banner in the taskbar while WS_EX_NOACTIVATE still avoids stealing focus. */
    window=CreateWindowExW(WS_EX_APPWINDOW|WS_EX_NOACTIVATE,klass.lpszClassName,L"RICOH & PENTAX SOOC Simulator",WS_POPUP,
        area.left+(area.right-area.left-frame_w)/2,area.top+(area.bottom-area.top-frame_h)/2,
        frame_w,frame_h,NULL,NULL,instance,NULL);
    if(!window){launcher_splash_close();return 0;}
    HRGN corner=CreateRoundRectRgn(0,0,frame_w+1,frame_h+1,px(18,frame_w),px(18,frame_w));
    if(corner&&!SetWindowRgn(window,corner,FALSE))DeleteObject(corner);
    phase=SPLASH_PREPARE;progress=0;started=GetTickCount64();SetTimer(window,1,50,NULL);
    if(visible){ShowWindow(window,SW_SHOWNOACTIVATE);UpdateWindow(window);}
    return 1;
}
void launcher_splash_stage(int stage,int percent){
    if(!window)return;
    if(percent<0)percent=0;if(percent>100)percent=100;
    if(phase==stage&&progress==percent)return;
    phase=stage;progress=percent;RECT rect={0,px(250,frame_w),frame_w,frame_h};InvalidateRect(window,&rect,FALSE);
}
void launcher_splash_pump(void){
    ULONGLONG now=GetTickCount64();if(now-last_pump<25)return;last_pump=now;
    MSG message;
    for(unsigned int i=0;i<128&&PeekMessageW(&message,NULL,0,0,PM_REMOVE);i++){
        if(message.message==WM_QUIT)continue;TranslateMessage(&message);DispatchMessageW(&message);
    }
}
void launcher_splash_close(void){
    if(window){KillTimer(window,1);DestroyWindow(window);window=NULL;}
    if(art_dc&&art_previous){SelectObject(art_dc,art_previous);art_previous=NULL;}
    if(art_bitmap){DeleteObject(art_bitmap);art_bitmap=NULL;}
    if(art_dc){DeleteDC(art_dc);art_dc=NULL;}
}
HWND launcher_splash_window(void){return window;}
int launcher_splash_preview(const wchar_t *filename){
    const int width=ART_W*2,height=ART_H*2;void *pixels=NULL;
    HDC dc=CreateCompatibleDC(NULL);HBITMAP image=bitmap(dc,width,height,&pixels);
    if(!dc||!image){if(image)DeleteObject(image);if(dc)DeleteDC(dc);return 0;}
    HGDIOBJ previous=SelectObject(dc,image);artwork(dc,pixels,width,height);loading(dc,width,SPLASH_COPY,42,650);
    BITMAPFILEHEADER header={0};BITMAPINFOHEADER info={0};DWORD count=0;
    header.bfType=0x4d42;header.bfOffBits=sizeof(header)+sizeof(info);header.bfSize=header.bfOffBits+width*height*4;
    info.biSize=sizeof(info);info.biWidth=width;info.biHeight=-height;info.biPlanes=1;info.biBitCount=32;info.biCompression=BI_RGB;
    HANDLE file=CreateFileW(filename,GENERIC_WRITE,0,NULL,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,NULL);
    int ok=file!=INVALID_HANDLE_VALUE;
    if(ok)ok=WriteFile(file,&header,sizeof(header),&count,NULL)&&count==sizeof(header)&&
        WriteFile(file,&info,sizeof(info),&count,NULL)&&count==sizeof(info)&&
        WriteFile(file,pixels,width*height*4,&count,NULL)&&count==width*height*4;
    if(file!=INVALID_HANDLE_VALUE)CloseHandle(file);
    SelectObject(dc,previous);DeleteObject(image);DeleteDC(dc);return ok;
}
