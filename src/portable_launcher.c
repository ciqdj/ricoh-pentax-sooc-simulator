/* Select writable storage before the Python bundle is extracted. */
#ifndef UNICODE
#define UNICODE
#endif
#ifndef _UNICODE
#define _UNICODE
#endif
#include <windows.h>
#include <wincrypt.h>
#include <stdio.h>
#include <string.h>
#include <wchar.h>
#include "payload_hash.h"
#include "launcher_splash.h"

static int quiet=0;
/* A GUI launcher must keep servicing its own message queue while the child
   runs. Windows/IME components can create windows on this thread indirectly. */
static int wait_child(HANDLE child,HANDLE ready,DWORD *exitcode) {
    for(;;){
        HANDLE handles[2]={child,ready};DWORD count=ready?2:1;
        DWORD state=MsgWaitForMultipleObjectsEx(count,handles,INFINITE,QS_ALLINPUT,MWMO_INPUTAVAILABLE);
        if(state==WAIT_OBJECT_0){launcher_splash_close();return GetExitCodeProcess(child,exitcode)!=0;}
        if(ready && state==WAIT_OBJECT_0+1){launcher_splash_close();ready=NULL;continue;}
        if(state!=WAIT_OBJECT_0+count)return 0;
        MSG message;
        /* Bound each batch so a busy queue cannot starve the child-exit check.
           WM_QUIT is not permission to delete files a live child still uses. */
        for(unsigned int count=0;count<128 && PeekMessageW(&message,NULL,0,0,PM_REMOVE);count++){
            if(message.message==WM_QUIT)continue;
            TranslateMessage(&message);DispatchMessageW(&message);
        }
    }
}
static void clean_owned(const wchar_t *root,const wchar_t *path,int depth) {
    if(depth>32)return;
    wchar_t *absolute=HeapAlloc(GetProcessHeap(),HEAP_ZERO_MEMORY,3*32768*sizeof(wchar_t));
    if(!absolute)return;
    wchar_t *pattern=absolute+32768,*child=pattern+32768;
    if(!GetFullPathNameW(path,32768,absolute,NULL))goto done;
    size_t length=wcslen(root);
    if(_wcsnicmp(root,absolute,length) || (absolute[length] && absolute[length]!=L'\\'))goto done;
    DWORD attr=GetFileAttributesW(absolute);
    if(attr==INVALID_FILE_ATTRIBUTES || (attr&FILE_ATTRIBUTE_REPARSE_POINT))goto done;
    if(!(attr&FILE_ATTRIBUTE_DIRECTORY)){DeleteFileW(absolute);goto done;}
    _snwprintf(pattern,32768,L"%ls\\*",absolute);WIN32_FIND_DATAW item;
    HANDLE find=FindFirstFileW(pattern,&item);
    if(find!=INVALID_HANDLE_VALUE){
        do {
            if(!wcscmp(item.cFileName,L".")||!wcscmp(item.cFileName,L"..")|| (item.dwFileAttributes&FILE_ATTRIBUTE_REPARSE_POINT))continue;
            _snwprintf(child,32768,L"%ls\\%ls",absolute,item.cFileName);
            clean_owned(root,child,depth+1);
        }while(FindNextFileW(find,&item));FindClose(find);
    }
    RemoveDirectoryW(absolute);
done:
    HeapFree(GetProcessHeap(),0,absolute);
}
static int fail(const wchar_t *message) {
    launcher_splash_close();
    if(!quiet) MessageBoxW(NULL,message,L"RICOH & PENTAX SOOC Simulator",MB_OK|MB_ICONERROR);
    return 12;
}
static wchar_t *arguments(void) {
    wchar_t *p=GetCommandLineW();
    if(*p==L'"'){p++;while(*p && *p!=L'"')p++;if(*p)p++;}
    else while(*p && *p!=L' ' && *p!=L'\t')p++;
    while(*p==L' ' || *p==L'\t')p++;
    return p;
}
static int candidate(const wchar_t *base, ULONGLONG need, wchar_t *work,HCRYPTPROV crypto) {
    ULARGE_INTEGER available;BYTE random[12];wchar_t probe[32768];
    if(!base[0] || wcslen(base)>32000)return 0;
    CreateDirectoryW(base,NULL);
    if(!GetDiskFreeSpaceExW(base,&available,NULL,NULL) || available.QuadPart<need)return 0;
    if(!CryptGenRandom(crypto,sizeof(random),random))return 0;
    _snwprintf(work,32768,L"%ls\\launch-351-%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",base,
        random[0],random[1],random[2],random[3],random[4],random[5],random[6],random[7],random[8],random[9],random[10],random[11]);
    if(!CreateDirectoryW(work,NULL))return 0;
    _snwprintf(probe,32768,L"%ls\\write-check",work);
    HANDLE f=CreateFileW(probe,GENERIC_WRITE,0,NULL,CREATE_NEW,FILE_ATTRIBUTE_TEMPORARY|FILE_FLAG_DELETE_ON_CLOSE,NULL);
    if(f==INVALID_HANDLE_VALUE){RemoveDirectoryW(work);return 0;}
    CloseHandle(f);return 1;
}
enum { PREPARE_OK, PREPARE_FAILED, PREPARE_CRYPTO_FAILED, PREPARE_NO_SPACE };
typedef struct {
    wchar_t self[32768],base[32768],work[32768],payload[32768];
    HCRYPTPROV crypto;
    volatile LONG stage,percent;
    int selected,error;
} PayloadWork;

/* Only this worker owns the file/hash handles. It performs all preparation
   I/O, including the final CloseHandle, and never touches a window or GDI.
   The UI owns the context and crypto provider until the thread has exited. */
static DWORD WINAPI prepare_payload(void *parameter) {
    PayloadWork *job=(PayloadWork *)parameter;
    HANDLE input=INVALID_HANDLE_VALUE,output=INVALID_HANDLE_VALUE;
    HCRYPTHASH hash=0;BYTE digest[32],buffer[65536];
    DWORD n,written,hash_size=32;LARGE_INTEGER offset,length;
    struct {char magic[8];ULONGLONG offset,size;} footer;
    job->error=PREPARE_FAILED;
    if(!CryptAcquireContextW(&job->crypto,NULL,NULL,PROV_RSA_AES,CRYPT_VERIFYCONTEXT)){
        job->error=PREPARE_CRYPTO_FAILED;goto done;
    }
    if(!GetModuleFileNameW(NULL,job->self,32768))goto done;
    input=CreateFileW(job->self,GENERIC_READ,FILE_SHARE_READ,NULL,OPEN_EXISTING,FILE_ATTRIBUTE_NORMAL,NULL);
    if(input==INVALID_HANDLE_VALUE)goto done;
    offset.QuadPart=-(LONGLONG)sizeof(footer);
    if(!SetFilePointerEx(input,offset,NULL,FILE_END)||!ReadFile(input,&footer,sizeof(footer),&n,NULL)||n!=sizeof(footer)||memcmp(footer.magic,"WB351END",8))goto done;
    if(!GetFileSizeEx(input,&length)||length.QuadPart<(LONGLONG)sizeof(footer)||
       footer.size>(ULONGLONG)length.QuadPart-sizeof(footer)||
       footer.offset!=(ULONGLONG)length.QuadPart-sizeof(footer)-footer.size)goto done;
    /* Reserve the measured bundle expansion and a bounded startup margin.
       Photo workers check their own image/cache requirements separately. */
    ULONGLONG need=footer.size+PAYLOAD_UNPACKED_BYTES+256ULL*1024*1024;
    if(GetEnvironmentVariableW(L"PHOTO_WORKBENCH_TEMP",job->base,32768))
        job->selected=candidate(job->base,need,job->work,job->crypto);
    /* Prefer the local disk carrying the portable EXE; never hardcode D. */
    if(!job->selected){
        wchar_t volume[32768];
        if(GetVolumePathNameW(job->self,volume,32768) && GetDriveTypeW(volume)==DRIVE_FIXED){
            _snwprintf(job->base,32768,L"%lsPhotoWorkbenchTemp",volume);
            job->selected=candidate(job->base,need,job->work,job->crypto);
        }
    }
    if(!job->selected && GetTempPathW(32768,job->base))
        job->selected=candidate(job->base,need,job->work,job->crypto);
    if(!job->selected){
        for(wchar_t drive=L'C';drive<=L'Z';drive++){
            wchar_t root[]={drive,L':',L'\\',0};
            if(GetDriveTypeW(root)!=DRIVE_FIXED)continue;
            _snwprintf(job->base,32768,L"%lsPhotoWorkbenchTemp",root);
            if(candidate(job->base,need,job->work,job->crypto)){job->selected=1;break;}
        }
    }
    if(!job->selected){job->error=PREPARE_NO_SPACE;goto done;}
    _snwprintf(job->payload,32768,L"%ls\\PhotoWorkbench.exe",job->work);
    output=CreateFileW(job->payload,GENERIC_WRITE,0,NULL,CREATE_NEW,FILE_ATTRIBUTE_NORMAL,NULL);
    if(output==INVALID_HANDLE_VALUE)goto done;
    if(!CryptCreateHash(job->crypto,CALG_SHA_256,0,0,&hash))goto done;
    offset.QuadPart=footer.offset;if(!SetFilePointerEx(input,offset,NULL,FILE_BEGIN))goto done;
    ULONGLONG remaining=footer.size;
    InterlockedExchange(&job->stage,SPLASH_COPY);
    while(remaining){
        DWORD block=(DWORD)(remaining>sizeof(buffer)?sizeof(buffer):remaining);
        if(!ReadFile(input,buffer,block,&n,NULL)||n!=block||!CryptHashData(hash,buffer,n,0)||!WriteFile(output,buffer,n,&written,NULL)||written!=n)goto done;
        remaining-=n;
        InterlockedExchange(&job->percent,(LONG)((footer.size-remaining)*100/footer.size));
    }
    InterlockedExchange(&job->stage,SPLASH_VERIFY);
    CloseHandle(output);output=INVALID_HANDLE_VALUE;
    if(!CryptGetHashParam(hash,HP_HASHVAL,digest,&hash_size,0)||memcmp(digest,PAYLOAD_HASH,32))goto done;
    job->error=PREPARE_OK;
done:
    if(output!=INVALID_HANDLE_VALUE)CloseHandle(output);
    if(hash)CryptDestroyHash(hash);
    if(input!=INVALID_HANDLE_VALUE)CloseHandle(input);
    return (DWORD)job->error;
}

static void wait_payload(HANDLE worker,PayloadWork *job) {
    for(;;){
        LONG stage=InterlockedCompareExchange(&job->stage,0,0);
        LONG percent=InterlockedCompareExchange(&job->percent,0,0);
        launcher_splash_stage((int)stage,(int)percent);
        DWORD state=MsgWaitForMultipleObjectsEx(1,&worker,25,QS_ALLINPUT,MWMO_INPUTAVAILABLE);
        if(state==WAIT_OBJECT_0)break;
        /* An unexpected message-wait failure must not free a live worker's
           context or delete its files. Retain a bounded handle wait. */
        if(state==WAIT_FAILED && WaitForSingleObject(worker,25)==WAIT_OBJECT_0)break;
        MSG message;
        for(unsigned int count=0;count<128 && PeekMessageW(&message,NULL,0,0,PM_REMOVE);count++){
            if(message.message==WM_QUIT)continue;
            TranslateMessage(&message);DispatchMessageW(&message);
        }
    }
}

typedef struct {
    const wchar_t *application;
    wchar_t *command;
    PROCESS_INFORMATION process;
    DWORD error;
    int created;
} ChildLaunch;

/* Creating a large executable can wait on the file system or a scanner.
   Keep that wait off the splash thread too, not just payload extraction.
   The caller retains the mutable command and context until this thread exits. */
static DWORD WINAPI start_child(void *parameter) {
    ChildLaunch *launch=(ChildLaunch *)parameter;
    STARTUPINFOW startup={sizeof(startup)};
    if(quiet){startup.dwFlags=STARTF_USESHOWWINDOW;startup.wShowWindow=SW_HIDE;}
    launch->created=CreateProcessW(launch->application,launch->command,NULL,NULL,FALSE,
        quiet?CREATE_NO_WINDOW:0,NULL,NULL,&startup,&launch->process)!=0;
    launch->error=launch->created?ERROR_SUCCESS:GetLastError();
    return launch->error;
}

int WINAPI wWinMain(HINSTANCE instance,HINSTANCE previous,LPWSTR unused,int show) {
    wchar_t preview_path[32768];
    wchar_t *args=arguments();quiet=wcsncmp(args,L"--",2)==0;
    /* AUDIT-P25: a second double-click must not unpack 857 MB again or race
       the first instance for the same temp workspace. One named mutex, checked
       before anything is allocated. Command-line workers keep running in
       parallel on purpose, so the guard only applies to interactive launches. */
    HANDLE single=NULL;
    if(!quiet){
        single=CreateMutexW(NULL,FALSE,L"Local\\PhotoWorkbenchSingleInstance");
        if(single && GetLastError()==ERROR_ALREADY_EXISTS){
            MessageBoxW(NULL,
                launcher_splash_english()
                    ? L"RICOH & PENTAX SOOC Simulator is already running.\n\nSwitch to the existing window, or close it before starting again."
                    : L"程序已经在运行中。\n\n请切换到已打开的窗口，或先关闭它再重新启动。",
                L"RICOH & PENTAX SOOC Simulator",MB_OK|MB_ICONINFORMATION);
            CloseHandle(single);
            return 0;
        }
    }
    const wchar_t *preview_option=L"--launcher-splash-preview";size_t preview_length=wcslen(preview_option);
    if(!wcsncmp(args,preview_option,preview_length) && (args[preview_length]==L' '||args[preview_length]==L'\t')){
        wchar_t *name=args+preview_length;while(*name==L' '||*name==L'\t')name++;
        size_t count=wcslen(name);if(count>=32768)return 12;
        if(count>=2&&name[0]==L'"'&&name[count-1]==L'"'){name++;count-=2;}
        wmemcpy(preview_path,name,count);preview_path[count]=0;return launcher_splash_preview(preview_path)?0:12;
    }
    if(!quiet)launcher_splash_open(instance,1);
    PayloadWork *job=HeapAlloc(GetProcessHeap(),HEAP_ZERO_MEMORY,sizeof(PayloadWork));
    if(!job)return fail(launcher_splash_english()?L"Could not prepare startup files.":L"无法准备启动资源。");
    job->error=PREPARE_FAILED;
    HANDLE ready=NULL;
    int result=12,may_clean=1;
    if(quiet)prepare_payload(job);
    else{
        HANDLE worker=CreateThread(NULL,0,prepare_payload,job,0,NULL);
        if(!worker)goto cleanup;
        wait_payload(worker,job);CloseHandle(worker);
    }
    if(job->error!=PREPARE_OK)goto cleanup;
    SetEnvironmentVariableW(L"TEMP",job->work);SetEnvironmentVariableW(L"TMP",job->work);SetEnvironmentVariableW(L"PHOTO_WORKBENCH_TEMP",job->base);
    if(!quiet && launcher_splash_window()){
        BYTE token[12];wchar_t event_name[128];
        if(CryptGenRandom(job->crypto,sizeof(token),token)){
            _snwprintf(event_name,128,L"Local\\PhotoWorkbenchReady-%lu-%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x%02x",
                GetCurrentProcessId(),token[0],token[1],token[2],token[3],token[4],token[5],token[6],token[7],token[8],token[9],token[10],token[11]);
            ready=CreateEventW(NULL,TRUE,FALSE,event_name);
            if(ready && !SetEnvironmentVariableW(L"PHOTO_WORKBENCH_READY_EVENT",event_name)){CloseHandle(ready);ready=NULL;}
        }
        /* The splash must not outlive readiness if event creation is unavailable. */
        if(!ready)launcher_splash_close();
    }
    launcher_splash_stage(SPLASH_START,0);launcher_splash_pump();
    /* Internal workers start the unpacked executable directly; only the user's
       outer launch does this disk selection and payload integrity check. */
    size_t capacity=wcslen(job->payload)+wcslen(args)+8;
    wchar_t *command=HeapAlloc(GetProcessHeap(),HEAP_ZERO_MEMORY,capacity*sizeof(wchar_t));
    if(!command)goto cleanup;
    _snwprintf(command,capacity,L"\"%ls\" %ls",job->payload,args);
    ChildLaunch launch={0};launch.application=job->payload;launch.command=command;
    if(quiet)start_child(&launch);
    else{
        InterlockedExchange(&job->stage,SPLASH_START);InterlockedExchange(&job->percent,0);
        HANDLE worker=CreateThread(NULL,0,start_child,&launch,0,NULL);
        if(worker){wait_payload(worker,job);CloseHandle(worker);}
        else launch.error=GetLastError();
    }
    if(launch.created){
        DWORD exitcode=12;may_clean=0;
        if(wait_child(launch.process.hProcess,ready,&exitcode)){may_clean=1;result=(int)exitcode;}
        CloseHandle(launch.process.hThread);CloseHandle(launch.process.hProcess);
    }
    HeapFree(GetProcessHeap(),0,command);
cleanup:
    /* Scope cleanup to the newly created private directory, checking every
       resolved child stays beneath it; never follow reparse points. */
    launcher_splash_close();
    wchar_t owned_root[32768];
    if(may_clean && job->selected && GetFullPathNameW(job->work,32768,owned_root,NULL))clean_owned(owned_root,owned_root,0);
    if(ready)CloseHandle(ready);
    if(job->crypto)CryptReleaseContext(job->crypto,0);
    if(single){ReleaseMutex(single);CloseHandle(single);single=NULL;}
    int prepare_error=job->error;HeapFree(GetProcessHeap(),0,job);
    int english=launcher_splash_english();
    if(result==12 && prepare_error==PREPARE_CRYPTO_FAILED)return fail(english?L"Could not initialize the file check.":L"无法初始化完整性检查。");
    if(result==12 && prepare_error==PREPARE_NO_SPACE)return fail(english?L"No writable drive has enough free space. Free some space and try again.":L"没有找到空间足够且可写的本地磁盘。请留出至少 4 GB 空间后重试；程序未修改照片。");
    if(result==12)return fail(english?L"The app could not start or verify its files. Download a fresh EXE and check the temporary drive.":L"程序启动或文件完整性检查失败。请重新取得完整 EXE，并检查临时磁盘可写及空间充足。");
    return result;
}
