/*
 * FatFS Stub Functions for NN-only mode
 * For testing NN inference without audio recording
 */

#include "ff.h"
#include "diskio.h"

/* Global file object stub */
FIL file;
UINT bw;

/* FatFS function stubs - return success for compatibility */
FRESULT f_mount(FATFS* fs, const TCHAR* path, BYTE opt) {
    return FR_OK;
}

FRESULT f_open(FIL* fp, const TCHAR* path, BYTE mode) {
    return FR_OK;
}

FRESULT f_close(FIL* fp) {
    return FR_OK;
}

FRESULT f_read(FIL* fp, void* buff, UINT btr, UINT* br) {
    if (br) *br = btr;
    return FR_OK;
}

FRESULT f_write(FIL* fp, const void* buff, UINT btw, UINT* bw_ptr) {
    if (bw_ptr) *bw_ptr = btw;
    return FR_OK;
}

FRESULT f_sync(FIL* fp) {
    return FR_OK;
}

FRESULT f_lseek(FIL* fp, FSIZE_t ofs) {
    return FR_OK;
}

/* Note: f_size is defined as macro in ff.h, we override it here */
#undef f_size
FSIZE_t f_size(FIL* fp) {
    (void)fp; /* Suppress unused parameter warning */
    return 0;
}

FRESULT f_stat(const TCHAR* path, FILINFO* fno) {
    return FR_OK;
}

FRESULT f_mkdir(const TCHAR* path) {
    return FR_OK;
}

FRESULT f_rename(const TCHAR* path_old, const TCHAR* path_new) {
    return FR_OK;
}

/* Disk I/O stubs */
DSTATUS disk_initialize(BYTE pdrv) {
    return 0; /* Success */
}

DSTATUS disk_status(BYTE pdrv) {
    return 0; /* Success */
}

DRESULT disk_read(BYTE pdrv, BYTE* buff, DWORD sector, BYTE count) {
    (void)pdrv; (void)buff; (void)sector; (void)count;
    return RES_OK;
}

DRESULT disk_write(BYTE pdrv, const BYTE* buff, DWORD sector, BYTE count) {
    (void)pdrv; (void)buff; (void)sector; (void)count;
    return RES_OK;
}

DRESULT disk_ioctl(BYTE pdrv, BYTE cmd, void* buff) {
    return RES_OK;
}

/* Note: get_fattime already defined in audiomoth.c */