/* FatFS disk layer backed by a raw image file (512-byte sectors), in place of microsd.c / SPI. */
#include <stdio.h>
#include <stdlib.h>

#include "diskio.h"
#include "ff.h"

static FILE *img;
static unsigned long sectors;
unsigned long g_writes, g_reads;

DSTATUS disk_initialize(BYTE pdrv) {
    (void)pdrv;
    const char *path = getenv("SD_IMAGE");
    if (!path) return STA_NOINIT;
    img = fopen(path, "r+b");
    if (!img) return STA_NODISK;
    fseek(img, 0, SEEK_END);
    sectors = (unsigned long)(ftell(img) / 512);
    return 0;
}
DSTATUS disk_status(BYTE pdrv) { (void)pdrv; return img ? 0 : STA_NOINIT; }
DRESULT disk_read(BYTE pdrv, BYTE *buff, DWORD sector, BYTE count) {
    (void)pdrv;
    if (!img || sector + count > sectors) return RES_PARERR;
    fseek(img, (long)sector * 512, SEEK_SET);
    g_reads += count;
    return fread(buff, 512, count, img) == count ? RES_OK : RES_ERROR;
}
DRESULT disk_write(BYTE pdrv, const BYTE *buff, DWORD sector, BYTE count) {
    (void)pdrv;
    if (!img || sector + count > sectors) return RES_PARERR;
    fseek(img, (long)sector * 512, SEEK_SET);
    g_writes += count;
    return fwrite(buff, 512, count, img) == count ? RES_OK : RES_ERROR;
}
DRESULT disk_ioctl(BYTE pdrv, BYTE cmd, void *buff) {
    (void)pdrv;
    switch (cmd) {
        case CTRL_SYNC: if (img) fflush(img); return RES_OK;
        case GET_SECTOR_COUNT: *(DWORD *)buff = (DWORD)sectors; return RES_OK;
        case GET_SECTOR_SIZE: *(WORD *)buff = 512; return RES_OK;
        case GET_BLOCK_SIZE: *(DWORD *)buff = 8; return RES_OK;
        default: return RES_PARERR;
    }
}
DWORD get_fattime(void) { return ((DWORD)(2026 - 1980) << 25) | (10u << 21) | (21u << 16) | (7u << 11); }
