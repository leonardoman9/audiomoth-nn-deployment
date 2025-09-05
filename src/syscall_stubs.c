/**
 * @file syscall_stubs.c
 * @brief Minimal syscall stubs for bare-metal AudioMoth environment
 * 
 * These stubs are required when using libc_nano.a but running in a 
 * bare-metal environment without a full operating system.
 */

#include <sys/stat.h>
#include <sys/types.h>
#include <errno.h>

/**
 * Exit program - in bare-metal, just infinite loop
 */
void _exit(int status) {
    (void)status;
    while(1) {
        // Infinite loop - no OS to return to
    }
}

/**
 * Heap management - not implemented in bare-metal
 */
void* _sbrk(ptrdiff_t incr) {
    (void)incr;
    errno = ENOMEM;
    return (void*)-1;
}

/**
 * Close file descriptor - not implemented
 */
int _close(int fd) {
    (void)fd;
    errno = EBADF;
    return -1;
}

/**
 * Get file status - not implemented
 */
int _fstat(int fd, struct stat *buf) {
    (void)fd;
    (void)buf;
    errno = EBADF;
    return -1;
}

/**
 * Check if file descriptor is a terminal - not in bare-metal
 */
int _isatty(int fd) {
    (void)fd;
    return 0;  // Not a terminal
}

/**
 * Seek in file - not implemented
 */
off_t _lseek(int fd, off_t offset, int whence) {
    (void)fd;
    (void)offset;
    (void)whence;
    errno = EBADF;
    return -1;
}

/**
 * Read from file descriptor - not implemented
 */
ssize_t _read(int fd, void *buf, size_t count) {
    (void)fd;
    (void)buf;
    (void)count;
    errno = EBADF;
    return -1;
}

/**
 * Write to file descriptor - AudioMoth already has this implemented
 * Removed to avoid linker conflict
 */

/**
 * Kill process - not applicable in bare-metal
 */
int _kill(pid_t pid, int sig) {
    (void)pid;
    (void)sig;
    errno = EINVAL;
    return -1;
}

/**
 * Get process ID - return dummy value
 */
pid_t _getpid(void) {
    return 1;  // Dummy PID
}