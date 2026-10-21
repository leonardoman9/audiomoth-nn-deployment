// Optional phase markers for the instruction/memory-access counting plugin (tools/memcount.c).
// A store to g_bird_marker makes the plugin print the counters accumulated since the previous marker.
#ifndef BIRD_MARKER_H
#define BIRD_MARKER_H
#ifdef BIRD_MARKERS
#ifdef __cplusplus
extern "C" {
#endif
extern volatile unsigned g_bird_marker;
#ifdef __cplusplus
}
#endif
#define BIRD_MARK(n) (g_bird_marker = (unsigned)(n))
#else
#define BIRD_MARK(n) ((void)0)
#endif
#endif
