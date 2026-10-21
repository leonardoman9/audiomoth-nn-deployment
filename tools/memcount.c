/*
 * QEMU TCG plugin: counts executed instructions and memory accesses (bytes) per address region,
 * with snapshots at marker writes. Used to estimate the run time on the AudioMoth, where the
 * external SRAM sits behind an 8-bit bus with wait states (see README).
 *
 * Regions (match tflm_port/link.ld):  flash 0x00000000-0x000FFFFF | test data 0x00100000-0x001FFFFF |
 *   internal RAM 0x20000000-0x200FFFFF | external SRAM stand-in 0x20100000-0x2FFFFFFF
 * Options: marker=<hex address>. A 32-bit store to that address prints the counters accumulated since the
 * previous marker, labelled with the stored value (the program writes 1, 2, 3, ... at phase boundaries).
 */
#include <glib.h>
#include <inttypes.h>
#include <stdio.h>
#include <string.h>
#include <qemu-plugin.h>

QEMU_PLUGIN_EXPORT int qemu_plugin_version = QEMU_PLUGIN_VERSION;

enum { FLASH, TESTDATA, IRAM, EXT, NREG };
typedef struct { uint64_t insns, rbytes[NREG], wbytes[NREG], racc[NREG], wacc[NREG]; } Counters;
static Counters cur, last_snapshot;
static uint64_t marker_addr;

static int Region(uint64_t a) {
    if (a < 0x00100000) return FLASH;
    if (a < 0x00200000) return TESTDATA;
    if (a >= 0x20000000 && a < 0x20100000) return IRAM;
    if (a >= 0x20100000 && a < 0x30000000) return EXT;
    return -1;
}

static void Print(const char *label, const Counters *c) {
    static const char *names[NREG] = {"flash", "testdata", "iram", "ext"};
    g_autoptr(GString) s = g_string_new(NULL);
    g_string_append_printf(s, "PHASE %s: insns=%" PRIu64, label, c->insns);
    for (int i = 0; i < NREG; ++i)
        g_string_append_printf(s, " | %s r=%" PRIu64 "B/%" PRIu64 " w=%" PRIu64 "B/%" PRIu64, names[i], c->rbytes[i], c->racc[i], c->wbytes[i], c->wacc[i]);
    g_string_append_c(s, '\n');
    qemu_plugin_outs(s->str);
}

static void InsnExec(unsigned int vcpu, void *udata) { cur.insns++; }

static void MemAccess(unsigned int vcpu, qemu_plugin_meminfo_t info, uint64_t vaddr, void *udata) {
    const unsigned size = 1u << qemu_plugin_mem_size_shift(info);
    const bool store = qemu_plugin_mem_is_store(info);
    if (store && marker_addr && vaddr == marker_addr) {
        char label[32];
        snprintf(label, sizeof label, "%u", (unsigned)0);
        Print("marker", &cur);
        memset(&cur, 0, sizeof cur);
        return;
    }
    const int r = Region(vaddr);
    if (r < 0) return;
    if (store) { cur.wbytes[r] += size; cur.wacc[r]++; } else { cur.rbytes[r] += size; cur.racc[r]++; }
}

static void TbTrans(struct qemu_plugin_tb *tb, void *userdata) {
    const size_t n = qemu_plugin_tb_n_insns(tb);
    for (size_t i = 0; i < n; ++i) {
        struct qemu_plugin_insn *insn = qemu_plugin_tb_get_insn(tb, i);
        qemu_plugin_register_vcpu_insn_exec_cb(insn, InsnExec, QEMU_PLUGIN_CB_NO_REGS, NULL);
        qemu_plugin_register_vcpu_mem_cb(insn, MemAccess, QEMU_PLUGIN_CB_NO_REGS, QEMU_PLUGIN_MEM_RW, NULL);
    }
}

static void AtExit(void *p) { Print("end", &cur); }

QEMU_PLUGIN_EXPORT int qemu_plugin_install(qemu_plugin_id_t id, const qemu_info_t *info, int argc, char **argv) {
    for (int i = 0; i < argc; ++i)
        if (g_str_has_prefix(argv[i], "marker=")) marker_addr = g_ascii_strtoull(argv[i] + 7, NULL, 16);
    qemu_plugin_register_vcpu_tb_trans_cb(id, TbTrans, NULL);
    qemu_plugin_register_atexit_cb(id, AtExit, NULL);
    return 0;
}
