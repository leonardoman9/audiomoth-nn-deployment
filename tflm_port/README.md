# Porting TFLM + CMSIS-NN (Cortex-M4F)

Sostituisce i percorsi simulati del firmware con TensorFlow Lite Micro reale.
Per ora e' un test autonomo, non ancora integrato nel firmware AudioMoth.

`main.cc` esegue backbone e streaming su Cortex-M4F emulato (QEMU `mps2-an386`),
con kernel CMSIS-NN e istruzioni DSP, e confronta backbone, stato nascosto e
logits con i vettori di riferimento di WrenNet (`docs/deployment/reference`).

```sh
# TFLM compilato per M4: gmake -f tensorflow/lite/micro/tools/make/Makefile \
#   TARGET=cortex_m_generic TARGET_ARCH=cortex-m4+fp OPTIMIZED_KERNEL_DIR=cmsis_nn \
#   BUILD_TYPE=release_with_logs microlite
RAM_KB=64 ./build.sh /percorso/tflite-micro /percorso/WrenNet 29696
./run_qemu.sh
```

## Risultati (TFLM e0021ca, GCC 14.3.1, `-Os`, `NDEBUG`)

- Parita': **0 differenze** su 12 casi backbone e 216 frame streaming (stato
  ricircolato con requantizzazione). Non e' una prova su scheda ne' di accuratezza.
- Flash firmware (`.text`, include i due modelli da 76 KB): 210.872 B su 262.144 B.
  Il firmware AudioMoth esistente non e' incluso.
- Arena: dopo l'inizializzazione servono 20.404 B (backbone) e 5.448 B (streaming),
  ma `AllocateTensors` richiede un'arena di **almeno ~29,2-29,7 KB** per il backbone
  (fallisce a 29.184 B, riesce a 29.696 B): il picco e' nella fase di preparazione.
- Stack massimo osservato: 1.632 B.

## RAM: arena nella SRAM esterna (risolto nel test)

L'AudioMoth ha 256 KB di SRAM esterna (`0x80000000`), gia' usata per i buffer di
registrazione. Con l'arena li' (`-DARENA_IN_EXTSRAM`, 32 KB; in QEMU una regione
sostitutiva a `0x20100000`) e il vincolo reale di 32 KB per la RAM interna:

- RAM interna statica: 680 B; stack massimo 1.632 B. Resta ampio margine.
- Parita': 0 differenze, come prima.
- Flash: 210.872 B, ma lo spazio applicativo AudioMoth e' 240 KB (0x4000-0x40000):
  restano ~34 KB per tutto il firmware non-NN. Da verificare.
- Il capitolo di deployment della tesi (`example.tex`) descrive gia' questa scelta
  (arena da 128 KiB in SRAM esterna, partizioni 40/20 KiB). Il backbone ne richiede
  ~29,7 KB, quindi 40 KiB bastano.
- Da fare sul firmware reale: abilitare la SRAM esterna prima di `AllocateTensors`
  e sottrarre la zona ai buffer di registrazione (`NUMBER_OF_BUFFERS` in `main.c`).
  Gli accessi alla SRAM esterna sono piu' lenti: latenza ed energia vanno misurate.

## Piano di memoria offline (provato, nessun vantaggio)

`add_offline_plan.cc` crea una copia del `.tflite` con il metadato
`OfflineMemoryAllocation` (stessi pesi e operatori; +448 B backbone, +240 B
streaming). Resta bit-exact, ma l'arena minima non cambia: fallisce a 28.672 B e
riesce a 29.696 B come prima. Il picco e' in `CommitStaticMemoryPlan`, dove servono
`AllocationInfo` e memoria del planner proporzionali ai 202 tensori del grafo; gli
activation tensor sono solo ~2,3 KB (CORREZIONE: valeva per il modello KWS a 18 frame; per il backbone degli uccelli, a 301 frame, la parte non persistente dell'arena e' ~67 KB). Ridurre il picco richiederebbe un grafo con
meno tensori (43 RESHAPE nel backbone), cioe' una trasformazione del modello.

## Nota sulle misure della tesi

Latenza (1,692 s/inferenza), energia (0,0771 J) e memoria riportate in `example.tex`
vengono dal firmware in cui backbone e GRU erano simulati: non misurano la rete
vera. Vanno rifatte con questo port, una volta integrato.

## Firmware AudioMoth con TFLM reale (`firmware/`)

`build_fw.sh` compila per EFM32WG380F256 con le sorgenti reali del repository
(device, emlib, USB, `audiomoth.c`), TFLM + CMSIS-NN e i due modelli. `main.cc`
esegue un self-test all'avvio (backbone su un blocco noto + 18 passi streaming,
confronto byte per byte col riferimento) e segnala l'esito: 3 lampeggi verdi =
PASS, 10 rossi = FAIL. Risultati leggibili via GDB (`g_result*`, cicli in
`g_result.cycles_*`, clock in `g_core_clock_hz`). Arene in SRAM esterna:
40 KiB backbone + 20 KiB streaming (`.extsram`, `0x80000000`).

Misurato sul build (GCC 14.3.1, `-Os`, `NDEBUG`):

- Flash: `.text` 226.272 B; lo spazio applicativo e' 245.760 B (0x4000-0x40000),
  restano ~19,5 KB per la logica applicativa (registrazione, FatFS, configurazione):
  **margine stretto**, da verificare con l'applicazione completa.
- RAM interna: .data 112 + .bss 2.144 + heap 3.072 + stack 1.024 = ~6,3 KB su 32 KB.
- SRAM esterna: 61.440 B.
- `build_qemu.sh` esegue lo stesso `nn_real.cc` + `selftest.cc` su Cortex-M4 emulato
  con un `audiomoth.h` fittizio: **PASS, 0 differenze**.

**Non verificato:** esecuzione su scheda (QEMU non emula l'EFM32), tempi e consumi
con SRAM esterna reale, e soprattutto il front-end audio -> spettrogramma: non e'
implementato perche' la sua definizione non e' stata recuperata. Quindi questo
firmware verifica la rete, non ancora il riconoscimento da microfono.
