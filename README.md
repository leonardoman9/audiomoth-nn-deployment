# WrenNet su AudioMoth

Firmware per **AudioMoth Dev** (EFM32WG380F256, Cortex-M4F a 48 MHz). Registra dal microfono e classifica il canto
degli uccelli con la rete **WrenNet** (8 specie + "non-bird"). Tutto il calcolo avviene sulla scheda:

```
microfono -> ADC 384 kHz -> decimazione a 32 kHz -> front-end C (STFT + 64 filtri appresi, dB, normalizzazione)
          -> backbone CNN int8 (TFLM + CMSIS-NN) -> GRU + attenzione + FC in C -> 9 logit -> LOG.TXT su microSD
```

Ogni giorno alle **07:00** (orologio della scheda) classifica circa **5 minuti** di audio in finestre da 3 s. Per ogni
finestra aggiunge una riga a `LOG.TXT`. Il resto del tempo la scheda dorme (EM4). Non usa USB, GPS né l'app di
configurazione.

> **Stato.** Verificato solo in emulazione: simulatore su computer, Cortex-M4 in QEMU, FatFS su immagine FAT32.
> **Non è ancora stato eseguito su una scheda reale.** Tempi ed energia sono stime. Risultati completi in
> [docs/VERIFICATION.md](docs/VERIFICATION.md).

| | |
|---|---|
| Flash | 230.124 B su 245.760 (margine ~15,6 KB) |
| RAM interna | .data 192 B + .bss 14.112 B, stack ~2,7 KB |
| SRAM esterna | ~160 KB (arena TFLM 96 KiB + buffer) |
| Modello | backbone int8 94,7 KB, 151 operatori; coda (GRU 64) in C con pesi int8 |
| Accuratezza (8 specie, 161 finestre) | float 76,4 %, pipeline del dispositivo 75,8 % |

Le classi, nell'ordine dei logit, sono: Bubo bubo, Certhia familiaris, Apus apus, Certhia brachydactyla, Emberiza cia,
Lophophanes cristatus, Periparus ater, Poecile montanus, non-bird.

## Cosa serve

**Hardware**
- AudioMoth **Dev** con SRAM esterna e slot microSD. Le versioni con microfono USB (≥ AM_VERSION_4) non hanno né SD né
  SRAM esterna.
- microSD formattata **FAT32**. FatFS è ridotto: niente exFAT, solo nomi 8.3.
- Per programmare la scheda: AudioMoth Flash App (bootloader) oppure J-Link/SWD.

**Software (macOS o Linux)**
- `git`, `curl`, **GNU make ≥ 4**. Su macOS: `brew install make`, che installa `gmake`; il `make` di sistema è troppo vecchio.
- `python3` con `numpy`.
- La toolchain ARM **non** va installata: la scarica TFLM (GCC 14.3.1). Funziona anche GNU Arm 10.3-2021.10, con la
  stessa dimensione (vedi VERIFICATION.md).
- Solo per i test: `cc`/`clang`, `qemu-system-arm`, `mtools`. Per i confronti con il modello serve anche un ambiente
  Python con `ai_edge_litert`, `torch`, `torchaudio`, `soundfile`.

**Dal repository WrenNet** (`docs/deployment/bird/`)
- `bird_backbone_int8.tflite`: il modello che finisce nel firmware.
- `bird_reference.npz` e il checkpoint `best_distillation_model.pt`: servono solo per i test di confronto con PyTorch.

## Compilare

```sh
scripts/setup_tflm.sh    # scarica tflite-micro al commit verificato (e0021ca) e applica firmware/patches/tflm-no-int4.patch
scripts/build_tflm.sh    # compila TFLM per Cortex-M4F; la prima volta scarica GCC e CMSIS (qualche minuto)
firmware/app/build_app.sh third_party/tflite-micro /percorso/WrenNet/docs/deployment/bird/bird_backbone_int8.tflite
```

Il risultato è `firmware/app/out/bird_logger.bin`, insieme a `bird_logger.axf`. Lo script stampa l'occupazione:
`flash image: 230124 bytes (limit 245760)`. Le opzioni di `build_tflm.sh` (kernel a `-Os`) sono necessarie: senza, il
firmware supera i 244 KB.

Il **firmware diagnostico** registra file WAV invece di classificare e serve a controllare il microfono sulla scheda vera:

```sh
EXTRA_CF=-DBIRD_DIAG_RECORD firmware/app/build_app.sh third_party/tflite-micro .../bird_backbone_int8.tflite
```

## Usare la scheda

1. Carica il `.bin` con il bootloader; l'applicazione parte da `0x4000`.
2. Sulla microSD crea `CONFIG.TXT` con una sola riga, l'ora attuale: `2026-10-21 06:58:30`.
3. Inserisci la scheda e alimenta. All'avvio l'orologio viene impostato e `CONFIG.TXT` viene riscritto come
   `APPLIED <ora>`, così non viene applicato di nuovo.
4. Dalle 07:00 alle 07:05 il firmware scrive una riga ogni ~6 s in `LOG.TXT`:

   ```
   2026-10-21 07:00:09,Bubo bubo,69,Y,241,-87,-15,-19,-65,-56,-37,-39,-79
   ora, classe più probabile, probabilità %, Y se >= 50 %, 9 logit x100
   ```

Se manca l'alimentazione l'ora si perde e serve un nuovo `CONFIG.TXT`. La deriva dell'orologio è di ±0,9 s al giorno.
Orario e durata si cambiano in `firmware/app/schedule.h` (`SCHED_WINDOW_*`).

**Modalità replay.** Se sulla SD c'è `REPLAY.PCM`, il firmware classifica quella finestra invece del microfono e scrive
una riga `REPLAY,...` nel log. Il file contiene una finestra nota da 3 s e si prepara con `tests/sim/run_replay.py`.
Serve a verificare che il calcolo sulla scheda coincida con quello del computer.

**Primo test su una scheda reale**, in quest'ordine:
1. firmware diagnostico → WAV registrati: controlla livelli e rumore del microfono;
2. firmware normale con `REPLAY.PCM` → confronta i logit con quelli del computer;
3. sessione completa alle 07:00 → misura tempi e consumi.

## Verificare senza scheda

```sh
T=third_party/tflite-micro; M=/percorso/WrenNet/docs/deployment/bird/bird_backbone_int8.tflite

# logica di orario e log: 1.805 casi confrontati con datetime di Python
cc -Ifirmware/app firmware/app/schedule.c tests/schedule/test_schedule.c -lm -o /tmp/ts && python3 tests/schedule/test_schedule.py /tmp/ts

# simulatore: main_app.c e la pipeline vera su un modello software della scheda, per più giorni simulati
tests/sim/build_sim.sh $T $M
python3 tests/sim/make_audio.py audio.raw <file o cartelle audio>
python3 tests/sim/run_scenario.py tests/sim/out/bird_logger_sim audio.raw /tmp/work 2
python3 tests/sim/verify_log.py /percorso/WrenNet best_distillation_model.pt /tmp/work    # confronto con PyTorch

# file system: FatFS reale su un'immagine FAT32 (per esempio: mformat -C -F -T 262144 -i card.img ::)
tests/fstest/run_fstest.sh card.img
```

La pipeline completa in QEMU (da PCM a logit su Cortex-M4 emulato, confronto byte per byte con il computer) e il profilo
di istruzioni e memoria usano `firmware/bird/build_bird_qemu.sh` e `tools/memcount.c`. Procedura e risultati sono in
[docs/VERIFICATION.md](docs/VERIFICATION.md).

## Struttura

```
firmware/app      applicazione: main_app.c (hardware), schedule.c (orario e log), fatfs_conf/, build_app.sh
firmware/bird     front-end, backbone TFLM, coda GRU/attenzione, tabelle generate, script QEMU e di generazione
firmware/patches  patch TFLM (toglie i kernel int4, risparmia flash)
vendor/audiomoth  libreria hardware AudioMoth (emlib, FatFS, driver, audiomoth.c) da AudioMoth-Firmware-Basic
tests/            sim (simulatore), fstest (FatFS), schedule, qemu (harness)
tools/            generatore di tabelle, plugin QEMU conta-istruzioni, parser energia da oscilloscopio
scripts/          setup_tflm.sh, build_tflm.sh
docs/             VERIFICATION.md (tutti i risultati), TFLM_BUILD.md, datasheet
benchmarks/       benchmark Raspberry Pi (lavoro precedente; audio-model-int8.tflite, 41 MB, non è nel repo)
```

## Limiti noti

- Mai eseguito su hardware. Restano da provare: l'abilitazione della SRAM esterna, il driver SD (SPI), i livelli reali
  del microfono, i tempi (stima: ~6–8 s di calcolo per finestra) e l'energia (stima: ~0,8–1 J per finestra). Le stime
  possono sbagliare di circa 2×.
- Il modello è stato addestrato su clip centrate sui richiami; qui si usano finestre fisse da 3 s, quindi sul campo
  l'accuratezza può essere più bassa. Non c'è ancora mitigazione dei falsi positivi.
- Il vecchio deployment "KWS" simulato, che non eseguiva la rete addestrata, è conservato solo nel tag git `pre-cleanup`.

## Licenza

MIT, vedi [LICENSE](LICENSE). La libreria in `vendor/audiomoth` è di Open Acoustic Devices.
