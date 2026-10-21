# Applicazione "bird logger" per AudioMoth

Ogni giorno alle **07:00** (orologio del dispositivo) per **5 minuti**: registra finestre da 3 s a
32 kHz, le classifica con il modello di uccelli e aggiunge una riga a `LOG.TXT` sulla SD:

    2026-10-21 07:00:03,Certhia brachydactyla,55,Y     (data ora, classe, probabilita' %, Y = >= 50%)

Fuori dalla finestra il dispositivo e' spento (power-down, sveglia a ore). L'orologio si imposta con
`CONFIG.TXT` sulla SD (una riga `YYYY-MM-DD hh:mm:ss`), che viene poi riscritto come `APPLIED <ora>`.
Niente USB, GPS, scrittura WAV, app di configurazione. Orario e durata: `SCHED_WINDOW_*` in
`schedule.h`. L'ordine delle classi e' quello degli INDICI DI USCITA del modello (ordine di `allowed_bird_classes` nella configurazione di training), non quello della riga "Class names" del log: vedi `docs/VERIFICATION.md` (`kClassNames` in `schedule.c`).

File: `main_app.c` (colla hardware, segue `makeRecording` dell'originale), `schedule.c` (logica pura,
verificata su 1.805 casi (`tests/schedule/`) contro `datetime` di Python), `fatfs_conf/` (FatFS ridotto), `build_app.sh`
(firmware), `tests/sim/` (simulatore su computer).

## Firmware (misurato, GCC 14.3.1 `-Os`)

    firmware/app/build_app.sh TFLM_ROOT bird_backbone_int8.tflite     # TFLM da scripts/build_tflm.sh

- Flash: **230.124 B** su 245.760 disponibili (margine ~15,6 KB; vedi `docs/VERIFICATION.md`). Modello 94,7 KB, TFLM+CMSIS-NN
  ~57 KB, coda C 27,6 KB, libc/libm ~23 KB, FatFS+SD 7,4 KB, emlib 6,7 KB, audiomoth.c 5 KB.
- FatFS ridotto (CP437, nomi 8.3, **niente exFAT**: usare una scheda FAT32): evita ~60 KB di tabelle.
  Il firmware AudioMoth completo originale (USB, GPS, filtri, FatFS standard) occupa ~150 KB.
- RAM interna: .data 192 B + .bss 14.112 B (+ heap 3 KB, stack in coda). SRAM esterna: 182.576 B su 262.144.

## Simulatore (`tests/sim/`)

Esegue `main_app.c` e la pipeline vera (front-end, backbone TFLM, coda, `digitalfilter.c`) su un modello
software dell'hardware (`hal_sim.c`): RTC e SD persistono tra gli avvii (ogni processo = un ciclo di
accensione), il microfono legge un file audio (campioni a 384 kHz ricostruiti per il vero filtro di
decimazione).

    tests/sim/build_sim.sh TFLM_ROOT bird_backbone_int8.tflite
    python tests/sim/make_audio.py audio.raw <file o cartelle audio>
    python tests/sim/run_scenario.py tests/sim/out/bird_logger_sim audio.raw WORKDIR 2
    python tests/sim/verify_log.py WRENNET_ROOT best_distillation_model.pt WORKDIR

Risultato (2 giorni simulati, orologio impostato a 06:58:30 da CONFIG.TXT): 28 avvii, 49 righe di log
al giorno (una ogni 6 s, da 07:00:03 a 07:04:55), nessun errore; il resto del giorno solo risvegli
orari. Sulle stesse 49 finestre PyTorch concorda sulla classe in 48 (il cambio ha confidenza 0,28);
nomi di classe e percentuali nelle righe di log coincidono con i logit del dispositivo.

**Cosa NON modella**: temporizzazione ADC/DMA, parte analogica del microfono, driver SD reale, deriva
dell'RTC, consumi, tempi con SRAM esterna, spegnimento reale. Il decimatore dell'AudioMoth somma 12
campioni (media a blocchi), un anti-alias piu' povero del ricampionamento usato nel training
(a 16 kHz circa -4 dB): differenza da tenere presente sul campo.
