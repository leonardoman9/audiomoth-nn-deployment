# Verifiche del port WrenNet su AudioMoth (21 ottobre 2026)

Scopo: confermare il piu' possibile, senza scheda, cio' che era in dubbio. Per ogni voce: cosa e' stato fatto,
risultato, e cosa resta aperto. Tutto e' ripetibile con gli script indicati. Nulla e' stato eseguito sul chip.

## 1. Configurazione del microfono a 32 kHz
Lo script ufficiale dell'app di configurazione AudioMoth (`constants.js`) usa per 32 kHz: `sampleRate 384000`,
`sampleRateDivider 12`, `clockDivider 4`, `acquisitionCycles 16`, `oversampleRate 1`; guadagno predefinito medio.
**Coincide** con `main_app.c`. L'ADC e' differenziale a 12 bit con riferimento 2,5 V (+-1,25 V, LSB 0,61 mV).

## 2. Watchdog
`WDOG` periodo 64k su ULFRCO 1 kHz = ~65 s (`audiomoth.c`, manuale RM). Il blocco piu' lungo (la rete) dura pochi secondi:
**nessun rischio**; il watchdog e' comunque alimentato nei cicli.

## 3. SRAM esterna: bus a 8 bit con stati di attesa (scoperta che ha cambiato il progetto)
`enableEBI()`: modo D8A8, lettura 2+3+1 cicli, scrittura 6 cicli **per byte**: un accesso a 32 bit costa ~25 cicli
(contro 1 della RAM interna). `enableExternalSRAM()` e `enableFileSystem()` restituiscono false su hardware `AM_VERSION_4`
(USB Microphone: senza SD ne' SRAM); la scheda Dev, che ha entrambe, deve rilevarsi < 4 (non verificabile senza scheda).

Profilo con plugin QEMU (`tools/memcount.c`, conta istruzioni e byte per regione), una finestra da 3 s:

| fase | primo progetto | dopo la riprogettazione |
|---|---|---|
| front-end (301 frame) | 68 M istruzioni, **86 MB** verso SRAM esterna (~13 s: overrun, ogni finestra scartata) | 44,8 M istruzioni, 0,85 MB (~1,3-1,8 s, 45-60% di CPU durante la registrazione) |
| backbone (modello 4D) | 105 M istr., 16,3 MB (attivazioni: ~67 KB nell'arena) | invariato: ~4,9-6,2 s |
| coda in C | 55 M istr., nessun accesso esterno | invariata: ~1,4-2,1 s |

Riprogettazione del front-end: tabelle costanti in Flash (generate da `tools/gen_frontend_tables.c`), buffer di lavoro in
RAM interna (6 KB), FFT reale da 1024 punti come FFT complessa da 512 + passo di separazione; in SRAM esterna restano solo
l'anello di campioni e lo spettrogramma dB (accessi sequenziali). Risultato verificato: <= 1 LSB int8 rispetto a PyTorch.
Stima per finestra (CPI 1,2-1,8 sul conteggio di istruzioni, +ritardi EBI): backbone+coda 6-8 s dopo i 3 s di registrazione:
~27-31 finestre in 5 minuti; energia ~0,8-1,0 J per finestra a 21 mA/4,1 V (non 0,077 J come nel paper). Stime, non misure.

## 4. Stack e RAM interna
Stack massimo misurato in QEMU sull'intera pipeline: **2.708 B** (2,7 KB); statici della pipeline 7.024 B. Applicazione completa:
`.data` 192 + `.bss` 14.112 B (+ heap 3 KB riservato). Margine ampio su 32 KB.

## 5. Flash
Firmware completo (applicazione + classificatore): **230.124 B su 245.760** (margine 15,6 KB), TFLM+CMSIS-NN 53,6 KB (build per dimensione,
patch senza int4 in `patches/`), modello 94,7 KB, coda 27,5 KB, tabelle front-end ~10 KB. Il linker script originale AudioMoth
confronta una LUNGHEZZA con un INDIRIZZO (`__etext`): e' piu' severo di 16 KB del limite fisico; corretto nella copia
(`firmware/audiomoth_tflm.ld`). Il resto del firmware (FatFS, audiomoth.c, emlib, filtri) pesa ~36 KB (build diagnostica senza rete).
Il firmware AudioMoth originale completo era ~150 KB: FatFS con code page 932 (60 KB) e libc/printf (33 KB); qui FatFS e' ridotto
(CP437, nomi 8.3, **niente exFAT: scheda FAT32**).

## 6. FatFS + wrapper del file system AudioMoth (test su immagine FAT32)
`app/fstest/`: FatFS vero con la configurazione ridotta + le funzioni file di `audiomoth.c` copiate alla lettera, su un'immagine
FAT32 da 128 MB. **PASS**: lettura/marcatura di CONFIG.TXT, 120 righe accodate (apri/scrivi/chiudi), file letti correttamente da un
lettore FAT indipendente (mtools). Trovato e corretto un difetto: `f_rename` fallisce se la destinazione esiste (il simulatore lo
nascondeva); ora CONFIG.TXT viene sovrascritto con "APPLIED <ora>". Il driver SPI (`microsd.c`, codice del produttore, non modificato)
non e' stato eseguito: resta da provare sulla scheda.

## 7. Orologio, sonno
EM4 (RAM persa, resta il BURTC): ogni risveglio e' un riavvio, come previsto. Sonno lungo con `powerDownAndWake(s, true)` (la versione
non sincronizzata moltiplicherebbe ms*1024 e traboccherebbe oltre ~70 min), blocchi da 6 h. L'ora si perde se manca l'alimentazione
(datasheet): serve un nuovo CONFIG.TXT. Deriva +-0,89 s/giorno.

## 8. Ordine delle classi (errore trovato, ora confermato su tutte le specie)
L'ordine reale e' quello di `allowed_bird_classes` (0 Bubo bubo, 1 Certhia familiaris, 2 Apus apus, 3 Certhia brachydactyla, 4 Emberiza cia,
5 Lophophanes cristatus, 6 Periparus ater, 7 Poecile montanus, 8 non-bird), non quello del log. Prova diretta con registrazioni vere dal Drive
del progetto (finestre da 3 s, indice con piu' voti per specie vera): Bubo 19/23 -> 0, Certhia fam. 13/14 -> 1, Apus 20/23 -> 2,
Certhia brach. 12/14 -> 3, Emberiza 13/23 -> 4, Lophophanes 10/14 -> 5, Periparus 19/24 -> 6, Poecile 17/24 -> 7: **8 specie su 8**, 76% delle
finestre corrette. Corretto in `schedule.c` e negli strumenti.

## 9. Accuratezza sulle specie vere (dati dal Drive del progetto)
161 finestre da 3 s (le 2 piu' forti per file, fino a 12 file per specie, 7-12 file per specie), tutte e 8 le specie.

| | accuratezza |
|---|---|
| PyTorch float, audio pulito | 76,4% |
| pipeline del dispositivo (front-end C + backbone int8 + coda C), audio pulito | 75,8% |
| PyTorch float, catena analogica simulata 70 / 60 / 50 dB SPL | 73,3% / 73,9% / 73,3% |
| pipeline del dispositivo, catena simulata 70 / 60 / 50 dB SPL | 75,8% / 75,8% / 75,2% |

Per classe (float, pulito): Bubo 0,79; Certhia fam. 0,93; Apus 0,87; Certhia brach. 0,86; Emberiza 0,54; Lophophanes 0,71; Periparus 0,79;
Poecile 0,71. Letture: int8 + C costano <1 punto; la catena analogica simulata costa ~2-3 punti sul float e quasi nulla sulla pipeline int8.
**Limiti:** campione piccolo (+-3-4 punti), finestre scelte per energia e non con l'estrazione dei richiami del training, nessuna finestra
non-bird, e i file del Drive sono verosimilmente anche quelli del training (accuratezza assoluta ottimistica: conta il confronto relativo).
Script: `tools/bird_eval_prepare.py` + `tools/bird_eval_pipeline.py`; scarico da Xeno-canto: `tools/xc_fetch.py` (chiave API).

## 10. Modalita' di prova sul dispositivo
`REPLAY.PCM` ("RPLY" + 96.000 campioni int16 + "END!") sulla SD: il dispositivo lo classifica e scrive `REPLAY,<riga di log con i 9 logit>`;
provato nel simulatore (differenza 0,044 da PyTorch). Build diagnostica `EXTRA_CF=-DBIRD_DIAG_RECORD ./build_app.sh ...`: registra 3 WAV
(`HHMMSS.WAV`) senza rete, 36 KB di Flash; provata nel simulatore (errore 5% sopra 300 Hz rispetto alla sorgente quantizzata a 12 bit).

## 11. Ancora aperto (richiede la scheda o altri dati)
Avvio e abilitazione della SRAM esterna sul chip; cattura DMA/ADC reale e livelli; driver SD reale; tempi e consumi veri (le stime sopra
possono sbagliare di ~2x); accuratezza vera sul test set (dataset su Drive
senza la partizione di test del training: ordine di `os.listdir` non riproducibile); divario di dominio con audio reale del dispositivo.
