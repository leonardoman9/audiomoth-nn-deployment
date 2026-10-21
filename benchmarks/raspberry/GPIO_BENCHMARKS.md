Benchmark GPIO Scripts for Raspberry Pi

- Scripts: `benchmark_raspberry/rpi_benchmark_gpio.py` and `benchmark_raspberry/rpi_benchmark_two_markers.py`
- Numbering: BCM
- Marker: GPIO set HIGH during the measured window for oscilloscope integration

Requisiti
- Python 3 su Raspberry Pi
- RPi.GPIO: `sudo apt-get update && sudo apt-get install -y python3-pip && pip3 install RPi.GPIO`
- Opzionale (stabilità): `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`

Cablaggio per l’oscilloscopio
- GND del Pi → GND dello scope
- CH1 sul pin BCM del marker (default 18 per singolo; 23 e 24 per due marker)
- Shunt in serie al 5V USB o su GND per corrente totale del Pi; CH2/CH3 ai capi dello shunt; canale Math = CH2–CH3

Esempi d’uso
- Marker singolo (pin 18) che chiama `run_inference(x)` del modulo `my_model` su una cartella di input

  `python3 benchmark_raspberry/rpi_benchmark_gpio.py \
    --module my_model \
    --func run_inference \
    --inputs /path/a/input_wav/ \
    --pin 18 \
    --loops 100 --warmup 10 \
    --csv rpi3b_plus.csv`

- Due marker (preproc sul pin 23, NN sul 24)

  `python3 benchmark_raspberry/rpi_benchmark_two_markers.py \
    --module my_model \
    --pre preprocess \
    --infer run_inference \
    --inputs /path/a/input_wav/ \
    --pin-a 23 --pin-b 24 \
    --loops 100 --warmup 10 \
    --csv rpi_zero2w.csv`

Output
- Statistiche mean/median/std a video
- CSV con timestamp e durate per integrazione/analisi

Parser energetico (CSV scope)
- Script: `tools/energy_parser.py`
- Esempio (marker su CH1, vshunt su Math):

  `python3 tools/energy_parser.py --csv scope.csv --time-col time --vshunt-col Math --marker-col CH1 --rshunt 1.0 --vdd 3.3 --count 100`

- Risultati: `E_inf`, `T_inf`, `I_avg`, `P_avg` con soglia configurabile (`--threshold`)

Snippet C (firmware)
- File: `firmware/snippets/gpio_cycle_marker.c`
- Abilita DWT->CYCCNT e toggla un GPIO per misure allineate a firmware

