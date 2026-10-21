# GPIO benchmark scripts for Raspberry Pi

- Script: `benchmarks/raspberry/rpi_benchmark_two_markers.py`
- Pin numbering: BCM
- Marker: the GPIO is held HIGH during the measured window, for integration on the oscilloscope

## Requirements
- Python 3 on the Raspberry Pi
- RPi.GPIO: `sudo apt-get update && sudo apt-get install -y python3-pip && pip3 install RPi.GPIO`
- Optional (stability): `echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor`

## Oscilloscope wiring
- Pi GND → scope GND
- CH1 on the marker's BCM pin (23 and 24 by default for the two markers)
- Shunt in series with the USB 5 V line or on GND to measure the Pi's total current; CH2/CH3 across the shunt;
  Math channel = CH2 − CH3

## Usage
Two markers (preprocessing on pin 23, network on pin 24):

```sh
python3 benchmarks/raspberry/rpi_benchmark_two_markers.py \
  --module my_model \
  --pre preprocess \
  --infer run_inference \
  --inputs /path/to/input_wav/ \
  --pin-a 23 --pin-b 24 \
  --loops 100 --warmup 10 \
  --csv rpi_zero2w.csv
```

## Output
- mean/median/std statistics on screen
- CSV with timestamps and durations for integration and analysis

## Energy parser (scope CSV)
- Script: `tools/energy_parser.py`
- Example (marker on CH1, shunt voltage on Math):

  ```sh
  python3 tools/energy_parser.py --csv scope.csv --time-col time --vshunt-col Math --marker-col CH1 --rshunt 1.0 --vdd 3.3 --count 100
  ```

- Results: `E_inf`, `T_inf`, `I_avg`, `P_avg`, with a configurable threshold (`--threshold`)

## C snippet (firmware)
- File: `firmware/snippets/gpio_cycle_marker.c`
- Enables DWT->CYCCNT and toggles a GPIO for measurements aligned with the firmware
