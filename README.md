# 🎯 AudioMoth Neural Network Model Conversion Pipeline

Questo repository contiene gli strumenti per convertire modelli PyTorch in TensorFlow Lite quantizzati int8 per il deployment su AudioMoth.

## 📋 **Panoramica**

Il processo di conversione segue questa pipeline:
```
PyTorch Model → ONNX → TensorFlow SavedModel → TensorFlow Lite int8
```

Il sistema è progettato per modelli a due stadi:
- **Backbone CNN**: Estrazione features da spettrogrammi audio
- **Streaming GRU**: Elaborazione sequenziale per classificazione

## 🚀 **Quick Start**

### 1. Setup Ambiente

```bash
# Crea virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Installa dipendenze
pip install torch torchvision torchaudio
pip install onnx tensorflow tf_keras
pip install ai-edge-litert tensorflow-probability
```

### 2. Conversione Completa

```bash
cd streamable-kws

# Step 1: PyTorch → ONNX (se necessario)
python export_onnx.py model_path=/path/to/model.pth +output_path=logs/export_custom

# Step 2: ONNX → TFLite int8 (processo completo)
./run_conversion.sh
source ../.venv/bin/activate
TF_LITE_DISABLE_XNNPACK=1 python3 onnx_to_int8_simple.py
```

## 📖 **Guida Dettagliata**

### **STEP 1: PyTorch → ONNX**

Il file `export_onnx.py` converte i modelli PyTorch in formato ONNX:

```bash
cd streamable-kws
python export_onnx.py
```

**Output generati:**
- `logs/export_*/cnn_backbone_simplified.onnx` - Backbone CNN
- `logs/export_*/streaming_processor_simplified.onnx` - Streaming GRU

**Configurazione modello:**
- Input backbone: `[1, 40, 18]` (batch, mels, time)  
- Output backbone: `[1, 18, 32]` (batch, time, features)
- Input streaming: `[1, 32]` feature + `[1, 32]` hidden state
- Output streaming: `[1, 5]` logits + `[1, 32]` new hidden state

### **STEP 2: ONNX → TFLite int8**

Il processo di conversione avviene in due fasi:

#### **Fase A: ONNX → SavedModel (Docker)**

```bash
./run_conversion.sh
```

**Cosa fa:**
- Utilizza Docker con ambiente Linux controllato
- Converte ONNX in SavedModel usando `onnx2tf`
- Prova 3 strategie diverse per ogni modello
- Genera SavedModel in `tflite_models/`

**File generati:**
- `tflite_models/backbone_model_attempt2/` - SavedModel backbone
- `tflite_models/streaming_model_attempt2/` - SavedModel streaming

#### **Fase B: SavedModel → TFLite int8 (Python)**

```bash
source .venv/bin/activate
TF_LITE_DISABLE_XNNPACK=1 python onnx_to_int8_simple.py
```

**Cosa fa:**
- Carica i SavedModel generati dal Docker
- Applica quantizzazione int8 usando il metodo del tutor
- Genera representative dataset per calibrazione
- Produce i modelli finali ottimizzati

**File finali:**
- `backbone_int8_FINAL.tflite` (53.3 KB)
- `streaming_int8_FINAL.tflite` (21.0 KB)

### **STEP 3: Verifica Quantizzazione**

```bash
python verify_int8_quantization.py
```

**Output esempio:**
```
✅ backbone_int8_FINAL.tflite: 54,592 bytes (53.3 KB)
✅ streaming_int8_FINAL.tflite: 21,552 bytes (21.0 KB)
📊 Backbone: 100% int8 quantization
📊 Streaming: 76.6% int8 quantization  
```

## 🔧 **Configurazione Avanzata**

### **Docker Settings** (`Dockerfile`)

```dockerfile
FROM python:3.10-slim
RUN pip install tensorflow==2.15.0 tf_keras onnx==1.15.0 onnx2tf
```

### **Quantizzazione Settings** (`onnx_to_int8_simple.py`)

```python
# Configurazione TFLite Converter
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8

# Representative dataset per calibrazione
def representative_dataset():
    for _ in range(100):
        # Backbone: spettrogramma dummy
        yield [np.random.rand(1, 18, 40).astype(np.float32)]
        
        # Streaming: feature + hidden state dummy  
        yield [np.random.rand(1, 32).astype(np.float32),
               np.zeros((1, 32), dtype=np.float32)]
```

## 📊 **Specifiche Modelli**

### **Backbone CNN**
- **Input**: Spettrogramma `[1, 18, 40]` (batch, time, mels)
- **Output**: Features `[1, 18, 32]` (batch, time, features)
- **Dimensione**: 53.3 KB (int8)
- **Architettura**: CNN con residual connections + attention

### **Streaming GRU**  
- **Input**: Features `[1, 32]` + Hidden State `[1, 32]`
- **Output**: Logits `[1, 5]` + New Hidden `[1, 32]`
- **Dimensione**: 21.0 KB (int8)
- **Architettura**: GRU + attention + fully connected

### **Totale Sistema**
- **Memoria richiesta**: 74.4 KB (29% di 256KB AudioMoth)
- **Classi supportate**: 5 categorie audio
- **Quantizzazione**: int8 per efficienza embedded

## 🛠️ **Troubleshooting**

### **Errore XNNPACK**
```bash
# Disabilita XNNPACK durante conversione
export TF_LITE_DISABLE_XNNPACK=1
```

### **Errore Docker Permission**
```bash
# Su macOS, assicurati che Docker Desktop sia in esecuzione
docker --version
```

### **Errore Dependencies**
```bash
# Reinstalla dipendenze in virtual environment pulito
rm -rf .venv
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install tensorflow==2.15.0 tf_keras onnx==1.15.0
```

## 📂 **Struttura File**

```
convert-and-quantize/
├── README.md                          # Questa guida
├── streamable-kws/
│   ├── export_onnx.py                 # PyTorch → ONNX
│   ├── run_conversion.sh              # Docker orchestrator  
│   ├── convert_models.sh              # Script Docker interno
│   ├── Dockerfile                     # Ambiente Docker
│   ├── onnx_to_int8_simple.py         # SavedModel → TFLite int8
│   ├── verify_int8_quantization.py    # Verifica quantizzazione
│   ├── backbone_int8_FINAL.tflite     # ✅ Modello finale backbone
│   ├── streaming_int8_FINAL.tflite    # ✅ Modello finale streaming
│   └── logs/                          # File ONNX generati
└── .venv/                             # Virtual environment Python
```

## ✅ **Checklist Conversione**

- [ ] Virtual environment attivato
- [ ] Dipendenze installate correttamente  
- [ ] File ONNX presenti in `logs/`
- [ ] Docker in esecuzione
- [ ] `run_conversion.sh` completato con successo
- [ ] SavedModel generati in `tflite_models/`
- [ ] `onnx_to_int8_simple.py` eseguito senza errori
- [ ] Solo 2 file `*_FINAL.tflite` presenti
- [ ] Quantizzazione verificata con `verify_int8_quantization.py`

## 🎯 **Prossimi Passi**

1. **Conversione C Arrays**: `xxd -i model.tflite > model.h`
2. **Integrazione AudioMoth**: Include nei file firmware
3. **TensorFlow Lite Micro**: Setup runtime embedded
4. **Testing Hardware**: Deploy su AudioMoth fisico

---

## 📚 **Riferimenti**

- [TensorFlow Lite Micro Guide](https://www.tensorflow.org/lite/microcontrollers)
- [AudioMoth Documentation](https://github.com/OpenAcousticDevices/AudioMoth-Project)
- [ONNX to TFLite Conversion](https://github.com/PINTO0309/onnx2tf)
- [Int8 Quantization Guide](https://www.tensorflow.org/lite/performance/post_training_quantization)

**Creato per il progetto AudioMoth Neural Network Deployment** 🦎🎵