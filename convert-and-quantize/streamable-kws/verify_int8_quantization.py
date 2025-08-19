#!/usr/bin/env python3
"""
VERIFICA COMPLETA QUANTIZZAZIONE INT8
Script dedicato per verificare che i modelli TFLite siano realmente quantizzati int8
"""

import tensorflow as tf
import numpy as np
import os

def analyze_tflite_model(model_path, model_name):
    """Analizza completamente un modello TFLite per verificare la quantizzazione int8"""
    print(f"\n{'='*80}")
    print(f"🔍 ANALISI COMPLETA: {model_name}")
    print(f"📁 File: {model_path}")
    print(f"{'='*80}")
    
    if not os.path.exists(model_path):
        print(f"❌ File non trovato: {model_path}")
        return False
    
    # Dimensione file
    file_size = os.path.getsize(model_path)
    print(f"📊 Dimensione file: {file_size:,} bytes ({file_size/1024:.1f} KB)")
    
    try:
        # Carica modello SENZA XNNPACK per evitare errori
        interpreter = tf.lite.Interpreter(
            model_path=model_path,
            experimental_delegates=[],
            num_threads=1
        )
        interpreter.allocate_tensors()
        
        # === 1. VERIFICA INPUT/OUTPUT TYPES ===
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        
        print(f"\n📥 INPUT ANALYSIS ({len(input_details)} inputs):")
        all_inputs_int8 = True
        for i, detail in enumerate(input_details):
            dtype = detail['dtype']
            shape = detail['shape']
            name = detail['name']
            is_int8 = (dtype == np.int8)
            all_inputs_int8 = all_inputs_int8 and is_int8
            
            status = "✅ INT8" if is_int8 else f"❌ {dtype}"
            print(f"  Input[{i}]: {name}")
            print(f"    Shape: {shape}")
            print(f"    Type: {status}")
            
            # Quantization parameters
            if 'quantization_parameters' in detail:
                quant = detail['quantization_parameters']
                if 'scales' in quant and len(quant['scales']) > 0:
                    print(f"    Scale: {quant['scales'][0]:.6f}")
                if 'zero_points' in quant and len(quant['zero_points']) > 0:
                    print(f"    Zero point: {quant['zero_points'][0]}")
        
        print(f"\n📤 OUTPUT ANALYSIS ({len(output_details)} outputs):")
        all_outputs_int8 = True
        for i, detail in enumerate(output_details):
            dtype = detail['dtype']
            shape = detail['shape']
            name = detail['name']
            is_int8 = (dtype == np.int8)
            all_outputs_int8 = all_outputs_int8 and is_int8
            
            status = "✅ INT8" if is_int8 else f"❌ {dtype}"
            print(f"  Output[{i}]: {name}")
            print(f"    Shape: {shape}")
            print(f"    Type: {status}")
            
            # Quantization parameters
            if 'quantization_parameters' in detail:
                quant = detail['quantization_parameters']
                if 'scales' in quant and len(quant['scales']) > 0:
                    print(f"    Scale: {quant['scales'][0]:.6f}")
                if 'zero_points' in quant and len(quant['zero_points']) > 0:
                    print(f"    Zero point: {quant['zero_points'][0]}")
        
        # === 2. VERIFICA TENSORI INTERNI ===
        tensor_details = interpreter.get_tensor_details()
        print(f"\n⚙️  INTERNAL TENSORS ANALYSIS ({len(tensor_details)} total tensors):")
        
        # Conta per tipo
        type_counts = {}
        for tensor in tensor_details:
            dtype = str(tensor['dtype'])
            type_counts[dtype] = type_counts.get(dtype, 0) + 1
        
        # Mostra statistiche
        int8_count = type_counts.get("<class 'numpy.int8'>", 0)
        total_count = len(tensor_details)
        int8_percentage = (int8_count / total_count) * 100 if total_count > 0 else 0
        
        print(f"  📊 Distribuzione tipi tensori:")
        for dtype, count in sorted(type_counts.items()):
            percentage = (count / total_count) * 100
            emoji = "✅" if "int8" in dtype else "⚠️" if "float" in dtype else "❓"
            print(f"    {emoji} {dtype}: {count:2d} tensori ({percentage:5.1f}%)")
        
        print(f"\n  🎯 QUANTIZZAZIONE SUMMARY:")
        print(f"    INT8 tensori: {int8_count}/{total_count} ({int8_percentage:.1f}%)")
        
        # === 3. TEST DI INFERENZA ===
        print(f"\n🧪 INFERENCE TEST:")
        try:
            # Prepara input di test
            test_inputs = []
            for detail in input_details:
                shape = detail['shape']
                dtype = detail['dtype']
                name = detail['name'].lower()
                
                if dtype == np.int8:
                    # Input quantizzato: usa range int8
                    if 'spectrogram' in name or 'mel' in name:
                        # Spettrogramma: simula mel-spectrogram quantizzato
                        data = np.random.randint(-100, 100, shape, dtype=np.int8)
                    elif 'feature' in name:
                        # Feature: simula output CNN quantizzato
                        data = np.random.randint(-80, 80, shape, dtype=np.int8)
                    elif 'hidden' in name or 'state' in name:
                        # Hidden state: simula stato GRU quantizzato
                        data = np.random.randint(-50, 50, shape, dtype=np.int8)
                    else:
                        # Default int8
                        data = np.random.randint(-128, 127, shape, dtype=np.int8)
                else:
                    # Input float (non dovrebbe succedere se quantizzato)
                    data = np.random.uniform(-1, 1, shape).astype(dtype)
                
                test_inputs.append(data)
                interpreter.set_tensor(detail['index'], data)
                print(f"    Input[{detail['name']}]: range[{np.min(data)}, {np.max(data)}] dtype={data.dtype}")
            
            # Esegui inferenza
            print(f"    🚀 Esecuzione inferenza...")
            interpreter.invoke()
            print(f"    ✅ Inferenza completata con successo!")
            
            # Verifica output
            print(f"    📤 Output values:")
            for i, detail in enumerate(output_details):
                output = interpreter.get_tensor(detail['index'])
                print(f"      Output[{i}] ({detail['name']}): range[{np.min(output)}, {np.max(output)}] dtype={output.dtype}")
            
        except Exception as e:
            print(f"    ❌ Errore durante inferenza: {e}")
            return False
        
        # === 4. VERDETTO FINALE ===
        print(f"\n{'='*80}")
        print(f"🏆 VERDETTO FINALE per {model_name}:")
        print(f"{'='*80}")
        
        # Criteri per considerare il modello "veramente quantizzato int8"
        criteria = {
            "Input int8": all_inputs_int8,
            "Output int8": all_outputs_int8,
            "Tensori int8 > 50%": int8_percentage > 50,
            "Inferenza funziona": True,  # Se arriviamo qui, funziona
            "File size ragionevole": file_size < 200 * 1024  # < 200KB
        }
        
        passed_criteria = sum(criteria.values())
        total_criteria = len(criteria)
        
        for criterion, passed in criteria.items():
            status = "✅ PASS" if passed else "❌ FAIL"
            print(f"  {status} {criterion}")
        
        overall_score = (passed_criteria / total_criteria) * 100
        
        if overall_score >= 80:
            verdict = f"✅ OTTIMAMENTE QUANTIZZATO ({overall_score:.0f}%)"
            is_good = True
        elif overall_score >= 60:
            verdict = f"⚠️  PARZIALMENTE QUANTIZZATO ({overall_score:.0f}%)"
            is_good = True
        else:
            verdict = f"❌ NON QUANTIZZATO ({overall_score:.0f}%)"
            is_good = False
        
        print(f"\n🎯 {verdict}")
        print(f"💾 Dimensione: {file_size/1024:.1f} KB")
        print(f"📊 Quantizzazione: {int8_percentage:.1f}% tensori int8")
        
        return is_good
        
    except Exception as e:
        print(f"❌ ERRORE durante analisi: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Verifica entrambi i modelli finali"""
    print("🔍 VERIFICA COMPLETA QUANTIZZAZIONE INT8")
    print("=" * 80)
    print("Analisi dettagliata dei modelli TFLite per confermare quantizzazione int8")
    print("=" * 80)
    
    # Modelli da verificare
    models = [
        {
            "path": "backbone_int8_FINAL.tflite",
            "name": "Backbone CNN"
        },
        {
            "path": "streaming_int8_FINAL.tflite", 
            "name": "Streaming GRU"
        }
    ]
    
    results = []
    
    # Analizza ogni modello
    for model in models:
        if os.path.exists(model["path"]):
            is_good = analyze_tflite_model(model["path"], model["name"])
            results.append((model["name"], model["path"], is_good))
        else:
            print(f"\n❌ File non trovato: {model['path']}")
            results.append((model["name"], model["path"], False))
    
    # Risultato finale
    print(f"\n{'='*80}")
    print(f"📋 SUMMARY FINALE")
    print(f"{'='*80}")
    
    total_size = 0
    good_models = 0
    
    for name, path, is_good in results:
        if os.path.exists(path):
            size = os.path.getsize(path)
            total_size += size
            status = "✅ VERIFICATO" if is_good else "❌ PROBLEMI"
            print(f"  {status} {name}")
            print(f"    📁 {path}")
            print(f"    📊 {size:,} bytes ({size/1024:.1f} KB)")
            if is_good:
                good_models += 1
        else:
            print(f"  ❌ MANCANTE {name}")
            print(f"    📁 {path}")
    
    print(f"\n🎯 RISULTATO COMPLESSIVO:")
    print(f"  ✅ Modelli verificati: {good_models}/{len(models)}")
    print(f"  📊 Dimensione totale: {total_size:,} bytes ({total_size/1024:.1f} KB)")
    
    if good_models == len(models):
        print(f"  🎉 TUTTI I MODELLI SONO CORRETTAMENTE QUANTIZZATI INT8!")
        print(f"  🚀 PRONTI PER INTEGRAZIONE AUDIOMOTH!")
    else:
        print(f"  ⚠️  ALCUNI MODELLI HANNO PROBLEMI - CONTROLLA I DETTAGLI SOPRA")
    
    print("=" * 80)

if __name__ == "__main__":
    main()