#!/usr/bin/env python3
"""
Analisi completa dei modelli TFLite per AudioMoth NN deployment
"""

import tensorflow as tf
import numpy as np
import json
import os

def analyze_tflite_model(tflite_path, model_name):
    """Analizza un modello TFLite e ne estrae la topologia completa"""
    print(f"\n{'='*50}")
    print(f"📊 ANALISI MODELLO: {model_name}")
    print(f"File: {tflite_path}")
    print(f"{'='*50}")
    
    # Carica l'interprete
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    # Input/Output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"\n🔍 INPUT TENSORS:")
    for i, inp in enumerate(input_details):
        print(f"  [{i}] Name: {inp['name']}")
        print(f"      Shape: {inp['shape']}")
        print(f"      Type: {inp['dtype']}")
        if 'quantization_parameters' in inp:
            quant = inp['quantization_parameters']
            print(f"      Quantization: scale={quant['scales']}, zero_point={quant['zero_points']}")
    
    print(f"\n🎯 OUTPUT TENSORS:")
    for i, out in enumerate(output_details):
        print(f"  [{i}] Name: {out['name']}")
        print(f"      Shape: {out['shape']}")
        print(f"      Type: {out['dtype']}")
        if 'quantization_parameters' in out:
            quant = out['quantization_parameters']
            print(f"      Quantization: scale={quant['scales']}, zero_point={quant['zero_points']}")
    
    # Estrai dettagli sui tensori interni
    print(f"\n🏗️  TENSOR DETAILS (tutti i tensori interni):")
    tensor_details = interpreter.get_tensor_details()
    
    analysis = {
        'model_name': model_name,
        'file_path': tflite_path,
        'input_details': input_details,
        'output_details': output_details,
        'tensor_details': tensor_details,
        'layers': []
    }
    
    for i, tensor in enumerate(tensor_details):
        print(f"  Tensor[{i:2d}]: {tensor['name']:40} Shape:{str(tensor['shape']):15} Type:{str(tensor['dtype']):15}")
        
        # Estrai quantization parameters se presenti
        if 'quantization_parameters' in tensor:
            quant = tensor['quantization_parameters'] 
            if quant['scales'].size > 0:
                print(f"              Quant: scale={quant['scales']} zero_point={quant['zero_points']}")
    
    # Cerca di estrarre informazioni sui layer (operazioni)
    print(f"\n⚙️  OPERATIONS/LAYERS:")
    
    # Get model flatbuffer
    with open(tflite_path, 'rb') as f:
        model_data = f.read()
    
    model = tf.lite.schema_py_generated.Model.Model.GetRootAs(model_data, 0)
    subgraph = model.Subgraphs(0)
    
    print(f"   Numero operatori: {subgraph.OperatorsLength()}")
    
    # Estrai informazioni sui layer
    builtin_op_names = [
        'ADD', 'AVERAGE_POOL_2D', 'CONCATENATION', 'CONV_2D', 'DEPTHWISE_CONV_2D',
        'DEQUANTIZE', 'EMBEDDING_LOOKUP', 'FLOOR', 'FULLY_CONNECTED', 'HASHTABLE_LOOKUP',
        'L2_NORMALIZATION', 'L2_POOL_2D', 'LOCAL_RESPONSE_NORMALIZATION', 'LOGISTIC',
        'LSH_PROJECTION', 'LSTM', 'MAX_POOL_2D', 'MUL', 'RELU', 'RELU_N1_TO_1', 'RELU6',
        'RESHAPE', 'RESIZE_BILINEAR', 'RNN', 'SOFTMAX', 'SPACE_TO_DEPTH', 'SVDF',
        'TANH', 'CONCAT_EMBEDDINGS', 'SKIP_GRAM', 'CALL', 'CUSTOM', 'EMBEDDING_LOOKUP_SPARSE',
        'PAD', 'UNIDIRECTIONAL_SEQUENCE_RNN', 'GATHER', 'BATCH_TO_SPACE_ND', 'SPACE_TO_BATCH_ND',
        'TRANSPOSE', 'MEAN', 'SUB', 'DIV', 'SQUEEZE', 'UNIDIRECTIONAL_SEQUENCE_LSTM',
        'STRIDED_SLICE', 'BIDIRECTIONAL_SEQUENCE_RNN', 'EXP', 'TOPK_V2', 'SPLIT',
        'LOG_SOFTMAX', 'DELEGATE', 'BIDIRECTIONAL_SEQUENCE_LSTM', 'CAST', 'PRELU',
        'MAXIMUM', 'ARG_MAX', 'MINIMUM', 'LESS', 'NEG', 'PADV2', 'GREATER',
        'GREATER_EQUAL', 'LESS_EQUAL', 'SELECT', 'SLICE', 'SIN', 'TRANSPOSE_CONV',
        'SPARSE_TO_DENSE', 'TILE', 'EXPAND_DIMS', 'EQUAL', 'NOT_EQUAL', 'LOG',
        'SUM', 'SQRT', 'RSQRT', 'SHAPE', 'POW', 'ARG_MIN', 'FAKE_QUANT',
        'REDUCE_PROD', 'REDUCE_MAX', 'PACK', 'LOGICAL_OR', 'ONE_HOT', 'LOGICAL_AND',
        'LOGICAL_NOT', 'UNPACK', 'REDUCE_MIN', 'FLOOR_DIV', 'REDUCE_ANY', 'SQUARE',
        'ZEROS_LIKE', 'FILL', 'FLOOR_MOD', 'RANGE', 'RESIZE_NEAREST_NEIGHBOR',
        'LEAKY_RELU', 'SQUARED_DIFFERENCE', 'MIRROR_PAD', 'ABS', 'SPLIT_V',
        'UNIQUE', 'CEIL', 'REVERSE_V2', 'ADD_N', 'GATHER_ND', 'COS', 'WHERE',
        'RANK', 'ELU', 'REVERSE_SEQUENCE', 'MATRIX_DIAG', 'QUANTIZE', 'MATRIX_SET_DIAG',
        'ROUND', 'HARD_SWISH', 'IF', 'WHILE', 'NON_MAX_SUPPRESSION_V4', 'NON_MAX_SUPPRESSION_V5',
        'SCATTER_ND', 'SELECT_V2', 'DENSIFY', 'SEGMENT_SUM', 'BATCH_MATMUL'
    ]
    
    for i in range(subgraph.OperatorsLength()):
        op = subgraph.Operators(i)
        opcode_idx = op.OpcodeIndex()
        opcode = model.OperatorCodes(opcode_idx)
        
        # Get operation name
        if opcode.BuiltinCode() < len(builtin_op_names):
            op_name = builtin_op_names[opcode.BuiltinCode()]
        else:
            op_name = f"CUSTOM_{opcode.BuiltinCode()}"
        
        # Get input/output tensors for this operation
        inputs = []
        for j in range(op.InputsLength()):
            inputs.append(op.Inputs(j))
        
        outputs = []
        for j in range(op.OutputsLength()):
            outputs.append(op.Outputs(j))
        
        layer_info = {
            'op_index': i,
            'op_name': op_name,
            'input_tensors': inputs,
            'output_tensors': outputs
        }
        analysis['layers'].append(layer_info)
        
        print(f"   Op[{i:2d}]: {op_name:20} Inputs:{inputs} Outputs:{outputs}")
        
        # Stampa dettagli tensori input/output
        for inp_idx in inputs:
            if inp_idx >= 0:  # Può essere -1 per optional tensors
                inp_tensor = tensor_details[inp_idx]
                print(f"        Input[{inp_idx}]:  {inp_tensor['name']:30} {inp_tensor['shape']}")
        
        for out_idx in outputs:
            if out_idx >= 0:
                out_tensor = tensor_details[out_idx]
                print(f"        Output[{out_idx}]: {out_tensor['name']:30} {out_tensor['shape']}")
    
    return analysis

def compare_with_python_model():
    """Confronta con il modello Python fornito"""
    print(f"\n{'='*60}")
    print(f"🧠 CONFRONTO CON MODELLO PYTHON ORIGINALE")
    print(f"{'='*60}")
    
    print("""
    MODELLO PYTHON - Improved_Phi_GRU_ATT:
    
    1. INPUT: Audio waveform [batch, time_samples]
    
    2. SPECTROGRAM PROCESSING:
       - Tipo: mel/linear_stft/linear_triangular/combined_log_linear
       - n_fft=400, hop_length=160
       - Output: spettrogramma [batch, freq, time]
       - AmplitudeToDB conversion
       - Normalizzazione (mean=0, std=1)
    
    3. BACKBONE (MatchboxNet):
       - Input: [batch, features, time] 
       - MatchboxNetSkip con configurazione matchbox_cfg
       - Output: [batch, base_filters=32, time]
    
    4. SEQUENCE PROCESSING:
       - Permute: [batch, time, features]
       - GRU: input_size=32, hidden_size=32, bidirectional=False
       - Projection: Linear(32, 32)
       - AttentionLayer(32)
       - FC: Linear(32, num_classes)
    
    5. OUTPUT: [batch, num_classes] logits
    """)

def main():
    """Analisi completa"""
    print("🚀 ANALISI TOPOLOGIA MODELLI AUDIOMOTH-NN")
    
    # Percorsi dei modelli TFLite
    backbone_path = "convert-and-quantize/streamable-kws/backbone_int8_FINAL.tflite"
    streaming_path = "convert-and-quantize/streamable-kws/streaming_int8_FINAL.tflite"
    
    # Verifica esistenza file
    if not os.path.exists(backbone_path):
        print(f"❌ File backbone non trovato: {backbone_path}")
        return
    
    if not os.path.exists(streaming_path):
        print(f"❌ File streaming non trovato: {streaming_path}")
        return
    
    # Analizza entrambi i modelli
    backbone_analysis = analyze_tflite_model(backbone_path, "BACKBONE")
    streaming_analysis = analyze_tflite_model(streaming_path, "STREAMING")
    
    # Confronto con modello Python
    compare_with_python_model()
    
    # Salva analisi su file JSON
    full_analysis = {
        'backbone': backbone_analysis,
        'streaming': streaming_analysis,
        'python_model_info': {
            'class_name': 'Improved_Phi_GRU_ATT',
            'components': [
                'Spectrogram Processing',
                'MatchboxNetSkip (Backbone)',
                'GRU + Attention (Streaming)',
                'Classification Head'
            ]
        }
    }
    
    with open('model_analysis.json', 'w') as f:
        json.dump(full_analysis, f, indent=2, default=str)
    
    print(f"\n✅ Analisi completa salvata in 'model_analysis.json'")
    
    # Summary della differenza CRITICA
    print(f"\n🎯 DIFFERENZE CRITICHE TROVATE:")
    print(f"   - Backbone TFLite: Input shape probabilmente diverso da [18,40,1]")
    print(f"   - Streaming TFLite: Potrebbe avere più layer del semplice GRU")
    print(f"   - Quantizzazione: Parametri scale/zero_point specifici per ogni tensore")
    print(f"   - Layer order: Sequenza esatta potrebbe differire dall'implementazione attuale")

if __name__ == "__main__":
    main()