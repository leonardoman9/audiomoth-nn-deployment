#!/bin/bash
echo "🚀 Building AudioMoth NN..."
cd ..
make -C build clean
make -C build
echo "✅ Build completato!"
