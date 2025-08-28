#!/bin/bash
# Wrapper for SS5 build - AudioMoth specific
cd ..

# Cerca 'all' tra tutti gli argomenti
if [[ "$*" == *"all"* ]]; then
    # Se c'è 'all', buildare il default target (rimuovi 'all' dagli argomenti)
    args=$(echo "$*" | sed 's/all//g')
    exec /usr/bin/make -C build $args
else
    exec /usr/bin/make -C build "$@"
fi
