#!/bin/bash

ARQUIVO_CSV="resultados_benchmark.csv"
TOTAL_EXECUCOES=30

CORES_PERMITIDOS=$(lscpu -p=cpu,core | grep -v '^#' | awk -F, '!seen[$2]++ {print $1}' | paste -sd, -)

echo "Desativando o Turbo Boost da CPU..."
echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost > /dev/null

echo "codigo_ingenuo,opemMP,otimizacao_sequencial,pThread" > "$ARQUIVO_CSV"

for (( i=1; i<=TOTAL_EXECUCOES; i++ ))
do
    cd matrizes/codigo_ingenuo || exit
    g++ -Wall -Wextra -std=c++20 -O3 main.cpp -o executar_testes
    tempo_ingenuo=$(sudo nice -n -20 taskset -c $CORES_PERMITIDOS ./executar_testes)
    cd ../..

    cd matrizes/opemMP || exit
    g++ -Wall -Wextra -std=c++20 -O3 -fopenmp -mavx2 -mfma main.cpp -o executar_testes
    tempo_openmp=$(sudo nice -n -20 taskset -c $CORES_PERMITIDOS ./executar_testes)
    cd ../..

    cd matrizes/otimizacao_sequencial || exit
    g++ -O3 -mavx2 -mfma -march=native main.cpp -o executar_testes
    tempo_seq=$(sudo nice -n -20 taskset -c $CORES_PERMITIDOS ./executar_testes)
    cd ../..

    cd matrizes/pThread || exit
    g++ -Wall -Wextra -std=c++20 -O3 -pthread -mavx2 -mfma main.cpp -o executar_testes
    tempo_pthread=$(sudo nice -n -20 taskset -c $CORES_PERMITIDOS ./executar_testes)
    cd ../..

    echo "$tempo_ingenuo,$tempo_openmp,$tempo_seq,$tempo_pthread" >> "$ARQUIVO_CSV"
    
    echo "Rodada $i de $TOTAL_EXECUCOES concluída com sucesso."
done

echo "Reativando o Turbo Boost da CPU..."
echo 1 | sudo tee /sys/devices/system/cpu/cpufreq/boost > /dev/null

echo "Processo finalizado. Os dados foram salvos no arquivo $ARQUIVO_CSV."