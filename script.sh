#!/bin/bash

ARQUIVO_CSV="resultados_benchmark.csv"
TOTAL_EXECUCOES=30

# Fixando os núcleos permitidos para 0, 2 e 6
CORES_PERMITIDOS="0,2,6"

echo 0 | sudo tee /sys/devices/system/cpu/cpufreq/boost > /dev/null

# Atualizando o cabeçalho do CSV para comportar tempo e GFLOPS de cada teste
echo "tempo_ingenuo,gflops_ingenuo,tempo_openmp,gflops_openmp,tempo_seq,gflops_seq,tempo_pthread,gflops_pthread" > "$ARQUIVO_CSV"

for (( i=1; i<=TOTAL_EXECUCOES; i++ ))
do
    echo "Execução $i de $TOTAL_EXECUCOES..."
    
    cd matrizes/codigo_ingenuo || exit
    g++ -Wall -Wextra -std=c++20 -O3 main.cpp -o executar_testes
    tempo_ingenuo=$(sudo nice -n -20 taskset -c $CORES_PERMITIDOS ./executar_testes)
    gflops_ingenuo=$(echo $tempo_ingenuo | awk '{printf "%.2f", (2 * 4096^3) / ($1 * 1000000000)}')
    cd ../..

    cd matrizes/opemMP || exit
    g++ -Wall -Wextra -std=c++20 -O3 -fopenmp -mavx2 -mfma main.cpp -o executar_testes
    tempo_openmp=$(sudo nice -n -20 taskset -c $CORES_PERMITIDOS ./executar_testes)
    gflops_openmp=$(echo $tempo_openmp | awk '{printf "%.2f", (2 * 4096^3) / ($1 * 1000000000)}')
    cd ../..

    cd matrizes/otimizacao_sequencial || exit
    g++ -O3 -mavx2 -mfma -march=native main.cpp -o executar_testes
    tempo_seq=$(sudo nice -n -20 taskset -c $CORES_PERMITIDOS ./executar_testes)
    gflops_seq=$(echo $tempo_seq | awk '{printf "%.2f", (2 * 4096^3) / ($1 * 1000000000)}')
    cd ../..

    # Assumindo que o diretório do pThread seja matrizes/pThread
    cd matrizes/pThread || exit
    g++ -Wall -Wextra -std=c++20 -O3 -pthread main.cpp -o executar_testes
    tempo_pthread=$(sudo nice -n -20 taskset -c $CORES_PERMITIDOS ./executar_testes)
    gflops_pthread=$(echo $tempo_pthread | awk '{printf "%.2f", (2 * 4096^3) / ($1 * 1000000000)}')
    cd ../..

    # Salvar todos os dados da rodada atual como uma linha no arquivo CSV
    echo "$tempo_ingenuo,$gflops_ingenuo,$tempo_openmp,$gflops_openmp,$tempo_seq,$gflops_seq,$tempo_pthread,$gflops_pthread" >> "$ARQUIVO_CSV"
done


echo 1 | sudo tee /sys/devices/system/cpu/cpufreq/boost > /dev/null

echo "Benchmark finalizado. Resultados salvos em $ARQUIVO_CSV"
