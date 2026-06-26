#include <iostream>
#include <initializer_list>
#include <algorithm>
#include <experimental/simd>
#include <thread>
#include <vector>
#include <pthread.h>

// Código feito para verificar se as classes matrizes funcionavam.

using namespace std;
namespace stdx = std::experimental;

template <typename T>
class Matrix {

    private:
    size_t r; // Número de linhas (rows)
    size_t c; // Número de colunas (columns)
    
    T* d; 

    public:

    // Estrutura para passar os argumentos para as threads do pthreads.
    struct ThreadDataSIMD {
        const Matrix* mat_a;
        const Matrix* mat_o_t; // Matriz B transposta
        Matrix* res;           // Onde o resultado será gravado
        size_t linha_inicio;
        size_t linha_fim;
        size_t cols_a;
        size_t cols_o;
    };

    // Estrutur para a paralelisação do tranpose da multiplicação.
    struct ThreadTranspose {
        const Matrix* origem;
        Matrix* destino;
        size_t linha_inicio;
        size_t linha_fim;
    };

    size_t column() const { return this->c; }
    size_t row() const { return this->r; }

    // Construtor básico: inicializa a matriz com zeros.
    Matrix(size_t i, size_t j) {
        this->r = i;
        this->c = j;
        d = new T[r * c];

        size_t sum = r * c;
        for(size_t k = 0; k < sum; k++) {
            d[k] = 0;
        }
    }

    // Construtor usando initializer_list
    Matrix(size_t i, size_t j, initializer_list<T> list) {
        this->r = i;
        this->c = j;
        d = new T[r * c];
        
        size_t k = 0;
        for (const T& val : list) {
            if (k >= r * c) break; // Proteção contra listas maiores que a matriz
            d[k] = val;
            k++;
        }
        
        // Preenche o resto com zero se a lista for menor que o tamanho da matriz
        while (k < r * c) {
            d[k] = 0;
            k++;
        }
    }

    // Construtor a partir de um array C-style preexistente.
    Matrix(size_t i, size_t j, const T* data_array) {
        this->r = i;
        this->c = j;
        d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = data_array[k];
        }
    }

    // Construtor de cópia
    Matrix(const Matrix& o) {
        this->r = o.r;
        this->c = o.c;
        this->d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = o.d[k];
        }
    }

    // Destrutor.
    ~Matrix() {
        delete[] d;
    }

    // Sobrecarga do operador ()
    T& operator()(size_t i, size_t j) {
        return d[j + (this->c * i)];
    }

    // Versão const
    const T& operator()(size_t i, size_t j) const {
        return d[j + (this->c * i)];
    }

    // Operador de atribuição
    Matrix& operator=(const Matrix& o) {
        // Evita auto-atribuição (ex: matriz = matriz;)
        if (this == &o) { 
            return *this;
        }

        delete[] d; // Limpa os dados antigos

        r = o.r;
        c = o.c;
        d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = o.d[k];
        }
        return *this;
    }

    // Soma de matrizes elemento a elemento
    Matrix operator+(const Matrix& o) const {
        Matrix res(r, c);
        for (size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) + o(i, j);
            }
        }
        return res;
    }

    // Soma escalar (matriz + um número único)
    Matrix operator+(T v) const {
        Matrix res(r, c);
        for (size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) + v;
            }
        }
        return res;
    }

    // Subtração de matrizes
    Matrix operator-(const Matrix& o) const {
        Matrix res(r, c);
        for (size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) - o(i, j);
            }
        }
        return res;
    }

    // Subtração escalar
    Matrix operator-(T v) const {
        Matrix res(r, c);
        for (size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) - v;
            }
        }
        return res;
    }

    //Rotina para usar no Pthread.
    static void* rotina_transpose(void* argumento) {
    ThreadTranspose* dados = static_cast<ThreadTranspose*>(argumento);

    for (size_t i = dados->linha_inicio; i < dados->linha_fim; i++) {
        for (size_t j = 0; j < dados->origem->column(); j++) {
            (*dados->destino)(j, i) = (*dados->origem)(i, j);
        }
    }

    return nullptr;
}
    //Rotina para usar no Pthread.
    static void* rotina_simd(void* argumento) {
        ThreadDataSIMD* dados = static_cast<ThreadDataSIMD*>(argumento);
        
        constexpr size_t BLOCK = 256;
        constexpr size_t SUB_BLOCK = 16; 
        
        using simd_t = stdx::native_simd<T>;
        constexpr size_t SIMD_WIDTH = simd_t::size();

        // Loop Tiling
        for (size_t i_b = dados->linha_inicio; i_b < dados->linha_fim; i_b += BLOCK) {
            for (size_t j_b = 0; j_b < dados->cols_o; j_b += BLOCK) {
                for (size_t k_b = 0; k_b < dados->cols_a; k_b += BLOCK) {
                    
                    // Limites seguros para não estourar o tamanho da matriz
                    size_t i_b_max = min(i_b + BLOCK, dados->linha_fim);
                    size_t j_b_max = min(j_b + BLOCK, dados->cols_o);
                    size_t k_b_max = min(k_b + BLOCK, dados->cols_a);


                    for (size_t i_sb = i_b; i_sb < i_b_max; i_sb += SUB_BLOCK) {
                        for (size_t j_sb = j_b; j_sb < j_b_max; j_sb += SUB_BLOCK) {
                            for (size_t k_sb = k_b; k_sb < k_b_max; k_sb += SUB_BLOCK) {
                                
                                size_t i_max = min(i_sb + SUB_BLOCK, i_b_max);
                                size_t j_max = min(j_sb + SUB_BLOCK, j_b_max);
                                size_t k_max = min(k_sb + SUB_BLOCK, k_b_max);

                                // Multiplicação
                                for (size_t i = i_sb; i < i_max; i++) {
                                    for (size_t j = j_sb; j < j_max; j++) {
                                        
                                        simd_t sum_vec = 0;
                                        size_t k = k_sb;

                                        // Vetorização (SIMD): processa vários elementos em um único ciclo de clock.
                                        for (; k + SIMD_WIDTH <= k_max; k += SIMD_WIDTH) {
                                            simd_t a_vec(&((*dados->mat_a)(i, k)), stdx::element_aligned);
                                            simd_t b_vec(&((*dados->mat_o_t)(j, k)), stdx::element_aligned);
                                            sum_vec += a_vec * b_vec; 
                                        }

                                        // Soma todo vetor em um único elemento.
                                        T scalar_sum = stdx::reduce(sum_vec);

                                        // Tratamento do resto
                                        for (; k < k_max; k++) {
                                            scalar_sum += (*dados->mat_a)(i, k) * (*dados->mat_o_t)(j, k);
                                        }

                                        (*dados->res)(i, j) += scalar_sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return nullptr;
    }

    // Realizao Transpose da matriz.
    static Matrix transpose_pthreads(const Matrix& m) {
        // Função separada da matriz para melhorar a visualização de código da multiplicação. 
        Matrix t(m.column(), m.row());

        size_t num_threads = 3; 

        vector<pthread_t> threads(num_threads);
        vector<ThreadTranspose> dados(num_threads);

        size_t linhas_por_thread = m.row() / num_threads;
        size_t linhas_restantes = m.row() % num_threads;
        size_t linha_atual = 0;

        for (size_t i = 0; i < num_threads; i++) {
            dados[i].origem = &m;
            dados[i].destino = &t;

            dados[i].linha_inicio = linha_atual;
            linha_atual += linhas_por_thread + (i < linhas_restantes ? 1 : 0);
            dados[i].linha_fim = linha_atual;

            pthread_create(&threads[i], nullptr, rotina_transpose, &dados[i]);
        }

        for (size_t i = 0; i < num_threads; i++) {
            pthread_join(threads[i], nullptr);
        }

        return t;
    }

    // Operador de Multiplicação de Matrizes
    Matrix operator*(const Matrix& o) const {
        Matrix res(r, o.column());

        // Transpoem Matriz B
        Matrix o_t = transpose_pthreads(o);

        // Divide o trabalho com base no número de núcleos físicos/lógicos do CPU
        size_t num_threads = 3; // Feito só para o meu computador
        vector<pthread_t> threads(num_threads);
        vector<ThreadDataSIMD> dados_das_threads(num_threads);

        size_t linhas_por_thread = r / num_threads;
        size_t linhas_restantes = r % num_threads;
        size_t linha_atual = 0;

        // Dispara as threads
        for (int i = 0; i < num_threads; ++i) {
            dados_das_threads[i].mat_a = this;
            dados_das_threads[i].mat_o_t = &o_t; // Passamos a versão transposta
            dados_das_threads[i].res = &res;
            dados_das_threads[i].cols_a = c;
            dados_das_threads[i].cols_o = o.column();
            
            // Lógica para distribuir o resto da divisão de forma justa caso 
            // o número de linhas não seja divisível pelo número de threads.
            dados_das_threads[i].linha_inicio = linha_atual;
            linha_atual += linhas_por_thread + (i < linhas_restantes ? 1 : 0);
            dados_das_threads[i].linha_fim = linha_atual;

            pthread_create(&threads[i], nullptr, rotina_simd, &dados_das_threads[i]);
        }

        // Aguarda todas as threads finalizarem antes de devolver o resultado
        for (int i = 0; i < num_threads; ++i) {
            pthread_join(threads[i], nullptr);
        }

        return res;
    }
    
    // Multiplicação por um valor escalar
    Matrix operator*(T v) const {
        Matrix res(r, c);
        for(size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) * v;
            }
        }
        return res;
    }

    // Transforma a matriz atual em sua própria transposta
    void transpose() {
        Matrix n(c, r);
        for(size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                n(j, i) = (*this)(i, j);
            }
        }
        (*this) = n; 
    }

    // Transforma a matriz atual em uma Matriz Identidade
    void I() {
        for(size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                if(i == j) {
                    (*this)(i, j) = 1; // Diagonal principal
                } else {
                    (*this)(i, j) = 0; // Restante
                }
            }
        }
    }

    void operator*=(const Matrix& o) { (*this) = (*this) * o; }
    void operator*=(T v) { (*this) = (*this) * v; }
    void operator+=(const Matrix& o) { (*this) = (*this) + o; }
    void operator+=(T v) { (*this) = (*this) + v; }
    void operator-=(const Matrix& o) { (*this) = (*this) - o; }
    void operator-=(T v) { (*this) = (*this) - v; }
};
