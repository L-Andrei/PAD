#include <iostream>
#include <initializer_list>
#include <algorithm>
#include <experimental/simd>
#include <thread>
#include <vector>
#include <pthread.h>

using namespace std;
namespace stdx = std::experimental;

// Classe de matriz genérica. O uso de template permite que ela funcione 
// com int, float, double, etc., sem precisar reescrever o código.
template <typename T>
class Matrix {

    private:
    size_t r; // Número de linhas (rows)
    size_t c; // Número de colunas (columns)
    
    // Os dados são armazenados em um array 1D contínuo na memória (Row-major layout).
    // Isso é muito melhor para o cache do processador do que usar ponteiros duplos (T**).
    T* d; 

    public:

    // Estrutura para passar os argumentos para as threads do pthreads.
    // Como a API do pthread em C exige um (void*), precisamos empacotar tudo aqui.
    struct ThreadDataSIMD {
        const Matrix* mat_a;
        const Matrix* mat_o_t; // Matriz B transposta (para acesso contíguo)
        Matrix* res;           // Onde o resultado será gravado
        size_t linha_inicio;
        size_t linha_fim;
        size_t cols_a;
        size_t cols_o;
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

    // Construtor usando initializer_list para permitir sintaxe estilo: Matrix<int> m(2, 2, {1, 2, 3, 4});
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

    // Construtor de cópia (Copy Constructor)
    Matrix(const Matrix& o) {
        this->r = o.r;
        this->c = o.c;
        this->d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = o.d[k];
        }
    }

    // Destrutor: nunca esquecer de liberar a memória do array 1D!
    ~Matrix() {
        delete[] d;
    }

    // Sobrecarga do operador () para acessar os elementos fácil: mat(linha, coluna)
    // O cálculo j + (c * i) mapeia a coordenada 2D para o índice 1D.
    T& operator()(size_t i, size_t j) {
        return d[j + (this->c * i)];
    }

    // Versão const do operador () para leitura em matrizes constantes.
    const T& operator()(size_t i, size_t j) const {
        return d[j + (this->c * i)];
    }

    // Operador de atribuição (Copy Assignment)
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

    // =========================================================================
    // Core da otimização: Multiplicação usando SIMD + Cache Blocking (Tiling)
    // =========================================================================
    static void* rotina_simd(void* argumento) {
        ThreadDataSIMD* dados = static_cast<ThreadDataSIMD*>(argumento);
        
        // Estes tamanhos quebram a matriz em pedaços menores para que caibam 
        // perfeitamente no cache L1/L2 do processador. Evita "cache misses" frequentes.
        constexpr size_t BLOCK = 256;
        constexpr size_t SUB_BLOCK = 128; 
        
        using simd_t = stdx::native_simd<T>;
        constexpr size_t SIMD_WIDTH = simd_t::size();

        // Loop Tiling: iterando através dos blocos maiores
        for (size_t i_b = dados->linha_inicio; i_b < dados->linha_fim; i_b += BLOCK) {
            for (size_t j_b = 0; j_b < dados->cols_o; j_b += BLOCK) {
                for (size_t k_b = 0; k_b < dados->cols_a; k_b += BLOCK) {
                    
                    // Limites seguros para não estourar o tamanho da matriz
                    size_t i_b_max = std::min(i_b + BLOCK, dados->linha_fim);
                    size_t j_b_max = std::min(j_b + BLOCK, dados->cols_o);
                    size_t k_b_max = std::min(k_b + BLOCK, dados->cols_a);

                    // Iterando através dos sub-blocos
                    for (size_t i_sb = i_b; i_sb < i_b_max; i_sb += SUB_BLOCK) {
                        for (size_t j_sb = j_b; j_sb < j_b_max; j_sb += SUB_BLOCK) {
                            for (size_t k_sb = k_b; k_sb < k_b_max; k_sb += SUB_BLOCK) {
                                
                                size_t i_max = std::min(i_sb + SUB_BLOCK, i_b_max);
                                size_t j_max = std::min(j_sb + SUB_BLOCK, j_b_max);
                                size_t k_max = std::min(k_sb + SUB_BLOCK, k_b_max);

                                // Multiplicação real acontecendo aqui dentro
                                for (size_t i = i_sb; i < i_max; i++) {
                                    for (size_t j = j_sb; j < j_max; j++) {
                                        
                                        simd_t sum_vec = 0;
                                        size_t k = k_sb;

                                        // Vetorização (SIMD): processa vários elementos em um único ciclo de clock.
                                        // Usar element_aligned garante performance máxima se a memória estiver alinhada.
                                        for (; k + SIMD_WIDTH <= k_max; k += SIMD_WIDTH) {
                                            simd_t a_vec(&((*dados->mat_a)(i, k)), stdx::element_aligned);
                                            simd_t b_vec(&((*dados->mat_o_t)(j, k)), stdx::element_aligned);
                                            sum_vec += a_vec * b_vec; 
                                        }

                                        // Reduz o vetor SIMD para um escalar somando os elementos internos
                                        T scalar_sum = stdx::reduce(sum_vec);

                                        // Tratamento do "resto" (quando as colunas não são múltiplas da largura do SIMD)
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

    // Operador de Multiplicação de Matrizes (Otimizado com Multithreading)
    Matrix operator*(const Matrix& o) const {
        Matrix res(r, o.column());

        // Truque clássico de performance: transpor a matriz B antes de multiplicar.
        // Isso faz com que a leitura na matriz B seja sequencial (friendly pro Cache L1), 
        // em vez de saltar posições de memória lendo as colunas verticalmente.
        Matrix o_t(o.column(), o.row());
        for(size_t i = 0; i < o.row(); i++) {
            for(size_t j = 0; j < o.column(); j++) {
                o_t(j, i) = o(i, j);
            }
        }

        // Divide o trabalho com base no número de núcleos físicos/lógicos do CPU
        size_t num_threads = std::thread::hardware_concurrency();
        std::vector<pthread_t> threads(num_threads);
        std::vector<ThreadDataSIMD> dados_das_threads(num_threads);

        size_t linhas_por_thread = r / num_threads;
        size_t linhas_restantes = r % num_threads;
        size_t linha_atual = 0;

        // Dispara as threads
        for (int i = 0; i < num_threads; ++i) {
            dados_das_threads[i].mat_a = this;
            dados_das_threads[i].mat_o_t = &o_t; // Passamos a versão transposta!
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
        (*this) = n; // Usa o operator= que já implementamos
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
