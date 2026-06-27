#include <iostream>
#include <initializer_list>
#include <algorithm>
#include <experimental/simd>
#include <omp.h> 

using namespace std;
// Alias para encurtar a digitação da biblioteca SIMD (que ainda é experimental no C++)
namespace stdx = std::experimental;

template <typename T>
class Matrix {

    private:
    size_t r; // Número de linhas (rows)
    size_t c; // Número de colunas (columns)
    
    // Row-major
    T* d;

    public:
    
    // Getters rápidos para as dimensões
    size_t column() const { return this->c; }
    size_t row() const { return this->r; }

    // Construtor padrão: inicializa a matriz com zeros.
    Matrix(size_t i, size_t j) {
        this->r = i;
        this->c = j;
        d = new T[r * c];

        size_t sum = r * c;

        #pragma omp parallel for
        for(size_t k = 0; k < sum; k++) {
            d[k] = 0;
        }
    }

    // Construtor usando initializer_list (permite sintaxe como: Matrix<int> m(2, 2, {1, 2, 3, 4}))
    Matrix(size_t i, size_t j, initializer_list<T> list) {
        this->r = i;
        this->c = j;
        d = new T[r * c];
        
        size_t k = 0;

        for (const T& val : list) {
            // Prevenção de estouro de memória caso a lista seja maior que a matriz
            if (k >= r * c) {
                break;
            }
            d[k] = val;
            k++;
        }
        
        // Completa com zeros caso a lista seja menor que a capacidade
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

        #pragma omp parallel for
        for (size_t k = 0; k < r * c; ++k) {
            d[k] = data_array[k];
        }
    }

    // Construtor de cópia (fundamental para evitar que duas matrizes apontem para a mesma memória)
    Matrix(const Matrix& o) {
        this->r = o.r;
        this->c = o.c;
        this->d = new T[r * c];

        #pragma omp parallel for
        for (size_t k = 0; k < r * c; ++k) {
            d[k] = o.d[k];
        }
    }

    // Destrutor: limpa o array alocado dinamicamente.
    ~Matrix() {
        delete[] d;
    }

    // Sobrecarga do operador () para acessar elementos via mat(linha, coluna).
    T& operator()(size_t i, size_t j) {
        return d[j + (this->c * i)];
    }

    // Versão const do acesso, garantindo que matrizes constantes possam ser lidas.
    const T& operator()(size_t i, size_t j) const {
        return d[j + (this->c * i)];
    }

    // Operador de atribuição (Copy Assignment)
    Matrix& operator=(const Matrix& o) {
        // Proteção contra auto-atribuição (ex: mat = mat)
        if (this == &o) { 
            return *this;
        }

        delete[] d; // Descarta a memória antiga

        r = o.r;
        c = o.c;
        d = new T[r * c];

        #pragma omp parallel for
        for (size_t k = 0; k < r * c; ++k) {
            d[k] = o.d[k];
        }
        return *this;
    }

    // Soma elemento a elemento
    Matrix operator+(const Matrix& o) const {
        Matrix res(r, c);

        // Paralelisa laços mais extremos
        #pragma omp parallel for
        for (size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) + o(i, j);
            }
        }
        return res;
    }

    // Soma de escalar (Matriz + Número)
    Matrix operator+(T v) const {
        Matrix res(r, c);

        #pragma omp parallel for
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

        #pragma omp parallel for
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

        #pragma omp parallel for
        for (size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) - v;
            }
        }
        return res;
    }

    //Multiplicação de matrizes otimizada
    Matrix operator*(const Matrix& o) const {
        Matrix res(r, o.column());

        // Tamanhos dos blocos para o "Tiling".
        constexpr size_t BLOCK = 256;
        constexpr size_t SUB_BLOCK = 16; // pequeno o suficiente para caber no L1 do meu computador.
        
        // Setup para as instruções SIMD 
        using simd_t = stdx::native_simd<T>;
        constexpr size_t SIMD_WIDTH = simd_t::size();

        // Transpor a Matriz B.
        Matrix o_t(o.column(), o.row());
        #pragma omp parallel for
        for(size_t i = 0; i < o.row(); i++) {
            for(size_t j = 0; j < o.column(); j++) {
                o_t(j, i) = o(i, j);
            }
        }


        #pragma omp parallel for
        for (size_t i_b = 0; i_b < r; i_b += BLOCK) {
            for (size_t j_b = 0; j_b < o.column(); j_b += BLOCK) {
                for (size_t k_b = 0; k_b < c; k_b += BLOCK) {
                    
                    // Limites seguros do bloco principal
                    size_t i_b_max = min(i_b + BLOCK, r);
                    size_t j_b_max = min(j_b + BLOCK, o.column());
                    size_t k_b_max = min(k_b + BLOCK, c);

                    // Refinando o Tiling com os sub-blocos
                    for (size_t i_sb = i_b; i_sb < i_b_max; i_sb += SUB_BLOCK) {
                        for (size_t j_sb = j_b; j_sb < j_b_max; j_sb += SUB_BLOCK) {
                            for (size_t k_sb = k_b; k_sb < k_b_max; k_sb += SUB_BLOCK) {
                                
                                // Limites seguros do bloco secundário.
                                size_t i_max = min(i_sb + SUB_BLOCK, i_b_max);
                                size_t j_max = min(j_sb + SUB_BLOCK, j_b_max);
                                size_t k_max = min(k_sb + SUB_BLOCK, k_b_max);

                                // Multiplicação real acontecendo no núcleo dos blocos
                                for (size_t i = i_sb; i < i_max; i++) {
                                    for (size_t j = j_sb; j < j_max; j++) {
                                        
                                        simd_t sum_vec = 0;
                                        size_t k = k_sb;

                                        // For para realizar as operações SIMD
                                        for (; k + SIMD_WIDTH <= k_max; k += SIMD_WIDTH) {
                                            // 'element_aligned' afirma que a memória está perfeitamente alinhada
                                            simd_t a_vec(&((*this)(i, k)), stdx::element_aligned);
                                            simd_t b_vec(&(o_t(j, k)), stdx::element_aligned);
                                            sum_vec += a_vec * b_vec; 
                                        }

                                        // Soma todo vetor em um único elemento.
                                        T scalar_sum = stdx::reduce(sum_vec);

                                        // Calcula as sobras (desnecessario nessa versão)
                                        for (; k < k_max; k++) {
                                            scalar_sum += (*this)(i, k) * o_t(j, k);
                                        }

                                        res(i, j) += scalar_sum;
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        return res;
    }
    
    // Multiplicação por valor escalar
    Matrix operator*(T v) const {
        Matrix res(r, c);

        #pragma omp parallel for
        for(size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) * v;
            }
        }
        return res;
    }

    // Transposição da própria matriz
    void transpose() {
        Matrix n(c, r);

        #pragma omp parallel for
        for(size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                n(j, i) = (*this)(i, j);
            }
        }
        (*this) = n; 
    }

    // Converte a matriz atual em uma Matriz Identidade
    void I() {
        #pragma omp parallel for
        for(size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                if(i == j) {
                    (*this)(i, j) = 1; 
                } else {
                    (*this)(i, j) = 0; 
                }
            }
        }
    }

    void operator*=(const Matrix& o) { (*this) = (*this) * o; }
    void operator*=(T v) { (*this) = (*this) * v; }
    void operator+=(const Matrix& o) { (*this) = (*this) + o; }
    void operator+=(T v) { (*this) = (*this) + v; }
    void operator-=(const Matrix& o) { (*this) =  (*this) - o; }
    void operator-=(T v) { (*this) = (*this) - v; }
};
