#include <iostream>
#include <initializer_list>
#include <algorithm>
#include <experimental/simd> 

using namespace std;
namespace stdx = std::experimental;

template <typename T>
class Matrix {

    private:
    size_t r; // Linhas (rows)
    size_t c; // Colunas (columns)
    
    // Matriz armazenada internamente como um array 1D (Row-major layout).
    T* d;

    public:
    size_t column() const { return this->c; }
    size_t row() const { return this->r; }

    // Construtor básico: aloca memória e preenche com zeros
    Matrix(size_t i, size_t j) {
        this->r = i;
        this->c = j;
        d = new T[r * c];

        size_t sum = r * c;
        for(size_t k = 0; k < sum; k++) {
            d[k] = 0;
        }
    }

    // Construtor via lista de inicialização (ex: Matrix<int> m(2,2, {1,2,3,4}))
    Matrix(size_t i, size_t j, initializer_list<T> list) {
        this->r = i;
        this->c = j;
        d = new T[r * c];
        
        size_t k = 0;
        for (const T& val : list) {
            if (k >= r * c) break; // Prevenção contra listas maiores que a matriz
            d[k] = val;
            k++;
        }
        
        while (k < r * c) {
            d[k] = 0;
            k++;
        }
    }

    // Construtor a partir de um array C-style
    Matrix(size_t i, size_t j, const T* data_array) {
        this->r = i;
        this->c = j;
        d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = data_array[k];
        }
    }

    // Construtor de cópia: essencial para evitar que duas matrizes apontem para a mesma memória
    Matrix(const Matrix& o) {
        this->r = o.r;
        this->c = o.c;
        this->d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = o.d[k];
        }
    }

    // Destrutor: libera a memória alocada dinamicamente
    ~Matrix() {
        delete[] d;
    }

    // Mapeia a coordenada 2D (linha, coluna) para o índice 1D do array
    T& operator()(size_t i, size_t j) {
        return d[j + (this->c * i)];
    }

    // Versão constante do mapeamento para leitura
    const T& operator()(size_t i, size_t j) const {
        return d[j + (this->c * i)];
    }

    // Operador de atribuição com proteção contra auto-atribuição
    Matrix& operator=(const Matrix& o) {
        if (this == &o) { 
            return *this;
        }

        delete[] d; // Limpa o estado atual antes de receber a cópia

        r = o.r;
        c = o.c;
        d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = o.d[k];
        }
        return *this;
    }

    Matrix operator+(const Matrix& o) const {
        Matrix res(r, c);
        for (size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) + o(i, j);
            }
        }
        return res;
    }

    Matrix operator+(T v) const {
        Matrix res(r, c);
        for (size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) + v;
            }
        }
        return res;
    }

    Matrix operator-(const Matrix& o) const {
        Matrix res(r, c);
        for (size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) - o(i, j);
            }
        }
        return res;
    }

    Matrix operator-(T v) const {
        Matrix res(r, c);
        for (size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) - v;
            }
        }
        return res;
    }

    // Algoritmo clássico de multiplicação de matrizes cúbica O(N^3)
    Matrix operator*(const Matrix& o) const {
        Matrix res(r, o.column());

        for (size_t i = 0; i < r; ++i) {
            for (size_t j = 0; j < o.column(); ++j) {
                for (size_t k = 0; k < c; ++k) {
                    res(i, j) += (*this)(i, k) * o(k, j);
                }
            }
        }

        return res;
    }
    
    Matrix operator*(T v) const {
        Matrix res(r, c);
        for(size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) * v;
            }
        }
        return res;
    }

    // Transposição criando uma nova matriz e realocando os eixos
    void transpose() {
        Matrix n(c, r);
        for(size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                n(j, i) = (*this)(i, j);
            }
        }
        (*this) = n; // Reutiliza o operador de atribuição
    }

    // Transforma a matriz na Identidade correspondente
    void I() {
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
