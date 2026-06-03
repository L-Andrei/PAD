#include <iostream>
#include <initializer_list>
#include <algorithm>
#include <experimental/simd>

using namespace std;
// Usamos um alias para facilitar a digitação, já que a biblioteca SIMD ainda está na namespace experimental
namespace stdx = std::experimental;

// Transformar a classe em um Template permite que a mesma lógica funcione 
// para matrizes de int, float, double, etc., sem precisarmos reescrever o código.
template <typename T>
class Matrix {

    private:
    size_t r; // Quantidade de linhas (rows)
    size_t c; // Quantidade de colunas (columns)
    
    // Os dados são armazenados em um array 1D contínuo na memória (Row-major layout).
    // Por que não usar ponteiros duplos (T**)? Porque arrays contínuos são muito mais 
    // amigáveis para o cache do processador, evitando saltos desnecessários de memória.
    T* d;

    public:
    
    // Getters simples para as dimensões
    size_t column() const { return this->c; }
    size_t row() const { return this->r; }

    // Construtor padrão: aloca a memória e inicializa a matriz inteira com zeros
    Matrix(size_t i, size_t j) {
        this->r = i;
        this->c = j;
        d = new T[r * c];

        size_t sum = r * c;

        for(size_t k = 0; k < sum; k++) {
            d[k] = 0;
        }
    }

    // Construtor com initializer_list. Permite criar a matriz usando uma sintaxe super limpa.
    // Exemplo: Matrix<int> m(2, 2, {1, 2, 3, 4});
    Matrix(size_t i, size_t j, initializer_list<T> list) {
        this->r = i;
        this->c = j;
        d = new T[r * c];
        
        size_t k = 0;
        for (const T& val : list) {
            // Prevenção caso passem mais elementos do que o tamanho da matriz suporta
            if (k >= r * c) {
                break;
            }
            d[k] = val;
            k++;
        }
        
        // Se passarem menos elementos, preenchemos o resto com zeros por segurança
        while (k < r * c) {
            d[k] = 0;
            k++;
        }
    }

    // Construtor que copia os dados de um array "C-style" cru já existente
    Matrix(size_t i, size_t j, const T* data_array) {
        this->r = i;
        this->c = j;
        d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = data_array[k];
        }
    }

    // Construtor de Cópia. Extremamente necessário quando lidamos com ponteiros crus (T* d).
    // Isso evita que duas matrizes apontem para o mesmo endereço de memória.
    Matrix(const Matrix& o) {
        this->r = o.r;
        this->c = o.c;
        this->d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = o.d[k];
        }
    }

    // Destrutor. A regra de ouro do C++: se você deu 'new[]', você é obrigado a dar 'delete[]'.
    ~Matrix() {
        delete[] d;
    }

    // Sobrecarga do operador () para acessar elementos facilmente: mat(linha, coluna).
    // A fórmula 'j + (c * i)' converte as coordenadas 2D para o índice exato no array 1D.
    T& operator()(size_t i, size_t j) {
        return d[j + (this->c * i)];
    }

    // Versão 'const' do acesso de elementos, para quando passamos a matriz como referência constante.
    const T& operator()(size_t i, size_t j) const {
        return d[j + (this->c * i)];
    }

    // Operador de Atribuição (matrizA = matrizB). Também exige cuidado manual com a memória.
    Matrix& operator=(const Matrix& o) {
        // Proteção contra auto-atribuição (alguém fazendo mat = mat)
        if (this == &o) { 
            return *this;
        }

        delete[] d; // Limpamos a memória atual antes de receber os dados novos

        r = o.r;
        c = o.c;
        d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = o.d[k];
        }
        return *this;
    }

    // =========================================================================
    // Operações Matemáticas Básicas Elemento-a-Elemento
    // =========================================================================
    
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

    // =========================================================================
    // Core da Otimização: Multiplicação com Tiling e SIMD
    // =========================================================================
    Matrix operator*(const Matrix& o) const {
        Matrix res(r, o.column());

        // Tamanhos dos blocos para o Cache Tiling.
        // O objetivo é quebrar a matriz em pedaços que caibam inteiros no cache L1/L2 do CPU.
        // Se a CPU não precisar buscar dados na RAM a todo momento, a velocidade decola.
        constexpr size_t BLOCK = 256;
        constexpr size_t SUB_BLOCK = 128; 
        
        // Configurando os registradores vetoriais para processar múltiplos elementos por vez.
        using simd_t = stdx::native_simd<T>;
        constexpr size_t SIMD_WIDTH = simd_t::size();

        // Truque clássico: Transpor a segunda matriz antes de começar.
        // Ao transpor, passamos a acessar a memória de forma contínua (horizontal) na matriz o_t,
        // o que o hardware prefetcher da CPU adora. Acessar colunas verticalmente destruiria a performance.
        Matrix o_t(o.column(), o.row());
        for(size_t i = 0; i < o.row(); i++) {
            for(size_t j = 0; j < o.column(); j++) {
                o_t(j, i) = o(i, j);
            }
        }

        // Loop Tiling: Navegando pelos Blocos Principais
        for (size_t i_b = 0; i_b < r; i_b += BLOCK) {
            for (size_t j_b = 0; j_b < o.column(); j_b += BLOCK) {
                for (size_t k_b = 0; k_b < c; k_b += BLOCK) {
                    
                    // Garantindo que não vamos tentar ler fora da matriz
                    size_t i_b_max = min(i_b + BLOCK, r);
                    size_t j_b_max = min(j_b + BLOCK, o.column());
                    size_t k_b_max = min(k_b + BLOCK, c);

                    // Loop Tiling: Navegando pelos Sub-Blocos (refinando ainda mais o cache)
                    for (size_t i_sb = i_b; i_sb < i_b_max; i_sb += SUB_BLOCK) {
                        for (size_t j_sb = j_b; j_sb < j_b_max; j_sb += SUB_BLOCK) {
                            for (size_t k_sb = k_b; k_sb < k_b_max; k_sb += SUB_BLOCK) {
                                
                                size_t i_max = min(i_sb + SUB_BLOCK, i_b_max);
                                size_t j_max = min(j_sb + SUB_BLOCK, j_b_max);
                                size_t k_max = min(k_sb + SUB_BLOCK, k_b_max);

                                // Multiplicação de fato
                                for (size_t i = i_sb; i < i_max; i++) {
                                    for (size_t j = j_sb; j < j_max; j++) {
                                        
                                        simd_t sum_vec = 0;
                                        size_t k = k_sb;

                                        // Aqui a mágica do SIMD acontece. Em vez de multiplicar um por um,
                                        // pegamos pacotes de dados (SIMD_WIDTH) e multiplicamos no mesmo ciclo de clock.
                                        for (; k + SIMD_WIDTH <= k_max; k += SIMD_WIDTH) {
                                            // 'element_aligned' avisa ao compilador que a memória tá alinhadinha, garantindo máxima velocidade.
                                            simd_t a_vec(&((*this)(i, k)), stdx::element_aligned);
                                            simd_t b_vec(&(o_t(j, k)), stdx::element_aligned);
                                            sum_vec += a_vec * b_vec; 
                                        }

                                        // Pega o vetor de somas e junta tudo num único número escalar
                                        T scalar_sum = stdx::reduce(sum_vec);

                                        // Tratamento do resto. Se a quantidade de elementos não for um múltiplo exato 
                                        // do tamanho do nosso pacote SIMD, fazemos o restinho do jeito tradicional.
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
    
    // Multiplicação por um valor escalar (Matriz * Número)
    Matrix operator*(T v) const {
        Matrix res(r, c);

        for(size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                res(i, j) = (*this)(i, j) * v;
            }
        }
        return res;
    }

    // Inverte as linhas com as colunas
    void transpose() {
        Matrix n(c, r);

        for(size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                n(j, i) = (*this)(i, j);
            }
        }
        (*this) = n; // Reutiliza o nosso operador de atribuição seguro
    }

    // Transforma a matriz atual em uma Matriz Identidade (1 na diagonal principal, 0 no resto)
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