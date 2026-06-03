#include <iostream>
#include <initializer_list>
#include <algorithm>
#include <experimental/simd>

using namespace std;
namespace stdx = std::experimental;

template <typename T>
class Matrix {

    private:
    size_t r;
    size_t c;
    T* d;

    // Row major layout.

    public:
    size_t column() const {
        return this->c;
    }

    size_t row() const {
        return this->r;
    }

    Matrix(size_t i, size_t j) {
        this->r = i;
        this->c = j;
        d = new T[r * c];

        size_t sum = r * c;

        for(size_t k = 0; k < sum; k++) {
            d[k] = 0;
        }
    }

    Matrix(size_t i, size_t j, initializer_list<T> list) {
        this->r = i;
        this->c = j;
        d = new T[r * c];
        
        size_t k = 0;
        for (const T& val : list) {
            if (k >= r * c) {
                break;
            }
            d[k] = val;
            k++;
        }
        
        while (k < r * c) {
            d[k] = 0;
            k++;
        }
    }

    Matrix(size_t i, size_t j, const T* data_array) {
        this->r = i;
        this->c = j;
        d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = data_array[k];
        }
    }

    Matrix(const Matrix& o) {
        this->r = o.r;
        this->c = o.c;
        this->d = new T[r * c];

        for (size_t k = 0; k < r * c; ++k) {
            d[k] = o.d[k];
        }
    }

    ~Matrix() {
        delete[] d;
    }

    T& operator()(size_t i, size_t j) {
        return d[j + (this->c * i)];
    }

    const T& operator()(size_t i, size_t j) const {
        return d[j + (this->c * i)];
    }

    Matrix& operator=(const Matrix& o) {
        if (this == &o) { 
            return *this;
        }

        delete[] d;

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

    void transpose() {
        Matrix n(c, r);

        for(size_t i = 0; i < r; i++) {
            for(size_t j = 0; j < c; j++) {
                n(j, i) = (*this)(i, j);
            }
        }
        (*this) = n;
    }

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

    void operator*=(const Matrix& o) {
        (*this) = (*this) * o;
    }

    void operator*=(T v) {
        (*this) = (*this) * v;
    }

    void operator+=(const Matrix& o) {
        (*this) = (*this) + o;
    }

    void operator+=(T v) {
        (*this) = (*this) + v;
    }

    void operator-=(const Matrix& o) {
        (*this) =  (*this) - o;
    }

    void operator-=(T v) {
        (*this) = (*this) - v;
    }
};