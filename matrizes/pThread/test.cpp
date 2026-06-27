#include <gtest/gtest.h>
#include <random>
#include "matrix.hpp"

// Código feito para verificar se as classes matrizes funcionavam.

template <typename T>
Matrix<T> simple_multiply(const Matrix<T>& m1, const Matrix<T>& m2) {
    Matrix<T> res(m1.row(), m2.column());
    for (size_t i = 0; i < m1.row(); ++i) {
        for (size_t j = 0; j < m2.column(); ++j) {
            T sum = 0;
            for (size_t k = 0; k < m1.column(); ++k) {
                sum += m1(i, k) * m2(k, j);
            }
            res(i, j) = sum;
        }
    }
    return res;
}

TEST(MatrixTest, ConstructorAndDimensions) {
    Matrix<int> m(3, 4);
    EXPECT_EQ(m.row(), 3);
    EXPECT_EQ(m.column(), 4);

    for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 4; j++) {
            EXPECT_EQ(m(i, j), 0);
        }
    }
}

TEST(MatrixTest, InitializerListConstructor) {
    Matrix<int> m1(2, 2, {1, 2, 3, 4});
    EXPECT_EQ(m1.row(), 2);
    EXPECT_EQ(m1.column(), 2);
    EXPECT_EQ(m1(0, 0), 1);
    EXPECT_EQ(m1(0, 1), 2);
    EXPECT_EQ(m1(1, 0), 3);
    EXPECT_EQ(m1(1, 1), 4);

    Matrix<int> m2(2, 2, {9, 8});
    EXPECT_EQ(m2(0, 0), 9);
    EXPECT_EQ(m2(0, 1), 8);
    EXPECT_EQ(m2(1, 0), 0);
    EXPECT_EQ(m2(1, 1), 0);
}

TEST(MatrixTest, ArrayConstructor) {
    int arr[] = {10, 20, 30, 40, 50, 60};
    Matrix<int> m(2, 3, arr);
    
    EXPECT_EQ(m.row(), 2);
    EXPECT_EQ(m.column(), 3);
    EXPECT_EQ(m(0, 0), 10);
    EXPECT_EQ(m(0, 1), 20);
    EXPECT_EQ(m(0, 2), 30);
    EXPECT_EQ(m(1, 0), 40);
    EXPECT_EQ(m(1, 1), 50);
    EXPECT_EQ(m(1, 2), 60);
}

TEST(MatrixTest, AccessOperator) {
    Matrix<int> m(2, 2);
    m(0, 1) = 5;
    m(1, 0) = 10;

    EXPECT_EQ(m(0, 0), 0);
    EXPECT_EQ(m(0, 1), 5);
    EXPECT_EQ(m(1, 0), 10);
    EXPECT_EQ(m(1, 1), 0);
}

TEST(MatrixTest, CopyConstructor) {
    Matrix<int> original(2, 2);
    original(0, 0) = 1;
    original(1, 1) = 2;

    Matrix<int> copy(original);
    EXPECT_EQ(copy(0, 0), 1);
    EXPECT_EQ(copy(1, 1), 2);

    copy(0, 0) = 99;
    EXPECT_EQ(original(0, 0), 1); 
}

TEST(MatrixTest, AssignmentOperator) {
    Matrix<int> m1(2, 2);
    m1(0, 0) = 7;

    Matrix<int> m2(3, 3);
    m2 = m1;

    EXPECT_EQ(m2.row(), 2);
    EXPECT_EQ(m2.column(), 2);
    EXPECT_EQ(m2(0, 0), 7);

    m2(0, 0) = 42;
    EXPECT_EQ(m1(0, 0), 7);
}

TEST(MatrixTest, AdditionMatrix) {
    Matrix<int> m1(2, 2);
    m1(0, 0) = 1; m1(0, 1) = 2;
    m1(1, 0) = 3; m1(1, 1) = 4;

    Matrix<int> m2(2, 2);
    m2(0, 0) = 5; m2(0, 1) = 6;
    m2(1, 0) = 7; m2(1, 1) = 8;

    Matrix<int> res = m1 + m2;
    EXPECT_EQ(res(0, 0), 6);
    EXPECT_EQ(res(0, 1), 8);
    EXPECT_EQ(res(1, 0), 10);
    EXPECT_EQ(res(1, 1), 12);
}

TEST(MatrixTest, AdditionScalar) {
    Matrix<int> m(2, 2);
    m(0, 0) = 1; m(1, 1) = 2;

    Matrix<int> res = m + 5;
    EXPECT_EQ(res(0, 0), 6);
    EXPECT_EQ(res(0, 1), 5);
    EXPECT_EQ(res(1, 0), 5);
    EXPECT_EQ(res(1, 1), 7);
}

TEST(MatrixTest, SubtractionMatrix) {
    Matrix<int> m1(2, 2);
    m1(0, 0) = 5; m1(0, 1) = 5;
    
    Matrix<int> m2(2, 2);
    m2(0, 0) = 2; m2(0, 1) = 8;

    Matrix<int> res = m1 - m2;
    EXPECT_EQ(res(0, 0), 3);
    EXPECT_EQ(res(0, 1), -3);
}

TEST(MatrixTest, SubtractionScalar) {
    Matrix<int> m(2, 2);
    m(0, 0) = 10;
    
    Matrix<int> res = m - 3;
    EXPECT_EQ(res(0, 0), 7);
    EXPECT_EQ(res(1, 1), -3);
}

TEST(MatrixTest, MultiplicationMatrix) {
    Matrix<int> m1(2, 3);
    m1(0, 0) = 1; m1(0, 1) = 2; m1(0, 2) = 3;
    m1(1, 0) = 4; m1(1, 1) = 5; m1(1, 2) = 6;

    Matrix<int> m2(3, 2);
    m2(0, 0) = 7; m2(0, 1) = 8;
    m2(1, 0) = 9; m2(1, 1) = 10;
    m2(2, 0) = 11; m2(2, 1) = 12;

    Matrix<int> res = m1 * m2;
    
    EXPECT_EQ(res.row(), 2);
    EXPECT_EQ(res.column(), 2);
    EXPECT_EQ(res(0, 0), 58);
    EXPECT_EQ(res(0, 1), 64);
    EXPECT_EQ(res(1, 0), 139);
    EXPECT_EQ(res(1, 1), 154);
}

TEST(MatrixTest, MultiplicationScalar) {
    Matrix<int> m(2, 2);
    m(0, 0) = 2; m(1, 1) = 3;
    
    Matrix<int> res = m * 4;
    EXPECT_EQ(res(0, 0), 8);
    EXPECT_EQ(res(1, 1), 12);
}

TEST(MatrixTest, Transpose) {
    Matrix<int> m(2, 3);
    m(0, 0) = 1; m(0, 1) = 2; m(0, 2) = 3;
    m(1, 0) = 4; m(1, 1) = 5; m(1, 2) = 6;

    m.transpose();

    EXPECT_EQ(m.row(), 3);
    EXPECT_EQ(m.column(), 2);
    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(0, 1), 4);
    EXPECT_EQ(m(1, 0), 2);
    EXPECT_EQ(m(1, 1), 5);
    EXPECT_EQ(m(2, 0), 3);
    EXPECT_EQ(m(2, 1), 6);
}

TEST(MatrixTest, IdentityMatrix) {
    Matrix<int> m(3, 3);
    m.I();

    EXPECT_EQ(m(0, 0), 1);
    EXPECT_EQ(m(1, 1), 1);
    EXPECT_EQ(m(2, 2), 1);
    EXPECT_EQ(m(0, 1), 0);
    EXPECT_EQ(m(2, 0), 0);
}

TEST(MatrixTest, CompoundOperators) {
    Matrix<int> m(2, 2);
    m(0, 0) = 2;
    
    m += 3;
    EXPECT_EQ(m(0, 0), 5);
    
    m -= 1;
    EXPECT_EQ(m(0, 0), 4);
    
    m *= 2;
    EXPECT_EQ(m(0, 0), 8);
    
    Matrix<int> m2(2, 2);
    m2(0, 0) = 2;
    
    m += m2;
    EXPECT_EQ(m(0, 0), 10);
    
    m -= m2;
    EXPECT_EQ(m(0, 0), 8);

    Matrix<int> id(2, 2);
    id.I();
    m *= id;
    EXPECT_EQ(m(0, 0), 8);
}


TEST(MatrixTest, MultiplicationLarge2048_RandomDouble) {
    size_t size = 2048;
    
    Matrix<double> m1(size, size);
    Matrix<double> m2(size, size);

    std::mt19937 rng(42);
    std::uniform_real_distribution<double> dist(-10.0, 10.0);

    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            m1(i, j) = dist(rng);
            m2(i, j) = dist(rng);
        }
    }

    // Executa a multiplicação pesada otimizada (SIMD + Pthreads)
    Matrix<double> res_optimized = m1 * m2;

    EXPECT_EQ(res_optimized.row(), size);
    EXPECT_EQ(res_optimized.column(), size);

    // Executa a multiplicação simples para gerar o gabarito confiável
    // Nota: Esta linha será o gargalo de tempo do seu teste
    Matrix<double> res_expected = simple_multiply(m1, m2);

    // Varre todas as células garantindo que a diferença entre a 
    // versão otimizada e a simples seja mínima (devido ao ponto flutuante)
    for (size_t i = 0; i < size; i++) {
        for (size_t j = 0; j < size; j++) {
            EXPECT_NEAR(res_optimized(i, j), res_expected(i, j), 1e-5);
        }
    }
}