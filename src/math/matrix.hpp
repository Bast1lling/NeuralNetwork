#pragma once
#include "vector.hpp"
#include <vector>
#include <tuple>
#include <random>

namespace math {
    class Matrix {
        std::vector<Vector> _rows;
        std::vector<Vector> _cols;
        size_t _n;
        size_t _m;

    public:
        // constructors
        Matrix(std::initializer_list<std::initializer_list<float>> data);

        explicit Matrix(const std::vector<std::vector<float>>& data); // init with rows

        Matrix(size_t n, size_t m);

        Matrix(size_t n, size_t m, float value);

        Matrix(size_t n, size_t m, const std::vector<Vector> &data); // init with rows

        Matrix(const std::vector<Vector> &rows, const std::vector<Vector> &cols);

        // getter
        std::tuple<size_t, size_t> shape() const;

        size_t n() const;

        size_t m() const;

        // array operations
        const Vector &operator[](size_t index) const;

        Vector &operator[](size_t index);

        const Vector &get_col(size_t index) const;

        Vector &get_col(size_t index);

        // arithmetic operators
        Matrix operator+() const;

        Matrix operator-() const;

        Matrix operator+(const Matrix &other) const;

        Matrix operator-(const Matrix &other) const;

        Matrix operator*(const Matrix &other) const;

        Matrix operator/(const Matrix &other) const;

        Matrix operator+(const Vector &other) const;

        Matrix operator-(const Vector &other) const;

        Vector operator*(const Vector &other) const;

        Vector operator/(const Vector &other) const;

        Matrix operator+(const float &other) const;

        Matrix operator-(const float &other) const;

        Matrix operator*(const float &other) const;

        Matrix operator/(const float &other) const;

        Matrix exp() const;

        Matrix inv(float other = 1);

        Matrix had(const Matrix &other) const; // hadamard-product

        Matrix max(const float &other) const;

        Matrix T() const;

        // relational operators
        bool symmetric() const;

        bool operator==(const Matrix &other) const;

        // other
        void print() const;

        static Matrix getRandom(size_t n, size_t m, float std);
    };
} // namespace math
