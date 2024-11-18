#pragma once
#include "vector.hpp"
#include <vector>
#include <tuple>

namespace math {
    class Matrix {
        std::vector<Vector> _rows;
        std::vector<Vector> _cols;
        size_t _n;
        size_t _m;

    public:
        // constructors
        Matrix(size_t n, size_t m);

        Matrix(size_t n, size_t m, float value);

        Matrix(size_t n, size_t m, const std::vector<Vector> &data);

        Matrix(const std::vector<Vector> &rows, const std::vector<Vector> &cols);

        // array operations
        std::tuple<size_t, size_t> shape() const;

        const Vector &operator[](size_t index) const;

        Vector &operator[](size_t index);

        // arithmetic operators
        Matrix operator+() const;

        Matrix operator-() const;

        Matrix operator+(const Matrix &other) const;

        Matrix operator-(const Matrix &other) const;

        Vector operator*(const Vector &other) const;

        Vector operator/(const Vector &other) const;

        Matrix operator*(const Matrix &other) const;

        Matrix operator/(const Matrix &other) const;

        Matrix operator+(const float &other) const;

        Matrix operator-(const float &other) const;

        Matrix operator*(const float &other) const;

        Matrix operator/(const float &other) const;

        Matrix T() const;

        // relational operators
        bool symmetric() const;

        bool operator==(const Matrix &other) const;

        // other
        void print() const;
    };
} // namespace math
