#include "matrix.hpp"
#include <cassert>
#include <print>

namespace math {
    Matrix::Matrix(const size_t n, const size_t m) : _n(n), _m(m) {
        _rows = std::vector<Vector>(n);
        _cols = std::vector<Vector>(m);
        for (size_t i = 0; i < std::max(n, m); i++) {
            if (i < n) {
                _rows.emplace_back(m);
            }
            if (i < m) {
                _cols.emplace_back(n);
            }
        }
    }

    Matrix::Matrix(size_t n, size_t m, float value) : _n(n), _m(m) {
        _rows = std::vector<Vector>(n);
        _cols = std::vector<Vector>(m);
        for (size_t i = 0; i < std::max(n, m); i++) {
            if (i < n) {
                _rows.emplace_back(m, value);
            }
            if (i < m) {
                _cols.emplace_back(n, value);
            }
        }
    }

    Matrix::Matrix(size_t n, size_t m, const std::vector<Vector> &data) : _n(n), _m(m) {
        assert((data.size() == n || data.size() == m) && "Can not construct Matrix with these rows/cols");
        if (data.size() == n) {
            _rows = data;
            _cols = std::vector<Vector>(m);
            for (size_t i = 0; i < _m; i++) {
                Vector col = Vector(n);
                for (size_t j = 0; j < _n; j++) {
                    assert(_rows[j].size() == _m && "Constructor input has row sizes different from N");
                    col[j] = _rows[j][i];
                }
                _cols[i] = col;
            }
        } else {
            _rows = std::vector<Vector>(n);
            _cols = data;
            for (size_t i = 0; i < _n; i++) {
                Vector row = Vector(m);
                for (size_t j = 0; j < _m; j++) {
                    assert(_cols[j].size() == _n && "Constructor input has column sizes different from M");
                    row[j] = _cols[j][i];
                }
                _rows[i] = row;
            }
        }
    }

    Matrix::Matrix(const std::vector<Vector> &rows, const std::vector<Vector> &cols) : _rows(rows), _cols(cols),
        _n(rows.size()), _m(cols.size()) {
    }

    std::tuple<size_t, size_t> Matrix::shape() const {
        return std::make_tuple(_n, _m);
    }

    const Vector &Matrix::operator[](size_t index) const {
        return _rows[index];
    }

    Vector &Matrix::operator[](size_t index) {
        return _rows[index];
    }

    Matrix Matrix::operator+() const {
        return {_rows, _cols};
    }

    Matrix Matrix::operator-() const {
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = -_rows[i];
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::operator+(const Matrix &other) const {
        assert(shape() == other.shape() && "Adding matrices of different shapes!");
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i] + other._rows[i];
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::operator-(const Matrix &other) const {
        assert(shape() == other.shape() && "Subtracting matrices of different shapes!");
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i] - other._rows[i];
        }
        return {_n, _m, new_data};
    }

    Vector Matrix::operator*(const Vector &other) const {
        assert(_m == other.size() && "Multiplying with Vector of wrong size!");
        std::vector<float> new_data = std::vector<float>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i] * other;
        }
        return Vector(new_data);
    }

    Vector Matrix::operator/(const Vector &other) const {
        assert(_m == other.size() && "Dividing by Vector of wrong size!");
        std::vector<float> new_data = std::vector<float>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i] / other;
        }
        return Vector(new_data);
    }

    Matrix Matrix::operator*(const Matrix &other) const {
        assert(_m == other._n && "Multiplying matrices of different shapes!");
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            std::vector<float> new_row = std::vector<float>(other._m);
            for (size_t j = 0; j < other._m; j++) {
                new_row[j] = _rows[i] * other._cols[j];
            }
            new_data[i] = Vector(new_row);
        }
        return {_n, other._m, new_data};
    }

    Matrix Matrix::operator/(const Matrix &other) const {
        assert(_m == other._n && "Dividing matrices of different shapes!");
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            std::vector<float> new_row = std::vector<float>(other._m);
            for (size_t j = 0; j < _m; j++) {
                new_row[j] = _rows[j] / other._cols[j];
            }
            new_data[i] = Vector(new_row);
        }
        return {_n, other._m, new_data};
    }

    Matrix Matrix::operator+(const float &other) const {
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i] + other;
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::operator-(const float &other) const {
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i] - other;
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::operator*(const float &other) const {
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i] * other;
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::operator/(const float &other) const {
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i] / other;
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::T() const {
        return {_cols, _rows};
    }

    bool Matrix::symmetric() const {
        if (_n != _m) {
            return false;
        }

        for (size_t i = 0; i < _n; i++) {
            if (_rows[i] != _cols[i]) {
                return false;
            }
        }
        return true;
    }

    bool Matrix::operator==(const Matrix &other) const {
        if (shape() != other.shape()) {
            return false;
        }
        for (size_t i = 0; i < _n; i++) {
            if (_rows[i] != other._rows[i]) {
                return false;
            }
        }
        return true;
    }

    void Matrix::print() const {
        for (size_t i = 0; i < _n; i++) {
            _rows[i].print();
        }
    }
} // namespace math
