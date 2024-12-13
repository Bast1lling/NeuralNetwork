#include <nn/math/matrix.hpp>
#include <cassert>
#include <print>
namespace {
    std::vector<std::vector<float>> convertToVector(
        const std::initializer_list<std::initializer_list<float>> init) {
        std::vector<std::vector<float>> result;
        for (const auto& row : init) {
            result.emplace_back(row); // Construct vector<float> from initializer_list<float>
        }
        return result;
    }
} // namespace
namespace math {
    Matrix::Matrix(const std::initializer_list<std::initializer_list<float>> data) : Matrix(convertToVector(data)) {
    }

    Matrix::Matrix(const std::vector<std::vector<float>>& data) : _n(data.size()), _m(data[0].size()) {
        _rows = std::vector<Vector>(_n);
        _cols = std::vector<Vector>(_m);

        for (size_t i = 0; i < _n; i++) {
            _rows[i] = Vector(data[i]);
        }

        for (size_t i = 0; i < _m; i++) {
            std::vector<float> col = std::vector<float>(_n);
            for (size_t j = 0; j < _n; j++) {
                col[j] = _rows[j][i];
            }
            _cols[i] = Vector(col);
        }
    }

    Matrix::Matrix(const Vector &data) : _n(data.size()), _m(1) {
        _rows = std::vector<Vector>(_n);
        _cols = {data};
        for (size_t i = 0; i < _n; i++) {
            _rows[i] = Vector(1, data[i]);
        }
    }

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
                _rows[i] = Vector(m, value);
            }
            if (i < m) {
                _cols[i] = Vector(n, value);
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

    size_t Matrix::n() const {
        return _n;
    }

    size_t Matrix::m() const {
        return _m;
    }

    const Vector &Matrix::operator[](size_t index) const {
        assert(index < _rows.size() && "Accessing non-existing row in Matrix!");
        return _rows[index];
    }

    Vector &Matrix::operator[](size_t index) {
        assert(index < _rows.size() && "Accessing non-existing row in Matrix!");
        return _rows[index];
    }

    const Vector & Matrix::get_col(size_t index) const {
        assert(index < _cols.size() && "Accessing non-existing column in Matrix!");
        return _cols[index];
    }

    Vector & Matrix::get_col(size_t index) {
        assert(index < _cols.size() && "Accessing non-existing column in Matrix!");
        return _cols[index];
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

    Matrix Matrix::operator+(const Vector &other) const {
        assert(_m == other.size() && "Adding Vector of wrong size!");
        std::vector<Vector> new_cols = std::vector<Vector>(_m);
        for (size_t i = 0; i < _m; i++) {
            new_cols[i] = _cols[i] + other[i];
        }
        return Matrix{_m, _n, new_cols}.T();
    }

    Matrix Matrix::operator-(const Vector &other) const {
        assert(_m == other.size() && "Subtracting Vector of wrong size!");
        std::vector<Vector> new_cols = std::vector<Vector>(_m);
        for (size_t i = 0; i < _m; i++) {
            new_cols[i] = _cols[i] - other[i];
        }
        return Matrix{_m, _n, new_cols}.T();
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

    Vector Matrix::sum() const {
        std::vector<float> new_data = std::vector<float>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i].sum();
        }
        return Vector(new_data);
    }

    Vector Matrix::mean() const {
        std::vector<float> new_data = std::vector<float>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i].mean();
        }
        return Vector(new_data);
    }

    Matrix Matrix::pow(float exponent) const {
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i].pow(exponent);
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::exp() const {
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i].exp();
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::log() const {
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i].log();
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::inv(const float other) {
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i].inv(other);
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::had(const Matrix &other) const {
        assert(shape() == other.shape() && "Matrices need to be of same shape for Hadamard-Product!");
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            Vector v = Vector(_m);
            for (size_t j = 0; j < _m; j++) {
                v[j] = _rows[i][j] * other[i][j];
            }
            new_data[i] = v;
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::max(const float &other) const {
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i].max(other);
        }
        return {_n, _m, new_data};
    }

    Matrix Matrix::abs() const {
        std::vector<Vector> new_data = std::vector<Vector>(_n);
        for (size_t i = 0; i < _n; i++) {
            new_data[i] = _rows[i].abs();
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

    Matrix Matrix::getRandom(size_t n, size_t m, float std) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, std);
        std::vector<std::vector<float>> data = std::vector<std::vector<float>>(n);
        for (size_t i = 0; i < n; i++) {
            std::vector<float> row = std::vector<float>(m);
            for (size_t j = 0; j < m; j++) {
                row[j] = static_cast<float>(dist(gen));
            }
            data[i] = row;
        }
        return Matrix(data);
    }

    Matrix fromLabels(const std::vector<size_t> &labels, size_t num_classes) {
        Matrix one_hot_encodings = Matrix(labels.size(), num_classes);
        for (size_t i = 0; i < labels.size(); ++i) {
            one_hot_encodings[i][labels[i]] = 1;
        }
        return one_hot_encodings;
    }
} // namespace math
