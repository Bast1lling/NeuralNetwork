#include "vector.hpp"
#include <cassert>
#include <print>

namespace math {
    Vector::Vector(const size_t size) : _size(size) {
        _data = std::vector<float>(size);
    }

    Vector::Vector(const size_t size, const float value) : _size(size) {
        _data = std::vector(size, value);
    }

    Vector::Vector(const std::vector<float>& data) : _size(data.size()), _data(data) {
    }

    size_t Vector::size() const {
        return _size;
    }

    const float &Vector::operator[](const size_t index) const {
        assert(index < _size && "Accessing element outside of vector");
        return _data[index];
    }

    float &Vector::operator[](const size_t index) {
        assert(index < _size && "Accessing element outside of vector");
        return _data[index];
    }

    Vector Vector::operator+() const {
        return Vector(_data);
    }

    Vector Vector::operator-() const {
        std::vector<float> new_data = std::vector<float>(_size);
        for (size_t i = 0; i < _size; i++) {
            new_data[i] = -_data[i];
        }
        return Vector(new_data);
    }

    Vector Vector::operator+(const Vector &other) const {
        assert(_size == other._size && "Adding vectors of different size.");
        std::vector<float> new_data = std::vector<float>(_size);
        for (size_t i = 0; i < _size; i++) {
            new_data[i] = _data[i] + other._data[i];
        }
        return Vector(new_data);
    }

    Vector Vector::operator-(const Vector &other) const {
        assert(_size == other._size && "Subtracting vectors of different size.");
        std::vector<float> new_data = std::vector<float>(_size);
        for (size_t i = 0; i < _size; i++) {
            new_data[i] = _data[i] - other._data[i];
        }
        return Vector(new_data);
    }

    Vector Vector::operator*(const Vector &other) const {
        assert(_size == other._size && "Multiplying vectors of different size.");
        std::vector<float> new_data = std::vector<float>(_size);
        for (size_t i = 0; i < _size; i++) {
            new_data[i] = _data[i] * other._data[i];
        }
        return Vector(new_data);
    }

    Vector Vector::operator/(const Vector &other) const {
        assert(_size == other._size && "Dividing vectors of different size.");
        std::vector<float> new_data = std::vector<float>(_size);
        for (size_t i = 0; i < _size; i++) {
            new_data[i] = _data[i] / other._data[i];
        }
        return Vector(new_data);
    }

    bool Vector::operator==(const Vector &other) const {
        assert(_size == other._size && "Comparing vectors of different size.");
        for (size_t i = 0; i < _size; i++) {
            if (_data[i] != other._data[i]) {
                return false;
            }
        }
        return true;
    }

    void Vector::print() const {
        std::print("[");
        for (size_t i = 0; i < _size - 1; i++) {
            std::print("{}, ", _data[i]);
        }
        std::println("{}]", _data[_size - 1]);
    }
} // namespace math
