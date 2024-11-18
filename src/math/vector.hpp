#pragma once
#include <vector>

namespace math {
    class Vector {
        size_t _size{};
        std::vector<float> _data;

    public:
        // constructors
        Vector();

        explicit Vector(size_t size);

        Vector(size_t size, float value);

        explicit Vector(const std::vector<float>& data);

        // array operations
        size_t size() const;

        const float &operator[](size_t index) const;
        
        float &operator[](size_t index);

        // arithmetic operators
        Vector operator+() const;

        Vector operator-() const;

        Vector operator+(const Vector &other) const;

        Vector operator-(const Vector &other) const;

        float operator*(const Vector &other) const;

        float operator/(const Vector &other) const;

        Vector operator+(const float &other) const;

        Vector operator-(const float &other) const;

        Vector operator*(const float &other) const;

        Vector operator/(const float &other) const;

        Vector exp() const;

        Vector inv(float other = 1) const;

        // relational operators
        bool operator==(const Vector &other) const;

        // conversions
        explicit operator std::vector<float>() const;

        // other
        void print() const;
    };
} // namespace math

