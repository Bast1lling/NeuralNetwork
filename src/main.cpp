//
// Created by sebas on 17.11.2024.
//
#include <matrix.hpp>
#include<print>
#include<vector.hpp>
using namespace math;

int main() {
    Vector vector = Vector(4, 5.f);
    Vector vector2 = Vector({1, 1, 2, 3});
    Matrix matrix = Matrix(2, 4, {vector, vector2});
    Matrix matrix2 = Matrix(4, 2, {vector, vector2});
    Matrix matrix3 = matrix * matrix2;
    Matrix matrix4 = matrix2 * matrix;
    matrix.print();
    matrix2.print();
    matrix3.print();
    matrix4.print();
    // float dot_product = vector / vector2;
    // std::println("Dot: {}", dot_product);
}
