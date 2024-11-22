#include <iostream>
#include <vector>
#include <cstdlib>
#include <ctime>
#include <type_traits>

class RandomArray {
public:
    int* array;

    RandomArray(size_t n, int seed) : size(n) {
        array = new int[n];
        std::srand(seed);
        generateArray();
    }

    ~RandomArray() {
        delete[] array;
    }

    void printArray() const {
        for (size_t i = 0; i < size; ++i) {
            std::cout << array[i] << " ";
        }
    }

private:
    size_t size;

    void generateArray() {
        for (size_t i = 0; i < size; ++i)
        {
            array[i] = std::rand();
        }
    }
};
