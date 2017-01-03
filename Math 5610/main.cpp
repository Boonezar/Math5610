#include "MathOperations.h"

int main(){
	int size = 5;
	MyMatrix A = generateRandomMatrix(size);
	cout << "Printing Matrix A: " << endl;
	A.print();
}