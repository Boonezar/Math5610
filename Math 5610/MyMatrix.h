//Sam Christiansen
//10/15/2016
//Programing Language: C++
//Creating my own Matrix class
#ifndef MY_MATRIX_H
#define MY_MATRIX_H
#include <iostream>

class MyMatrix{
	double** myMatrix;				//Dynamic double array

public:
	int rowCount;					//Variable to hold the number of rows
	int columnCount;				//Variable to hold the number of columns

	MyMatrix(int size){				//Constructor for square matrix
		rowCount = size;			//Sets the row and column count to size
		columnCount = size;
		myMatrix = new double*[size];
		for (int i = 0; i < size; i++)
			myMatrix[i] = new double[size];
		for (int i = 0; i < size; i++)
			for (int j = 0; j < size; j++)
				myMatrix[i][j] = 0;	//Initializes each position to 0
	}

	MyMatrix(int rows, int columns){	//Constructor for non-square matrix
		rowCount = rows;				//Sets the rowCount
		columnCount = columns;			//Sets the columnCount
		myMatrix = new double*[rowCount];
		for (int i = 0; i < rowCount; i++)
			myMatrix[i] = new double[columnCount];
		for (int i = 0; i < rowCount; i++)
			for (int j = 0; j < columnCount; j++)
				myMatrix[i][j] = 0;	//Initializes each position to 0
	}

	void print(){					//Function to print the matrix
		for (int i = 0; i < columnCount; i++){
			for (int j = 0; j < rowCount; j++){
				std::cout << myMatrix[i][j] << "  ";
			}
			std::cout << std::endl;
		}
		std::cout << std::endl;
	}

	double& operator() (int i, int j) {
		return myMatrix[i][j];		//Overloading () to access elements
	}
};

#endif