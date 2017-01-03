//Sam Christiansen
//10/15/2016
//Programing Language: C++
//Creating my own Vector class
#ifndef MY_VECTOR_H
#define MY_VECTOR_H
#include <iostream>

class MyVector{
	double* myArray;		//Dynamic Array
	int arraySize;			//Variable for the size

public:						//The constructor
	MyVector(int size) : arraySize(size), myArray(new double[size]){
		for (int i = 0; i < arraySize; i++)
			myArray[i] = 0;	//Initializes each position to 0
	}
	int size() const{		//Function to return the size
		return arraySize;
	}
	void print(){			//Function to print the Vector
		for (int i = 0; i < arraySize; i++){
			std::cout << myArray[i] << std::endl;
		}
		std::cout << std::endl;
	}
	double& operator[](int i){	//Overloading [] to access elements
		if (i < 0 || i >= arraySize)
			throw("range error");
		return myArray[i];
	}
};

#endif