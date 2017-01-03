#ifndef MATH_OPERATIONS_H
#define MATH_OPERATIONS_H
#include "MyMatrix.h"
#include "MyVector.h"
#include <stdlib.h>
#include <time.h>
#include <string>

using namespace std;

#pragma region Prototypes
MyMatrix generateSimpleMatrix(int size);
MyMatrix generateRandomSquareMatrix(int size);
MyMatrix generateRandomSymmetricSquareMatrix(int size);
MyMatrix generateTridiagonalMatrix(int size);
void singlePrecision();
void doublePrecision();
double l1Norm(MyVector u);
double l2Norm(MyVector u);
double lInfinityNorm(MyVector u);
double matrix1Norm(MyMatrix A, int size);
double matrixInfinityNorm(MyMatrix A, int size);
double conditionNumber1Norm(MyMatrix A, int size);
double conditionNumberInfNorm(MyMatrix A, int size);
double dotProduct(MyVector u, MyVector v);
MyVector crossProduct(MyVector u, MyVector v);
MyVector scalarVectorProduct(MyVector b, double c, int size);
MyVector vectorVectorAddition(MyVector a, MyVector b, bool isSubtraction);
MyVector matrixVectorProduct(MyMatrix A, MyVector y, int size);
MyMatrix scalarMatrixProduct(MyMatrix A, double c, int size);
MyMatrix matrixMatrixProduct(MyMatrix A, MyMatrix B, int size);
MyMatrix invertMatrix(MyMatrix A, int size);
MyMatrix transposeMatrix(MyMatrix A);
double bisectionMethod(double &a, double &b, double tol, int maxIter);
double functionalIteration(double xk, int maxIter, double tol);
double newtonsMethod(double xk, int maxIter, double tol);
double secantMethod(double x0, double x1, double tol, int maxIter);
MyVector backSubstitution(MyMatrix A, MyVector b, int size);
MyVector forwardSubstitution(MyMatrix A, MyVector b, int size);
void gaussianElimination(MyMatrix& A, MyVector& b, int size);
MyVector SolveWithGEandBS(MyMatrix A, MyVector b, int size);
MyVector jacobiIteration(MyMatrix A, MyVector b, double x0);
MyVector gaussSeidelIteration(MyMatrix A, MyVector b, double x0);
MyVector conjugateGradientMethod(MyMatrix A, MyVector b, double initGuess);
MyVector LUwithScaledPartialPivoting(MyMatrix& A, MyVector& b, int size);
MyVector GEwithScaledPartialPivoting(MyMatrix& A, MyVector& b, int size);
double powerMethod(MyMatrix A, MyVector v0);
double inversePowerMethod(MyMatrix A, MyVector v0);
#pragma endregion

#pragma region Matrix Constructors
//Function to create a simple matrix, with 1 parameter:
//The parameter, size, is the size of the matrix to be made
//The function returns the created matrix
MyMatrix generateSimpleMatrix(int size){
	MyMatrix A = MyMatrix(size);
	for (int i = 0; i < size; i++){
		for (int j = 0; j < size; j++){
			A(i, j) = i + i + j;
		}
	}
	A(0, 0) = 1;
	return A;
}
//Function to create a random, square matrix without a dominate diagonal, with 1 parameter:
//The parameter, size, is the size of the square matrix to be made
//The function returns the created matrix
MyMatrix generateRandomMatrix(int size){
	srand(time(NULL));					//Setting seed for rand()

	MyMatrix A = MyMatrix(size);		//Declaring new matrix
	for (int i = 0; i < size; i++){		//Loops 'size' times
		for (int j = 0; j < size; j++){	//Loops 'size' times
			A(i, j) = (double(rand()) / //Setting each entry to a 
				double(RAND_MAX)) * 100;//random number between 0 and 100
		}
	}
	return A;							//Returning new matrix A
}

//Function to create a random, square, diagonally dominate matrix, with 1 parameter:
//The parameter, size, is the size of the matrix to be made
//The function returns the created matrix
MyMatrix generateRandomSquareMatrix(int size){
	srand(time(NULL));						//Setting seed for rand()

	MyMatrix A = MyMatrix(size);			//Declaring new matrix
	for (int i = 0; i < size; i++){			//Loops 'size' times
		for (int j = 0; j < size; j++){		//Loops 'size' times
			A(i, j) = double(rand())		//Setting each entry to a 
				/ double(RAND_MAX);			//random number between 0 and 1
		}
	}
	for (int i = 0; i < size; i++){			//Loops 'size' times
		A(i, i) = A(i, i) + (10 * size);	//Sets each i=j entry A[i][j] to a 
	}										//much larger number (as directed)
	return A;								//Returning new matrix A
}

//Function to create a random, square and symmetric matrix, with 1 parameter:
//The parameter, size, is the size of the matrix to be made
//The function returns the created matrix
MyMatrix generateRandomSymmetricSquareMatrix(int size){
	srand(time(NULL));				//Setting seed for rand()

	MyMatrix A = MyMatrix(size);			//Declaring new matrix
	for (int i = 0; i < size; i++){			//Loops 'size' times
		for (int j = i; j < size; j++){		//Loops 'size' times
			A(i, j) = rand() % 4 + 1;	//Setting each entry to a 
			//random number between 1 and 4.
			A(j, i) = A(i, j);		//This makes the matrix symmetric
		}
	}
	return A;					//Returning new symmetric matrix A
}

//Function to create a Tridiagonal Matrix, as directed, with 1 parameter:
//The parameter, size, is the size of the matrix to be made
//The function returns the Tridiagonal Matrix
MyMatrix generateTridiagonalMatrix(int size){
	MyMatrix A = MyMatrix(size);

	for (int i = 0; i < size; i++){
		A(i, i) = -2.0;				//For main diagonal entries
		if (i>0)
			A(i - 1, i) = A(i, i - 1) = 1.0; //For lower and upper diagonal entries
	}
	return A;					//Returning new tridiagonal matrix A
}
#pragma endregion

#pragma region Basic Computational Routines
//Function to find the single precision of a computer:
//The function first prints out the smallest number computed, 
//and then prints out the power of 2
void singlePrecision(){
	float num = 1.0;				//Number of type float for single precision
	int exp = 0;				//Number for the exponent value
	while (num + 1 != 1){			//Main loop
		exp--;				//Decrement exponent value
		num /= 2;			//Divide the number by 2
	}
	cout << "Single Precision: " <<		//Printing results
		num << "\t Exponent: " <<
		exp << endl;
}

//Function to find the double precision of a computer:
//The function first prints out the smallest number computed, 
//and then prints out the power of 2
void doublePrecision(){
	double num = 1.0;			//Number of type double for double precision
	int exp = 0;				//Number for the exponent value
	while (num + 1 != 1){			//Main loop
		exp--;				//Decrement exponent value
		num /= 2;			//Divide the number by 2
	}
	cout << "Double Precision: " <<		//Printing results
		num << "\t Exponent: " <<
		exp << endl;
}

//Function to find the l1 Norm of a vector, with 1 parameter:
//The parameter is the vector we want the l1 Norm for
//The function returns the value of the l1 Norm
double l1Norm(MyVector u){
	double num = 0;				//Initialing variable to be returned
	for (int i = 0; i < u.size(); i++){	//Loop for size of the vector
		num += abs(u[i]);		//Summing the absolute value of each element
	}
	return num / u.size();			//Returning the sum divided by the vector size
}

//Function to find the l2 Norm of a vector, with 1 parameter:
//The parameter is the vector we want the l2 Norm for
//The function returns the value of the l2 Norm
double l2Norm(MyVector u){
	double num = 0;				//Initialing variable to be returned
	for (int i = 0; i < u.size(); i++){	//Loop for size of the vector
		num += pow(u[i], 2);		//Summing the square of each element
	}
	return sqrt(num);			//Returning square root of the sum
}

//Function to find the l∞ Norm of a vector, with 1 parameter:
//The parameter is the vector we want the l∞ Norm for
//The function returns the value of the l∞ Norm
double lInfinityNorm(MyVector u){
	int max = 0;				//Initialing variable for the vector index
	for (int i = 0; i < u.size(); i++){	//Loop for the size of the vector
		if (u[i]>u[max])			//Checking which element is larger
			max = i;			//If the new element is larger, set max to it
	}
	return u[max];				//Returning the max value of the vector
}

//Function to find the 1-Norm of a Real Square Matrix
//The first parameter, A, is a real, nonsingular nxn matrix.
//The second parameter, size, is the size of the matrix.
//The function returns the value of the 1-Norm
double matrix1Norm(MyMatrix A, int size){
	double max = 0;
	double temp;
	for (int i = 0; i < size; i++){
		temp = 0;
		for (int j = 0; j < size; j++){	//temp holds the sum of the absolute value 					temp += fabs(A(i, j));	//of each element for each row
		}

		if (temp > max)			//If temp > max, the temp value becomes the max value
			max = temp;
	}
	//Return the max absolute row sum
	return max;
}

//Function to find the Infinity-Norm of a Real Square Matrix
//The first parameter, A, is a real, nonsingular nxn matrix.
//The second parameter, size, is the size of the matrix.
//The function returns the value of the Infinity-Norm
double matrixInfinityNorm(MyMatrix A, int size){
	double max = 0;
	double temp;
	for (int i = 0; i < size; i++){
		temp = 0;
		for (int j = 0; j < size; j++){
			temp += fabs(A(j, i)); 	//temp holds the sum of the absolute value 
		}				//of each element for each column
		if (temp > max)
			max = temp;		//If temp > max, the temp value becomes the max value
	}
	return max;				//Return the max absolute column sum
}

//Function to find the Condition Number of a Square Matrix
//		using the 1-Norm
//The first parameter, A, is a real, nonsingular nxn matrix.
//The second parameter, size, is the size of the matrix.
//The function returns the value of the 1-Norm Condition Number
double conditionNumber1Norm(MyMatrix A, int size){
	MyMatrix Ainverse = invertMatrix(A, size);		//Inverting the Matrix
	return matrix1Norm(A, size)*matrix1Norm(Ainverse, size);	//Returning the product of A’s 1norm 
}

//Function to find the Condition Number of a Square Matrix
//		using the Infinity-Norm
//The first parameter, A, is a real, nonsingular nxn matrix.
//The second parameter, size, is the size of the matrix.
//The function returns the value of the Infinity-Norm Condition Number
double conditionNumberInfNorm(MyMatrix A, int size){
	MyMatrix Ainverse = invertMatrix(A, size);		//Inverting the Matrix	
	return matrixInfinityNorm(A, size) * 			//Returning the product of A’s InfNorm
		matrixInfinityNorm(Ainverse, size);		//and A’s inverse’s InfNorm
}
#pragma endregion

#pragma region Basic Linear Operations
//Function to find the dot product of 2 vector, with 2 parameters:
//The first and second parameter are the vectors supplied for the dot product
//The function returns the value of the dot product
double dotProduct(MyVector u, MyVector v){
	double num = 0;				//Initializing variable to hold the answer
	for (int i = 0; i < u.size(); i++){	//Looping for the size of the vector
		num += u[i] * v[i];		//Summing the products of u[i] and v[i]
	}
	return num;				//Returning the result
}

//Function to find the cross product of 2 vector of size 3, with 2 parameters:
//The first and second parameter are the vectors supplied for the cross product
//The function returns new vector fesulting from the cross product
MyVector crossProduct(MyVector u, MyVector v){
	MyVector w(3);				//Declaring new vector to be returned
	double num;				//Initializing variable to hold the values
	num = (u[1] * v[2]) - (u[2] * v[1]);	//Solving for w[0]
	w[0] = num;
	num = (u[0] * v[2]) - (u[2] * v[0]);	//Solving for w[1]
	w[1] = num;
	num = (u[0] * v[1]) - (u[1] * v[0]);	//Solving for w[2]
	w[2] = num;
	return w;				//Returning the resulting vector, w
}

//Function to calculate the product of a scalar and vector
//The first parameter, b, is the vector
//The second parameter, c, is the scalar value
//The third parameter, size, is the size of the vector
MyVector scalarVectorProduct(MyVector b, double c, int size){
	MyVector x = MyVector(size);	//Initializing the vector to be returned
	for (int i = 0; i < size; i++)	//For loop for the size of the vector
		x[i] = b[i] * c;		//Sets the value of the product for each element
	return x;			//Returns the solved vector
}

//Function to calculate adding two vectors together
//The first parameter, a, is the first vector
//The second parameter, b, is the second vector
//The third parameter, isSubraction, is a boolean to say if 
//	the function should do subtraction:
//	if true: solve a-b
//	if false: solve a+b
MyVector vectorVectorAddition(MyVector a, MyVector b, bool isSubtraction){
	MyVector c = MyVector(a.size());			//Initializing vector to be returned
	if (isSubtraction){				//If subtraction is wanted
		for (int i = 0; i < a.size(); i++)	//Loop for the size of the vectors
			c[i] = a[i] - b[i];		//Sets value of the difference for each element
	}
	else{						//If subtraction is not wanted
		for (int i = 0; i < a.size(); i++)	//Loop for the size of the vectors
			c[i] = a[i] + b[i];		//Sets value of the sum for each element
	}
	return c;					//Returns the solved vector
}

//Function to calculate the product of a matrix and vector, with 3 parameters:
//The first parameter, A, is the matrix provided
//The second parameter, y, is the vector provided
//The third parameter, size, is the size of the vector
//The function returns the new resulting vector
MyVector matrixVectorProduct(MyMatrix A, MyVector y, int size){
	MyVector b = MyVector(size);			//Declaring new vector to be solved and returned
	for (int i = 0; i < size; i++){			//Will loop 'size' times
		for (int j = 0; j < size; j++){		//Will loop 'size' times
			b[i] = b[i] + A(i, j)*y[j];	//Will sum the product for each matrix row 
		}					//i element and vector element,		
	}						//and saving it in b[i]
	return b;					//Returning the solved vector b
}

//Function to calculate the product of a scalar and matrix
//The first parameter, B, is the matrix
//The second parameter, c, is the scalar value
//The third parameter, size, is the size of the matrix
MyMatrix scalarMatrixProduct(MyMatrix A, double c, int size){
	MyMatrix B = MyMatrix(size);	//Initializing the matrix to be returned
	for (int i = 0; i < size; i++)	//Double for loop for the size of the matrix
		for (int j = 0; j < size; j++)
			B(i, j) = B(i, j) * c;	//Sets the value of the product for each element
	return B;				//Returns the solved vector
}

//Function to calculate the product of two matrices, with 3 parameters
//The first parameter, A, is the first matrix provided
//The second parameter, B, is the second matrix provided
//The third parameter, size, is the size of the matrices
//The function returns the resulting matrix
MyMatrix matrixMatrixProduct(MyMatrix A, MyMatrix B, int size){
	MyMatrix C = MyMatrix(size);			//Initializing matrix to be returned
	for (int i = 0; i < size; ++i){			//Triple for loop
		for (int j = 0; j < size; ++j){
			for (int k = 0; k < size; ++k){
				C(i, j) = C(i, j)		//Setting result into each element 
					+ A(i, k)*B(k, j);
			}
		}
	}
	return C;					//Returning the resulting matrix
}

//Function to calculate the inverse of a matrix
//The first parameter, A, is the nonsingular nxn matrix to be inverted
//The second parameter, size, is the size of the matrix
//The function returns the inverted matrix
MyMatrix invertMatrix(MyMatrix A, int size)
{
	int n = size;
	MyMatrix B = MyMatrix(size);		//Initializing matrix to be returned

	for (int i = 0; i < n; i++){
		for (int j = 0; j < n; j++)
			B(i, j) = A(i, j);	//Copy the given matrix into this matrix
	}
	double t;
	for (int i = 0; i<n; i++)
	{
		for (int j = n; j<2 * n; j++)
		{
			if (i == j - n)
				B(i, j) = 1;
			else
				B(i, j) = 0;
		}
	}
	for (int i = 0; i<n; i++)
	{
		t = B(i, i);
		for (int j = i; j<2 * n; j++)
			B(i, j) = B(i, j) / t;
		for (int j = 0; j<n; j++)
		{
			if (i != j)
			{
				t = B(j, i);
				for (int k = 0; k<2 * n; k++)
					B(j, k) = B(j, k) - t*B(i, k);
			}
		}
	}
	return B;				//Return the inverted matrix
}

//Function to calculate the transpose of a matrix
//The first parameter is the matrix desired to be transposed
//The function returns the transposed matrix
MyMatrix transposeMatrix(MyMatrix A){
	MyMatrix B = MyMatrix(A.columnCount, A.rowCount);	//Creates new matrix to be returned
	for (int i = 0; i < A.columnCount; i++)		//Double for loop for size of matrix
		for (int j = 0; j < A.rowCount; j++)
			B(i, j) = A(j, i);		//Sets B(i,j) = A(j,i)
	return B;					//Returns the transposed matrix
}

#pragma endregion

#pragma region Root Finding Methods
//The function f that I will use to test Root Finding Methods
// f(x) = x^2 - 4
double f(double a){
	return (a*a) - 4;
}
//The derivative of f that I will use to test Newton's Method
// f'(x) = 2x
double fPrime(double a){
	return 2 * a;
}
//An example function f(x) that I will use to test Functional Iteration
// f(x) = cCosh(x/4) - x, where c is some constant
double f1(double a, double c){
	return c*cosh(a / 4) - a;
}
//The function g(x) such that f(x)=0 if and only if g(x) = x 
// g(x) = 2Cosh(x/4)
double g1(double a){
	return 2 * cosh(a / 4);
}

//The function to implement the Bisection Method with 4 parameters:
//The first parameter, a, is the min starting point
//The second parameter, b, is the max starting point
//The third parameter, tol, is the given tolerance
//The fourth parameter, maxIter, is the max number of iterations we will allow
double bisectionMethod(double &a, double &b, double tol, int maxIter){
	double fa = f(a);		 	//Initializing the value of the function at point a
	double fb = f(b);		 	//Initializing the value of the function at point b

	if (fa*fb > 0) 				//If fa*fb > 0, then we want to throw this error
		throw("There is no root (or there are multiple roots) between a = "+to_string(a)+" and b = " +to_string(b)+"\n");
		if (fa*fb == 0){  			//if fa*fb == 0, then at least one of them is a root
		if (fa == fb == 0)		//if both == 0, then both are roots.
			throw("Both a = " + to_string(a) + " and b = " + to_string(b) + " are roots\n");
		else if (fa == 0)
			return a; 		//return a as the root
		else
			return b;  		//return b as the root
		}
	int iter = 0; 				//Initializing the iterator
	double error = 10 * tol;  		//Initializing the error
	double c, fc;  				//Declaring variables for the 'mid' value c and
	// the value of the function at c 
	//Loop continues until the error > the tolerance
	// or until the maxIter is reached
	while (error > tol&&iter < maxIter){
		c = (a + b) / 2;    		//Setting the 'mid' value
		fc = f(c);			//Setting value of function at 'mid' value
		if (fa*fc < 0){			//If true, let the 'mid' value be the new max, b
			b = c;
			fb = fc;
		}
		else{				//Otherwise, let the 'mid' value be the new min, a
			a = c;
			fa = fc;
		}
		error = abs(b - a);		//Update the error 
		iter++;				//Update the iterator
	}
	if (iter == maxIter)
		throw("Reached the Maximum Iteration count before finding a root.");
	//Printing results
	cout << "Number of iterations: " << iter << endl;	//Print value of iter
	cout << "Error: " << error << endl;	//Print value of error
	return c;				//Return the root value
}

//The function to implement the Functional Iteration with 3 parameters:
//The first parameter, xk, is the initial guess
//The second parameter, maxIter, is the max number of iterations we will allow
//The third parameter, tol, is the given tolerance
double functionalIteration(double xk, int maxIter, double tol){
	if (g1(xk) == 0)			//If g(xk) == 0, then xk is a root
		return xk;

	int iter = 0;			//Initializing iterator			
	double error = 10 * tol;		//Initializing the error
	double xkp1;			//Declaring variable for x(k+1)
	//Loop continues until the error > the tolerance 
	while (iter<maxIter&&error>tol){ //or until the maxIter is reached
		xkp1 = g1(xk);		//Running the algorithm for Functional Iteration
		error = abs(xkp1 - xk);	//Updating error
		xk = xkp1;		//Updating x(k) to x(k+1) value
		iter++;			//Updating iterator
	}
	if (iter == maxIter)		//Throw an error if the max iterations was hit
		throw("Max Iterations hit, Root not found.");
	//Print the results
	cout << "Number of iterations: " << iter << endl;
	cout << "Error: " << error << endl;
	return xkp1;			//Return the root value
}

//The function to implement Newton's Method with 3 parameters:
//The first parameter, xk, is the initial guess
//The second parameter, maxIter, is the max number of iterations we will allow
//The third parameter, tol, is the given tolerance
double newtonsMethod(double xk, int maxIter, double tol){
	double xkp1;				//Declaring variable for x(k+1)
	double fx = f(xk);			//Initializing variable for value of f(x)
	double fPrimeX = fPrime(xk);		//Initializing variable for value of f'(x)
	int iter = 0;				//Initializing the iterator
	double error = 10 * tol;			//Initializing the error

	//Loop continues until the error > the tolerance 
	//or until the maxIter is reached
	while (iter<maxIter && error > tol){
		if (fPrimeX == 0){		//If f'(x) == 0, then we need to throw this 
			//error to prevent division by 0:
			throw("For iteration #" + to_string(iter) +
				", f'(x) = 0, so Newton's Method was unsuccesful at finding a root.");
		}
		xkp1 = xk - (fx / fPrimeX);	//Running algorithm for Newton's Method
		iter++;				//Updating the iterator
		error = abs(xkp1 - xk);		//Updating the error
		xk = xkp1;			//Setting x(k) value to x(k+1) for next iteration
		fx = f(xk);			//Setting value of f(x) for next iteration
		fPrimeX = fPrime(xk);		//Setting value of f'(x) for next iteration
	}
	if (iter == maxIter)			//Throw an error if the max iterations was hit
		throw("Max Iterations hit, Root not found.");
	//Printing results
	cout << "Number of Newton iterations: " << iter << endl;
	cout << "Error: " << error << endl;					
	return xkp1;				//Returning x(k+1) for root value
}

//The function to implement the Secant Method with 4 parameters:
//The first parameter, x0, is the first initial guess
//The second parameter, x1, is the second initial guess
//The third parameter, tol, is the given tolerance
//The fourth parameter, maxIter, is the max number of iterations we will allow
double secantMethod(double x0, double x1, double tol, int maxIter){
	double fkm1 = f(x0);		//Initializing variable for the functions value at x(k-1)
	double fk = f(x1);		//Initializing variable for the functions value at x(k)
	double xkp1;			//Declaring variable for x(k+1)
	double error = 10 * tol;		//Initializing the error
	int iter = 0;			//Initializing the iterator

	//Loop continues until the error > the tolerance 
	//or until the maxIter is reached
	while (iter < maxIter&&error > tol){
		if ((fk - fkm1) == 0){	//If f(x(k))-f(x(k-1)) == 0, then we need to throw 
			//this error to prevent division by 0:
			throw("For iteration #" + to_string(iter) +
				", f'(x) = 0, so Newton's Method was unsuccesful at finding a root.");
		}
		//Running algorithm for Secant Method
		xkp1 = x1 - ((fk)*(x1 - x0)) / (fk - fkm1);
		error = abs(xkp1 - x1);	//Updating the error
		iter++;			//Updating the iterator
		fkm1 = fk;		//Setting value of f(x(k-1)) to f(x) for the next iteration
		fk = f(xkp1);		//Setting value of f(x) to f(x(k+1)) for the next iteration
		x0 = x1;			//Setting value of x(k-1) to x(k) for the next iteration
		x1 = xkp1;		//Setting value of x(k) to x(k+1) for the next iteration
	}
	//Printing results
	cout << "Number of iterations: " << iter << endl;
	cout << "Error: " << error << endl;
	return x1;			//Return x(k+1) for the root value
}
#pragma endregion

#pragma region Solving Linear Systems
//Function to implement the Back Substitution method with 3 paramenters:
//The first parameter, A, is an Upper Triangular Matrix.
//The second parameter, b, is a right-hand-side vector.
//The third parameter, size, is the length of the square matrix A and the size of b
MyVector backSubstitution(MyMatrix A, MyVector b, int size){
	int n = size;					//Set n to size
	MyVector x = MyVector(size);		//Declaring new vector to be returned
	double s;					//Declaring variable for the factor to
	//be used in each answer 

	for (int i = n - 1; i >= 0; i--){		//This will loop n times
		s = 0;					//Setting the factor variable to 0
		for (int j = i + 1; j < n; j++){	//This will loop j times, depending 
			//on the i row
			s = s + A(i, j)*x[j];	//Updating the factor variable
		}
		x[i] = (b[i] - s) / A(i, i);	//Solving x[i] using the factor variable
	}
	return x;					//Return the solved vector variable
}

//Function to implement the Forward Substitution method with 3 paramenters:
//The first parameter, A, is an Lower Triangular Matrix.
//The second parameter, b, is a right-hand-side vector.
//The third parameter, size, is the length of the square matrix A and the size of b
MyVector forwardSubstitution(MyMatrix A, MyVector b, int size){
	int n = size;				//Setting n to the size
	MyVector x = MyVector(size);		//Declaring new vector to be solved

	x[0] = b[0] / A(0, 0);			//Solving the first element x
	for (int i = 1; i < size; i++){		//This will loop n-1 times
		x[i] = b[i];			//Updating the new x[i] to be used below
		for (int j = 0; j < i; j++){	//This will loop j times, depending on the i row
			x[i] = x[i] - A(i, j)*x[j];//Updating the x[i] again according to the algorithm
		}
		x[i] = x[i] / A(i, i);		//Giving x[i] its final, solved value
	}

	return x;				//Returning the solved vector
}

//Function to implement the Gaussian Elimination method with 3 paramenters:
//The first parameter, A, is a real, nonsingular nxn matrix.
//The second parameter, b, is a right-hand-side vector.
//The third parameter, size, is the length of the square matrix A and the size of b
void gaussianElimination(MyMatrix& A, MyVector& b, int size){
	int n = size;					//Sets n to size
	double factor = 0;				//Declaring factor variable

	for (int k = 0; k < n - 1; k++){			//Will loop n-1 times
		for (int i = k + 1; i < n; i++){		//Will loop i times depending on the row k
			factor = A(i, k) / A(k, k);	//Updating factor variable according to algorithm
			for (int j = 0; j < n; j++){	//Will loop n times for each element in the row
				A(i, j) = A(i, j) - factor*A(k, j);	//Setting each matrix element to
			}						 //the solved value

			b[i] = b[i] - factor*b[k];	//Setting each vector element to the solved value
		}
	}						//This doesn't return anything since the matrix 
}							//and vector were passed by reference

//Function to implement the Gaussian Elimination method with 3 paramenters:
//The first parameter, A, is a real, nonsingular nxn matrix.
//The second parameter, b, is a right-hand-side vector.
//The third parameter, size, is the length of the square matrix A and the size of b
MyVector SolveWithGEandBS(MyMatrix A, MyVector b, int size){
	cout << "Matrix before Solving: " << endl;
	for (int i = 0; i < size; i++){			//Printing the starting matrix and vector 
		for (int j = 0; j < size; j++){		//  before solving
			cout << A(i, j) << " ";
		}
		cout << "\t" << b[i] << endl;
	}
	cout << endl;					//End of printing starting matrix

	gaussianElimination(A, b, size);			//Calling GE function
	MyVector x = backSubstitution(A, b, size);	//Calling BS function on modified A and b
	cout << "Solution: " << endl; 			//Printing out the solved vector x
	for (int i = 0; i < size; i++){
		cout << x[i] << endl;
	}
	cout << endl; 					//End of printing solved vector
	return x;
}

//Function to calculate the QR factorization of a matrix using
//	the Modified Gram-Schmidt Orthogonalization, with 4 parameters
//The first parameter, A, is the given matrix
//The second parameter, Q, is an empty matrix that is passed by reference
//The third parameter, R, is also an empty matrix passed by reference
//The fourth parameter, size, is the size of the matrix.
//The function doesn't return anything, but the Q and R matrices are passed
//	by reference, so changes made to them in this function will persist outside
//  of this function.
//Source for this code: http://www.cplusplus.com/forum/general/88888/
void modifiedGramSchmidtOrthogonalization(MyMatrix A, MyMatrix &Q, MyMatrix &R, int size){
	for (int k = 0; k < size; k++){						//Main loop

		for (int i = 0; i < size; i++)					//These three lines compute the norm
			R(k, k) = R(k, k) + A(i, k) * A(i, k);		//	of A, which is then saved in
		R(k, k) = sqrt(R(k, k));						//	R(k,k)

		for (int i = 0; i < size; i++)					//Solves each element of one 
			Q(i, k) = A(i, k) / R(k, k);				//	column of Q


		for (int j = k + 1; j < size; j++) {
			for (int i = 0; i < size; i++)
				R(k, j) += Q(i, k) * A(i, j);			//Finds non-diagonal values for R

			for (int i = 0; i < size; i++)
				A(i, j) = A(i, j) - R(k, j) * Q(i, k);	//Updates A for the next loop
		}
	}
}

//Function to solve a linear system, Ax = b, with QR Factorization
//	with 3 parameters
//The first parameter, A, is the given matrix
//The second parameter, b, is the right-hand side vector
//The third parameter, size, is the size of the vector
//The function returns the solved vector x.
MyVector solveWithQR(MyMatrix A, MyVector b, int size){
	MyMatrix Q(size), R(size);								//Initializing the Q and R matrices
	modifiedGramSchmidtOrthogonalization(A, Q, R, size);	//Calling the QR factorization function
	MyMatrix transposeQ = transposeMatrix(Q);				//Transposing matrix Q
	MyVector c = matrixVectorProduct(transposeQ, b, size);	//Solving c = (Q^T)b
	return backSubstitution(R, c, size);					//Solving and returning Rx = c
}

//Function to solve a system of linear equations, Ax=b using
//    Jacobi Iteration with 3 parameters
//The first parameter, A, is the matrix from Ax=b
//The second parameter, b, is the right hand side vector from Ax=b
//The third parameter, x0, is a value for the initial guess
MyVector jacobiIteration(MyMatrix A, MyVector b, double x0){
	int n = b.size();					//Setting n to the size
	int iter = 0, maxIter = 20;				//Initializing the iterator and maxIter
	double tol = .0000001;					//Initializing the tolerance 
	double error = tol * 10, temp;				//Initializing the error and temp error
	MyVector xk = MyVector(n);				//Initializing the vector for x k
	MyVector xkm1 = MyVector(n);				//Initializing the vector for x k-1
	for (int i = 0; i < n; i++)				//Setting each value for the x k-1 to 
		xkm1[i] = x0;					//	the initial guess, x0
	while (iter<maxIter&&error>tol){				//Main while loop
		for (int i = 0; i < n; i++){			//Main for loop, for the size of vector
			xk[i] = b[i];
			for (int j = 0; j < i - 1; j++){		//Loop for values below 
				//	the main diagonal
				xk[i] = xk[i] - A(i, j)*xkm1[j];	//Main iteration
			}
			for (int j = i + 1; j < n; j++){		//Loop for the values above
				//	the main diagonal
				xk[i] = xk[i] - A(i, j)*xkm1[j];	//Main iteration
			}
			xk[i] = xk[i] / A(i, i);			//Last step of iteration
			temp = abs((xkm1[i] - xk[i]) / xk[i]);	//Check the error for the vector at i
			if (temp > error)			//If this temp error is bigger than
				error = temp;			//the existing error, set it as error
			xkm1[i] = xk[i];				//set x k-1 to x k for next iteration
		}
		iter++;						//Add a counter
	}
	if (iter == maxIter)					//Shows if the maxIter was reached
		cout << "Max Iterations of " << maxIter << " was reached." << endl;
	return xk;						//Return the solved vector
}

//Function to solve a system of linear equations, Ax=b using
//    Gauss-Seidel Iteration with 3 parameters
//The first parameter, A, is the matrix from Ax=b
//The second parameter, b, is the right hand side vector from Ax=b
//The third parameter, x0, is a value for the initial guess
MyVector gaussSeidelIteration(MyMatrix A, MyVector b, double x0){
	int n = b.size();					//Setting n to the size
	int iter = 0, maxIter = 20;				//Initializing the iterator and maxIter
	double tol = .0000001;					//Initializing the tolerance 
	double error = tol * 10, temp;				//Initializing the error and temp error
	MyVector xk = MyVector(n);				//Initializing the vector for x k
	MyVector xkm1 = MyVector(n);				//Initializing the vector for x k-1
	for (int i = 0; i < n; i++)				//Setting each value for the x k-1 to 
		xkm1[i] = x0;					//	the initial guess, x0

	while (iter<maxIter&&error>tol){				//Main while loop
		for (int i = 0; i < n; i++){			//Main for loop, for the size vector
			xk[i] = b[i];
			for (int j = 0; j < i - 1; j++){		//Loop for values below 
				//	the main diagonal
				xk[i] = xk[i] - A(i, j)*xk[j];	//Main iteration
			}
			for (int j = i + 1; j < n; j++){		//Loop for the values above
				//	the main diagonal
				xk[i] = xk[i] - A(i, j)*xk[j];	//Main iteration
			}
			xk[i] = xk[i] / A(i, i);			//Last step of iteration
			temp = abs((xkm1[i] - xk[i]) / xk[i]);	//Check the error for the vector at i
			if (temp > error)			//If this temp error is bigger than
				error = temp;			//the existing error, set it as error
			xkm1[i] = xk[i];				//set x k-1 to x k for next iteration
		}
		iter++;						//Add a counter
	}
	if (iter == maxIter)					//Shows if the maxIter was reached
		cout << "Max Iterations of " << maxIter << " was reached." << endl;
	return xk;						//Return the solved vector
}

//Function to solve a system of linear equations, Ax=b using the
//    Conjugate Gradient Method with 3 parameters
//The first parameter, A, is the matrix from Ax=b
//The second parameter, b, is the right hand side vector from Ax=b
//The third parameter, initGuess, is a value for the initial guess
MyVector conjugateGradientMethod(MyMatrix A, MyVector b, double initGuess){
	int size = b.size();				//Setting size
	double tol = .0000001;				//Initializing the tolerance
	MyVector xk = MyVector(size);			//Initializing x k for initial guess
	for (int i = 0; i < size; i++)
		xk[i] = initGuess;			//Setting each value of x k to the initial guess
	MyVector xkp1 = MyVector(size);			//Initializing vector for x k+1
	MyVector rk = vectorVectorAddition(b,		//Initializing the first value of r k, r0, to
		matrixVectorProduct(A, xk, size),	//	r0 = b-Ax0
		true);					//true is set because subtraction is wanted
	MyVector rkp1 = MyVector(size);			//Initializing vector for r k+1
	MyVector sk = MyVector(size);			//Initializing vector for s k
	MyVector pk = rk;				//Initializing vector for p k and settings to r0
	MyVector pkp1 = MyVector(size);			//Initializing vector for p k+1
	double deltaK = dotProduct(rk, rk);		//Initializing δ k, δ0 = <r0,r0>
	double deltaKp1;					//Initializing δ k+1
	double alphaK;					//Initializing α k
	double bDelta = dotProduct(b, b);		//Initializing bδ, bδ = <b,b>
	double totalTol = tol*tol*bDelta;		//Setting total tolerance for main loop
	int maxIter = 100, iter = 0;			//Initializing iterators

	while (deltaK > totalTol&&iter<maxIter){		//Main while loop
		sk = matrixVectorProduct(A, pk, size);	//Updating sk, sk = A*pk
		alphaK = deltaK / dotProduct(pk, sk);	//Updating αk, αk = δ k / <pk,sk>
		xkp1 = vectorVectorAddition(xk,		//Main Iteration, x k+1 = xk + αk*pk
			scalarVectorProduct(pk, alphaK, size),
			false);				//False is set because addition is wanted
		rkp1 = vectorVectorAddition(rk,		//Updating r k+1, r k+1 = rk - αk*sk
			scalarVectorProduct(sk, alphaK, size),
			true);				//True is set because subtraction is wanted
		deltaKp1 = dotProduct(rkp1, rkp1);	//Updating δ k+1, δ k+1 = <r k+1, r k+1> 
		pkp1 = vectorVectorAddition(rkp1,	//Updating p k+1, p k+1 = r k+1 + (δ k+1/δ k)*pk
			scalarVectorProduct(pk, deltaKp1 / deltaK, size),
			false);				//False is set because addition is wanted

		deltaK = deltaKp1;			//Set δ k to δ k+1 for next iteration
		xk = xkp1;				//Set x k to x k+1 for next iteration
		rk = rkp1;				//Set r k to r k+1 for next iteration
		pk = pkp1;				//Set p k to p k+1 for next iteration
		iter++;					//Update iteration counter
	}
	cout << "Num of iter: " << iter << endl;		//Prints the number of iterations needed to solve
	return xkp1;					//Returns the solved vector
}

//Function to implement the LU decomposition using Scaled Partial Pivoting with 3 paramenters:
//The first parameter, A, is a real, nonsingular nxn matrix.
//The second parameter, b, is a right-hand-side vector.
//The third parameter, size, is the length of the square matrix A and the size of b
MyVector LUwithScaledPartialPivoting(MyMatrix& A, MyVector& b, int size){
	int* pvt = new int[size];				//Pivoting vector
	double* scale = new double[size];			//Scaling vector
	for (int i = 0; i < size; i++)				//Initializing pivoting vector
		pvt[i] = i;

	for (int i = 0; i < size; i++){				//Loops 'size' times
		scale[i] = 0;					//Initialize scale to 0
		for (int j = 0; j < size; j++){			//Loops 'size' times
			if (fabs(scale[i]) < fabs(A(i, j)))	//if the next position is larger,
				scale[i] = fabs(A(i, j));	//set it for the scale value
		}
	}

	double temp;						//Variable to check each position
	for (int k = 0; k < size - 1; k++){			//Main loop
		int pc = k;					//pivot column
		double aet = fabs(A(pvt[k], k) / scale[k]);
		for (int i = k + 1; i < size; i++){		//Loops 'size'-1 times
			temp = fabs(A(pvt[i], k) / scale[i]);
			if (temp > aet){				//Comparing each position with aet
				aet = temp;			//If temp>aet, set aet = to temp
				pc = i;				//and set pivot column to i
			}
		}
		if (pc != k){					//If pivot column was changed, we need 
			int tempNum = pvt[k];			//to switch the pivot values
			pvt[k] = pvt[pc];
			pvt[pc] = tempNum;
		}
		for (int i = k + 1; i < size; i++){		//Now we will shift the rows in A
			if (A(pvt[i], k) != 0){
				double factor = A(pvt[i], k) / A(pvt[k], k);
				A(pvt[i], k) = factor;
				for (int j = k + 1; j < size; j++)
					A(pvt[i], j) -= factor*A(pvt[k], j);
			}
		}
	}
	cout << "Printing after SPP: " << endl;			//Printing to console to see result of 
	A.print();						//Scaled Partial Pivoting
	b.print();
	//Now computing the LU Decomposition
	for (int i = 1; i < size; i++)				//First find Ly = b
		for (int j = 0; j < i; j++)
			b[pvt[i]] -= A(pvt[i], j)*b[pvt[j]];
	//Now find Ux = y
	MyVector xx = MyVector(size);				//Creating solution vector
	for (int i = size - 1; i >= 0; i--){
		for (int j = i + 1; j < size; j++)
			b[pvt[i]] -= A(pvt[i], j)*xx[j];
		xx[i] = b[pvt[i]] / A(pvt[i], i);
	}

	return xx;
}

//Function to implement the GE and BS using Scaled Partial Pivoting with 3 paramenters:
//The first parameter, A, is a real, nonsingular nxn matrix.
//The second parameter, b, is a right-hand-side vector.
//The third parameter, size, is the length of the square matrix A and the size of b
MyVector GEwithScaledPartialPivoting(MyMatrix& A, MyVector& b, int size){
	int* pvt = new int[size];			//Pivoting vector
	double* scale = new double[size];		//Scaling vector
	for (int i = 0; i < size; i++)			//Initializing pivoting vector
		pvt[i] = i;

	for (int i = 0; i < size; i++){			//Loops 'size' times
		scale[i] = 0;				//Initialize scale to 0
		for (int j = 0; j < size; j++){		//Loops size times 
			if (fabs(scale[i]) < fabs(A(i, j)))	//f the next position is larger
				scale[i] = fabs(A(i, j));	// set it for the scale value
		}
	}

	double temp;					//Variable to check each position
	for (int k = 0; k < size - 1; k++){		//Main loop
		int pc = k;				//pivot column
		double aet = fabs(A(pvt[k], k) / scale[k]);
		for (int i = k + 1; i < size; i++){	//Loops 'size'-1 times
			temp = fabs(A(pvt[i], k) / scale[i]);
			if (temp > aet){			//Comparing each position with aet
				aet = temp;		//If temp>aet, set aet = to temp
				pc = i;			//and set pivot column to i
			}
		}
		if (pc != k){			 	//If pivot column was changed, we 
			int tempNum = pvt[k];		//need to switch the pivot values
			pvt[k] = pvt[pc];
			pvt[pc] = tempNum;
		}
		for (int i = k + 1; i < size; i++){	//Now we will shift the rows in the matrix A
			if (A(pvt[i], k) != 0){
				double factor = A(pvt[i], k) / A(pvt[k], k);
				A(pvt[i], k) = factor;
				for (int j = k + 1; j < size; j++)
					A(pvt[i], j) -= factor*A(pvt[k], j);
			}
		}

	}
	cout << "Printing after SPP: " << endl;		//Printing to console to see result 
	A.print();
	b.print();
	return SolveWithGEandBS(A, b, size);			//Now plug the adjusted matrix GE function
}
#pragma endregion

#pragma region Finding Eigenvalues
//Function to find the largest eigenvalue of a matrix
//    using the Power Method
//The first parameter, A, is the matrix
//The second parameter, v0, is the vector with the initial guess
double powerMethod(MyMatrix A, MyVector v0){
	int size = v0.size();					//Initializing the size
	int iter = 0, maxIter = 1000;				//Initializing values for iterators
	double tol = .000001;					//Initializing value for tolerance
	double error = tol * 10;					//Initializing the error
	double lamda0 = 0, lamda1;				//Initializing λ0 and λ1
	MyVector vk = MyVector(size);				//Initializing vector for vk

	while (error > tol&&iter < maxIter){			//Main loop
		vk = matrixVectorProduct(A, v0, size);		//Compute vk = Av0
		vk = scalarVectorProduct(vk,		//Compute vk = (1/||vk||)*vk
			(1 / l2Norm(vk)), size);
		lamda1 = dotProduct(vk,				//Compute λ1 = <vk^T,Avk>
			matrixVectorProduct(A, vk, size));
		error = abs(lamda1 - lamda0);			//Update error
		lamda0 = lamda1;					//Update λ0
		v0 = vk;						//Update v0
		iter++;						//Update iterator
	}
	return lamda1;						//Return resulting λ1
}

//Function to find the smallest eigenvalue of a matrix
//    using the Inverse Power Method
//The first parameter, A, is the matrix
//The second parameter, v0, is the vector with the initial guess
double inversePowerMethod(MyMatrix A, MyVector v0){
	int size = v0.size();				//Initializing the size
	int iter = 0, maxIter = 1000;			//Initializing values for iterators
	double tol = .0001;				//Initializing value for tolerance
	double error = tol * 10;				//Initializing the error
	double lamda0 = 0, lamda1;			//Initializing λ0 and λ1
	MyVector vk = MyVector(size);			//Initializing vector for vk
	MyVector y = LUwithScaledPartialPivoting(A,	//Initializing y by solving Ay=v0
		v0, size);

	while (error > tol&&iter < maxIter){		//Main loop
		vk = scalarVectorProduct(y,	//Compute vk = (1/||vk||)*vk
			(1 / l2Norm(y)), size);
		y = LUwithScaledPartialPivoting(A,	//Update y, solving Ay=vk
			vk, size);
		lamda1 = 1 / dotProduct(vk, y);		//Compute λ1 = 1/<vk^T,y>
		error = abs(lamda1 - lamda0);		//Updating error
		lamda0 = lamda1;				//Updating λ0 
		v0 = vk;					//Updating v0
		iter++;					//Updating iterator
	}
	return lamda1;					//Return resulting λ1
}
#pragma endregion

#endif