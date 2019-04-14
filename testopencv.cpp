// testopencv.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <random>
#include "math.h"
#include <stdlib.h>
#include <stdio.h>

using namespace cv;
using namespace std;

Mat detect_edge(Mat image_test, int scale, int delta, int ddepth) {
Mat grad;
GaussianBlur(image_test, image_test, Size(3, 3), 0, 0, BORDER_DEFAULT);
Mat grad_x, grad_y;
Mat abs_grad_x, abs_grad_y;
Sobel(image_test, grad_x, ddepth, 1, 0, 3, scale, delta, BORDER_DEFAULT);
convertScaleAbs(grad_x, abs_grad_x);
Sobel(image_test, grad_y, ddepth, 0, 1, 3, scale, delta, BORDER_DEFAULT);
convertScaleAbs(grad_y, abs_grad_y);
addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0, grad);
//imshow("Edge detection", grad);
return grad;
}
void transpose(float(&inputMatrix)[100][2], float(&outputMatrix)[2][100], int n, int m) {
	int i, j;
	for (i = 0; i < n; ++i) {
		for (j = 0; j < m; ++j)
		{
			outputMatrix[j][i] = inputMatrix[i][j];
		}
	}
}
void multiplyMatrices(float(&inputMatrix1)[2][100], float(&inputMatrix2)[100][2], float(&outputMatrix)[2][2], int n, int m) {
	//MatrixA[i][j] = MatrixB[i][k] * MatrixC[k][j]
	int i, j, k;
	for (i = 0; i < 2; ++i) {
		for (j = 0; j < m; ++j) {
			for (k = 0; k < 100; ++k) {
				outputMatrix[i][j] += inputMatrix1[i][k] * inputMatrix2[k][j];
			}
		}
	}
}
void inverse(float(&inputMatrix)[2][2], float(&outputMatrix)[2][2], float detA) {
	double invdet = 1 / detA;
	outputMatrix[0][0] = inputMatrix[1][1] * invdet;
	outputMatrix[0][1] = -1 * inputMatrix[0][1] * invdet;
	outputMatrix[1][0] = -1 * inputMatrix[1][0] * invdet;
	outputMatrix[1][1] = inputMatrix[0][0] * invdet;
}
float det(float(&inputMatrix)[2][2]) {
	return inputMatrix[0][0] * inputMatrix[1][1] - inputMatrix[1][0] * inputMatrix[0][1];
}
bool findIntersection(Point line1_1, Point line1_2, Point line2_1, Point line2_2, Point &intersection) {
	Point x = line2_1 - line1_1;
	Point d1 = line1_2 - line1_1;
	Point d2 = line2_2 - line2_1;

	float cross = d1.x*d2.y - d1.y*d2.x;
	if (fabs(cross) < 1e-8) {
		return false;
	}

	float t1 = (x.x*d2.y - x.y*d2.x) / cross;
	intersection = line1_1 + d1 * t1;
	return true;
}
int calculateIntersections(Point pt1[10000], Point pt2[10000], Point (&intersections)[100000], int size) {
	int counter = 0; Point temp;
	for (int i = 0; i < size; i++) {
		for (int j = 0; j < size; j++) {
			if (i != j) {
				if (findIntersection(pt1[i], pt2[i], pt1[j], pt2[j], temp)) {
					intersections[counter] = temp;
					//cout << "Intersection #" << i << "and " << j << ": (" << temp.x << ", " << temp.y << ")" << endl;
					counter++;
				}
			}
		}
	}
	return counter;
}
float calculateEuclideanDistance(Point pt1, Point pt2) {
	float x = pt1.x - pt2.x;
	float y = pt1.y - pt2.y;
	return sqrt(pow(x, 2) + pow(y, 2));
}
vector<Point> clearOffImage(vector<Point> &intersections){
	vector<Point> newInters;
	for (int i = 0; i < intersections.size(); i++) {
		if (intersections.at(i).x >= 0 && intersections.at(i).y >= 0) {
			newInters.push_back(intersections.at(i));
		}
	}
	return newInters;
}
char intToChar(int i /*limited to 0-3*/) {
	char ch;
	switch (i) {
	case 0:
		ch = '0';
		break;
	case 1:
		ch = '1';
		break;
	case 2:
		ch = '2';
		break;
	default:
		ch = 3;
		break;
	}
	return ch;
}
/*Mat singleVP(Mat cdst, float (&matrixA)[100][2], float (&matrixX)[2][2], float (&matrixRho)[100][2], vector<Vec2f> lines) {
	float matrixTA[2][100];
	transpose(matrixA, matrixTA, lines.size(), 2);
	float matrixATA[2][2];
	multiplyMatrices(matrixTA, matrixA, matrixATA, lines.size(), 2);
	float matrixATRho[2][2];
	multiplyMatrices(matrixTA, matrixRho, matrixATRho, lines.size(), 1);
	float detA = det(matrixATA);
	if (detA == 0) {
		cout << "Since the determinant is zero, the vanishing point seems to be at infinity. (off-image)" << endl;
	}
	else {
		float matrixIATA[2][2];
		inverse(matrixATA, matrixIATA, detA);
		float As[2][100], Rhos[100][2];
		for (int i = 0; i < 2; i++) {
			for (int j = 0; j < 2; j++) {
				As[i][j] = matrixIATA[i][j];
				Rhos[i][j] = matrixATRho[i][0];
			}
		}
		multiplyMatrices(As, Rhos, matrixX, 2, 1);
		Point vp;
		vp.x = matrixX[0][0];
		vp.y = matrixX[1][0];
		circle(cdst, vp, 3, CV_RGB(0, 255, 0), 1);
	}
	return cdst;
}
Mat multipleVP(int k, Mat cdst, vector<vector<Point>> linesegments) {
	//Calculate intersections
	vector<Point> intersections;
	vector<int> intersectcounts;
	calculateIntersections(intersections, intersectcounts, linesegments);

	//solve vanishing points with k-means
	vector<Point> vanishing_points;
	intersections = clearOffImage(intersections);
	k_means(intersections, vanishing_points, k, 5);
	for (int i = 0; i < k; i++) {
		cout << vanishing_points.at(i).x;
		printf("\t");
		cout << vanishing_points.at(i).y << endl;
		circle(cdst, vanishing_points.at(i), 3, CV_RGB(0, 0, 250), 1);
	}
	return cdst;
}*/

int main(int argc, char** argv)
{
	Mat image; int MAX_ITERATIONS = 100;
	image = imread("test33.jpg", CV_LOAD_IMAGE_COLOR); // Read the file
	if (!image.data) // Check for invalid input
	{
		cout << " Could not open or find the image " << std::endl;
		return -1;
	}
	cvtColor(image, image, CV_BGR2GRAY);
	imshow("Original image", image);
	
	int k;
	cout << "Input K :";
	cin >> k;
	cout << "Input max iterations:";
	cin >> MAX_ITERATIONS;

	//Edge Detection
	Mat image_test = image.clone();
	int scale = 1;
	int delta = 0;
	int ddepth = CV_16S;
	image_test = detect_edge(image_test, scale, delta, ddepth);
	Mat dst, cdst; dst = image_test.clone();
	Canny(dst, dst, 50, 100, 3);
	cvtColor(dst, cdst, CV_GRAY2BGR);
	cvtColor(image_test, image_test, CV_GRAY2BGR);
	Mat image_test2 = image_test.clone();

	//Hough Transform
	vector<Vec2f> lines;
	int houghThreshold = 0;
	Point pt1[10000], pt2[10000];
	HoughLines(dst, lines, 1, CV_PI / 180, 195, 0, 0);
	for (size_t i = 0; i < lines.size(); i++)
	{
		float rho = lines[i][0], theta = lines[i][1]; //A set of Hough peaks
		float a = cos(theta), b = sin(theta);
		float x0 = a*rho, y0 = b*rho;
		pt1[i].x = cvRound(x0 + 1000 * (-b));
		pt1[i].y = cvRound(y0 + 1000 * (a));
		pt2[i].x = cvRound(x0 - 1000 * (-b));
		pt2[i].y = cvRound(y0 - 1000 * (a));
		//Draw lines according to Hough peaks
		line(image_test, pt1[i], pt2[i], Scalar(0, 0, 255), 1, CV_AA);
	}
	//Calculate intersections
	int numOfIntersections;
	Point intersections[100000];
	numOfIntersections = calculateIntersections(pt1, pt2, intersections, lines.size());
	printf("Completed intersections calculation.\n");

	//Apply kmeans
	Point means[5];
	//Initialization
	char labels[100000]; int iteration = 0;
	int meanX[100000], meanY[100000], counters[100000];

	//Random k centroids
	float mindist = 100, dist;
	int r; double d = 0;	std::default_random_engine generator;
	std::uniform_int_distribution<int> distribution(0, numOfIntersections - 1);
	for (int i = 0; i < k; i++) {
		r = distribution(generator);
		means[i] = intersections[r];
		meanX[i] = 0; meanY[i] = 0; counters[i] = 0;
	}
	for (int i = 0; i < numOfIntersections; i++) {
		char ch = '1';
		labels[i] = ch;
	}

	int label;
	while (iteration < MAX_ITERATIONS) {
		//Label data to the nearest centroids
		for (int i = 0; i < numOfIntersections; i++) {
			for (int j = 0; j < k; j++) {
				//Calculate distance to the jth centroid
				dist = calculateEuclideanDistance(intersections[i], means[j]);
				if (mindist == 100)
					mindist = dist;
				if (dist <= mindist) {
					mindist = dist;
					labels[i] = intToChar(j);
					label = j;
				}
			}
			meanX[label] += intersections[i].x;
			meanY[label] += intersections[i].y;
			counters[label]++;
		}
		//Calculate new centroids
		for (int i = 0; i < k; i++) {
			cout << meanX[i]; printf("\t");
			cout << meanY[i] << endl;
			cout << counters[i] << endl;
			meanX[i] /= counters[i];
			meanY[i] /= counters[i];
			Point pt;
			pt.x = (int)meanX[i];
			pt.y = (int)meanY[i];
			means[i] = pt;
		}
		iteration++;
	}
	printf("Finished kmean\n");
	for (int i = 0; i < k; i++) {
		circle(cdst, means[i], 3, CV_RGB(0, 255, 0), 1);
	}

	imshow("Edge Detection", cdst);
	imshow("Hough lines (standard)", image_test);
	waitKey(0); // Wait for a keystroke in the window
	return 0;
}