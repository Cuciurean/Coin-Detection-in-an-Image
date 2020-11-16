// OpenCVApplication.cpp : Defines the entry point for the console application.
//
#include "stdafx.h"
#include "common.h"
#include <random>


#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "iostream"

using namespace cv;
using namespace std;

void testOpenImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		imshow("image",src);
		waitKey();
	}
}

void testOpenImagesFld()
{
	char folderName[MAX_PATH];
	if (openFolderDlg(folderName)==0)
		return;
	char fname[MAX_PATH];
	FileGetter fg(folderName,"bmp");
	while(fg.getNextAbsFile(fname))
	{
		Mat src;
		src = imread(fname);
		imshow(fg.getFoundFileName(),src);
		if (waitKey()==27) //ESC pressed
			break;
	}
}

void testImageOpenAndSave()
{
	Mat src, dst;

	src = imread("Images/Lena_24bits.bmp", CV_LOAD_IMAGE_COLOR);	// Read the image

	if (!src.data)	// Check for invalid input
	{
		printf("Could not open or find the image\n");
		return;
	}

	// Get the image resolution
	Size src_size = Size(src.cols, src.rows);

	// Display window
	const char* WIN_SRC = "Src"; //window for the source image
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Dst"; //window for the destination (processed) image
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, src_size.width + 10, 0);

	cvtColor(src, dst, CV_BGR2GRAY); //converts the source image to a grayscale one

	imwrite("Images/Lena_24bits_gray.bmp", dst); //writes the destination to file

	imshow(WIN_SRC, src);
	imshow(WIN_DST, dst);

	printf("Press any key to continue ...\n");
	waitKey(0);
}

void testNegativeImage()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		double t = (double)getTickCount(); // Get the current time [s]
		
		Mat src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = Mat(height,width,CV_8UC1);
		// Asa se acceseaaza pixelii individuali pt. o imagine cu 8 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				uchar val = src.at<uchar>(i,j);
				uchar neg = MAX_PATH-val;
				dst.at<uchar>(i,j) = neg;
			}
		}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image",src);
		imshow("negative image",dst);
		waitKey();
	}
}

void testParcurgereSimplaDiblookStyle()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname, CV_LOAD_IMAGE_GRAYSCALE);
		int height = src.rows;
		int width = src.cols;
		Mat dst = src.clone();

		double t = (double)getTickCount(); // Get the current time [s]

		// the fastest approach using the “diblook style”
		uchar *lpSrc = src.data;
		uchar *lpDst = dst.data;
		int w = src.step; // no dword alignment is done !!!
		for (int i = 0; i<height; i++)
			for (int j = 0; j < width; j++) {
				uchar val = lpSrc[i*w + j];
				lpDst[i*w + j] = 255 - val;
			}

		// Get the current time again and compute the time difference [s]
		t = ((double)getTickCount() - t) / getTickFrequency();
		// Print (in the console window) the processing time in [ms] 
		printf("Time = %.3f [ms]\n", t * 1000);

		imshow("input image", src);
		imshow("negative image", dst);
		waitKey();
	}
}

void testColor2Gray()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src = imread(fname);

		int height = src.rows;
		int width = src.cols;

		Mat dst = Mat(height,width,CV_8UC1);
		
		// Asa se acceseaaza pixelii individuali pt. o imagine RGB 24 biti/pixel
		// Varianta ineficienta (lenta)
		for (int i=0; i<height; i++)
		{
			for (int j=0; j<width; j++)
			{
				Vec3b v3 = src.at<Vec3b>(i,j);
				uchar b = v3[0];
				uchar g = v3[1];
				uchar r = v3[2];
				dst.at<uchar>(i,j) = (r+g+b)/3;
			}
		}
		
		imshow("input image",src);
		imshow("gray image",dst);
		waitKey();
	}
}

void testBGR2HSV()
{
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		Mat src = imread(fname);
		int height = src.rows;
		int width = src.cols;

		// Componentele d eculoare ale modelului HSV
		Mat H = Mat(height, width, CV_8UC1);
		Mat S = Mat(height, width, CV_8UC1);
		Mat V = Mat(height, width, CV_8UC1);

		// definire pointeri la matricele (8 biti/pixeli) folosite la afisarea componentelor individuale H,S,V
		uchar* lpH = H.data;
		uchar* lpS = S.data;
		uchar* lpV = V.data;

		Mat hsvImg;
		cvtColor(src, hsvImg, CV_BGR2HSV);

		// definire pointer la matricea (24 biti/pixeli) a imaginii HSV
		uchar* hsvDataPtr = hsvImg.data;

		for (int i = 0; i<height; i++)
		{
			for (int j = 0; j<width; j++)
			{
				int hi = i*width * 3 + j * 3;
				int gi = i*width + j;

				lpH[gi] = hsvDataPtr[hi] * 510 / 360;		// lpH = 0 .. 255
				lpS[gi] = hsvDataPtr[hi + 1];			// lpS = 0 .. 255
				lpV[gi] = hsvDataPtr[hi + 2];			// lpV = 0 .. 255
			}
		}

		imshow("input image", src);
		imshow("H", H);
		imshow("S", S);
		imshow("V", V);

		waitKey();
	}
}

void testResize()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src;
		src = imread(fname);
		Mat dst1,dst2;
		//without interpolation
		resizeImg(src,dst1,320,false);
		//with interpolation
		resizeImg(src,dst2,320,true);
		imshow("input image",src);
		imshow("resized image (without interpolation)",dst1);
		imshow("resized image (with interpolation)",dst2);
		waitKey();
	}
}

void testCanny()
{
	char fname[MAX_PATH];
	while(openFileDlg(fname))
	{
		Mat src,dst,gauss;
		src = imread(fname,CV_LOAD_IMAGE_GRAYSCALE);
		int k = 0.4;
		int pH = 50;
		int pL = k*pH;
		GaussianBlur(src, gauss, Size(5, 5), 0.8, 0.8);
		Canny(gauss,dst,pL,pH,3);
		imshow("input image",src);
		imshow("canny",dst);
		waitKey();
	}
}

void testVideoSequence()
{
	VideoCapture cap("Videos/rubic.avi"); // off-line video from file
	//VideoCapture cap(0);	// live video from web cam
	if (!cap.isOpened()) {
		printf("Cannot open video capture device.\n");
		waitKey(0);
		return;
	}
		
	Mat edges;
	Mat frame;
	char c;

	while (cap.read(frame))
	{
		Mat grayFrame;
		cvtColor(frame, grayFrame, CV_BGR2GRAY);
		Canny(grayFrame,edges,40,100,3);
		imshow("source", frame);
		imshow("gray", grayFrame);
		imshow("edges", edges);
		c = cvWaitKey(0);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished\n"); 
			break;  //ESC pressed
		};
	}
}


void testSnap()
{
	VideoCapture cap(0); // open the deafult camera (i.e. the built in web cam)
	if (!cap.isOpened()) // openenig the video device failed
	{
		printf("Cannot open video capture device.\n");
		return;
	}

	Mat frame;
	char numberStr[256];
	char fileName[256];
	
	// video resolution
	Size capS = Size((int)cap.get(CV_CAP_PROP_FRAME_WIDTH),
		(int)cap.get(CV_CAP_PROP_FRAME_HEIGHT));

	// Display window
	const char* WIN_SRC = "Src"; //window for the source frame
	namedWindow(WIN_SRC, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_SRC, 0, 0);

	const char* WIN_DST = "Snapped"; //window for showing the snapped frame
	namedWindow(WIN_DST, CV_WINDOW_AUTOSIZE);
	cvMoveWindow(WIN_DST, capS.width + 10, 0);

	char c;
	int frameNum = -1;
	int frameCount = 0;

	for (;;)
	{
		cap >> frame; // get a new frame from camera
		if (frame.empty())
		{
			printf("End of the video file\n");
			break;
		}

		++frameNum;
		
		imshow(WIN_SRC, frame);

		c = cvWaitKey(10);  // waits a key press to advance to the next frame
		if (c == 27) {
			// press ESC to exit
			printf("ESC pressed - capture finished");
			break;  //ESC pressed
		}
		if (c == 115){ //'s' pressed - snapp the image to a file
			frameCount++;
			fileName[0] = NULL;
			sprintf(numberStr, "%d", frameCount);
			strcat(fileName, "Images/A");
			strcat(fileName, numberStr);
			strcat(fileName, ".bmp");
			bool bSuccess = imwrite(fileName, frame);
			if (!bSuccess) 
			{
				printf("Error writing the snapped image\n");
			}
			else
				imshow(WIN_DST, frame);
		}
	}

}

void MyCallBackFunc(int event, int x, int y, int flags, void* param)
{
	//More examples: http://opencvexamples.blogspot.com/2014/01/detect-mouse-clicks-and-moves-on-image.html
	Mat* src = (Mat*)param;
	if (event == CV_EVENT_LBUTTONDOWN)
		{
			printf("Pos(x,y): %d,%d  Color(RGB): %d,%d,%d\n",
				x, y,
				(int)(*src).at<Vec3b>(y, x)[2],
				(int)(*src).at<Vec3b>(y, x)[1],
				(int)(*src).at<Vec3b>(y, x)[0]);
		}
}

void testMouseClick()
{
	Mat src;
	// Read image from file 
	char fname[MAX_PATH];
	while (openFileDlg(fname))
	{
		src = imread(fname);
		//Create a window
		namedWindow("My Window", 1);

		//set the callback function for any mouse event
		setMouseCallback("My Window", MyCallBackFunc, &src);

		//show the image
		imshow("My Window", src);

		// Wait until user press some key
		waitKey(0);
	}
}




/////***************************/////
/************Proiect****************/
/////***************************/////


void cerc()
{

	Mat img;
	img = imread("imagini/multe.jpg", CV_BGR2GRAY);

	
	/****** Canny Edge *******/
	Mat imgBlurred;
	Mat imgCanny;
	Mat img1;
    GaussianBlur(img, imgBlurred,Size(5, 5),1.5);
	Canny(imgBlurred,imgCanny,100,200);

	img1 = 255 - imgCanny;

	/*NUmarul de puncte din imagine*/
	std::vector<Point> points;
	for (int i = 0; i < img1.rows; i++) {
		for (int j = 0; j < img1.cols; j++) {
			if (img1.at<uchar>(i, j) == 0) {
				Point p;
				p.x = j;
				p.y = i;
				points.push_back(p);
			}
		}
	}
	printf("\n");
	printf("Numarul de puncte %d \n", points.size());


	float t = 5.0, p = 0.99, q = 0.3, s = 3.0;
	float T = 0.0, N = 0.0;

	/*Numarul de esantioane*/
	N = log(1.0 - p) / log(1.0 - pow(q, s));
	N = 1000;
	/*pragul T*/
	T = points.size() * q;
	
	Point center;
	Point bestcenter;
	float radius;
	float distanta;
	bool finish = false;
	float MAXinliners = 0;
	float bestradius;


	int maxInliners = 0, parameterA, parameterB, parameterC;

	/*N incercari pentru gasirea cercului cel mai bun*/
	for (int i = 0; i < N; i++) {

		if (finish) {
			break;
		}
	
		/*cele 3 puncte*/
		int Point1 = rand() % (int)points.size();
		int Point2 = rand() % (int)points.size();
		int Point3 = rand() % (int)points.size();

		/*sa nu fie acelasi punct - puncte diferite*/
		if (Point1 == Point2) {
			Point2 = rand() % (int)points.size();
		}
		else if (Point1 == Point3)
		{
			Point3 = rand() % (int)points.size();
		}
		else if (Point2 == Point3)
		{
			Point3 = rand() % (int)points.size();
		}

		/*Cele 3 puncte*/
		Point p1, p2, p3;
		p1 = points.at(Point1);
		p2 = points.at(Point2);
		p3 = points.at(Point3);

		/*Coordonate puncte*/
		float x1 = p1.x;
		float x2 = p2.x;
		float x3 = p3.x;

		float y1 = p1.y;
		float y2 = p2.y;
		float y3 = p3.y;

		/*Coordonatele centrului*/
		center.x = (x1*x1 + y1 * y1)*(y2 - y3) + (x2*x2 + y2 * y2)*(y3 - y1) + (x3*x3 + y3 * y3)*(y1 - y2);

		if ((2 * (x1*(y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2)) != 0)
			center.x /= (2 * (x1*(y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2));

		if ((x1*x1 + y1 * y1)*(x3 - x2) + (x2*x2 + y2 * y2)*(x1 - x3) + (x3*x3 + y3 * y3)*(x2 - x1) != 0)
			center.y = (x1*x1 + y1 * y1)*(x3 - x2) + (x2*x2 + y2 * y2)*(x1 - x3) + (x3*x3 + y3 * y3)*(x2 - x1);

		center.y /= (2 * (x1*(y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2));

		/*Raza cercului*/
		radius = sqrt((center.x - x1)*(center.x - x1) + (center.y - y1)*(center.y - y1));


		int inliners = 0;
		for (int k = 0; k < points.size(); k++) {
			/*Calculul distantei dintre un punct (x0, y0) si un cerc*/
			/*distanta lui k fata de centru */
			distanta = abs(sqrt((points[k].x - center.x)*(points[k].x - center.x) + (points[k].y - center.y)*(points[k].y - center.y)) - radius);
			/*numararea punctelor distanta mai mica decat t*/
			if (distanta < t) {
				inliners++;
			}
		}

		if (inliners > T) {
			finish = true;
		}

		/*max inliners raza si center */
		if (inliners > MAXinliners)
		{
			MAXinliners = inliners;
			bestcenter = center;
			bestradius = radius;

		}	
		
	}
	circle(img, bestcenter, bestradius, Scalar(0, 255, 0), 2);



	printf("\n");
	printf("N= %f \n", N);
	printf("\n");
	printf("T= %f \n", T);
	printf("\n");
	printf("center.x= %d \n", center.x);
	printf("\n");
	printf("center.y= %d \n", center.y);
	printf("\n");
	printf("Radius= %f \n", radius);
	printf("\n");
	printf("Distanta= %f \n", distanta);
	

	//namedWindow("Cerc", CV_WINDOW_AUTOSIZE);
	imshow("Cerc", img);

	//namedWindow("CannyEdge", CV_WINDOW_AUTOSIZE);
	imshow("CannyEdge", imgCanny);

	waitKey(0);

}





int main()
{
	int op;
	do
	{
		system("cls");
		destroyAllWindows();
		printf("Menu:\n");
		printf(" 1 - Cerc!!\n");

		printf(" 0 - Exit\n\n");
		printf("Option: ");
		scanf("%d",&op);
		switch (op)
		{
			case 1:
				cerc();
				break;
				testMouseClick();
				break;
		}
	}
	while (op!=0);
	return 0;
}