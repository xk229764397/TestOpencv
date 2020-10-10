#include<opencv2/opencv.hpp>

#include<iostream>

using namespace cv;
using namespace std;

//int main(int argc, char** argv) {
//	Mat src = imread("C:/Users/xk/Desktop/1.jpg");
//	Mat dst = Mat::zeros(src.size(),src.type());
//	if (!src.data)
//	{
//		cout << "could not load";
//		return -1;
//	}
//	namedWindow("src", WINDOW_AUTOSIZE);
//	imshow("src", src);
//
//	int cols = (src.cols-1) * src.channels();
//	int offsets = src.channels();
//	int rows = src.rows;
//	for (int row = 1; row < rows - 1; row++)
//	{
//		const uchar* precious = src.ptr<uchar>(row - 1);
//		const uchar* current = src.ptr<uchar>(row);
//		const uchar* next = src.ptr<uchar>(row + 1);
//		uchar* output = dst.ptr<uchar>(row);
//		for (int col = offsets; col < cols; col++)
//		{
//			output[col] = saturate_cast<uchar> (5 * current[col] - (current[col - offsets] + current[col + offsets] + precious[col] + next[col]));
//			
//		}
//	}
//
//	namedWindow("dst", WINDOW_AUTOSIZE);
//	imshow("dst", dst);
//
//	double t = getTickCount();
//
//	Mat kernel = (Mat_<char>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
//	filter2D(src, dst, src.depth(), kernel);
//
//	double timeconsume = (getTickCount() - t) / getTickFrequency();
//	cout << "time consume:" << timeconsume;
//
//	namedWindow("dst2", WINDOW_AUTOSIZE);
//	imshow("dst2", dst);
//	waitKey(0);
//	return 0;
//}





//int main()
//{
//	Mat src = imread("C:/Users/xk/Desktop/1.jpg");
//	namedWindow("src", WINDOW_AUTOSIZE);
//	imshow("src", src);
//	/*Mat dst;
//	dst = Mat(src.size(), src.type());
//	dst = Scalar(255, 255, 255);
//	namedWindow("dst", WINDOW_AUTOSIZE);
//	imshow("dst", dst);*/
//
//	Mat dst;
//	namedWindow("dst", WINDOW_AUTOSIZE);
//	cvtColor(src, dst, CV_BGR2GRAY);
//	imshow("dst", dst);
//	cout << "src channels:" << src.channels() << endl;
//	cout << "dst channels:" << dst.channels() << endl;
//	
//	const uchar* firstRow = dst.ptr<uchar>(0);
//	cout << "first pixel value:" << (int)*firstRow << endl;
//	int cols = dst.cols;
//	int rows = dst.rows;
//	cout << "cols:" << cols << " rows:" << rows << endl;
//
//	Mat M(100, 100, CV_8UC3, Scalar(0, 0, 255));
//	//cout << "M=" << endl << M << endl;
//
//	Mat m1;
//	m1.create(src.size(), src.type());
//	m1 = Scalar(0, 0, 255);
//
//	imshow("dst", m1);
//	waitKey(0);
//	return 0;
//}



//int main()
//{
//	Mat src = imread("C:/Users/xk/Desktop/1.jpg");
//	namedWindow("src", WINDOW_AUTOSIZE);
//	imshow("src", src);
//	Mat gray;
//	cvtColor(src, gray, CV_BGR2GRAY);
//	int height = gray.rows;
//	int width = gray.cols;
//	//namedWindow("output", WINDOW_AUTOSIZE);
//	//imshow("output", gray);
//	//单通道
//	for (int row = 0; row < height; row++)
//	{
//		for (int col = 0; col < width; col++)
//		{
//			int gray_value = gray.at<uchar>(row, col);
//			gray.at<uchar>(row, col) = 255 - gray_value;
//		}
//	}
//	//imshow("output2", gray);
//	Mat dst;
//	dst.create(src.size(), src.type());
//	height = src.rows;
//	width = src.cols;
//	int channels = src.channels();
//	for (int row = 0; row < height; row++)
//	{
//		for (int col = 0; col < width; col++)
//		{
//			if (channels == 1)
//			{
//				int gray_value = gray.at<uchar>(row, col);
//				gray.at<uchar>(row, col) = 255 - gray_value;
//			}
//			else if (channels == 3)
//			{
//				for (int channel = 0; channel < channels; channel++)
//				{
//					dst.at<Vec3b>(row, col)[channel] = 255 - src.at<Vec3b>(row, col)[channel];
//				}
//				
//			}
//		}
//		
//	}
//	bitwise_not(src, dst);
//	imshow("output3", dst);
//
//	waitKey(0);
//	return 0;
//}



//int main()
//{
//	Mat src1 = imread("C:/Users/xk/Desktop/b1.jpg");
//	Mat src2 = imread("C:/Users/xk/Desktop/b2.jpg");
//	Mat dst;
//
//	double alpha = 0.5;
//	if (src1.rows == src2.rows && src1.cols == src2.cols && src1.type() == src2.type())
//	{
//		addWeighted(src1, alpha, src2, (1.0 - alpha), 0.0, dst);
//		//add(src1, src2, dst, Mat());
//		//multiply(src1, src2, dst, 1.0);
//		imshow("src1", src1);
//		imshow("src2", src2);
//		imshow("output", dst);
//	}
//	else
//	{
//		cout << "不匹配";
//	}
//	
//
//	waitKey(0);
//	return 0;
//}



//int main()
//{
//	Mat src ,dst,dst2;
//	src = imread("C:/Users/xk/Desktop/1.jpg");
//	if (!src.data)
//	{
//		cout << "could not load...";
//		return -1;
//	}
//	imshow("input image", src);
//
//	int height = src.rows;
//	int width = src.cols;
//	dst = Mat::zeros(src.size(), src.type());
//	dst2 = Mat::zeros(src.size(), src.type());
//	float alpha = 0.8;
//	float beta = 100;
//
//	Mat m1;
//	src.convertTo(m1, CV_32F);
//
//	for (int row = 0; row < height; row++)
//	{
//		for (int col = 0; col < width; col++)
//		{
//			if (src.channels() == 3)
//			{
//				float b = m1.at<Vec3f>(row, col)[0];
//				float g = m1.at<Vec3f>(row, col)[1];
//				float r = m1.at<Vec3f>(row, col)[2];
//				float b2 = src.at<Vec3b>(row, col)[0];
//				float g2 = src.at<Vec3b>(row, col)[1];
//				float r2 = src.at<Vec3b>(row, col)[2];
//				dst.at<Vec3b>(row,col)[0]	= saturate_cast<uchar>(b * alpha + beta);
//				dst.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g * alpha + beta);
//				dst.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r * alpha + beta);
//				dst2.at<Vec3b>(row, col)[0] = saturate_cast<uchar>(b2 * alpha + beta);
//				dst2.at<Vec3b>(row, col)[1] = saturate_cast<uchar>(g2 * alpha + beta);
//				dst2.at<Vec3b>(row, col)[2] = saturate_cast<uchar>(r2 * alpha + beta);
//			}
//			else if (src.channels() == 1)
//			{
//				float v = src.at<uchar>(row, col);
//				dst.at<uchar>(row, col) = saturate_cast<uchar>(v * alpha + beta);
//			}
//		}
//	}
//
//	imshow("contrast and brightness change demo-m1", dst);
//	imshow("contrast and brightness change demo-src", dst2);
//	waitKey(0);
//	return 0;
//}




//Mat bgImage;
//const char* drawdemo_win = "draw shapes and text demo";
//void Mylines()
//{
//	Point p1 = Point(200, 100);
//	Point p2;
//	p2.x = 500;
//	p2.y = 400;
//	Scalar color = Scalar(0, 0, 255);
//	line(bgImage, p1, p2, color, 1, LINE_8);
//}
//void Myrectangle()
//{
//	Rect rect = Rect(200, 100, 300, 300);
//	Scalar color = Scalar(255, 0, 0);
//	rectangle(bgImage, rect, color, 1, LINE_8);
//}
//void Myellipse()
//{
//	Scalar color = Scalar(0, 255, 0);
//	ellipse(bgImage, Point(bgImage.cols / 2, bgImage.rows / 2), Size(bgImage.cols / 4, bgImage.rows / 8), 90, 0,360,color,1,LINE_8);
//
//}
//void Mycircle()
//{
//	Scalar color = Scalar(0, 255, 255);
//	circle(bgImage, Point(bgImage.cols / 2, bgImage.rows / 2), 150, color, 2, 8);
//}
//void Mypolygon()
//{
//	Point pts[1][5];
//	pts[0][0] = Point(100, 100);
//	pts[0][1] = Point(100, 200);
//	pts[0][2] = Point(200, 200);
//	pts[0][3] = Point(200, 100);
//	pts[0][4] = Point(100, 100);
//	const Point* ppts[] = { pts[0] };
//	int ppt[] = { 5 };
//	Scalar color = Scalar(255, 0, 255);
//	fillPoly(bgImage, ppts, ppt, 1, color, 8);
//}
//void RandomLineDemo()
//{
//	RNG rng(12345);
//	Point pt1, pt2;
//	Mat bg = Mat::zeros(bgImage.size(), bgImage.type());
//
//	for (int i = 0; i < 100000; i++)
//	{
//		pt1.x = rng.uniform(0, bgImage.cols);
//		pt2.x = rng.uniform(0, bgImage.cols);
//		pt1.y = rng.uniform(0, bgImage.rows);
//		pt2.y = rng.uniform(0, bgImage.rows);
//		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//		if (waitKey(50) > 0) break;
//		line(bgImage, pt1, pt2, color, 1, 8);
//		imshow(drawdemo_win, bgImage);
//	}
//	
//}
//int main()
//{
//	bgImage = imread("C:/Users/xk/Desktop/1.jpg");
//	if (!bgImage.data)
//	{
//		cout << "could not load...";
//		return -1;
//	}
//	Mylines();
//	Myrectangle();
//	Myellipse();
//	Mycircle();
//	Mypolygon();
//
//	putText(bgImage, "Hello OpenCV", Point(300, 300), CV_FONT_BLACK, 2.0, Scalar(180, 125, 200), 3, 8);
//	imshow(drawdemo_win, bgImage);
//	RandomLineDemo();
//	waitKey(0);
//	return 0;
//}



//bool matIsEqual(const cv::Mat mat1, const cv::Mat mat2) {
//	if (mat1.empty() && mat2.empty()) {
//		return true;
//	}
//	if (mat1.cols != mat2.cols || mat1.rows != mat2.rows || mat1.dims != mat2.dims ||
//		mat1.channels() != mat2.channels()) {
//		return false;
//	}
//	if (mat1.size() != mat2.size() || mat1.channels() != mat2.channels() || mat1.type() != mat2.type()) {
//		return false;
//	}
//	int nrOfElements1 = mat1.total() * mat1.elemSize();
//	if (nrOfElements1 != mat2.total() * mat2.elemSize()) return false;
//	bool lvRet = memcmp(mat1.data, mat2.data, nrOfElements1) == 0;
//	return lvRet;
//}
//均值模糊和高斯模糊
//int main()
//{
//	Mat src, dst,dst2;
//	src = imread("C:/Users/xk/Desktop/b2.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	dst2 = Mat::zeros(src.size(), src.type());
//	char input_title[] = "intput image";
//	char output_title[] = "blur image";
//	imshow(input_title, src);
//
//	blur(src, dst, Size(3, 3), Point(-1, -1));
//	imshow("dst", dst);
//	Mat kernel = (Mat_<double>(3, 3) << 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9, 1.0/9);
//	filter2D(src, dst2, src.depth(), kernel);
//	imshow("dst2", dst2);
//	if(matIsEqual(dst,dst2)) cout<<"=";
//	Mat gs;
//	GaussianBlur(src, gs, Size(11, 11), 11, 11);
//	//imshow("高斯blur", gs);
//	
//	waitKey(0);
//	return 0;
//} 




//中值模糊(去椒盐噪声)、双边模糊(相比高斯保留了边缘像素)
//int main()
//{
//	Mat src, dst, dst2, dst3;
//	src = imread("C:/Users/xk/Desktop/1.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("input image", src);
//
//	//medianBlur(src, dst, 3);
//	//imshow("dst",dst);
//
//	bilateralFilter(src, dst2, 15, 150, 3);
//	imshow("dst2",dst2);
//
//	Mat kernel = (Mat_<int>(3, 3) << 0, -1, 0, -1, 5, -1, 0, -1, 0);
//	filter2D(dst2, dst3, dst2.depth(), kernel);
//	
//	//GaussianBlur(src, dst3, Size(15, 15), 3, 3);
//	imshow("dst3", dst3);
//
//	waitKey(0);
//	return 0;
//}




//Mat src, dst, dst2, dst3;
//int element_size = 3;
//int max_size = 21;
//void CallBack_Demo(int, void*)
//{
//	int s = element_size * 2 + 1;
//	Mat structureElement = getStructuringElement(MORPH_RECT, Size(s, s), Point(-1, -1));
//	//dilate(src, dst, structureElement, Point(-1, -1), 1);   //膨胀
//	erode(src, dst, structureElement, Point(-1, -1), 1);      //腐蚀
//	imshow("output", dst);
//}
//int main()
//{
//
//	src = imread("C:/Users/xk/Desktop/p1.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//
//	namedWindow("output", WINDOW_AUTOSIZE);
//	createTrackbar("Element Size:", "output", &element_size, max_size, CallBack_Demo);
//	CallBack_Demo(0, 0);
//
//
//	waitKey(0);
//	return 0;
//}




//开操作、闭操作、形态学梯度、顶帽、黑帽
//int main()
//{
//	Mat src, dst;
//	src = imread("C:/Users/xk/Desktop/hb.png");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//
//	Mat kernel = getStructuringElement(MORPH_RECT, Size(11, 11), Point(-1, -1));
//	morphologyEx(src, dst, CV_MOP_BLACKHAT, kernel);
//	imshow("output", dst);
//
//	waitKey(0);
//	return 0;
//}




//提取水平与垂直线
//int main()
//{
//	Mat src, dst,gray_src;
//	src = imread("C:/Users/xk/Desktop/3.png");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//
//	cvtColor(src, gray_src, CV_BGR2GRAY);
//	imshow("gray_src", gray_src);
//
//	Mat binImg;
//	adaptiveThreshold(~gray_src, binImg, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY,15, -2);
//	imshow("binary Image", binImg);
//	//水平结构元素	
//	Mat hline = getStructuringElement(MORPH_RECT, Size(src.cols / 16, 1), Point(-1, -1));
//	//垂直结构元素
//	Mat vline = getStructuringElement(MORPH_RECT, Size(1, src.rows / 16), Point(-1, -1));
//	//矩形结构
//	Mat kernel = getStructuringElement(MORPH_RECT, Size(2, 2), Point(-1, -1));
//
//	Mat temp;
//	erode(binImg, temp, kernel);	
//	dilate(temp, dst, kernel);
//	//morphologyEx(binImg, dst, CV_MOP_OPEN, vline);    e+d=m开操作
//	bitwise_not(dst, dst);
//	//blur(dst, dst, Size(3, 3), Point(-1, -1));
//	imshow("Final Result",dst);
//
//	waitKey(0);
//	return 0;
//}




//int main()
//{
//	Mat src, dst;
//	src = imread("C:/Users/xk/Desktop/p1.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//
//	//上采样
//	pyrUp(src, dst, Size(src.cols * 2, src.rows * 2));
//	imshow("up", dst);
//
//	//降采样
//	Mat down;
//	pyrDown(src, down, Size(src.cols / 2, src.rows / 2));
//	imshow("down", down);
//
//	//DOG 高斯不同
//	Mat g1, g2, gray_src,dogImg;
//	cvtColor(src, gray_src, CV_BGR2GRAY);
//	GaussianBlur(gray_src, g1, Size(5, 5), 0, 0);
//	GaussianBlur(g1, g2, Size(5, 5), 0, 0);
//	subtract(g1, g2, dogImg, Mat());
//	//归一化显示
//	normalize(dogImg, dogImg, 255, 0, NORM_MINMAX);
//	imshow("dogImg", dogImg);
//
//	waitKey(0);
//	return 0;
//}




//基本阈值操作
//Mat src, dst,gray_src;
//int threshold_value = 127;
//int threshold_max = 255;
//int type_value = 2;
//int type_max = 4;
//void Threshold_Demo(int, void*)
//{
//	cvtColor(src, gray_src, CV_BGR2GRAY);
//	threshold(gray_src, dst, 0, 255, THRESH_TRIANGLE|type_value);
//	imshow("binary image", dst);
//}
//int main()
//{
//	src = imread("C:/Users/xk/Desktop/p1.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//	cvtColor(src, gray_src, CV_BGR2GRAY);
//	namedWindow("binary image", WINDOW_AUTOSIZE);
//	createTrackbar("Threshold Value", "binary image", &threshold_value, threshold_max, Threshold_Demo);
//	createTrackbar("Type Value", "binary image", &type_value, type_max, Threshold_Demo);
//	Threshold_Demo(0, 0);
//	
//
//	waitKey(0);
//	return 0;
//}




//int main()
//{
//	Mat src, dst,dst_x,dst_y, s_x,s_y,lpls,gray_src;
//	src = imread("C:/Users/xk/Desktop/p1.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//	
//	////Robert算子 X方向
//	//Mat kernel_x = (Mat_<int>(2, 2) << 1, 0, 0, -1);
//	//filter2D(src, dst_x, -1, kernel_x,Point(-1,-1),0.0);
//	//imshow("Robert x", dst_x);
//
//	////Robert算子 Y方向
//	//Mat kernel_y = (Mat_<int>(2, 2) << 0, 1, -1, 0);
//	//filter2D(src, dst_y, -1, kernel_y, Point(-1, -1), 0.0);
//	//imshow("Robert y", dst_y);
//
//	//Mat Sobel_x = (Mat_<int>(3, 3) << -1, 0, 1, -2, 0, 2, -1, 0, 1);
//	//filter2D(src, s_x, -1, Sobel_x, Point(-1, -1), 0.0);
//	//imshow("Sobel x", s_x);
//
//	//Mat Sobel_y = (Mat_<int>(3, 3) << -1, -2, -1, 0, 0, 0, 1, 2, 1);
//	//filter2D(src, s_y, -1, Sobel_y, Point(-1, -1), 0.0);
//	//imshow("Sobel y", s_y);
//
//	////拉普拉斯算子
//	//Mat Lpls = (Mat_<int>(3, 3) << 0,-1,0,-1,4,-1,0,-1,0);
//	//filter2D(src, lpls, -1, Lpls, Point(-1, -1), 0.0);
//	//imshow("Lpls x", lpls);
//
//	//自定义卷积模糊
//	int c = 0, index = 0, ksize = 3;
//	while (true)
//	{
//		c = waitKey(500);
//		if ((char)c == 27) break;
//		ksize = 4 + (index % 8) * 2 + 1;
//		Mat kernel = Mat::ones(Size(ksize, ksize), CV_32F) / (float)(ksize * ksize);
//		filter2D(src, dst, -1, kernel, Point(-1, -1),0);
//		index++;
//		imshow("custom blur", dst);
//
//	}
//
//
//	waitKey(0);
//	return 0;
//}




//卷积边缘问题
//int main()
//{
//	Mat src, dst,dst2,dst3,dst4;
//	src = imread("C:/Users/xk/Desktop/p1.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//
//	int top = (int)(0.05 * src.rows);
//	int bottom = (int)(0.05 * src.rows);
//	int left = (int)(0.05 * src.cols);
//	int right = (int)(0.05 * src.cols);
//	RNG rng(12345);
//	int borderType = BORDER_DEFAULT;
//
//	int c = 0;
//	//while (true)
//	//{
//	//	c = waitKey(500);
//	//	//ESC
//	//	if ((char)c == 27) break;
//	//	else if ((char)c == 'r') borderType = BORDER_REPLICATE;
//	//	else if ((char)c == 'w')borderType = BORDER_WRAP;
//	//	else if ((char)c == 'c') borderType = BORDER_CONSTANT;
//	//	Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//	//	copyMakeBorder(src, dst, top, bottom, left, right, borderType, color);
//	//	imshow("dst", dst);
//	//}
//
//	GaussianBlur(src, dst, Size(5, 5), 0, 0, BORDER_DEFAULT);
//	imshow("dst", dst);
//	GaussianBlur(src, dst2, Size(5, 5), 0, 0, BORDER_REPLICATE);
//	imshow("dst2", dst2);
//	GaussianBlur(src, dst3, Size(5, 5), 0, 0, BORDER_WRAP);
//	imshow("dst3", dst3);
//	GaussianBlur(src, dst4, Size(5, 5), 0, 0, BORDER_CONSTANT);
//	imshow("dst4", dst4);
//
//	waitKey(0);
//	return 0;
//}




//Sobel算子
//卷积应用-图像边缘提取
//int main()
//{
//	Mat src, dst;
//	src = imread("C:/Users/xk/Desktop/p1.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//
//
//	GaussianBlur(src, dst, Size(3, 3), 0, 0);
//	Mat gray_dst;
//	cvtColor(dst, gray_dst, CV_BGR2GRAY);
//	imshow("gray_dst", gray_dst);
//
//	Mat xgrad,ygrad;
//	Scharr(gray_dst, xgrad, CV_16S, 1, 0);
//	Scharr(gray_dst, ygrad, CV_16S, 0, 1);
//	//Sobel(gray_dst, xgrad, CV_16S, 1, 0, 3);
//	//Sobel(gray_dst, ygrad, CV_16S, 0, 1, 3);
//	convertScaleAbs(xgrad, xgrad);
//	convertScaleAbs(ygrad, ygrad);
//	imshow("xgrad", xgrad);
//	imshow("ygrad", ygrad);
//
//	Mat xygrad= Mat(xgrad.size(),xgrad.type());
//	cout << "type:" << xgrad.type();
//	int width = xgrad.cols;
//	int height = xgrad.rows;
//	for (int row = 0; row < height; row++)
//	{
//		for (int col = 0; col < width; col++)
//		{
//			int xg = xgrad.at<uchar>(row, col);
//			int yg = ygrad.at<uchar>(row, col);
//			int xy = xg + yg;
//			xygrad.at<uchar>(row, col) = saturate_cast<uchar>(xy);
//		}
//	}
//	//addWeighted(xgrad, 0.5, ygrad, 0.5, 0, xygrad);
//	imshow("xygrad", xygrad);
//
//	waitKey(0);
//	return 0;
//}



//Laplance算子
//int main()
//{
//	Mat src, dst;
//	src = imread("C:/Users/xk/Desktop/p1.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//
//
//	GaussianBlur(src, dst, Size(3, 3), 0, 0);
//	Mat gray_dst,edge_image;
//	cvtColor(dst, gray_dst, CV_BGR2GRAY);
//
//	Laplacian(gray_dst, edge_image, CV_16S, 3);
//	convertScaleAbs(edge_image, edge_image);
//
//	imshow("edge_image", edge_image);
//
//	threshold(edge_image, edge_image, 0, 255, THRESH_OTSU | THRESH_BINARY);
//	imshow("Laplacian image", edge_image);
//
//	waitKey(0);
//	return 0;
////}