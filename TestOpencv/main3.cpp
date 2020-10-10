#include<opencv2/opencv.hpp>

#include<iostream>

#include<math.h>

using namespace cv;

using namespace std;

//凸包
Mat src, src_gray, dst;
int threshold_value = 100;
int threshold_max = 255;
const char* output_win = "convex hull demo";
void Threshold_Callback(int, void*);
RNG rng(12345);
int main(int argc, char** argv) {
	src = imread("C:/Users/xk/Desktop/hand2.png");
	if (!src.data) {
		printf("could not load image...\n");
		return -1;
	}
	const char* input_win = "input image";
	namedWindow(input_win, CV_WINDOW_AUTOSIZE);
	namedWindow(output_win, CV_WINDOW_NORMAL);
	const char* trackbar_label = "Threshold : ";

	cvtColor(src, src_gray, CV_BGR2GRAY);
	blur(src_gray, src_gray, Size(3, 3), Point(-1, -1), BORDER_DEFAULT);
	imshow(input_win, src_gray);

	createTrackbar(trackbar_label, output_win, &threshold_value, threshold_max, Threshold_Callback);
	Threshold_Callback(0, 0);
	waitKey(0);
	return 0;
}

void Threshold_Callback(int, void*) {
	Mat bin_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierachy;

	threshold(src_gray, bin_output, threshold_value, threshold_max, THRESH_BINARY);
	findContours(bin_output, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));

	vector<vector<Point>> convexs(contours.size());
	for (size_t i = 0; i < contours.size(); i++) {
		convexHull(contours[i], convexs[i], false, true);//计算出图像的凸包，根据图像的轮廓点，通过函数convexhull转化成凸包的点点坐标
	}

	// 绘制
	dst = Mat::zeros(src.size(), CV_8UC3);
	vector<Vec4i> empty(0);
	for (size_t k = 0; k < contours.size(); k++) {
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
		drawContours(dst, contours, k, color, 2, LINE_8, hierachy, 0, Point(0, 0));
		drawContours(dst, convexs, k, color, 2, LINE_8, empty, 0, Point(0, 0));
	}
	imshow(output_win, dst);

	return;
}




//轮廓发现
//Mat src, dst;
//const char* output_win = "findcontours-demo";
//int threshold_value = 100;
//int threshold_max = 255;
//RNG rng;
//void Demo_Contours(int, void*);
//int main(int argc, char** argv) {
//	src = imread("C:/Users/xk/Desktop/1.jpg");
//	if (src.empty()) {
//		printf("could not load image...\n");
//		return -1;
//	}
//	namedWindow("input-image", CV_WINDOW_AUTOSIZE);
//	namedWindow(output_win, CV_WINDOW_AUTOSIZE);
//	imshow("input-image", src);
//	cvtColor(src, src, CV_BGR2GRAY);
//
//	const char* trackbar_title = "Threshold Value:";
//	createTrackbar(trackbar_title, output_win, &threshold_value, threshold_max, Demo_Contours);
//	Demo_Contours(0, 0);
//
//	waitKey(0);
//	return 0;
//}
//
//void Demo_Contours(int, void*) {
//	Mat canny_output;
//	vector<vector<Point>> contours;
//	vector<Vec4i> hierachy;
//	Canny(src, canny_output, threshold_value, threshold_value * 2, 3, false);
//	findContours(canny_output, contours, hierachy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
//
//	dst = Mat::zeros(src.size(), CV_8UC3);
//	RNG rng(12345);
//	for (size_t i = 0; i < contours.size(); i++) {
//		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
//		drawContours(dst, contours, i, color, 2, 8, hierachy, 0, Point(0, 0));
//	}
//	imshow(output_win, dst);
//}



//模板匹配Template match
//Mat src, temp, dst;
//int match_method = TM_SQDIFF;
//int max_track = 5;
//const char* INPUT_T = "input image";
//const char* OUTPUT_T = "result image";
//const char* match_t = "template match-demo";
//void Match_Demo(int, void*);
//int main()
//{
//	src = imread("C:/Users/xk/Desktop/p1.jpg");
//	temp = imread("C:/Users/xk/Desktop/p1eye.png");
//	if (src.empty() || temp.empty()) {
//		printf("could not load image...\n");
//		return -1;
//	}
//	namedWindow(INPUT_T, CV_WINDOW_AUTOSIZE);
//	namedWindow(OUTPUT_T, CV_WINDOW_NORMAL);
//	namedWindow(match_t, CV_WINDOW_AUTOSIZE);
//	imshow(INPUT_T, temp);
//	const char* trackbar_title = "Match Algo Type:";
//	createTrackbar(trackbar_title, OUTPUT_T, &match_method, max_track, Match_Demo);
//	Match_Demo(0, 0);
//
//	waitKey(0);
//	return 0;
//}
//void Match_Demo(int, void*) {
//	int width = src.cols - temp.cols + 1;
//	int height = src.rows - temp.rows + 1;
//	Mat result(width, height, CV_32FC1);
//
//	matchTemplate(src, temp, result, match_method, Mat());
//	normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
//
//	Point minLoc;
//	Point maxLoc;
//	double min, max;
//	src.copyTo(dst);
//	Point temLoc;
//	minMaxLoc(result, &min, &max, &minLoc, &maxLoc, Mat());
//	if (match_method == TM_SQDIFF || match_method == TM_SQDIFF_NORMED) {//TM_CCORR没找到正确的匹配位置
//		temLoc = minLoc;
//	}
//	else {
//		temLoc = maxLoc;
//	}
//
//	// 绘制矩形
//	rectangle(dst, Rect(temLoc.x, temLoc.y, temp.cols, temp.rows), Scalar(0, 0, 255), 2, 8);
//	rectangle(result, Rect(temLoc.x, temLoc.y, temp.cols, temp.rows), Scalar(0, 0, 255), 2, 8);
//
//	imshow(OUTPUT_T, result);
//	imshow(match_t, dst);
//}



//直方图反向投影
//Mat src; Mat hsv; Mat hue;
//int bins = 12;
//void Hist_And_Backprojection(int, void*);
//int main()
//{
//	src = imread("C:/Users/xk/Desktop/hand1.png");
//	if (src.empty()) {
//		printf("could not load image...\n");
//		return -1;
//	}
//	const char* window_image = "input image";
//	namedWindow(window_image, CV_WINDOW_NORMAL);
//	namedWindow("BackProj", CV_WINDOW_NORMAL);
//	namedWindow("Histogram", CV_WINDOW_NORMAL);
//	
//	cvtColor(src, hsv, CV_BGR2HSV);
//	hue.create(hsv.size(), hsv.depth());
//	int nchannels[] = { 0,0 };
//	mixChannels(&hsv, 1, &hue, 1, nchannels, 1);
//
//	createTrackbar("Histogram Bins:", window_image, &bins, 180, Hist_And_Backprojection);
//	Hist_And_Backprojection(0, 0);
//	
//	imshow(window_image, src);
//	waitKey(0);
//	return 0;
//}
//void Hist_And_Backprojection(int, void*)
//{
//	float range[] = { 0,180 };
//	const float* histRange = { range };
//	Mat h_hist;
//	calcHist(&hue, 1, 0, Mat(), h_hist, 1,&bins ,&histRange, true, false);
//	normalize(h_hist, h_hist, 0, 255, NORM_MINMAX, -1, Mat());
//
//	Mat backPrjImage;
//	calcBackProject(&hue, 1, 0, h_hist, backPrjImage, &histRange, 1, true);
//	imshow("BackProj", backPrjImage);
//	
//	int hist_h = 400;
//	int hist_w = 400;
//	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
//	int bin_w = hist_w / bins ;
//	for (int i = 1; i < bins; i++)
//	{
//		rectangle(histImage,
//			Point((i - 1) * bin_w, hist_h - cvRound(h_hist.at<float>(i - 1) * (400 / 255))),
//			//Point(i * bin_w, hist_h - cvRound(h_hist.at<float>(i) * (400 / 255))),
//			Point(i * bin_w, hist_h),
//			Scalar(0, 0, 255), -1);
//	}
//	imshow("Histogram", histImage);
//	return;
//}


//直方图比较
//int main()
//{
//	Mat base1,base2, test1, test2;
//	base1 = imread("C:/Users/xk/Desktop/1.jpg");
//	base2 = imread("C:/Users/xk/Desktop/2.jpg");
//	if (!base1.data || !base2.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	test1 = imread("C:/Users/xk/Desktop/4.jpg");
//	test2 = imread("C:/Users/xk/Desktop/3.jpg");
//	
//	cvtColor(base1, base1, CV_BGR2HSV);
//	cvtColor(base2, base2, CV_BGR2HSV);
//	cvtColor(test1, test1, CV_BGR2HSV);
//	cvtColor(test2, test2, CV_BGR2HSV);
//
//	int h_bins = 50,s_bins = 60;
//	int histSize[] = { h_bins,s_bins };
//	float h_ranges[] = { 0,180 };
//	float s_ranges[] = { 0,256 };
//	const float* ranges[] = { h_ranges,s_ranges };
//	int channels[] = { 0,1 };
//	MatND hist_base1;
//	MatND hist_base2;
//	MatND hist_test1;
//	MatND hist_test2;
//
//	
//	calcHist(&base1, 1, channels, Mat(), hist_base1, 2, histSize, ranges, true, false);
//	normalize(hist_base1, hist_base1, 0, 1, NORM_MINMAX, -1, Mat());
//
//	calcHist(&base2, 1, channels, Mat(), hist_base2, 2, histSize, ranges, true, false);
//	normalize(hist_base2, hist_base2, 0, 1, NORM_MINMAX, -1, Mat());
//
//	calcHist(&test1, 1, channels, Mat(), hist_test1, 2, histSize, ranges, true, false);
//	normalize(hist_test1, hist_test1, 0, 1, NORM_MINMAX, -1, Mat());
//
//	calcHist(&test2, 1, channels, Mat(), hist_test2, 2, histSize, ranges, true, false);
//	normalize(hist_test2, hist_test2, 0, 1, NORM_MINMAX, -1, Mat());
//
//	double base1test1 = compareHist(hist_base1, hist_test1, CV_COMP_CORREL);
//	double base1base1 = compareHist(hist_base1, hist_base1, CV_COMP_CORREL);
//	double base2test2 = compareHist(hist_base2, hist_test2, CV_COMP_CORREL);
//
//	putText(base1, to_string(base1test1), Point(50, 50), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);
//	putText(base1, to_string(base1base1), Point(50, 100), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);
//	putText(base2, to_string(base2test2), Point(50, 50), CV_FONT_HERSHEY_COMPLEX, 1, Scalar(0, 0, 255), 2, LINE_AA);
//	imshow("base1", base1);
//	imshow("base2", base2);
//
//	waitKey(0);
//	return 0;
//}


//直方图计算
//int main()
//{
//	Mat src, src_gray, dst, median;
//	src = imread("C:/Users/xk/Desktop/1.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//
//	//分通道显示
//	vector<Mat> bgr_planes;
//	split(src, bgr_planes);
//	//imshow("single channel demo", bgr_planes[0]);
//
//	//计算直方图
//	int histSize = 256;
//	float range[] = { 0,256 };
//	const float* histRange = { range };
//	Mat b_hist, g_hist, r_hist;
//	calcHist(&bgr_planes[0], 1, 0, Mat(), b_hist, 1, &histSize, &histRange, true, false);
//	calcHist(&bgr_planes[1], 1, 0, Mat(), g_hist, 1, &histSize, &histRange, true, false);
//	calcHist(&bgr_planes[2], 1, 0, Mat(), r_hist, 1, &histSize, &histRange, true, false);
//
//	//归一化
//	int hist_h = 400;
//	int hist_w = 512;
//	int bin_w = hist_w / histSize;
//	Mat histImage(hist_w, hist_h, CV_8UC3, Scalar(0, 0, 0));
//	normalize(b_hist, b_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
//	normalize(g_hist, g_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
//	normalize(r_hist, r_hist, 0, hist_h, NORM_MINMAX, -1, Mat());
//
//	//render histogram chart
//	for (int i = 1; i < histSize; i++)
//	{
//		line(histImage, Point((i - 1) * bin_w, hist_h - cvRound(b_hist.at<float>(i - 1))),
//			Point((i)*bin_w, hist_h - cvRound(b_hist.at<float>(i))), Scalar(255, 0, 0), 2, LINE_AA);
//
//		line(histImage, Point((i - 1) * bin_w, hist_h - cvRound(g_hist.at<float>(i - 1))),
//			Point((i)*bin_w, hist_h - cvRound(g_hist.at<float>(i))), Scalar(0, 255, 0), 2, LINE_AA);
//
//		line(histImage, Point((i - 1) * bin_w, hist_h - cvRound(r_hist.at<float>(i - 1))),
//			Point((i)*bin_w, hist_h - cvRound(r_hist.at<float>(i))), Scalar(0, 0, 255), 2, LINE_AA);
//
//	}
//	imshow("output", histImage);
//
//	waitKey(0);
//	return 0;
//}



//直方图均衡化(单通道图像)   增强对比度
//int main()
//{
//	Mat src, src_gray,dst,median;
//	src = imread("C:/Users/xk/Desktop/p1.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	
//
//	cvtColor(src, src_gray, CV_BGR2GRAY);
//	imshow("src_gray", src_gray);
//	equalizeHist(src_gray, dst);
//
//	imshow("dst", dst);
//	waitKey(0);
//	return 0;
//}




//像素重映射
//Mat src, dst, map_x, map_y;
//int index = 0;
//void update_map(void)
//{
//	for (int row = 0; row < src.rows; row++)
//	{
//		for (int col = 0; col < src.cols; col++)
//		{
//			switch (index)
//			{
//			case 0:
//				if (col > src.cols*0.25 && col<src.cols*0.75 && row>src.rows*0.25 && row<src.rows*0.75)
//				{
//					map_x.at<float>(row, col) = 2 * (col - src.cols * 0.25);
//					map_y.at<float>(row, col) = 2 * (row - src.rows * 0.25);
//				}
//				else
//				{
//					map_x.at<float>(row, col) = 0;
//					map_y.at<float>(row, col) = 0;
//				}
//				break;
//			case 1:
//				map_x.at<float>(row, col) = src.cols - col - 1;
//				map_y.at<float>(row, col) = row;
//				break;
//			case 2:
//				map_x.at<float>(row, col) = col;
//				map_y.at<float>(row, col) = src.rows - row - 1;
//				break;
//			case 3:
//				map_x.at<float>(row, col) = src.cols - col - 1;
//				map_y.at<float>(row, col) = src.rows - row - 1;
//				break;
//			}
//		}
//	}
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
//	map_x.create(src.size(), CV_32FC1);
//	map_y.create(src.size(), CV_32FC1);
//	int c = 0;
//	while (true)
//	{
//		c = waitKey(500);
//		
//		if ((char)c==27)
//		{
//			break;
//		}
//		index = c % 4;
//		update_map();
//		remap(src, dst, map_x, map_y, INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 255, 255));
//		imshow("remap", dst);
//	}
//	
//	return 0;
//}







//霍夫圆检测
//int main()
//{
//	Mat src, src_gray,dst,median;
//	src = imread("C:/Users/xk/Desktop/circles.jpg");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//
//	medianBlur(src, median, 7);
//	cvtColor(median, median, CV_BGR2GRAY);
//
//	vector<Vec3f> pcircles;
//	HoughCircles(median, pcircles, CV_HOUGH_GRADIENT, 1, 10, 100, 30, 5, 50);
//	src.copyTo(dst);
//	for (size_t i = 0; i < pcircles.size(); i++)
//	{
//		Vec3f cc = pcircles[i];
//		circle(dst, Point(cc[0], cc[1]), cc[2], Scalar(0, 0, 255), 2, LINE_AA);
//		circle(dst, Point(cc[0], cc[1]), 2, Scalar(198, 23, 165), 2, LINE_AA);
//	}
//
//
//	imshow("dst", dst);
//	waitKey(0);
//	return 0;
//}





//霍夫直线变换
//int main()
//{
//	Mat src, src_gray,dst;
//	src = imread("C:/Users/xk/Desktop/3.png");
//	if (!src.data)
//	{
//		cout << "no";
//		return -1;
//	}
//	imshow("src", src);
//
//	Canny(src, src_gray, 150, 200);
//	cvtColor(src_gray, dst, CV_GRAY2BGR);
//	imshow("edge image", dst);
//
//	vector<Vec4f> plines;
//	HoughLinesP(src_gray, plines, 1, CV_PI / 180.0, 10, 10, 5);
//	Scalar color = Scalar(0, 0, 255);
//	for (size_t i = 0; i < plines.size(); i++)
//	{
//		Vec4f hline = plines[i];
//		line(dst, Point(hline[0], hline[1]), Point(hline[2], hline[3]), color, 3, LINE_AA);
//	}
//	imshow("dst", dst);
//	waitKey(0);
//	return 0;
//}



//canny边缘检测
//Mat src, dst, dst2, dst3, gray_src;
//int t1_value = 50;
//int max_value = 255;
//void Canny_Demo(int, void*)
//{
//	Mat edge_image;//	blur(gray_src, gray_src, Size(3, 3), Point(-1, -1));//	Canny(gray_src, edge_image, t1_value, t1_value * 2, 3, false);//	//	dst.create(src.size(), src.type());//	src.copyTo(dst, edge_image);//	imshow("output", dst);
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
//	cvtColor(src, gray_src, CV_BGR2GRAY);
//	createTrackbar("Threshold Value", "output", &t1_value, max_value, Canny_Demo);
//	Canny_Demo(0,0);
//
//
//	waitKey(0);
//	return 0;
//}


