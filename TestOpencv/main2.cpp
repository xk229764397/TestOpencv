#include<opencv2/opencv.hpp>

#include<iostream>

using namespace cv;

using namespace std;
//
//int main(int, char**)
//{
//    
//    VideoCapture cap;    //创建存储视频文件或者设备的对象
//    cap.open("D:/应用软件/爱剪辑/人脸识别.mp4");    //打开视频文件或者视频设备
//    if (!cap.isOpened())
//    {
//        cout << "could not open the VideoCapture !" << endl;
//        system("pause");
//        return -1;
//    }
//
//    const char* windowsName = "Example";
//    int k = -1;
//
//    while (true)
//    {
//        Mat frame;
//        bool ok = cap.read(frame);
//        if (!ok)    //判断视频文件是否读取结束
//            break;
//        imshow(windowsName, frame);    //从视频对象中获取图片显示到窗口
//        k = waitKey(33);    //每33毫秒一张图片
//        if (k == 27) break;    //按下退出键：Esc
//    }
//
//    waitKey(-1);
//    return 0;
//}


//int t1_value = 50;
//int max_value = 255;
//Mat gray_src;
//Mat src, dst;
//void Canny_Demo(int, void*)
//{
//	Mat edge_image;
//	blur(gray_src, gray_src, Size(3, 3), Point(-1, -1));
//	Canny(gray_src, edge_image, t1_value, t1_value * 2);
//
//	dst.create(src.size(), src.type());
//	src.copyTo(dst, edge_image);
//	imshow("Canny Result", dst);
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
//
//	cvtColor(src, gray_src, CV_BGR2GRAY);
//	createTrackbar("Threshold Value:", "Canny Result", &t1_value, max_value, Canny_Demo);
//	Canny_Demo(0, 0);
//
//
//	waitKey(0);
//	return 0;
//}