#include<opencv2/opencv.hpp>

#include<iostream>

using namespace cv;

using namespace std;
//
//int main(int, char**)
//{
//    
//    VideoCapture cap;    //�����洢��Ƶ�ļ������豸�Ķ���
//    cap.open("D:/Ӧ�����/������/����ʶ��.mp4");    //����Ƶ�ļ�������Ƶ�豸
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
//        if (!ok)    //�ж���Ƶ�ļ��Ƿ��ȡ����
//            break;
//        imshow(windowsName, frame);    //����Ƶ�����л�ȡͼƬ��ʾ������
//        k = waitKey(33);    //ÿ33����һ��ͼƬ
//        if (k == 27) break;    //�����˳�����Esc
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