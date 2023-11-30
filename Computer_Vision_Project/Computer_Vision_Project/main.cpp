#include "opencv2/opencv.hpp" //��� ���� 
#include <iostream> //��� ����
using namespace std; //std:: ����
using namespace cv; //cv:: ����
using namespace cv::dnn; //dnn:: ����

Mat src, res_src, dst; //���� ���� ������ ����� ��ü, ���� ������ (600x600)���� ũ�� ������ ��ü, ũ�� ������ �Ϸ�� ��ü�� color�������� ������ ��ü.   src �� res_src �� dst
Mat dst1 ,res_dst ; //���� ��ȯ ��� ������ ����� ��ü, �ش� ��ü�� (500x500)���� ũ�� ������ ��ü
Mat  gray_res_dst, bin; //res_dst�� ��� �������� ��ȯ�� ����� ����� ��ü, gray_res_dst�� ����ȭ ����� ������ ��ü.
Point2f srcPts[4], dstPts[4]; //�Է� ����� ��� ���󿡼��� �� �� ��ǥ�� ������ Point2f�� �迭
Mat dst2;
void Save_trainData(int event, int x, int y, int flags, void* userdata); //���콺 �̺�Ʈ �Լ� �����
int main()
{
	src = imread("a.jpg",IMREAD_GRAYSCALE); //�̸� �����ص� jpg ����(���� ����)�� ��� �������� �ҷ���
	resize(src, res_src, Size(1000, 1000)); //���� ������ (600x600)���� ũ�� ����
	cvtColor(res_src, dst, COLOR_GRAY2BGR); //ũ�� ������ �Ϸ�� ��ü�� color�������� ��ȯ (���콺 �̺�Ʈ�� ���� ������ circle(��)�� �׸��� ����. 
	cvtColor(res_src, dst2, COLOR_GRAY2BGR); //ũ�� ������ �Ϸ�� ��ü�� color�������� ��ȯ (���콺 �̺�Ʈ�� ���� ������ circle(��)�� �׸��� ����. 

	namedWindow("src"); //src �̸��� ������ â ����
	setMouseCallback("src", Save_trainData); //src ������ â�� ���콺 �̺�Ʈ call

	imshow("src", dst); //dst �̹����� ȭ�鿡 ��� 

	waitKey(0); //���� �Է±��� ���
	return 0; //�Լ� ����� ��ȯ 
}
void Save_trainData(int event, int x, int y, int flags, void*) //���콺 �̺�Ʈ �Լ� ���Ǻ�
{
	// ******************* 2022.11.27 ���� �ۼ� *******************

	vector<string> classNames{ "A","B","C","E","H","I","L","M","N","O","R","T","V" };
	

	// ***********************************************************

	static int cnt = 0; //int�� ���� ���� ���� �� ���� ������ �������� ���� ���, ���콺 �̺�Ʈ�� call�� ������ cnt���� 0���� �ʱ�ȭ��. �̿� ���� �ϱ��� if���� ������� ����. 

	clock_t old_time = clock(); //�Լ� ���� �ð��� �߻��ϴ� clock ��

	//[ ������ ���ϴ� �簢�� ������ ���� ��ȯ�� �̿��Ͽ� �������ִ� �ڵ� ]
	if (event == EVENT_LBUTTONDOWN) //���콺 ��Ŭ���� �� ��,
	{
		if (cnt < 4) //cnt�� ���� 4�̸��� ���, 
		{
			//���� ��ȯ�� ���� �簢�� ������ ���� ��� �𼭸� ������ �����Ͽ� �ð� ���� ������ ������.
			srcPts[cnt++] = Point2f(x, y); //srcPts[0] => ���� ��� �𼭸� ��ǥ   srcPts[1] => ���� ��� �𼭸� ��ǥ  srcPts[2] => ���� �ϴ� �𼭸� ��ǥ  srcPts[3] => ���� �ϴ� �𼭸� ��ǥ 
			circle(dst, Point(x, y), 5, Scalar(0, 0, 255), -1); //���콺 ��Ŭ���� �� �κ��� ��ǥ�� ������ circle�� �׸�. 
			imshow("src", dst); //������ circle�� �׸� ����� ȭ�鿡 ��� 

			if (cnt == 4) //cnt�� ���� 4�� ��, 
			{
				int w = res_src.cols, h = res_src.rows; //int�� ���� w�� res_src ������ �� ������, h�� res_src ������ �� ������ ������.  => ���� ��ȯ�Ͽ� ���� ��� ������ ���ο� ���� ũ�� 

				//������ â���� ����ڰ� ���콺�� ������ �簢�� �������� �̵��� ��� ���� ��ǥ�� ����
				dstPts[0] = Point2f(0, 0); dstPts[1] = Point2f(w - 1, 0); 
				dstPts[2] = Point2f(w - 1, h - 1); dstPts[3] = Point2f(0, h - 1);

				Mat pers = getPerspectiveTransform(srcPts, dstPts); //3x3 ���� ��ȯ ����� pers ������ ���� 
				
				warpPerspective(dst2, dst1, pers, Size(w, h)); //���� ��ȯ�� �����Ͽ� w x h ũ���� ��� ���� dst1�� �����Ѵ�. 
				resize(dst1, res_dst, Size(500, 500)); //���� ��ȯ�� �Ϸ�� ��ü�� (500x500)���� ũ�� ���� 
				imshow("res_dst", res_dst); //���� ��ȯ �� ũ�� ������ �޷�� ������ "res_dst"������ â�� ���  ************************
\
				cvtColor(res_dst, gray_res_dst, COLOR_BGR2GRAY); //res_dst ��ü�� ��� �������� ��ȯ (����ȭ ������ ����)
				threshold(gray_res_dst, bin, 0, 255, THRESH_BINARY_INV | THRESH_OTSU); //gray_res_dst�� ����ȭ ������ �Ϸ�� ������ bin ��ü�� ����. ����� ���, �۾��� �������� ��� THRESH_BINARY �� THRESH_BINARY_INV�� �̿�
				
				Mat labels, stats, centroids; //bin ���� ���� ���̺��� �����ϰ�, �� ��ü ������ ��� ������ ����� ��ü 

				Mat color_dilate_dst, dilate_dst;
				dilate(bin, dilate_dst, Mat(), Point(-1, -1), 8); //bin ���� ���ؼ� ��â ������ ����(10ȸ)
				//cvtColor(bin, my_dst, COLOR_GRAY2BGR);
				cvtColor(dilate_dst, color_dilate_dst, COLOR_GRAY2BGR); //dilate_dst ������ �÷� �������� ����
				int cnt_object = connectedComponentsWithStats(dilate_dst, labels, stats, centroids); //��ü�� ������ ������ int�� ���� 

				for (int i = 1; i < cnt_object; i++) //����� �ܾ� ���� ��ü�� ����ŭ �ݺ�  
				{
					
					int* p = stats.ptr<int>(i); //��� ������ �����ϰ� ��� ��ü ������ ���ؼ��� for �ݺ����� ���� 
					if (p[4] < 20) continue; //��ü�� �ȼ� ������ 20���� ������ �������� �����ϰ� ����
					

					rectangle(color_dilate_dst, Rect(p[0], p[1], p[2], p[3]), Scalar(0, 0, 255), 2); //����� ��ü ������ ���δ� �ٿ�� �ڽ��� ���������� �׸� 

					string desc = format("%d", i); //��ü�� ��ȣ�� ������ string�� ����
					putText(color_dilate_dst, desc, Point(centroids.at<double>(i, 0), centroids.at<double>(i,1))
						, FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0),1); //��ü�� ���� �߽� ��ǥ�� ��ü ��ȣ�� �Ķ������� ���

					Mat words = bin(Rect(p[0], p[1], p[2], p[3])); //����ȭ�� �Ϸ�� i��° ��ü�� �ٿ�� �ڽ�(�ܾ� �������� ����) ��ǥ�� ��ü�� ����
					Mat res_words; 
					resize(words, res_words, Size(500, 500)); //����� �ܾ� ������ (400x400)���� ũ�� ���� 

					//[��� Ȯ���� ���� �ڵ�]
					

					Mat labels_for_words, stats_for_words, centroids_for_words; //�ܾ� ������ ���� ���� ���̺��� �����ϰ�, �� ��ü ������ ��� ������ ����� ��ü 
					
					Mat color_res_words;
					cvtColor(res_words, color_res_words, COLOR_GRAY2BGR); //res_words ��ü�� �÷� �������� ��ȯ 
					int cnt_object2 = connectedComponentsWithStats(res_words, labels_for_words, stats_for_words, centroids_for_words); //��ü�� ������ ������ int�� ����  

					for (int j = 1; j < cnt_object2; j++) //����� ���� ���� ��ü�� ����ŭ �ݺ�  
					{
						int* k = stats_for_words.ptr<int>(j); //��� ������ �����ϰ� ��� ��ü ������ ���ؼ��� for �ݺ��� ����
						if (k[4] < 20) continue; //��ü�� �ȼ� ������ 20���� ������ �������� �����ϰ� ���� 

						rectangle(color_res_words, Rect(k[0], k[1], k[2], k[3]), Scalar(0, 0, 255), 2); //����� ��ü ������ ���δ� �ٿ�� �ڽ��� ���������� �׸�

						string desc_words = format("%d", j); //��ü�� ��ȣ�� ������ string�� ���� 
						putText(color_res_words, desc_words, Point(centroids_for_words.at<double>(j, 0), centroids_for_words.at<double>(j, 1))
							, FONT_HERSHEY_COMPLEX, 0.5, Scalar(255, 0, 0), 1); //��ü�� ���� �߽� ��ǥ�� ��ü ��ȣ�� �Ķ������� ���

						Mat chars = res_words(Rect(k[0], k[1], k[2], k[3])); //j��° ��ü�� �ٿ�� �ڽ�(���� ����) ��ǥ�� ��ü�� ���� 
						Mat res_chars; 
						resize(chars, res_chars, Size(100, 100)); //����� ���� ������ (200x200)���� ũ�� ����
						Mat M = Mat_<double>({ 2,3 }, { 1,0,100,0,1,100 });
						Mat rrr; 
						warpAffine(res_chars, rrr, M, Size(300, 300));

						// ******************* 2022.11.27 ���� �ۼ� *******************

						//cout << i << "��° �ܾ��� ���� �ν� ��� : ";
						Mat res_chars_color;
						cvtColor(rrr, res_chars_color, COLOR_GRAY2BGR);
						
						string filename = "word_boxN2" + to_string(i) + "_" + to_string(j) + ".jpg";
						imwrite(filename, res_chars_color);
						
					}


				}

				imshow("bin", bin); //����ȭ�� �Ϸ�� ������ "bin"â�� ���
				imshow("dilate", color_dilate_dst); //��â ������ �Ϸ�� ������ "dilate"���� ��� 

			}
		}
	}
}
