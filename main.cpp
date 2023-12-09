#include <iostream>
#include <opencv2/opencv.hpp>
#include <librealsense2/rs.hpp>
#include <sstream>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>
#include <random>
#include <limits>
#include <opencv2/core.hpp>

#include "omp.h"




using namespace std;
using namespace cv;
using namespace rs2;

Mat Road_element_erode, Road_element_dilate;		//ͼ������ĺ� 
Mat Road_HSV_image, Road_dst_image;
Mat Road_edge_0, Road_edge_dst;

//������
vector<vector<cv::Point>> contours;		//�����㼯�ļ���
vector<cv::Vec4i> hierarchy;			//�㼶


// ������ ����ֱ�߽������ֵ
int AlphaValuesSlider;					//��������Ӧ�ı���
const int MaxAlphaValue = 500;			//Alpha�����ֵ

int H, S, V;
int buttondown = 0;
int px = 0, py = 0;					// �������
float dist_to_center = 0.0;			// ����
float fps;							// ֡��
float REJECT_DEGREE_TH = 4.0;		// ��㣿
char fps_char[50];					// ��Ļ��ʾ
clock_t time1, time2;				// ����fps����

// https://dev.intelrealsense.com/docs/rs-align-advanced
// �ڸ���վ�п��Կ��������ľ������

// https://dev.intelrealsense.com/docs
// ����վΪ�����ĵ�

//��Ӧ�������Ļص�����
void on_Trackbar(int, void*)
{

}

//������С����
static inline bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2)
{
	return (cv::contourArea(contour1) > cv::contourArea(contour2));
}

//�������ͷ���ݹܵ������Ƿ�ı�
bool profile_changed(const vector<stream_profile>& current, const vector<stream_profile>& prev);

// ��ȡ��Ⱥ���ɫͼ�񣨼ٶ��˴˶��룩����ȱ�����λ���û�ϣ����ʾ�������룬
// ��������ɫ���Ա�ɾ���䱳������Ⱦ��������������ֵ���κ����أ���
void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist);

// ���������� ���ص�����
void on_Mouse_HSV(int event, int x, int y, int flags, void*);
void on_Mouse_Distance(int event, int x, int y, int flags, void*);

// ��� ��ȡֱ��
std::vector<std::vector<double>> GetLines(cv::Mat Image);
// ��� ��ȡֱ�ߵĺ���
std::vector<std::vector<double>> FilterLines(std::vector<cv::Vec4i> Lines);
// ��� ��ȡ���
int* GetVanishingPoint(std::vector<std::vector<double>> Lines);

// ��ֱ�߽��о���
std::vector<std::vector<cv::Vec4i>> clusterLines(std::vector<cv::Vec4i>& lines, double distanceThresh);
// ���С��ͨ��
void Clear_MicroConnected_Areas(cv::Mat src, cv::Mat& dst, double min_area);


void fitLineRansac(const std::vector<cv::Point2f>& points, cv::Vec4f& line, int iterations , double sigma, double k_min , double k_max );

void draw_line_k(Mat image, float k, cv::Point point, const cv::Scalar& color, int thickness);

class D435if_Camera
{
public:
	rs2::config cfg;
	rs2::pipeline pipe;				//����realsense�ܵ�
	rs2::frameset frame;
	rs2::pipeline_profile profile;
	rs2_intrinsics depth_intrinsics; //�������ͷ�ڲ�
	rs2_intrinsics color_intrinsics; //��ɫ����ͷ�ڲ�

	D435if_Camera();
	void start();
	void stop();
	float get_depth_scale(rs2::device dev); // ��ȡ������ȵ�λ��ʾ������
	rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);
	
	Mat readColorCamera();
	Mat readDepthCamera();
	Mat readAlignedCamera(rs2_stream align_to, rs2::align align, frameset frameset_data,float depth_scale, float depth_clipping_distance);
	Mat frame_to_mat(const rs2::frame& f);	
};

D435if_Camera::D435if_Camera()
{
	// ���캯��
}

void D435if_Camera::start()
{
	//��ȡ�豸�ͺź����к�
	rs2::context ctx;
	auto devicelist = ctx.query_devices();
	rs2::device dev = *devicelist.begin(); // �����豸
	cout << "RS2_CAMERA_NAME�� " << dev.get_info(RS2_CAMERA_INFO_NAME) << endl;
	cout << "RS2_CAMERA_SERIAL_NUMBER�� " << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << endl;
	cout << "RS2_FIRMWARE_VERSION�� " << dev.get_info(RS2_CAMERA_INFO_RECOMMENDED_FIRMWARE_VERSION) << endl;
	
	// ����������
	cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
	cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);
	
	//�����豸
	profile = pipe.start(cfg);
	
	//����������
	auto _stream_depth = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
	auto _stream_color = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();
	
	//�������ͷ�ڲ�
	depth_intrinsics = _stream_depth.get_intrinsics();
	color_intrinsics = _stream_color.get_intrinsics();
	
	//��ӡ�ڲ�
	char str[124] = { 0 };
	snprintf(str, 124, "%d_%d_%f_%f_%f_%f_%d_%f_%f_%f_%f_%f",
		depth_intrinsics.width, depth_intrinsics.height, depth_intrinsics.ppx, depth_intrinsics.ppy,
		depth_intrinsics.fx, depth_intrinsics.fy, depth_intrinsics.model, depth_intrinsics.coeffs[0],
		depth_intrinsics.coeffs[1], depth_intrinsics.coeffs[2], depth_intrinsics.coeffs[3], depth_intrinsics.coeffs[4]);
	printf("depth intrinsics: %s:%d\n", str, strlen(str));
	memset(str, 0x00, 124);
	
	snprintf(str, 124, "%d_%d_%f_%f_%f_%f_%d_%f_%f_%f_%f_%f",
		color_intrinsics.width, color_intrinsics.height, color_intrinsics.ppx, color_intrinsics.ppy,
		color_intrinsics.fx, color_intrinsics.fy, color_intrinsics.model, color_intrinsics.coeffs[0],
		color_intrinsics.coeffs[1], color_intrinsics.coeffs[2], color_intrinsics.coeffs[3], color_intrinsics.coeffs[4]);
	printf("color intrinsics: %s:%d\n", str, strlen(str));



	////����ǰ��֡
	//for (int i = 0; i < 5; i++)
	//{
	//	pipe.wait_for_frames();
	//}
}

Mat D435if_Camera::frame_to_mat(const rs2::frame& f)
{
	auto vf = f.as<rs2::video_frame>();
	const int w = vf.get_width();
	const int h = vf.get_height();

	if (f.get_profile().format() == RS2_FORMAT_BGR8)
	{
		//cout << "RS2_FORMAT_BGR8" << endl;
		return Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
	}
	else if (f.get_profile().format() == RS2_FORMAT_RGB8)
	{
		//cout << "RS2_FORMAT_RGB8" << endl;
		auto r = Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
		cvtColor(r, r, CV_RGB2BGR);
		return r;
	}
	else if (f.get_profile().format() == RS2_FORMAT_Z16)
	{
		//cout << "RS2_FORMAT_Z16" << endl;
		return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
	}
	else if (f.get_profile().format() == RS2_FORMAT_Y8)
	{
		//cout << "RS2_FORMAT_Y8" << endl;
		return Mat(Size(w, h), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP);
	}

	throw std::runtime_error("Frame format is not supported yet!");
}

Mat D435if_Camera::readColorCamera()
{
	frame = pipe.wait_for_frames();
	auto color_frame = frame.get_color_frame();
	//cout << "- start ---------------------------------" << endl;
	Mat color = frame_to_mat(color_frame);
	//cout << "- end ---------------------------------" << endl;
	return color;
}

Mat D435if_Camera::readDepthCamera()
{
	frame = pipe.wait_for_frames();
	auto depth_frame = frame.get_depth_frame();
	Mat depth = frame_to_mat(depth_frame);
	return depth;
}

Mat D435if_Camera::readAlignedCamera(rs2_stream align_to, rs2::align align, frameset frameset_data, float depth_scale, float depth_clipping_distance)
{
	//��ȡ�Ѵ���Ķ���֡
	auto frameset_processed = align.process(frameset_data);

	// ��ȡ��ɫ�Ͷ�������֡
	// ��ɫ֡��rs2::video_frame���ͣ����֡if��rs2:depth_frame
	//��������rs2:��video_frame�������������������ع��ܣ�
	rs2::video_frame other_frame = frameset_processed.first(align_to);  //video_frame
	rs2::depth_frame aligned_depth_frame = frameset_processed.get_depth_frame(); //depth_frame

	//If one of them is unavailable, continue iteration
	//�������һ�������ã��ͼ�����ȡ��һ֡
	if (!aligned_depth_frame || !other_frame)
	{
		cout<<"����֡�����ã���ȡ��һ֡"<<endl;
	}
	else
	{
		remove_background(other_frame, aligned_depth_frame, depth_scale, depth_clipping_distance);
	}

	//Mat aligned_image_color(Size(640, 480), CV_8UC3, (void*)other_frame.get_data(), Mat::AUTO_STEP);
	Mat aligned_image_color = frame_to_mat(other_frame);

	return aligned_image_color;
}

void D435if_Camera::stop()
{
	pipe.stop();
}

float D435if_Camera::get_depth_scale(rs2::device dev)
{
	// Go over the device's sensors
	// �鿴�û�������豸
	for (rs2::sensor& sensor : dev.query_sensors())
	{
		// Check if the sensor if a depth sensor
		if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
		{
			// When called on a depth sensor, this method will return the number of meters represented by a single depth unit
			// ������ȴ������ϵ���ʱ���˷����������ɵ�����ȵ�λ��ʾ������
			return dpt.get_depth_scale();
		}
	}
	throw std::runtime_error("Device does not have a depth sensor");
}

// ��һ��stream which ���ͼ�����������
rs2_stream D435if_Camera::find_stream_to_align(const std::vector<rs2::stream_profile>& streams)
{
	// Given a vector of streams, we try to find a depth stream and another stream to align depth with.
	// We prioritize color streams to make the view look better.
	// If color is not available, we take another stream that (other than depth)
	rs2_stream align_to = RS2_STREAM_ANY;
	bool depth_stream_found = false;
	bool color_stream_found = false;
	for (rs2::stream_profile sp : streams)
	{
		rs2_stream profile_stream = sp.stream_type();
		if (profile_stream != RS2_STREAM_DEPTH)
		{
			if (!color_stream_found)         //Prefer color
				align_to = profile_stream;

			if (profile_stream == RS2_STREAM_COLOR)
			{
				color_stream_found = true;
			}
		}
		else
		{
			depth_stream_found = true;
		}
	}

	if (!depth_stream_found)
		throw std::runtime_error("No Depth stream available");

	if (align_to == RS2_STREAM_ANY)
		throw std::runtime_error("No stream found to align with Depth");

	return align_to;
}



int main() try
{
	D435if_Camera My_D435if_Camera;
	My_D435if_Camera.start();

	rs2::config cfg = My_D435if_Camera.cfg;
	rs2::pipeline pipe = My_D435if_Camera.pipe;
	rs2::frameset frame = My_D435if_Camera.frame;
	rs2::pipeline_profile profile = My_D435if_Camera.profile;

	// align_to �Ǽƻ������֡��֮����������͡�
	rs2_stream align_to = My_D435if_Camera.find_stream_to_align(profile.get_streams());
	rs2::align align(align_to);

	// ��ʵ��ʹ��֮֡ǰ�����Ի�ȡ��������������ŵ�λ��
	// ���磬���������һ��ֵΪ 2 ��������أ�������ȱ�����λΪ 0.5��������ؾ������һ�ס�
	float depth_scale = My_D435if_Camera.get_depth_scale(profile.get_device());

	// Ҫȥ���ı����ľ��룬0.9f �� 90cm
	float depth_clipping_distance = 1.0f;

	while (true)
	{
		time1 = clock();// ��ʱ
		if (time1 != 0)
		{
			fps = 1000 / (time1 - time2);
			time2 = time1;
		}

		/* ===================================== realsense ��ȡ���� ===================================== */

			// �����������װ�и����������������Ҫ����������
			// ����������������� ������kinect����� ����и���ʱʹ�á�
			// 
			// https://dev.intelrealsense.com/docs/rs-measure


		// pipeline::wait_for_frames �������豸�����Ͽ�����ʱ�滻��ʹ�õ��豸
		frameset frame_set_data = My_D435if_Camera.pipe.wait_for_frames(); //��������ֱ���µ�һ֡����

		if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
		{
			// If the profile was changed, update the align object, and also get the new device's depth scale
			// ��������ļ��Ѹ��ģ����¶�����󣬲���ȡ���豸����ȿ̶�
			profile = pipe.get_active_profile();
			align_to = My_D435if_Camera.find_stream_to_align(profile.get_streams());
			align = rs2::align(align_to);
			depth_scale = My_D435if_Camera.get_depth_scale(profile.get_device());
		}

		auto frameset_processed = align.process(frame_set_data);
		rs2::depth_frame aligned_depth_frame = frameset_processed.get_depth_frame(); // ��ȡ���

/* ================================================ ͼ���� ================================================ */

		/*--------------------------------------------------------------------------------------------*/
		// 
		// ����·��һֱ����ɫ����ɫ�������ĸ�Ч���ã��ı߽磨���Բõ����ĵ�ͼ��
		// ��ԭʼ�Ĳ�ͼ����Ҫ��������Ķ��뵽���ӵ����ģ���	�ȽϺ���
		// ��ERRORע�⣺��ͼ������ɶ��������ͼ����֪��Ϊʲô��
		// 
		// �����ͼ����Ҫ��������Ķ��뵽���ӵ����ģ���		��̫���ʣ����������ͼ�кܶ�հ�
		// 
		// ��̨�ף������״�����ݷ����Ӿ����ĵ��������
		// �����ͼ��
		// 
		// ��˫ľ�ţ��������ͼȡ��Ե�������� ��ˮ���� ȡ��Ե����
		// 
		/*--------------------------------------------------------------------------------------------*/

		Mat aligned_image_color = My_D435if_Camera.readAlignedCamera(align_to, align, frame_set_data, depth_scale, depth_clipping_distance);
		Mat src_color_image = My_D435if_Camera.readColorCamera();

		// imshow("src_color_image", src_color_image);


		/* #################################################### ��·��Ե ��ʼ #################################################### */

		// ��ͼHSV������ֵ��ڰף��������ֱ��
		// ���������ݣ� �Ƕ� float �жϽǶ��Ƿ�Ϊ90��-90���Ҽ��ɡ�

		cvtColor(src_color_image, Road_HSV_image, cv::COLOR_BGR2HSV);
		//inRange(Road_HSV_image, cv::Scalar(11, 43, 46), cv::Scalar(34, 255, 255), Road_dst_image); // ��+��ɫ
		inRange(Road_HSV_image, cv::Scalar(0, 0, 0), cv::Scalar(150, 255, 46), Road_dst_image); // ��ɫ

		//-------------------------- �����ȡHSV
		//imshow("HSV color", HSV_image);
		//setMouseCallback("HSV color", on_Mouse_HSV, &HSV_image);
		//if (buttondown == 1)-
		//{
		//	cout << "H: " << (int)HSV_image.at<cv::Vec3b>(py, px)[0] << '\t';
		//	cout << "S: " << (int)HSV_image.at<cv::Vec3b>(py, px)[1] << '\t';
		//	cout << "V: " << (int)HSV_image.at<cv::Vec3b>(py, px)[2] << endl;
		//	buttondown = 0;
		//}
		//-------------------------- �����ȡHSV

		Road_element_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		Road_element_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

		erode(Road_dst_image, Road_dst_image, Road_element_erode); //��ʴ
		erode(Road_dst_image, Road_dst_image, Road_element_erode); //��ʴ
		//imshow("��·-��ֵ����", Road_dst_image);

		GaussianBlur(Road_dst_image, Road_dst_image, cv::Size(5, 5), 0, 0);
		//imshow("��·-Road_dst_image", Road_dst_image);
		//Mat Road_dst_image2;
		//Clear_MicroConnected_Areas(Road_dst_image, Road_dst_image2, 4000);
		//imshow("��·-Road_dst_image-2", Road_dst_image2);
		Canny(Road_dst_image, Road_edge_0, 100, 300, 3); // 3��4λ ��ֵԽСϸ��Խ��
		//imshow("Canny", edge_0);
		dilate(Road_edge_0, Road_edge_dst, Road_element_dilate);//����
		//imshow("edge dilate", edge_0);
		GaussianBlur(Road_edge_dst, Road_edge_dst, cv::Size(5, 5), 0, 0);
		//imshow("edge", edge_dst);

		Mat Road_mask; // ʵ�ز��Ժ��ͼ���������
		Road_edge_dst.copyTo(Road_mask);

		// ���� ROI ������� mask (400, 0) (220, 420) (200, 860), (400, 1280)
		for (int i = 0; i < Road_mask.rows; ++i) //y
		{
			for (int j = 0; j < Road_mask.cols; ++j) //x
			{
				// 640 (x-j) * 480 (y-i)
				if (i >= (480 - 2 * j) && i >= (2 * j - 800))
				{
					// ��������ϵ ��ˮƽ��Ϊ�� ����ȥ
					Road_mask.at<uchar>(i, j) = 0;
					//src_color_image.at<cv::Vec3b>(i, j)[0] = 0;
					//src_color_image.at<cv::Vec3b>(i, j)[1] = 0;
					//src_color_image.at<cv::Vec3b>(i, j)[2] = 0;
				}
			}
		}

		imshow("Road_mask", Road_mask);

		//Mat edge0;
		//Canny(src_color_image, edge0, 100, 300, 3);
		//// ��color_img�ı�Եͨ��canny_img���в�ȫ
		//cv::Mat result_img;
		//cv::bitwise_or(edge0, Road_dst_image, result_img);
		////Clear_MicroConnected_Areas(result_img, result_img, 50);
		//imshow("��·��Ե-reault", result_img);



		/* ================================================ ��·��Ե ���� ================================================= */

		//// ��ȡ����
		//findContours(edge_dst, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
		////��������С����
		//std::sort(contours.begin(), contours.end(), ContoursSortFun);
		//Vec4f lines; //�����Ϻ��ֱ��
		//vector<Point>points; //������Ƿ����ֱ�ߵ����е�
		// 
		//for (int i = 0; i < contours.size(); i++)
		//{
		//	if (contours[i].size() < 20)
		//	{
		//		break;
		//	}
		//	points = contours[i];
		//	
		//	//��������
		//	double param = 0; //����ģ���е���ֵ����C
		//	double reps = 0.01; //����ԭ����ֱ��֮��ľ���
		//	double aeps = 0.01; //�ǶȾ���
		//	fitLine(points, lines, DIST_L1, 0, 0.01, 0.01);
		//	
		//	// 0��1����б�ʣ�2��3�����
		//	// lines[0] dx
		//	// lines[1] dy
		//	// lines[2] x
		//	// lines[3] x
		//
		//	float k, b;
		//	k = lines[1] / lines[0];				//   dy / dx    ������ֱ�ߵ�б��
		//	b = lines[3] - k * lines[2];		//   y=kx+b	 ������ֱ�ߵĽؾ�
		//	int minx = 640;
		//	int maxx = 0;
		//
		//	//�����������е����е㣬�ҵ�������С��x����
		//	for (int t = 0; t < contours[i].size(); t++)
		//	{
		//		if (minx > contours[i][t].x)
		//		{
		//			minx = contours[i][t].x;
		//		}
		//		if (maxx < contours[i][t].x)
		//		{
		//			maxx = contours[i][t].x;
		//		}
		//	}
		//	int miny = k * minx + b;
		//	int maxy = k * maxx + b;
		//	line(aligned_image_color, cv::Point(minx, miny), cv::Point(maxx, maxy), cv::Scalar(0, 255, 0), 1);
		//}

		/* ================================================ ��·��Ե ���� ================================================= */

		namedWindow("��·-��ʶ���");
		//�ڴ��������д���һ��������
		char TranckbarName[50] = "���򽻵���ֵ ���1000";
		createTrackbar(TranckbarName, "��·-��ʶ���", &AlphaValuesSlider, MaxAlphaValue, on_Trackbar, 0);
		on_Trackbar(AlphaValuesSlider, 0);

		/* ================================================ ��·��Ե ����ֱ����� ================================================= */

		//-------------------------- ���� �߶����
		vector<cv::Vec4i> Road_Hough_lines;
		HoughLinesP(Road_mask, Road_Hough_lines, 1, CV_PI / 180, AlphaValuesSlider, 30, 30);

		for (size_t i = 0; i < Road_Hough_lines.size(); i++)
		{
			Vec4i Road_linex = Road_Hough_lines[i];
			//int dx = Road_linex[2] - Road_linex[0];
			//int dy = Road_linex[3] - Road_linex[1];
			//double angle = atan2(double(dy), dx) * 180 / CV_PI;
			
			//line(aligned_image_color, cv::Point(Road_linex[0], Road_linex[1]), cv::Point(Road_linex[2], Road_linex[3]), cv::Scalar(255, 0, 0), 1);
		}
		//-------------------------- ���� �߶����

		//// ��1 ��㡿------------------------- ��� ʹ����ɫ��ֵ [��ɫ��+��ɫ��]
		//// ͨ������ houghp ��ȡ�����߶Σ���ȡ���λ��
		//// Check if lines found and exit if not.
		//if (Road_Hough_lines.size() == 0)
		//{
		//	std::cout << "Not enough lines found in the image for Vanishing Point detection." << std::endl;
		//}
		////Filtering Lines wrt angle
		//std::vector<std::vector<double>> FilteredLines;
		//FilteredLines = FilterLines(Road_Hough_lines);
		//// Get vanishing point
		//int* VanishingPoint = GetVanishingPoint(FilteredLines);
		//// Checking if vanishing point found
		//if (VanishingPoint[0] == -1 && VanishingPoint[1] == -1)
		//{
		//	std::cout << "Vanishing Point not found. Possible reason is that not enough lines are found in the image for determination of vanishing point." << std::endl;
		//	continue;
		//}
		//// Drawing linesand vanishing point
		//for (int i = 0; i < FilteredLines.size(); i++)
		//{
		//	std::vector<double> Line = FilteredLines[i];
		//	//cv::line(aligned_image_color, cv::Point((int)Line[0], (int)Line[1]), cv::Point((int)Line[2], (int)Line[3]), cv::Scalar(0, 0, 255), 2);
		//}
		////cv::circle(aligned_image_color, cv::Point(VanishingPoint[0], VanishingPoint[1]), 10, cv::Scalar(0, 0, 255), -1);
		//// ��1 ��㡿------------------------- ��� ʹ����ɫ��ֵ [��ɫ��+��ɫ��]



		//// ��2 ��㡿------------------------- ��� ʹ�ûҶ�ͼ+canny [��ɫ��+��ɫ��]
		//// ͨ���Ҷ�ͼ��canny��ȡ�߶Σ��ټ������λ��
		//std::vector<std::vector<double>> Lines;
		//Lines = GetLines(src_color_image);
		//// Get vanishing point
		//int* VanishingPoint_1 = GetVanishingPoint(Lines);
		//// Checking if vanishing point found
		//if (VanishingPoint_1[0] == -1 && VanishingPoint_1[1] == -1)
		//{
		//	std::cout << "Vanishing Point not found. Possible reason is that not enough lines are found in the image for determination of vanishing point." << std::endl;
		//	continue;
		//}
		//// Drawing linesand vanishing point
		//for (int i = 0; i < Lines.size(); i++)
		//{
		//	std::vector<double> Line = Lines[i];
		//	//cv::line(aligned_image_color, cv::Point((int)Line[0], (int)Line[1]), cv::Point((int)Line[2], (int)Line[3]), cv::Scalar(0, 255, 0), 2);
		//}
		////cv::circle(aligned_image_color, cv::Point(VanishingPoint_1[0], VanishingPoint_1[1]), 10, cv::Scalar(0, 255, 0), -1);
		////��2 ��㡿 ------------------------- ��� ʹ�ûҶ�ͼ+canny [��ɫ��+��ɫ��]



		// �����߷����ң�
		// https://zhuanlan.zhihu.com/p/60891432
		// https://zhuanlan.zhihu.com/p/630083399

		// ��3 ���ࡿ------------------------- ���� [����ɫ��]
		// ��֮ǰ��ȡ����ֱ�ߣ����򣩽��о��࣬�ֳ������ߵ�����

		// ======================== ��step 1���߶ξ��� & ��step 2���ҳ�ÿ���߶εĴ���ֱ��
		// ���������ֵ������ʵ���������
		float cluster_distanceThreshold = 20.0;

		// ���߶ν��о���
		std::vector<std::vector<cv::Vec4i>> clusters = clusterLines(Road_Hough_lines, cluster_distanceThreshold);
		vector<cv::Vec4f> represent_line;//ÿ���߶εĴ����߶�

		float represent_line_slope = 0.0, represent_line_slope_sum = 0.0; //б�ʺ�
		float represent_line_midpoint_x = 0.0, represent_line_midpoint_x_sum = 0.0;
		float represent_line_midpoint_y = 0.0, represent_line_midpoint_y_sum = 0.0;
		float represent_line_b = 0.0, represent_line_b_sum = 0.0;
		float represent_line_angle = 0.0;

		// ���Ը�����Ҫ����������������������
		for (const auto& cluster : clusters)
		{
			represent_line_slope_sum = 0.0;
			represent_line_b_sum = 0.0;
			represent_line_midpoint_x_sum = 0.0;
			represent_line_midpoint_y_sum = 0.0;

			for (const auto& line : cluster)
			{
				// cluster�����һ��ֱ�ߣ�һֱѭ�����cluster
				// �������ĸ�����������ֱ��ǰ���࣬�����ĸ���
				// �е�
				represent_line_midpoint_x = (line[0] + line[2]) * 0.5;
				represent_line_midpoint_y = (line[1] + line[3]) * 0.5;
				represent_line_midpoint_x_sum = represent_line_midpoint_x_sum + represent_line_midpoint_x;
				represent_line_midpoint_y_sum = represent_line_midpoint_y_sum + represent_line_midpoint_y;
				
				// б�� & b
				int dx = line[2] - line[0];
				int dy = line[3] - line[1];
				if (dx == 0)
				{
					represent_line_slope_sum = 1000000 + represent_line_slope_sum;
					represent_line_b = line[3] - 1000000 * line[2];
					represent_line_b_sum = represent_line_b + represent_line_b_sum;
				}
				else
				{
					represent_line_slope_sum = dy / dx + represent_line_slope_sum;
					represent_line_b = line[3] - dy / dx * line[2];
					represent_line_b_sum = represent_line_b + represent_line_b_sum;
				}

				//cv::line(aligned_image_color, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 156, 28), 1);
			}
	
			// ÿ��cluster�Ĵ�����
			represent_line_slope = (float)represent_line_slope_sum / cluster.size();
			represent_line_midpoint_x = (float)represent_line_midpoint_x_sum / cluster.size();
			represent_line_midpoint_y = (float)represent_line_midpoint_y_sum / cluster.size();
			represent_line_b =(float)represent_line_b_sum / cluster.size();
			represent_line_angle = atan(represent_line_slope) * 180.0 / 3.1415926;

			cv::Vec4f line(1, represent_line_slope, represent_line_midpoint_x, represent_line_midpoint_y);
			represent_line.push_back(line);
			
			//cout << "�������߶Ρ��߶��е�Ϊ����" << represent_line_midpoint_x << "��" << represent_line_midpoint_y << "����";
			//cout << "�߶���ˮƽ�ߵĽǶ�Ϊ��" << represent_line_angle << endl;
			
			draw_line_k(src_color_image, represent_line_slope, Point(represent_line_midpoint_x, represent_line_midpoint_y),
				cv::Scalar(28, 156, 255), 1);


		}
		// ======================== ��step 1���߶ξ��� & ��step 2���ҳ�ÿ���߶εĴ���ֱ��
		
		if (clusters.size() < 2)
		{
			cout << "���಻�����࣡" << endl;
			continue;
		}

	 

		// -------------------------��step 2��һ���߶��ҽ�ƽ����

		// ��ȷ��б�ʣ�����һ���㼴�ɻ�����ƽ���ߡ��жϽ�ƽ������ˮƽ�ߵĽǶȣ���90�����Ҽ��ɡ�
		// �������֮��Ĵ���ֱ��ֻ������

		float bisector_k1, bisector_k2, bisector_k_average;
		bisector_k1 = represent_line[0][1] / represent_line[0][0];
		bisector_k2 = represent_line[1][1] / represent_line[1][0];
		bisector_k_average = (float)(bisector_k1 + bisector_k2) / 2.0;

		float bisector_midX1 = represent_line[0][2];
		float bisector_midY1 = represent_line[0][3];
		float bisector_midX2 = represent_line[1][2];
		float bisector_midY2 = represent_line[1][3];
		float bisector_X_average, bisector_Y_average;
		bisector_X_average = (float)(bisector_midX1 + bisector_midX2) / 2.0;
		bisector_Y_average = (float)(bisector_midY1 + bisector_midY2) / 2.0;

		float bisector_b = bisector_Y_average - (bisector_k_average * bisector_X_average);

		float bisector_angle_mid = atan(bisector_k_average) * 180.0 / 3.1415926;

		draw_line_k(src_color_image, bisector_k_average, cv::Point(bisector_X_average, bisector_Y_average),
			cv::Scalar(255, 255, 255), 1);


		//cout << "����ƽ�֡���ƽ���� y = " << bisector_k_average << "x + " << bisector_b ;
		//cout << "���߶���ˮƽ�ߵĽǶ�Ϊ��" << bisector_angle_mid << endl;

		// -------------------------��step 3��һ��ֱ���ҽ�ƽ����
		

		// ��3 ���ࡿ------------------------- ���� [����ɫ��]

		// ��������
		// ���� ��ƽ������ˮƽ�ߵļн� bisector_angle_mid

		// ��������͸�ӱ任
		// https://www.zhihu.com/question/53021663

		// ��㣿
		// ȷ�����֮��Ͳ���Ҫ�����ߵľ�������
		// https://blog.csdn.net/A_L_A_N/article/details/89575707
		// https://blog.csdn.net/ydy1107/article/details/121355836


/* ================================================ ��·��Ե ����ֱ����� ================================================= */


		sprintf_s(fps_char, "fps: %f", fps);
		cv::putText(src_color_image, (string)fps_char, cv::Point(20, 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 55), 2, 5);
		imshow("��·-��ʶ���", src_color_image);

		// ��������
		// ��·�е���ͼ�����ĵľ���


		
		////-------------------------- �����ȡ���
		//imshow("��ʶ���", aligned_image_color);
		//setMouseCallback("��ʶ���", on_Mouse_Distance, &aligned_image_color);
		//if (buttondown == 2)
		//{
		//	// ע�⣺��Ȼ��ʾ��ͼ�����ǻ�ɫ�㣬���ǵ����ʱ���ǻ����������
		//	dist_to_center = aligned_depth_frame.get_distance(px, py);
		//	cout << "The object is " << dist_to_center << " meters away "<<endl;
		//	buttondown = 0;
		//}
		////-------------------------- �����ȡ���


/* #################################################### ��·��Ե ���� #################################################### */


/* #################################################### ̨�� ��ʼ #################################################### */

		//// �״��յ��ĵ�		received_x		received_y
		//// ���Ե����ŵ�		best_x			best_y
		//// �����״�Ͳ��ԵĽ��������Ĳ��������޸�
		//// ��Χ			+- 
		//int received_x, received_y;
		//int best_x, best_y;
		//float distance_footstep = 0.0; // ����̨�׵ľ���

		//// ���վλ׼ȷ���������� ����������ת����
		//if (received_x - best_x > -50 && received_x - best_x<50 && received_y - best_y>-50 && received_y - best_y < 50)
		//{
		//	for (int x_i = 0; x_i < 5; x_i++)
		//	{
		//		for (int y_i = 0; y_i < 5; y_i++)
		//		{
		//			distance_footstep = aligned_depth_frame.get_distance(320 - x_i, 240 - y_i) + distance_footstep; // �е�+-5�ķ�Χ
		//			distance_footstep = aligned_depth_frame.get_distance(320 + x_i, 240 + y_i) + distance_footstep;
		//		}
		//	}
		//	distance_footstep = distance_footstep / 25;

		//	// ��������
		//	// �����е�����

		//}
		//
		//// ���վλ��׼ȷ������ʵ�ʲ��Խ���������ݵ���վ�ˡ��ܵ�������������ת����
		//// 
		//
		//


/* #################################################### ̨�� ���� #################################################### */



/* #################################################### ˫ľ�� ��ʼ #################################################### */


		// 
		// ���Ϊ�����׵ĵ㣬RANSAC���ֱ�ߣ����ߣ��Ƕ�90�����ң����ҽ��㣬�����ڻ������߸�����
		// 


		Mat Bridge_depth_image = My_D435if_Camera.readDepthCamera();
		Mat Bridge_depth_image_colored;
		float diatance = 0;

		convertScaleAbs(Bridge_depth_image, Bridge_depth_image, 1.0, 0.0); //�ڰ�
		applyColorMap(Bridge_depth_image, Bridge_depth_image_colored, COLORMAP_AUTUMN); //��ɫ

		// ������ݴ洢��ά����
		int height = Bridge_depth_image_colored.rows;	//y
		int width = Bridge_depth_image_colored.cols;	//x
		vector<vector<float>> Bridge_depthArray(height, vector<float>(width, 0.0));
		//eg ͼƬ�ǿ�200����100�������������100�У�200�е����顣[100][200]
		//�ڲ���vector<float>��ʾÿһ�е����������Ĵ�СΪwidth���ҳ�ʼֵΪ0��


		// �����ͼ�����ݴ洢����ά������
		for (int i = 0; i < Bridge_depth_image_colored.rows; i++)
		{
			for (int j = 0; j < Bridge_depth_image_colored.cols; j++)
			{
				// ��ȡ���ֵ
				diatance = aligned_depth_frame.get_distance(j, i);
				// �浽��ά������
				Bridge_depthArray[i][j] = diatance;
			}
		}


		// ����� 20cm ���ҵ����ݵ�
		vector<Point2f> Bridge_specfic_Points; 
		// 1 cm ��0.01
		for (int i = 1; i < height - 1; i++) 
		{
			for (int j = 1; j < width - 1; j++) 
			{
				int depthDiff = Bridge_depthArray[i][j] * 100;

				if (depthDiff >= 25 && depthDiff <= 30) //25���׵�30����
				{
					cv::circle(Bridge_depth_image_colored, Point(j, i), 1, Scalar(255, 255, 255));
					Bridge_specfic_Points.push_back(Point(j, i));
				}
			}
		}

		// ��֪б�� k ����֪����һ�� x1,y1 ��ֱ�ߣ������ң�
		// ��Ҫ���ֱ��������������
		// P1�� 0 ��x1 + k*(0-x1) ��
		// P2��640��x1 + k*(640-x1) ��
		// cv::line(img, P1, P2, cv::Scalar(255, 0, 0), 1)


		clock_t time11 = clock();
		Vec4f Bridge_line_Ransac;
		fitLineRansac(Bridge_specfic_Points, Bridge_line_Ransac, 300, 1, -10, 10);  // ����fitLineRansac�������һ��ֱ��
		
		//100 30
		//300 100

		clock_t time22 = clock();
		cout << "RANSAC��ʱ"<<time22 - time11 << endl;

		float k = Bridge_line_Ransac[1] / Bridge_line_Ransac[0];
		float b = Bridge_line_Ransac[3] - k * Bridge_line_Ransac[2];
		cv::Point p1, p2, p3;
		p1.y = 640;
		p1.x = (p1.y - b) / k;
		p2.y = 0;
		p2.x = (p2.y - b) / k;
		
		p3.y = 320;
		p3.x = (p3.y - b) / k;
		
		cv::line(Bridge_depth_image_colored, p1, p2, cv::Scalar(0, 255, 0), 2);

		double vertical_line_k = -Bridge_line_Ransac[0] / Bridge_line_Ransac[1]; // ����б��
		draw_line_k(Bridge_depth_image_colored, vertical_line_k, p3, cv::Scalar(255, 0, 0), 2);

		sprintf_s(fps_char, "fps: %f", fps);
		cv::putText(Bridge_depth_image_colored, (string)fps_char, cv::Point(20, 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 55), 2, 5);
		imshow("˫ľ��-depth", Bridge_depth_image_colored);

		

/* #################################################### ˫ľ�� ���� #################################################### */

		contours.clear();
		hierarchy.clear();
		//Road_lines.clear();
		aligned_image_color.release();
		


		if (waitKey(1) == 27)
		{
			destroyAllWindows();
			break;
		}

	}
	return EXIT_SUCCESS;
}

catch (const rs2::error& e) {
	cout << "RealSense error calling" << e.get_failed_function() << "(" << e.get_failed_args() << "):\n"
		<< e.what() << endl;
	return EXIT_FAILURE;
}

catch (const exception& e) {
	cout << e.what() << endl;
	return EXIT_FAILURE;
}

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
	for (auto&& sp : prev)
	{
		// If previous profile is in current (maybe just added another)
		// �豸����
		auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp) { return sp.unique_id() == current_sp.unique_id(); });
		if (itr == std::end(current)) //If it previous stream wasn't found in current
		{
			return true;
		}
	}
	return false;
}

void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist)
{
	//��ȡָ������֡��ԭʼ��������ָ�룬�Ա����ǿ��Ը��Ĳ�ɫͼ�񣨶����Ǵ����µĻ���������
	const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
	uint8_t* p_other_frame = reinterpret_cast<uint8_t*>(const_cast<void*>(other_frame.get_data()));

	int width = other_frame.get_width();
	int height = other_frame.get_height();
	int other_bpp = other_frame.get_bytes_per_pixel();

#pragma omp parallel for schedule(dynamic) 
	//Using OpenMP to try to parallelise the loop
	//����֡��ÿ������
	for (int y = 0; y < height; y++)
	{
		auto depth_pixel_index = y * width;
		for (int x = 0; x < width; x++, ++depth_pixel_index)
		{
			// Get the depth value of the current pixel
			// ��������ص���Ⱦ���
			auto pixels_distance = depth_scale * p_depth_frame[depth_pixel_index];

			// Check if the depth value is invalid (<=0) or greater than the threashold
			// ����þ�����Ч����û�������������Զ����ô�����ɵĲ�ɫͼ����ȥ�������ء�
			if (pixels_distance <= 0.f || pixels_distance > clipping_dist)
			{
				//�û�ɫ���Ƹ����ز��ڷ�Χ�ڵ�����
				// Calculate the offset in other frame's buffer to current pixel
				auto offset = depth_pixel_index * other_bpp;

				// Set pixel to "background" color (0x999999)
				std::memset(&p_other_frame[offset], 0x99, other_bpp);
			}
		}
	}
}

//���ص� hsv
void on_Mouse_HSV(int event, int x, int y, int flags, void* param)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		px = x;
		py = y;
		buttondown = 1;
	}
}

//���ص� ����
void on_Mouse_Distance(int event, int x, int y, int flags, void* param)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		px = x;
		py = y;
		buttondown = 2;
	}
}

std::vector<std::vector<double>> GetLines(cv::Mat Image)
{
	cv::Mat GrayImage, BlurGrayImage, EdgeImage;
	// Converting to grayscale
	cv::cvtColor(Image, GrayImage, cv::COLOR_BGR2GRAY);
	// Blurring image to reduce noise.
	cv::GaussianBlur(GrayImage, BlurGrayImage, cv::Size(5, 5), 1);
	// Generating Edge image
	cv::Canny(BlurGrayImage, EdgeImage, 50, 150);
	imshow("GetLines-EdgeImage", EdgeImage);

	// Finding Lines in the image
	std::vector<cv::Vec4i> Lines;
	cv::HoughLinesP(EdgeImage, Lines, 1, CV_PI / 180, 50, 15);

	// Check if lines found and exit if not.
	if (Lines.size() == 0)
	{
		std::cout << "Not enough lines found in the image for Vanishing Point detection." << std::endl;
		//exit(3);
	}

	//Filtering Lines wrt angle
	std::vector<std::vector<double>> FilteredLines;
	FilteredLines = FilterLines(Lines);

	return FilteredLines;
}

std::vector<std::vector<double>> FilterLines(std::vector<cv::Vec4i> Lines)
{
	std::vector<std::vector<double>> FinalLines;

	for (int i = 0; i < Lines.size(); i++)
	{
		cv::Vec4i Line = Lines[i];
		int x1 = Line[0], y1 = Line[1];
		int x2 = Line[2], y2 = Line[3];

		double m, c;

		// Calculating equation of the line : y = mx + c
		if (x1 != x2)
			m = (double)(y2 - y1) / (double)(x2 - x1);
		else
			m = 100000000.0;
		c = y2 - m * x2;

		// theta will contain values between - 90 -> + 90.
		double theta = atan(m) * (180.0 / CV_PI);

		/*# Rejecting lines of slope near to 0 degree or 90 degree and storing others
		if REJECT_DEGREE_TH <= abs(theta) <= (90 - REJECT_DEGREE_TH):
			l = math.sqrt( (y2 - y1)**2 + (x2 - x1)**2 )    # length of the line
			FinalLines.append([x1, y1, x2, y2, m, c, l])*/
			// Rejecting lines of slope near to 0 degree or 90 degree and storing others
		if (REJECT_DEGREE_TH <= abs(theta) && abs(theta) <= (90.0 - REJECT_DEGREE_TH))
		{
			double l = pow((pow((y2 - y1), 2) + pow((x2 - x1), 2)), 0.5);	// length of the line
			std::vector<double> FinalLine{ (double)x1, (double)y1, (double)x2, (double)y2, m, c, l };
			FinalLines.push_back(FinalLine);
		}
	}

	// Removing extra lines
	// (we might get many lines, so we are going to take only longest 15 lines 
	// for further computation because more than this number of lines will only
	// contribute towards slowing down of our algo.)
	if (FinalLines.size() > 15)
	{
		std::sort(FinalLines.begin(), FinalLines.end(),
			[](const std::vector< double >& a,
				const std::vector< double >& b)
			{ return a[6] > b[6]; });

		std::vector<std::vector<double>> FinalLines2;
		FinalLines = std::vector<std::vector<double>>(FinalLines.begin(), FinalLines.begin() + 15);
	}

	return FinalLines;
}

int* GetVanishingPoint(std::vector<std::vector<double>> Lines)
{
	// We will apply RANSAC inspired algorithm for this.We will take combination
	// of 2 lines one by one, find their intersection point, and calculate the
	// total error(loss) of that point.Error of the point means root of sum of
	// squares of distance of that point from each line.
	int* VanishingPoint = new int[2];
	VanishingPoint[0] = -1; 
	VanishingPoint[1] = -1;

	double MinError = 1000000000.0;

	for (int i = 0; i < Lines.size(); i++)
	{
		for (int j = i + 1; j < Lines.size(); j++)
		{
			double m1 = Lines[i][4], c1 = Lines[i][5];
			double m2 = Lines[j][4], c2 = Lines[j][5];

			if (m1 != m2)
			{
				double x0 = (c1 - c2) / (m2 - m1);
				double y0 = m1 * x0 + c1;

				double err = 0;
				for (int k = 0; k < Lines.size(); k++)
				{
					double m = Lines[k][4], c = Lines[k][5];
					double m_ = (-1 / m);
					double c_ = y0 - m_ * x0;

					double x_ = (c - c_) / (m_ - m);
					double y_ = m_ * x_ + c_;

					double l = pow((pow((y_ - y0), 2) + pow((x_ - x0), 2)), 0.5);

					err += pow(l, 2);
				}

				err = pow(err, 0.5);

				if (MinError > err)
				{
					MinError = err;
					VanishingPoint[0] = (int)x0;
					VanishingPoint[1] = (int)y0;
				}
			}
		}
	}

	return VanishingPoint;
}

// ������ຯ��
std::vector<std::vector<cv::Vec4i>> clusterLines(std::vector<cv::Vec4i>& lines, double distanceThresh) 
{
	std::vector<std::vector<cv::Vec4i>> clusters; // �洢������������

	for (const auto& line : lines) //line�����޸ģ�����ֻ��ȡlines������
	{
		bool assignedToCluster = false;

		// �������еľ���
		for (auto& cluster : clusters) 
		{
			// �����߶ε����ĵ�����
			cv::Point2f lineCenter = cv::Point2f((line[0] + line[2]) / 2.0, (line[1] + line[3]) / 2.0);

			// ���������е��߶�
			for (const auto& existingLine : cluster) 
			{
				// ����������߶ε����ĵ�����
				cv::Point2f existingLineCenter = cv::Point2f((existingLine[0] + existingLine[2]) / 2.0,	(existingLine[1] + existingLine[3]) / 2.0);

				// �����߶�֮��ľ���
				double distance = cv::norm(lineCenter - existingLineCenter);

				// �������С����ֵ���򽫵�ǰ�߶μ������
				if (distance <= distanceThresh) 
				{
					cluster.push_back(line);
					assignedToCluster = true;
					break; // ��ֹ�ڲ�ѭ��
				}
			}

			if (assignedToCluster)
				break; // ��ֹ���ѭ��
		}

		// �����ǰ�߶�û�б����䵽�κξ����У��򴴽��µľ���
		if (!assignedToCluster) 
		{
			clusters.push_back({ line });
		}
	}

	return clusters;
}

void Clear_MicroConnected_Areas(cv::Mat src, cv::Mat& dst, double min_area)
{
	// ���ݸ���
	dst = src.clone();
	std::vector<std::vector<cv::Point> > contours;  // ������������
	std::vector<cv::Vec4i> 	hierarchy;

	// Ѱ�������ĺ���
	// ���ĸ�����CV_RETR_EXTERNAL����ʾѰ������Χ����
	// ���������CV_CHAIN_APPROX_NONE����ʾ��������߽������������������㵽contours������
	cv::findContours(src, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE, cv::Point());

	if (!contours.empty() && !hierarchy.empty())
	{
		std::vector<std::vector<cv::Point> >::const_iterator itc = contours.begin();
		// ������������
		while (itc != contours.end())
		{
			// ��λ��ǰ��������λ��
			cv::Rect rect = cv::boundingRect(cv::Mat(*itc));
			// contourArea����������ͨ�����
			double area = contourArea(*itc);
			// �����С�����õ���ֵ
			if (area < min_area)
			{
				// ������������λ���������ص�
				for (int i = rect.y; i < rect.y + rect.height; i++)
				{
					uchar* output_data = dst.ptr<uchar>(i);
					for (int j = rect.x; j < rect.x + rect.width; j++)
					{
						// ����ͨ����ֵ��0
						if (output_data[j] == 255)
						{
							output_data[j] = 0;
						}
					}
				}
			}
			itc++;
		}
	}
}


float Get_K(float x1, float y1, float x2, float y2)
{
	float k;
	if (x1 != x2)
		k = (float)(y2 - y1) / (float)(x2 - x1);
	else
		k = 100000000.0;
	return k;
}


//RANSAC ���2D ֱ��
//���������
//			points--����㼯
//			iterations--��������
//			sigma--���ݺ�ģ��֮��ɽ��ܵĲ�ֵ,���������ؿ��һ��Ϊ10����
//              ��Parameter use to compute the fitting score��
//			k_min/k_max--��ϵ�ֱ��б�ʵ�ȡֵ��Χ.
//                     ���ǵ����ҳ�������ͼ���е�б��λ��һ����Χ�ڣ�
//                      ��Ӵ˲�����ͬʱ���Ա����ⴹ�ߺ�ˮƽ��

void fitLineRansac(const std::vector<cv::Point2f>& points,cv::Vec4f& line,int iterations = 1000,double sigma = 1.0,double k_min = -7.0,double k_max = 7.0)
{
	unsigned int n = points.size(); // �㼯��С

	if (n < 2) 
	{
		return;// �㼯�еĵ�С��2���˳�
	}

	cv::RNG rng; // �����������
	double bestScore = -1.0; // ���ƥ��÷֣���ʼ��Ϊ-1.0

	int i1 = 0, i2 = 0;

	for (int k = 0; k < iterations; k++)
	{
		// ���ѡ��������ͬ�ĵ������
		i1 = rng(n);
		i2 = rng(n);
		const cv::Point2f& p1 = points[i1];
		const cv::Point2f& p2 = points[i2];

		cv::Point2f dp = p2 - p1; //ֱ�ߵķ�������
		double score = 0; // ��ǰģ�͵ĵ÷�
		dp = dp * 1.0 / norm(dp); // ��һ����������
		
		if (dp.y / dp.x <= k_max && dp.y / dp.x >= k_min)
		{
			// ���б���Ƿ��ڸ�����Χ��
			for (int i = 0; i < n; i++)
			{
				// ����㵽ֱ�ߵ�����
				cv::Point2f v = points[i] - p1;
				double d = (double)v.y * dp.x - (double)v.x * dp.y; //����a��b���/����b��ģ.||b||=1./norm(dp)
				if (fabs(d) < sigma)
				{
					// ʹ������ķ�ʽ����÷�
					score = score+1;
				}
			}
		}
		if (score > bestScore)
		{
			// �����ǰ�÷ֱ���ѵ÷ִ��������ѵ÷ֺ�ֱ�߲���
			line = cv::Vec4f(dp.x, dp.y, p1.x, p1.y);
			bestScore = score;
		}
	}

}


void draw_line_k(Mat image,float k, cv::Point point, const cv::Scalar &color,int thickness )
{
	cv::Point p1, p2;
	p1.x = 0;
	p1.y = point.y + k * (p1.x - point.x);
	p2.x = 640;
	p2.y = point.y + k * (p2.x - point.x);
	cv::line(image, p1, p2, color, thickness);
};

