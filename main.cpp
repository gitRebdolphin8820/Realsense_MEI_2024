#include <iostream>
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

#include <librealsense2/rs.hpp>
#include <opencv2/core.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc/imgproc_c.h>
#include <opencv2/highgui/highgui_c.h>

#include "omp.h"

// ROS 
//#include "ros/ros.h"
//#include <std_msgs/Float64MultiArray.h>
//#include <nav_msgs/Path.h>
//#include <sstream>
//
//float route[2];
float cam_data[6];
//
//void pathCallback(const nav_msgs::Path::ConstPtr& pathMsg)
//{
//	// 处理接收到的路径数据
//	const auto& pose = pathMsg->poses[0];
//	route[0] = pose.pose.position.x;
//	route[1] = pose.pose.position.y;
//}


using namespace std;
using namespace cv;
using namespace rs2;

Mat Road_element_erode, Road_element_dilate; // 图像运算的核
Mat Road_HSV_image, Road_dst_image;
Mat Road_edge_0, Road_edge_dst;

// 画轮廓
vector<vector<cv::Point>> contours; // 轮廓点集的集合
vector<cv::Vec4i> hierarchy;		// 层级

// 滑动条 霍夫直线交点的阈值
int AlphaValuesSlider;		   // 滑动条对应的变量
const int MaxAlphaValue = 500; // Alpha的最大值

int H, S, V;
int buttondown = 0;
int px = 0, py = 0;			  // 鼠标点击点
float dist_to_center = 0.0;	  // 距离
float fps;					  // 帧率
float REJECT_DEGREE_TH = 4.0; // 灭点？
char fps_char[50];			  // 屏幕显示
clock_t time1, time2;		  // 计算fps变量

// https://dev.intelrealsense.com/docs/rs-align-advanced
// 在该网站中可以看到函数的具体解释

// https://dev.intelrealsense.com/docs
// 该网站为帮助文档

// 响应滑动条的回调函数
void on_Trackbar(int v, void*)
{
	AlphaValuesSlider = v;
}

// 轮廓大小排序
static inline bool ContoursSortFun(vector<cv::Point> contour1, vector<cv::Point> contour2)
{
	return (cv::contourArea(contour1) > cv::contourArea(contour2));
}

// 检查摄像头数据管道设置是否改变
bool profile_changed(const vector<stream_profile>& current, const vector<stream_profile>& prev);

// 获取深度和颜色图像（假定彼此对齐）、深度比例单位和用户希望显示的最大距离，
// 并更新颜色框，以便删除其背景（深度距离大于允许的最大值的任何像素）。
void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist);

// 点击输出数据 鼠标回调函数
void on_Mouse_HSV(int event, int x, int y, int flags, void*);
void on_Mouse_Distance(int event, int x, int y, int flags, void*);

// 灭点 获取直线
std::vector<std::vector<double>> GetLines(cv::Mat Image);
// 灭点 获取直线的函数
std::vector<std::vector<double>> FilterLines(std::vector<cv::Vec4i> Lines);
// 灭点 获取灭点
int* GetVanishingPoint(std::vector<std::vector<double>> Lines);

// 对直线进行聚类
std::vector<std::vector<cv::Vec4i>> clusterLines(std::vector<cv::Vec4i>& lines, double distanceThresh);

void fitLineRansac(const std::vector<cv::Point2f>& points, cv::Vec4f& line, int iterations, double sigma, double k_min, double k_max);

void draw_line_k(Mat image, float k, cv::Point point, const cv::Scalar& color, int thickness);

class D435if_Camera
{
public:
	rs2::config cfg;
	rs2::pipeline pipe; // 声明realsense管道
	rs2::frameset frame;
	rs2::pipeline_profile profile;
	rs2_intrinsics depth_intrinsics; // 深度摄像头内参
	rs2_intrinsics color_intrinsics; // 彩色摄像头内参

	D435if_Camera();
	void start();
	void stop();
	float get_depth_scale(rs2::device dev); // 获取单个深度单位表示的米数
	rs2_stream find_stream_to_align(const std::vector<rs2::stream_profile>& streams);

	Mat readColorCamera();
	Mat readDepthCamera();
	Mat readAlignedCamera(rs2_stream align_to, rs2::align align, frameset frameset_data, float depth_scale, float depth_clipping_distance);
	Mat frame_to_mat(const rs2::frame& f);
};

D435if_Camera::D435if_Camera()
{
	// 构造函数
}

void D435if_Camera::start()
{
	// 读取设备型号和序列号
	rs2::context ctx;
	auto devicelist = ctx.query_devices();
	rs2::device dev = *devicelist.begin(); // 单个设备
	cout << "RS2_CAMERA_NAME： " << dev.get_info(RS2_CAMERA_INFO_NAME) << endl;
	cout << "RS2_CAMERA_SERIAL_NUMBER： " << dev.get_info(RS2_CAMERA_INFO_SERIAL_NUMBER) << endl;
	cout << "RS2_FIRMWARE_VERSION： " << dev.get_info(RS2_CAMERA_INFO_RECOMMENDED_FIRMWARE_VERSION) << endl;

	// 数据流配置
	cfg.enable_stream(RS2_STREAM_COLOR, 640, 480, RS2_FORMAT_BGR8, 30);
	cfg.enable_stream(RS2_STREAM_DEPTH, 640, 480, RS2_FORMAT_Z16, 30);

	// 启动设备
	profile = pipe.start(cfg);

	// 声明数据流
	auto _stream_depth = profile.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
	auto _stream_color = profile.get_stream(RS2_STREAM_COLOR).as<rs2::video_stream_profile>();

	// 获得摄像头内参
	depth_intrinsics = _stream_depth.get_intrinsics();
	color_intrinsics = _stream_color.get_intrinsics();

	// 打印内参
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
}

Mat D435if_Camera::frame_to_mat(const rs2::frame& f)
{
	auto vf = f.as<rs2::video_frame>();
	const int w = vf.get_width();
	const int h = vf.get_height();

	if (f.get_profile().format() == RS2_FORMAT_BGR8)
	{
		// cout << "RS2_FORMAT_BGR8" << endl;
		return Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
	}
	else if (f.get_profile().format() == RS2_FORMAT_RGB8)
	{
		// cout << "RS2_FORMAT_RGB8" << endl;
		auto r = Mat(Size(w, h), CV_8UC3, (void*)f.get_data(), Mat::AUTO_STEP);
		cvtColor(r, r, CV_RGB2BGR);
		return r;
	}
	else if (f.get_profile().format() == RS2_FORMAT_Z16)
	{
		// cout << "RS2_FORMAT_Z16" << endl;
		return Mat(Size(w, h), CV_16UC1, (void*)f.get_data(), Mat::AUTO_STEP);
	}
	else if (f.get_profile().format() == RS2_FORMAT_Y8)
	{
		// cout << "RS2_FORMAT_Y8" << endl;
		return Mat(Size(w, h), CV_8UC1, (void*)f.get_data(), Mat::AUTO_STEP);
	}

	throw std::runtime_error("Frame format is not supported yet!");
}

Mat D435if_Camera::readColorCamera()
{
	frame = pipe.wait_for_frames();
	auto color_frame = frame.get_color_frame();
	// cout << "- start ---------------------------------" << endl;
	Mat color = frame_to_mat(color_frame);
	// cout << "- end ---------------------------------" << endl;
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
	// 获取已处理的对齐帧
	auto frameset_processed = align.process(frameset_data);

	// 获取颜色和对齐的深度帧
	// 彩色帧是rs2::video_frame类型，深度帧if是rs2:depth_frame
	// （派生自rs2:：video_frame，并添加了特殊的深度相关功能）
	rs2::video_frame other_frame = frameset_processed.first(align_to);			 // video_frame
	rs2::depth_frame aligned_depth_frame = frameset_processed.get_depth_frame(); // depth_frame

	// If one of them is unavailable, continue iteration
	// 如果其中一个不能用，就继续获取下一帧
	if (!aligned_depth_frame || !other_frame)
	{
		cout << "数据帧不可用，获取下一帧" << endl;
	}
	else
	{
		remove_background(other_frame, aligned_depth_frame, depth_scale, depth_clipping_distance);
	}

	// Mat aligned_image_color(Size(640, 480), CV_8UC3, (void*)other_frame.get_data(), Mat::AUTO_STEP);
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
	// 查看用户的深度设备
	for (rs2::sensor& sensor : dev.query_sensors())
	{
		// Check if the sensor if a depth sensor
		if (rs2::depth_sensor dpt = sensor.as<rs2::depth_sensor>())
		{
			// When called on a depth sensor, this method will return the number of meters represented by a single depth unit
			// 当在深度传感器上调用时，此方法将返回由单个深度单位表示的米数
			return dpt.get_depth_scale();
		}
	}
	throw std::runtime_error("Device does not have a depth sensor");
}

// 找一个stream which 深度图可以向其对齐
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
			if (!color_stream_found) // Prefer color
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

int main()
try
{
	// ROS 
	//ros::init(argc, argv, "cam_node");
	//ros::NodeHandle nh;
	//ros::Publisher cam_pub = nh.advertise<std_msgs::Float64MultiArray>("/campub", 10);
	//ros::Subscriber cam_sub = nh.subscribe<nav_msgs::Path::ConstPtr>("/path", 10, pathCallback);
	//std_msgs Float64MultiArray cam_data;
	//ros::Rate r(20);

	D435if_Camera My_D435if_Camera;
	My_D435if_Camera.start();

	rs2::config cfg = My_D435if_Camera.cfg;
	rs2::pipeline pipe = My_D435if_Camera.pipe;
	rs2::frameset frame = My_D435if_Camera.frame;
	rs2::pipeline_profile profile = My_D435if_Camera.profile;

	// align_to 是计划将深度帧与之对齐的流类型。
	rs2_stream align_to = My_D435if_Camera.find_stream_to_align(profile.get_streams());
	rs2::align align(align_to);

	// 在实际使用帧之前，尝试获取深度相机的深度缩放单位。
	// 例如，如果我们有一个值为 2 的深度像素，并且深度比例单位为 0.5，则该像素距离相机一米。
	float depth_scale = My_D435if_Camera.get_depth_scale(profile.get_device());

	// 要去除的背景的距离，0.9f 是 90cm
	float depth_clipping_distance = 1.0f;

	namedWindow("道路-深彩对齐");

	// 在创建窗体中创建一个滑动条
	char TranckbarName[50] = "霍夫交点阈值 最大1000";
	createTrackbar(TranckbarName, "道路-深彩对齐", nullptr, MaxAlphaValue, on_Trackbar, 0);
	on_Trackbar(AlphaValuesSlider, 0);

	while (true)
	{
		time1 = clock(); // 计时
		if (time1 != 0)
		{
			fps = float(CLOCKS_PER_SEC) / (time1 - time2);
			time2 = time1;
		}

		/* ===================================== realsense 获取数据 ===================================== */

		// 【！！相机安装有俯仰！！深度数据需要修正！！】
		// 【相机点云数据生成 可以与kinect相配合 相机有俯仰时使用】
		//
		// https://dev.intelrealsense.com/docs/rs-measure

		// pipeline::wait_for_frames 可以在设备出错或断开连接时替换它使用的设备
		frameset frame_set_data = My_D435if_Camera.pipe.wait_for_frames(); // 堵塞程序直到新的一帧捕获

		if (profile_changed(pipe.get_active_profile().get_streams(), profile.get_streams()))
		{
			// If the profile was changed, update the align object, and also get the new device's depth scale
			// 如果配置文件已更改，更新对齐对象，并获取新设备的深度刻度
			profile = pipe.get_active_profile();
			align_to = My_D435if_Camera.find_stream_to_align(profile.get_streams());
			align = rs2::align(align_to);
			depth_scale = My_D435if_Camera.get_depth_scale(profile.get_device());
		}

		auto frameset_processed = align.process(frame_set_data);
		rs2::depth_frame aligned_depth_frame = frameset_processed.get_depth_frame(); // 获取深度

		/* ================================================ 图像处理 ================================================ */

		/*--------------------------------------------------------------------------------------------*/
		//
		// 对主路：一直看黄色（黑色线条？哪个效果好）的边界（可以裁掉中心的图像）
		// 【原始的彩图（需要把相机中心对齐到狗子的中心）】	比较合适
		// 【ERROR注意：彩图会跳变成对齐后的深度图，不知道为什么】
		//
		// 对台阶：根据雷达的数据发送视觉中心的深度数据
		// 【深度图】
		//
		// 对双木桥：？？？？？
		//
		/*--------------------------------------------------------------------------------------------*/

		Mat aligned_image_color = My_D435if_Camera.readAlignedCamera(align_to, align, frame_set_data, depth_scale, depth_clipping_distance);
		Mat src_color_image = My_D435if_Camera.readColorCamera();
		


		// imshow("src_color_image", src_color_image);

		/* #################################################### 道路边缘 开始 #################################################### */

		// 彩图HSV处理，阈值变黑白，霍夫拟合直线
		// 【发送数据： 角度 float 判断角度是否为90或-90左右即可】

		// cvtColor(src_color_image, Road_dst_image, cv::COLOR_BGR2GRAY);
		//// 大津法 阈值
		// threshold(Road_dst_image, Road_dst_image, 10, 255, CV_THRESH_OTSU);

		cvtColor(src_color_image, Road_HSV_image, cv::COLOR_BGR2HSV);
		inRange(Road_HSV_image, cv::Scalar(11, 43, 46), cv::Scalar(34, 255, 255), Road_dst_image); // 黄+橙色
		// inRange(Road_HSV_image, cv::Scalar(0, 0, 0), cv::Scalar(150, 255, 46), Road_dst_image); // 黑色

		//-------------------------- 点击获取HSV
		// imshow("HSV color", HSV_image);
		// setMouseCallback("HSV color", on_Mouse_HSV, &HSV_image);
		// if (buttondown == 1)-
		//{
		//	cout << "H: " << (int)HSV_image.at<cv::Vec3b>(py, px)[0] << '\t';
		//	cout << "S: " << (int)HSV_image.at<cv::Vec3b>(py, px)[1] << '\t';
		//	cout << "V: " << (int)HSV_image.at<cv::Vec3b>(py, px)[2] << endl;
		//	buttondown = 0;
		//}
		//-------------------------- 点击获取HSV

		Road_element_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
		Road_element_dilate = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));

		// erode(Road_dst_image, Road_dst_image, Road_element_erode); //腐蚀
		erode(Road_dst_image, Road_dst_image, Road_element_erode); // 腐蚀
		// imshow("道路-二值化后", Road_dst_image);
		GaussianBlur(Road_dst_image, Road_dst_image, cv::Size(5, 5), 0, 0);
		// imshow("道路-Road_dst_image", Road_dst_image);
		Canny(Road_dst_image, Road_edge_0, 100, 300, 3); // 3、4位 数值越小细节越多
		// imshow("Canny", edge_0);
		dilate(Road_edge_0, Road_edge_dst, Road_element_dilate); // 膨胀
		// imshow("edge dilate", edge_0);

		Mat Road_mask; // 实地测试后对图像进行遮罩
		Road_edge_dst.copyTo(Road_mask);

		// 梯形 ROI 区域进行 mask
		for (int i = 0; i < Road_mask.rows; ++i) // y
		{
			for (int j = 0; j < Road_mask.cols; ++j) // x
			{
				// 640 (x->j) * 480 (y->i)
				// if (i >= (480 - 3 * j) && i >= (3 * j - 800))
				if (j > 210 && j < 420)
				{
					// 正常坐标系 以水平线为轴 翻上去
					Road_edge_dst.at<uchar>(i, j) = 0;
					// src_color_image.at<cv::Vec3b>(i, j)[0] = 0;
					// src_color_image.at<cv::Vec3b>(i, j)[1] = 0;
					// src_color_image.at<cv::Vec3b>(i, j)[2] = 0;
				}
			}
		}
		imshow("道路-Road_mask", Road_mask);

		/* ================================================ 道路边缘 霍夫直线拟合 ================================================= */

		//-------------------------- 霍夫 线段拟合
		vector<cv::Vec4i> Road_Hough_lines;
		HoughLinesP(Road_mask, Road_Hough_lines, 1, CV_PI / 180, AlphaValuesSlider, 30, 30);

		for (size_t i = 0; i < Road_Hough_lines.size(); i++)
		{
			Vec4i Road_linex = Road_Hough_lines[i];
			// int dx = Road_linex[2] - Road_linex[0];
			// int dy = Road_linex[3] - Road_linex[1];
			// double angle = atan2(double(dy), dx) * 180 / CV_PI;

			// line(aligned_image_color, cv::Point(Road_linex[0], Road_linex[1]), cv::Point(Road_linex[2], Road_linex[3]), cv::Scalar(255, 0, 0), 1);
		}
		//-------------------------- 霍夫 线段拟合

		// 车道线分左右？
		// https://zhuanlan.zhihu.com/p/60891432
		// https://zhuanlan.zhihu.com/p/630083399

		// 【聚类】------------------------- 聚类 [天蓝色线]
		// 对之前获取到的直线（霍夫）进行聚类，分出车道线的左右

		// ======================== 【step 1】线段聚类 & 【step 2】找出每组线段的代表直线

		// 【距离聚类阈值，根据实际情况调整】
		float cluster_distanceThreshold = 30.0;

		// 对线段进行聚类
		std::vector<std::vector<cv::Vec4i>> clusters = clusterLines(Road_Hough_lines, cluster_distanceThreshold);
		vector<cv::Vec4f> represent_line; // 每组线段的代表线段

		float represent_line_slope = 0.0, represent_line_slope_sum = 0.0; // 斜率和
		float represent_line_midpoint_x = 0.0, represent_line_midpoint_x_sum = 0.0;
		float represent_line_midpoint_y = 0.0, represent_line_midpoint_y_sum = 0.0;
		float represent_line_b = 0.0, represent_line_b_sum = 0.0;
		float represent_line_angle = 0.0;

		// 可以根据需要输出聚类结果或进行其他操作
		for (const auto& cluster : clusters)
		{
			represent_line_slope_sum = 0.0;
			represent_line_b_sum = 0.0;
			represent_line_midpoint_x_sum = 0.0;
			represent_line_midpoint_y_sum = 0.0;

			for (const auto& line : cluster)
			{
				// cluster里放了一堆直线，一直循环这个cluster
				// 【属于哪个聚类的类里的直线前二多，就找哪个】
				// 中点
				represent_line_midpoint_x = (line[0] + line[2]) * 0.5;
				represent_line_midpoint_y = (line[1] + line[3]) * 0.5;
				represent_line_midpoint_x_sum = represent_line_midpoint_x_sum + represent_line_midpoint_x;
				represent_line_midpoint_y_sum = represent_line_midpoint_y_sum + represent_line_midpoint_y;

				// 斜率 & b
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

				// cv::line(aligned_image_color, cv::Point(line[0], line[1]), cv::Point(line[2], line[3]), cv::Scalar(255, 156, 28), 1);
			}

			// 每个cluster的代表线
			represent_line_slope = (float)represent_line_slope_sum / cluster.size();
			represent_line_midpoint_x = (float)represent_line_midpoint_x_sum / cluster.size();
			represent_line_midpoint_y = (float)represent_line_midpoint_y_sum / cluster.size();

			represent_line_b = (float)represent_line_b_sum / cluster.size();
			represent_line_angle = atan(represent_line_slope) * 180.0 / 3.1415926;

			cv::Vec4f line(1, represent_line_slope, represent_line_midpoint_x, represent_line_midpoint_y);
			// if (represent_line_slope > 10 && represent_line_slope < 30)
			{
				represent_line.push_back(line);
			}

			// cout << "【代表线段】线段中点为：（" << represent_line_midpoint_x << "，" << represent_line_midpoint_y << "），";
			// cout << "线段与水平线的角度为：" << represent_line_angle << endl;

			draw_line_k(src_color_image, represent_line_slope, Point(represent_line_midpoint_x, represent_line_midpoint_y),
				cv::Scalar(28, 156, 255), 1);
		}

		// ======================== 【step 1】线段聚类 & 【step 2】找出每组线段的代表直线


		// -------------------------【step 2】一对线段找角平分线

		// 不确定的地方：聚类的类别可能不只两类，这里选的是0和1，可能会出错？

		// 先确定斜率，再找一个点即可画出角平分线。判断角平分线与水平线的角度，在90度左右即可。
		// 假设聚类之后的代表直线只有两条

		if (clusters.size() >= 2)
		{
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

			float Road_x_send = 0, y = 240;
			Road_x_send = (y - bisector_b) / bisector_k_average;

			draw_line_k(src_color_image, bisector_k_average, cv::Point(bisector_X_average, bisector_Y_average),
				cv::Scalar(255, 255, 255), 1);

			// cout << "【角平分】角平分线 y = " << bisector_k_average << "x + " << bisector_b ;
			// cout << "，线段与水平线的角度为：" << bisector_angle_mid << endl;

			// -------------------------【step 3】一对直线找角平分线

			// 【3 聚类】------------------------- 聚类 [天蓝色线]

			/* ================================================ 道路边缘 霍夫直线拟合 ================================================= */


			// 发送数据
			// bisector_angle_mid ：道路中线与水平线的夹角度数
			// Road_x_send - 320：道路中点与图像中心的距离
			cam_data[0] = bisector_angle_mid;
			cam_data[1] = Road_x_send - 320;

		}

		else
		{
			cout << "聚类不足两类！" << endl;
		}

		sprintf_s(fps_char, "fps: %f", fps);
		cv::putText(src_color_image, (string)fps_char, cv::Point(20, 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 55), 2, 5);
		imshow("道路-深彩对齐", src_color_image);

		////-------------------------- 点击获取深度
		// imshow("深彩对齐", aligned_image_color);
		// setMouseCallback("深彩对齐", on_Mouse_Distance, &aligned_image_color);
		// if (buttondown == 2)
		//{
		//	// 注意：虽然显示的图像上是灰色点，但是点击的时候还是会有深度数据
		//	dist_to_center = aligned_depth_frame.get_distance(px, py);
		//	cout << "The object is " << dist_to_center << " meters away "<<endl;
		//	buttondown = 0;
		// }
		////-------------------------- 点击获取深度

		/* #################################################### 道路边缘 结束 #################################################### */

		/* #################################################### 台阶 开始 #################################################### */

		//// 雷达收到的点		received_x		received_y
		//// 测试的最优点		best_x			best_y
		//// 根据雷达和测试的结果对上面的参数进行修改
		//// 误差范围			+-
		// int received_x, received_y;
		// int best_x, best_y;
		// float distance_footstep = 0.0; // 距离台阶的距离

		//// 如果站位准确，发送数据 【自身无旋转？】
		// if (received_x - best_x > -50 && received_x - best_x<50 && received_y - best_y>-50 && received_y - best_y < 50)
		//{
		//	for (int x_i = 0; x_i < 5; x_i++)
		//	{
		//		for (int y_i = 0; y_i < 5; y_i++)
		//		{
		//			distance_footstep = aligned_depth_frame.get_distance(320 - x_i, 240 - y_i) + distance_footstep; // 中点+-5的范围
		//			distance_footstep = aligned_depth_frame.get_distance(320 + x_i, 240 + y_i) + distance_footstep;
		//		}
		//	}
		//	distance_footstep = distance_footstep / 25;

		//	// 发送数据
		//	// 距离中点的深度
		// cmda.data.push_back(distance_footstep);

		//}
		//
		//// 如果站位不准确，根据实际测试结果发送数据调整站姿【能调整吗？自身有旋转？】
		////
		//
		//

		/* #################################################### 台阶 结束 #################################################### */


		/* #################################################### 双木桥 开始 #################################################### */

		//
		// 深度为？厘米的点，RANSAC拟合直线，求垂线，角度90度左右；（找交点，交点在画面中线附近）
		//

		// 先雷达判断 然后进判断，开始发数
		// or 一直发数 需要的时候 机器人自己判断，自己用

		Mat Bridge_depth_image = My_D435if_Camera.readDepthCamera();
		Mat Bridge_depth_image_colored;
		float diatance = 0;


		convertScaleAbs(Bridge_depth_image, Bridge_depth_image, 1.0, 0.0);				// 黑白
		applyColorMap(Bridge_depth_image, Bridge_depth_image_colored, COLORMAP_AUTUMN); // 彩色

		// 深度数据存储二维数组
		int height = Bridge_depth_image_colored.rows; // y
		int width = Bridge_depth_image_colored.cols;  // x
		vector<vector<float>> Bridge_depthArray(height, vector<float>(width, -0.5));
		// eg 图片是宽200，高100，上面语句生成100行，200列的数组。[100][200]
		// 内部的vector<float>表示每一行的向量，它的大小为width，且初始值为0。



		// 将深度图像数据存储到二维数组中
		for (int i = 0;i<Bridge_depth_image_colored.rows ; i++) //height
		{
			for (int j = 0; j < Bridge_depth_image_colored.cols; j++) //width
			{
				if (i > 100 && i < 400)
				{
					// 获取深度值
					diatance = aligned_depth_frame.get_distance(j, i);
					// 存到二维数组中
					Bridge_depthArray[i][j] = diatance;
				}
				else
				{
					//Bridge_depth_image_colored.at<uchar>(i, j) = 255;
					Bridge_depth_image_colored.at<cv::Vec3b>(i, j)[0] = 0;
					Bridge_depth_image_colored.at<cv::Vec3b>(i, j)[1] = 0;
					Bridge_depth_image_colored.at<cv::Vec3b>(i, j)[2] = 0;
				}
			}
		}

		float i_sum = 0, i_mid;
		int count = 0;

		// 检测在 ? 左右的数据点 1 cm ：0.01
		vector<Point2f> Bridge_specfic_Points;



		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				int depthDiff = Bridge_depthArray[i][j] * 100;
				// 640 (x->j) * 480 (y->i)
				if (depthDiff >= 90 && depthDiff <= 95)
				{
					i_sum = i + i_sum;
					count = count + 1;
				}
			}
		}

		i_mid = i_sum / count;

		int flag = 0;

		for (int i = 1; i < height - 1; i++)
		{
			for (int j = 1; j < width - 1; j++)
			{
				int depthDiff = Bridge_depthArray[i][j] * 100;
				// 640 (x->j) * 480 (y->i)
				if (i > 100 && j < 540 && depthDiff >= 20 && depthDiff <= 30)
				{
					if (flag == 0)
					{
						cv::circle(Bridge_depth_image_colored, Point(j, i), 1, Scalar(255, 255, 255));
						Bridge_specfic_Points.push_back(Point(j, i));
						flag = 1;
					}
					else
					{
						flag = 0;
					}
				}
			}
		}

		// 已知斜率 k ，已知线上一点 x1,y1 画直线（从左到右）
		// 需要算出直线上另外两个点
		// P1（ 0 ，x1 + k*(0-x1) ）
		// P2（640，x1 + k*(640-x1) ）
		// cv::line(img, P1, P2, cv::Scalar(255, 0, 0), 1)

		clock_t time11 = clock();
		Vec4f Bridge_line_Ransac;
		fitLineRansac(Bridge_specfic_Points, Bridge_line_Ransac, 300, 1, -10, 10); // 调用fitLineRansac函数拟合一条直线

		// 100 30
		// 300 100

		clock_t time22 = clock();
		cout << "RANSAC用时" << time22 - time11 << endl;

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

		double vertical_line_k = -Bridge_line_Ransac[0] / Bridge_line_Ransac[1]; // 垂线斜率
		draw_line_k(Bridge_depth_image_colored, vertical_line_k, p3, cv::Scalar(255, 0, 0), 2);

		sprintf_s(fps_char, "fps: %f", fps);
		cv::putText(Bridge_depth_image_colored, (string)fps_char, cv::Point(20, 20), cv::FONT_HERSHEY_PLAIN, 2, cv::Scalar(0, 255, 55), 2, 5);
		imshow("双木桥-depth", Bridge_depth_image_colored);


		// 【发数据】
		float Bridge_angle_mid = atan(vertical_line_k) * 180.0 / 3.1415926;
		cam_data[4] = Bridge_angle_mid;
		cam_data[5] = p3.x - 320;

		/* #################################################### 双木桥 结束 #################################################### */

		//pub.publish(cam_data);
		//r.sleep();
		//ros::spinOnce();


		aligned_image_color.release();
		src_color_image.release();
		Road_mask.release();
		Bridge_depth_image.release();
		Bridge_depth_image_colored.release();
		contours.clear();
		hierarchy.clear();
		clusters.clear();
		represent_line.clear();
		Bridge_depthArray.clear();
		Bridge_specfic_Points.clear();

		if (waitKey(1) == 27)
		{
			destroyAllWindows();
			break;
		}
	}
	return EXIT_SUCCESS;
}

catch (const rs2::error& e)
{
	cout << "RealSense error calling" << e.get_failed_function() << "(" << e.get_failed_args() << "):\n"
		<< e.what() << endl;
	return EXIT_FAILURE;
}

catch (const exception& e)
{
	cout << e.what() << endl;
	return EXIT_FAILURE;
}

bool profile_changed(const std::vector<rs2::stream_profile>& current, const std::vector<rs2::stream_profile>& prev)
{
	for (auto&& sp : prev)
	{
		// If previous profile is in current (maybe just added another)
		// 设备换了
		auto itr = std::find_if(std::begin(current), std::end(current), [&sp](const rs2::stream_profile& current_sp)
			{ return sp.unique_id() == current_sp.unique_id(); });
		if (itr == std::end(current)) // If it previous stream wasn't found in current
		{
			return true;
		}
	}
	return false;
}

void remove_background(rs2::video_frame& other_frame, const rs2::depth_frame& depth_frame, float depth_scale, float clipping_dist)
{
	// 获取指向两个帧的原始缓冲区的指针，以便我们可以更改彩色图像（而不是创建新的缓冲区）。
	const uint16_t* p_depth_frame = reinterpret_cast<const uint16_t*>(depth_frame.get_data());
	uint8_t* p_other_frame = reinterpret_cast<uint8_t*>(const_cast<void*>(other_frame.get_data()));

	int width = other_frame.get_width();
	int height = other_frame.get_height();
	int other_bpp = other_frame.get_bytes_per_pixel();

#pragma omp parallel for schedule(dynamic)
	// Using OpenMP to try to parallelise the loop
	// 遍历帧的每个像素
	for (int y = 0; y < height; y++)
	{
		auto depth_pixel_index = y * width;
		for (int x = 0; x < width; x++, ++depth_pixel_index)
		{
			// Get the depth value of the current pixel
			// 计算该像素的深度距离
			auto pixels_distance = depth_scale * p_depth_frame[depth_pixel_index];

			// Check if the depth value is invalid (<=0) or greater than the threashold
			// 如果该距离无效或比用户请求的最大距离更远，那么从生成的彩色图像中去除该像素。
			if (pixels_distance <= 0.f || pixels_distance > clipping_dist)
			{
				// 用灰色绘制该像素不在范围内的像素
				//  Calculate the offset in other frame's buffer to current pixel
				auto offset = depth_pixel_index * other_bpp;

				// Set pixel to "background" color (0x999999)
				std::memset(&p_other_frame[offset], 0x99, other_bpp);
			}
		}
	}
}

// 鼠标回调 hsv
void on_Mouse_HSV(int event, int x, int y, int flags, void* param)
{
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		px = x;
		py = y;
		buttondown = 1;
	}
}

// 鼠标回调 距离
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
		// exit(3);
	}

	// Filtering Lines wrt angle
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
			double l = pow((pow((y2 - y1), 2) + pow((x2 - x1), 2)), 0.5); // length of the line
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
			[](const std::vector<double>& a,
				const std::vector<double>& b)
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

// 距离聚类函数
std::vector<std::vector<cv::Vec4i>> clusterLines(std::vector<cv::Vec4i>& lines, double distanceThresh)
{
	std::vector<std::vector<cv::Vec4i>> clusters; // 存储聚类结果的容器

	for (const auto& line : lines) // line不可修改，用于只读取lines中内容
	{
		bool assignedToCluster = false;

		// 遍历已有的聚类
		for (auto& cluster : clusters)
		{
			// 计算线段的中心点坐标
			cv::Point2f lineCenter = cv::Point2f((line[0] + line[2]) / 2.0, (line[1] + line[3]) / 2.0);

			// 遍历聚类中的线段
			for (const auto& existingLine : cluster)
			{
				// 计算聚类中线段的中心点坐标
				cv::Point2f existingLineCenter = cv::Point2f((existingLine[0] + existingLine[2]) / 2.0, (existingLine[1] + existingLine[3]) / 2.0);

				// 计算线段之间的距离
				double distance = cv::norm(lineCenter - existingLineCenter);

				// 如果距离小于阈值，则将当前线段加入聚类
				if (distance <= distanceThresh)
				{
					cluster.push_back(line);
					assignedToCluster = true;
					break; // 终止内层循环
				}
			}

			if (assignedToCluster)
				break; // 终止外层循环
		}

		// 如果当前线段没有被分配到任何聚类中，则创建新的聚类
		if (!assignedToCluster)
		{
			clusters.push_back({ line });
		}
	}

	return clusters;
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

// RANSAC 拟合2D 直线
// 输入参数：
//			points--输入点集
//			iterations--迭代次数
//			sigma--数据和模型之间可接受的差值,车道线像素宽带一般为10左右
//               （Parameter use to compute the fitting score）
//			k_min/k_max--拟合的直线斜率的取值范围.
//                      考虑到左右车道线在图像中的斜率位于一定范围内，
//                       添加此参数，同时可以避免检测垂线和水平线

void fitLineRansac(const std::vector<cv::Point2f>& points, cv::Vec4f& line, int iterations = 1000, double sigma = 1.0, double k_min = -7.0, double k_max = 7.0)
{
	// prosac ><
	// lo-ransac
	// groupsac

	unsigned int n = points.size(); // 点集大小

	if (n < 2)
	{
		return; // 点集中的点小于2，退出
	}

	cv::RNG rng;			 // 随机数生成器
	double bestScore = -1.0; // 最佳匹配得分，初始化为-1.0

	int i1 = 0, i2 = 0;

	for (int k = 0; k < iterations; k++)
	{
		// 随机选择两个不同的点的索引
		i1 = rng(n);
		i2 = rng(n);
		const cv::Point2f& p1 = points[i1];
		const cv::Point2f& p2 = points[i2];

		cv::Point2f dp = p2 - p1; // 直线的方向向量
		double score = 0;		  // 当前模型的得分
		dp = dp * 1.0 / norm(dp); // 归一化方向向量

		if (dp.y / dp.x <= k_max && dp.y / dp.x >= k_min)
		{
			// 检查斜率是否在给定范围内
			for (int i = 0; i < n; i++)
			{
				// 计算点集内所有点到现在选择的直线的向量的距离
				cv::Point2f v = points[i] - p1;
				double d = (double)v.y * dp.x - (double)v.x * dp.y; // 向量a与b叉乘/向量b的模.||b||=1./norm(dp)
				if (fabs(d) < sigma)
				{
					// 使用误差定义的方式计算得分
					score = score + 1;
				}
			}
		}
		if (score > bestScore)
		{
			// 如果当前得分比最佳得分大，则更新最佳得分和直线参数
			line = cv::Vec4f(dp.x, dp.y, p1.x, p1.y);
			bestScore = score;
		}
	}
}

void draw_line_k(Mat image, float k, cv::Point point, const cv::Scalar& color, int thickness)
{
	cv::Point p1, p2;
	p1.x = 0;
	p1.y = point.y + k * (p1.x - point.x);
	p2.x = 640;
	p2.y = point.y + k * (p2.x - point.x);
	cv::line(image, p1, p2, color, thickness);
};
