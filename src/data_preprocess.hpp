/*
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef DATA_PREPROCESS_HPP
#define DATA_PREPROCESS_HPP

#include <Eigen/Core>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl_conversions/pcl_conversions.h>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp/serialization.hpp>
#include <rosbag2_cpp/converter_options.hpp>
#include <rosbag2_cpp/readers/sequential_reader.hpp>
#include <rosbag2_storage/storage_filter.hpp>
#include <rosbag2_storage/storage_options.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#ifdef FAST_CALIB_LIVOX_ENABLED
#include <livox_ros_driver2/msg/custom_msg.hpp>
#endif
#include <fstream>
#include "common_lib.h"

using namespace std;

enum class LiDARType : int {
    Unknown = 0,
    Solid   = 1,   // 固态（如 Livox）
    Mech    = 2,   // 机械式多线
    Grid    = 3    // 栅格占据法（与扫描模式无关，适合玫瑰线/MEMS 雷达）
};

class DataPreprocess
{
public:
    // 改成带线号的点云
    pcl::PointCloud<Common::Point>::Ptr cloud_input_;
    cv::Mat img_input_;
    LiDARType lidar_type_{LiDARType::Unknown};
    LiDARType lidarType() const { return lidar_type_; }

    DataPreprocess(const rclcpp::Node::SharedPtr &node, Params &params)
        : cloud_input_(new pcl::PointCloud<Common::Point>)
    {
        // 图像：rtsp:// 地址走实时抓帧；以 '/' 开头且磁盘上不存在的路径当作
        // ROS 图像 topic 实时订阅一帧；否则按文件读取
        if (params.image_path.rfind("rtsp://", 0) == 0)
        {
            captureImageFromRtsp(params.image_path, params.rtsp_warmup_frames);
            saveFrameForInspection(params.output_path, "rtsp_frame.jpg");
        }
        else if (!params.image_path.empty() && params.image_path[0] == '/' &&
                 !std::ifstream(params.image_path).good())
        {
            captureImageFromTopic(node, params.image_path);
            saveFrameForInspection(params.output_path, "topic_frame.png");
        }
        else
        {
            img_input_ = cv::imread(params.image_path, cv::IMREAD_UNCHANGED);
            if (img_input_.empty())
            {
                RCLCPP_ERROR(rclcpp::get_logger("fast_calib"), "Loading the image %s failed", params.image_path.c_str());
                return;
            }
        }

        // 点云：bag_path 为空时走实时订阅，否则按 bag 读取
        if (params.bag_path.empty())
        {
            captureLidarLive(node, params.lidar_topic, params.live_capture_seconds);
        }
        else
        {
            loadLidarFromBag(params.bag_path, params.lidar_topic);
        }
    }

private:
    void saveFrameForInspection(const std::string &output_path, const std::string &filename)
    {
        if (img_input_.empty()) return;
        std::string out = output_path;
        if (!out.empty() && out.back() != '/') out += '/';
        const std::string frame_path = out + filename;
        if (cv::imwrite(frame_path, img_input_))
            RCLCPP_INFO(rclcpp::get_logger("fast_calib"),
                        "Saved the captured frame to %s for inspection.", frame_path.c_str());
    }

    // 把 ROS Image 消息转成 BGR 的 cv::Mat（只处理本项目会遇到的编码，避免引入 cv_bridge 依赖）
    static cv::Mat imageMsgToBgr(const sensor_msgs::msg::Image &msg)
    {
        const int h = static_cast<int>(msg.height);
        const int w = static_cast<int>(msg.width);
        cv::Mat bgr;
        if (msg.encoding == "bgr8" || msg.encoding == "rgb8")
        {
            const cv::Mat wrap(h, w, CV_8UC3, const_cast<uint8_t *>(msg.data.data()), msg.step);
            if (msg.encoding == "rgb8")
                cv::cvtColor(wrap, bgr, cv::COLOR_RGB2BGR);
            else
                bgr = wrap.clone();
        }
        else if (msg.encoding == "mono8")
        {
            const cv::Mat wrap(h, w, CV_8UC1, const_cast<uint8_t *>(msg.data.data()), msg.step);
            cv::cvtColor(wrap, bgr, cv::COLOR_GRAY2BGR);
        }
        else
        {
            RCLCPP_ERROR(rclcpp::get_logger("fast_calib"),
                         "Unsupported image encoding '%s' (expected rgb8/bgr8/mono8).", msg.encoding.c_str());
        }
        return bgr;
    }

    void captureImageFromTopic(const rclcpp::Node::SharedPtr &node, const std::string &topic,
                               double timeout_sec = 10.0)
    {
        RCLCPP_INFO(rclcpp::get_logger("fast_calib"),
                    "Waiting for one image on ROS topic %s (sensor-data QoS) ...", topic.c_str());

        auto sub = node->create_subscription<sensor_msgs::msg::Image>(
            topic, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::Image::SharedPtr msg)
            {
                if (!img_input_.empty()) return;
                img_input_ = imageMsgToBgr(*msg);
            });

        const rclcpp::Time start = node->now();
        rclcpp::Rate rate(50);
        while (rclcpp::ok() && img_input_.empty() && (node->now() - start).seconds() < timeout_sec)
        {
            rclcpp::spin_some(node);
            rate.sleep();
        }

        if (img_input_.empty())
            RCLCPP_ERROR(rclcpp::get_logger("fast_calib"),
                         "No image received on %s within %.1f s (if you meant an image file, check that the path exists).",
                         topic.c_str(), timeout_sec);
        else
            RCLCPP_INFO(rclcpp::get_logger("fast_calib"), "Captured a %dx%d frame from %s.",
                        img_input_.cols, img_input_.rows, topic.c_str());
    }

    void captureImageFromRtsp(const std::string &url, int warmup_frames)
    {
        // TCP transport avoids the RTP-over-UDP packet loss that leaves the HEVC
        // decoder without reference frames and produces flat gray output.
        setenv("OPENCV_FFMPEG_CAPTURE_OPTIONS", "rtsp_transport;tcp", 0);

        RCLCPP_INFO(rclcpp::get_logger("fast_calib"), "Connecting to RTSP camera stream %s ...", url.c_str());
        cv::VideoCapture cap(url, cv::CAP_FFMPEG);
        if (!cap.isOpened())
        {
            RCLCPP_ERROR(rclcpp::get_logger("fast_calib"), "Failed to connect to RTSP stream: %s", url.c_str());
            return;
        }

        cv::Mat frame;
        // Drain buffered/stale frames so the one we keep reflects the current scene.
        for (int i = 0; i < warmup_frames; ++i) cap.read(frame);

        // A frame decoded without its keyframe comes out as near-uniform gray;
        // keep reading (up to ~200 more frames) until one has real texture.
        const double kMinStdDev = 8.0;
        for (int i = 0; i < 200; ++i)
        {
            if (!frame.empty())
            {
                cv::Mat gray;
                cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
                cv::Scalar mean, stddev;
                cv::meanStdDev(gray, mean, stddev);
                if (stddev[0] > kMinStdDev) break;
                if (i == 0)
                    RCLCPP_WARN(rclcpp::get_logger("fast_calib"),
                                "Frame looks corrupted (flat gray, stddev %.1f) — waiting for a keyframe ...", stddev[0]);
            }
            if (!cap.read(frame)) break;
        }

        if (frame.empty())
        {
            RCLCPP_ERROR(rclcpp::get_logger("fast_calib"), "Failed to grab a frame from RTSP stream: %s", url.c_str());
            return;
        }
        img_input_ = frame.clone();
        RCLCPP_INFO(rclcpp::get_logger("fast_calib"), "Captured a %dx%d frame from %s.",
                    img_input_.cols, img_input_.rows, url.c_str());
    }

    void appendPointCloud2(const sensor_msgs::msg::PointCloud2 &pcl_msg)
    {
        // 优先判断是否有 ring 字段
        bool has_ring = false;
        for (const auto &f : pcl_msg.fields)
        {
            if (f.name == "ring") { has_ring = true; break; }
        }
        lidar_type_ = has_ring ? LiDARType::Mech : LiDARType::Solid;

        // 使用 iterator 安全读取
        sensor_msgs::PointCloud2ConstIterator<float> it_x(pcl_msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> it_y(pcl_msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> it_z(pcl_msg, "z");

        // ring 可能不存在：不存在时用 0xFFFF 表示未知
        std::unique_ptr<sensor_msgs::PointCloud2ConstIterator<std::uint16_t>> it_ring_ptr;
        if (has_ring)
        {
            it_ring_ptr.reset(new sensor_msgs::PointCloud2ConstIterator<std::uint16_t>(pcl_msg, "ring"));
        }

        const size_t n = static_cast<size_t>(pcl_msg.width) * pcl_msg.height;
        cloud_input_->reserve(cloud_input_->size() + n);

        for (size_t i = 0; i < n; ++i, ++it_x, ++it_y, ++it_z)
        {
            Common::Point p;
            p.x = *it_x;
            p.y = *it_y;
            p.z = *it_z;

            if (has_ring)
            {
                p.ring = **it_ring_ptr;
                ++(*it_ring_ptr);
            }
            else
            {
                p.ring = 0xFFFF;
            }

            cloud_input_->push_back(p);
        }
    }

    void captureLidarLive(const rclcpp::Node::SharedPtr &node, const std::string &lidar_topic, double duration_sec)
    {
        RCLCPP_INFO(rclcpp::get_logger("fast_calib"), "Subscribing to live LiDAR topic %s for %.1f s ...",
                    lidar_topic.c_str(), duration_sec);

        auto sub = node->create_subscription<sensor_msgs::msg::PointCloud2>(
            lidar_topic, rclcpp::SensorDataQoS(),
            [this](const sensor_msgs::msg::PointCloud2::SharedPtr msg) { appendPointCloud2(*msg); });

        const rclcpp::Time start = node->now();
        rclcpp::Rate rate(50);
        while (rclcpp::ok() && (node->now() - start).seconds() < duration_sec)
        {
            rclcpp::spin_some(node);
            rate.sleep();
        }

        RCLCPP_INFO(rclcpp::get_logger("fast_calib"), "Captured %zu points from live topic %s over %.1f s.",
                    cloud_input_->size(), lidar_topic.c_str(), duration_sec);
    }

    void loadLidarFromBag(const std::string &bag_path, const std::string &lidar_topic)
    {
        // 打开 ROS 2 bag（一个包含 metadata.yaml 的目录）
        rosbag2_storage::StorageOptions storage_options;
        storage_options.uri = bag_path;
        storage_options.storage_id = "sqlite3";

        rosbag2_cpp::ConverterOptions converter_options;
        converter_options.input_serialization_format = "cdr";
        converter_options.output_serialization_format = "cdr";

        rosbag2_cpp::readers::SequentialReader reader;
        try
        {
            reader.open(storage_options, converter_options);
        }
        catch (const std::exception &e)
        {
            RCLCPP_ERROR(rclcpp::get_logger("fast_calib"), "Loading the rosbag %s failed: %s", bag_path.c_str(), e.what());
            return;
        }
        RCLCPP_INFO(rclcpp::get_logger("fast_calib"), "Loading the rosbag %s", bag_path.c_str());

        // 查找雷达 topic 的消息类型
        std::string lidar_msg_type;
        for (const auto &topic_info : reader.get_all_topics_and_types())
        {
            if (topic_info.name == lidar_topic)
            {
                lidar_msg_type = topic_info.type;
                break;
            }
        }
        if (lidar_msg_type.empty())
        {
            RCLCPP_ERROR(rclcpp::get_logger("fast_calib"), "Topic %s not found in the rosbag.", lidar_topic.c_str());
            return;
        }

        rosbag2_storage::StorageFilter filter;
        filter.topics.push_back(lidar_topic);
        reader.set_filter(filter);

#ifdef FAST_CALIB_LIVOX_ENABLED
        rclcpp::Serialization<livox_ros_driver2::msg::CustomMsg> livox_serialization;
#endif
        rclcpp::Serialization<sensor_msgs::msg::PointCloud2> pc2_serialization;

        // 累计读取
        while (reader.has_next())
        {
            auto bag_message = reader.read_next();
            rclcpp::SerializedMessage serialized_msg(*bag_message->serialized_data);

            // 1) Livox 自定义消息（含 line 字段）
#ifdef FAST_CALIB_LIVOX_ENABLED
            if (lidar_msg_type == "livox_ros_driver2/msg/CustomMsg")
            {
                livox_ros_driver2::msg::CustomMsg livox_msg;
                livox_serialization.deserialize_message(&serialized_msg, &livox_msg);

                lidar_type_ = LiDARType::Solid;
                cloud_input_->reserve(cloud_input_->size() + livox_msg.point_num);
                for (uint32_t i = 0; i < livox_msg.point_num; ++i)
                {
                    Common::Point p;
                    p.x = livox_msg.points[i].x;
                    p.y = livox_msg.points[i].y;
                    p.z = livox_msg.points[i].z;
                    // Livox 的 CustomPoint 有 line 字段（uint8 / uint16 视版本而定）
                    p.ring = static_cast<std::uint16_t>(livox_msg.points[i].line);
                    cloud_input_->push_back(p);
                }
                continue;
            }
#endif

            // 2) 机械雷达 / 通用 PointCloud2
            if (lidar_msg_type == "sensor_msgs/msg/PointCloud2")
            {
                sensor_msgs::msg::PointCloud2 pcl_msg;
                pc2_serialization.deserialize_message(&serialized_msg, &pcl_msg);
                appendPointCloud2(pcl_msg);
                continue;
            }

            // 其他类型忽略
        }

        RCLCPP_INFO(rclcpp::get_logger("fast_calib"), "Loaded %zu points from the rosbag.", cloud_input_->size());
    }
};

typedef std::shared_ptr<DataPreprocess> DataPreprocessPtr;

#endif // DATA_PREPROCESS_HPP
