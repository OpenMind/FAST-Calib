/*
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#include "qr_detect.hpp"
#include "lidar_detect.hpp"
#include "data_preprocess.hpp"

int main(int argc, char **argv)
{
    rclcpp::init(argc, argv);
    auto node = std::make_shared<rclcpp::Node>("mono_qr_pattern");

    // 读取参数
    Params params = loadParameters(node.get());

    // 初始化 QR 检测和 LiDAR 检测
    QRDetectPtr qrDetectPtr;
    qrDetectPtr.reset(new QRDetect(node, params));

    LidarDetectPtr lidarDetectPtr;
    lidarDetectPtr.reset(new LidarDetect(node, params));

    DataPreprocessPtr dataPreprocessPtr;
    dataPreprocessPtr.reset(new DataPreprocess(node, params));

    // 读取图像和点云
    cv::Mat img_input = dataPreprocessPtr->img_input_;
    undistortFisheyeImage(img_input, params);  // no-op unless fisheye_enable is set
    pcl::PointCloud<Common::Point>::Ptr cloud_input = dataPreprocessPtr->cloud_input_;

    // 检测 QR 码
    PointCloud<PointXYZ>::Ptr qr_center_cloud(new PointCloud<PointXYZ>);
    qr_center_cloud->reserve(4);
    qrDetectPtr->detect_qr(img_input, qr_center_cloud);

    // 检测 LiDAR 数据
    PointCloud<PointXYZ>::Ptr lidar_center_cloud(new PointCloud<PointXYZ>);
    lidar_center_cloud->reserve(4);

    LiDARType lidar_type = dataPreprocessPtr->lidar_type_;
    if (params.lidar_type == "solid") lidar_type = LiDARType::Solid;
    else if (params.lidar_type == "mech") lidar_type = LiDARType::Mech;
    else if (params.lidar_type == "grid") lidar_type = LiDARType::Grid;

    switch (lidar_type)
    {
        case LiDARType::Solid:
            lidarDetectPtr->detect_solid_lidar(cloud_input, lidar_center_cloud);
            break;

        case LiDARType::Mech:
            lidarDetectPtr->detect_mech_lidar(cloud_input, lidar_center_cloud);
            break;

        case LiDARType::Grid:
            lidarDetectPtr->detect_grid_lidar(cloud_input, lidar_center_cloud);
            break;

        default:
            std::cerr << BOLDYELLOW
                    << "[Main] Unknown LiDAR type."
                    << RESET << std::endl;
            break;
    }

    // 保存中间点云，便于检测失败时离线分析
    {
        std::string outDir = params.output_path;
        if (!outDir.empty() && outDir.back() != '/') outDir += '/';
        auto dump = [&outDir, &node](const std::string &name, const auto &cloudPtr) {
            if (cloudPtr && !cloudPtr->empty())
                pcl::io::savePCDFileBinary(outDir + name, *cloudPtr);
        };
        dump("dbg_filtered.pcd", lidarDetectPtr->getFilteredCloud());
        dump("dbg_plane.pcd", lidarDetectPtr->getPlaneCloud());
        dump("dbg_edge.pcd", lidarDetectPtr->getEdgeCloud());
        dump("dbg_aligned_edge.pcd", lidarDetectPtr->getAlignedCloud());
        dump("dbg_circle_centers.pcd", lidarDetectPtr->getCenterZ0Cloud());
        RCLCPP_INFO(node->get_logger(), "Saved intermediate debug clouds (dbg_*.pcd) to %s", outDir.c_str());
    }

    // 对 QR 和 LiDAR 检测到的圆心进行排序
    PointCloud<PointXYZ>::Ptr qr_centers(new PointCloud<PointXYZ>);
    PointCloud<PointXYZ>::Ptr lidar_centers(new PointCloud<PointXYZ>);
    sortPatternCenters(qr_center_cloud, qr_centers, "camera");
    sortPatternCenters(lidar_center_cloud, lidar_centers, "lidar", params.lidar_frame);

    // 计算外参。雷达与相机之间可能存在 90°/180° 相对滚转（如侧装雷达），
    // 角度排序无法保证两侧起始角点一致，因此穷举 4 个循环偏移 × 2 个方向
    // 共 8 种对应关系，取 RMSE 最小者。
    Eigen::Matrix4f transformation = Eigen::Matrix4f::Identity();
    pcl::registration::TransformationEstimationSVD<pcl::PointXYZ, pcl::PointXYZ> svd;
    double best_rmse = -1.0;
    if (lidar_centers->size() == 4 && qr_centers->size() == 4)
    {
        PointCloud<PointXYZ>::Ptr best_order(new PointCloud<PointXYZ>);
        for (int rev = 0; rev < 2; ++rev)
        {
            for (int shift = 0; shift < 4; ++shift)
            {
                PointCloud<PointXYZ>::Ptr cand(new PointCloud<PointXYZ>);
                for (int k = 0; k < 4; ++k)
                {
                    int idx = rev ? (shift + 4 - k) % 4 : (shift + k) % 4;
                    cand->push_back(lidar_centers->points[idx]);
                }
                Eigen::Matrix4f T;
                svd.estimateRigidTransformation(*cand, *qr_centers, T);

                pcl::PointCloud<pcl::PointXYZ>::Ptr aligned(new pcl::PointCloud<pcl::PointXYZ>);
                alignPointCloud(cand, aligned, T);
                double rmse = computeRMSE(qr_centers, aligned);
                if (rmse >= 0 && (best_rmse < 0 || rmse < best_rmse))
                {
                    best_rmse = rmse;
                    transformation = T;
                    *best_order = *cand;
                }
            }
        }
        if (!best_order->empty()) *lidar_centers = *best_order;
    }
    else
    {
        svd.estimateRigidTransformation(*lidar_centers, *qr_centers, transformation);
    }

    // 保存中间结果：已按最优对应关系排列的 LiDAR 圆心和 QR 圆心
    saveTargetHoleCenters(lidar_centers, qr_centers, params);

    // 将 LiDAR 点云转换到 QR 码坐标系
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_lidar_centers(new pcl::PointCloud<pcl::PointXYZ>);
    aligned_lidar_centers->reserve(lidar_centers->size());
    alignPointCloud(lidar_centers, aligned_lidar_centers, transformation);

    double rmse = computeRMSE(qr_centers, aligned_lidar_centers);
    if (rmse > 0)
    {
      std::cout << BOLDYELLOW << "[Result] RMSE: " << BOLDRED << std::fixed << std::setprecision(4)
      << rmse << " m" << RESET << std::endl;
    }

    std::cout << BOLDYELLOW << "[Result] Single-scene calibration: extrinsic parameters T_cam_lidar = " << RESET << std::endl;
    std::cout << BOLDCYAN << std::fixed << std::setprecision(6) << transformation << RESET << std::endl;

    pcl::PointCloud<pcl::PointXYZRGB>::Ptr colored_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
    projectPointCloudToImage(cloud_input, transformation, qrDetectPtr->cameraMatrix_, qrDetectPtr->distCoeffs_, img_input, colored_cloud);

    saveCalibrationResults(params, transformation, colored_cloud, qrDetectPtr->imageCopy_);

    auto colored_cloud_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("colored_cloud", 1);
    auto aligned_lidar_centers_pub = node->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_lidar_centers", 1);

    // 主循环
    rclcpp::Rate rate(1);
    while (rclcpp::ok())
    {
      if (DEBUG)
      {
        // 发布 QR 检测结果
        sensor_msgs::msg::PointCloud2 qr_centers_msg;
        pcl::toROSMsg(*qr_centers, qr_centers_msg);
        qr_centers_msg.header.stamp = node->now();
        qr_centers_msg.header.frame_id = "map";
        qrDetectPtr->qr_pub_->publish(qr_centers_msg);

        // 发布 LiDAR 检测结果
        sensor_msgs::msg::PointCloud2 lidar_centers_msg;
        pcl::toROSMsg(*lidar_centers, lidar_centers_msg);
        lidar_centers_msg.header = qr_centers_msg.header;
        lidarDetectPtr->center_pub_->publish(lidar_centers_msg);

        // 发布中间结果
        sensor_msgs::msg::PointCloud2 filtered_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getFilteredCloud(), filtered_cloud_msg);
        filtered_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->filtered_pub_->publish(filtered_cloud_msg);

        sensor_msgs::msg::PointCloud2 plane_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getPlaneCloud(), plane_cloud_msg);
        plane_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->plane_pub_->publish(plane_cloud_msg);

        sensor_msgs::msg::PointCloud2 aligned_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getAlignedCloud(), aligned_cloud_msg);
        aligned_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->aligned_pub_->publish(aligned_cloud_msg);

        sensor_msgs::msg::PointCloud2 edge_cloud_msg;
        pcl::toROSMsg(*lidarDetectPtr->getEdgeCloud(), edge_cloud_msg);
        edge_cloud_msg.header = qr_centers_msg.header;
        lidarDetectPtr->edge_pub_->publish(edge_cloud_msg);

        sensor_msgs::msg::PointCloud2 lidar_centers_z0_msg;
        pcl::toROSMsg(*lidarDetectPtr->getCenterZ0Cloud(), lidar_centers_z0_msg);
        lidar_centers_z0_msg.header = qr_centers_msg.header;
        lidarDetectPtr->center_z0_pub_->publish(lidar_centers_z0_msg);

        // 发布外参变换后的LiDAR点云
        sensor_msgs::msg::PointCloud2 aligned_lidar_centers_msg;
        pcl::toROSMsg(*aligned_lidar_centers, aligned_lidar_centers_msg);
        aligned_lidar_centers_msg.header = qr_centers_msg.header;
        aligned_lidar_centers_pub->publish(aligned_lidar_centers_msg);

        // 发布彩色点云
        sensor_msgs::msg::PointCloud2 colored_cloud_msg;
        pcl::toROSMsg(*colored_cloud, colored_cloud_msg);
        colored_cloud_msg.header = qr_centers_msg.header;
        colored_cloud_pub->publish(colored_cloud_msg);

        // cv::imshow("result", qrDetectPtr->imageCopy_);
      }
      // cv::waitKey(1);
      rclcpp::spin_some(node);
      rate.sleep();
    }

    rclcpp::shutdown();
    return 0;
}
