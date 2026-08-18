/* 
Developer: Chunran Zheng <zhengcr@connect.hku.hk>

This file is subject to the terms and conditions outlined in the 'LICENSE' file,
which is included as part of this source code package.
*/

#ifndef LIDAR_DETECT_HPP
#define LIDAR_DETECT_HPP
#define PCL_NO_PRECOMPILE

#include <sensor_msgs/msg/point_cloud2.hpp>
#include <Eigen/Dense>
#include <rclcpp/rclcpp.hpp>
#include <pcl/filters/voxel_grid.h>
#include "common_lib.h"

class LidarDetect
{
private:
    double x_min_, x_max_, y_min_, y_max_, z_min_, z_max_;
    double circle_radius_, delta_width_circles_, delta_height_circles_;
    rclcpp::Logger logger_;

    // 存储中间结果的点云
    pcl::PointCloud<Common::Point>::Ptr filtered_cloud_;
    pcl::PointCloud<Common::Point>::Ptr plane_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr aligned_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr edge_cloud_;
    pcl::PointCloud<pcl::PointXYZ>::Ptr center_z0_cloud_;

public:
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr filtered_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr plane_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr aligned_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr edge_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr center_z0_pub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr center_pub_;

    LidarDetect(const rclcpp::Node::SharedPtr &node, Params &params)
        : logger_(node->get_logger()),
          filtered_cloud_(new pcl::PointCloud<Common::Point>),
          plane_cloud_(new pcl::PointCloud<Common::Point>),
          aligned_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          edge_cloud_(new pcl::PointCloud<pcl::PointXYZ>),
          center_z0_cloud_(new pcl::PointCloud<pcl::PointXYZ>)
    {
        x_min_ = params.x_min;
        x_max_ = params.x_max;
        y_min_ = params.y_min;
        y_max_ = params.y_max;
        z_min_ = params.z_min;
        z_max_ = params.z_max;
        circle_radius_ = params.circle_radius;
        delta_width_circles_ = params.delta_width_circles;
        delta_height_circles_ = params.delta_height_circles;

        filtered_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("filtered_cloud", 1);
        plane_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("plane_cloud", 1);
        aligned_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("aligned_cloud", 1);
        edge_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("edge_cloud", 1);
        center_z0_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("center_z0_cloud", 10);
        center_pub_ = node->create_publisher<sensor_msgs::msg::PointCloud2>("center_cloud", 10);
    }

    void detect_mech_lidar(pcl::PointCloud<Common::Point>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // 1. X、Y、Z方向滤波
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<Common::Point> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);  // 设置X轴范围
        pass_x.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);  // 设置Y轴范围
        pass_y.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);  // 设置Z轴范围
        pass_z.filter(*filtered_cloud_);

        RCLCPP_INFO(logger_, "Depth filtered cloud size: %zu", filtered_cloud_->size());

        // 2. 拟合平面，提取法向量
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.01);  // 平面分割阈值
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);
    
        pcl::ExtractIndices<Common::Point> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        RCLCPP_INFO(logger_, "Plane cloud size: %zu", plane_cloud_->size());
    
        // 3. 根据每条ring相邻点距离提取边缘点
        edge_cloud_->reserve(filtered_cloud_->size());

        // 先按 ring 分组
        std::unordered_map<unsigned int, std::vector<int>> ring2indices;
        ring2indices.reserve(64);

        for (int i = 0; i < static_cast<int>(filtered_cloud_->size()); ++i)
        {
            const auto &pt = filtered_cloud_->points[i];
            ring2indices[pt.ring].push_back(i);
        }

        const auto &c = plane_coefficients->values; // [a, b, c, d]
        Eigen::Vector3d n(c[0], c[1], c[2]);
        double norm_n = n.norm();
        Eigen::Vector3d normal = n / norm_n;

        // 在每条 ring 内，用相邻点距离检测跳变点作为边缘点
        const double neighbor_gap_threshold = 0.10;  // 邻近点距离阈值
        const int    min_points_per_ring    = 10;    // 太短的 ring 不处理

        for (auto &kv : ring2indices)
        {
            auto &idx_vec = kv.second;
            if (static_cast<int>(idx_vec.size()) < min_points_per_ring) continue;

            for (size_t k = 1; k + 1 < idx_vec.size(); ++k)
            {
                const auto &p_prev = filtered_cloud_->points[idx_vec[k - 1]];
                const auto &p_cur  = filtered_cloud_->points[idx_vec[k]];
                const auto &p_next = filtered_cloud_->points[idx_vec[k + 1]];

                // 只保留落在拟合平面附近的点
                double dist_plane = std::fabs(c[0]*p_cur.x + c[1]*p_cur.y + c[2]*p_cur.z + c[3]) / norm_n;
                if (dist_plane >= 0.03) continue;

                // cur 与 prev 的距离
                double dx1 = static_cast<double>(p_cur.x) - static_cast<double>(p_prev.x);
                double dy1 = static_cast<double>(p_cur.y) - static_cast<double>(p_prev.y);
                double dz1 = static_cast<double>(p_cur.z) - static_cast<double>(p_prev.z);
                double dist_prev = std::sqrt(dx1 * dx1 + dy1 * dy1 + dz1 * dz1);

                // cur 与 next 的距离
                double dx2 = static_cast<double>(p_cur.x) - static_cast<double>(p_next.x);
                double dy2 = static_cast<double>(p_cur.y) - static_cast<double>(p_next.y);
                double dz2 = static_cast<double>(p_cur.z) - static_cast<double>(p_next.z);
                double dist_next = std::sqrt(dx2 * dx2 + dy2 * dy2 + dz2 * dz2);

                // 只要和前一个或后一个之间有一个距离超过阈值，就认为当前点是边缘点
                if (dist_prev > neighbor_gap_threshold || dist_next > neighbor_gap_threshold)
                {
                    edge_cloud_->push_back(pcl::PointXYZ(p_cur.x, p_cur.y, p_cur.z));
                }
            }
        }

        RCLCPP_INFO(logger_, "Extracted %zu edge points (mechanical LiDAR by neighbor distance).", edge_cloud_->size());

        // 4. 将边缘点对齐到 Z=0 平面
        aligned_cloud_->reserve(edge_cloud_->size());
        Eigen::Vector3d z_axis(0, 0, 1);
        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(normal.dot(z_axis));

        Eigen::AngleAxisd rotation(angle, axis);
        Eigen::Matrix3d R_align = rotation.toRotationMatrix();

        float average_z = 0.0;
        int cnt = 0;
        for (const auto& pt : *edge_cloud_) {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R_align * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= cnt;

        // 5. 在对齐后的点云中检测圆形，提取圆心
        
        // 拷贝一份工作点云，后面要不停“删掉已拟合的圆”
        pcl::PointCloud<pcl::PointXYZ>::Ptr xy_cloud(new pcl::PointCloud<pcl::PointXYZ>(*aligned_cloud_));

        RCLCPP_INFO(logger_, "[LiDAR] Start circle detection, initial cloud size = %zu", xy_cloud->points.size());

        // 在对齐后的平面上，用 RANSAC 反复检测 2D 圆
        pcl::SACSegmentation<pcl::PointXYZ> circle_segmentation;
        circle_segmentation.setModelType(pcl::SACMODEL_CIRCLE2D);
        circle_segmentation.setMethodType(pcl::SAC_RANSAC);
        circle_segmentation.setDistanceThreshold(0.02);
        circle_segmentation.setOptimizeCoefficients(true);
        circle_segmentation.setMaxIterations(1000);
        circle_segmentation.setRadiusLimits(circle_radius_ - 0.03, circle_radius_ + 0.03);

        pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
        pcl::ExtractIndices<pcl::PointXYZ> extract2;

        // 不停在剩余点云中找圆
        while (xy_cloud->points.size() > 3)
        {
            RCLCPP_INFO(logger_, "[LiDAR] RANSAC on cloud of size %lu", xy_cloud->points.size());

            circle_segmentation.setInputCloud(xy_cloud);
            circle_segmentation.segment(*inliers, *coefficients);

            // 没有 inliers，说明没有可用的圆，结束
            if (inliers->indices.empty())
            {
                RCLCPP_INFO(logger_, "[LiDAR] No more circles can be found, stop.");
                break;
            }

            // 内点太少就认为是噪声，直接结束
            if ((int)inliers->indices.size() < 5)
            {
                RCLCPP_INFO(logger_, "[LiDAR] Found circle but inliers too few (%zu < 3), stop.",
                            inliers->indices.size());
                break;
            }

            // RCLCPP_INFO(logger_, "[LiDAR] Circle found: inliers = %zu, coeffs size = %zu",
            //         inliers->indices.size(), coefficients->values.size());
            // 对 Circle2D 而言，coeffs 通常是 [xc, yc, r]
            // if (coefficients->values.size() >= 3)
            // {
            //     RCLCPP_INFO(logger_, "[LiDAR]   center = (%.4f, %.4f), r = %.4f",
            //             coefficients->values[0],
            //             coefficients->values[1],
            //             coefficients->values[2]);
            // }
            
            // 记录当前这个圆
            pcl::PointXYZ center_point;
            center_point.x = coefficients->values[0];
            center_point.y = coefficients->values[1];
            center_point.z = 0;
            center_z0_cloud_->push_back(center_point);

            // 把当前圆的 inliers 从点云中移除，继续在剩余点中找
            extract2.setInputCloud(xy_cloud);
            extract2.setIndices(inliers);
            extract2.setNegative(true);  // 保留非 inliers（即剩余点）
            pcl::PointCloud<pcl::PointXYZ>::Ptr remaining(new pcl::PointCloud<pcl::PointXYZ>);
            extract2.filter(*remaining);
            xy_cloud.swap(remaining);

            // 清空 inliers，避免下一轮残留
            inliers->indices.clear();
        }

        // 6. Geometric consistency check
        std::vector<std::vector<int>> groups;
        comb(center_z0_cloud_->size(), TARGET_NUM_CIRCLES, groups);
        double groups_scores[groups.size()];  // -1: invalid; 0-1 normalized score
        for (int i = 0; i < groups.size(); ++i) 
        {
            std::vector<pcl::PointXYZ> candidates;
            // Build candidates set
            for (int j = 0; j < groups[i].size(); ++j) 
            {
                pcl::PointXYZ center;
                center.x = center_z0_cloud_->at(groups[i][j]).x;
                center.y = center_z0_cloud_->at(groups[i][j]).y;
                center.z = center_z0_cloud_->at(groups[i][j]).z;
                candidates.push_back(center);
            }

            // Compute candidates score
            Square square_candidate(candidates, delta_width_circles_, delta_height_circles_);
            groups_scores[i] = square_candidate.is_valid() ? 1.0 : -1;  // -1 when it's not valid, 1 otherwise
        }

        int best_candidate_idx = -1;
        double best_candidate_score = -1;
        for (int i = 0; i < groups.size(); ++i) 
        {
            if (best_candidate_score == 1 && groups_scores[i] == 1) 
            {
                // Exit 4: Several candidates fit target's geometry
                RCLCPP_ERROR(logger_, 
                    "[LiDAR] More than one set of candidates fit target's geometry. "
                    "Please, make sure your parameters are well set. Exiting callback");
                return;
            }
            if (groups_scores[i] > best_candidate_score) 
            {
                best_candidate_score = groups_scores[i];
                best_candidate_idx = i;
            }
        }
        if (best_candidate_idx == -1) 
        {
            // Exit 5: No candidates fit target's geometry
            RCLCPP_WARN(logger_, 
                "[LiDAR] Unable to find a candidate set that matches target's "
                "geometry");
            return;
        }
        
        // 7. 将选中的圆心逆变换回原始坐标系
        Eigen::Matrix3d R_inv = R_align.inverse();
        for (int j = 0; j < groups[best_candidate_idx].size(); ++j) 
        {
            pcl::PointXYZ center;
            center.x = center_z0_cloud_->at(groups[best_candidate_idx][j]).x;
            center.y = center_z0_cloud_->at(groups[best_candidate_idx][j]).y;
            center.z = center_z0_cloud_->at(groups[best_candidate_idx][j]).z;

            // 将圆心坐标逆变换回原始坐标系
            Eigen::Vector3d aligned_point(center.x, center.y, center.z + average_z);
            Eigen::Vector3d original_point = R_inv * aligned_point;

            pcl::PointXYZ center_point_origin;
            center_point_origin.x = original_point.x();
            center_point_origin.y = original_point.y();
            center_point_origin.z = original_point.z();
            center_cloud->points.push_back(center_point_origin);
        }
    }

    // 栅格占据法：将拟合平面上的点栅格化成 2D 占据图，用形态学闭运算桥接扫描线间隙，
    // 再把板内的圆形空洞当作靶孔。与扫描模式无关（机械式 / MEMS 玫瑰线均适用）。
    void detect_grid_lidar(pcl::PointCloud<Common::Point>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // 1. X、Y、Z方向滤波
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<Common::Point> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);
        pass_x.filter(*filtered_cloud_);

        pcl::PassThrough<Common::Point> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);
        pass_y.filter(*filtered_cloud_);

        pcl::PassThrough<Common::Point> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);
        pass_z.filter(*filtered_cloud_);

        RCLCPP_INFO(logger_, "Depth filtered cloud size: %zu", filtered_cloud_->size());

        // 2. 拟合平面
        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.02);
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);

        pcl::ExtractIndices<Common::Point> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        RCLCPP_INFO(logger_, "Plane cloud size: %zu", plane_cloud_->size());
        if (plane_cloud_->size() < 100)
        {
            RCLCPP_WARN(logger_, "[LiDAR/grid] Too few plane points, abort.");
            return;
        }

        // 3. 平面点对齐到 Z=0
        Eigen::Vector3d normal(plane_coefficients->values[0],
                               plane_coefficients->values[1],
                               plane_coefficients->values[2]);
        normal.normalize();
        Eigen::Vector3d z_axis(0, 0, 1);
        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(normal.dot(z_axis));
        Eigen::AngleAxisd rotation(angle, axis);
        Eigen::Matrix3d R_align = rotation.toRotationMatrix();

        aligned_cloud_->reserve(plane_cloud_->size());
        double average_z = 0.0;
        for (const auto &pt : *plane_cloud_)
        {
            Eigen::Vector3d p = R_align * Eigen::Vector3d(pt.x, pt.y, pt.z);
            aligned_cloud_->push_back(pcl::PointXYZ(p.x(), p.y(), 0.0));
            average_z += p.z();
        }
        average_z /= aligned_cloud_->size();

        // 4. 栅格化成占据图（1 cm 分辨率，四周留边让洞不贴图像边界）
        const float res = 0.01f;
        float gx_min = std::numeric_limits<float>::max(), gx_max = std::numeric_limits<float>::lowest();
        float gy_min = gx_min, gy_max = gx_max;
        for (const auto &pt : *aligned_cloud_)
        {
            gx_min = std::min(gx_min, pt.x); gx_max = std::max(gx_max, pt.x);
            gy_min = std::min(gy_min, pt.y); gy_max = std::max(gy_max, pt.y);
        }
        const int pad = 4;
        const int W = static_cast<int>((gx_max - gx_min) / res) + 1 + 2 * pad;
        const int H = static_cast<int>((gy_max - gy_min) / res) + 1 + 2 * pad;
        if (W < 10 || H < 10 || W > 4000 || H > 4000)
        {
            RCLCPP_WARN(logger_, "[LiDAR/grid] Degenerate grid %dx%d, abort.", W, H);
            return;
        }

        cv::Mat occ = cv::Mat::zeros(H, W, CV_8UC1);
        for (const auto &pt : *aligned_cloud_)
        {
            int ix = static_cast<int>((pt.x - gx_min) / res) + pad;
            int iy = static_cast<int>((pt.y - gy_min) / res) + pad;
            occ.at<uint8_t>(iy, ix) = 255;
        }

        // 5. 闭运算桥接相邻扫描线之间的空隙（7 cm 内视为连续板面）
        cv::Mat closed;
        cv::morphologyEx(occ, closed, cv::MORPH_CLOSE,
                         cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7)));

        // 6. 找板内部的空洞：从图像边界向内 flood fill 得到"外部"，
        //    剩下的空像素即被板面包围的洞
        cv::Mat empty = 255 - closed;
        cv::Mat outside = empty.clone();
        cv::floodFill(outside, cv::Point(0, 0), 128);
        cv::Mat holes = (empty == 255) & (outside != 128);

        cv::Mat labels, stats, centroids;
        int n = cv::connectedComponentsWithStats(holes, labels, stats, centroids, 8, CV_32S);

        // 7. 按面积/长宽比筛选圆洞，并用最小二乘在环带点上精修圆心
        const double r_px = circle_radius_ / res;
        const double area_lo = M_PI * (r_px - 5) * (r_px - 5);
        const double area_hi = M_PI * (r_px + 5) * (r_px + 5);
        center_z0_cloud_->reserve(4);
        edge_cloud_->reserve(1024);

        for (int i = 1; i < n; ++i)
        {
            double area = stats.at<int>(i, cv::CC_STAT_AREA);
            double w = stats.at<int>(i, cv::CC_STAT_WIDTH), h = stats.at<int>(i, cv::CC_STAT_HEIGHT);
            if (area < area_lo || area > area_hi) continue;
            if (std::fabs(w - h) > 0.6 * r_px) continue;

            double cx = (centroids.at<double>(i, 0) - pad) * res + gx_min;
            double cy = (centroids.at<double>(i, 1) - pad) * res + gy_min;

            // 收集圆环带内的平面点做最小二乘圆拟合
            std::vector<Eigen::Vector2d> ring_pts;
            for (const auto &pt : *aligned_cloud_)
            {
                double d = std::hypot(pt.x - cx, pt.y - cy);
                if (d > circle_radius_ - 0.05 && d < circle_radius_ + 0.05)
                    ring_pts.emplace_back(pt.x, pt.y);
            }
            if (ring_pts.size() >= 10)
            {
                Eigen::MatrixXd A(ring_pts.size(), 3);
                Eigen::VectorXd b(ring_pts.size());
                for (size_t k = 0; k < ring_pts.size(); ++k)
                {
                    A(k, 0) = 2 * ring_pts[k].x();
                    A(k, 1) = 2 * ring_pts[k].y();
                    A(k, 2) = 1.0;
                    b(k) = ring_pts[k].squaredNorm();
                }
                Eigen::Vector3d sol = A.colPivHouseholderQr().solve(b);
                double r_fit = std::sqrt(sol(2) + sol(0) * sol(0) + sol(1) * sol(1));
                if (std::fabs(r_fit - circle_radius_) < 0.03)
                {
                    cx = sol(0);
                    cy = sol(1);
                }
                for (const auto &rp : ring_pts)
                    edge_cloud_->push_back(pcl::PointXYZ(rp.x(), rp.y(), 0.0));
                RCLCPP_INFO(logger_, "[LiDAR/grid] Hole at (%.3f, %.3f), fitted r = %.3f (%zu rim pts)",
                            cx, cy, r_fit, ring_pts.size());
            }

            pcl::PointXYZ center;
            center.x = cx; center.y = cy; center.z = 0;
            center_z0_cloud_->push_back(center);
        }
        RCLCPP_INFO(logger_, "[LiDAR/grid] %zu circular hole(s) detected.", center_z0_cloud_->size());

        // 8. Geometric consistency check（与 mech 路径一致）
        std::vector<std::vector<int>> groups;
        comb(center_z0_cloud_->size(), TARGET_NUM_CIRCLES, groups);
        std::vector<double> groups_scores(groups.size(), -1.0);
        for (size_t i = 0; i < groups.size(); ++i)
        {
            std::vector<pcl::PointXYZ> candidates;
            for (size_t j = 0; j < groups[i].size(); ++j)
                candidates.push_back(center_z0_cloud_->at(groups[i][j]));
            Square square_candidate(candidates, delta_width_circles_, delta_height_circles_);
            groups_scores[i] = square_candidate.is_valid() ? 1.0 : -1;
        }

        int best_candidate_idx = -1;
        double best_candidate_score = -1;
        for (size_t i = 0; i < groups.size(); ++i)
        {
            if (best_candidate_score == 1 && groups_scores[i] == 1)
            {
                RCLCPP_ERROR(logger_,
                    "[LiDAR/grid] More than one set of candidates fit target's geometry. "
                    "Please, make sure your parameters are well set. Exiting callback");
                return;
            }
            if (groups_scores[i] > best_candidate_score)
            {
                best_candidate_score = groups_scores[i];
                best_candidate_idx = i;
            }
        }
        if (best_candidate_idx == -1)
        {
            RCLCPP_WARN(logger_,
                "[LiDAR/grid] Unable to find a candidate set that matches target's geometry");
            return;
        }

        // 9. 将选中的圆心逆变换回原始坐标系
        Eigen::Matrix3d R_inv = R_align.inverse();
        for (size_t j = 0; j < groups[best_candidate_idx].size(); ++j)
        {
            const auto &c = center_z0_cloud_->at(groups[best_candidate_idx][j]);
            Eigen::Vector3d original_point = R_inv * Eigen::Vector3d(c.x, c.y, average_z);
            center_cloud->points.push_back(pcl::PointXYZ(original_point.x(), original_point.y(), original_point.z()));
        }
    }

    void detect_solid_lidar(pcl::PointCloud<Common::Point>::Ptr cloud, pcl::PointCloud<pcl::PointXYZ>::Ptr center_cloud)
    {
        // 1. X、Y、Z方向滤波
        filtered_cloud_->reserve(cloud->size());

        pcl::PassThrough<Common::Point> pass_x;
        pass_x.setInputCloud(cloud);
        pass_x.setFilterFieldName("x");
        pass_x.setFilterLimits(x_min_, x_max_);  // 设置X轴范围
        pass_x.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_y;
        pass_y.setInputCloud(filtered_cloud_);
        pass_y.setFilterFieldName("y");
        pass_y.setFilterLimits(y_min_, y_max_);  // 设置Y轴范围
        pass_y.filter(*filtered_cloud_);
    
        pcl::PassThrough<Common::Point> pass_z;
        pass_z.setInputCloud(filtered_cloud_);
        pass_z.setFilterFieldName("z");
        pass_z.setFilterLimits(z_min_, z_max_);  // 设置Z轴范围
        pass_z.filter(*filtered_cloud_);
    
        RCLCPP_INFO(logger_, "Filtered cloud size: %zu", filtered_cloud_->size());
        
        pcl::VoxelGrid<Common::Point> voxel_filter;
        voxel_filter.setInputCloud(filtered_cloud_);
        voxel_filter.setLeafSize(0.005f, 0.005f, 0.005f);
        voxel_filter.filter(*filtered_cloud_);
        RCLCPP_INFO(logger_, "Filtered cloud size: %zu", filtered_cloud_->size());

        // 2. 平面分割
        plane_cloud_->reserve(filtered_cloud_->size());

        pcl::ModelCoefficients::Ptr plane_coefficients(new pcl::ModelCoefficients);
        pcl::PointIndices::Ptr plane_inliers(new pcl::PointIndices);
        pcl::SACSegmentation<Common::Point> plane_segmentation;
        plane_segmentation.setModelType(pcl::SACMODEL_PLANE);
        plane_segmentation.setMethodType(pcl::SAC_RANSAC);
        plane_segmentation.setDistanceThreshold(0.01);  // 平面分割阈值
        plane_segmentation.setInputCloud(filtered_cloud_);
        plane_segmentation.segment(*plane_inliers, *plane_coefficients);
    
        pcl::ExtractIndices<Common::Point> extract;
        extract.setInputCloud(filtered_cloud_);
        extract.setIndices(plane_inliers);
        extract.filter(*plane_cloud_);
        RCLCPP_INFO(logger_, "Plane cloud size: %zu", plane_cloud_->size());
    
        // 3. 平面点云对齐   
        aligned_cloud_->reserve(plane_cloud_->size());

        Eigen::Vector3d normal(plane_coefficients->values[0],
            plane_coefficients->values[1],
            plane_coefficients->values[2]);
        normal.normalize();
        Eigen::Vector3d z_axis(0, 0, 1);

        Eigen::Vector3d axis = normal.cross(z_axis);
        double angle = acos(normal.dot(z_axis));

        Eigen::AngleAxisd rotation(angle, axis);
        Eigen::Matrix3d R = rotation.toRotationMatrix();

        // 应用旋转矩阵，将平面对齐到 Z=0 平面
        float average_z = 0.0;
        int cnt = 0;
        for (const auto& pt : *plane_cloud_) {
            Eigen::Vector3d point(pt.x, pt.y, pt.z);
            Eigen::Vector3d aligned_point = R * point;
            aligned_cloud_->push_back(pcl::PointXYZ(aligned_point.x(), aligned_point.y(), 0.0));
            average_z += aligned_point.z();
            cnt++;
        }
        average_z /= cnt;

        // 4. 提取边缘点
        edge_cloud_->reserve(aligned_cloud_->size());

        pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> normal_estimator;
        pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
        normal_estimator.setInputCloud(aligned_cloud_);
        normal_estimator.setRadiusSearch(0.03); // 设置法线估计的搜索半径
        normal_estimator.compute(*normals);
    
        pcl::PointCloud<pcl::Boundary> boundaries;
        pcl::BoundaryEstimation<pcl::PointXYZ, pcl::Normal, pcl::Boundary> boundary_estimator;
        boundary_estimator.setInputCloud(aligned_cloud_);
        boundary_estimator.setInputNormals(normals);
        boundary_estimator.setRadiusSearch(0.03); // 设置边界检测的搜索半径
        boundary_estimator.setAngleThreshold(M_PI / 4); // 设置角度阈值
        boundary_estimator.compute(boundaries);
    
        for (size_t i = 0; i < aligned_cloud_->size(); ++i) {
            if (boundaries.points[i].boundary_point > 0) {
                edge_cloud_->push_back(aligned_cloud_->points[i]);
            }
        }
        RCLCPP_INFO(logger_, "Extracted %zu edge points.", edge_cloud_->size());

        // 5. 对边缘点进行聚类
        pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
        tree->setInputCloud(edge_cloud_);
    
        std::vector<pcl::PointIndices> cluster_indices;
        pcl::EuclideanClusterExtraction<pcl::PointXYZ> ec;
        ec.setClusterTolerance(0.05); // 设置聚类距离阈值
        ec.setMinClusterSize(50);     // 最小点数
        ec.setMaxClusterSize(1000);   // 最大点数
        ec.setSearchMethod(tree);
        ec.setInputCloud(edge_cloud_);
        ec.extract(cluster_indices);
    
        RCLCPP_INFO(logger_, "Number of edge clusters: %zu", cluster_indices.size());
    
        // 6. 对每个聚类进行圆拟合
        center_z0_cloud_->reserve(4);
        Eigen::Matrix3d R_inv = R.inverse();
    
        // 对每个聚类进行圆拟合
        for (size_t i = 0; i < cluster_indices.size(); ++i) 
        {
            pcl::PointCloud<pcl::PointXYZ>::Ptr cluster(new pcl::PointCloud<pcl::PointXYZ>);
            for (const auto& idx : cluster_indices[i].indices) {
                cluster->push_back(edge_cloud_->points[idx]);
            }
    
            // 圆拟合
            pcl::ModelCoefficients::Ptr coefficients(new pcl::ModelCoefficients);
            pcl::PointIndices::Ptr inliers(new pcl::PointIndices);
            pcl::SACSegmentation<pcl::PointXYZ> seg;
            seg.setOptimizeCoefficients(true);
            seg.setModelType(pcl::SACMODEL_CIRCLE2D);
            seg.setMethodType(pcl::SAC_RANSAC);
            seg.setDistanceThreshold(0.01); // 设置距离阈值
            seg.setMaxIterations(1000);     // 设置最大迭代次数
            seg.setInputCloud(cluster);
            seg.segment(*inliers, *coefficients);
    
            if (inliers->indices.size() > 0) 
            {
                // 计算拟合误差
                double error = 0.0;
                for (const auto& idx : inliers->indices) 
                {
                    double dx = cluster->points[idx].x - coefficients->values[0];
                    double dy = cluster->points[idx].y - coefficients->values[1];
                    double distance = sqrt(dx * dx + dy * dy) - circle_radius_; // 距离误差
                    error += abs(distance);
                }
                error /= inliers->indices.size();
    
                // 如果拟合误差较小，则认为是一个圆洞
                if (error < 0.025) 
                {
                    // 将恢复后的圆心坐标添加到点云中
                    pcl::PointXYZ center_point;
                    center_point.x = coefficients->values[0];
                    center_point.y = coefficients->values[1];
                    center_point.z = 0.0;
                    center_z0_cloud_->push_back(center_point);

                    // 将圆心坐标逆变换回原始坐标系
                    Eigen::Vector3d aligned_point(center_point.x, center_point.y, center_point.z + average_z);
                    Eigen::Vector3d original_point = R_inv * aligned_point;

                    pcl::PointXYZ center_point_origin;
                    center_point_origin.x = original_point.x();
                    center_point_origin.y = original_point.y();
                    center_point_origin.z = original_point.z();
                    center_cloud->points.push_back(center_point_origin);
                }
            }
        }
    }
    // 获取中间结果的点云
    pcl::PointCloud<Common::Point>::Ptr getFilteredCloud() const { return filtered_cloud_; }
    pcl::PointCloud<Common::Point>::Ptr getPlaneCloud() const { return plane_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getAlignedCloud() const { return aligned_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getEdgeCloud() const { return edge_cloud_; }
    pcl::PointCloud<pcl::PointXYZ>::Ptr getCenterZ0Cloud() const { return center_z0_cloud_; }
};

typedef std::shared_ptr<LidarDetect> LidarDetectPtr;

#endif
