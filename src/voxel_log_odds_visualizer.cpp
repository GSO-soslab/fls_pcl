#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <Eigen/Dense>

using namespace std::chrono_literals;

struct VoxelKey {
    int x, y, z;
    bool operator==(const VoxelKey &other) const {
        return x == other.x && y == other.y && z == other.z;
    }
};

struct KeyHash {
    std::size_t operator()(const VoxelKey &k) const {
        return ((std::hash<int>()(k.x) ^
                (std::hash<int>()(k.y) << 1)) >> 1) ^
               (std::hash<int>()(k.z) << 1);
    }
};

class VoxelLogOddsVisualizer : public rclcpp::Node {
public:
    VoxelLogOddsVisualizer() : Node("voxel_logodds_visualizer") {
        voxel_res_ = this->declare_parameter<double>("voxel_resolution", 1.0);
        grid_size_ = this->declare_parameter<double>("grid_size", 200.0);
        logodds_min_ = this->declare_parameter<double>("logodds_min", -20.0);
        logodds_max_ = this->declare_parameter<double>("logodds_max", 20.0);
        frame_id_ = this->declare_parameter<std::string>("frame_id", "alpha_rise/odom");
        prob_threshold_ = 0.3;

        n_voxels_ = static_cast<int>(std::ceil(grid_size_ / voxel_res_));

        // ROS subscriptions and publishers
        pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            "/alpha_rise/fls/pointcloud/post", 10,
            std::bind(&VoxelLogOddsVisualizer::pcCallback, this, std::placeholders::_1)
        );
        marker_pub_ = this->create_publisher<visualization_msgs::msg::MarkerArray>(
            "/alpha_rise/voxel_map", 10
        );

        // TF listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        RCLCPP_INFO(this->get_logger(), "VoxelLogOddsVisualizer initialized. Grid size: %.2fm, resolution: %.2fm",
                    grid_size_, voxel_res_);
    }

private:
    double voxel_res_, grid_size_, logodds_min_, logodds_max_;
    std::string frame_id_;
    double prob_threshold_;
    int n_voxels_;

    std::unordered_map<VoxelKey, double, KeyHash> logodds_grid_;
    std::unordered_map<VoxelKey, visualization_msgs::msg::Marker, KeyHash> markers_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
    rclcpp::Publisher<visualization_msgs::msg::MarkerArray>::SharedPtr marker_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    void pcCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Lookup transform
        geometry_msgs::msg::TransformStamped trans;
        try {
            trans = tf_buffer_->lookupTransform(
                frame_id_, msg->header.frame_id, msg->header.stamp, 100ms
            );
        } catch (tf2::TransformException &ex) {
            RCLCPP_WARN(this->get_logger(), "TF lookup failed: %s", ex.what());
            return;
        }

        // Convert PointCloud2 → xyz + intensity
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");
        sensor_msgs::PointCloud2ConstIterator<float> iter_i(*msg, "intensity");

        std::vector<Eigen::Vector4f> points;
        std::vector<float> probs;
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
            if (std::isfinite(*iter_x) && std::isfinite(*iter_y) && std::isfinite(*iter_z)) {
                points.emplace_back(*iter_x, *iter_y, *iter_z, 1.0f);
                probs.push_back(*iter_i);
            }
        }

        if (points.empty()) return;

        // Transform points to odom frame using TF
        Eigen::Matrix4f T = transformToMatrix(trans);

        std::vector<VoxelKey> voxel_keys;
        std::vector<float> logodds_updates;
        double half_grid = grid_size_ / 2.0;

        for (size_t i = 0; i < points.size(); i++) {
            Eigen::Vector4f p_odom = T * points[i];

            int ix = static_cast<int>(std::floor((p_odom.x() + half_grid) / voxel_res_));
            int iy = static_cast<int>(std::floor((p_odom.y() + half_grid) / voxel_res_));
            int iz = static_cast<int>(std::floor((p_odom.z() + half_grid) / voxel_res_));

            if (ix < 0 || iy < 0 || iz < 0 || ix >= n_voxels_ || iy >= n_voxels_ || iz >= n_voxels_)
                continue;

            VoxelKey key{ix, iy, iz};
            float logodds = std::log(probs[i] / (1.0 - probs[i]));

            if (logodds_grid_.find(key) != logodds_grid_.end()) {
                logodds_grid_[key] += logodds;
            } else {
                logodds_grid_[key] = logodds;
            }

            // Clip
            logodds_grid_[key] = std::min(std::max(logodds_grid_[key], logodds_min_), logodds_max_);
        }

        updateMarkers();
    }
    void updateMarkers() {
        visualization_msgs::msg::MarkerArray marker_array;
        double half_grid = grid_size_ / 2.0;

        for (auto &kv : logodds_grid_) {
            VoxelKey key = kv.first;
            double logodds = kv.second;
            double prob = 1.0 - 1.0 / (1.0 + std::exp(logodds));

            // Skip voxels below threshold
            if (prob < prob_threshold_) continue;

            // Create marker if not exists
            if (markers_.find(key) == markers_.end()) {
                visualization_msgs::msg::Marker marker;
                marker.header.frame_id = frame_id_;
                marker.ns = "voxels";
                marker.id = markers_.size();
                marker.type = visualization_msgs::msg::Marker::CUBE;
                marker.action = visualization_msgs::msg::Marker::ADD;
                marker.pose.position.x = key.x * voxel_res_ - half_grid + voxel_res_/2;
                marker.pose.position.y = key.y * voxel_res_ - half_grid + voxel_res_/2;
                marker.pose.position.z = key.z * voxel_res_ - half_grid + voxel_res_/2;
                marker.pose.orientation.w = 1.0;
                marker.scale.x = marker.scale.y = marker.scale.z = voxel_res_;
                markers_[key] = marker;
            }

            // Update color and alpha
            visualization_msgs::msg::Marker &marker = markers_[key];
            marker.color.r = 1.0 - prob;
            marker.color.g = prob;
            marker.color.b = 0.0;

            // Alpha scaled between 0.1 and 0.9
            marker.color.a = 0.1 + 0.8 * prob;

            marker_array.markers.push_back(marker);
        }

        auto now = this->get_clock()->now();
        for (auto &m : marker_array.markers) {
            m.header.stamp = now;
        }
        marker_pub_->publish(marker_array);
    }


    Eigen::Matrix4f transformToMatrix(const geometry_msgs::msg::TransformStamped &trans) {
        Eigen::Quaternionf q(trans.transform.rotation.w,
                             trans.transform.rotation.x,
                             trans.transform.rotation.y,
                             trans.transform.rotation.z);
        Eigen::Vector3f t(trans.transform.translation.x,
                          trans.transform.translation.y,
                          trans.transform.translation.z);
        Eigen::Matrix4f T = Eigen::Matrix4f::Identity();
        T.block<3,3>(0,0) = q.toRotationMatrix();
        T.block<3,1>(0,3) = t;
        return T;
    }
};

int main(int argc, char **argv) {
    rclcpp::init(argc, argv);
    auto node = std::make_shared<VoxelLogOddsVisualizer>();
    rclcpp::spin(node);
    rclcpp::shutdown();
    return 0;
}
