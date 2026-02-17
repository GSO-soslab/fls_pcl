#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/point_cloud2.hpp>
#include <sensor_msgs/point_cloud2_iterator.hpp>
#include <visualization_msgs/msg/marker.hpp>
#include <visualization_msgs/msg/marker_array.hpp>
#include <tf2_ros/transform_listener.hpp>
#include <tf2_ros/buffer.hpp>
#include <tf2_geometry_msgs/tf2_geometry_msgs.hpp>
#include <geometry_msgs/msg/transform_stamped.hpp>
#include <nav_msgs/msg/occupancy_grid.hpp>
#include <unordered_map>
#include <tuple>
#include <cmath>
#include <fstream>
#include <iomanip>
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
    
struct Key2D { int x, y; bool operator==(const Key2D& o) const { return x==o.x && y==o.y; } };
struct Key2DHash {
    std::size_t operator()(const Key2D& k) const {
        return std::hash<int>()(k.x) ^ (std::hash<int>()(k.y) << 1);
    }
};

class VoxelLogOddsVisualizer : public rclcpp::Node {
public:
    VoxelLogOddsVisualizer() : Node("voxel_logodds_visualizer") {
        this->declare_parameter<double>("voxel_resolution", 1.0);
        this->get_parameter("voxel_resolution", voxel_res_);

        this->declare_parameter<double>("grid_size", 100.0);
        this->get_parameter("grid_size", grid_size_);

        this->declare_parameter<double>("logodds_min", -5.0);
        this->get_parameter("logodds_min", logodds_min_);

        this->declare_parameter<double>("logodds_max", 5.0);
        this->get_parameter("logodds_max", logodds_max_);

        this->declare_parameter<std::string>("frame_id", "map");
        this->get_parameter("frame_id", frame_id_);

        this->declare_parameter<double>("probability_threshold", 0.1);
        this->get_parameter("probability_threshold", prob_threshold_);

        this->declare_parameter<std::string>("sub_pointcloud_topic", "/pointcloud");
        this->get_parameter("sub_pointcloud_topic", sub_pointcloud_topic_);

        // Changed parameter name to reflect PointCloud2 output
        this->declare_parameter<std::string>("pub_pointcloud_topic", "/occupancy_grid");
        this->get_parameter("pub_pointcloud_topic", pub_pointcloud_topic_);

        this->declare_parameter<std::string>("output_pcd_file", "occupancy_grid.pcd");
        this->get_parameter("output_pcd_file", output_pcd_file_);

        this->declare_parameter<bool>("save_pcd", false);
        this->get_parameter("save_pcd", save_pcd_);

        this->declare_parameter<double>("depth_min", -1.0);
        this->get_parameter("depth_min", depth_min_);
        this->declare_parameter<double>("depth_max",  1.0);
        this->get_parameter("depth_max", depth_max_);
        this->declare_parameter<std::string>("pub_occupancy_grid_topic", "/occupancy_grid_2d");
        this->get_parameter("pub_occupancy_grid_topic", pub_og_topic_);

        n_voxels_ = static_cast<int>(std::ceil(grid_size_ / voxel_res_));
        half_grid_ = grid_size_ / 2.0;

        // ROS subscriptions and publishers
        pc_sub_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
            sub_pointcloud_topic_, 10,
            std::bind(&VoxelLogOddsVisualizer::pcCallback, this, std::placeholders::_1)
        );
        
        // Changed to PointCloud2 publisher instead of MarkerArray
        pc_pub_ = this->create_publisher<sensor_msgs::msg::PointCloud2>(
           pub_pointcloud_topic_, 10
        );

        og_pub_ = this->create_publisher<nav_msgs::msg::OccupancyGrid>(pub_og_topic_, 10);

        // TF listener
        tf_buffer_ = std::make_shared<tf2_ros::Buffer>(this->get_clock());
        tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

        RCLCPP_INFO(this->get_logger(), "VoxelLogOddsVisualizer initialized. Grid size: %.2fm, resolution: %.2fm",
                    grid_size_, voxel_res_);
    }
        
    ~VoxelLogOddsVisualizer() { 
        if (save_pcd_) {
            RCLCPP_INFO(this->get_logger(), "Shutting down, saving PCD file...");
            savePCD(output_pcd_file_);
        }
    }

private:
    std::string output_pcd_file_;
    double voxel_res_, grid_size_, logodds_min_, logodds_max_;
    double half_grid_;
    std::string frame_id_;
    std::string sub_pointcloud_topic_;
    std::string pub_pointcloud_topic_;
    double prob_threshold_;
    int n_voxels_;
    bool save_pcd_;

    std::string pub_og_topic_;
    double depth_min_, depth_max_;
    std::unordered_map<VoxelKey, double, KeyHash> logodds_grid_;
    std::unordered_map<Key2D, double, Key2DHash> logodds_2d_grid_;

    rclcpp::Subscription<sensor_msgs::msg::PointCloud2>::SharedPtr pc_sub_;
    rclcpp::Publisher<sensor_msgs::msg::PointCloud2>::SharedPtr pc_pub_;
    rclcpp::Publisher<nav_msgs::msg::OccupancyGrid>::SharedPtr og_pub_;

    std::shared_ptr<tf2_ros::Buffer> tf_buffer_;
    std::shared_ptr<tf2_ros::TransformListener> tf_listener_;

    inline double to_logodds(double p) const {
        return std::log(p / (1.0 - p));
    }

    inline double to_prob(double logodds) const {
        return 1.0 / (1.0 + std::exp(-logodds));
    }
 
    void savePCD(const std::string& filename) {
        std::ofstream pcd_file(filename);
        if (!pcd_file.is_open()) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open PCD file: %s", filename.c_str());
            return;
        }

        // Count valid voxels (above threshold)
        size_t num_voxels = 0;
        for (const auto& kv : logodds_grid_) {
            double prob = to_prob(kv.second);
            if (prob >= prob_threshold_) {
                num_voxels++;
            }
        }

        // Write PCD header
        pcd_file << "# .PCD v.7 - Point Cloud Data file format\n";
        pcd_file << "VERSION .7\n";
        pcd_file << "FIELDS x y z rgb occupancy\n";
        pcd_file << "SIZE 4 4 4 4 4\n";
        pcd_file << "TYPE F F F U F\n";
        pcd_file << "COUNT 1 1 1 1 1\n";
        pcd_file << "WIDTH " << num_voxels << "\n";
        pcd_file << "HEIGHT 1\n";
        pcd_file << "VIEWPOINT 0 0 0 1 0 0 0\n";
        pcd_file << "POINTS " << num_voxels << "\n";
        pcd_file << "DATA ascii\n";

        // Write voxel data
        for (const auto& kv : logodds_grid_) {
            const VoxelKey& key = kv.first;
            double logodds = kv.second;
            double prob = to_prob(logodds);

            if (prob < prob_threshold_) continue;

            // Calculate voxel center in world coordinates
            float x = key.x * voxel_res_ - half_grid_ + voxel_res_ / 2.0f;
            float y = key.y * voxel_res_ - half_grid_ + voxel_res_ / 2.0f;
            float z = key.z * voxel_res_ - half_grid_ + voxel_res_ / 2.0f;

            // Calculate color (red = occupied, blue = free)
            uint8_t red = static_cast<uint8_t>(prob * 255);
            uint8_t green = 0;
            uint8_t blue = static_cast<uint8_t>((1.0 - prob) * 255);

            // Pack RGB into single 32-bit integer
            uint32_t rgb = (static_cast<uint32_t>(red) << 16) |
                          (static_cast<uint32_t>(green) << 8) |
                          static_cast<uint32_t>(blue);

            pcd_file << std::fixed << std::setprecision(6)
                    << x << " " << y << " " << z << " "
                    << rgb << " "
                    << prob << "\n";
        }

        pcd_file.close();
        RCLCPP_INFO(this->get_logger(), "Saved %zu voxels to %s", num_voxels, filename.c_str());
    }

    void pcCallback(const sensor_msgs::msg::PointCloud2::SharedPtr msg) {
        // Convert to frame_id_
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
        std::vector<float> sensor_model_prob;
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z, ++iter_i) {
            if (std::isfinite(*iter_x) && std::isfinite(*iter_y) && std::isfinite(*iter_z)) {
                points.emplace_back(*iter_x, *iter_y, *iter_z, 1.0f);
                sensor_model_prob.push_back(*iter_i);
            }
        }

        if (points.empty()) return;

        // Transform points to map frame using TF
        Eigen::Matrix4f T = transformToMatrix(trans);

        for (size_t i = 0; i < points.size(); i++) {
            Eigen::Vector4f p_map = T * points[i];
            
            // Voxel indices
            int ix = static_cast<int>(std::floor((p_map.x() + half_grid_) / voxel_res_));
            int iy = static_cast<int>(std::floor((p_map.y() + half_grid_) / voxel_res_));
            int iz = static_cast<int>(std::floor((p_map.z() + half_grid_) / voxel_res_));

            if (ix < 0 || iy < 0 || iz < 0 || ix >= n_voxels_ || iy >= n_voxels_ || iz >= n_voxels_)
                continue;

            VoxelKey key{ix, iy, iz};
            
            // Bayesian log-odds update
            double logodds_measurement = to_logodds(sensor_model_prob[i]);
            double logodds_prior = to_logodds(0.5);
            double evidence = logodds_measurement - logodds_prior;
            
            if (logodds_grid_.find(key) != logodds_grid_.end()) {
                logodds_grid_[key] += evidence;
            } else {
                logodds_grid_[key] = evidence;
            }

            // Clip
            logodds_grid_[key] = std::min(std::max(logodds_grid_[key], logodds_min_), logodds_max_);

            // 2D occupancy grid update (depth-filtered)
            if (p_map.z() >= depth_min_ && p_map.z() <= depth_max_) {
                Key2D key2d{ix, iy};
                logodds_2d_grid_[key2d] = std::min(std::max(
                    logodds_2d_grid_[key2d] + evidence, logodds_min_), logodds_max_);
            }
        }

        publishPointCloud();
        publishOccupancyGrid();
    }

    void publishPointCloud() {
        // Count valid voxels
        size_t num_voxels = 0;
        for (const auto& kv : logodds_grid_) {
            double prob = to_prob(kv.second);
            if (prob >= prob_threshold_) {
                num_voxels++;
            }
        }

        if (num_voxels == 0) return;

        // Create PointCloud2 message
        sensor_msgs::msg::PointCloud2 cloud_msg;
        cloud_msg.header.stamp = this->get_clock()->now();
        cloud_msg.header.frame_id = frame_id_;
        cloud_msg.height = 1;
        cloud_msg.width = num_voxels;
        cloud_msg.is_dense = true;
        cloud_msg.is_bigendian = false;

        // // Define fields: x, y, z, intensity
        sensor_msgs::PointCloud2Modifier modifier(cloud_msg);
        modifier.setPointCloud2Fields(4,
            "x", 1, sensor_msgs::msg::PointField::FLOAT32,
            "y", 1, sensor_msgs::msg::PointField::FLOAT32,
            "z", 1, sensor_msgs::msg::PointField::FLOAT32,
            "intensity", 1, sensor_msgs::msg::PointField::FLOAT32
        );
        
        modifier.resize(num_voxels);
        
        cloud_msg.point_step = 16;  // 12 (xyz) + 4 (intensity)
        cloud_msg.row_step = cloud_msg.point_step * num_voxels;
        cloud_msg.data.resize(cloud_msg.row_step);

        // Create iterators
        sensor_msgs::PointCloud2Iterator<float> iter_x(cloud_msg, "x");
        sensor_msgs::PointCloud2Iterator<float> iter_y(cloud_msg, "y");
        sensor_msgs::PointCloud2Iterator<float> iter_z(cloud_msg, "z");
        sensor_msgs::PointCloud2Iterator<float> iter_intensity(cloud_msg, "intensity");

        // Fill point cloud data
        for (const auto& kv : logodds_grid_) {
            const VoxelKey& key = kv.first;
            double logodds = kv.second;
            double prob = to_prob(logodds);

            if (prob < prob_threshold_) continue;

            // Calculate voxel center in world coordinates
            *iter_x = key.x * voxel_res_ - half_grid_ + voxel_res_ / 2.0f;
            *iter_y = key.y * voxel_res_ - half_grid_ + voxel_res_ / 2.0f;
            *iter_z = key.z * voxel_res_ - half_grid_ + voxel_res_ / 2.0f;

            // Set intensity as probability (0-1 range)
            *iter_intensity = static_cast<float>(prob);

            ++iter_x;
            ++iter_y;
            ++iter_z;
            ++iter_intensity;
        }

        pc_pub_->publish(cloud_msg);
    }

    void publishOccupancyGrid() {
        nav_msgs::msg::OccupancyGrid og;
        og.header.stamp = this->get_clock()->now();
        og.header.frame_id = frame_id_;

        // Grid dimensions match the 3D voxel map (same resolution and extent)
        og.info.resolution = static_cast<float>(voxel_res_);
        og.info.width  = n_voxels_;
        og.info.height = n_voxels_;

        // Origin placed so the grid is centered at (0, 0) in frame_id_
        og.info.origin.position.x = -half_grid_;
        og.info.origin.position.y = -half_grid_;
        og.info.origin.orientation.w = 1.0;

        // Initialize all cells to -1 (unknown); only observed cells will be filled
        og.data.assign(n_voxels_ * n_voxels_, -1);

        for (const auto& kv : logodds_2d_grid_) {
            const Key2D& k = kv.first;

            // Skip cells that fell outside the grid bounds
            if (k.x < 0 || k.y < 0 || k.x >= n_voxels_ || k.y >= n_voxels_) continue;

            double prob = to_prob(kv.second);

            // Only publish cells whose accumulated probability exceeds the threshold
            if (prob < prob_threshold_) continue;

            // OccupancyGrid expects [0, 100]; scale probability accordingly
            int8_t val = static_cast<int8_t>(std::round(prob * 100.0));
            og.data[k.y * n_voxels_ + k.x] = val;
        }
        og_pub_->publish(og);
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