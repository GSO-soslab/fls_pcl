#include "fls_pcl/loop.hpp"

PointCloudSubscriber::PointCloudSubscriber()
: Node("pointcloud_subscriber")
{
  subscription_ = this->create_subscription<sensor_msgs::msg::PointCloud2>(
    "/alpha_rise/fls/pointcloud",
    10,
    std::bind(&PointCloudSubscriber::pointcloud_callback, this, std::placeholders::_1)
  );
}

void PointCloudSubscriber::pointcloud_callback(const sensor_msgs::msg::PointCloud2::SharedPtr msg)
{
  RCLCPP_INFO(this->get_logger(), "Received PointCloud2 message with %d points", msg->width * msg->height);

  sensor_msgs::PointCloud2ConstIterator<float> iter_x(*msg, "x");
  sensor_msgs::PointCloud2ConstIterator<float> iter_y(*msg, "y");
  sensor_msgs::PointCloud2ConstIterator<float> iter_z(*msg, "z");

}

int main(int argc, char * argv[])
{
  rclcpp::init(argc, argv);
  rclcpp::spin(std::make_shared<PointCloudSubscriber>());
  rclcpp::shutdown();
  return 0;
}