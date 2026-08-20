#ifndef UTILS_ROS_PARAMS_H
#define UTILS_ROS_PARAMS_H

#include <string>
#include <rclcpp/rclcpp.hpp>

namespace ros_params
{

// Declared with dynamic typing so a YAML value written as an integer (e.g. `min_detected_markers: 3`)
// doesn't get rejected when the parameter is read back as a double, and vice versa.
inline rclcpp::Parameter declareDynamic(rclcpp::Node *node, const std::string &name, const rclcpp::ParameterValue &default_value)
{
  if (!node->has_parameter(name))
  {
    rcl_interfaces::msg::ParameterDescriptor descriptor;
    descriptor.dynamic_typing = true;
    node->declare_parameter(name, default_value, descriptor);
  }
  return node->get_parameter(name);
}

inline void get(rclcpp::Node *node, const std::string &name, int &value, int default_value)
{
  const rclcpp::Parameter p = declareDynamic(node, name, rclcpp::ParameterValue(default_value));
  switch (p.get_type())
  {
  case rclcpp::ParameterType::PARAMETER_INTEGER: value = static_cast<int>(p.as_int()); break;
  case rclcpp::ParameterType::PARAMETER_DOUBLE: value = static_cast<int>(p.as_double()); break;
  default: value = default_value; break;
  }
}

inline void get(rclcpp::Node *node, const std::string &name, double &value, double default_value)
{
  const rclcpp::Parameter p = declareDynamic(node, name, rclcpp::ParameterValue(default_value));
  switch (p.get_type())
  {
  case rclcpp::ParameterType::PARAMETER_DOUBLE: value = p.as_double(); break;
  case rclcpp::ParameterType::PARAMETER_INTEGER: value = static_cast<double>(p.as_int()); break;
  default: value = default_value; break;
  }
}

inline void get(rclcpp::Node *node, const std::string &name, std::string &value, const std::string &default_value)
{
  const rclcpp::Parameter p = declareDynamic(node, name, rclcpp::ParameterValue(default_value));
  value = (p.get_type() == rclcpp::ParameterType::PARAMETER_STRING) ? p.as_string() : default_value;
}

inline void get(rclcpp::Node *node, const std::string &name, bool &value, bool default_value)
{
  const rclcpp::Parameter p = declareDynamic(node, name, rclcpp::ParameterValue(default_value));
  value = (p.get_type() == rclcpp::ParameterType::PARAMETER_BOOL) ? p.as_bool() : default_value;
}

/// Convenience wrapper returning the value instead of writing it out.
template <typename T> T value(rclcpp::Node *node, const std::string &name, const T &default_value)
{
  T out = default_value;
  get(node, name, out, default_value);
  return out;
}

} // namespace ros_params

#endif // UTILS_ROS_PARAMS_H
