#include <ros/ros.h>
#include <tf2_ros/transform_broadcaster.h>
#include <geometry_msgs/TransformStamped.h>
#include <nav_msgs/Odometry.h>

void odomCallback(const nav_msgs::Odometry::ConstPtr &odom)
{
    static tf2_ros::TransformBroadcaster broadcaster;
    geometry_msgs::TransformStamped transform;

    transform.header.stamp = ros::Time::now();
    transform.header.frame_id = odom->header.frame_id;
    transform.child_frame_id = odom->child_frame_id;

    transform.transform.translation.x = odom->pose.pose.position.x;
    transform.transform.translation.y = odom->pose.pose.position.y;
    transform.transform.translation.z = odom->pose.pose.position.z;

    transform.transform.rotation.x = odom->pose.pose.orientation.x;
    transform.transform.rotation.y = odom->pose.pose.orientation.y;
    transform.transform.rotation.z = odom->pose.pose.orientation.z;
    transform.transform.rotation.w = odom->pose.pose.orientation.w;

    broadcaster.sendTransform(transform);
}

int main(int argc, char *argv[]) {
    ros::init(argc, argv, "rr100_odom_broadcaster");

    ros::NodeHandle nh;
    ros::Subscriber sub = nh.subscribe("odom", 10, &odomCallback);

    ros::spin();
    return 0;
}