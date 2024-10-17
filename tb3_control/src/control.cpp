
/*
Used for teaching controller design for Turtlebot3
Lantao Liu
ISE, Indiana University
*/

#include "ros/ros.h"
#include <tf/transform_listener.h>
#include <nav_msgs/Odometry.h>
#include <visualization_msgs/Marker.h>
#include <std_srvs/Empty.h>
#include <sstream>
#include "geometry.h"
#include "pid.h"
#include <std_msgs/Float32MultiArray.h>
#include <math.h>
//#include <unistd.h>

// global vars
//tf::Point Odom_pos;    //odometry position (x, y, z)
double Odom_yaw, qx, qy, qz, qw, siny_cosp, cosy_cosp;    //odometry orientation (yaw)
double Odom_v, Odom_w;    //odometry linear and angular speeds
double Odom_x, Odom_y;
//std::vector<float> waypoints;
float waypoints[8] = {0,0,0.5,2.5,2.5,2.5,4.5,4.5};
//float waypoints[4] = {0,0,0.5,2.5};
// ROS Topic Subscribers
ros::Subscriber odom_sub;
ros::Subscriber waypoints_sub;
// ROS Topic Publishers
ros::Publisher cmd_vel_pub;
ros::Publisher marker_pub;



//Global variables for PID

int num_waypoints = 4;               // number of waypoints

double maxSpeed = 0.26;
double distanceConst = 0.2;//0.2;
double dt = 0.1, maxT = M_PI, minT = -M_PI, Kp = 1.5, Ki = 0.001, Kd = 0.4;
//double dt = 0.05, maxT = M_PI, minT = -M_PI, Kp = 3, Ki = 0.001, Kd = 0.4;
double dtS = 0.333, maxS = maxSpeed, minS = 0.0, KpS = 0.8, KiS = 0.0001, KdS = 0.1;


void odomCallback(const nav_msgs::Odometry odom_msg) {
    Odom_v = odom_msg.twist.twist.linear.x;
    Odom_w = odom_msg.twist.twist.angular.z;
    qx = odom_msg.pose.pose.orientation.x;
    qy = odom_msg.pose.pose.orientation.y;
    qz = odom_msg.pose.pose.orientation.z;
    qw = odom_msg.pose.pose.orientation.w;
    siny_cosp = 2 * (qw * qz + qx * qy);
    cosy_cosp = 1 - 2 * (qy * qy + qz * qz);
    Odom_yaw = std::atan2(siny_cosp, cosy_cosp);
    Odom_x = odom_msg.pose.pose.position.x;
    Odom_y = odom_msg.pose.pose.position.y;
    //update observed linear and angular speeds (real speeds published from simulation)
}



/*
 * display function that draws a circular lane in Rviz, with function  (x+0.5)^2 + (y-1)^2 = 4^2 
 */

void displayLane(bool isTrajectoryPushed, Geometry &geometry) {
    static visualization_msgs::Marker path;
    path.type = visualization_msgs::Marker::LINE_STRIP;

    path.header.frame_id = "odom";  //NOTE: this should be "paired" to the fixed frame id entry in Rviz, the default setting in Rviz for tb3-fake is "odom". Keep this line as is if you don't have an issue.
    path.header.stamp = ros::Time::now();
    path.ns = "odom";
    path.id = 0;
    path.action = visualization_msgs::Marker::ADD; // use line marker
    path.lifetime = ros::Duration();

    // path line strip is blue
    path.color.b = 1.0;
    path.color.a = 1.0;

    path.scale.x = 0.02;
    path.pose.orientation.w = 1.0;


    int slice_index2 = 0;


    VECTOR2D *prev = NULL, *current = NULL;
    while (path.points.size() > 0) {
        path.points.pop_back();
    }

    while (path.points.size() < num_waypoints) {
        geometry_msgs::Point p;

        
        p.x = waypoints[slice_index2*2+0];       //some random circular trajectory, with radius 4, and offset (-0.5, 1, 0)
        p.y = waypoints[slice_index2*2+1];
        ROS_INFO("%f,%f",p.x,p.y);
        p.z = 0;
        slice_index2++;
        path.points.push_back(p);         //for drawing path, which is line strip type
        //ROS_INFO("%f,%f,%f,%f,%f,%f",waypoints[0],waypoints[1],waypoints[2],waypoints[3],waypoints[4],waypoints[5]);

        //Add points for PID use only in 1st execution
        if (!isTrajectoryPushed) {

            VECTOR2D *temp = new VECTOR2D(p.x, p.y);

            geometry.trajectory.push_back(*temp);

            current = temp;

            if (prev != NULL) {

                geometry.path.push_back(geometry.getLineSegment(*prev, *current));

            }
            prev = current;

        }

    }

    //If you want to connect start and END points
    //if (prev != NULL && current != NULL && current != prev)
     //   geometry.path.push_back(geometry.getLineSegment(*prev, *current));


    marker_pub.publish(path);
}

void waypointsCallback(const std_msgs::Float32MultiArray::ConstPtr& waypoints_msg) {
    for (int i = 0; i < num_waypoints; i++){
                waypoints[2*i] = waypoints_msg->data[2*i];
                waypoints[2*i+1] = waypoints_msg->data[2*i+1];
            }
    //update observed linear and angular speeds (real speeds published from simulation)


}
double getDistance(Point &p1, Point &p2);
/*
 * main function 
 */
int main(int argc, char **argv) {

    ros::init(argc, argv, "control");
    ros::NodeHandle n("~");
    //tf::TransformListener m_listener;
    //tf::StampedTransform transform;
    
    
    cmd_vel_pub = n.advertise<geometry_msgs::Twist>("cmd_vel", 1);
    marker_pub = n.advertise<visualization_msgs::Marker>("visualization_marker", 1);

    odom_sub = n.subscribe("odometry_2", 10, odomCallback);
    waypoints_sub = n.subscribe("waypoints", 10, waypointsCallback);
    //do_waypoints();
    ros::Rate loop_rate(10); // ros spins 10 frames per second

    //we use geometry_msgs::twist to specify linear and angular speeds (v, w) which also denote our control inputs to pass to turtlebot
    geometry_msgs::Twist tw_msg;


    //trajectory details are here
    Geometry geometry;

    double angleError = 0.0;
    double speedError = 0.0;

    int frame_count = 0;
    PID pidTheta = PID(dt, maxT, minT, Kp, Kd, Ki);
    PID pidVelocity = PID(dtS, maxS, minS, KpS, KdS, KiS);
    
    while (ros::ok()) {
        if (frame_count == 0){
            boost::shared_ptr<std_msgs::Float32MultiArray const> Topic;
            Topic = ros::topic::waitForMessage<std_msgs::Float32MultiArray>("waypoints");
           }
/*
        if (frame_count%10 == 0){
            boost::shared_ptr<std_msgs::Float32MultiArray const> Topic;
            Topic = ros::topic::waitForMessage<std_msgs::Float32MultiArray>("waypoints");
            //num_waypoints = Topic->data.size() / 2;
            ROS_INFO("%ld", Topic->data.size());
            //num_waypoints = 4;
            for (int i = 0; i < num_waypoints; i++){
                waypoints[2*i] = Topic->data[2*i];
                waypoints[2*i+1] = Topic->data[2*i+1];
                //waypoints.push_back(Topic->data[2*i]);
                //waypoints.push_back(Topic->data[2*i+1]);
            }
            //ROS_INFO("%f,%f",waypoints[6],waypoints[7]);
            ROS_INFO("%d", num_waypoints);
            displayLane(false, geometry);
            }
        else
            displayLane(false, geometry);
        //ROS_INFO("frame %d", frame_count);
*/
        displayLane(false, geometry);

/*

YOUR CONTROL STRETEGY HERE
you need to define vehicle dynamics first (dubins car model)
after you computed your control input (here angular speed) w, pass w value to "tw_msg.angular.z" below

*/

        double omega = 0.0;
        double speed = 0.0;
        double prevDistError = 1.0;


        double tb3_lenth = 0.125;

        /*Error calculation*/
        VECTOR2D current_pos, pos_error, final_pos;
        final_pos.x = waypoints[2*num_waypoints-2];
        final_pos.y = waypoints[2*num_waypoints-1];
        current_pos.x = Odom_x;
        current_pos.y = Odom_y;
        //ROS_INFO("%f,%f",waypoints[0],waypoints[1]);
        ROS_INFO("Nearest %f,%f ", current_pos.x, current_pos.y);

        Geometry::LineSegment *linesegment = geometry.getNearestLine(current_pos);

        Geometry::LineSegment linesegmentPerpen = geometry.getMinimumDistanceLine(*linesegment, current_pos);

        //Get Next LineSegment to do velocity PID

        Geometry::LineSegment *nextLinesegment = geometry.getNextLineSegment(linesegment);

        double targetDistanceEnd = geometry.getDistance(current_pos, linesegment->endP);
        double targetDistanceStart = geometry.getDistance(current_pos, linesegment->startP);

        //Distance Error
        double distError = geometry.getDistance(current_pos, final_pos);

        double targetAnglePerpen = geometry.getGradient(current_pos, linesegmentPerpen.endP);

        VECTOR2D target = linesegment->endP;
        double targetAngle = geometry.getGradient(current_pos, target);
        double distanceToClosestPath = abs(linesegment->disatanceToAObj);

        //Error calculation based on angles
        if (distanceToClosestPath < distanceConst) {

            // This goes towards the end point of the line segment-> Select vary small distanceConst

            //angleError = targetAngle - Odom_yaw;
            double directional = targetAngle;

            double discripancy = targetAnglePerpen - directional;
            discripancy = geometry.correctAngle(discripancy);

            //Adding some potion of perpendicular angle to the error
            discripancy = 0.5* discripancy / distanceConst * abs(distanceToClosestPath);

            double combined = targetAngle + discripancy;

            angleError = combined - Odom_yaw;


        } else {

            //This goes toward the minimum distance point of the path

            angleError = targetAnglePerpen - Odom_yaw;

        }

//        speed = maxSpeed;

//If lines are long and it has sharp edges
/*        if (nextLinesegment->disatance > 8.0 && linesegment->disatance > 8.0) {
            //angleError correction for sharp turns
            if (targetDistanceEnd < 0.5) {
                double futureAngleChange = nextLinesegment->gradient - linesegment->gradient;
                futureAngleChange = geometry.correctAngle(futureAngleChange);

                //Adding some potion of perpendicular angle to the error
                futureAngleChange = futureAngleChange / distanceConst * abs(targetDistanceEnd);

                double combined = targetAngle + futureAngleChange;

                angleError = combined - Odom_yaw;
            }

            //Velocity Error Calculation for sharp turns
            if (targetDistanceStart < 0.7 || targetDistanceEnd < 0.7) {

                double targetDistance = targetDistanceEnd;

                if (targetDistanceStart < targetDistanceEnd)
                    targetDistance = targetDistanceStart;

                double speedError = 0.3 * maxSpeed * exp(-abs(targetDistance));

                speed = pidVelocity.calculate(maxSpeed, -speedError);
            }

        }

        double targetDistance = targetDistanceEnd;

        if (targetDistanceStart < targetDistanceEnd)
            targetDistance = targetDistanceStart;



        double speedError = maxSpeed * angleError * 2;

        speed = pidVelocity.calculate(maxSpeed, speedError);
*/

        //Error Angle correction for large angles
        angleError = geometry.correctAngle(angleError);
        speed = maxSpeed;
/*
        if (abs(angleError) > 0.01)
            speed = 0.4 * maxSpeed;
        if (targetDistanceEnd < 0.3)
            speed = 0.3 * maxSpeed;
*/
        //PID for the angle
        omega = pidTheta.calculate(0, -angleError);
        //PID for the speed

        if (abs(omega) > 0.5) {
            speed = 0.05;
        }

        if (distError < 0.15) {
            speed = 0;
            omega = 0;
        }
        //speed = pidVelocity.calculate(0, -speedError);
        //ROS_INFO("Nearest %f,%f, dist %f ,ShortestDistanceVecAngle %f, Odom_yaw %f, Error: %f , omega: %f", linesegment->startP.x,linesegment->startP.y, linesegment->disatanceToAObj,angleError,Odom_yaw,angleError,omega);

        //ROS_INFO("Odom_yaw %f, Angle Error: %f , omega: %f Odom_v %f, distance Error: %f , speed: %f", Odom_yaw, angleError, omega, Odom_v, distError, speed);
        ROS_INFO("Nearest %f,%f ", current_pos.x, current_pos.y);

        //for linear speed, we only use the first component of 3D linear velocity "linear.x" to represent the speed "v"
        tw_msg.linear.x = speed;
        //for angular speed, we only use the third component of 3D angular velocity "angular.z" to represent the speed "w" (in radian)
        tw_msg.angular.z = omega;

        //publish this message to the robot
        cmd_vel_pub.publish(tw_msg);

        frame_count++;
        ros::spinOnce();
        loop_rate.sleep();
    }

    return 0;

}


/*
 * This function calculate euclidean distance between two given points
 * Remove sqrt if performance hungry
 **/
double getDistance(Point &p1, Point &p2) {
    return sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
}


