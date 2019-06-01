#!/usr/bin/env python

import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped
from styx_msgs.msg import Lane, Waypoint
from std_msgs.msg import Header
from scipy.spatial import KDTree
from std_msgs.msg import Int32

import math

'''
This node will publish waypoints from the car's current position to some `x` distance ahead.

As mentioned in the doc, you should ideally first implement a version which does not care
about traffic lights or obstacles.

Once you have created dbw_node, you will update this node to use the status of traffic lights too.

Please note that our simulator also provides the exact location of traffic lights and their
current status in `/vehicle/traffic_lights` message. You can use this message to build this node
as well as to verify your TL classifier.

TODO (for Yousuf and Aaron): Stopline location for each traffic light.
'''

LOOKAHEAD_WPS = 200 # Number of waypoints we will publish. You can change this number
MAX_DECEL = 0.5
RATE = 50

class WaypointUpdater(object):
    def __init__(self):
        rospy.init_node('waypoint_updater')

        rospy.Subscriber('/current_pose', PoseStamped, self.pose_cb)
        rospy.Subscriber('/base_waypoints', Lane, self.waypoints_cb)

        # TODO: Add a subscriber for /traffic_waypoint and /obstacle_waypoint below
        rospy.Subscriber("/traffic_waypoint", Int32, self.traffic_cb)

        self.final_waypoints_pub = rospy.Publisher('final_waypoints', Lane, queue_size=5)

        # Declaring important fields
        self.__base_waypoints = None
        self.__base_waypoints_array = None
        self.__current_pose = None
        self.__waypoints_tree = None
        self.__stopline_wp_idx = -1


    def spin(self, rate=RATE):
        """
        Spins the ros node at a given rate
        """
        spin_rate = rospy.Rate(rate)
        while not rospy.is_shutdown():
            if self.__current_pose and self.__waypoints_tree:
                idx = self.get_nearest_waypoint_id(self.__current_pose)
                self.update_waypoints(idx)

            spin_rate.sleep()


    def pose_cb(self, pose):
        # TODO: Implement
        self.__current_pose = pose.pose


    def waypoints_cb(self, lane):
        # Check out document for PoseStamped Message: http://docs.ros.org/melodic/api/geometry_msgs/html/msg/PoseStamped.html
        rospy.loginfo("Received base waypoints")
        if not self.__waypoints_tree:
            self.__base_waypoints = lane.waypoints
            self.__base_waypoints_array = [[w.pose.pose.position.x, w.pose.pose.position.y] for w in self.__base_waypoints]
            self.__waypoints_tree = KDTree(self.__base_waypoints_array)
            rospy.loginfo("Successfully ingested based waypoints")

    def get_nearest_waypoint_id(self, pose):
        """
        Returns the nearest waypoint position index for the current pose
        """
        idx = self.__waypoints_tree.query([pose.position.x, pose.position.y])[1]

        closest_point = self.__base_waypoints_array[idx]
        previous_point = self.__base_waypoints_array[idx - 1]

        closest_vector = np.array(closest_point)
        previous_vector = np.array(previous_point)
        current_pos_vector =  np.array([self.__current_pose.position.x, self.__current_pose.position.y])


        val = np.dot(closest_vector - previous_vector, current_pos_vector - closest_vector)
        if val > 0:
            return (idx + 1) % len(self.__base_waypoints_array)

        return idx

    def update_waypoints(self, idx):
        """
        Publishes the latest look-ahead waypoints
        """
        # Create the header and set its timestamp
        header = Header()
        header.stamp = rospy.Time.now()

        msg = Lane()
        msg.header = header
        # Keep the copy of base_waypoints so that you don't have to recompute them
        # we are using the same base_waypoints when we get multiple messages for stopping
        # at a stopline.
        base_waypoints = self.__base_waypoints[idx: idx + LOOKAHEAD_WPS]
        msg.waypoints = base_waypoints
        # If you find out that one of the generated waypoints lies on a stop line
        # that we should be stopping at then start decelerating
        if self.__stopline_wp_idx != -1 and self.__stopline_wp_idx < (idx + LOOKAHEAD_WPS):
            rospy.logdebug('Planning to stop at '+str(self.__stopline_wp_idx)+' from total '+str(idx + LOOKAHEAD_WPS))
            msg.waypoints = self.__decelerate(base_waypoints, idx)

        self.final_waypoints_pub.publish(msg)

    def __decelerate(self, waypoints, idx):
        temp = []
        for i, wp in enumerate(waypoints):
            p = Waypoint()
            p.pose = wp.pose
            # we should be stoping a bit before the stop line
            stop_idx = max(self.__stopline_wp_idx - idx - 2, 0)
            # The velocity should be decreased based on the distance between
            # waypoint and stop point.
            dist = self.distance(waypoints, i, stop_idx)
            vel = math.sqrt(2 * MAX_DECEL * dist)
            if vel < 1.:
                vel = 0.
            # If the stop points is way ahead of us we may end up computing
            # higher velocity than we are at right now. To avoid that, following
            p.twist.twist.linear.x = min(vel, wp.twist.twist.linear.x)
            temp.append(p)
        return temp

    def traffic_cb(self, msg):
        self.__stopline_wp_idx = msg.data
        pass

    def obstacle_cb(self, msg):
        # TODO: Callback for /obstacle_waypoint message. We will implement it later
        pass

    def get_waypoint_velocity(self, waypoint):
        return waypoint.twist.twist.linear.x

    def set_waypoint_velocity(self, waypoints, waypoint, velocity):
        waypoints[waypoint].twist.twist.linear.x = velocity

    def distance(self, waypoints, wp1, wp2):
        dist = 0
        dl = lambda a, b: math.sqrt((a.x-b.x)**2 + (a.y-b.y)**2  + (a.z-b.z)**2)
        for i in range(wp1, wp2+1):
            dist += dl(waypoints[wp1].pose.pose.position, waypoints[i].pose.pose.position)
            wp1 = i
        return dist


if __name__ == '__main__':
    try:
        WaypointUpdater().spin(rate=20)
    except rospy.ROSInterruptException:
        rospy.logerr('Could not start waypoint updater node.')
