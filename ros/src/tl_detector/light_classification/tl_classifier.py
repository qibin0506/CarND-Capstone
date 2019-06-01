import rospy
from styx_msgs.msg import TrafficLight
import rospkg
import os,sys
import tensorflow as tf
import numpy as np
import time
import os
import cv2

def load_graph (graph_file):
    """
    Loads the frozen inference protobuf file 
    """
    graph = tf.Graph()
    with graph.as_default():
        od_graph_def = tf.GraphDef()
        with tf.gfile.GFile(graph_file, 'rb') as fid:
            serialized_graph = fid.read()
            od_graph_def.ParseFromString(serialized_graph)
            tf.import_graph_def(od_graph_def, name='prefix')
    return graph


def filter_results(min_score, scores, classes):
    """Return tuple (scores, classes) where score[i] >= `min_score`"""
    n = len(classes)
    idxs = []
    for i in range(n):
        if scores[i] >= min_score:
            idxs.append(i)
        
    filtered_scores = scores[idxs, ...]
    filtered_classes = classes[idxs, ...]
    return filtered_scores, filtered_classes


class TLClassifier(object):
    
    def __init__(self):       
        self.__model_loaded = False
        self.tf_session = None
        self.prediction = None
        self.path_to_model = './res/models/frozen_graphs/'
        self.load_model()

    def load_model(self):
        detect_path = rospkg.RosPack().get_path('tl_detector')
        self.path_to_model += 'mobilenets_ssd.pb'
        rospy.loginfo('model going to be loaded from '+self.path_to_model)

        self.tf_graph = load_graph(self.path_to_model)
        self.config = tf.ConfigProto(log_device_placement=False)

        self.config.gpu_options.per_process_gpu_memory_fraction = 0.8  

        self.config.operation_timeout_in_ms = 50000 

        self.image_tensor = self.tf_graph.get_tensor_by_name('prefix/image_tensor:0')

        self.detection_scores = self.tf_graph.get_tensor_by_name('prefix/detection_scores:0')

        self.num_detections = self.tf_graph.get_tensor_by_name('prefix/num_detections:0')

        self.detection_classes = self.tf_graph.get_tensor_by_name('prefix/detection_classes:0')

        with self.tf_graph.as_default():      
            self.tf_session = tf.Session(graph=self.tf_graph, config=self.config)        

        self.__model_loaded = True 
        rospy.loginfo("Successfully loaded model")

    def get_classification(self, image, confidence_cutoff=0.3):
        """Determines the color of the traffic light in the image
        Args:
            image (cv::Mat): image containing the traffic light
        Returns:
            int: ID of traffic light color (specified in styx_msgs/TrafficLight)
        """        
        if not self.__model_loaded:
            return TrafficLight.UNKNOWN
        
        colors = ["RED", "YELLOW", "GREEN"]
        image_np = np.expand_dims(np.asarray(image, dtype=np.uint8), 0)

        (scores, classes, num) = self.tf_session.run(
            [self.detection_scores, self.detection_classes, self.num_detections],
            feed_dict={self.image_tensor: image_np})

        scores = np.squeeze(scores)
        classes = np.squeeze(classes).astype(np.int32)

        final_scores, final_classes = filter_results(confidence_cutoff, scores, classes)

        end_time = time.time()

        if len(final_classes) == 0:    
            return TrafficLight.UNKNOWN

        rospy.logerr("Predicted color is " + colors[final_classes[0] - 1] + " and score is " + str(final_scores[0])) 
        return final_classes[0] - 1 
    