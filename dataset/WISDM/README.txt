This dataset includes activity data from 51 participants from an activity
recognition project. Each participant performed each of the 18 activities 
(listed in activity_key.txt) for 3 minutes and the sensor data (accelerometer 
and gyroscope for smartphone and smartwatch) was recorded at a rate of 20 Hz. 
The smartphones used were Nexus 5, Nexus 5X, and Galaxy S6 while the smartwatch
used was the LG G watch. This dataset contains the raw data.

A much more detailed description of the two main data sets is provided in the
wisdm-dataset-description.pdf document at the top level of the data directory.

################################################################################
################################################################################

The raw sensor data is located in the raw directory. Each user has its own data
file which is tagged with their subject id, the sensor, and the device.  Within
the data file, each line is:

Subject-id, Activity Label, Timestamp, x, y, z

The features are defined as follows:

subject-id: Identfies the subject and is an integer value between 1600 and 1650.

activity-label: see activity_key.txt for a mapping from 18 characters to the
       activity name

timestamp: time that the reading was taken (Unix Time)

x: x sensor value (real valued) 
y: y sensor value (real valued) 
z: z sensor value (real valued) 
