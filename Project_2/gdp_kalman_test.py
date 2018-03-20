# Calvin He, Max Li, Ilya Lifshits, Rohit Voleti
# CS 534 Aritificial Intelligence Spring 2018
# Group 11
# Assignment 2 Part 2
# Kalman Filter

import numpy as np

# Q is the estimated process error covariance, while R is estimated measure covariance.
# Finding precise values for Q and R are beyond the scope of this assignment, so they have
# been assigned Q = identity matrix with 0.00001 along the diagonals to show that there is very little
# error in the process. And R = identity matrix with 0.1 along diagonal which is a conservative
# arbitrary value.
Q = np.zeros((2,2))
R = np.identity(2)*0.1

# The time step delta_t is 1 year since the measurements are recorded every year.
delta_t = 1

# The state vector x consists of the GDP position and GDP velocity (or rate of change of GDP).
# Our initial guess of the state vector can be any number, and with the Kalman filtering, the
# GDP estimations will eventually converge to the true GDP.
x_initial = np.matrix([[50], [1]])

# This A matrix is the state transition matrix to get from the previous state to the next state.
# Since we are using GDP position and velocity, the A matrix will follow the general physics concept
# of position and velocity kinematics of an object trajectory.
A = np.matrix([[1, delta_t], [0, 1]])

# We assume that there is no control input from the user or system, and that there is no noise.
# So the B control matrix will not be used since the u control vector is the zero vector.

# P is the estimated error of the states, or the covariance of the state. Our inital guess of the
# P matrix is just the identity matrix, but P will be updated over time due to the Kalman gain.
P_initial = P = np.identity(2)

x_predicted = A*x_initial
P_predicted = A*P_initial*np.transpose(A) + Q

# Measurements will be taking from two sensors:
# (1) Exports of goods and services of the U.S.
# (2) Government total expenditures of the U.S.
# These two sensors were chosen because they are the subparts of the GDP Formula using
# the expenditure approach (https://corporatefinanceinstitute.com/resources/knowledge/economics/gdp-formula/).
# The formula is GDP = C + G + I + NX,
# where C = consumption or private consumer spending, G = total government expenditures,
# I = sum of a country's investments, and NX = net exports or a country's total exports

z1 = 2.215
z2 = 6.347

measurement = np.matrix([[z1],[z2]])

# H is an observation matrix with the purpose to translate a vector to a measurment vector.
# Since the current state vector has components consisteing od GDP and the rate of change of GDP,
# then in order to translate GDP to is subparts (exports and expedientures according to the GDP Formula),
# the components of H as to be a constants between 0 and 1 for reducing.
# This H has arbitratily chosen values.
h1 = 0.1
h2 = 0
h3 = 0.3
h4 = 0.001
H = np.matrix([[h1, h2],[h3, h4]])

# Innovation is the difference of the measurement and the state vector translated to measurements by H.
state_to_measurement = H*x_predicted
innovation = measurement - state_to_measurement

# Using print statements to debug.
#print "MEASUREMENT:"
#print measurement
#print "STATE TO MEASUREMENT:"
#print state_to_measurement

# This is the innovation covariance matrix, which will be used to compare measurement error
# with prediction error.
innovation_covariance = H*P_predicted*np.transpose(H) + R

# The Kalman gain is a weight determined by the measurements and prediction, the value is determined
# by the prevoius computed predicted error and the innovation covariance matrix.
K = P_predicted * np.transpose(H) * np.linalg.inv(innovation_covariance)

# Now update our state estimation and state error using the Kalman gain with the two predictions.
x_updated = x_predicted + K*innovation
P_updated = (np.identity(2) - K*H)*P_predicted

# Without any sensors, there would be no z vectors to be used as measurements, and as a result
# it would not be possible to calculate the innovation and weigh it would the Kalman gain.
# That means that equation x_updated = x_predicted + K*innovation would not have the term
# K*innovation. Thus, without any sensors, x_updated = x_predicted

x_updated_no_sensors = x_predicted

# Outputs using print statements
print "Initial guess of state vector:"
print x_initial
print "\nEstimated State without using sensor information:"
print x_updated_no_sensors
print "Estimated GDP (in trillion) without any sensor information =", x_updated_no_sensors[0,0]
print "\nEstimated State using sensor information:"
print x_updated
print "Estimated GDP (in trillion) using sensor information =", x_updated[0,0]


