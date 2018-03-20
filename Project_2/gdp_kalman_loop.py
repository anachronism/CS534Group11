# Calvin He, Max Li, Ilya Lifshits, Rohit Voleti
# CS 534 Aritificial Intelligence Spring 2018
# Group 11
# Assignment 2 Part 2
# Kalman Filter

import numpy as np
import matplotlib.pyplot as plt

# Recorded true GDP (in trillions) from 1980-2016. NOTE: The true GDP was NOT used at all
# for any calculations of the Kalman filer. It is only used for plotting at the end of
# the script in order to visualize how well the estimation has done.
# Statistic source: Google "What is the current US GDP"
true_GDP = [2.863, 3.211, 3.345, 3.638, 4.041, 4.347, 4.590, 4.870, 5.253, 5.658,
            5.980, 6.174, 6.539, 6.879, 7.309, 7.664, 8.100, 8.609, 9.089, 9.661,
            10.28, 10.62, 10.98, 11.51, 12.27, 13.09, 13.86, 14.48, 14.72, 14.42,
            14.96, 15.52, 16.16, 16.69, 17.39, 18.04, 18.57]

# Recorded U.S. exports (in trillions) from 1980-2016. These will be used as sensor
# measurements to estimate the GDP using the Kalman filter.
# Statistic source: https://data.worldbank.org/indicator/NE.EXP.GNFS.CD?locations=US
measured_exports = [0.281, 0.305, 0.283, 0.277, 0.302, 0.303, 0.321, 0.364, 0.445, 0.504,
                    0.552, 0.595, 0.633, 0.655, 0.721, 0.813, 0.868, 0.954, 0.953, 0.992,
                    1.097, 1.027, 1.003, 1.040, 1.182, 1.309, 1.476, 1.665, 1.842, 1.588,
                    1.852, 2.106, 2.198, 2.277, 2.374, 2.265, 2.215]

# Recorded U.S. total expenditures (in trillions) from 1980-2016. These will be used as sensor
# measurements to estimate the GDP using the Kalman filter.
# Statistic source: https://fred.stlouisfed.org/series/W068RCQ027SBEA
measured_expenditures = [0.961, 1.087, 1.205, 1.277, 1.400, 1.514, 1.599, 1.691, 1.779, 1.919,
                         2.100, 2.202, 2.342, 2.407, 2.505, 2.576, 2.678, 2.772, 2.865, 3.035,
                         3.156, 3.379, 3.618, 3.813, 4.040, 4.356, 4.485, 4.851, 5.361, 5.629,
                         5.784, 5.813, 5.841, 5.776, 5.967, 6.150, 6.347]

# This will hold all the estimated GDP that will be calculted in the forloop below.
estimated_GDP = []

# Q is the estimated process error covariance, while R is estimated measure covariance.
# Finding precise values for Q and R are beyond the scope of this assignment, so they have
# been assigned Q = identity matrix with 0.00001 along the diagonals to show that there is very little
# error in the process. And R = identity matrix with 0.1 along diagonal which is a conservative
# arbitrary value.
# Q = np.zeros((2,2))
Q = np.identity(2)*0.00001
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



# This forloop has the same estimation and update content as the code in gpd_kalman_test.py,
# but instead of just running the Kalman filter calculations once and finding one estimation,
# the forloop will continue the process for a set number of measurements taken from online sources
# as described in the writeup. Then we would plot the estimated GDP against the true GDP of the U.S.
# from 1980 to 2016.

# NOTE: The true GDP was not used at all for any calculations of the Kalman filter. The only
# measurments used were exports and total expenditures of the U.S., and the true GDP was only used
# for plotting for comparison between estimated value produced by Kalman and the true value.

x_previous = x_initial
P_previous = P_initial
for index in range(len(measured_exports)):
    x_predicted = A*x_previous
    P_predicted = A*P_previous*np.transpose(A) + Q
    
    # Measurements will be taking from two sensors:
    # (1) Exports of goods and services of the U.S.
    # (2) Government total expenditures of the U.S.
    # These two sensors were chosen because they are the subparts of the GDP Formula using
    # the expenditure approach (https://corporatefinanceinstitute.com/resources/knowledge/economics/gdp-formula/).
    # The formula is GDP = C + G + I + NX,
    # where C = consumption or private consumer spending, G = total government expenditures,
    # I = sum of a country's investments, and NX = net exports or a country's total exports

    z1 = measured_exports[index]
    z2 = measured_expenditures[index]
    measurement = np.matrix([[z1],[z2]]) # Measurement is a 2x1 vector

    # H is an observation matrix with the purpose to translate a vector to a measurment vector.
    # Since the current state vector has components consisteing od GDP and the rate of change of GDP,
    # then in order to translate GDP to is subparts (exports and expedientures according to the GDP Formula),
    # the components of H as to be a constants between 0 and 1 for reducing.
    # This H has arbitratily chosen values.

    h1 = 0.01
    h2 = 1
    h3 = 0.3
    h4 = 0.001
    
    H = np.matrix([[h1, h2],[h3, h4]])

    # Innovation is the difference of the measurement and the state vector translated to measurements by H.
    state_to_measurement = H*x_predicted
    innovation = measurement - state_to_measurement

    # This is the innovation covariance matrix, which will be used to compare measurement error
    # with prediction error.
    innovation_covariance = H*P_predicted*np.transpose(H) + R

    # The Kalman gain is a weight determined by the measurements and prediction, the value is determined
    # by the prevoius computed predicted error and the innovation covariance matrix.
    K = P_predicted * np.transpose(H) * np.linalg.inv(innovation_covariance)

    # Now update our state estimation and state error using the Kalman gain with the two predictions.
    x_updated = x_predicted + K*innovation
    P_updated = (np.identity(2) - K*H)*P_predicted

    # Keeping track of all the estimated GDP so far.
    estimated_GDP.append(x_updated[0,0])

    # Updating variables for the next iteration of the loop.
    x_previous = x_updated
    P_previous = P_updated


# Outputs using print statements:
    
# Plotting the calculated estimated GDP using the Kalman filter against
# the true GDP record in 1980-2016.
time = np.arange(1980, 1980+len(true_GDP), delta_t)
plt.plot(time, true_GDP, 'bo-', time, estimated_GDP, 'go-')
plt.xlabel("Year")
plt.ylabel("GDP in trillions")
plt.title("Kalman Filter Estimation of GDP vs. True GDP\nInitial State Guess = %d --> Initial Estimate = %d"
          %(x_initial[0,0], estimated_GDP[0]))
plt.text(1987, 10.5, "Blue = True GDP\nGreen = Estimated GDP")
plt.show()
