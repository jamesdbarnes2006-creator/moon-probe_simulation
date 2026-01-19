# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 10:52:14 2025

@author: james
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.constants import G

#-------------------
#Define Functions
#-------------------

def equations(time, state, M_e, M_m):
     
    """"This function takes initial parameters of thge Moon and Probe in relation to stationary Earth, 
    and utilises second order differential equations to find the rate of changes in each variable, for RK45 to use"""
     
    x_m, y_m, v_mx, v_my, x_p, y_p, v_px, v_py = state    #put our variables into tuple "state"
    
    #for the moon
    dv_mxdt = -(M_e * G * x_m)/(x_m**2 + y_m**2)**(3/2)   
    dx_mdt = v_mx
    dv_mydt = -(M_e * G * y_m)/(x_m**2 + y_m**2)**(3/2)
    dy_mdt = v_my 
    
    #defining the distance of the probe from the moon in terms of x_pm and y_pm
    x_pm = x_p - x_m
    y_pm = y_p - y_m
    
    #for the probe
    dv_pxdt = - (M_e * G * x_p)/(x_p**2 + y_p**2)**(3/2) - (M_m * G * x_pm)/(x_pm**2 + y_pm**2)**(3/2)
    dx_pdt = v_px
    dv_pydt = - (M_e * G * y_p)/(x_p**2 + y_p**2)**(3/2) - (M_m * G * y_pm)/(x_pm**2 + y_pm**2)**(3/2)
    dy_pdt = v_py
     
    return (dx_mdt, dy_mdt, dv_mxdt, dv_mydt, dx_pdt, dy_pdt, dv_pxdt, dv_pydt)  
    #These are the values that our RK45 will use


def energy_check(results, M_e, M_m):
    x_m, y_m, v_mx, v_my = results.y[0], results.y[1], results.y[2], results.y[3]
    x_p, y_p, v_px, v_py = results.y[4], results.y[5], results.y[6], results.y[7]
    #First extracting our values from the results_p
     
    x_pm = x_p - x_m #Defining our probe-moon distances in the x and y direction
    y_pm = y_p - y_m 
    M_p = 1000 #kg   we need to give our probe a mass for energy conservation even though it was negligible for the gravitational force
                     #(1000kg is the general mass for a lunar probe)
    K_m = 1/2 * M_m * (v_mx **2 + v_my **2) #for the moon  (J)
    K_p = 1/2 * M_p * (v_px **2 + v_py **2) #for the probe (J)
    #Lets calculate our Kinetic Energies
     
    U_em = -(G * M_e * M_m) / (x_m**2 + y_m**2)**(1/2) #Potential of Earth-Moon (J)
    U_ep = -(G * M_e * M_p) / (x_p**2 + y_p**2)**(1/2) #Potential of Earth-Probe (J)
    U_pm = -(G * M_p * M_m) / (x_pm**2 + y_pm**2)**(1/2) #Potential of Probe-Moon (J)
    #Now calculate the Potentials for all the Bodies
    
    E = K_m + K_p + U_em + U_ep + U_pm #Total Energy (J)
    E_0 = E[0] #Make our initial Energy a Variable
    deviation = np.abs(E - E_0) / np.abs(E_0) #check the deviation from E_0 of each value in the numpy array 
    deviation_max = np.max(deviation)
    
    return deviation_max

#-------------------
#Main Function
#-------------------

def main():
    t0 = 0 #s  our initial time
    tol = 1e-12 #tolerance for conservation of energy check

    #initial conditions for the moon
    x_m0, y_m0, v_mx0, v_my0 = 0, 3.84e8, 1018, 0  #m, m, m/s, m/s
    M_e = 5.972e24 # kg  mass of the earth

    #initaial conditions for the probe
    x_p0, y_p0, v_px0, v_py0 = 0, 3.96e8, 1655, 0  #m, m, m/s, m/s, m, m
    M_m = 7.35e22 # kg  mass of the moon

    #now our parameters for the RK45
    numpoints = 1001  #number of points to consider
    rtol = 1e-7 #accuracy for most values
    atol = 1e-8  #accuracy for small values


    """Ask for the timescale"""

    MyInput = '0'
    while MyInput != "q":  #setting a while loop to ask what timescale it wants until one is picked or q is pressed
        MyInput = input('Enter a choice, "1" for long timescale, "2" for short timescale or "q" to quit')
        if MyInput == '1':
            print('You have chosen part (1): long timescale (approx 1 lunar year)')
            tmax = 2.36e6      #s   (maximum time set to ≈ one lunar year)
                                                         
        elif MyInput == '2':
            print('You have chosen part (2): short timescale (approx 3-4 days)')
            tmax = 3.36e5     #s   (maximum time set to ≈ same time frame as in the example pdf approx 3-4 days)
         
        elif MyInput != "q":
            print('This is not a valid choice')

        print("\n") #creates a space in the console text to make it easier to read

        
        """RK45"""

        time = np.linspace(t0, tmax, numpoints) #created an array of all our time values, what this does is 
                                                #autmoatically create "numpoints" number of timesteps between 
                                                #t0 and tmax instead of inputting it manually

        results = solve_ivp(equations, (t0, tmax), (x_m0, y_m0, v_mx0, v_my0, x_p0, y_p0, v_px0, v_py0), args=(M_e, M_m),  method = 'RK45', t_eval = time, rtol = rtol, atol = atol)
        #This is how we run RK45 and name it results

        """Plot Simulation"""

        ax=plt.axes()    # This creates some axes, so that we
        ax.set_aspect(1) # can set the aspect ratio to 1 i.e. 
                              # x and y axes are scaled equally.                                
        ax.set_xlabel("x coordinate (m)") # Must label axes (with
        ax.set_ylabel("y coordinate (m)") # units) and give
        ax.set_title("Orbit of moon-probe system around earth") # plot title.
            
        ax.plot(results.y[0], results.y[1], label='Moon') # Make the plot for the moon
        ax.plot(results.y[4], results.y[5], label='Probe') # Make the plot for the probe
        ax.legend()  # and add a key.
        plt.show()
            
             
        deviation_max = energy_check(results, M_e, M_m)
        
        if tol < deviation_max:  #if the deviation is greater than a threshold (tol) then it will flag this
                     
            print("Energy in the system was not conserved, max deviation is equal to", deviation_max, "\n")

        else: print("Energy in the system was conserved within tolerance", "\n")
       
            
    print('You have chosen to finish - goodbye.')
    
    
# -----------------------------
# Run main only if executed directly
# -----------------------------

if __name__ == "__main__":
    main()