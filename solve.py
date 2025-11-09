import pandas as pd
import numpy as np
from scipy.optimize import differential_evolution
import sys

def predict_curve(params, t):
    theta_deg, M, X = params
    
    theta_rad = np.deg2rad(theta_deg)
    
    sin_theta = np.sin(theta_rad)
    cos_theta = np.cos(theta_rad)
    sin_0_3_t = np.sin(0.3 * t)
    
    exp_M_t = np.exp(M * np.abs(t))

    x_pred = (t * cos_theta - exp_M_t * sin_0_3_t * sin_theta + X)
    
    y_pred = (42 + t * sin_theta + exp_M_t * sin_0_3_t * cos_theta)
    
    return x_pred, y_pred

def calculate_l1_loss(params, t_actual, x_actual, y_actual):
    
    x_pred, y_pred = predict_curve(params, t_actual)
    
    error_x = np.abs(x_actual - x_pred)
    error_y = np.abs(y_actual - y_pred)
    
    total_error = np.mean(error_x) + np.mean(error_y)
    
    return total_error

def main():
    data_filename = 'xy_data.csv'
    try:
        data = pd.read_csv(data_filename)
        
        # --- MODIFIED SECTION ---
        # Check for 'x' and 'y' (we will create 't' ourselves)
        if not {'x', 'y'}.issubset(data.columns):
            print(f"Error: CSV file must contain 'x' and 'y' columns.")
            sys.exit()
            
        # Get the number of points
        num_points = len(data)
        
        # Generate the 't' data
        # As per the assignment, t is uniformly spaced from 6 to 60
        # We use np.linspace to create 'num_points' values
        t_data = np.linspace(6, 60, num_points)
        
        x_data = data['x'].values
        y_data = data['y'].values
        
        print(f"Data loaded successfully. Found {len(x_data)} (x,y) points.")
        print(f"Generated {len(t_data)} 't' values (uniformly spaced from 6 to 60).")
        # --- END OF MODIFIED SECTION ---

    except FileNotFoundError:
        print(f"Error: '{data_filename}' not found.")
        print("Please make sure the file is in the same directory as this script.")
        sys.exit()
    except Exception as e:
        print(f"An error occurred while loading the data: {e}")
        sys.exit()

    bounds = [(0, 50), (-0.05, 0.05), (0, 100)]

    print("Starting optimization... This may take a minute or two.")

    result = differential_evolution(
        func=calculate_l1_loss,
        bounds=bounds,
        args=(t_data, x_data, y_data), # Pass all three arrays
        strategy='best1bin',
        maxiter=1000,
        popsize=20,
        tol=0.01,
        mutation=(0.5, 1),
        recombination=0.7,
        disp=True
    )

    if result.success:
        print("\n" + "="*30)
        print("Optimization Successful! ")
        print("="*30)
        
        best_params = result.x
        best_loss = result.fun
        
        print(f"Minimum L1 Loss (Error): {best_loss}")
        print("-------------------------------------")
        print(f"Optimal theta (θ): {best_params[0]}")
        print(f"Optimal M:         {best_params[1]}")
        print(f"Optimal X:         {best_params[2]}")
        print("-------------------------------------")
       
        
    else:
        print("\nOptimization failed or did not converge.")
        print(f"Message: {result.message}")

if __name__ == "__main__":
    main()