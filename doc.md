# File content and location

1. Filtered precipitation signal
     - File: /home/b11209013/2025_Research/Obs/Files/IMERG/Hovmoller.h5
     - Corresponding Python script: /home/b11209013/2025_Research/Obs/Code/IMERG/bandpass.ipynb
     - Description: Time-longitude map of precipitation signal with filtered.
     - File Structure: kw/mjo_{kl}_{kr}
2. Regression coefficient of LW and SW from CloudSat
     - File: /work/b11209013/2025_Research/regression/IMERG_CLOUDSAT.h5
     - Corresponding Python script: /home/b11209013/2025_Research/Obs/Code/IMERG/radiation_regression.ipynb
     - Description: LW and SW regression coefficient from CloudSat.
     - File Structure: 
       - lw
         - lon={central_lon}
           - kw/mjo_{kl}_{kr}
       - sw
         - lon={central_lon}
           - kw/mjo_{kl}_{kr}
       - lw_comp
         - lon={central_lon}
           - kw/mjo_{kl}_{kr}
       - sw_comp
         - lon={central_lon}
           - kw/mjo_{kl}_{kr}
3. Moisture and temperature regression coefficient from ERA5