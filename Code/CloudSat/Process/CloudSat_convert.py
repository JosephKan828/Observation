# This program is to convert CloudSat from hdf to nc and filter out unreasonable value.
# Import package
import os;
import sys;
import json;
import numpy as np;
import datetime;
import pickle as pkl;
from glob import glob;
from itertools import product;

from pyhdf.HC  import HC;
from pyhdf.SD  import SD, SDC;
from pyhdf.HDF import HDF;

from joblib import Parallel, delayed;
from netCDF4 import Dataset;
from pyhdf.error import HDF4Error;

from matplotlib import pyplot as plt

def load_data( fname ):
    
    from pyhdf.VS import VS;

    hdf = HDF( fname, SDC.READ );
    sd  = SD( fname, SDC.READ );
    vs  = hdf.vstart();
    lat = np.array( vs.attach( "Latitude" )[:] ).squeeze().astype( np.float32 );
    lon = np.array( vs.attach( "Longitude" )[:] ).squeeze().astype( np.float32 );
    vs.end();
    lon = np.where( lon <= 0.0, lon + 360.0, lon );
    hgt = np.array( sd.select( "Height" ).get() ).astype( np.float32 );
    qr  = np.array( sd.select( "QR" ).get() );  # Keep as original data type first
    sd.end();
    # Convert to float and apply scaling
    qr = qr.astype(np.float32);
    
    # Separate shortwave and longwave heating rates
    # According to documentation: first element is shortwave, second is longwave
    qsw = qr[0];  # Shortwave (first element)
    qlw = qr[1];  # Longwave (second element)
    
    # Setting mask 
    hgt_invalid_mask = ( hgt < 0.0 );
    
    # Apply reasonable physical bounds (K/day) and combine with fill value mask
    # # Typical atmospheric heating rates are within ±50 K/day
    # Apply masks
    hgt = np.where(hgt<0.0, np.nan, hgt)
    qlw = np.where(qlw<=-9999, np.nan, qlw)
    qsw = np.where(qsw<=-9999, np.nan, qsw)

    return { "lat": lat, "lon": lon, "hgt": hgt, "qlw": qlw, "qsw": qsw };


def main( year, date ):
    # Load data
    file_path = f"/work/DATA/Satellite/CloudSat/{year:04d}/{date:03d}/"
    if os.path.exists( file_path ):

        file_list = glob( file_path+"*.hdf" );
        extract_file = [];
        for f in file_list:
            try:
                extract_file.append( load_data( f ) );
            except Exception as e:
                print(f"Error processing file {f}: {e}");
                pass;
        
        if not extract_file:
            print(f"No valid files found for {year:04d}_{date:03d}");
            return;
            
        # concatenate array
        lat_concat, lon_concat, hgt_concat, qlw_concat, qsw_concat = [], [], [], [], [];
        for i in range( len( extract_file ) ):
            lat_concat.append( extract_file[i]["lat"] );
            lon_concat.append( extract_file[i]["lon"] );
            hgt_concat.append( extract_file[i]["hgt"] );
            qlw_concat.append( extract_file[i]["qlw"] );
            qsw_concat.append( extract_file[i]["qsw"] );
            
        lat_merge = np.concatenate( lat_concat, axis=0 );
        lon_merge = np.concatenate( lon_concat, axis=0 );
        hgt_merge = np.concatenate( hgt_concat, axis=0 );
        qlw_merge = np.concatenate( qlw_concat, axis=0 );
        qsw_merge = np.concatenate( qsw_concat, axis=0 );
        
        output = { 
                "lat": lat_merge.tolist(),
                "lon": lon_merge.tolist(),
                "hgt": hgt_merge.tolist(),
                "qlw": qlw_merge.tolist(),
                "qsw": qsw_merge.tolist(),
                }
    
        # Uncomment to save data
        with open( f"/work/b11209013/2024_Research/CloudSat/Stage1/{year:04d}_{date:03d}.json", "w" ) as f:
            json.dump( output, f );
    else:
        print(f"Path does not exist: {file_path}");


if __name__ == "__main__":
    year, date = sys.argv[1:3];
    # year = 2006; date = 170
    
    year, date = int( year ), int( date )

    main( int( year ), int( date ) );
    print( f"{year:04d}_{date:03d} is finished" )

